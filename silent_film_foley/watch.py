"""
Foley v2: the VLM "watches" the film strip in overlapping windows, carrying a running story.

    frames (4 fps stills) -> 4 s windows, 2 s apart, in order
        -> VLM: what happens, timed sound events, updated story so far
        -> foley_v2/windows (one row per window, resumable)
        -> foley_v2/log (events from each window's central 2 s) -> CLAP search + vote

Usage:
    python -m silent_film_foley.watch cannon                 # all windows, then the log
    python -m silent_film_foley.watch cannon --start 20 --end 30   # only windows in a range
    python -m silent_film_foley.watch cannon --log-only      # rebuild the log from stored windows
"""

import argparse
import time

import pixeltable as pxt

from silent_film_foley.v1_udfs import VLM_MODEL, _client, _jpeg_b64, _parse_json

DIR = 'foley_v2'
WINDOW, HOP = 4.0, 2.0

PROMPT = """You are watching a silent film clip titled "{title}", in order, a few seconds at a time,
to prepare sound effects (Foley).

The story so far:
{story}

Below are {n} consecutive frames from {a:.2f}s to {b:.2f}s, 0.25 s apart, each preceded by its
timestamp.{cuts} Treat them like a flip-book: follow motion from frame to frame. The camera is
the listener: describe every sound as heard from the camera's position.

Return only JSON:
{{
  "what_happens": "<what happens in these seconds, in order>",
  "setting": {{"location": "", "indoor_outdoor": "indoor | outdoor",
               "listener": "<where the camera is>", "ambience": "<continuous background sound>"}},
  "events": [{{"start": <s>, "end": <s>, "action": "", "sound": "<as heard from the camera>",
               "queries": ["<phrasing 1>", "<phrasing 2>", "<phrasing 3>"],
               "distance": "near | far"}}],
  "story": "<the story so far, updated to include these seconds, at most 120 words>"
}}

Rules:
- Physical actions make sound: include visible impacts, hits, shots, explosions, steps, jumps,
  falls, scraping and objects landing. If a frame shows something about to happen and a later
  frame shows its result, the event happened between them.
- Leave out only sounds you would have to invent with no visual evidence.
- Keep the people, place and objects consistent with the story so far unless the frames
  clearly show something new.
- Event times must lie between {a:.2f} and {b:.2f}."""


def windows_for(duration: float) -> list[tuple[float, float]]:
    out, a = [], 0.0
    while a < duration - 0.5:
        out.append((round(a, 2), round(min(a + WINDOW, duration), 2)))
        a += HOP
    return out


def ensure_tables() -> pxt.Table:
    return pxt.create_table(f'{DIR}/windows', {
        'key': pxt.String, 'start': pxt.Float, 'end': pxt.Float, 'frame_times': pxt.Json,
        'story_before': pxt.String, 'result': pxt.Json, 'story_after': pxt.String,
        'finish_reason': pxt.String, 'reasoning_tokens': pxt.Int, 'seconds': pxt.Float,
    }, if_exists='ignore')


def watch(key: str, lo: float | None, hi: float | None) -> None:
    frames = pxt.get_table(f'{DIR}/frames')
    clips = pxt.get_table(f'{DIR}/clips')
    clip = clips.where(clips.key == key).select(clips.title, clips.cuts).collect()[0]
    cut_times = [round(float(c['start_time']), 2) for c in (clip['cuts'] or [])[1:]]
    stills = frames.where(frames.key == key).order_by(frames.t).select(frames.t, frames.still).collect()
    duration = stills[-1]['t'] + 0.25

    win = ensure_tables()
    done = {(r['start']): r['story_after'] for r in
            win.where(win.key == key).select(win.start, win.story_after).collect()}
    story = 'Nothing yet; this is the start of the clip.'

    for a, b in windows_for(duration):
        if a in done:
            story = done[a] or story
            continue
        if (lo is not None and a < lo) or (hi is not None and a >= hi):
            continue
        shown = [r for r in stills if a <= r['t'] < b]
        cuts_here = [c for c in cut_times if a < c < b]
        cuts_note = (f' Camera cuts at {", ".join(f"{c:.2f}s" for c in cuts_here)}; the action may'
                     ' continue across a cut.') if cuts_here else ''
        # film name only: the clip label ("the cannon") could prime the model to report events
        film = clip['title'].split(':')[0]
        content = [{'type': 'text', 'text': PROMPT.format(
            title=film, story=story, n=len(shown), a=a, b=b, cuts=cuts_note)}]
        for r in shown:
            content.append({'type': 'text', 'text': f"t = {r['t']:.2f}s"})
            content.append({'type': 'image_url',
                            'image_url': {'url': f"data:image/jpeg;base64,{_jpeg_b64(r['still'])}"}})

        t0 = time.time()
        resp = _client.chat.completions.create(
            model=VLM_MODEL, messages=[{'role': 'user', 'content': content}],
            temperature=0, max_tokens=16384,
        )
        secs = time.time() - t0
        choice = resp.choices[0]
        result = _parse_json(choice.message.content)
        details = getattr(resp.usage, 'completion_tokens_details', None)
        new_story = result.get('story') if 'error' not in result else None
        win.insert([{
            'key': key, 'start': a, 'end': b, 'frame_times': [r['t'] for r in shown],
            'story_before': story, 'result': result, 'story_after': new_story or story,
            'finish_reason': choice.finish_reason,
            'reasoning_tokens': getattr(details, 'reasoning_tokens', None), 'seconds': round(secs, 1),
        }])
        story = new_story or story
        status = 'ok' if 'error' not in result else result['error']
        print(f"[{a:5.1f}-{b:5.1f}] {secs:5.0f}s {choice.finish_reason} {status} | "
              f"{result.get('what_happens', '')[:140]}")


def build_log(key: str) -> pxt.Table:
    from silent_film_foley.v1_library import nearest
    from silent_film_foley.v1_udfs import vote

    clips = pxt.get_table(f'{DIR}/clips')
    title = clips.where(clips.key == key).select(clips.title).collect()[0]['title']
    win = pxt.get_table(f'{DIR}/windows')
    rows_in = win.where(win.key == key).order_by(win.start).select(
        win.start, win.end, win.result).collect()

    log = pxt.create_table(f'{DIR}/log', {
        'title': pxt.String, 'start': pxt.Float, 'end': pxt.Float, 'layer': pxt.String,
        'action': pxt.String, 'sound': pxt.String, 'distance': pxt.String, 'setting': pxt.Json,
        'q1': pxt.String, 'q2': pxt.String, 'q3': pxt.String,
    }, if_exists='ignore')
    for i in (1, 2, 3):
        log.add_computed_column(**{f'm{i}': nearest(log[f'q{i}'])}, if_exists='ignore')
    log.add_computed_column(candidates=vote(log.m1, log.m2, log.m3), if_exists='ignore')
    log.delete(log.title == title)

    rows = []
    for i, r in enumerate(rows_in):
        res = r['result'] or {}
        if 'error' in res:
            continue
        # each window owns its central 2 s (the first and last also own the clip edges)
        own_a = r['start'] if i == 0 else r['start'] + (WINDOW - HOP) / 2
        own_b = r['end'] if i == len(rows_in) - 1 else r['start'] + (WINDOW + HOP) / 2
        setting = res.get('setting') or {}
        if setting.get('ambience'):
            amb = setting['ambience']
            rows.append({'title': title, 'start': own_a, 'end': own_b, 'layer': 'ambience',
                         'action': 'ambience', 'sound': amb, 'distance': 'far', 'setting': setting,
                         'q1': amb, 'q2': amb, 'q3': amb})
        for ev in res.get('events', []):
            s = float(ev.get('start', r['start']))
            if not (own_a <= s < own_b):
                continue
            qs = [q for q in (ev.get('queries') or []) if q] or [ev.get('sound', '')]
            qs = (qs * 3)[:3]
            rows.append({'title': title, 'start': s,
                         'end': max(float(ev.get('end', s)), s), 'layer': 'event',
                         'action': ev.get('action', ''), 'sound': ev.get('sound', ''),
                         'distance': ev.get('distance', ''), 'setting': setting,
                         'q1': qs[0], 'q2': qs[1], 'q3': qs[2]})
    if rows:
        log.insert(rows)

    for r in log.where(log.title == title).order_by(log.start).select(
            log.start, log.end, log.layer, log.action, log.sound, log.candidates).collect():
        top = (r['candidates'] or [{}])[0]
        print(f"{r['start']:5.1f}-{r['end']:5.1f} {r['layer']:8s} {r['action']}")
        print(f"            sound: {r['sound']} -> {top.get('name')} ({top.get('best_sim', 0):.2f})")
    return log


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('key')
    p.add_argument('--start', type=float)
    p.add_argument('--end', type=float)
    p.add_argument('--log-only', action='store_true')
    args = p.parse_args()
    if not args.log_only:
        watch(args.key, args.start, args.end)
    build_log(args.key)
