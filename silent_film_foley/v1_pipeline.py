"""
Silent Film Foley v1

    clips -> scene detection -> shots (video_splitter at the cuts)
          -> VLM shot timeline: setting + timed events, with three search phrasings each
          -> action log (one row per ambience / event, with start and end)
          -> CLAP search per phrasing -> vote -> chosen sound (or silence below a threshold)
          -> soundtrack (ambience bed + events) -> with_audio -> MP4

Runs locally (the VLM is served by LM Studio). THINK=0 disables reasoning.

Usage:
    python -m silent_film_foley.v1_pipeline            # build / continue
    TL=timeline_v2 python -m silent_film_foley.v1_pipeline   # motion-aware sampling variant
    FRESH=1 python -m silent_film_foley.v1_pipeline    # rebuild everything except the sound library
"""

import os
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions.video import scene_detect_content, video_splitter

from silent_film_foley.models import clap_audio, clap_text
from silent_film_foley.v1_udfs import (
    cut_times,
    repair_timeline,
    shot_timeline,
    shot_timeline_v2,
    vote,
)

DATA = Path(__file__).parent / 'data'
DIR = 'foley_v1'
FRESH = os.environ.get('FRESH') == '1'
TL = os.environ.get('TL', 'timeline_final')  # which timeline column feeds the log

CLIPS = {
    'The General (1926): the cannon': 'general_cannon.mp4',
    'Spring (2019): the monolith': 'spring_monolith.mp4',
}


def build_library() -> None:
    """All local ESC-50 clips (200, uncurated) with a CLAP audio embedding index."""
    pxt.create_dir(DIR, if_exists='ignore')
    sounds = pxt.create_table(
        f'{DIR}/sounds', {'name': pxt.String, 'label': pxt.String, 'audio': pxt.Audio},
        if_exists='ignore',
    )
    if sounds.count() > 0:
        return
    sounds.add_embedding_index('audio', audio_embed=clap_audio, string_embed=clap_text)
    files = sorted((DATA / 'sounds').glob('*.wav'))
    sounds.insert([
        {'name': f.stem, 'label': f.stem.rsplit('_', 1)[0], 'audio': str(f)} for f in files
    ])


def build_shots() -> pxt.Table:
    if FRESH:
        for name in ('log', 'shots', 'clips'):
            try:
                pxt.drop_table(f'{DIR}/{name}', force=True)
            except Exception:
                pass

    clips = pxt.create_table(f'{DIR}/clips', {'title': pxt.String, 'video': pxt.Video},
                             if_exists='ignore')
    clips.add_computed_column(cuts=scene_detect_content(clips.video, fps=10.0), if_exists='ignore')
    clips.add_computed_column(split_times=cut_times(clips.cuts), if_exists='ignore')

    have = {r['title'] for r in clips.select(clips.title).collect()}
    new = [{'title': t, 'video': str(DATA / f)} for t, f in CLIPS.items()
           if t not in have and (DATA / f).exists()]
    if new:
        clips.insert(new)

    shots = pxt.create_view(
        f'{DIR}/shots', clips,
        iterator=video_splitter(clips.video, segment_times=clips.split_times),
        if_exists='ignore',
    )
    shots.add_computed_column(
        timeline=shot_timeline(shots.video, shots.segment_start, shots.segment_end, shots.title),
        if_exists='ignore',
    )
    shots.add_computed_column(
        timeline_final=repair_timeline(shots.timeline, shots.video, shots.segment_start,
                                       shots.segment_end, shots.title),
        if_exists='ignore',
    )
    if TL == 'timeline_v2':
        shots.add_computed_column(
            timeline_v2=shot_timeline_v2(shots.video, shots.segment_start, shots.segment_end,
                                         shots.title),
            if_exists='ignore',
        )
    return shots


def build_log(shots: pxt.Table) -> pxt.Table:
    from silent_film_foley.v1_library import nearest

    # The log is cheap to rebuild (text search only), so it always reflects the latest timelines.
    try:
        pxt.drop_table(f'{DIR}/log', force=True)
    except Exception:
        pass
    log = pxt.create_table(f'{DIR}/log', {
        'title': pxt.String, 'shot': pxt.Int, 'start': pxt.Float, 'end': pxt.Float,
        'layer': pxt.String, 'action': pxt.String, 'sound': pxt.String, 'distance': pxt.String,
        'setting': pxt.Json, 'q1': pxt.String, 'q2': pxt.String, 'q3': pxt.String,
    }, if_exists='ignore')
    for i in (1, 2, 3):
        log.add_computed_column(**{f'm{i}': nearest(log[f'q{i}'])}, if_exists='ignore')
    log.add_computed_column(candidates=vote(log.m1, log.m2, log.m3), if_exists='ignore')

    rows = []
    for r in shots.select(shots.title, shots.pos, shots.segment_start, shots.segment_end,
                          tl=shots[TL]).order_by(shots.title, shots.pos).collect():
        tl, s0, s1 = r['tl'] or {}, float(r['segment_start'] or 0), float(r['segment_end'] or 0)
        if 'error' in tl:
            print(f"[{r['title']} shot {r['pos']}] VLM output not parsed: {tl['error']} "
                  f"(finish_reason={tl.get('_finish_reason')}, reasoning_tokens={tl.get('_reasoning_tokens')})")
            continue
        setting = tl.get('setting', {}) or {}
        events = [e for e in tl.get('events', []) if e.get('layer') != 'ambience']
        ambience = setting.get('ambience') or next(
            (e.get('sound') for e in tl.get('events', []) if e.get('layer') == 'ambience'), None)
        if ambience:
            events.insert(0, {'start': s0, 'end': s1, 'layer': 'ambience', 'action': 'shot ambience',
                              'sound': ambience, 'queries': [ambience], 'distance': 'far'})
        for ev in events:
            qs = [q for q in (ev.get('queries') or []) if q] or [ev.get('sound', '')]
            qs = (qs + qs * 3)[:3]
            start = min(max(float(ev.get('start', s0)), s0), s1)
            end = min(max(float(ev.get('end', s1)), start), s1)
            rows.append({
                'title': r['title'], 'shot': r['pos'], 'start': start, 'end': end,
                'layer': ev.get('layer', 'event'), 'action': ev.get('action', ''),
                'sound': ev.get('sound', ''), 'distance': ev.get('distance', ''),
                'setting': tl.get('setting', {}), 'q1': qs[0], 'q2': qs[1], 'q3': qs[2],
            })
    if rows:
        log.insert(rows)
    return log


def report(log: pxt.Table) -> None:
    rows = log.select(log.title, log.start, log.end, log.layer, log.action, log.sound,
                      log.candidates).order_by(log.title, log.start).collect()
    title = None
    for r in rows:
        if r['title'] != title:
            title = r['title']
            print(f'\n===== {title} =====')
        top = (r['candidates'] or [{}])[0]
        match = f"{top.get('name')} (sim {top.get('best_sim', 0):.2f}, votes {top.get('votes')})" if top else '-'
        print(f"{r['start']:5.1f}-{r['end']:5.1f}  {r['layer']:8s}  {r['action']}")
        print(f"             sound: {r['sound']}  ->  {match}")


if __name__ == '__main__':
    build_library()
    shots = build_shots()
    log = build_log(shots)
    report(log)
