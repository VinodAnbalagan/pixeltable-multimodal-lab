"""UDFs for Silent Film Foley v1: shot timelines from a local VLM, and sound voting."""

import base64
import io
import json
import os
import re

import PIL.Image
from openai import OpenAI

import pixeltable as pxt

VLM_BASE_URL = os.environ.get('VLM_BASE_URL', 'http://localhost:1234/v1')
VLM_MODEL = os.environ.get('VLM_MODEL', 'google/gemma-4-12b')
THINK = os.environ.get('THINK', '1') == '1'
MAX_FRAMES = 12  # frames per shot sent to the VLM; long shots are sampled evenly

_client = OpenAI(base_url=VLM_BASE_URL, api_key='local', timeout=1200)

PROMPT = """You are a Foley artist preparing sound effects for a silent film clip titled "{title}".
The images below are frames from ONE camera shot, from {start:.1f}s to {end:.1f}s of the clip.
Each image is preceded by its timestamp. The camera is the listener: describe every sound as
heard from the camera's position (for example, a train seen from beside the track sounds like a
train passing outside, not the inside of a carriage).

Return only JSON, with this structure:
{{
  "frames": [{{"t": <seconds>, "see": "<what is visible>"}}],
  "setting": {{
    "location": "<place>",
    "indoor_outdoor": "indoor | outdoor",
    "listener": "<where the camera is relative to the action>",
    "ambience": "<continuous background sound for this shot>"
  }},
  "events": [{{
    "start": <seconds>, "end": <seconds>,
    "layer": "ambience | event",
    "action": "<what happens>",
    "sound": "<the sound it makes, as heard from the camera>",
    "queries": ["<phrasing 1>", "<phrasing 2>", "<phrasing 3>"],
    "distance": "near | far"
  }}]
}}

Rules:
- One "ambience" entry covering the whole shot, then one "event" entry per distinct sound.
- Times must lie between {start:.1f} and {end:.1f}.
- Only include sounds caused by something visible. If unsure, leave it out: silence is better
  than a wrong sound.
- "queries" are three short, different descriptions of the same sound for searching a sound
  effects library (for example "cannon blast", "loud explosion outdoors", "artillery gunfire")."""


def _sample_times(start: float, end: float) -> list[float]:
    dur = max(end - start, 0.0)
    n = int(min(MAX_FRAMES, max(3, round(dur))))
    lo, hi = start + 0.05 * dur, end - 0.05 * dur
    if n == 1 or hi <= lo:
        return [round((start + end) / 2, 2)]
    return [round(lo + i * (hi - lo) / (n - 1), 2) for i in range(n)]


def _frames_at(path: str, times: list[float]) -> list[tuple[float, PIL.Image.Image]]:
    import av

    out, i = [], 0
    with av.open(path) as container:
        for frame in container.decode(video=0):
            t = float(frame.time or 0.0)
            while i < len(times) and t >= times[i]:
                out.append((times[i], frame.to_image()))
                i += 1
            if i >= len(times):
                break
    return out


def _motion_times(path: str, start: float, end: float, max_frames: int = 16) -> list[float]:
    """Frame times that follow the action: a sparse even spread across the shot, plus frames
    at the biggest moments of change between consecutive frames, with a close-up of the top one."""
    import av
    import numpy as np

    ts, diffs, prev, last_t = [], [], None, -1.0
    with av.open(path) as container:
        for frame in container.decode(video=0):
            t = float(frame.time or 0.0)
            if t < start or t - last_t < 0.1:
                if t > end:
                    break
                continue
            if t > end:
                break
            g = np.asarray(frame.to_image().convert('L').resize((64, 36)), dtype=np.float32)
            if prev is not None:
                ts.append(t)
                diffs.append(float(np.mean(np.abs(g - prev))))
            prev, last_t = g, t

    base = _sample_times(start, end)
    if len(base) > 6:  # keep 6 frames spread across the whole shot
        base = [base[round(i * (len(base) - 1) / 5)] for i in range(6)]
    if not diffs:
        return base

    order = np.argsort(diffs)[::-1]
    peaks: list[float] = []
    for i in order:
        if all(abs(ts[i] - p) >= 0.5 for p in peaks):
            peaks.append(ts[i])
        if len(peaks) == 4:
            break
    top = peaks[0]
    close = [top - 0.3, top - 0.15, top + 0.15, top + 0.3]
    times = sorted({round(t, 2) for t in base + peaks + close if start <= t <= end})
    if len(times) > max_frames:
        keep = set(round(t, 2) for t in peaks + close)
        rest = [t for t in times if t not in keep]
        step = max(1, len(rest) // max(1, max_frames - len(keep)))
        times = sorted(keep | set(rest[::step][: max_frames - len(keep)]))
    return times


def _jpeg_b64(img: PIL.Image.Image) -> str:
    img = img.convert('RGB')
    img.thumbnail((512, 512))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json(text: str) -> dict:
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL)
    a, b = text.find('{'), text.rfind('}')
    if a == -1 or b <= a:
        return {'error': 'no json', 'raw': text}
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError as e:
        return {'error': f'bad json: {e}', 'raw': text}


@pxt.udf
def cut_times(cuts: list) -> list[float]:
    """Split points for video_splitter: the start of every detected shot after the first."""
    return [round(float(c['start_time']), 3) for c in (cuts or [])[1:] if float(c['start_time']) > 0.2]


PROMPT_V2_RULES = """

Additional rules for this shot:
- The frames are not evenly spaced: extra frames were taken where the picture changes most.
  Compare consecutive frames and describe what changed between them.
- Physical actions make sound. Always include visible impacts, hits, strikes, steps, falls,
  swings and objects landing. If one frame shows something about to strike and a later frame
  shows the aftermath, the impact happened between them: give it a time between the two frames.
- Leave out only sounds you would have to invent with no visual evidence."""


PROMPT_CONTEXT_RULES = """

Some images are marked CONTEXT: they come from just before or just after this shot, across a
camera cut. Use them to understand actions that continue across the cut (for example, a swing
at the end of this shot and the impact at the start of the next). Place every event inside
{start:.1f}-{end:.1f}s; an impact that lands exactly at a cut goes at the cut time."""


def _context_times(start: float, end: float, cuts: list | None) -> tuple[list[float], list[float]]:
    """Two frames from the end of the previous shot and two from the start of the next one."""
    if not cuts:
        return [], []
    clip_end = max(float(c['start_time']) + float(c['duration']) for c in cuts)
    before = [round(start - d, 2) for d in (0.6, 0.25)] if start > 0.7 else []
    after = [round(end + d, 2) for d in (0.25, 0.6)] if end + 0.7 < clip_end else []
    return before, after


def _timeline(video: str, start: float, end: float, title: str, think: bool,
              motion: bool = False, cuts: list | None = None) -> dict:
    start, end = float(start or 0.0), float(end or 0.0)
    times = _motion_times(video, start, end) if motion else _sample_times(start, end)
    before, after = _context_times(start, end, cuts) if cuts is not None else ([], [])
    frames = _frames_at(video, sorted(set(before + times + after)))
    prompt = PROMPT.format(title=title, start=start, end=end) + (PROMPT_V2_RULES if motion else '')
    if before or after:
        prompt += PROMPT_CONTEXT_RULES.format(start=start, end=end)
    content: list[dict] = [{'type': 'text', 'text': prompt}]
    for t, img in frames:
        tag = ' (CONTEXT: previous shot)' if t in before else ' (CONTEXT: next shot)' if t in after else ''
        content.append({'type': 'text', 'text': f't = {t:.1f}s{tag}'})
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{_jpeg_b64(img)}'}})
    response = _client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{'role': 'user', 'content': content}],
        temperature=0,
        max_tokens=16384 if think else 4096,
        extra_body={} if think else {'reasoning_effort': 'none'},
    )
    choice = response.choices[0]
    result = _parse_json(choice.message.content)
    details = getattr(response.usage, 'completion_tokens_details', None)
    result['_frames_sent'] = [t for t, _ in frames]
    result['_finish_reason'] = choice.finish_reason
    result['_reasoning_tokens'] = getattr(details, 'reasoning_tokens', None)
    result['_thinking'] = think
    return result


@pxt.udf
def shot_timeline(video: pxt.Video, start: float, end: float, title: str) -> dict:
    """Ask the VLM for the setting and the timed sound events of one shot."""
    return _timeline(video, start, end, title, think=THINK)


@pxt.udf
def shot_timeline_v2(video: pxt.Video, start: float, end: float, title: str) -> dict:
    """v2: motion-aware frame sampling and a prompt that asks for visible impacts.
    Reasoning is off by default (THINK_V2=1 turns it on)."""
    think = os.environ.get('THINK_V2', '0') == '1'
    result = _timeline(video, start, end, title, think=think, motion=True)
    if 'error' in result and think:
        result = _timeline(video, start, end, title, think=False, motion=True)
        result['_retried'] = True
    return result


@pxt.udf
def repair_timeline(timeline: dict, video: pxt.Video, start: float, end: float, title: str) -> dict:
    """Keep a parsed timeline; for a shot whose output could not be parsed, ask again
    with reasoning disabled."""
    if timeline and 'error' not in timeline:
        return timeline
    retry = _timeline(video, start, end, title, think=False)
    retry['_retried'] = True
    return retry


@pxt.udf
def vote(m1: list, m2: list, m3: list) -> list:
    """Combine three ranked candidate lists by reciprocal rank fusion.
    Each result keeps its best cosine similarity and how many lists it appeared in."""
    agg: dict[str, dict] = {}
    for ranked in (m1, m2, m3):
        for rank, r in enumerate(ranked or []):
            entry = agg.setdefault(r['name'], {'name': r['name'], 'label': r['label'],
                                               'rrf': 0.0, 'best_sim': 0.0, 'votes': 0})
            entry['rrf'] += 1.0 / (10 + rank)
            entry['best_sim'] = max(entry['best_sim'], float(r['score']))
            entry['votes'] += 1
    return sorted(agg.values(), key=lambda e: e['rrf'], reverse=True)[:5]
