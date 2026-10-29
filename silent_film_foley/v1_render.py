"""
Render Foley v1: build a soundtrack from the action log and add it to each clip.

    ambience rows -> looped bed at lower volume, across the row's time span
    event rows    -> one-shot sound at the row's start time
    a row whose best match is below its threshold stays silent

Usage:
    python -m silent_film_foley.v1_render
    AMB_THRESH=0.30 EVT_THRESH=0.38 python -m silent_film_foley.v1_render
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import pixeltable as pxt
from pixeltable.functions.video import with_audio

from silent_film_foley.render import SR, _fade, _load

DIR = os.environ.get('FOLEY_DIR', 'foley_v1')
OUT = Path(__file__).parent / 'outputs'
AMB_THRESH = float(os.environ.get('AMB_THRESH', '0.30'))
EVT_THRESH = float(os.environ.get('EVT_THRESH', '0.38'))
AMB_GAIN, EVT_GAIN = 0.35, 1.0
TAG = os.environ.get('TAG', 'v1')  # output file prefix


def _duration(path: str) -> float:
    import av

    with av.open(path) as c:
        return float(c.duration or 0) / 1_000_000


def _slug(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')


def build_soundtrack(rows: list[dict], duration: float, path: Path,
                     generated: list[dict] | None = None) -> list[str]:
    track = np.zeros(int((duration + 1) * SR), dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    notes = []
    for r in sorted(rows, key=lambda r: r['start']):
        top = (r['candidates'] or [None])[0]
        thresh = AMB_THRESH if r['layer'] == 'ambience' else EVT_THRESH
        span = f"{r['start']:5.1f}-{r['end']:5.1f} {r['layer']:8s}"
        if not top or top['best_sim'] < thresh:
            notes.append(f"{span} silent   ({top['name'] if top else '-'} {top['best_sim'] if top else 0:.2f} < {thresh})")
            continue
        wav = cache.setdefault(top['name'], _load(top['name']))
        a = int(r['start'] * SR)
        gain = r.get('gain')
        if r['layer'] == 'ambience':
            n = max(int((r['end'] - r['start']) * SR), 1)
            g0 = gain or AMB_GAIN
            g1 = r.get('gain_end') or g0  # optional ramp, e.g. tension rising toward a reveal
            seg = _fade(np.tile(wav, int(np.ceil(n / len(wav))))[:n].copy()) * np.linspace(g0, g1, n)
        else:
            n = int(min(max(r['end'] - r['start'], 1.0) * SR, len(wav)))
            seg = _fade(wav[:n].copy()) * (gain or EVT_GAIN)
        seg = seg[:max(len(track) - a, 0)]
        track[a:a + len(seg)] += seg
        notes.append(f"{span} {top['name']} ({top['best_sim']:.2f})")
    for g in generated or []:  # generated score / sound design, placed as is (no looping)
        import librosa

        wav, _ = librosa.load(g['audio'], sr=SR, mono=True)
        a = int(g['start'] * SR)
        n = min(len(wav), int((g['end'] - g['start']) * SR), max(len(track) - a, 0))
        seg = wav[:n].copy()
        fade = min(int(1.0 * SR), n // 2)  # long fades so overlapping pieces crossfade
        if fade:
            ramp = np.linspace(0.0, 1.0, fade)
            seg[:fade] *= ramp
            if g['end'] < duration - 0.1:  # pieces that run to the end stay at full level
                seg[-fade:] *= ramp[::-1]
        g0 = g.get('gain') or 0.5
        g1 = g.get('gain_end') or g0
        x = np.linspace(0.0, 1.0, n)
        if g1 > g0:
            x = x ** 2  # rising pieces: slow at first, then keep climbing to the end
        seg *= g0 + (g1 - g0) * x
        track[a:a + n] += seg
        notes.append(f"{g['start']:5.1f}-{g['end']:5.1f} generated {g['role']}")

    peak = float(np.max(np.abs(track))) or 1.0
    sf.write(path, track / peak * 0.9, SR)
    return notes


def main() -> None:
    OUT.mkdir(exist_ok=True)
    log = pxt.get_table(f'{DIR}/log')
    clips = pxt.get_table(f'{DIR}/clips')
    videos = {r['title']: r['video'] for r in clips.select(clips.title, clips.video).collect()}

    per_clip = defaultdict(list)
    cols = [log.title, log.start, log.end, log.layer, log.candidates]
    for name in ('gain', 'gain_end'):  # per-row volume (foley_v3); older logs have none
        try:
            cols.append(getattr(log, name))
        except AttributeError:
            pass
    for r in log.select(*cols).collect():
        per_clip[r['title']].append(r)

    per_clip_gen = defaultdict(list)
    try:
        gen = pxt.get_table(f'{DIR}/generated')
        for g in gen.select(gen.title, gen.start, gen.end, gen.role, gen.gain, gen.gain_end,
                            gen.audio).collect():
            per_clip_gen[g['title']].append(g)
    except Exception:
        pass  # no generated layer for this run

    try:
        pxt.drop_table(f'{DIR}/renders', force=True)
    except Exception:
        pass
    renders = pxt.create_table(f'{DIR}/renders',
                               {'title': pxt.String, 'video': pxt.Video, 'soundtrack': pxt.Audio})
    renders.add_computed_column(final=with_audio(renders.video, renders.soundtrack))

    new = []
    for title, rows in per_clip.items():
        wav = OUT / f'{TAG}_{_slug(title)}_soundtrack.wav'
        notes = build_soundtrack(rows, _duration(videos[title]), wav, per_clip_gen.get(title))
        print(f'\n===== {title} =====')
        print('\n'.join(notes))
        new.append({'title': title, 'video': videos[title], 'soundtrack': str(wav)})
    renders.insert(new)

    for r in renders.select(renders.title, renders.final).collect():
        dest = OUT / f'{TAG}_{_slug(r["title"])}.mp4'
        dest.write_bytes(Path(r['final']).read_bytes())
        print(f'wrote {dest}')


if __name__ == '__main__':
    main()
