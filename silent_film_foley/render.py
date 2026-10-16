"""
Build a soundtrack from the matched sounds and add it to each silent clip.

Matches are read from the Foley tables (local or Cloud). The soundtrack is laid out
locally from the ESC-50 files in data/sounds/, and Pixeltable's `with_audio` joins it
to the clip in a computed column.

Usage:
    python -m silent_film_foley.render pxt://<org>:main
    python -m silent_film_foley.render /foley
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import pixeltable as pxt

from silent_film_foley.load_data import CLIPS, DATA, SOUNDS_DIR, table

OUT = Path(__file__).parent / 'outputs'
SR = 44_100
SEGMENT = 2.0   # seconds between sampled scenes (fps=0.5)
FADE = 0.15     # seconds of fade in/out at each sound change


def _load(name: str) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(SOUNDS_DIR / f'{name}.wav', sr=SR, mono=True)
    return wav


def _fade(seg: np.ndarray) -> np.ndarray:
    n = min(int(FADE * SR), len(seg) // 2)
    if n:
        ramp = np.linspace(0.0, 1.0, n)
        seg[:n] *= ramp
        seg[-n:] *= ramp[::-1]
    return seg


def build_soundtrack(scenes: list[tuple[float, str]], path: Path) -> None:
    """Place the top match for each scene at its timestamp. A sound that repeats in
    consecutive scenes keeps playing instead of restarting."""
    scenes = sorted(scenes)
    total = scenes[-1][0] + SEGMENT
    track = np.zeros(int(total * SR) + SR, dtype=np.float32)
    cache: dict[str, np.ndarray] = {}

    run_start, i = 0, 0
    while i < len(scenes):
        name = scenes[i][1]
        j = i
        while j + 1 < len(scenes) and scenes[j + 1][1] == name:
            j += 1
        start, end = scenes[i][0], scenes[j][0] + SEGMENT
        wav = cache.setdefault(name, _load(name))
        length = int((end - start) * SR)
        reps = int(np.ceil(length / len(wav)))
        seg = _fade(np.tile(wav, reps)[:length].copy())
        a = int(start * SR)
        track[a:a + len(seg)] += seg
        i = j + 1

    peak = float(np.max(np.abs(track))) or 1.0
    sf.write(path, track / peak * 0.9, SR)


def main(target: str) -> None:
    OUT.mkdir(exist_ok=True)
    scenes = table(target, 'scenes')
    rows = scenes.select(scenes.title, t=scenes.frame_attrs.time, matches=scenes.matches).collect()

    per_clip: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        if r['matches']:
            per_clip[r['title']].append((float(r['t']), r['matches'][0]['name']))

    pxt.create_dir('foley_out', if_exists='ignore')
    renders = pxt.create_table(
        'foley_out/renders',
        {'title': pxt.String, 'video': pxt.Video, 'soundtrack': pxt.Audio},
        if_exists='ignore',
    )
    renders.add_computed_column(
        final=renders.video.with_audio(renders.soundtrack), if_exists='ignore'
    )

    done = {r['title'] for r in renders.select(renders.title).collect()}
    new = []
    for title, timeline in per_clip.items():
        if title in done or title not in CLIPS:
            continue
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        wav = OUT / f'{slug}_soundtrack.wav'
        build_soundtrack(timeline, wav)
        new.append({'title': title, 'video': str(DATA / CLIPS[title]), 'soundtrack': str(wav)})
    if new:
        renders.insert(new)

    for r in renders.select(renders.title, renders.final).collect():
        slug = re.sub(r'[^a-z0-9]+', '_', r['title'].lower()).strip('_')
        dest = OUT / f'{slug}_foley.mp4'
        dest.write_bytes(Path(r['final']).read_bytes())
        print(f'wrote {dest}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pxt://vinod-lab:main')
