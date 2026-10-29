"""
Step 2: how accurate is Whisper on this lecture? Compares the stored Whisper transcript with
MIT's official captions (data/how_to_speak_official.srt) using word error rate (WER).

Usage:
    python -m lecture.accuracy             # whole lecture
    python -m lecture.accuracy 0 4         # only minutes 0-4 (e.g. the subtitle segment)
"""

import re
import sys
from pathlib import Path

import jiwer

import pixeltable as pxt

from lecture.transcribe import DIR

SRT = Path(__file__).parent / 'data' / 'how_to_speak_official.srt'
TS = re.compile(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)')


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def official_cues() -> list[tuple[float, float, str]]:
    cues = []
    for block in SRT.read_text(encoding='utf-8', errors='ignore').split('\n\n'):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = TS.search(ln)
            if m:
                g = m.groups()
                cues.append((_secs(*g[:4]), _secs(*g[4:]), ' '.join(lines[i + 1:])))
                break
    return cues


def normalize(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]|\([^)]*\)', ' ', text.lower())  # [LAUGHTER], (APPLAUSE)
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def main(a: float | None, b: float | None) -> None:
    seg = pxt.get_table(f'{DIR}/segments')
    whisper = [(r['start'], r['end'], r['text']) for r in
               seg.order_by(seg.start).select(seg.start, seg.end, seg.text).collect()]
    official = official_cues()

    def window(items):
        if a is None:
            return items
        return [x for x in items if a * 60 <= (x[0] + x[1]) / 2 < b * 60]

    ref = normalize(' '.join(t for *_, t in window(official)))
    hyp = normalize(' '.join(t for *_, t in window(whisper)))
    out = jiwer.process_words(ref, hyp)
    span = 'whole lecture' if a is None else f'minutes {a:g}-{b:g}'
    print(f'{span}: {len(ref.split())} reference words')
    print(f'WER {out.wer:.1%}  (substitutions {out.substitutions}, deletions {out.deletions}, '
          f'insertions {out.insertions})')

    print('\nexamples (MIT caption | Whisper), nearest Whisper segment by time:')
    for s, e, text in window(official)[:: max(1, len(window(official)) // 6)][:6]:
        mid = (s + e) / 2
        near = min(whisper, key=lambda w: abs((w[0] + w[1]) / 2 - mid))
        m, sec = divmod(int(s), 60)
        print(f'[{m:02d}:{sec:02d}] MIT:     {text}')
        print(f'        Whisper: {near[2]}\n')


if __name__ == '__main__':
    args = [float(x) for x in sys.argv[1:3]]
    main(*(args if len(args) == 2 else (None, None)))
