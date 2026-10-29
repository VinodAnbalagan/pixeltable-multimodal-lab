"""
Step 1: lecture -> audio -> Whisper transcript -> one row per spoken segment, searchable by meaning.

    lecture/lectures  : video -> audio (extract_audio) -> transcript (Whisper, with timestamps)
    lecture/segments  : one row per transcript segment (start, end, text) + text embedding index

Usage:
    python -m lecture.transcribe                          # transcribe + build the segment table
    python -m lecture.transcribe "how should a talk end"  # search what was said

"How to Speak" by Patrick Winston, MIT OpenCourseWare (RES.TLL-005), CC BY-NC-SA.
"""

import os
import sys
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.functions.whisper import transcribe

DIR = os.environ.get('LECTURE_DIR', 'lecture')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base.en')
DATA = Path(__file__).parent / 'data'
LECTURES = {'How to Speak (Patrick Winston, MIT OCW)': 'how_to_speak.mp4'}
EMBED = sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')


def build() -> None:
    pxt.create_dir(DIR, if_exists='ignore')
    lectures = pxt.create_table(f'{DIR}/lectures', {'title': pxt.String, 'video': pxt.Video},
                                if_exists='ignore')
    lectures.add_computed_column(audio=extract_audio(lectures.video, format='mp3'),
                                 if_exists='ignore')
    lectures.add_computed_column(transcript=transcribe(lectures.audio, model=WHISPER_MODEL),
                                 if_exists='ignore')

    have = {r['title'] for r in lectures.select(lectures.title).collect()}
    new = [{'title': t, 'video': str(DATA / f)} for t, f in LECTURES.items() if t not in have]
    if new:
        print(f'transcribing {len(new)} lecture(s) with Whisper {WHISPER_MODEL} ...')
        lectures.insert(new)

    try:  # segments are cheap to rebuild from the stored transcripts
        pxt.drop_table(f'{DIR}/segments', force=True)
    except Exception:
        pass
    segments = pxt.create_table(f'{DIR}/segments', {
        'title': pxt.String, 'start': pxt.Float, 'end': pxt.Float, 'text': pxt.String,
    })
    segments.add_embedding_index('text', string_embed=EMBED)

    rows = []
    for r in lectures.select(lectures.title, lectures.transcript).collect():
        for s in (r['transcript'] or {}).get('segments', []):
            text = (s.get('text') or '').strip()
            if text:
                rows.append({'title': r['title'], 'start': round(float(s['start']), 2),
                             'end': round(float(s['end']), 2), 'text': text})
    segments.insert(rows)
    print(f'{len(rows)} segments indexed')


def search(question: str, k: int = 5) -> None:
    segments = pxt.get_table(f'{DIR}/segments')
    sim = segments.text.similarity(string=question)
    for r in (segments.order_by(sim, asc=False).limit(k)
              .select(segments.start, segments.end, segments.text, score=sim).collect()):
        m, s = divmod(int(r['start']), 60)
        print(f"[{m:02d}:{s:02d}] ({r['score']:.2f}) {r['text']}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        search(' '.join(sys.argv[1:]))
    else:
        build()
