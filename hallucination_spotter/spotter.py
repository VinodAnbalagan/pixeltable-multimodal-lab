"""
Hallucination Spotter v0

Does a vision-language model mention objects that aren't in the frame?

    video -> frames (1 fps) -> VLM caption -> mentioned objects
                            -> YOLOX detections -> detected objects
          -> hallucinated = mentioned - detected   (CHAIR-i)
          -> omitted      = detected - mentioned

Two caption variants share the same frames and detections:
    caption        reasoning disabled
    caption_think  reasoning enabled (THINK=1)

Only objects in the 80-class COCO vocabulary can be checked, and detector
misses can show up as false positives.

Usage (from the repo root, LM Studio server running):
    python -m hallucination_spotter.spotter
    THINK=1 python -m hallucination_spotter.spotter   # add the reasoning variant
    FRESH=1 python -m hallucination_spotter.spotter   # rebuild from scratch
"""

import os

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.yolox import yolox

from hallucination_spotter.udfs import (
    describe,
    describe_thinking,
    detected_objects,
    hallucinated_objects,
    hallucination_rate,
    mentioned_objects,
    omitted_objects,
)

DIR = 'spotter'
FRESH = os.environ.get('FRESH') == '1'
THINK = os.environ.get('THINK') == '1'
VIDEO_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/bangkok.mp4'

if FRESH:
    pxt.drop_dir(DIR, force=True)
pxt.create_dir(DIR, if_exists='ignore')

videos = pxt.create_table(f'{DIR}/videos', {'video': pxt.Video}, if_exists='ignore')
if videos.count() == 0:
    videos.insert([{'video': VIDEO_URL}])

frames = pxt.create_view(
    f'{DIR}/frames',
    videos,
    iterator=frame_iterator(videos.video, fps=1.0),
    if_exists='ignore',
)

frames.add_computed_column(
    detections=yolox(frames.frame, model_id='yolox_m', threshold=0.3), if_exists='ignore'
)
frames.add_computed_column(detected=detected_objects(frames.detections), if_exists='ignore')


def add_variant(suffix: str, caption_fn) -> None:
    """Caption -> mentioned -> hallucinated / omitted -> rate, for one VLM variant."""
    cols = {name: f'{name}{suffix}' for name in
            ('caption', 'mentioned', 'hallucinated', 'omitted', 'halluc_rate')}
    frames.add_computed_column(**{cols['caption']: caption_fn(frames.frame)}, if_exists='ignore')
    caption = frames[cols['caption']]
    frames.add_computed_column(**{cols['mentioned']: mentioned_objects(caption)}, if_exists='ignore')
    mentioned = frames[cols['mentioned']]
    frames.add_computed_column(
        **{cols['hallucinated']: hallucinated_objects(mentioned, frames.detected)}, if_exists='ignore'
    )
    frames.add_computed_column(
        **{cols['omitted']: omitted_objects(mentioned, frames.detected)}, if_exists='ignore'
    )
    frames.add_computed_column(
        **{cols['halluc_rate']: hallucination_rate(mentioned, frames[cols['hallucinated']])},
        if_exists='ignore',
    )


def report(suffix: str, label: str) -> None:
    rows = (
        frames.select(
            frames.pos,
            caption=frames[f'caption{suffix}'],
            mentioned=frames[f'mentioned{suffix}'],
            hallucinated=frames[f'hallucinated{suffix}'],
            omitted=frames[f'omitted{suffix}'],
            detected=frames.detected,
        )
        .order_by(frames.pos)
        .collect()
    )
    print(f'\n===== {label} =====\n')
    n_mentioned = n_halluc = n_empty = 0
    for r in rows:
        n_mentioned += len(r['mentioned'])
        n_halluc += len(r['hallucinated'])
        n_empty += not r['caption']
        flag = '  <-- hallucination' if r['hallucinated'] else ''
        print(f"[{r['pos']:>2}] {r['caption'] or '(empty)'}")
        print(f"     mentioned={r['mentioned']}  detected={r['detected']}")
        print(f"     hallucinated={r['hallucinated']}{flag}  omitted={r['omitted']}\n")
    rate = n_halluc / n_mentioned if n_mentioned else 0.0
    print(f'{label}: frames={len(rows)}  empty={n_empty}  mentioned={n_mentioned}  '
          f'hallucinated={n_halluc}  CHAIR-i={rate:.0%}')


add_variant('', describe)
if THINK:
    add_variant('_think', describe_thinking)

report('', 'reasoning off')
if THINK:
    report('_think', 'reasoning on')
