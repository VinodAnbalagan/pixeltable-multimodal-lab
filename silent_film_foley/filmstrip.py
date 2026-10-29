"""
Film strip for Foley v2: a screenshot every 0.25 s, stored in Pixeltable.

    clips  -> frames (frame_iterator, 4 fps) -> t, still (stored image)

Usage:
    python -m silent_film_foley.filmstrip cannon            # build the strip
    python -m silent_film_foley.filmstrip cannon 22 26      # contact sheet for 22-26 s
    python -m silent_film_foley.filmstrip cannon sheets     # whole clip, a still every 0.5 s
"""

import sys
from pathlib import Path

import PIL.Image
from PIL import ImageDraw, ImageFont

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator, scene_detect_content

from silent_film_foley.v2_udfs import frame_time, still

DIR = 'foley_v2'
FPS = 4.0
DATA = Path(__file__).parent / 'data'
OUT = Path(__file__).parent / 'outputs'

CLIPS = {
    'cannon': ('The General (1926): the cannon', 'general_cannon.mp4'),
    'spring': ('Spring (2019): the monolith', 'spring_monolith.mp4'),
}


def build(key: str) -> None:
    pxt.create_dir(DIR, if_exists='ignore')
    clips = pxt.create_table(
        f'{DIR}/clips', {'key': pxt.String, 'title': pxt.String, 'video': pxt.Video},
        if_exists='ignore',
    )
    clips.add_computed_column(cuts=scene_detect_content(clips.video, fps=10.0), if_exists='ignore')
    if key not in {r['key'] for r in clips.select(clips.key).collect()}:
        title, file = CLIPS[key]
        clips.insert([{'key': key, 'title': title, 'video': str(DATA / file)}])

    frames = pxt.create_view(f'{DIR}/frames', clips,
                             iterator=frame_iterator(clips.video, fps=FPS), if_exists='ignore')
    frames.add_computed_column(t=frame_time(frames.frame_attrs), if_exists='ignore')
    frames.add_computed_column(still=still(frames.frame), if_exists='ignore')

    n = frames.where(frames.key == key).count()
    print(f'{key}: {n} stills at {FPS:g} per second')


def all_sheets(key: str, step: float = 0.5, per_page: float = 8.0) -> list[Path]:
    """Contact sheets covering the whole clip: one still every `step` seconds, `per_page` s per sheet."""
    frames = pxt.get_table(f'{DIR}/frames')
    end = frames.where(frames.key == key).select(frames.t).order_by(frames.t, asc=False).limit(1).collect()[0]['t']
    paths, a = [], 0.0
    while a <= end:
        paths.append(contact_sheet(key, a, min(a + per_page, end + 0.01), step=step))
        a += per_page
    return paths


def contact_sheet(key: str, a: float, b: float, step: float | None = None) -> Path:
    frames = pxt.get_table(f'{DIR}/frames')
    clips = pxt.get_table(f'{DIR}/clips')
    cuts = clips.where(clips.key == key).select(clips.cuts).collect()[0]['cuts'] or []
    cut_times = [float(c['start_time']) for c in cuts[1:]]
    rows = (frames.where((frames.key == key) & (frames.t >= a) & (frames.t <= b))
            .order_by(frames.t).select(frames.t, frames.still).collect())
    if step:
        keep = max(1, round(step * FPS))
        rows = [r for i, r in enumerate(rows) if i % keep == 0]

    w, h, cols = 320, 240, 4
    font = ImageFont.load_default(size=18)
    sheet = PIL.Image.new('RGB', (cols * w, ((len(rows) + cols - 1) // cols) * (h + 26)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * w, (i // cols) * (h + 26)
        img = r['still'].copy()
        img.thumbnail((w, h))
        sheet.paste(img, (x + (w - img.width) // 2, y + 26))
        cut = any(abs(r['t'] - c) < 1 / FPS for c in cut_times)
        draw.text((x + 6, y + 4), f"t = {r['t']:.2f}s" + ('   CUT' if cut else ''),
                  font=font, fill=(255, 90, 90) if cut else (240, 240, 240))
    OUT.mkdir(exist_ok=True)
    path = OUT / f'strip_{key}_{a:g}-{b:g}.png'
    sheet.save(path)
    return path


if __name__ == '__main__':
    key = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == 'sheets':
        for p in all_sheets(key):
            print('wrote', p)
    elif len(sys.argv) == 4:
        print('wrote', contact_sheet(key, float(sys.argv[2]), float(sys.argv[3])))
    else:
        build(key)
