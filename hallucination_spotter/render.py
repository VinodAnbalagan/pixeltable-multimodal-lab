"""
Export the annotated frames as an MP4 and a GIF.

The video is assembled inside Pixeltable with the make_video aggregate;
ffmpeg only re-encodes it for sharing.

Usage (after spotter.py has run):
    python -m hallucination_spotter.render
"""

import shutil
import subprocess
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions.video import make_video

OUT = Path(__file__).parent / 'outputs'


def main() -> None:
    OUT.mkdir(exist_ok=True)
    frames = pxt.get_table('spotter/frames')
    result = frames.select(video=make_video(frames.pos, frames.annotated, fps=1)).collect()
    raw = OUT / 'spotter_raw.mp4'
    shutil.copy(result[0]['video'], raw)

    mp4 = OUT / 'hallucination_spotter.mp4'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw),
         '-vf', 'fps=10', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(mp4)],
        check=True,
    )

    gif = OUT / 'hallucination_spotter.gif'
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw),
         '-vf', 'fps=1,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse',
         str(gif)],
        check=True,
    )
    print(f'wrote {mp4}\nwrote {gif}')


if __name__ == '__main__':
    main()
