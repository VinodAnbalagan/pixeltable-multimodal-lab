"""
Step 3: multilingual subtitles for one segment of the lecture.

    lecture/segments (Whisper) -> subtitle lines for [START, END] (<= 2 lines each, timed)
    lecture/sublines : en -> es, fr, de, zh (Helsinki-NLP opus-mt models) -> wrapped text
    lecture/subclip  : the segment cut from the lecture (clip) -> one subtitled video per
                       language (overlay_text for each line, at its start/end time)

Usage:
    python -m lecture.subtitles                      # default segment: "How to stop" (56:49-58:47)
    SEG_START=250 SEG_END=345 SEG_NAME=how_to_start python -m lecture.subtitles   # another segment

Output: outputs/{SEG_NAME}_multilingual.mp4 (one video, five selectable subtitle tracks) plus
.srt / .vtt files per language. BURN=1 also writes one video per language with burned-in text.

"How to Speak" by Patrick Winston, MIT OpenCourseWare (RES.TLL-005), CC BY-NC-SA.
Subtitled versions are shared under the same license.
"""

import os
from pathlib import Path

import pixeltable as pxt
from pixeltable.functions.huggingface import translation
from pixeltable.functions.video import clip, overlay_text

from lecture.sub_udfs import wrap, zh_input
from lecture.transcribe import DIR

SEG_START = float(os.environ.get('SEG_START', '3409'))  # 56:49, "end with a joke"
SEG_END = float(os.environ.get('SEG_END', '3527'))      # 58:47, "...don't say thank you"
MAX_CHARS = 84                                          # per subtitle (two lines of ~42)
NAME = os.environ.get('SEG_NAME', 'how_to_stop')
BURN = os.environ.get('BURN') == '1'                    # also burn subtitles into the picture
OUT = Path(__file__).parent / 'outputs'

MODELS = {
    'es': 'Helsinki-NLP/opus-mt-en-es',
    'fr': 'Helsinki-NLP/opus-mt-en-fr',
    'de': 'Helsinki-NLP/opus-mt-en-de',
    'zh': 'Helsinki-NLP/opus-mt-en-zh',
}
FONTS = {  # macOS fonts for burned-in subtitles; Chinese needs a CJK font
    'zh': '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
}
DEFAULT_FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'


def subtitle_lines() -> list[dict]:
    """Split the Whisper segments inside the window into short, timed subtitle lines."""
    seg = pxt.get_table(f'{DIR}/segments')
    rows = (seg.where((seg.end > SEG_START) & (seg.start < SEG_END)).order_by(seg.start)
            .select(seg.start, seg.end, seg.text).collect())
    lines = []
    for r in rows:
        words, chunks, cur = r['text'].split(), [], ''
        for w in words:
            if cur and len(cur) + 1 + len(w) > MAX_CHARS:
                chunks.append(cur)
                cur = w
            else:
                cur = f'{cur} {w}'.strip()
        if cur:
            chunks.append(cur)
        total = sum(len(c) for c in chunks) or 1
        t = max(r['start'], SEG_START)
        span = min(r['end'], SEG_END) - t
        for c in chunks:
            d = span * len(c) / total
            lines.append({'start': round(t - SEG_START, 2), 'end': round(t + d - SEG_START, 2), 'en': c})
            t += d
    return lines


def build() -> None:
    for name in ('subclip', 'sublines'):
        try:
            pxt.drop_table(f'{DIR}/{name}', force=True)
        except Exception:
            pass

    sub = pxt.create_table(f'{DIR}/sublines', {'idx': pxt.Int, 'start': pxt.Float,
                                                'end': pxt.Float, 'en': pxt.String})
    for lang, model in MODELS.items():
        src = zh_input(sub.en) if lang == 'zh' else sub.en
        sub.add_computed_column(**{f'{lang}_raw': translation(src, model_id=model)})
    sub.add_computed_column(en_wrapped=wrap(sub.en, 42))
    for lang in MODELS:
        sub.add_computed_column(**{lang: wrap(sub[f'{lang}_raw'], 20 if lang == 'zh' else 42)})

    lines = subtitle_lines()
    sub.insert([{**ln, 'idx': i} for i, ln in enumerate(lines)])
    print(f'{len(lines)} subtitle lines, {SEG_END - SEG_START:.0f} s segment, translated into {list(MODELS)}')

    texts = sub.order_by(sub.idx).select(sub.start, sub.end, sub.en_wrapped,
                                         *[sub[l] for l in MODELS]).collect()

    # A copy of the lecture with only the main video and audio streams (same timing): the
    # archive.org file carries extra streams that clip() cannot write back into an mp4.
    video = str(Path(__file__).parent / 'data' / 'how_to_speak_clean.mp4')
    sc = pxt.create_table(f'{DIR}/subclip', {'video': pxt.Video})
    sc.add_computed_column(segment=clip(sc.video, start_time=SEG_START, end_time=SEG_END))

    langs = ['en', *MODELS]
    if BURN:  # optional: one video per language with subtitles burned into the picture
        for lang in langs:
            col = 'en_wrapped' if lang == 'en' else lang
            expr = sc.segment
            for t in texts:  # one timed overlay per subtitle line
                expr = overlay_text(
                    expr, t[col], font=FONTS.get(lang, DEFAULT_FONT), font_size=20,
                    color='white', box=True, box_color='black', box_opacity=0.6,
                    box_border=[6, 12], vertical_align='bottom', vertical_margin=24,
                    line_spacing=4, start_time=t['start'], end_time=t['end'],
                )
            sc.add_computed_column(**{f'sub_{lang}': expr})

    print('cutting the segment' + (' and burning subtitles' if BURN else '') + ' ...')
    sc.insert([{'video': video}])
    OUT.mkdir(exist_ok=True)

    # Selectable subtitles: one subtitle file per language, plus one MP4 carrying all of them
    # as tracks the viewer can switch between (QuickTime, VLC, browsers via the .vtt files).
    for lang in langs:
        col = 'en_wrapped' if lang == 'en' else lang
        (OUT / f'{NAME}_{lang}.srt').write_text(to_srt(texts, col), encoding='utf-8')
        (OUT / f'{NAME}_{lang}.vtt').write_text(to_vtt(texts, col), encoding='utf-8')
    segment = sc.select(sc.segment).collect()[0]['segment']
    mux(segment, [OUT / f'{NAME}_{lang}.srt' for lang in langs], langs,
        OUT / f'{NAME}_multilingual.mp4')
    print('wrote', OUT / f'{NAME}_multilingual.mp4', 'and .srt / .vtt files for', langs)

    if BURN:
        r = sc.select(*[sc[f'sub_{l}'] for l in langs]).collect()[0]
        for lang in langs:
            dest = OUT / f'{NAME}_{lang}_burned.mp4'
            dest.write_bytes(Path(r[f'sub_{lang}']).read_bytes())
            print('wrote', dest)


def _ts(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}'


def to_srt(texts: list[dict], col: str) -> str:
    return '\n'.join(f"{i}\n{_ts(t['start'], ',')} --> {_ts(t['end'], ',')}\n{t[col]}\n"
                     for i, t in enumerate(texts, 1))


def to_vtt(texts: list[dict], col: str) -> str:
    return 'WEBVTT\n\n' + '\n'.join(f"{_ts(t['start'], '.')} --> {_ts(t['end'], '.')}\n{t[col]}\n"
                                   for t in texts)


ISO639_2 = {'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu', 'zh': 'zho'}
LABELS = {'en': 'English', 'es': 'Español', 'fr': 'Français', 'de': 'Deutsch', 'zh': '中文'}


def mux(video: str, srts: list[Path], langs: list[str], dest: Path) -> None:
    import subprocess

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', video]
    for s in srts:
        cmd += ['-i', str(s)]
    cmd += ['-map', '0:v:0', '-map', '0:a:0']
    for i in range(len(srts)):
        cmd += ['-map', f'{i + 1}:0']
    cmd += ['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text']
    for i, lang in enumerate(langs):
        cmd += [f'-metadata:s:s:{i}', f'language={ISO639_2[lang]}',
                f'-metadata:s:s:{i}', f'title={LABELS[lang]}']
    cmd += ['-disposition:s:0', 'default', str(dest)]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    build()
