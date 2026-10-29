"""UDFs for multilingual subtitles."""

import textwrap

import pixeltable as pxt


@pxt.udf
def zh_input(text: str) -> str:
    """opus-mt-en-zh needs a target-language token at the start of the sentence."""
    return f'>>cmn_Hans<< {text}'


@pxt.udf
def wrap(text: str, width: int) -> str:
    """Wrap a subtitle into at most two lines (CJK text is wrapped by characters)."""
    # Typographic apostrophe: a straight ' inside overlay_text's ffmpeg filter ends the quoted
    # text, so a later comma is read as a filter separator.
    text = (text or '').strip().replace("'", '’')
    if any('一' <= ch <= '鿿' for ch in text):
        lines = [text[i:i + width] for i in range(0, len(text), width)]
    else:
        lines = textwrap.wrap(text, width=width)
    if len(lines) > 2:  # keep everything, but on two lines
        mid = (len(lines) + 1) // 2
        lines = [' '.join(lines[:mid]) if ' ' in text else ''.join(lines[:mid]),
                 ' '.join(lines[mid:]) if ' ' in text else ''.join(lines[mid:])]
    return '\n'.join(lines)
