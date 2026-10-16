"""UDFs for the Hallucination Spotter pipeline."""

import base64
import io
import re

import PIL.Image
from openai import OpenAI

import pixeltable as pxt

LMSTUDIO_URL = 'http://localhost:1234/v1'
VLM_MODEL = 'google/gemma-4-12b'

PROMPT = 'Describe this image in one or two sentences. Mention every object you can see.'

# YOLOX label ids map to the 80 COCO classes, in this order.
COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)

# Common caption words mapped to their COCO class.
SYNONYMS = {
    'people': 'person', 'man': 'person', 'men': 'person', 'woman': 'person',
    'women': 'person', 'pedestrian': 'person', 'boy': 'person', 'girl': 'person',
    'child': 'person', 'children': 'person', 'kid': 'person', 'crowd': 'person',
    'rider': 'person', 'driver': 'person',
    'bike': 'bicycle', 'cyclist': 'bicycle',
    'taxi': 'car', 'sedan': 'car', 'suv': 'car', 'van': 'car',
    'motorbike': 'motorcycle', 'scooter': 'motorcycle', 'moped': 'motorcycle',
    'lorry': 'truck', 'phone': 'cell phone', 'bag': 'handbag', 'purse': 'handbag',
    'sofa': 'couch', 'television': 'tv', 'table': 'dining table',
}

# Phrases where a COCO word is used as a modifier, not as the object itself
# ("pedestrian overpass" is not a person; a truck "bed" is not a bed).
NON_OBJECT_PHRASES = (
    r'pedestrian (?:overpass|bridge|crossing|walkway|path)s?',
    r'(?:truck|cargo|pickup|flat) beds?',
    r'on (?:their |its )?beds',
    r'flower beds?',
    r'traffic (?:cones?|lanes?)',
)

VOCAB = {name: name for name in COCO_CLASSES} | SYNONYMS

_client = OpenAI(base_url=LMSTUDIO_URL, api_key='lm-studio', timeout=600)


def _caption(frame: PIL.Image.Image, thinking: bool) -> str:
    img = frame.convert('RGB')
    img.thumbnail((768, 768))  # fewer image tokens, faster and cooler on a laptop
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    extra = {} if thinking else {'reasoning_effort': 'none'}
    response = _client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': PROMPT},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            ],
        }],
        temperature=0,
        max_tokens=8192 if thinking else 512,
        extra_body=extra,
    )
    content = response.choices[0].message.content or ''
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()


@pxt.udf
def describe(frame: PIL.Image.Image) -> str:
    """Caption a frame with the local VLM, reasoning disabled."""
    return _caption(frame, thinking=False)


@pxt.udf
def describe_thinking(frame: PIL.Image.Image) -> str:
    """Caption a frame with the local VLM, reasoning enabled."""
    return _caption(frame, thinking=True)


@pxt.udf
def mentioned_objects(caption: str) -> list[str]:
    """COCO classes named in a caption (plurals and common synonyms included)."""
    text = (caption or '').lower()
    for phrase in NON_OBJECT_PHRASES:
        text = re.sub(phrase, ' ', text)
    found = {
        coco_class
        for word, coco_class in VOCAB.items()
        if re.search(rf'\b{re.escape(word)}(?:s|es)?\b', text)
    }
    return sorted(found)


@pxt.udf
def detected_objects(detections: dict) -> list[str]:
    """COCO class names from a YOLOX detection result."""
    labels = (detections or {}).get('labels') or []
    return sorted({COCO_CLASSES[int(i)] for i in labels if 0 <= int(i) < len(COCO_CLASSES)})


@pxt.udf
def hallucinated_objects(mentioned: list[str], detected: list[str]) -> list[str]:
    """Objects mentioned in the caption but not found by the detector."""
    return sorted(set(mentioned) - set(detected))


@pxt.udf
def omitted_objects(mentioned: list[str], detected: list[str]) -> list[str]:
    """Objects found by the detector but not mentioned in the caption."""
    return sorted(set(detected) - set(mentioned))


@pxt.udf
def hallucination_rate(mentioned: list[str], hallucinated: list[str]) -> float:
    """Per-frame CHAIR-i: hallucinated / mentioned."""
    return len(hallucinated) / len(mentioned) if mentioned else 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_WHITE = (240, 240, 240)
_GREEN = (80, 200, 120)
_RED = (235, 64, 52)
_YELLOW = (250, 200, 40)
_GRAY = (150, 150, 150)
_BG = (12, 12, 12)


def _word_class(word: str) -> str | None:
    w = re.sub(r'[^a-z]', '', word.lower())
    for cand in (w, w[:-2] if w.endswith('es') else '', w[:-1] if w.endswith('s') else ''):
        if cand and cand in VOCAB:
            return VOCAB[cand]
    return None


def _wrap(draw, words, font, width):
    lines, line = [], []
    for word in words:
        trial = ' '.join(w for w, _ in line + [(word, None)])
        if line and draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = []
        line.append((word, None))
    if line:
        lines.append(line)
    return [[w for w, _ in ln] for ln in lines]


@pxt.udf
def annotate(
    frame: PIL.Image.Image,
    caption: str,
    mentioned: list[str],
    hallucinated: list[str],
    omitted: list[str],
    second: int,
) -> PIL.Image.Image:
    """Frame with the caption underneath: green = confirmed by the detector,
    red = not found by the detector, plus the objects the caption left out."""
    from PIL import ImageDraw, ImageFont

    width, pad = 960, 24
    img = frame.convert('RGB')
    img = img.resize((width, round(img.height * width / img.width)))

    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    probe = ImageDraw.Draw(PIL.Image.new('RGB', (1, 1)))
    lines = _wrap(probe, (caption or '').split(), font, width - 2 * pad)
    line_h, panel_h = 32, 0
    panel_h = pad + len(lines) * line_h + 12 + 2 * 28 + pad

    canvas = PIL.Image.new('RGB', (width, img.height + panel_h), _BG)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, 200, 40), fill=_BG)
    draw.text((12, 8), f'Gemma 4  |  t = {second}s', font=small, fill=_WHITE)

    y = img.height + pad
    for line in lines:
        x = pad
        for word in line:
            cls = _word_class(word)
            color = _WHITE
            if cls in hallucinated:
                color = _RED
            elif cls in mentioned:
                color = _GREEN
            draw.text((x, y), word, font=font, fill=color)
            x += draw.textlength(word + ' ', font=font)
        y += line_h

    y += 12
    flagged = ', '.join(hallucinated) if hallucinated else 'none'
    missed = ', '.join(omitted) if omitted else 'none'
    draw.text((pad, y), f'Not found by detector: {flagged}', font=small,
              fill=_RED if hallucinated else _GRAY)
    draw.text((pad, y + 28), f'Detected but not mentioned: {missed}', font=small,
              fill=_YELLOW if omitted else _GRAY)
    return canvas
