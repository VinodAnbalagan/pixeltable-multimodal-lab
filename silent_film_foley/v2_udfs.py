"""UDFs for Foley v2 (film strip)."""

import PIL.Image

import pixeltable as pxt

STILL_SIZE = (512, 512)  # bounding box; aspect ratio is kept


@pxt.udf
def frame_time(attrs: dict) -> float:
    """Timestamp of a frame in seconds, from frame_iterator's frame_attrs."""
    return round(float((attrs or {}).get('time', 0.0)), 3)


@pxt.udf
def still(frame: PIL.Image.Image) -> PIL.Image.Image:
    """A small stored copy of the frame: the screenshot the VLM will see."""
    img = frame.convert('RGB')
    img.thumbnail(STILL_SIZE)
    return img
