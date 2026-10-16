"""
Silent Film Foley

A silent clip gets sound effects chosen by AI:

    clip -> frames (every 2 s) -> BLIP caption -> sound prompt
         -> nearest sound in the CLAP-indexed library (audio <-> text similarity)

Create the tables (local catalog or Pixeltable Cloud):
    pxt schema update silent_film_foley/app.py /foley
    pxt schema update silent_film_foley/app.py pxt://<org>:main
"""

import pixeltable as pxt
from pixeltable.functions.huggingface import image_captioning
from pixeltable.functions.video import frame_iterator

from silent_film_foley.models import clap_audio, clap_text, sound_prompt

TableModel = pxt.model_base()

BLIP_ID = 'Salesforce/blip-image-captioning-base'


class Sounds(TableModel, name='sounds'):
    """Sound-effect library, searchable by text through a CLAP embedding index."""

    name: pxt.String
    label: pxt.String
    audio: pxt.Audio

    __indexes__ = [
        pxt.EmbeddingIndex(audio, audio_embed=clap_audio, string_embed=clap_text, name='clap_idx')
    ]


@pxt.query
def nearest_sounds(prompt: str):
    sim = Sounds.audio.similarity(string=prompt, idx='clap_idx')
    return Sounds.order_by(sim, asc=False).limit(3).select(Sounds.name, Sounds.label, score=sim)


class Clips(TableModel, name='clips'):
    """Silent film clips."""

    title: pxt.String
    video: pxt.Video


class Scenes(
    TableModel,
    name='scenes',
    base=Clips,
    iterator=frame_iterator(Clips.video, fps=0.5),
):
    """One row per sampled frame; 'frame' and 'frame_attrs' come from the iterator."""

    caption = image_captioning(
        frame,  # noqa: F821
        model_id=BLIP_ID,
        model_kwargs={'max_new_tokens': 30},
    )
    prompt = sound_prompt(caption)
    matches = nearest_sounds(prompt)
