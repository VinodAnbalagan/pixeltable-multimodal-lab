"""Sound library for Foley v1 and the CLAP search query. Import after build_library() has run."""

import os

import pixeltable as pxt

# local: 'foley_v1/sounds'; Pixeltable Cloud: e.g. 'pxt://vinod-lab:main/sounds'
LIBRARY = os.environ.get('FOLEY_LIBRARY', 'foley_v1/sounds')

SOUNDS = pxt.get_table(LIBRARY)


@pxt.query
def nearest(prompt: str):
    sim = SOUNDS.audio.similarity(string=prompt)
    return SOUNDS.order_by(sim, asc=False).limit(5).select(SOUNDS.name, SOUNDS.label, score=sim)
