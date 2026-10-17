# Pixeltable Multimodal Lab

Multimodal AI experiments built on [Pixeltable](https://docs.pixeltable.com) with open models, run locally and on Pixeltable Cloud (beta).

## Projects

### 1. Hallucination Spotter (`hallucination_spotter/`)

Does a vision-language model mention objects that aren't in the frame?

```
video -> frames (1 fps) -> Gemma 4 caption (LM Studio) -> objects mentioned
                        -> YOLOX detections            -> objects detected
      -> hallucinated = mentioned - detected   (CHAIR-i)
      -> omitted      = detected - mentioned
      -> annotated frames -> make_video -> MP4 / GIF
```

**v0 result** (19 s street scene, reasoning off): 52 COCO objects mentioned, 4 flagged (CHAIR-i 8%). Each flagged object was detected in the neighbouring frames, so these look like detector misses rather than model hallucinations. The clearer pattern was omission: people were detected in most frames but rarely mentioned. Word-matching also produced false flags ("pedestrian overpass" → person, truck "beds" → bed), which the matcher now filters.

**Next:** count an object as present if it is detected within ±1 s; compare reasoning on/off; test stylised animation (Blender open movies).

### 2. Silent Film Foley (`silent_film_foley/`)

Can a pipeline choose sound effects for a silent film?

```
sounds (ESC-50, 200 clips, 50 categories, uncurated) -> CLAP audio embedding index
clips (Buster Keaton, The General, 1926) -> scenes (frame every 2 s)
      -> BLIP caption -> sound prompt -> nearest sounds (CLAP text search)
      -> soundtrack -> with_audio -> MP4
```

Runs on Pixeltable Cloud (1 vCPU / 2 GB): BLIP, CLAP and the audio index all compute there.

**v0 result:** the pipeline works end to end, but the sound choices are poor. Captions describe what is visible, not what happens, so a visible cannon triggers explosion sounds before it fires; single frames miss events; low-confidence matches still play; and the library lacks key sounds (explosion, crash, outdoor train). BLIP errors also carry through (a caption with a hallucinated dog retrieves sheep and dog sounds).

**v1 plan:** short video segments with several frames each, a reasoning model that describes the action and the expected sound events using the previous segment as context, CLAP retrieval with voting across several phrasings, a confidence threshold (silence when unsure), separate ambience and event layers, and a larger sound library (FSD50K subset).

## Running

```bash
uv sync && source .venv/bin/activate

# Hallucination Spotter (local; LM Studio serving a vision model on :1234)
python -m hallucination_spotter.spotter
python -m hallucination_spotter.render

# Silent Film Foley (clips cut into silent_film_foley/data/)
pxt schema update silent_film_foley/app.py pxt://<org>:main   # or /foley for local
python -m silent_film_foley.load_data pxt://<org>:main
python -m silent_film_foley.render pxt://<org>:main
```

### Data

- **Hallucination Spotter:** no files needed; it uses a sample video from Pixeltable's docs. The VLM is any OpenAI-compatible server with a vision model: set `VLM_BASE_URL` and `VLM_MODEL` (defaults: LM Studio on `localhost:1234`, `google/gemma-4-12b`).
- **Silent Film Foley:** `load_data` downloads the ESC-50 sounds from Hugging Face automatically. The three Keaton clips and example outputs are attached to the GitHub release; place the clips in `silent_film_foley/data/`.

Media files are not committed to the repository.

## Credits

- *The General* (1926), Buster Keaton: public domain.
- ESC-50: K. J. Piczak, *ESC: Dataset for Environmental Sound Classification* (CC BY-NC).
- Models: Gemma 4, YOLOX, BLIP (Salesforce), CLAP (LAION).
- CHAIR metric: Rohrbach et al., *Object Hallucination in Image Captioning* (2018).

## Beta feedback

Notes for the Pixeltable team are in [FEEDBACK.md](FEEDBACK.md).
