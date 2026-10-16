"""CLAP audio/text embeddings and prompt helpers for Silent Film Foley."""

import functools
import re

import numpy as np

import pixeltable as pxt

CLAP_ID = 'laion/clap-htsat-unfused'
CLAP_DIM = 512
CLAP_SR = 48_000


@functools.cache
def _clap():
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(CLAP_ID).eval()
    processor = ClapProcessor.from_pretrained(CLAP_ID)
    return model, processor


def _features(output) -> np.ndarray:
    import torch

    emb = output if torch.is_tensor(output) else output.pooler_output
    return emb[0].detach().float().numpy()


@pxt.udf
def clap_audio(audio: pxt.Audio) -> pxt.Array[(CLAP_DIM,), pxt.Float]:
    """CLAP embedding of an audio file (resampled to 48 kHz mono)."""
    import librosa
    import torch

    model, processor = _clap()
    wav, _ = librosa.load(audio, sr=CLAP_SR, mono=True)
    try:
        inputs = processor(audio=wav, sampling_rate=CLAP_SR, return_tensors='pt')
    except TypeError:  # older transformers
        inputs = processor(audios=wav, sampling_rate=CLAP_SR, return_tensors='pt')
    with torch.no_grad():
        return _features(model.get_audio_features(**inputs))


@pxt.udf
def clap_text(text: str) -> pxt.Array[(CLAP_DIM,), pxt.Float]:
    """CLAP embedding of a text description, in the same space as clap_audio."""
    import torch

    model, processor = _clap()
    inputs = processor(text=[text], return_tensors='pt', padding=True)
    with torch.no_grad():
        return _features(model.get_text_features(**inputs))


_CAPTION_PREFIX = re.compile(
    r'^(?:an? )?(?:old |vintage )?(?:black and white |black-and-white )?'
    r'(?:photo|picture|image|still|scene|film still) of ',
)


@pxt.udf
def sound_prompt(caption: str) -> str:
    """Turn an image caption into a CLAP-friendly sound description."""
    text = _CAPTION_PREFIX.sub('', (caption or '').strip().lower())
    return f'the sound of {text}'
