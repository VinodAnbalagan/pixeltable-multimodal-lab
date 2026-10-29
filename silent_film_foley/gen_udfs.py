"""Generated score and sound design with MusicGen (text -> music)."""

import functools
import hashlib
import os
import tempfile
from pathlib import Path

import pixeltable as pxt

MUSICGEN_ID = os.environ.get('MUSICGEN_ID', 'facebook/musicgen-small')
MAX_SECONDS = 30.0  # MusicGen is trained on 30 s segments


@functools.cache
def _musicgen():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device = os.environ.get('MUSICGEN_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_ID).to(device).eval()
    return model, AutoProcessor.from_pretrained(MUSICGEN_ID), device


@pxt.udf
def musicgen(prompt: str, seconds: float, seed: int) -> pxt.Audio:
    """Generate `seconds` of music for a text prompt (deterministic for a given seed)."""
    import soundfile as sf
    import torch

    model, processor, device = _musicgen()
    seconds = min(float(seconds), MAX_SECONDS)
    torch.manual_seed(int(seed))
    inputs = processor(text=[prompt], padding=True, return_tensors='pt').to(device)
    frame_rate = model.config.audio_encoder.frame_rate  # tokens per second of audio
    with torch.no_grad():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0,
                               max_new_tokens=int(seconds * frame_rate) + 4)
    sr = model.config.audio_encoder.sampling_rate
    wav = audio[0, 0].float().cpu().numpy()
    key = hashlib.sha1(f'{MUSICGEN_ID}|{prompt}|{seconds}|{seed}'.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f'musicgen_{key}.wav'
    sf.write(path, wav, sr)
    return str(path)
