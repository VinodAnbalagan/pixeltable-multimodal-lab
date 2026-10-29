"""
Build a Foley sound library from FSD50K (eval split) with a CLAP index.

Selection (no hand-picking):
  - drop clips labelled Music, Speech or Singing
  - keep only CC0 and CC-BY clips (no NC, no Sampling+), so the library can be used in public demos
  - at most PER_LABEL clips per most-specific label, for balance
  - trim each clip to its first 10 s (CLAP's input length), so embeddings are deterministic

Usage:
    python -m silent_film_foley.fsd_library            # download, trim, index into foley_v3/library
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import pixeltable as pxt

from silent_film_foley.models import clap_audio, clap_text

REPO = 'Fhrozen/FSD50k'
DIR = os.environ.get('FOLEY_DIR', 'foley_v3')
PER_LABEL = int(os.environ.get('PER_LABEL', '10'))
OUT = Path(__file__).parent / 'data' / 'fsd'
EXCLUDE = {'Music', 'Speech', 'Singing', 'Musical_instrument', 'Human_voice'}
MAX_SECONDS = 10.0


def _licence_ok(url: str) -> str | None:
    if 'publicdomain/zero' in url:
        return 'CC0'
    if '/licenses/by/' in url:
        return 'CC-BY'
    return None


def select() -> list[dict]:
    from huggingface_hub import hf_hub_download

    labels_csv = hf_hub_download(REPO, 'labels/eval.csv', repo_type='dataset')
    info_json = hf_hub_download(REPO, 'metadata/eval_clips_info_FSD50K.json', repo_type='dataset')
    info = json.loads(Path(info_json).read_text())

    per_label: dict[str, int] = defaultdict(int)
    chosen = []
    with open(labels_csv, newline='') as f:
        for r in csv.DictReader(f):
            labels = r['labels'].split(',')
            if EXCLUDE & set(labels):
                continue
            meta = info.get(r['fname'], {})
            lic = _licence_ok(meta.get('license', ''))
            if not lic:
                continue
            leaf = labels[0]
            if per_label[leaf] >= PER_LABEL:
                continue
            per_label[leaf] += 1
            chosen.append({'fname': r['fname'], 'label': leaf, 'labels': labels,
                           'licence': lic, 'uploader': meta.get('uploader', ''),
                           'title': meta.get('title', '')})
    print(f'selected {len(chosen)} clips across {len(per_label)} labels')
    return chosen


def download_and_trim(chosen: list[dict]) -> None:
    import soundfile as sf
    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)
    todo = [c for c in chosen if not (OUT / f"fsd_{c['fname']}.wav").exists()]
    if todo:
        root = snapshot_download(REPO, repo_type='dataset', max_workers=8,
                                 allow_patterns=[f"clips/eval/{c['fname']}.wav" for c in todo])
        for c in todo:
            data, sr = sf.read(Path(root) / 'clips' / 'eval' / f"{c['fname']}.wav", always_2d=True)
            mono = data.mean(axis=1)[: int(MAX_SECONDS * sr)]
            sf.write(OUT / f"fsd_{c['fname']}.wav", mono, sr)


def build() -> None:
    chosen = select()
    download_and_trim(chosen)

    pxt.create_dir(DIR, if_exists='ignore')
    lib = pxt.create_table(f'{DIR}/library', {
        'name': pxt.String, 'label': pxt.String, 'labels': pxt.Json, 'licence': pxt.String,
        'uploader': pxt.String, 'title': pxt.String, 'audio': pxt.Audio,
    }, if_exists='ignore')
    if lib.count() == 0:
        lib.add_embedding_index('audio', audio_embed=clap_audio, string_embed=clap_text)
    have = {r['name'] for r in lib.select(lib.name).collect()}
    rows = [{'name': f"fsd_{c['fname']}", 'label': c['label'], 'labels': c['labels'],
             'licence': c['licence'], 'uploader': c['uploader'], 'title': c['title'],
             'audio': str(OUT / f"fsd_{c['fname']}.wav")}
            for c in chosen if f"fsd_{c['fname']}" not in have]
    if rows:
        lib.insert(rows)
    print(f'library: {lib.count()} sounds')


if __name__ == '__main__':
    build()
