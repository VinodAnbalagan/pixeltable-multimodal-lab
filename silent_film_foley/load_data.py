"""
Load the sound library (ESC-50 subset) and the silent clips into the Foley tables.

Usage (after `pxt schema update`):
    python -m silent_film_foley.load_data /foley
    python -m silent_film_foley.load_data pxt://<org>:main
"""

import sys
from pathlib import Path

import pixeltable as pxt

DATA = Path(__file__).parent / 'data'
SOUNDS_DIR = DATA / 'sounds'
PER_CATEGORY = 4  # clips per ESC-50 category; all 50 categories are used, uncurated

CLIPS = {
    'The General: cannon': 'general_cannon.mp4',
    'The General: bridge collapse': 'general_bridge.mp4',
    'The General: log on the tracks': 'general_log.mp4',
}


def export_sounds() -> list[dict]:
    """Write a balanced, uncurated ESC-50 subset to data/sounds/ and return rows to insert."""
    from datasets import Audio, load_dataset

    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('ashraq/esc50', split='train').cast_column('audio', Audio(decode=False))
    counts: dict[str, int] = {}
    rows = []
    for item in ds:
        label = item['category']
        if counts.get(label, 0) >= PER_CATEGORY:
            continue
        counts[label] = counts.get(label, 0) + 1
        name = f'{label}_{counts[label]}'
        path = SOUNDS_DIR / f'{name}.wav'
        if not path.exists():
            path.write_bytes(item['audio']['bytes'])
        rows.append({'name': name, 'label': label, 'audio': str(path)})
    short = sorted(c for c, n in counts.items() if n < PER_CATEGORY)
    print(f'sound library: {len(rows)} clips across {len(counts)} categories')
    if short:
        print(f'  fewer than {PER_CATEGORY} clips: {short}')
    return rows


def table(target: str, name: str) -> pxt.Table:
    base = target.rstrip('/')
    if not base.startswith('pxt://'):
        base = base.lstrip('/')  # local catalog: '/foley' -> 'foley'
    return pxt.get_table(f'{base}/{name}')


def main(target: str) -> None:
    sounds = table(target, 'sounds')
    if sounds.count() == 0:
        rows = export_sounds()
        print(f'inserting {len(rows)} sounds')
        sounds.insert(rows)

    clips = table(target, 'clips')
    existing = {r['title'] for r in clips.select(clips.title).collect()}
    new = [
        {'title': title, 'video': str(DATA / file)}
        for title, file in CLIPS.items()
        if title not in existing and (DATA / file).exists()
    ]
    if new:
        print(f'inserting {len(new)} clips')
        clips.insert(new)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/foley')
