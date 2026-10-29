"""
Foley v3: action log written from the stored stills (no VLM) -> sound search -> soundtrack.

    ground_truth/<key>_foley.csv -> foley_v3/log (three sound-word queries per row)
        -> CLAP search per query (ESC-50 library) -> vote -> chosen sound
    then: FOLEY_DIR=foley_v3 TAG=v3 python -m silent_film_foley.v1_render

Usage:
    python -m silent_film_foley.oracle                            # local
    FOLEY_DIR=pxt://<org>:main/foley_v3 FOLEY_LIBRARY=pxt://<org>:main/sounds \\
        python -m silent_film_foley.oracle                        # Pixeltable Cloud
"""

import csv
import os
from pathlib import Path

import pixeltable as pxt

from silent_film_foley.v1_library import nearest
from silent_film_foley.v1_udfs import vote

DIR = os.environ.get('FOLEY_DIR', 'foley_v3')  # Cloud: e.g. 'pxt://vinod-lab:main/foley_v3'
HERE = Path(__file__).parent
CLIPS = {
    'cannon': ('The General (1926): the cannon', 'general_cannon.mp4'),
    'spring': ('Spring (2019): the monolith', 'spring_monolith.mp4'),
}


def main() -> None:
    """Rebuild the log from every clip that has a ground_truth/<key>_foley.csv."""
    keys = [k for k in CLIPS if (HERE / 'ground_truth' / f'{k}_foley.csv').exists()]
    pxt.create_dir(DIR, if_exists='ignore')
    clips = pxt.create_table(f'{DIR}/clips', {'title': pxt.String, 'video': pxt.Video},
                             if_exists='ignore')
    have = {r['title'] for r in clips.select(clips.title).collect()}
    new = [{'title': CLIPS[k][0], 'video': str(HERE / 'data' / CLIPS[k][1])}
           for k in keys if CLIPS[k][0] not in have]
    if new:
        clips.insert(new)

    # The log is rebuilt on every run (text search only), so the schema can change freely.
    try:
        pxt.drop_table(f'{DIR}/log', force=True)
    except Exception:
        pass
    log = pxt.create_table(f'{DIR}/log', {
        'title': pxt.String, 'start': pxt.Float, 'end': pxt.Float, 'layer': pxt.String,
        'source': pxt.String, 'material': pxt.String, 'action': pxt.String,
        'heard_from': pxt.String, 'evidence': pxt.String,
        'gain': pxt.Float | None, 'gain_end': pxt.Float | None,  # optional per row
        'q1': pxt.String, 'q2': pxt.String, 'q3': pxt.String,
    }, if_exists='ignore')
    for i in (1, 2, 3):
        log.add_computed_column(**{f'm{i}': nearest(log[f'q{i}'])}, if_exists='ignore')
    log.add_computed_column(candidates=vote(log.m1, log.m2, log.m3), if_exists='ignore')

    rows = []
    for k in keys:
        title = CLIPS[k][0]
        with open(HERE / 'ground_truth' / f'{k}_foley.csv', newline='') as f:
            rows += [{
                'title': title, 'start': float(r['start']), 'end': float(r['end']),
                'layer': r['layer'], 'source': r['source'], 'material': r['material'],
                'action': r['action'], 'heard_from': r['heard_from'], 'evidence': r['evidence'],
                'gain': float(r['gain']) if r.get('gain') else None,
                'gain_end': float(r['gain_end']) if r.get('gain_end') else None,
                'q1': r['query_1'], 'q2': r['query_2'], 'q3': r['query_3'],
            } for r in csv.DictReader(f)]
    log.insert(rows)

    build_generated(keys)

    title = None
    for r in log.order_by(log.title, log.start).select(
            log.title, log.start, log.end, log.layer, log.source, log.action, log.q1,
            log.candidates).collect():
        if r['title'] != title:
            title = r['title']
            print(f'\n===== {title} =====')
        top = (r['candidates'] or [{}])[0]
        print(f"{r['start']:5.1f}-{r['end']:5.1f} {r['layer']:8s} {r['source']}: {r['action']}")
        print(f"            '{r['q1']}' -> {top.get('name')} "
              f"(sim {top.get('best_sim', 0):.2f}, votes {top.get('votes')})")


def build_generated(keys: list[str]) -> None:
    """Score and sound design from ground_truth/<key>_score.csv, generated with MusicGen.
    Generation is slow, so rows are kept and only new (prompt, seed, timing) rows are generated."""
    from silent_film_foley.gen_udfs import musicgen

    gen = pxt.create_table(f'{DIR}/generated', {
        'title': pxt.String, 'start': pxt.Float, 'end': pxt.Float, 'role': pxt.String,
        'prompt': pxt.String, 'seed': pxt.Int, 'seconds': pxt.Float,
        'gain': pxt.Float | None, 'gain_end': pxt.Float | None,
    }, if_exists='ignore')
    gen.add_computed_column(audio=musicgen(gen.prompt, gen.seconds, gen.seed), if_exists='ignore')

    have = {(r['title'], r['prompt'], r['seed'], r['start'], r['end'])
            for r in gen.select(gen.title, gen.prompt, gen.seed, gen.start, gen.end).collect()}
    wanted = set()
    new = []
    for k in keys:
        path = HERE / 'ground_truth' / f'{k}_score.csv'
        if not path.exists():
            continue
        title = CLIPS[k][0]
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                start, end, seed = float(r['start']), float(r['end']), int(r['seed'])
                wanted.add((title, r['prompt'], seed, start, end))
                gains = {'role': r['role'],
                         'gain': float(r['gain']) if r.get('gain') else None,
                         'gain_end': float(r['gain_end']) if r.get('gain_end') else None}
                if (title, r['prompt'], seed, start, end) in have:
                    # same audio, maybe new mix settings: update in place, no regeneration
                    gen.update(gains, where=(gen.title == title) & (gen.prompt == r['prompt'])
                               & (gen.seed == seed) & (gen.start == start) & (gen.end == end))
                    continue
                new.append({'title': title, 'start': start, 'end': end, 'role': r['role'],
                            'prompt': r['prompt'], 'seed': seed, 'seconds': end - start,
                            'gain': float(r['gain']) if r.get('gain') else None,
                            'gain_end': float(r['gain_end']) if r.get('gain_end') else None})
    for t, p, s, a, b in have - wanted:  # rows no longer in any CSV
        gen.delete((gen.title == t) & (gen.prompt == p) & (gen.seed == s)
                   & (gen.start == a) & (gen.end == b))
    if new:
        print(f'generating {len(new)} score / sound-design pieces with MusicGen ...')
        gen.insert(new)


if __name__ == '__main__':
    main()
