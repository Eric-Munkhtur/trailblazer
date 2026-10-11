"""Package recorded evidence and training-only lookup mappings for inference.

No labels or outcome fields enter the inference inputs. Assets stay gitignored.
"""
import argparse
import ast
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normal(s):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', s.lower()).split())


def name_keys(name):
    # Conservative typography aliases only. Do not drop stages/subtypes or use
    # fuzzy semantic matches. Conflicting mappings still remove an alias.
    variants = {name, re.sub(r"['’]s\b", '', name)}
    for variant in list(variants):
        parts = [p.strip() for p in variant.split(',')]
        if len(parts) == 2 and all(parts):
            variants.add(parts[1] + ' ' + parts[0])
    return {normal(v) for v in variants}


p = argparse.ArgumentParser()
p.add_argument('--results-root', type=Path, required=True)
p.add_argument('--assets', type=Path, default=Path('data/predictions'))
args = p.parse_args()
root, assets = args.results_root, args.assets
arms = {
    'attention': ('Verdia · sentence attention', root / 'continuation_s0/attention_s0'),
    'control': ('HINT · matched control', root / 'continuation_s0/original_s0'),
    'conditioned': ('Verdia · context attention', root / 'conditioned_s0'),
}
models, records, membership = {}, {}, {}
reference_ids = None
for arm, (name, folder) in arms.items():
    metrics = json.loads((folder / 'metrics.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    rows = list(csv.DictReader((folder / 'predictions_test.csv').open()))
    ids = {row['nctid'] for row in rows}
    if len(rows) != 1654 or len(ids) != len(rows):
        raise ValueError('Unexpected or duplicate benchmark rows')
    if reference_ids is not None and ids != reference_ids:
        raise ValueError('Benchmark model IDs differ')
    reference_ids = ids
    for row in rows:
        score = float(row['probability'])
        if not re.fullmatch(r'NCT\d{8}', row['nctid']) or not 0 <= score <= 1:
            raise ValueError('Invalid prediction')
        records.setdefault(row['nctid'], {})[arm] = score
    m = metrics['test']
    matrix = m['confusion_matrix']
    models[arm] = {
        'name': name, 'seed': config['seed'], 'selectedEpoch': metrics['best_epoch'],
        'checkpointSha256': metrics['best_checkpoint_sha256'],
        'predictionsSha256': digest(folder / 'predictions_test.csv'),
        'metrics': {'n': m['n'], 'rocAuc': m['roc_auc'],
                    'averagePrecision': m['average_precision'], 'f1': m['f1_at_0.5'],
                    'accuracy': (matrix[0][0] + matrix[1][1]) / m['n']},
    }
    if arm == 'attention':
        input_hashes = config['input_sha256']
        upstream = config['upstream_commit']

variants = defaultdict(set)
for split in ('train', 'valid', 'test'):
    path = assets / f'phase_II_{split}.csv'
    if digest(path) != input_hashes[f'phase_II_{split}.csv']:
        raise ValueError(f'CSV hash mismatch: {split}')
    for row in csv.DictReader(path.open()):
        if row['nctid'] in membership:
            raise ValueError('Split overlap')
        membership[row['nctid']] = split
        if split != 'train':
            continue
        diseases = ast.literal_eval(row['diseases'])
        groups = [ast.literal_eval(s) for s in ast.literal_eval(row['icdcodes'])]
        if len(diseases) != len(groups):
            continue
        for name, codes in zip(diseases, groups):
            codes = tuple(sorted(set(c for c in codes if c)))
            if codes:
                for key in name_keys(name):
                    variants[key].add(codes)

# An exact normalized name must have one consistent mapping in training data.
mapping = {k: list(next(iter(v))) for k, v in variants.items() if len(v) == 1}
evidence = {
    'schemaVersion': 1, 'dataset': 'TOP Phase II', 'nTest': 1654,
    'recordedDate': '2026-10-11', 'upstreamCommit': upstream,
    'upstreamUrl': 'https://github.com/futianfan/clinical-trial-outcome-prediction',
    'inputSha256': input_hashes, 'models': models, 'records': records,
    'conclusion': 'Neither attention modification demonstrated a reliable improvement over the matched control.',
}
for filename, data in [('benchmark.json', evidence), ('membership.json', membership),
                       ('disease-map.json', mapping)]:
    (assets / filename).write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')
print(json.dumps({'trials': len(records), 'diseaseNamesFromTraining': len(mapping),
                  'benchmarkBytes': (assets / 'benchmark.json').stat().st_size}))
