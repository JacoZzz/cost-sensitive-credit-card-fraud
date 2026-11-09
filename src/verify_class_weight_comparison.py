"""Audit paired class-weight differences using aligned real resampled records."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/class_weight/comparison'


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    assert s['status'] == 'SUCCESS' and s['sealed_test_accessed'] is False
    for name, digest in s['input_files_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    draws = pd.read_csv(OUT / 'bootstrap_replicates.csv.gz')
    folds = pd.read_csv(OUT / 'fold_bootstrap_replicates.csv.gz')
    assert len(draws) == 5000 and len(folds) == 25000
    assert not folds.duplicated(['replicate', 'outer_fold']).any()
    for name, m in s['metrics'].items():
        np.testing.assert_allclose(folds.groupby('replicate')[name].mean(), draws[name], atol=1e-14)
        np.testing.assert_allclose(draws[name].std(ddof=1), m['bootstrap_SE'], atol=1e-14)
        np.testing.assert_allclose(np.quantile(draws[name], [.025, .975]), m['conditional_95_percentile_interval'], atol=1e-14)
    for metric in ['AP', 'default_mean_cost', 'tuned_mean_cost']:
        np.testing.assert_allclose(draws[f'balanced_{metric}'] - draws[f'unweighted_{metric}'],
                                   draws[f'delta_{metric}'], atol=1e-14)
    base = pd.read_csv(ROOT / 'reports/baseline/development_outer_oof.csv.gz')
    balanced = pd.read_csv(ROOT / 'reports/class_weight/development_outer_oof.csv.gz').set_index('raw_row_id')
    f0 = base[base.outer_fold == 1].sort_values('probability', ascending=False, kind='stable')
    f1 = balanced.loc[f0.raw_row_id]
    y = f0.y_true.to_numpy()
    rng = np.random.default_rng(42)
    weights = np.zeros((25, len(y)), dtype=np.int64)
    for label in [0, 1]:
        idx = np.flatnonzero(y == label)
        weights[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=25)
    for i in range(3):
        indices = np.repeat(np.arange(len(y)), weights[i])
        row = folds[(folds.outer_fold == 1) & (folds.replicate == i + 1)].iloc[0]
        for method, data in [('unweighted', f0), ('balanced', f1)]:
            sample = data.iloc[indices]
            np.testing.assert_allclose(average_precision_score(sample.y_true, sample.probability), row[f'{method}_AP'], atol=1e-14)
            for mode in ['default', 'tuned']:
                fp = ((sample[f'{mode}_prediction'] == 1) & (sample.y_true == 0)).sum()
                fn = ((sample[f'{mode}_prediction'] == 0) & (sample.y_true == 1)).sum()
                np.testing.assert_allclose((fp + 50 * fn) / len(sample), row[f'{method}_{mode}_mean_cost'], atol=1e-14)
    for method, directory in [('unweighted', 'baseline'), ('balanced', 'class_weight')]:
        training = json.loads((ROOT / f'reports/{directory}/summary.json').read_text())
        for metric in ['AP', 'default_mean_cost', 'tuned_mean_cost']:
            np.testing.assert_allclose(s['metrics'][f'{method}_{metric}']['point_estimate'],
                                       training['metrics'][metric]['outer_fold_mean'], atol=1e-14)
    result = {'status': 'SUCCESS', 'same_fold_and_row_pairing_verified': True,
              'replicates': 5000, 'sealed_test_accessed': False,
              'checks': ['hashes unchanged', 'fold-mean aggregation and quantiles',
                         'paired differences', 'explicit real-row resampling matches both methods',
                         'point estimates match original nested CV']}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
