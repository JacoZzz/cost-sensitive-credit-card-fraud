"""Independently audit conditional bootstrap summaries and selected real replicates."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/baseline/uncertainty'


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['status'] == 'SUCCESS'
    assert summary['provenance']['sealed_test_accessed'] is False
    for name, digest in summary['provenance']['input_files_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    replicates = pd.read_csv(OUT / 'bootstrap_replicates.csv.gz')
    folds = pd.read_csv(OUT / 'fold_bootstrap_replicates.csv.gz')
    assert len(replicates) == 5000 and len(folds) == 25000
    assert replicates.replicate.is_unique
    assert not folds.duplicated(['replicate', 'outer_fold']).any()
    for name, result in summary['metrics'].items():
        expected = folds.groupby('replicate')[name].mean().to_numpy()
        np.testing.assert_allclose(replicates[name].to_numpy(), expected, atol=1e-14)
        np.testing.assert_allclose(replicates[name].std(ddof=1), result['bootstrap_SE'], atol=1e-14)
        np.testing.assert_allclose(np.quantile(replicates[name], [.025, .975], method='linear'),
                                   result['conditional_95_percentile_interval'], atol=1e-14)
    np.testing.assert_allclose(replicates.tuned_mean_cost - replicates.default_mean_cost,
                               replicates.cost_difference_tuned_minus_default, atol=1e-14)
    data = pd.read_csv(ROOT / 'reports/baseline/development_outer_oof.csv.gz')
    first = data[data.outer_fold == 1].sort_values('probability', ascending=False, kind='stable')
    y = first.y_true.to_numpy()
    rng = np.random.default_rng(42)
    weights = np.zeros((25, len(first)), dtype=np.int64)
    for label in [0, 1]:
        idx = np.flatnonzero(y == label)
        weights[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=25)
    for i in range(3):
        # Explicitly duplicate sampled rows instead of using the optimized weighted routine.
        indices = np.repeat(np.arange(len(first)), weights[i])
        sampled = first.iloc[indices]
        saved = folds[(folds.outer_fold == 1) & (folds.replicate == i + 1)].iloc[0]
        np.testing.assert_allclose(average_precision_score(sampled.y_true, sampled.probability), saved.AP, atol=1e-14)
        assert sampled.y_true.sum() == 75 and len(sampled) == 44106
        for mode in ['default', 'tuned']:
            fp = int(((sampled[f'{mode}_prediction'] == 1) & (sampled.y_true == 0)).sum())
            fn = int(((sampled[f'{mode}_prediction'] == 0) & (sampled.y_true == 1)).sum())
            np.testing.assert_allclose((fp + 50 * fn) / len(sampled), saved[f'{mode}_mean_cost'], atol=1e-14)
    result = {'status': 'SUCCESS', 'replicates_verified': 5000,
              'fold_replicates_verified': 25000, 'sealed_test_accessed': False,
              'checks': ['input hashes unchanged', 'bootstrap means match fold-level replicates',
                         'SE and percentile endpoints recomputed', 'paired cost difference recomputed',
                         'first three real replicates match explicit row resampling and sklearn AP']}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
