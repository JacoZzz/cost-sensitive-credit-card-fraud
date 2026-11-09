"""Audit saved nested-CV artifacts without opening the sealed test set."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from baseline import choose_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/baseline'


def check_threshold_optimizer():
    """Compare grouped optimization with exhaustive candidates, including ties."""
    cases = [([0, 1], [.8, .8]), ([0, 1], [.9, .1]),
             ([0, 0, 1], [.4, .2, .1]), ([1, 1, 0], [.7, .3, .2])]
    for labels, scores in cases:
        y, p = np.array(labels), np.array(scores)
        for fp_cost, fn_cost in [(1, 50), (1, 1), (50, 1)]:
            candidates = sorted(set(p.tolist() + [float(np.nextafter(p.max(), np.inf))]), reverse=True)
            totals = [fp_cost * int(((p >= t) & (y == 0)).sum())
                      + fn_cost * int(((p < t) & (y == 1)).sum()) for t in candidates]
            expected = candidates[int(np.argmin(totals))]
            actual, table = choose_threshold(y, p, fp_cost, fn_cost)
            assert actual == expected
            assert table.total_cost.min() == min(totals)


def main():
    check_threshold_optimizer()
    summary = json.loads((OUT / 'summary.json').read_text())
    assert json.loads((OUT / 'run_status.json').read_text())['status'] == 'SUCCESS'
    cfg = json.loads((ROOT / 'config/baseline_protocol.json').read_text())
    provenance = json.loads((OUT / 'provenance.json').read_text())
    for name, digest in provenance['input_files_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    data = pd.read_csv(ROOT / 'data/processed/development.csv').set_index('raw_row_id')
    features = provenance['features']
    folds = pd.read_csv(OUT / 'outer_fold_results.csv')
    oof = pd.read_csv(OUT / 'development_outer_oof.csv.gz').set_index('raw_row_id')
    assert oof.index.is_unique and set(oof.index) == set(data.index)
    assert (oof.y_true == data.loc[oof.index, 'Class']).all()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train, valid) in enumerate(splitter.split(data[features], data.Class), 1):
        actual = oof[oof.outer_fold == fold]
        assert set(actual.index) == set(data.index[valid])
        assert set(actual.index).isdisjoint(data.index[train])
        stage = f'outer_{fold}'
        model = joblib.load(ROOT / f'models/baseline/{stage}.joblib')
        estimator = model['pipeline']
        train_X = data.iloc[train][features].to_numpy()
        assert estimator['scaler'].n_samples_seen_ == len(train)
        np.testing.assert_allclose(estimator['scaler'].mean_, train_X.mean(axis=0), atol=1e-12, rtol=1e-10)
        assert estimator['logistic'].class_weight is None
        assert estimator['logistic'].l1_ratio == 0
        predicted_p = estimator.predict_proba(data.loc[actual.index, features].to_numpy())[:, 1]
        np.testing.assert_allclose(predicted_p, actual.probability, atol=1e-14, rtol=1e-12)
        row = folds[folds.outer_fold == fold].iloc[0]
        np.testing.assert_allclose(average_precision_score(actual.y_true, actual.probability), row.AP)
        for mode, threshold in [('default', .5), ('tuned', model['threshold'])]:
            labels = (actual.probability.to_numpy() >= threshold).astype(int)
            assert np.array_equal(labels, actual[f'{mode}_prediction'].to_numpy())
            tn, fp, fn, tp = confusion_matrix(actual.y_true, labels, labels=[0, 1]).ravel()
            assert [tn, fp, fn, tp] == [int(row[f'{mode}_{k}']) for k in ['TN', 'FP', 'FN', 'TP']]
            np.testing.assert_allclose((fp + 50 * fn) / len(actual), row[f'{mode}_mean_cost'])
        inner = pd.read_csv(OUT / f'{stage}_selected_inner_oof.csv.gz').set_index('raw_row_id')
        assert set(inner.index) == set(data.index[train])
        assert set(inner.index).isdisjoint(actual.index)
        threshold, _ = choose_threshold(inner.y_true, inner.probability, 1, 50)
        np.testing.assert_allclose(threshold, model['threshold'], atol=1e-14, rtol=1e-12)
        tuning = pd.read_csv(OUT / f'{stage}_inner_scores.csv')
        means = tuning.groupby('C', sort=True).AP.mean()
        assert float(means.idxmax()) == model['C'] == float(row.selected_C)
    full = joblib.load(ROOT / 'models/baseline/development_refit.joblib')
    full_inner = pd.read_csv(OUT / 'full_development_selected_inner_oof.csv.gz')
    assert set(full_inner.raw_row_id) == set(data.index)
    full_scores = pd.read_csv(OUT / 'full_development_inner_scores.csv')
    assert full_scores.groupby('C').AP.mean().idxmax() == full['C']
    final_threshold, _ = choose_threshold(full_inner.y_true, full_inner.probability, 1, 50)
    np.testing.assert_allclose(full['threshold'], final_threshold, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(full['pipeline']['scaler'].mean_, data[features].to_numpy().mean(axis=0), atol=1e-12, rtol=1e-10)
    for name in ['AP', 'default_mean_cost', 'tuned_mean_cost']:
        np.testing.assert_allclose(folds[name].mean(), summary['metrics'][name]['outer_fold_mean'])
        np.testing.assert_allclose(folds[name].std(ddof=1), summary['metrics'][name]['outer_fold_sample_SD'])
    assert provenance['transaction_files_read'] == ['data/processed/development.csv']
    assert summary['sealed_test_accessed'] is False
    result = {'status': 'SUCCESS', 'verified_outer_folds': 5,
              'verified_oof_rows': len(oof), 'sealed_test_accessed': False,
              'checks': ['threshold optimizer versus exhaustive candidates',
                         'source, protocol and development hashes',
                         'deterministic outer folds and complete OOF coverage',
                         'scaler means equal outer training data means',
                         'saved models reproduce outer predictions',
                         'AP, confusion matrices and costs recomputed',
                         'inner threshold tuning excludes outer validation IDs',
                         'selected C maximizes inner mean AP',
                         'full-development refit selection and scaler verified',
                         'summary means and descriptive SD recomputed']}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
