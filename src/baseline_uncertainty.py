"""Compute conditional paired bootstrap error bars from saved outer OOF scores."""
from pathlib import Path
import hashlib
import json
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/baseline/uncertainty'
METRICS = ['AP', 'default_mean_cost', 'tuned_mean_cost',
           'default_precision', 'default_recall', 'default_F1',
           'tuned_precision', 'tuned_recall', 'tuned_F1', 'cost_difference_tuned_minus_default']


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def prepare_fold(frame):
    frame = frame.sort_values('probability', ascending=False, kind='stable')
    y = frame.y_true.to_numpy(dtype=np.int64)
    probabilities = frame.probability.to_numpy()
    return {'y': y, 'probabilities': probabilities,
            'ends': np.flatnonzero(np.r_[probabilities[:-1] != probabilities[1:], True]),
            'default': frame.default_prediction.to_numpy(dtype=bool),
            'tuned': frame.tuned_prediction.to_numpy(dtype=bool)}


def weighted_metrics(prepared, weights):
    """Evaluate exact duplicate-row bootstrap statistics without sorting again."""
    y = prepared['y']
    cumulative_tp = np.cumsum(weights * y[None, :], axis=1)[:, prepared['ends']]
    cumulative_size = np.cumsum(weights, axis=1)[:, prepared['ends']]
    precision = np.divide(cumulative_tp, cumulative_size,
                          out=np.zeros_like(cumulative_tp, dtype=float), where=cumulative_size > 0)
    positive_increments = np.diff(cumulative_tp, axis=1, prepend=0)
    n_positive = (weights * y[None, :]).sum(axis=1)
    n_total = weights.sum(axis=1)
    result = {'AP': (positive_increments * precision).sum(axis=1) / n_positive}
    for mode in ['default', 'tuned']:
        predicted = prepared[mode]
        tp = weights[:, (y == 1) & predicted].sum(axis=1)
        fp = weights[:, (y == 0) & predicted].sum(axis=1)
        fn = n_positive - tp
        result[f'{mode}_mean_cost'] = (fp + 50 * fn) / n_total
        result[f'{mode}_precision'] = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=tp + fp > 0)
        result[f'{mode}_recall'] = tp / n_positive
        result[f'{mode}_F1'] = np.divide(2 * tp, 2 * tp + fp + fn,
                                        out=np.zeros_like(tp, dtype=float), where=2 * tp + fp + fn > 0)
    result['cost_difference_tuned_minus_default'] = result['tuned_mean_cost'] - result['default_mean_cost']
    return result


def verify_weighted_ap():
    """Compare optimized AP against sklearn, including ties and zero weights."""
    frame = pd.DataFrame({'y_true': [0, 1, 1, 0, 1, 0],
                          'probability': [.9, .8, .8, .4, .2, .2],
                          'default_prediction': [1, 1, 1, 0, 0, 0],
                          'tuned_prediction': [1, 1, 1, 1, 1, 1]})
    prepared = prepare_fold(frame)
    cases = np.array([[1, 1, 1, 1, 1, 1], [0, 2, 1, 0, 3, 0], [2, 0, 1, 4, 0, 1]])
    for weights in cases:
        actual = weighted_metrics(prepared, weights[None, :])['AP'][0]
        expected = average_precision_score(prepared['y'], prepared['probabilities'], sample_weight=weights)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        repeated_y = np.repeat(prepared['y'], weights)
        repeated_p = np.repeat(prepared['probabilities'], weights)
        np.testing.assert_allclose(actual, average_precision_score(repeated_y, repeated_p), atol=1e-14)


def run():
    started = time.perf_counter()
    verify_weighted_ap()
    protocol_path = ROOT / 'config/uncertainty_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['n_resamples'] == 5000 and protocol['random_seed'] == 42
    baseline_out = ROOT / 'reports/baseline'
    assert json.loads((baseline_out / 'verification.json').read_text())['status'] == 'SUCCESS'
    original_summary = json.loads((baseline_out / 'summary.json').read_text())
    frame_path = baseline_out / 'development_outer_oof.csv.gz'
    frame = pd.read_csv(frame_path)
    assert frame.raw_row_id.is_unique and len(frame) == 220530 and frame.y_true.sum() == 378
    assert set(frame.outer_fold) == {1, 2, 3, 4, 5}
    fold_results = pd.read_csv(baseline_out / 'outer_fold_results.csv')
    rng = np.random.default_rng(protocol['random_seed'])
    B = protocol['n_resamples']
    replicates = {name: np.zeros(B, dtype=float) for name in METRICS}
    point_values = {name: [] for name in METRICS}
    fold_bootstrap = []
    for fold in range(1, 6):
        prepared = prepare_fold(frame[frame.outer_fold == fold])
        y = prepared['y']
        n = len(y)
        point = weighted_metrics(prepared, np.ones((1, n), dtype=np.int64))
        row = fold_results[fold_results.outer_fold == fold].iloc[0]
        for name in METRICS:
            point_values[name].append(float(point[name][0]))
            if name in fold_results:
                np.testing.assert_allclose(point[name][0], row[name], atol=1e-14)
        indices = {label: np.flatnonzero(y == label) for label in [0, 1]}
        fold_values = {name: np.empty(B, dtype=float) for name in METRICS}
        for start in range(0, B, 25):
            stop = min(B, start + 25)
            weights = np.zeros((stop - start, n), dtype=np.int64)
            for label in [0, 1]:
                idx = indices[label]
                # Multinomial counts are exactly equivalent to sampling rows with replacement.
                weights[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=stop - start)
                assert np.all(weights[:, idx].sum(axis=1) == len(idx))
            result = weighted_metrics(prepared, weights)
            for name in METRICS:
                fold_values[name][start:stop] = result[name]
                replicates[name][start:stop] += result[name] / 5
        fold_bootstrap.append(pd.DataFrame({'replicate': np.arange(1, B + 1),
                                            'outer_fold': fold, **fold_values}))
        print(f'Completed fold {fold}: {B} paired, class-stratified bootstrap replicates', flush=True)
        save_json(OUT / 'run_status.json', {'status': 'RUNNING', 'completed_folds': fold})
    points = {name: float(np.mean(values)) for name, values in point_values.items()}
    for name in ['AP', 'default_mean_cost', 'tuned_mean_cost']:
        np.testing.assert_allclose(points[name], original_summary['metrics'][name]['outer_fold_mean'], atol=1e-14)
    pd.DataFrame({'replicate': np.arange(1, B + 1), **replicates}).to_csv(OUT / 'bootstrap_replicates.csv.gz', index=False)
    pd.concat(fold_bootstrap).to_csv(OUT / 'fold_bootstrap_replicates.csv.gz', index=False)
    alpha = (1 - protocol['confidence_level']) / 2
    estimates = {}
    for name in METRICS:
        low, high = np.quantile(replicates[name], [alpha, 1 - alpha], method='linear')
        estimates[name] = {'point_estimate': points[name],
                           'bootstrap_SE': float(np.std(replicates[name], ddof=1)),
                           'conditional_95_percentile_interval': [float(low), float(high)],
                           'bootstrap_mean': float(np.mean(replicates[name]))}
    provenance = {'input_files_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in
                   [frame_path, baseline_out / 'outer_fold_results.csv', baseline_out / 'summary.json',
                    baseline_out / 'verification.json', protocol_path, Path(__file__)]},
                  'transaction_files_read': ['reports/baseline/development_outer_oof.csv.gz'],
                  'sealed_test_accessed': False, 'models_refitted': 0}
    result = {'status': 'SUCCESS', 'protocol': protocol, 'metrics': estimates,
              'provenance': provenance, 'runtime_seconds': time.perf_counter() - started,
              'validation': {'weighted_AP_matches_sklearn_and_explicit_repetition': True,
                             'point_estimates_match_outer_cv': True,
                             'class_counts_preserved_within_each_fold': True}}
    save_json(OUT / 'summary.json', result)
    save_json(OUT / 'run_status.json', {'status': 'SUCCESS', 'completed_folds': 5})
    print(json.dumps(result, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'run_status.json', {'status': 'RUNNING', 'completed_folds': 0})
    try:
        with threadpool_limits(limits=1):
            run()
    except Exception:
        save_json(OUT / 'run_status.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
