"""Compare balanced and unweighted models with identical paired bootstrap rows."""
from pathlib import Path
import json
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from baseline_uncertainty import METRICS, prepare_fold, weighted_metrics, file_hash, save_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/class_weight/comparison'


def run():
    started = time.perf_counter()
    base_out = ROOT / 'reports/baseline'
    control_out = ROOT / 'reports/class_weight'
    for directory in [base_out, control_out]:
        assert json.loads((directory / 'verification.json').read_text())['status'] == 'SUCCESS'
    protocol_path = ROOT / 'config/uncertainty_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol.update({'comparison': 'balanced_minus_unweighted',
                     'pairing': 'identical_raw_row_ids_within_fold_and_class_across_methods',
                     'bootstrap_row_order': 'unweighted_probability_descending_stable'})
    save_json(OUT / 'protocol.json', protocol)
    base = pd.read_csv(base_out / 'development_outer_oof.csv.gz')
    control = pd.read_csv(control_out / 'development_outer_oof.csv.gz').set_index('raw_row_id')
    assert base.raw_row_id.is_unique and control.index.is_unique
    assert set(base.raw_row_id) == set(control.index)
    aligned = control.loc[base.raw_row_id]
    assert np.array_equal(base.y_true, aligned.y_true)
    assert np.array_equal(base.outer_fold, aligned.outer_fold)
    rng = np.random.default_rng(protocol['random_seed'])
    B = protocol['n_resamples']
    columns = [f'{method}_{metric}' for method in ['unweighted', 'balanced'] for metric in METRICS]
    delta_metrics = ['AP', 'default_mean_cost', 'tuned_mean_cost']
    columns += [f'delta_{metric}' for metric in delta_metrics]
    draws = {name: np.zeros(B) for name in columns}
    points = {name: [] for name in columns}
    fold_tables = []
    for fold in range(1, 6):
        f0 = base[base.outer_fold == fold].sort_values('probability', ascending=False, kind='stable')
        f1 = control.loc[f0.raw_row_id].copy()
        order1 = np.argsort(-f1.probability.to_numpy(), kind='stable')
        prepared0, prepared1 = prepare_fold(f0), prepare_fold(f1)
        y = f0.y_true.to_numpy(dtype=int)
        n = len(y)
        point0 = weighted_metrics(prepared0, np.ones((1, n), dtype=int))
        point1 = weighted_metrics(prepared1, np.ones((1, n), dtype=int))
        for method, result in [('unweighted', point0), ('balanced', point1)]:
            for metric in METRICS:
                points[f'{method}_{metric}'].append(float(result[metric][0]))
        for metric in delta_metrics:
            points[f'delta_{metric}'].append(float(point1[metric][0] - point0[metric][0]))
        fold_draws = {name: np.empty(B) for name in columns}
        indices = {label: np.flatnonzero(y == label) for label in [0, 1]}
        for start in range(0, B, 25):
            stop = min(B, start + 25)
            weights = np.zeros((stop - start, n), dtype=np.int64)
            for label in [0, 1]:
                idx = indices[label]
                weights[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=stop - start)
            result0 = weighted_metrics(prepared0, weights)
            result1 = weighted_metrics(prepared1, weights[:, order1])
            result = {f'{method}_{metric}': values[metric]
                      for method, values in [('unweighted', result0), ('balanced', result1)]
                      for metric in METRICS}
            result.update({f'delta_{metric}': result1[metric] - result0[metric] for metric in delta_metrics})
            for name in columns:
                fold_draws[name][start:stop] = result[name]
                draws[name][start:stop] += result[name] / 5
        fold_tables.append(pd.DataFrame({'replicate': np.arange(1, B + 1), 'outer_fold': fold, **fold_draws}))
        print(f'Paired class-weight comparison: fold {fold}/{5}, {B} resamples', flush=True)
        save_json(OUT / 'run_status.json', {'status': 'RUNNING', 'completed_folds': fold})
    # Reusing the original baseline ordering and RNG must reproduce its bootstrap.
    previous = pd.read_csv(base_out / 'uncertainty/bootstrap_replicates.csv.gz')
    for metric in METRICS:
        np.testing.assert_allclose(draws[f'unweighted_{metric}'], previous[metric], atol=1e-14)
    estimates = {}
    for name in columns:
        low, high = np.quantile(draws[name], [.025, .975], method='linear')
        estimates[name] = {'point_estimate': float(np.mean(points[name])),
                           'bootstrap_SE': float(np.std(draws[name], ddof=1)),
                           'conditional_95_percentile_interval': [float(low), float(high)]}
    pd.DataFrame({'replicate': np.arange(1, B + 1), **draws}).to_csv(OUT / 'bootstrap_replicates.csv.gz', index=False)
    pd.concat(fold_tables).to_csv(OUT / 'fold_bootstrap_replicates.csv.gz', index=False)
    inputs = [base_out / 'development_outer_oof.csv.gz', control_out / 'development_outer_oof.csv.gz',
              base_out / 'verification.json', control_out / 'verification.json', protocol_path,
              Path(__file__), ROOT / 'src/baseline_uncertainty.py']
    summary = {'status': 'SUCCESS', 'protocol': protocol, 'metrics': estimates,
               'same_row_ids_and_folds': True, 'baseline_bootstrap_reproduced': True,
               'sealed_test_accessed': False, 'models_refitted': 0,
               'input_files_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in inputs},
               'runtime_seconds': time.perf_counter() - started}
    save_json(OUT / 'summary.json', summary)
    save_json(OUT / 'run_status.json', {'status': 'SUCCESS', 'completed_folds': 5})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
