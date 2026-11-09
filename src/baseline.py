"""Tune and evaluate an unweighted logistic baseline without reading test data."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/baseline'


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def choose_threshold(y, probabilities, fp_cost, fn_cost):
    """Minimize OOF cost over tied-score groups; prefer the highest tied threshold."""
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    order = np.argsort(-probabilities, kind='stable')
    scores = probabilities[order]
    positives = np.cumsum(y[order], dtype=np.int64)
    endpoints = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    tp = np.r_[0, positives[endpoints]]
    fp = np.r_[0, endpoints + 1 - positives[endpoints]]
    fn = int(y.sum()) - tp
    thresholds = np.r_[np.nextafter(scores[0], np.inf), scores[endpoints]]
    costs = fp_cost * fp + fn_cost * fn
    # Candidates are descending, so argmin implements the declared tie rule.
    best = int(np.argmin(costs))
    table = pd.DataFrame({'threshold': thresholds, 'FP': fp, 'FN': fn,
                          'total_cost': costs, 'mean_cost': costs / len(y)})
    return float(thresholds[best]), table


def cv_splits(X, y, folds, seed):
    return list(StratifiedKFold(n_splits=folds, shuffle=True,
                               random_state=seed).split(X, y))


def fit_model(X, y, C, cfg):
    estimator = make_pipeline()
    estimator.set_params(logistic__C=C, logistic__class_weight=None,
                         logistic__solver='lbfgs',
                         logistic__random_state=cfg['random_seed'])
    # Convergence failures must stop the run rather than become reported results.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(X, y)
    assert int(estimator['scaler'].n_samples_seen_) == len(y)
    assert estimator['logistic'].classes_.tolist() == [0, 1]
    return estimator


def tune(X, y, raw_ids, cfg, stage):
    splits = cv_splits(X, y, cfg['inner_folds'], cfg['random_seed'])
    best_C, best_score, best_oof = None, -np.inf, None
    tuning_rows = []
    membership = np.full(len(y), -1, dtype=int)
    for fold, (train, valid) in enumerate(splits, 1):
        assert not np.intersect1d(train, valid).size
        membership[valid] = fold
    for C in sorted(cfg['C_grid']):
        oof = np.full(len(y), np.nan)
        scores = []
        for fold, (train, valid) in enumerate(splits, 1):
            estimator = fit_model(X[train], y[train], C, cfg)
            probabilities = estimator.predict_proba(X[valid])[:, 1]
            oof[valid] = probabilities
            score = float(average_precision_score(y[valid], probabilities))
            scores.append(score)
            tuning_rows.append({'stage': stage, 'C': C, 'inner_fold': fold,
                                'train_rows': len(train), 'validation_rows': len(valid),
                                'validation_frauds': int(y[valid].sum()), 'AP': score,
                                'solver_iterations': int(estimator['logistic'].n_iter_.max())})
        assert np.isfinite(oof).all() and (membership >= 1).all()
        mean_ap = float(np.mean(scores))
        # Ascending C and strict improvement implement the declared C tie rule.
        if mean_ap > best_score:
            best_C, best_score, best_oof = C, mean_ap, oof.copy()
        print(f'{stage}: C={C:g}, inner mean AP={mean_ap:.6f}', flush=True)
    pd.DataFrame(tuning_rows).to_csv(OUT / f'{stage}_inner_scores.csv', index=False)
    threshold, candidates = choose_threshold(y, best_oof, cfg['false_positive_cost'],
                                             cfg['false_negative_cost'])
    candidates.to_csv(OUT / f'{stage}_threshold_candidates.csv.gz', index=False)
    pd.DataFrame({'raw_row_id': raw_ids, 'inner_fold': membership,
                  'y_true': y, 'probability': best_oof}).to_csv(
                      OUT / f'{stage}_selected_inner_oof.csv.gz', index=False)
    selected = {'C': best_C, 'inner_mean_AP': best_score, 'threshold': threshold,
                'inner_threshold_selection_cost': float(candidates.mean_cost.min())}
    write_json(OUT / f'{stage}_selection.json', selected)
    print(f'{stage}: selected C={best_C:g}, threshold={threshold:.8f}', flush=True)
    return selected


def metrics(y, probabilities, threshold, cfg):
    predicted = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel()]
    return {'threshold': float(threshold), 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn),
            'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'mean_cost': (cfg['false_positive_cost'] * fp + cfg['false_negative_cost'] * fn) / len(y)}


def run():
    started = time.perf_counter()
    cfg_path = ROOT / 'config/baseline_protocol.json'
    cfg = json.loads(cfg_path.read_text())
    assert cfg['class_weight'] is None and cfg['regularization'] == 'L2'
    assert cfg['random_seed'] == 42 and cfg['outer_folds'] == cfg['inner_folds'] == 5
    assert cfg['hyperparameter_selection_metric'] == 'average_precision'
    assert cfg['false_positive_cost'] == 1 and cfg['false_negative_cost'] == 50
    data_path = ROOT / 'data/processed/development.csv'
    # This is the only transaction data file opened by the baseline runner.
    frame = pd.read_csv(data_path)
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    assert frame.columns.tolist() == ['raw_row_id'] + features + ['Class']
    raw_ids = frame.raw_row_id.to_numpy(dtype=np.int64)
    X = frame[features].to_numpy(dtype=np.float64)
    y = frame.Class.to_numpy(dtype=np.int64)
    assert len(frame) == 220530 and int(y.sum()) == 378
    assert set(np.unique(y)) == {0, 1} and np.isfinite(X).all()
    assert len(np.unique(raw_ids)) == len(raw_ids)
    provenance = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'python': platform.python_version(), 'python_executable': sys.executable,
                  'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
                  'input_files_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                      [data_path, cfg_path, Path(__file__), ROOT / 'src/pipeline.py']},
                  'transaction_files_read': ['data/processed/development.csv'],
                  'sealed_test_accessed': False, 'protocol': cfg,
                  'features': features, 'development_rows': len(y),
                  'development_frauds': int(y.sum()), 'thread_limit': 1}
    write_json(OUT / 'provenance.json', provenance)
    outer_predictions = []
    fold_results = []
    models_dir = ROOT / 'models/baseline'
    models_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train, valid) in enumerate(cv_splits(X, y, cfg['outer_folds'], cfg['random_seed']), 1):
        stage = f'outer_{fold}'
        assert not np.intersect1d(raw_ids[train], raw_ids[valid]).size
        selection = tune(X[train], y[train], raw_ids[train], cfg, stage)
        estimator = fit_model(X[train], y[train], selection['C'], cfg)
        probabilities = estimator.predict_proba(X[valid])[:, 1]
        assert np.isfinite(probabilities).all()
        default = metrics(y[valid], probabilities, .5, cfg)
        cost_tuned = metrics(y[valid], probabilities, selection['threshold'], cfg)
        row = {'outer_fold': fold, 'train_rows': len(train), 'validation_rows': len(valid),
               'validation_frauds': int(y[valid].sum()), 'selected_C': selection['C'],
               'inner_mean_AP': selection['inner_mean_AP'],
               'AP': float(average_precision_score(y[valid], probabilities)),
               'solver_iterations': int(estimator['logistic'].n_iter_.max()),
               **{f'default_{k}': v for k, v in default.items()},
               **{f'tuned_{k}': v for k, v in cost_tuned.items()}}
        fold_results.append(row)
        predictions = pd.DataFrame({'raw_row_id': raw_ids[valid], 'outer_fold': fold,
                                    'y_true': y[valid], 'probability': probabilities,
                                    'tuned_threshold': selection['threshold'],
                                    'default_prediction': (probabilities >= .5).astype(int),
                                    'tuned_prediction': (probabilities >= selection['threshold']).astype(int)})
        predictions.to_csv(OUT / f'{stage}_predictions.csv.gz', index=False)
        outer_predictions.append(predictions)
        joblib.dump({'pipeline': estimator, 'feature_names': features, **selection},
                    models_dir / f'{stage}.joblib')
        pd.DataFrame(fold_results).to_csv(OUT / 'outer_fold_results.csv', index=False)
        write_json(OUT / 'run_status.json', {'status': 'RUNNING', 'completed_outer_folds': fold})
        print(f'{stage}: HELD-OUT AP={row["AP"]:.6f}, default cost={default["mean_cost"]:.6f}, '
              f'tuned cost={cost_tuned["mean_cost"]:.6f}', flush=True)
    predictions = pd.concat(outer_predictions).sort_values('raw_row_id')
    assert len(predictions) == len(y) and predictions.raw_row_id.is_unique
    assert np.array_equal(predictions.raw_row_id.to_numpy(), np.sort(raw_ids))
    predictions.to_csv(OUT / 'development_outer_oof.csv.gz', index=False)
    results = pd.DataFrame(fold_results)
    # Full-development tuning selects the deployable artifact, not an evaluation score.
    final_selection = tune(X, y, raw_ids, cfg, 'full_development')
    final_model = fit_model(X, y, final_selection['C'], cfg)
    joblib.dump({'pipeline': final_model, 'feature_names': features,
                 'protocol': cfg, **final_selection}, models_dir / 'development_refit.joblib')
    summary = {'status': 'SUCCESS', 'evaluation': 'nested five-fold CV on development data only',
               'sealed_test_accessed': False, 'outer_folds': fold_results,
               'metrics': {name: {'outer_fold_mean': float(results[name].mean()),
                                  'outer_fold_sample_SD': float(results[name].std(ddof=1))}
                           for name in ['AP', 'default_mean_cost', 'tuned_mean_cost',
                                        'default_precision', 'default_recall',
                                        'tuned_precision', 'tuned_recall']},
               'pooled_outer_oof_AP': float(average_precision_score(predictions.y_true, predictions.probability)),
               'pooled_confusion': {mode: {key: int(results[f'{mode}_{key}'].sum())
                                          for key in ['TN', 'FP', 'FN', 'TP']}
                                    for mode in ['default', 'tuned']},
               'always_negative_cost_reference': float(cfg['false_negative_cost'] * y.mean()),
               'full_development_selection': final_selection,
               'total_pipeline_fits': cfg['outer_folds'] * (len(cfg['C_grid']) * cfg['inner_folds'] + 1)
                                      + len(cfg['C_grid']) * cfg['inner_folds'] + 1,
               'convergence_failures': 0, 'runtime_seconds': time.perf_counter() - started,
               'error_bar_status': 'not_yet_computed; fold SD is descriptive spread, not a confidence interval'}
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'run_status.json', {'status': 'SUCCESS', 'completed_outer_folds': 5,
                                       'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'run_status.json', {'status': 'RUNNING', 'completed_outer_folds': 0})
    try:
        with threadpool_limits(limits=1):
            run()
    except Exception:
        write_json(OUT / 'run_status.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
