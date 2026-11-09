"""Verify saved splits, row coverage, CV scaffold, and executed notebook."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pipeline import make_cv, make_pipeline
ROOT = Path(__file__).resolve().parents[1]
q = json.loads((ROOT / 'reports/data_quality.json').read_text())
assert q['matches_expected_284807_rows_492_frauds'] and q['shape'] == [284807, 31]
assert sum(q['missing_by_column'].values()) == sum(q['infinite_by_numeric_column'].values()) == 0
out = ROOT / 'data/processed'
dev = pd.read_csv(out / 'development.csv', index_col='raw_row_id')
test = pd.read_csv(out / 'sealed_test.csv', index_col='raw_row_id')
excluded = pd.read_csv(out / 'excluded_duplicate_indices.csv').raw_row_id
for name, frame in [('development', dev), ('sealed_test', test)]:
    assert frame.index.tolist() == pd.read_csv(out / f'{name}_indices.csv').raw_row_id.tolist()
    assert len(frame) == q['split'][name]['rows']
    assert {str(k): int(v) for k, v in frame.Class.value_counts().items()} == q['split'][name]['class_counts']
retained = pd.concat([dev, test]).sort_index()
assert retained.index.is_unique and len(retained) + len(excluded) == 284807
assert set(retained.index).isdisjoint(excluded)
assert set(retained.index) | set(excluded) == set(range(284807))
assert retained.shape[1] == 30 and 'Time' not in retained.columns
assert not retained.drop(columns='Class').duplicated().any()
a, b = train_test_split(retained.index.to_numpy(), test_size=.2, random_state=42, stratify=retained.Class)
assert set(a) == set(dev.index) and set(b) == set(test.index)
pipeline = make_pipeline()
assert not hasattr(pipeline['scaler'], 'mean_') and not hasattr(pipeline['logistic'], 'coef_')
folds = []
for train, valid in make_cv().split(dev.drop(columns='Class'), dev.Class):
    assert set(train).isdisjoint(valid)
    assert dev.iloc[train].Class.nunique() == dev.iloc[valid].Class.nunique() == 2
    folds.append({'train_rows': len(train), 'validation_rows': len(valid), 'validation_frauds': int(dev.iloc[valid].Class.sum())})
nb = json.loads((ROOT / 'notebooks/01_data_preparation.ipynb').read_text())
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert code and all(c['execution_count'] is not None for c in code)
assert not any(o['output_type'] == 'error' for c in code for o in c['outputs'])
result = {'status': 'SUCCESS', 'python_executable': sys.executable, 'checks': ['expected raw dimensions and labels', 'no missing or infinite values', 'saved row IDs and class counts', 'all raw IDs accounted for exactly once', '29 inputs plus Class; Time removed', 'no duplicate retained inputs', 'reproducible stratified 80/20 split', 'five stratified development folds', 'unfitted scaler and model', 'notebook executed without errors'], 'development_cv_folds': folds, 'preserved_files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['proposal.txt', 'Proposal Feedback.rtf']}}
(ROOT / 'reports/stage1_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
