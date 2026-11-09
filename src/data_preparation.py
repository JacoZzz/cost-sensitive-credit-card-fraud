"""Reproducible preparation; never fits a model or accesses test performance."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import platform
import ssl
import sys
import traceback
import urllib.request
import certifi

ROOT = Path(__file__).resolve().parents[1]

def download(url, path):
    temp = path.with_suffix(path.suffix + '.part')
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=120, context=context) as response, temp.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

def run():
    for directory in ['data/raw', 'data/processed', 'reports', 'notebooks']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    raw = ROOT / 'data/raw'
    meta_path = raw / 'openml_1597_metadata.json'
    if not meta_path.exists():
        download('https://www.openml.org/api/v1/json/data/1597', meta_path)
    meta = json.loads(meta_path.read_text())['data_set_description']
    data_path = raw / 'openml_1597.arff'
    if not data_path.exists():
        download(meta['url'], data_path)
    manifest = {
        'dataset_id': 1597, 'name': meta['name'], 'version': meta['version'],
        'source': 'https://www.openml.org/search?type=data&id=1597',
        'download_url': meta['url'], 'method': 'urllib.request; original OpenML ARFF bytes',
        'retrieved_or_verified_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'openml_md5': meta.get('md5_checksum'),
    }
    if meta.get('md5_checksum'):
        if hashlib.md5(data_path.read_bytes()).hexdigest() != meta['md5_checksum']:
            raise ValueError('OpenML MD5 checksum mismatch')
    (raw / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    import numpy as np
    import pandas as pd
    from scipy.io import arff
    from sklearn.model_selection import train_test_split
    import sklearn
    df = pd.DataFrame(arff.loadarff(data_path)[0])
    if 'Class' not in df or 'Time' not in df:
        raise ValueError(f'Unexpected schema: {df.columns.tolist()}')
    df['Class'] = df['Class'].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    df['Class'] = pd.to_numeric(df['Class'], errors='raise').astype('int64')
    if set(df.Class.unique()) != {0, 1}:
        raise ValueError('Class must contain only 0 and 1')
    counts = df.Class.value_counts().sort_index()
    numeric = df.select_dtypes(include='number')
    X = df.drop(columns=['Time', 'Class'])
    group = df.groupby(X.columns.tolist(), dropna=False, sort=False)['Class'].agg(['size', 'nunique'])
    quality = {
        'shape': list(df.shape), 'columns_and_types': df.dtypes.astype(str).to_dict(),
        'class_counts': {str(k): int(v) for k, v in counts.items()},
        'class_proportions': {str(k): float(v) for k, v in df.Class.value_counts(normalize=True).items()},
        'missing_by_column': df.isna().sum().astype(int).to_dict(),
        'infinite_by_numeric_column': np.isinf(numeric).sum().astype(int).to_dict(),
        'constant_columns': df.columns[df.nunique(dropna=False) <= 1].tolist(),
        'full_duplicate_excess_rows': int(df.duplicated().sum()),
        'input_duplicate_excess_rows_after_dropping_Time': int(X.duplicated().sum()),
        'conflicting_input_groups_after_dropping_Time': int((group['nunique'] > 1).sum()),
        'conflicting_input_rows_after_dropping_Time': int(group.loc[group['nunique'] > 1, 'size'].sum()),
        'matches_expected_284807_rows_492_frauds': bool(len(df) == 284807 and counts.get(1, 0) == 492),
    }
    numeric.describe().T.to_csv(ROOT / 'reports/numeric_summary.csv')
    # Persist diagnostics even when modeling inputs are invalid.
    (ROOT / 'reports/data_quality.json').write_text(json.dumps(quality, indent=2))
    if X.select_dtypes(include='number').shape[1] != X.shape[1] or X.isna().any().any() or np.isinf(X.to_numpy()).any():
        raise ValueError('Inputs require explicit missing/nonfinite/nonnumeric handling; diagnostics saved, split not created')
    # Exact equality across all retained inputs; no hash-only deduplication.
    # Keep first raw row by source order regardless of label; no label-based selection.
    keep = ~X.duplicated(keep='first')
    ids = df.index[keep].to_numpy()
    dev, test = train_test_split(ids, test_size=.2, random_state=42, stratify=df.loc[ids, 'Class'])
    dev, test = np.sort(dev), np.sort(test)
    overlap = pd.MultiIndex.from_frame(X.loc[dev]).intersection(pd.MultiIndex.from_frame(X.loc[test]))
    assert len(overlap) == 0
    assert set(dev).isdisjoint(test)
    assert len(dev) + len(test) == len(ids)
    out = ROOT / 'data/processed'
    pd.DataFrame({'raw_row_id': df.index[~keep]}).to_csv(out / 'excluded_duplicate_indices.csv', index=False)
    for name, indices in [('development', dev), ('sealed_test', test)]:
        pd.DataFrame({'raw_row_id': indices}).to_csv(out / f'{name}_indices.csv', index=False)
        df.loc[indices].drop(columns='Time').rename_axis('raw_row_id').to_csv(out / f'{name}.csv')
    quality['split'] = {name: {'rows': len(idx), 'class_counts': {str(k): int(v) for k, v in df.loc[idx, 'Class'].value_counts().items()}} for name, idx in [('development', dev), ('sealed_test', test)]}
    quality['cross_split_duplicate_inputs'] = len(overlap)
    quality['excluded_rows'] = int((~keep).sum())
    (ROOT / 'reports/data_quality.json').write_text(json.dumps(quality, indent=2))
    versions = {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__}
    import scipy
    versions['scipy'] = scipy.__version__
    (ROOT / 'requirements-verified.txt').write_text(''.join(f'{k}=={v}\n' for k, v in versions.items()))
    report = '''# Data quality report

Status: Real-data preparation completed. See data_quality.json, numeric_summary.csv, and data/raw/source_manifest.json.

## Cleaning and split rules
The raw ARFF remains unchanged. Class is the target; Time is excluded from the inputs.
No rows are removed solely because of extreme values. Full duplicates and conflicting labels are counted first.
For identical retained inputs, keep the first source row and record excluded row IDs in excluded_duplicate_indices.csv; never select rows based on their labels.
Deduplication precedes the fixed stratified 80/20 split to prevent identical inputs crossing the split.
This changes transaction frequencies and class proportions. Keeping the first conflicting label would be arbitrary; excluded rows are not necessarily erroneous. The study population is the deduplicated sample.
Use random_state=42. All indices are zero-based source ARFF row IDs.
The final test set is sealed. Only structure, label counts, and cross-split duplicates are checked; no model is trained or tuned and no test performance is computed.

## Standardization
src/pipeline.py constructs an unfitted StandardScaler-to-LogisticRegression pipeline.
Future cross-validation must receive unscaled development inputs and fit the scaler within each training fold.
No fit is called in this stage, no globally standardized data is produced, and labels are not standardized.
Missing, infinite, or nonnumeric inputs require explicit handling; this script does not silently impute them.

## Limitations
A random split does not simulate future-time generalization; dropping Time does not remove all temporal dependence.
Deduplication can change sample weights and must be disclosed in the paper.
The baseline checkpoint, CV AP, cost comparisons, and error bars have not been completed.
The pre-specified cost ratio and evaluation rules are in config/baseline_protocol.json and README.md.

## Observed checks
```json
''' + json.dumps(quality, ensure_ascii=False, indent=2) + '\n```\n'
    (ROOT / 'reports/data_quality_report.md').write_text(report)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return quality

def main():
    try:
        run()
    except Exception:
        failure = {'status': 'FAILED', 'python': platform.python_version(), 'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'traceback': traceback.format_exc()}
        (ROOT / 'reports/run_status.json').write_text(json.dumps(failure, indent=2))
        print(failure['traceback'], file=sys.stderr)
        return 1
    (ROOT / 'reports/run_status.json').write_text(json.dumps({'status': 'SUCCESS'}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
