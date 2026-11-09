"""Create and execute a reader for the saved baseline experiment artifacts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH'] = str(ROOT / '.venv/share/jupyter')
os.environ['IPYTHONDIR'] = str(ROOT / 'work/ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / 'work/jupyter_runtime')

import nbformat
from nbclient import NotebookClient

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Fraud Project (.venv)',
                          'language': 'python', 'name': 'fraud-project'}
nb.cells = [
    nbformat.v4.new_markdown_cell(
        '# Part 2: Tuned Logistic Regression Baseline\n'
        'Training was executed by `src/baseline.py`. This notebook displays its '
        'saved results and provenance; it does not retrain models. All evaluation '
        'uses development data only. The sealed test set is not opened.\n\n'
        'Reproduce training from the project root:\n'
        '```bash\n.venv/bin/python src/baseline.py\n'
        '.venv/bin/python src/verify_baseline.py\n```'),
    nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport hashlib\nimport json\nimport pandas as pd\n"
        "from IPython.display import display\n"
        "root = Path.cwd()\nif root.name == 'notebooks':\n    root = root.parent\n"
        "out = root / 'reports/baseline'\n"
        "summary = json.loads((out / 'summary.json').read_text())\n"
        "provenance = json.loads((out / 'provenance.json').read_text())\n"
        "verification = json.loads((out / 'verification.json').read_text())\n"
        "assert summary['status'] == verification['status'] == 'SUCCESS'\n"
        "assert summary['sealed_test_accessed'] is False\n"
        "for name in ['src/baseline.py', 'src/pipeline.py', 'config/baseline_protocol.json']:\n"
        "    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == provenance['input_files_sha256'][name]\n"
        "print('Training runtime (seconds):', summary['runtime_seconds'])\n"
        "print('Total pipeline fits:', summary['total_pipeline_fits'])\n"
        "print('Convergence failures:', summary['convergence_failures'])\n"
        "print('Transaction files read:', provenance['transaction_files_read'])\n"
        "print('Audit:', verification['status'])"),
    nbformat.v4.new_markdown_cell(
        '## Outer validation results\n'
        'Each outer validation fold was excluded from scaling, model fitting, '
        'C selection and threshold selection for that fold. Inner five-fold CV '
        'chooses C by mean AP. For the selected C, inner OOF predictions select '
        'the threshold minimizing `(FP + 50*FN) / N`.\n\n'
        'The default 0.5 threshold is reported separately from the cost-tuned '
        'threshold. AP uses probabilities and is identical for both decisions.'),
    nbformat.v4.new_code_cell(
        "folds = pd.read_csv(out / 'outer_fold_results.csv')\n"
        "columns = ['outer_fold', 'validation_rows', 'validation_frauds', 'selected_C',\n"
        "           'AP', 'default_FP', 'default_FN', 'default_mean_cost',\n"
        "           'tuned_threshold', 'tuned_FP', 'tuned_FN', 'tuned_mean_cost']\n"
        "display(folds[columns])\n"
        "display(pd.DataFrame(summary['metrics']).T)\n"
        "print('Pooled confusion counts:', summary['pooled_confusion'])"),
    nbformat.v4.new_markdown_cell(
        '## Full-development selection and limitations\n'
        'Full-development tuning selects the artifact for later evaluation. '
        'Its inner score is a tuning score, not independent held-out performance. '
        'The primary performance estimate is mean outer-fold AP. Pooled OOF AP '
        'is an additional summary and can differ because AP is nonlinear and '
        'the folds use different fitted models.\n\n'
        'Fold SD describes variability; it is not a confidence interval. Formal '
        'error bars, class-weight comparisons and the final submission are '
        'not completed by this part.'),
    nbformat.v4.new_code_cell(
        "grid = pd.read_csv(out / 'full_development_inner_scores.csv')\n"
        "display(grid.groupby('C')['AP'].agg(['mean', 'std']))\n"
        "print('Full-development selection:', summary['full_development_selection'])\n"
        "print('Primary mean outer-fold AP:', summary['metrics']['AP']['outer_fold_mean'])\n"
        "print('Additional pooled OOF AP:', summary['pooled_outer_oof_AP'])\n"
        "print('Error bar status:', summary['error_bar_status'])"),
]
p = ROOT / 'notebooks/02_baseline.ipynb'
try:
    NotebookClient(nb, timeout=120, kernel_name='fraud-project',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
finally:
    nbformat.write(nb, p)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
print('Baseline result notebook executed successfully.')
