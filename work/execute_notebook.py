import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH'] = str(root / '.venv/share/jupyter')
os.environ['IPYTHONDIR'] = str(root / 'work/ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / 'work/jupyter_runtime')
import nbformat
from nbclient import NotebookClient
p = root / 'notebooks/01_data_preparation.ipynb'
nb = nbformat.read(p, as_version=4)
nb.cells[0].source = '# Stage 1: Data Preparation\nThe project uses a fresh Python 3.12 virtual environment. The proposal and original feedback are preserved. Time is removed and Class is the target. This stage prepares data without fitting or tuning models. Outputs below come from real data.'
nb.cells[-1].source = '## Current status\nQuality checks, cleaning, and the fixed stratified split are complete. See reports/data_quality_report.md. The raw file passes the OpenML MD5 check and has a recorded SHA-256. The test set remains sealed. No model is trained and no test performance is reported. Baseline CV, class weights, and cost comparisons remain future work. The pre-specified protocol is in config/baseline_protocol.json.'
nb.metadata.kernelspec = {'display_name': 'Fraud Project (.venv)', 'language': 'python', 'name': 'fraud-project'}
for c in nb.cells:
    if c.cell_type == 'code':
        c.outputs = []
        c.execution_count = None
try:
    NotebookClient(nb, timeout=900, kernel_name='fraud-project', resources={'metadata': {'path': str(root)}}).execute()
finally:
    nbformat.write(nb, p)
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
print('Notebook executed successfully with project virtual environment')
