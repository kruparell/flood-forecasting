import os
import sys
from pathlib import Path

repo_root = Path('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.insert(0, str(repo_root))

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = repo_root / 'tutorial' / 'Finetuning_50_Basin_Report.ipynb'

# 1. Generate the base notebook structure
import generate_notebook_report

# 2. Read the notebook
with open(notebook_path) as f:
    nb = nbformat.read(f, as_version=4)

# 3. Execute all cells
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
print(f"Executing notebook: {notebook_path} ...")
ep.preprocess(nb, {'metadata': {'path': str(repo_root / 'tutorial')}})

# 4. Save executed notebook with all outputs and figures
with open(notebook_path, 'w') as f:
    nbformat.write(nb, f)

print(f"Successfully executed and saved notebook at {notebook_path}!")
