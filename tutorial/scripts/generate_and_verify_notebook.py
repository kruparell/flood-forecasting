"""Generate and verify the 50-basin fine-tuning report notebook."""

import sys
from pathlib import Path
import nbformat as nbf

_SCRIPT_DIR = Path(__file__).resolve().parent
_TUTORIAL_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TUTORIAL_DIR.parent

for _p in [_REPO_ROOT, _TUTORIAL_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

nb_path = _TUTORIAL_DIR / 'notebooks' / 'Finetuning_50_Basin_Report.ipynb'
nb = nbf.v4.new_notebook()

c0 = nbf.v4.new_markdown_cell("""# OpenHydroNet 50-Basin Fine-Tuning Performance Report

This report evaluates the **OpenHydroNet Fine-Tuning Workflow** across all **50 CAMELS basins** (`tutorial/basin-lists/50-basin-train.txt`).
""")

c1 = nbf.v4.new_code_cell("""%matplotlib inline
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nb_dir = Path.cwd()
tut_dir = nb_dir.parent if nb_dir.name == 'notebooks' else (nb_dir if nb_dir.name == 'tutorial' else nb_dir / 'tutorial')
repo_root = tut_dir.parent

for p in [repo_root, tut_dir, tut_dir / 'scripts', tut_dir / 'src']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import backend
""")

nb.cells = [c0, c1]
nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(nb_path, 'w') as f:
    nbf.write(nb, f)
print(f"Generated fine-tuning report notebook at: {nb_path}")
