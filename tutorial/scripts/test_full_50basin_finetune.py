import os
import sys
import copy
import time
from pathlib import Path
import torch
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np

repo_root = Path('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.insert(0, str(repo_root))

from googlehydrology.utils.config import Config
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.evaluation.metrics import calculate_metrics
from googlehydrology.datasetzoo.multimet import _convert_to_tensor

# Setup paths
run_dir_me = repo_root / 'tutorial' / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
cfg_me = Config(run_dir_me / 'config.yml')

print("Initializing tester for base model...")
tester = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='test', init_model=True)
basins = tester.basins
print(f"Total Basins: {len(basins)}")

# Verify model forward pass and fine-tuning mechanism
print(f"Model parts: {tester.model.module_parts}")
