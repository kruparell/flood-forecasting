import os
import sys
from pathlib import Path
import torch
import xarray as xr
import pandas as pd
import numpy as np

repo_root = Path('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.insert(0, str(repo_root))

from googlehydrology.utils.config import Config
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.evaluation.metrics import calculate_metrics

# Base Model directory for 50 basins
run_dir_me = repo_root / 'tutorial' / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
cfg_me = Config(run_dir_me / 'config.yml')

print(f"Base 50-basin model dir: {run_dir_me}")
print(f"Base experiment name: {cfg_me.experiment_name}")

# Let's inspect test_results.zarr in the base 50-basin model directory
epoch_dirs = sorted(list((run_dir_me / 'test').glob('model_epoch*')))
print(f"Epoch test dirs in base model: {epoch_dirs}")
if epoch_dirs:
    latest_test_dir = epoch_dirs[-1]
    p_zarr = latest_test_dir / 'test_results.zarr'
    print(f"p_zarr exists: {p_zarr.exists()}")
    if p_zarr.exists():
        ds = xr.open_zarr(p_zarr, consolidated=False).load()
        print(f"Variables in test_results: {list(ds.data_vars)}")
        print(f"Basins in test_results: {len(ds.basin.values)}")
