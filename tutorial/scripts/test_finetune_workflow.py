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

run_dir_me = repo_root / 'tutorial' / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
run_dir_ar = repo_root / 'tutorial' / 'model-runs' / 'arlstm-50basin-example_2107_080318'

print(f"MeanEmbedding Config exists: {(run_dir_me / 'config.yml').exists()}")
print(f"ARLSTM Config exists: {(run_dir_ar / 'config.yml').exists()}")

cfg_me = Config(run_dir_me / 'config.yml')
tester_me = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='test', init_model=True)
print(f"Model class: {type(tester_me.model).__name__}")
print(f"Model module parts: {tester_me.model.module_parts}")
print(f"Total test basins: {len(tester_me.basins)}")
print(f"First 5 basins: {tester_me.basins[:5]}")
