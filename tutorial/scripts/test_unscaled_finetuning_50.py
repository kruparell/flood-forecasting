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

# Load the 50-basin finetuning results CSV
csv_path = repo_root / 'tutorial' / 'model-runs' / '50_basin_finetuning_results.csv'
df = pd.read_csv(csv_path)

print("50-BASIN RESULTS SUMMARY:")
print(df.describe().to_string())

# Check percentage of basins that improved after fine-tuning
if 'NSE_Post_Finetuning' in df.columns:
    improved_nse = (df['NSE_Post_Finetuning'] > df['NSE_Pre_Finetuning']).sum()
    improved_kge = (df['KGE_Post_Finetuning'] > df['KGE_Pre_Finetuning']).sum()
    improved_r = (df['PearsonR_Post_Finetuning'] > df['PearsonR_Pre_Finetuning']).sum()
    improved_rmse = (df['RMSE_Post_Finetuning'] < df['RMSE_Pre_Finetuning']).sum()
else:
    improved_nse = (df['NSE'] > -0.328).sum()
    improved_kge = (df['KGE'] > 0.0).sum()
    improved_r = (df['Pearson-r'] > 0.0).sum()
    improved_rmse = (df['RMSE'] < 5.0).sum()

print(f"\nImprovements across 50 basins:")
print(f"  NSE improved/positive in:      {improved_nse}/50 basins ({improved_nse/50*100:.1f}%)")
print(f"  KGE improved/positive in:      {improved_kge}/50 basins ({improved_kge/50*100:.1f}%)")
print(f"  Pearson-r improved/positive in: {improved_r}/50 basins ({improved_r/50*100:.1f}%)")
print(f"  RMSE acceptable in:            {improved_rmse}/50 basins ({improved_rmse/50*100:.1f}%)")

