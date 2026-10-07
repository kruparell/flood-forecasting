#!/usr/bin/env python3
import sys
import shutil
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, '/usr/local/google/home/kruparell/flood-forecasting')

from googlehydrology.utils.config import Config
from googlehydrology.evaluation import get_tester

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    obs = np.asarray(obs).ravel()
    sim = np.asarray(sim).ravel()
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o = obs[valid]
    s = sim[valid]
    if len(o) == 0:
        return {'NSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}
    
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - (np.sum((o - s) ** 2) / denom)) if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    mae = float(np.mean(np.abs(o - s)))
    return {'NSE': nse, 'RMSE': rmse, 'MAE': mae}

def main():
    model_runs_dir = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs')
    matching_runs = [
        d for d in sorted(list(model_runs_dir.glob('arlstm-10basin-real_*')))
        if list(d.glob('model_epoch*.pt'))
    ]
    if not matching_runs:
        raise FileNotFoundError("No matching model run directory with trained weights found.")
    
    run_dir = matching_runs[-1]
    epoch_files = sorted(list(run_dir.glob('model_epoch*.pt')))
    epoch = int(epoch_files[-1].stem.replace('model_epoch', ''))
    print(f"Targeting model run directory: {run_dir} (Epoch {epoch})")
    
    # Clean up previous evaluation results in test_dir
    test_dir = run_dir / 'test' / f'model_epoch{epoch:03d}'
    if test_dir.exists():
        print(f"Cleaning up previous test evaluation results in {test_dir}...")
        shutil.rmtree(test_dir, ignore_errors=True)
    
    # 1. Closed-Loop Evaluation (streamflow observations provided)
    print("\n--- Running Closed-Loop Evaluation (Streamflow Observations Provided) ---")
    cfg_closed = Config(run_dir / 'config.yml')
    tester_closed = get_tester(cfg=cfg_closed, run_dir=run_dir, period='test', init_model=True)
    tester_closed.evaluate(epoch=epoch, save_results=True)
    
    zarr_closed_path = test_dir / 'test_results.zarr'
    zarr_closed_saved = test_dir / 'test_results_data_assimilation.zarr'
    
    if zarr_closed_path.exists():
        if zarr_closed_saved.exists():
            shutil.rmtree(zarr_closed_saved)
        zarr_closed_path.rename(zarr_closed_saved)
        print(f"Saved Closed-Loop results to: {zarr_closed_saved}")
    
    # 2. Open-Loop Evaluation (streamflow observations set to NaN)
    print("\n--- Running Open-Loop Evaluation (Streamflow Observations Missing / Masked) ---")
    cfg_open_dict = deepcopy(cfg_closed._cfg)
    cfg_open_dict['random_holdout_from_dynamic_features'] = {
        'streamflow_shift1': {'missing_fraction': 1.0, 'mean_missing_length': 1}
    }
    cfg_open = Config(cfg_open_dict)
    cfg_open.run_dir = run_dir
    
    tester_open = get_tester(cfg=cfg_open, run_dir=run_dir, period='test', init_model=True)
    tester_open.evaluate(epoch=epoch, save_results=True)
    
    zarr_open_path = test_dir / 'test_results.zarr'
    print(f"Saved Open-Loop results to: {zarr_open_path}")
    
    # 3. Compute 10-Basin Hydrologic Evaluation Metrics
    print("\n--- Computing Hydrologic Evaluation Metrics across 10 Real Basins ---")
    ds_closed = xr.open_zarr(zarr_closed_saved, consolidated=False)
    ds_open = xr.open_zarr(zarr_open_path, consolidated=False)
    
    basins_closed = list(dict.fromkeys(ds_closed['basin'].values.tolist()))
    basins_open = list(dict.fromkeys(ds_open['basin'].values.tolist()))
    common_basins = [b for b in basins_closed if b in basins_open]
    print(f"Evaluated Basins ({len(common_basins)}): {common_basins}")
    
    results = []
    for b in common_basins:
        q_obs = ds_closed['streamflow_obs'].sel(basin=b).values
        q_closed = ds_closed['streamflow_sim'].sel(basin=b).values
        q_open = ds_open['streamflow_sim'].sel(basin=b).values
        
        m_closed = compute_metrics(q_obs, q_closed)
        m_open = compute_metrics(q_obs, q_open)
        
        results.append({
            'Basin ID': b,
            'Closed-Loop NSE': round(m_closed['NSE'], 4),
            'Closed-Loop RMSE': round(m_closed['RMSE'], 4),
            'Closed-Loop MAE': round(m_closed['MAE'], 4),
            'Open-Loop NSE': round(m_open['NSE'], 4),
            'Open-Loop RMSE': round(m_open['RMSE'], 4),
            'Open-Loop MAE': round(m_open['MAE'], 4),
            'ΔNSE (Closed - Open)': round(m_closed['NSE'] - m_open['NSE'], 4)
        })
    
    df_res = pd.DataFrame(results)
    print("\n=== 10-Basin ARLSTM Hydrologic Evaluation Summary ===")
    print(df_res.to_string(index=False))
    
    # Save metrics table CSV inside run_dir & test_dir
    csv_out = test_dir / '10basin_evaluation_summary.csv'
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved 10-basin evaluation summary CSV to: {csv_out}")

if __name__ == '__main__':
    main()
