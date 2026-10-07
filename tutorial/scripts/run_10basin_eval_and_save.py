#!/usr/bin/env python3
"""
Evaluates trained 10-basin ARLSTM model in both Closed-Loop and Open-Loop modes using googlehydrology's tester.
Saves test_results_data_assimilation.zarr (Closed-Loop) and test_results.zarr (Open-Loop),
and produces per-basin NSE, RMSE, MAE metrics summary CSV.
"""

import sys
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from googlehydrology.utils.config import Config
from googlehydrology.evaluation.tester import RegressionTester

def compute_metrics(obs, sim):
    valid = np.isfinite(obs) & np.isfinite(sim)
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    o = obs[valid]
    s = sim[valid]
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = 1.0 - (np.sum((o - s) ** 2) / denom) if denom > 0 else np.nan
    rmse = np.sqrt(np.mean((o - s) ** 2))
    mae = np.mean(np.abs(o - s))
    return float(nse), float(rmse), float(mae)

def main():
    run_dir = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/model-runs/arlstm-10basin-real_2907_161743')
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        
    cfg = Config(run_dir / "config.yml")
    cfg.inference_mode = True
    
    test_epoch_dir = run_dir / "test" / "model_epoch002"
    
    print("=== Step 1: Closed-Loop Evaluation (Streamflow Observations Provided) ===")
    tester_closed = RegressionTester(cfg=cfg, run_dir=run_dir, period='test', init_model=True)
    tester_closed.evaluate(epoch=2, save_results=True, metrics=cfg.metrics)
    
    zarr_closed = test_epoch_dir / "test_results_data_assimilation.zarr"
    orig_zarr = test_epoch_dir / "test_results.zarr"
    orig_csv = test_epoch_dir / "test_metrics.csv"
    closed_csv = test_epoch_dir / "test_metrics_data_assimilation.csv"
    
    if zarr_closed.exists():
        shutil.rmtree(zarr_closed)
    if orig_zarr.exists():
        shutil.move(orig_zarr, zarr_closed)
    if orig_csv.exists():
        shutil.copy(orig_csv, closed_csv)
        
    print("=== Step 2: Open-Loop Evaluation (Streamflow Observations Missing / Feedback Unrolled) ===")
    tester_open = RegressionTester(cfg=cfg, run_dir=run_dir, period='test', init_model=True)
    
    # Monkey-patch tester_open._evaluate to mask autoregressive feature with NaN
    orig_eval_fn = tester_open._evaluate
    def masked_evaluate(model, loader, frequencies, basins, data_assimilation):
        import torch
        # We wrap the loader to yield batches with NaN in the last feature column of x_d
        class MaskedLoader:
            def __init__(self, original_loader):
                self.loader = original_loader
                self.dataset = original_loader.dataset
            def __iter__(self):
                for batch in self.loader:
                    if 'x_d' in batch:
                        if isinstance(batch['x_d'], dict):
                            for sub_k in batch['x_d']:
                                if isinstance(batch['x_d'][sub_k], torch.Tensor):
                                    batch['x_d'][sub_k][..., -1] = torch.nan
                        elif isinstance(batch['x_d'], torch.Tensor):
                            batch['x_d'][..., -1] = torch.nan
                    yield batch
            def __len__(self):
                return len(self.loader)
        return orig_eval_fn(model, MaskedLoader(loader), frequencies, basins, data_assimilation)
        
    tester_open._evaluate = masked_evaluate
    tester_open.evaluate(epoch=2, save_results=True, metrics=cfg.metrics)
    
    zarr_open = orig_zarr
    open_csv = orig_csv
    
    print("=== Step 3: Computing 10-Basin Hydrologic Evaluation Metrics (NSE, RMSE, MAE) ===")
    ds_closed = xr.open_zarr(zarr_closed, consolidated=False)
    ds_open = xr.open_zarr(zarr_open, consolidated=False)
    
    basins = sorted(ds_closed['basin'].values)
    metrics_list = []
    
    for b in basins:
        b_str = str(b)
        obs_cl = ds_closed['streamflow_obs'].sel(basin=b).values.squeeze()
        sim_cl = ds_closed['streamflow_sim'].sel(basin=b).values.squeeze()
        
        obs_op = ds_open['streamflow_obs'].sel(basin=b).values.squeeze()
        sim_op = ds_open['streamflow_sim'].sel(basin=b).values.squeeze()
        
        # Squeeze out potential frequency dimensions if any
        if obs_cl.ndim > 1:
            obs_cl = obs_cl.flatten()
            sim_cl = sim_cl.flatten()
            sim_op = sim_op.flatten()
            
        nse_cl, rmse_cl, mae_cl = compute_metrics(obs_cl, sim_cl)
        nse_op, rmse_op, mae_op = compute_metrics(obs_cl, sim_op)
        gain_nse = nse_cl - nse_op if not np.isnan(nse_cl) and not np.isnan(nse_op) else np.nan
        
        metrics_list.append({
            'Basin ID': b_str,
            'Closed-Loop NSE': round(nse_cl, 4),
            'Open-Loop NSE': round(nse_op, 4),
            'ΔNSE (Gain)': round(gain_nse, 4),
            'Closed-Loop RMSE': round(rmse_cl, 4),
            'Open-Loop RMSE': round(rmse_op, 4),
            'Closed-Loop MAE': round(mae_cl, 4),
            'Open-Loop MAE': round(mae_op, 4)
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    out_csv = run_dir / "10_basin_arlstm_evaluation_metrics.csv"
    df_metrics.to_csv(out_csv, index=False)
    print(f"\nSaved 10-basin metrics to {out_csv}:")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
