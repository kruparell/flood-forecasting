import os
import sys
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")
tut_dir = repo_root / "tutorial"

zarr_path = tut_dir / "model-runs" / "arlstm-50basin-example_2107_080318" / "test" / "model_epoch002" / "test_results.zarr"
pt_dir = repo_root / "pretrained-models" / "google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs"
scaler_path = pt_dir / "scaler.nc"

print("--- SCALER.NC ---")
scaler_ds = xr.open_dataset(scaler_path)
print(scaler_ds)
for v in scaler_ds.data_vars:
    print(v, scaler_ds[v].dims, scaler_ds[v].values if scaler_ds[v].size < 10 else scaler_ds[v].shape)

print("\n--- TEST_RESULTS.ZARR ---")
ds_zarr = xr.open_zarr(zarr_path, consolidated=False).compute()
print(ds_zarr)
print("Data vars:", list(ds_zarr.data_vars))
print("Coords / Dims:", ds_zarr.dims)
print("time_step values:", ds_zarr["time_step"].values if "time_step" in ds_zarr else "No time_step")
if "lead_time" in ds_zarr:
    print("lead_time values:", ds_zarr["lead_time"].values)

sample_basin = ds_zarr["basin"].values[0]
print(f"\nSample basin: {sample_basin}")
obs = ds_zarr["streamflow_obs"].sel(basin=sample_basin, freq="1D").values
sim = ds_zarr["streamflow_sim"].sel(basin=sample_basin, freq="1D").values
print(f"obs shape: {obs.shape}, min: {np.nanmin(obs):.4f}, max: {np.nanmax(obs):.4f}, mean: {np.nanmean(obs):.4f}")
print(f"sim shape: {sim.shape}, min: {np.nanmin(sim):.4f}, max: {np.nanmax(sim):.4f}, mean: {np.nanmean(sim):.4f}")

# Check all time_steps
for ts in range(sim.shape[0]):
    obs_ts = obs[ts]
    sim_ts = sim[ts]
    valid = ~np.isnan(obs_ts) & ~np.isnan(sim_ts)
    o, s = obs_ts[valid], sim_ts[valid]
    corr = np.corrcoef(o, s)[0, 1] if len(o) > 1 else np.nan
    denom = np.sum((o - np.mean(o))**2)
    nse = 1.0 - np.sum((o - s)**2) / denom if denom > 0 else np.nan
    print(f"time_step={ts}: NSE={nse:.4f}, r={corr:.4f}, mean_obs={np.mean(o):.4f}, mean_sim={np.mean(s):.4f}")
