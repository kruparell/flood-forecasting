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

scaler_ds = xr.open_dataset(scaler_path)
print("=== SCALER DATASET ===")
print("Scaler dims:", scaler_ds.dims)
print("Scaler coords:", dict(scaler_ds.coords))
for var in scaler_ds.data_vars:
    print(f"Var: {var}, shape: {scaler_ds[var].shape}, dims: {scaler_ds[var].dims}")
    if "basin" in scaler_ds[var].dims:
        print(f"  Basin dim present! Basins: {len(scaler_ds[var].basin)}")
    else:
        print(f"  Values: {scaler_ds[var].values}")

ds_zarr = xr.open_zarr(zarr_path, consolidated=False).compute()
print("\n=== ZARR DATASET ===")
print("Zarr dims:", ds_zarr.dims)
print("Zarr coords:", dict(ds_zarr.coords))
for var in ds_zarr.data_vars:
    print(f"Var: {var}, shape: {ds_zarr[var].shape}, dims: {ds_zarr[var].dims}")

print("\n=== TIME_STEP vs DATE in ZARR ===")
# What is the time_step dimension in ds_zarr['streamflow_sim']?
sim_da = ds_zarr['streamflow_sim']
obs_da = ds_zarr['streamflow_obs']
print(f"sim_da dims: {sim_da.dims}, shape: {sim_da.shape}")
print(f"obs_da dims: {obs_da.dims}, shape: {obs_da.shape}")
print("time_step values:", ds_zarr['time_step'].values if 'time_step' in ds_zarr.coords else None)
print("date values (first 10):", ds_zarr['date'].values[:10])
print("date values (last 10):", ds_zarr['date'].values[-10:])
