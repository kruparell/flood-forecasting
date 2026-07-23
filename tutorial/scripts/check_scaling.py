import xarray as xr
import numpy as np
from pathlib import Path

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")
tut_dir = repo_root / "tutorial"
zarr_path = tut_dir / "model-runs" / "arlstm-50basin-example_2107_080318" / "test" / "model_epoch002" / "test_results.zarr"
pt_dir = repo_root / "pretrained-models" / "google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs"
scaler_path = pt_dir / "scaler.nc"

scaler_ds = xr.open_dataset(scaler_path)
mu_q = float(scaler_ds["streamflow_sim"].sel(parameter="mean").values)
sigma_q = float(scaler_ds["streamflow_sim"].sel(parameter="std").values)

ds_zarr = xr.open_zarr(zarr_path, consolidated=False).compute()

print(f"Scaler mu: {mu_q}, sigma: {sigma_q}")

for b in ds_zarr["basin"].values[:5]:
    obs = ds_zarr["streamflow_obs"].sel(basin=b, freq="1D", time_step=0).values.flatten()
    sim_raw = ds_zarr["streamflow_sim"].sel(basin=b, freq="1D", time_step=0).values.flatten()
    sim_unscaled = sim_raw * sigma_q + mu_q
    
    valid = ~np.isnan(obs)
    print(f"\nBasin {b}:")
    print(f"  Obs: min={np.nanmin(obs):.3f}, max={np.nanmax(obs):.3f}, mean={np.nanmean(obs):.3f}, std={np.nanstd(obs):.3f}")
    print(f"  Sim_raw: min={np.nanmin(sim_raw):.3f}, max={np.nanmax(sim_raw):.3f}, mean={np.nanmean(sim_raw):.3f}, std={np.nanstd(sim_raw):.3f}")
    print(f"  Sim_unscaled (raw * sigma + mu): min={np.nanmin(sim_unscaled):.3f}, max={np.nanmax(sim_unscaled):.3f}, mean={np.nanmean(sim_unscaled):.3f}")
    
    # Check metrics comparing obs vs sim_raw AND obs vs sim_unscaled
    denom = np.sum((obs[valid] - np.mean(obs[valid]))**2)
    nse_raw = 1.0 - np.sum((obs[valid] - sim_raw[valid])**2) / denom
    nse_unscaled = 1.0 - np.sum((obs[valid] - sim_unscaled[valid])**2) / denom
    print(f"  NSE with sim_raw: {nse_raw:.4f}")
    print(f"  NSE with sim_unscaled: {nse_unscaled:.4f}")
