import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")
tut_dir = repo_root / "tutorial"
zarr_path = tut_dir / "model-runs" / "arlstm-50basin-example_2107_080318" / "test" / "model_epoch010" / "test_results.zarr"

ds_zarr = xr.open_zarr(zarr_path, consolidated=False).compute()

dates = pd.to_datetime(ds_zarr["date"].values)
mask_period = (dates >= pd.to_datetime("2011-10-01")) & (dates <= pd.to_datetime("2012-09-30"))

print(f"Total dates: {len(dates)}")
print(f"Dates in test period 2011-10-01 to 2012-09-30: {mask_period.sum()} days")

sample_basins = ["camels_01054200", "camels_01195100", "camels_01350000", "camels_01413500"]

for b in sample_basins:
    obs = ds_zarr["streamflow_obs"].sel(basin=b, freq="1D", time_step=0).values.flatten()
    sim_raw = ds_zarr["streamflow_sim"].sel(basin=b, freq="1D", time_step=0).values.flatten()
    sim_ar = np.maximum(0.0, sim_raw)
    
    obs_test = obs[mask_period]
    sim_test = sim_ar[mask_period]
    
    valid = ~np.isnan(obs_test) & ~np.isnan(sim_test)
    o, s = obs_test[valid], sim_test[valid]
    denom = np.sum((o - np.mean(o))**2)
    nse = 1.0 - np.sum((o - s)**2) / denom if denom > 0 else np.nan
    r = np.corrcoef(o, s)[0, 1] if len(o) > 1 else np.nan
    print(f"Basin {b}: Test Period Days={len(o)}, NSE={nse:.3f}, Pearson-r={r:.3f}")
