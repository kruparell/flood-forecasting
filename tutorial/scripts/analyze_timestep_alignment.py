import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")

# Let's inspect generic-meanembedding-50basin
p1 = repo_root / "tutorial/model-runs/generic-meanembedding-50basin_2107_080323/test/model_epoch055/test_results.zarr"
ds1 = xr.open_zarr(p1, consolidated=False).compute()

print("ds1 time_steps:", ds1["time_step"].values)
b = ds1["basin"].values[0]

for ts in ds1["time_step"].values:
    obs = ds1["streamflow_obs"].sel(basin=b, freq="1D", time_step=ts).values.flatten()
    sim = ds1["streamflow_sim"].sel(basin=b, freq="1D", time_step=ts).values.flatten()
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if valid.sum() > 10:
        o, s = obs[valid], sim[valid]
        r = np.corrcoef(o, s)[0, 1]
        denom = np.sum((o - np.mean(o))**2)
        nse = 1.0 - np.sum((o - s)**2) / denom if denom > 0 else np.nan
        print(f"generic time_step={ts:2d}: NSE={nse:+.4f}, r={r:+.4f}")

# Also check 5-basin-example
p2 = repo_root / "tutorial/model-runs/5-basin-example/finetune-camels_13235000/test/model_epoch025/test_results.zarr"
ds2 = xr.open_zarr(p2, consolidated=False).compute()
print("\nds2 time_steps:", ds2["time_step"].values)
b2 = ds2["basin"].values[0]
for ts in ds2["time_step"].values:
    obs = ds2["streamflow_obs"].sel(basin=b2, freq="1D", time_step=ts).values.flatten()
    sim = ds2["streamflow_sim"].sel(basin=b2, freq="1D", time_step=ts).values.flatten()
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if valid.sum() > 10:
        o, s = obs[valid], sim[valid]
        r = np.corrcoef(o, s)[0, 1]
        denom = np.sum((o - np.mean(o))**2)
        nse = 1.0 - np.sum((o - s)**2) / denom if denom > 0 else np.nan
        print(f"5-basin time_step={ts:2d}: NSE={nse:+.4f}, r={r:+.4f}")
