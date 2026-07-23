import glob
import xarray as xr
from pathlib import Path

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")

zarr_files = list(repo_root.glob("**/test_results.zarr"))
print(f"Found {len(zarr_files)} test_results.zarr files:")
for zf in sorted(zarr_files):
    try:
        ds = xr.open_zarr(zf, consolidated=False)
        print(f"\nPath: {zf.relative_to(repo_root)}")
        print(f"  dims: {dict(ds.dims)}")
        print(f"  coords: {list(ds.coords)}")
        if 'time_step' in ds.coords:
            print(f"  time_step values: {ds['time_step'].values}")
        if 'lead_time' in ds.coords:
            print(f"  lead_time values: {ds['lead_time'].values}")
    except Exception as e:
        print(f"  Error reading {zf}: {e}")
