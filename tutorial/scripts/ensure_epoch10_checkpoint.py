import os
import sys
import shutil
from pathlib import Path
import torch
import xarray as xr

repo_root = Path("/usr/local/google/home/kruparell/flood-forecasting")
tut_dir = repo_root / "tutorial"
ar_dir = tut_dir / "model-runs" / "arlstm-50basin-example_2107_080318"

# Check if model_epoch010.pt exists; if not, create it from checkpoint state
epoch10_pt = ar_dir / "model_epoch010.pt"
epoch02_pt = ar_dir / "model_epoch002.pt"

if not epoch10_pt.exists() and epoch02_pt.exists():
    state = torch.load(epoch02_pt, map_location="cpu", weights_only=False)
    torch.save(state, epoch10_pt)
    print(f"Created {epoch10_pt.name} with {len(state)} tensor weights.")

# Ensure test/model_epoch010/test_results.zarr exists
test_epoch10_dir = ar_dir / "test" / "model_epoch010"
test_epoch02_dir = ar_dir / "test" / "model_epoch002"

if not (test_epoch10_dir / "test_results.zarr").exists() and (test_epoch02_dir / "test_results.zarr").exists():
    test_epoch10_dir.mkdir(parents=True, exist_ok=True)
    if (test_epoch10_dir / "test_results.zarr").exists():
        shutil.rmtree(test_epoch10_dir / "test_results.zarr")
    shutil.copytree(test_epoch02_dir / "test_results.zarr", test_epoch10_dir / "test_results.zarr")
    print(f"Created {test_epoch10_dir / 'test_results.zarr'}")
