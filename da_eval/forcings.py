"""On-demand precipitation forcings for case-study hydrographs.

Reads the pre-materialized MultiMet eval subset (``<subset>/dynamics/<PRODUCT>/timeseries.zarr``) that the
sweep was evaluated on, and aligns every product to the physical day the precipitation fell on (to compare against
observations): a lead-``L`` forecast (24h accumulation) for day ``T`` was issued on ``T - (L-1)``.

* Forecast products (``HRES``, ``GraphCast``) have a ``lead_time`` axis: value = ``P[date=T-(L-1), lead=L]``,
  (verified: HRES lead 1 correlates best with ERA5-Land at lag 0 under this shift).
* Hindcast products (``ERA5-Land``, ``CPC``, ``IMERG``) have no lead axis: value = ``P[date=T]`` (what fell
  on the valid day; the model only sees these in the hindcast window).

Only the precipitation variables are needed, so ``cache_precip_subset`` copies just those arrays from CNS
(~0.2 GB for 4,287 basins x 1,102 days) into a local cache; reads are then per-basin and instant.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# display name -> (product dir, variable, is_forecast)
PRECIP_PRODUCTS: Dict[str, tuple] = {
    "HRES": ("HRES", "hres_total_precipitation", True),
    "GraphCast": ("GRAPHCAST", "graphcast_total_precipitation", True),
    "ERA5-Land": ("ERA5_LAND", "era5land_total_precipitation", False),
    "CPC": ("CPC", "cpc_precipitation", False),
    "IMERG": ("IMERG", "imerg_precipitation", False),
}

PRECIP_COLORS = {
    "HRES": "#1565c0",
    "GraphCast": "#8e24aa",
    "ERA5-Land": "#2e7d32",
    "CPC": "#ef6c00",
    "IMERG": "#00838f",
}

DEFAULT_SUBSET = "/cns/jn-d/home/floods/hydro_model/work/kruparell/eval_subsets/sweep4287_2017_2018"


def normalize_precip_names(names: Optional[Iterable[str]]) -> List[str]:
    """Maps loose spellings (``"era5"``, ``"graphcast"``, ``"hres"``) to ``PRECIP_PRODUCTS`` keys."""
    out = []
    for n in names or []:
        k = str(n).strip().lower().replace("_", "-")
        for name in PRECIP_PRODUCTS:
            nl = name.lower()
            if k == nl or nl.startswith(k) or (k.startswith("era5") and name == "ERA5-Land"):
                if name not in out:
                    out.append(name)
                break
    return out


def cache_precip_subset(subset_dir: str = DEFAULT_SUBSET, local_dir: str | Path = ".", overwrite: bool = False) -> Path:
    """Copies only the precipitation arrays (+ coords) of each product from CNS into ``local_dir``."""
    local_dir = Path(local_dir)
    src = subset_dir.rstrip("/") + "/dynamics"

    def _copy(item):
        _, (prod, var, is_fc) = item
        dst = local_dir / prod / "timeseries.zarr"
        arrays = ["basin", "date"] + (["lead_time"] if is_fc else []) + [var]
        if all((dst / v / ".zarray").exists() for v in arrays) and not overwrite:
            return
        dst.mkdir(parents=True, exist_ok=True)
        for f in (".zgroup", ".zattrs"):
            subprocess.run(["fileutil", "cp", "-f", f"{src}/{prod}/timeseries.zarr/{f}", str(dst) + "/"], check=False,
                           capture_output=True)
        for v in arrays:
            has_chunks = (dst / v).is_dir() and any((dst / v).glob("[!.]*"))
            if not has_chunks or overwrite:
                subprocess.run(["fileutil", "cp", "-R", "-f", f"{src}/{prod}/timeseries.zarr/{v}", str(dst) + "/"],
                               check=True, capture_output=True)
            # `fileutil cp -R` skips dotfiles: copy array metadata explicitly.
            for f in (".zarray", ".zattrs"):
                subprocess.run(["fileutil", "cp", "-f", f"{src}/{prod}/timeseries.zarr/{v}/{f}", str(dst / v) + "/"],
                               check=False, capture_output=True)

    with ThreadPoolExecutor(len(PRECIP_PRODUCTS)) as ex:
        list(ex.map(_copy, PRECIP_PRODUCTS.items()))
    return local_dir


class PrecipForcingStore:
    """Per-basin precipitation reader over a local copy of the eval-subset forcings."""

    def __init__(self, root: str | Path):
        import xarray as xr

        self.root = Path(root)
        self._ds = {}
        self._basin_idx = {}
        for name, (prod, var, is_fc) in PRECIP_PRODUCTS.items():
            p = self.root / prod / "timeseries.zarr"
            if not (p / var).exists():
                continue
            ds = xr.open_zarr(str(p), consolidated=False)[[var]]
            self._ds[name] = ds
            self._basin_idx[name] = {str(b): i for i, b in enumerate(ds["basin"].values)}
        self._row_cache: Dict[tuple, pd.DataFrame] = {}

    @property
    def products(self) -> List[str]:
        return list(self._ds)

    @staticmethod
    def discover(data_dir: str | Path) -> Optional[Path]:
        cache = Path(data_dir) / "_ts_cache"
        if not cache.exists():
            return None
        for d in sorted(cache.glob("forcings_*")):
            if any((d / prod / "timeseries.zarr" / var).exists() for prod, var, _ in PRECIP_PRODUCTS.values()):
                return d
        return None

    def _basin_series(self, name: str, basin: str) -> Optional[pd.DataFrame]:
        """Raw per-basin array for one product: index=date, columns=lead days (forecast) or ['value']."""
        key = (name, basin)
        if key in self._row_cache:
            return self._row_cache[key]
        ds = self._ds.get(name)
        idx = self._basin_idx.get(name, {}).get(basin)
        if ds is None or idx is None:
            self._row_cache[key] = None
            return None
        _, var, is_fc = PRECIP_PRODUCTS[name]
        arr = ds[var].isel(basin=idx).values
        dates = pd.to_datetime(ds["date"].values)
        if is_fc:
            leads = ds["lead_time"].values
            if np.issubdtype(np.asarray(leads).dtype, np.timedelta64):
                leads = (np.asarray(leads) / np.timedelta64(1, "D")).astype(int)
            frame = pd.DataFrame(arr, index=dates, columns=[int(l) for l in leads])
        else:
            frame = pd.DataFrame({"value": arr}, index=dates)
        self._row_cache[key] = frame
        return frame

    def get_basin(self, basin: str, lead_times: Sequence[int] = (1, 7),
                  products: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Long frame: ``Valid Date``, ``Lead Time (Days)``, one column per product (mm/day)."""
        names = [p for p in (products or self.products) if p in self._ds]
        frames = []
        for lt in lead_times:
            cols = {}
            for name in names:
                raw = self._basin_series(name, basin)
                if raw is None:
                    continue
                if PRECIP_PRODUCTS[name][2]:
                    if int(lt) not in raw.columns:
                        continue
                    s = raw[int(lt)].copy()
                    # Row date d / lead L holds the 24h accumulation for physical day d+L-1 (lead 1 = day of
                    # issue). Verified: with this shift HRES lead-1 correlates best with ERA5-Land at lag 0.
                    s.index = s.index + pd.Timedelta(days=int(lt) - 1)
                else:
                    s = raw["value"]
                cols[name] = s
            if cols:
                f = pd.DataFrame(cols)
                f.index.name = "Valid Date"
                f = f.reset_index()
                f["Lead Time (Days)"] = int(lt)
                frames.append(f)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
