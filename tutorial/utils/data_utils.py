"""Data and File I/O Utilities for Caravans & Flood Forecasting."""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Set, Dict, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fallback gfile import for CNS file I/O inside Borg task containers
try:
    from google3.pyglib import gfile
except ImportError:
    try:
        from pyglib import gfile
    except ImportError:
        gfile = None


def file_exists(path_or_str: Union[str, Path]) -> bool:
    """Checks file/directory existence using gfile for CNS paths or Path.exists() fallback."""
    path_str = str(path_or_str)
    if gfile is not None:
        try:
            return gfile.Exists(path_str)
        except Exception:
            pass
    return Path(path_str).exists()


def open_file(path_or_str: Union[str, Path], mode: str = "r"):
    """Opens file using gfile.GFile for CNS paths or built-in open() fallback."""
    path_str = str(path_or_str)
    if gfile is not None:
        try:
            return gfile.GFile(path_str, mode)
        except Exception:
            pass
    return open(path_str, mode)


def load_basin_list(
    file_path: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_basins: Optional[int] = None,
    random_seed: int = 42,
    exclude_basins: Optional[Set[str]] = None,
    caravan_base_dir: Optional[Union[str, Path]] = None,
    static_attributes: Optional[List[str]] = None,
) -> List[str]:
    """Reads basin list CSV or TXT file, filters for complete date coverage, excludes specified basins,
    and reproducibly samples num_basins.
    """
    file_path_str = str(file_path)
    if not file_exists(file_path_str):
        raise FileNotFoundError(f"Basin list file not found: {file_path_str}")

    if file_path_str.lower().endswith(".csv"):
        with open_file(file_path_str, "r") as f:
            df = pd.read_csv(f)

        if start_date and end_date and {"valid_start_date", "valid_end_date"}.issubset(df.columns):
            eval_start = pd.to_datetime(start_date)
            eval_end = pd.to_datetime(end_date)
            df["valid_start_date"] = pd.to_datetime(df["valid_start_date"])
            df["valid_end_date"] = pd.to_datetime(df["valid_end_date"])

            valid_mask = (df["valid_start_date"] <= eval_start) & (df["valid_end_date"] >= eval_end)
            df = df[valid_mask]

        if "basin_id" in df.columns:
            basins = df["basin_id"].dropna().astype(str).str.strip().unique().tolist()
        else:
            basins = df.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
    else:
        with open_file(file_path_str, "r") as f:
            basins = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if exclude_basins:
        basins = [b for b in basins if b not in exclude_basins]

    if caravan_base_dir and static_attributes:
        attrs_df = load_caravan_attributes(caravan_base_dir)
        if not attrs_df.empty:
            avail_cols = [c for c in static_attributes if c in attrs_df.columns]
            if avail_cols:
                valid_basin_set = set(attrs_df.dropna(subset=avail_cols).index)
                before_cnt = len(basins)
                basins = [b for b in basins if b in valid_basin_set]
                print(f"  -> Static Attribute NaN Filter: {len(basins)} basins remaining out of {before_cnt} (excluded {before_cnt - len(basins)} with NaN attributes).")

    if num_basins and 0 < num_basins < len(basins):
        rng = np.random.RandomState(random_seed)
        basins = sorted(list(rng.choice(basins, size=num_basins, replace=False)))
    else:
        basins = sorted(basins)

    return basins


def load_caravan_attributes(caravan_base_dir: Union[str, Path]) -> pd.DataFrame:
    """Loads and combines all CSV static attribute files across all source folders under attributes/."""
    caravan_base_dir_str = str(caravan_base_dir)
    attributes_dir = os.path.join(caravan_base_dir_str, "attributes")
    source_dfs = []

    if not file_exists(attributes_dir):
        return pd.DataFrame()

    if gfile is not None:
        try:
            csv_files = gfile.Glob(os.path.join(attributes_dir, "*", "*.csv"))
        except Exception:
            csv_files = glob.glob(os.path.join(attributes_dir, "*", "*.csv"))
    else:
        csv_files = glob.glob(os.path.join(attributes_dir, "*", "*.csv"))

    csv_by_dir: Dict[str, List[str]] = {}
    for csv_f in csv_files:
        d = os.path.dirname(csv_f)
        csv_by_dir.setdefault(d, []).append(csv_f)

    for src_dir, files in csv_by_dir.items():
        src_dfs = []
        for csv_file in files:
            try:
                with open_file(csv_file, "r") as f:
                    df = pd.read_csv(f)
                if "gauge_id" in df.columns:
                    src_dfs.append(df.set_index("gauge_id"))
            except Exception:
                pass
        if src_dfs:
            df_src = pd.concat(src_dfs, axis=1)
            df_src = df_src.loc[:, ~df_src.columns.duplicated()]
            source_dfs.append(df_src)

    if not source_dfs:
        return pd.DataFrame()

    combined = pd.concat(source_dfs, axis=0)
    return combined.loc[:, ~combined.columns.duplicated()]


def build_basin_nc_index(caravan_base_dir: Union[str, Path]) -> Dict[str, Union[str, Path]]:
    """Scans dataset directory once to build a fast O(1) basin_id -> NetCDF file path lookup index."""
    caravan_base_dir_str = str(caravan_base_dir)
    timeseries_dir = os.path.join(caravan_base_dir_str, "timeseries", "netcdf")
    if not file_exists(timeseries_dir):
        return {}

    if gfile is not None:
        try:
            nc_files = gfile.Glob(os.path.join(timeseries_dir, "*", "*.nc"))
        except Exception:
            nc_files = glob.glob(os.path.join(timeseries_dir, "*", "*.nc"))
    else:
        nc_files = glob.glob(os.path.join(timeseries_dir, "*", "*.nc"))

    index: Dict[str, Union[str, Path]] = {}
    for f in nc_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        index[stem] = f
    return index
