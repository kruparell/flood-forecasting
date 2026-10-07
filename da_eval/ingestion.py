"""Data ingestion, schema normalization, and attribute extraction for DA Evaluation.

Strictly scopes data loading to the specified experiment directory to prevent cross-contamination.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from da_eval.lazy_timeseries import block_to_frame, reduce_samples


def is_rho_config(cid: str) -> bool:
    """PP error-correction post-processing configs (fixed rho or per-basin rho); not DA."""
    c = str(cid).lower()
    return c.startswith("ar1_") or "postprocess_rho" in c or "best_rho" in c


def is_per_basin_rho_config(cid: str) -> bool:
    c = str(cid).lower()
    return is_rho_config(cid) and ("per_basin" in c or "per-basin" in c)


def sync_cns_shards_locally(
    cns_dir: str,
    local_dir: str,
    max_shards: Optional[int] = None,
    sync_timeseries: bool = False,
) -> None:
    """Synchronizes completed evaluation and timeseries shard parquets from CNS.

    Args:
        cns_dir: CNS root experiment directory.
        local_dir: Local staging directory on cloudtop.
        max_shards: If specified, only syncs up to max_shards shards for faster turnaround.
        sync_timeseries: If True, also syncs all basin timeseries parquets. Defaults to False.
    """
    os.makedirs(local_dir, exist_ok=True)
    try:
        parquet_lines = []
        if max_shards is not None:
            # Discover existing shard directories and only sync up to max_shards
            res = subprocess.run(f"fileutil ls -d {cns_dir}/shard_* 2>/dev/null", shell=True, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                shard_dirs = [d.strip() for d in res.stdout.strip().split("\n") if d.strip()]
                def _shard_key(s):
                    m = re.search(r'shard_(\d+)', s)
                    return int(m.group(1)) if m else 999999
                shard_dirs = sorted(shard_dirs, key=_shard_key)[:max_shards]
                search_paths = []
                for sd in shard_dirs:
                    search_paths.append(f"{sd}/")
                    if sync_timeseries:
                        search_paths.extend([f"{sd}/timeseries/", f"{sd}/timeseries_forecasts/"])
            else:
                search_paths = [f"{cns_dir}/shard_{i}/" for i in range(max_shards)]
                if sync_timeseries:
                    for i in range(max_shards):
                        search_paths.extend([f"{cns_dir}/shard_{i}/timeseries/", f"{cns_dir}/shard_{i}/timeseries_forecasts/"])
        else:
            search_paths = [f"{cns_dir}/shard_*/"]
            if sync_timeseries:
                search_paths.extend([
                    f"{cns_dir}/shard_*/timeseries/",
                    f"{cns_dir}/shard_*/timeseries_forecasts/",
                    f"{cns_dir}/timeseries/",
                ])

        for p in search_paths:
            res = subprocess.run(f"fileutil ls {p} 2>/dev/null", shell=True, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                parquet_lines.extend([l for l in lines if l.endswith(".parquet")])

        unique_parquets = list(set(parquet_lines))
        if unique_parquets:
            def _copy(line):
                fname = os.path.basename(line)
                dest = os.path.join(local_dir, fname)
                if not os.path.exists(dest) or os.path.getsize(dest) < 100:
                    subprocess.run(["fileutil", "cp", line, dest], capture_output=True)

            with ThreadPoolExecutor(max_workers=16) as ex:
                ex.map(_copy, unique_parquets)
            return

        # Fallback for canonical tester.py CNS layout:
        # <cns_dir>/<config_id>/shard_NN/test/model_epochNNN/test_metrics*.csv
        # or <cns_dir>/<config_id>/test/model_epochNNN/test_metrics*.csv
        csv_patterns = [
            f"{cns_dir}/*/shard_*/test/model_epoch*/test_metrics*.csv",
            f"{cns_dir}/*/test/model_epoch*/test_metrics*.csv",
        ]
        csv_lines = []
        for pat in csv_patterns:
            res = subprocess.run(f"fileutil ls \"{pat}\" 2>/dev/null", shell=True, capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                csv_lines.extend([l.strip() for l in res.stdout.strip().split("\n") if l.strip().endswith(".csv")])

        unique_csvs = sorted(list(set(csv_lines)))
        if unique_csvs:
            print(f"[CNS SYNC] Syncing {len(unique_csvs)} tester.py metric CSVs from {cns_dir} to {local_dir}...")
            cns_prefix = cns_dir.rstrip("/") + "/"

            def _copy_csv(cns_path: str):
                rel = cns_path[len(cns_prefix):] if cns_path.startswith(cns_prefix) else os.path.basename(cns_path)
                dest = os.path.join(local_dir, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if not os.path.exists(dest) or os.path.getsize(dest) < 50:
                    subprocess.run(["fileutil", "cp", cns_path, dest], capture_output=True)

            with ThreadPoolExecutor(max_workers=16) as ex:
                list(ex.map(_copy_csv, unique_csvs))

        if sync_timeseries:
            zarr_patterns = [
                f"{cns_dir}/*/shard_*/test/model_epoch*/test_results*.zarr",
                f"{cns_dir}/*/test/model_epoch*/test_results*.zarr",
            ]
            zarr_dirs = []
            for pat in zarr_patterns:
                res = subprocess.run(f"fileutil ls -d \"{pat}\" 2>/dev/null", shell=True, capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    zarr_dirs.extend([l.strip().rstrip("/") for l in res.stdout.strip().split("\n") if l.strip().endswith(".zarr") or ".zarr/" in l])
            unique_zarrs = sorted(list(set(zarr_dirs)))
            if unique_zarrs:
                cns_prefix = cns_dir.rstrip("/") + "/"
                missing_zarrs = []
                for zp in unique_zarrs:
                    rel = zp[len(cns_prefix):] if zp.startswith(cns_prefix) else os.path.basename(zp)
                    dest = os.path.join(local_dir, rel)
                    if not os.path.exists(os.path.join(dest, ".zmetadata")):
                        missing_zarrs.append((zp, dest))
                if missing_zarrs:
                    print(f"[CNS SYNC] Syncing {len(missing_zarrs)} tester.py consolidated Zarr stores from {cns_dir} to {local_dir}...")
                    def _copy_zarr(pair):
                        cns_zp, local_zp = pair
                        os.makedirs(os.path.dirname(local_zp), exist_ok=True)
                        subprocess.run(["fileutil", "cp", "-a", "-R", "-f", "-parallelism", "64", cns_zp, os.path.dirname(local_zp) + "/"], capture_output=True)
                    with ThreadPoolExecutor(max_workers=16) as ex:
                        list(ex.map(_copy_zarr, missing_zarrs))
    except KeyboardInterrupt:
        print("\n[INFO] CNS synchronization interrupted by user. Proceeding with local files available so far...")
    except Exception as e:
        print(f"[INFO] fileutil sync skipped or unavailable: {e}")


def _load_single_parquet(
    filepath: str,
    is_eval: bool,
    filter_configs: Optional[List[str]] = None,
    filter_basins: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """Reads and validates a single parquet shard with memory optimizations."""
    try:
        if not os.path.isfile(filepath) or os.path.getsize(filepath) < 100:
            return None

        # Apply pyarrow pushdown filter on Config ID for huge timeseries shards
        if not is_eval and filter_configs:
            try:
                df = pd.read_parquet(filepath, filters=[("Config ID", "in", filter_configs)])
            except Exception:
                try:
                    df = pd.read_parquet(filepath, filters=[("config_id", "in", filter_configs)])
                except Exception:
                    df = pd.read_parquet(filepath)
        else:
            df = pd.read_parquet(filepath)

        if df.empty:
            return None

        # Filter out non-eval files if searching for evaluation metrics
        if is_eval:
            eval_indicators = ["DA NSE", "da_nse", "Base NSE", "base_nse", "NSE Delta", "nse_delta"]
            if not any(k in df.columns for k in eval_indicators):
                return None
        else:
            ts_indicators = ["q_obs", "q_da", "q_base"]
            if not any(k in df.columns for k in ts_indicators):
                return None

        # Apply basin filter if specified
        if filter_basins and "Basin ID" in df.columns:
            df = df[df["Basin ID"].astype(str).isin(filter_basins)]
            if df.empty:
                return None
        elif filter_basins and "basin" in df.columns:
            df = df[df["basin"].astype(str).isin(filter_basins)]
            if df.empty:
                return None

        # Dtype optimizations: downcast float64 to float32
        float_cols = list(df.select_dtypes(include=["float64"]).columns)
        if float_cols:
            df[float_cols] = df[float_cols].astype(np.float32)

        return df
    except Exception:
        return None


def load_valid_parquets(
    data_dir: str,
    pattern: str,
    filter_configs: Optional[List[str]] = None,
    filter_basins: Optional[List[str] | set] = None,
    max_shards: Optional[int] = None,
    max_basins: Optional[int] = None,
    num_workers: int = 8,
) -> pd.DataFrame:
    """Discovers and concatenates non-empty parquet shards strictly within data_dir.

    Supports graceful early stopping via Ctrl+C / Jupyter Stop, max_shards limiting, and max_basins thresholding.
    """
    is_eval = "eval" in pattern.lower()

    if is_eval:
        glob_patterns = [
            "*detailed_basin_parameter_eval*.parquet",
            "eval_shard_*.parquet",
            "*detailed_eval*.parquet",
            "*eval*.parquet",
        ]
    else:
        glob_patterns = [
            "*timeseries_forecasts*.parquet",
            "ts_shard_*.parquet",
            "*timeseries*.parquet",
            "*ts*.parquet",
        ]

    # Search ONLY inside data_dir and its subdirectories
    search_dirs = [
        str(data_dir),
        os.path.join(str(data_dir), "detailed_eval" if is_eval else "timeseries_forecasts"),
        os.path.join(str(data_dir), "shard_*"),
    ]

    files = []
    for d in search_dirs:
        for gp in glob_patterns:
            matched = glob.glob(os.path.join(d, gp))
            if matched:
                files.extend(matched)

    unique_files = sorted(list(set(files)))
    if not unique_files:
        if is_eval:
            print(f"[INFO] No parquet files found for evaluation metrics. Attempting legacy CSV ingestion from {data_dir}...")
            return load_legacy_csv_metrics(data_dir, filter_basins=filter_basins)
        else:
            print(f"[INFO] No parquet files found for timeseries. Attempting legacy Zarr ingestion from {data_dir}...")
            return load_legacy_zarr_timeseries(data_dir, filter_configs=filter_configs, filter_basins=filter_basins)

    # If max_shards is specified, filter shard files
    if max_shards is not None:
        def _get_shard_num(fpath):
            m = re.search(r'shard_(\d+)', os.path.basename(fpath))
            return int(m.group(1)) if m else None
        shard_files = []
        for f in unique_files:
            s_num = _get_shard_num(f)
            if s_num is not None:
                if s_num < max_shards:
                    shard_files.append(f)
            else:
                shard_files.append(f)
        unique_files = shard_files[:max_shards] if len(shard_files) > max_shards else shard_files

    # For timeseries files, if filter_basins is provided, directly filter file list by basin name in filename
    if not is_eval and filter_basins:
        basin_set_lower = {str(b).lower() for b in filter_basins}
        matched_ts_files = []
        for f in unique_files:
            base_f = os.path.basename(f).lower()
            if any(b in base_f for b in basin_set_lower):
                matched_ts_files.append(f)
        if matched_ts_files:
            unique_files = matched_ts_files

    filter_basins_set = set(str(b) for b in filter_basins) if filter_basins else None

    valid_dfs = []
    accumulated_basins = set()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(_load_single_parquet, f, is_eval, filter_configs=filter_configs, filter_basins=filter_basins_set): f
                for f in unique_files
            }
            for future in as_completed(future_to_file):
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        valid_dfs.append(df)
                        if is_eval and "Basin ID" in df.columns:
                            accumulated_basins.update(df["Basin ID"].dropna().unique())
                            if max_basins and len(accumulated_basins) >= max_basins:
                                print(f"[INFO] Reached max_basins limit ({len(accumulated_basins)} catchments). Stopping loading gracefully.")
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
                except Exception:
                    pass
    except KeyboardInterrupt:
        print(f"\n[INFO] Graceful stop requested (KeyboardInterrupt)! Retaining {len(valid_dfs)} loaded parquet shard(s) ({len(accumulated_basins)} catchments). Proceeding with analysis on partial data...")

    if not valid_dfs:
        print(f"[WARNING] None of the found files were valid parquets for '{pattern}' in {data_dir}.")
        return pd.DataFrame()

    df_combined = pd.concat(valid_dfs, ignore_index=True)
    df_standard = _standardize_columns(df_combined)

    # Deduplicate records across shards
    if is_eval and "Basin ID" in df_standard.columns and "Config ID" in df_standard.columns and "Lead Time (Days)" in df_standard.columns:
        df_standard = df_standard.drop_duplicates(subset=["Basin ID", "Config ID", "Lead Time (Days)"])
    elif not is_eval and "Basin ID" in df_standard.columns and "Config ID" in df_standard.columns and "Lead Time (Days)" in df_standard.columns and "Valid Date" in df_standard.columns:
        df_standard = df_standard.drop_duplicates(subset=["Basin ID", "Config ID", "Lead Time (Days)", "Valid Date"])

    # If max_basins was requested, trim strictly to max_basins
    if is_eval and max_basins and "Basin ID" in df_standard.columns:
        top_basins = df_standard["Basin ID"].dropna().unique()[:max_basins]
        df_standard = df_standard[df_standard["Basin ID"].isin(top_basins)].reset_index(drop=True)

    return df_standard


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names across shards."""
    rename_dict = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ["lead_time", "lead_day", "lead_time_days", "lead", "lead time (days)"]:
            rename_dict[col] = "Lead Time (Days)"
        elif cl in ["basin", "basin_id", "gauge_id", "basin id"]:
            rename_dict[col] = "Basin ID"
        elif cl in ["config", "config_id", "configuration", "config id"]:
            rename_dict[col] = "Config ID"
        elif cl in ["date", "valid_date", "time", "valid date"]:
            rename_dict[col] = "Valid Date"
        elif cl in ["baseline nse", "base_nse"]:
            rename_dict[col] = "Base NSE"
        elif cl in ["baseline kge", "base_kge"]:
            rename_dict[col] = "Base KGE"
        elif cl in ["da nse", "da_nse"]:
            rename_dict[col] = "DA NSE"
        elif cl in ["da kge", "da_kge"]:
            rename_dict[col] = "DA KGE"
        elif cl in ["delta nse", "delta_nse", "nse_delta"]:
            rename_dict[col] = "NSE Delta"
        elif cl in ["delta kge", "delta_kge", "kge_delta"]:
            rename_dict[col] = "KGE Delta"

    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "Valid Date" in df.columns:
        df["Valid Date"] = pd.to_datetime(df["Valid Date"])

    if "Lead Time (Days)" in df.columns:
        df["Lead Time (Days)"] = df["Lead Time (Days)"].astype(int)

    if "NSE Delta" not in df.columns and {"DA NSE", "Base NSE"}.issubset(df.columns):
        df["NSE Delta"] = df["DA NSE"] - df["Base NSE"]

    if "KGE Delta" not in df.columns and {"DA KGE", "Base KGE"}.issubset(df.columns):
        df["KGE Delta"] = df["DA KGE"] - df["Base KGE"]

    for m in ("Pearson-r", "Alpha-NSE", "Beta-KGE", "Beta-NSE", "NSE (t+0 in-window)", "KGE (t+0 in-window)"):
        da_c, base_c, del_c = f"DA {m}", f"Base {m}", f"{m} Delta"
        if del_c not in df.columns and {da_c, base_c}.issubset(df.columns):
            df[del_c] = df[da_c] - df[base_c]

    return _ensure_skill_score_columns(df)


def _ensure_skill_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Computes standard Skill Score columns:
    - Unit-optimum correlation/efficiency metrics (NSE, KGE, Pearson-r, t+0 in-window):
      SS = 1 - (1 - DA) / (1 - Base) = (DA - Base) / (1 - Base)
    - Ratio/bias metrics with finite target M* (Alpha-NSE: M*=1, Beta-KGE: M*=1, Beta-NSE: M*=0):
      SS = 1 - |DA - M*| / |Base - M*|
    """
    for m in ("NSE", "KGE", "Pearson-r", "NSE (t+0 in-window)", "KGE (t+0 in-window)"):
        ss_col = f"{m} Skill Score" if "(t+0" not in m else m.replace(" (", " Skill Score (")
        da_c, base_c = f"DA {m}", f"Base {m}"
        if ss_col not in df.columns and {da_c, base_c}.issubset(df.columns):
            denom = 1.0 - df[base_c].astype(float)
            df[ss_col] = np.where(
                np.abs(denom) > 1e-9,
                1.0 - (1.0 - df[da_c].astype(float)) / denom,
                np.nan,
            )

    for m, opt in (("Alpha-NSE", 1.0), ("Beta-KGE", 1.0), ("Beta-NSE", 0.0)):
        ss_col = f"{m} Skill Score"
        da_c, base_c = f"DA {m}", f"Base {m}"
        if ss_col not in df.columns and {da_c, base_c}.issubset(df.columns):
            err_base = np.abs(df[base_c].astype(float) - opt)
            err_da = np.abs(df[da_c].astype(float) - opt)
            df[ss_col] = np.where(
                err_base > 1e-9,
                1.0 - err_da / err_base,
                np.nan,
            )
    return df


def parse_da_config_id(config_id: str) -> pd.Series:
    """Universal regex parser for hyperparameter components across all 3 DA modes:

    - Cell State DA: (c_fc, h_n, c_both)
    - Precipitation DA: (total_precipitation)
    - Embedding DA: (embedded_dynamics, embedded_statics, both, all)
    """
    cid = str(config_id)
    cid_lower = cid.lower()
    if any(p in cid_lower for p in ["per-basin", "oracle", "global best", "tuned per basin"]):
        return pd.Series({
            "Window (Days)": np.nan,
            "Learning Rate": np.nan,
            "Epochs": np.nan,
            "BG Weight": np.nan,
            "Stat Weight": np.nan,
            "Loss": np.nan,
            "Target": "Tuned Per Basin",
        })
    if cid_lower.startswith("ar1_") or "_postprocess_rho" in cid_lower:
        return pd.Series({
            "Window (Days)": np.nan,
            "Learning Rate": np.nan,
            "Epochs": np.nan,
            "BG Weight": np.nan,
            "Stat Weight": np.nan,
            "Loss": np.nan,
            "Target": "Post-processing (PP)",
        })
    if "baseline" in cid_lower:
        return pd.Series({
            "Window (Days)": np.nan,
            "Learning Rate": np.nan,
            "Epochs": np.nan,
            "BG Weight": np.nan,
            "Stat Weight": np.nan,
            "Loss": np.nan,
            "Target": "Open-Loop Baseline",
        })

    w_match = re.search(r"_w(\d+)", cid)
    lr_match = re.search(r"_(?:lr|lrd)([0-9.eE+-]+)", cid) or re.search(r"_lrs([0-9.eE+-]+)", cid)
    ep_match = re.search(r"_ep(\d+)", cid)
    bg_match = re.search(r"_(?:bg|bgd)([0-9.eE+-]+)", cid)
    stat_match = re.search(r"_(?:stat|bgs)([0-9.eE+-]+)", cid)

    w = int(w_match.group(1)) if w_match else np.nan
    lr = float(lr_match.group(1)) if lr_match else np.nan
    ep = int(ep_match.group(1)) if ep_match else np.nan
    bg = float(bg_match.group(1)) if bg_match else np.nan
    stat = float(stat_match.group(1)) if stat_match else np.nan

    if "_cmal_" in cid_lower:
        loss_fn = "CMAL"
    elif "_nse_" in cid_lower:
        loss_fn = "NSE"
    elif "_mse_" in cid_lower:
        loss_fn = "MSE"
    else:
        loss_fn = np.nan

    if "mf2lstm" in cid_lower:
        tgt = "MF2LSTM (Masked Streamflow)"
    elif (
        "embedded_all" in cid_lower
        or "_all_" in cid_lower
        or cid_lower.startswith("all_")
        or "triple" in cid_lower
        or "_t_w" in cid_lower
        or cid_lower.startswith("t_w")
        or "tr_w" in cid_lower
        or "_shf_w" in cid_lower
        or cid_lower.startswith("shf_w")
    ):
        tgt = "All Embeddings"
    elif (
        "dualdyn" in cid_lower
        or "_d_dual_" in cid_lower
        or "_d_w" in cid_lower
        or cid_lower.startswith("d_w")
        or "dd_w" in cid_lower
    ):
        tgt = "Dual Dynamic Embeddings"
    elif (
        "embedded_both" in cid_lower
        or "_both_" in cid_lower
        or cid_lower.startswith("both_")
        or "_hs_w" in cid_lower
        or cid_lower.startswith("hs_w")
        or "_sh_w" in cid_lower
        or cid_lower.startswith("sh_w")
    ):
        tgt = "Both Embeddings"
    elif "_fc_" in cid_lower or "forecast_embedding" in cid_lower:
        tgt = "Forecast Embeddings"
    elif "embedded_dynamics" in cid_lower or "_dynamics_" in cid_lower or cid_lower.startswith("dyn_") or "_dyn_" in cid_lower or "_h_w" in cid_lower or cid_lower.startswith("h_w"):
        tgt = "Dynamic Embeddings"
    elif (
        "embedded_statics" in cid_lower
        or "_statics_" in cid_lower
        or cid_lower.startswith("stat_")
        or "_stat_" in cid_lower
        or "_s_w" in cid_lower
        or cid_lower.startswith("s_w")
        or "sa_w" in cid_lower
    ):
        tgt = "Static Embeddings"
    elif "precip" in cid_lower:
        tgt = "Precipitation"
    elif "c_fc" in cid_lower or "c_n_forecast" in cid_lower:
        tgt = "c_n_forecast"
    elif "c_both" in cid_lower:
        tgt = "c_both"
    elif "h_keys" in cid_lower or "h_n" in cid_lower:
        tgt = "h_n"
    elif "embedded" in cid_lower:
        tgt = "Embeddings"
    elif re.search(r"_r[0-9][0-9.]*_", cid_lower):
        # Stage 51-style IDs: `_r<lr_stat/lr_dyn>_`; r0 = hindcast-only, r>0 = hindcast + static.
        ratio = float(re.search(r"_r([0-9][0-9.]*)_", cid_lower).group(1))
        tgt = "Dynamic Embeddings" if ratio == 0 else "Both Embeddings"
    else:
        parts = cid.split("_")
        tgt = "_".join(parts[5:]) if len(parts) > 5 else "State"

    if np.isnan(stat) and not np.isnan(bg) and tgt in ("Both Embeddings", "Static Embeddings", "All Embeddings"):
        stat = bg

    return pd.Series({
        "Window (Days)": w,
        "Learning Rate": lr,
        "Epochs": ep,
        "BG Weight": bg,
        "Stat Weight": stat,
        "Loss": loss_fn,
        "Target": tgt,
    })


def detect_lead_times(df: pd.DataFrame) -> List[int]:
    """Detects available leadtimes in dataset (supporting dynamic t+0 detection)."""
    if "Lead Time (Days)" not in df.columns or df.empty:
        return [1, 2, 3, 4, 5, 6, 7]
    leads = sorted([int(x) for x in df["Lead Time (Days)"].dropna().unique()])
    return leads


def load_basin_attributes(
    attr_zarr_path: Optional[str | Path] = None,
    requested_basins: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extracts physical and geospatial attributes from Caravans V2 zarr archive:

    Extracts: Area, Aridity Index, Snow Fraction, Elevation, Terrain Slope,
    Coordinates, Country.
    """
    candidates = [
        attr_zarr_path,
        Path("/usr/local/google/home/kruparell/Caravans_V2/attributes.zarr"),
        Path("/usr/local/google/home/kruparell/da-paper/tutorial/Caravan-zarr/attributes.zarr"),
    ]

    resolved_path = None
    for c in candidates:
        if c and Path(c).is_dir():
            resolved_path = Path(c)
            break

    if not resolved_path:
        print("[WARNING] attributes.zarr could not be located in candidate paths.")
        return pd.DataFrame()

    try:
        ds = xr.open_zarr(resolved_path, consolidated=True)
        avail_basins = [str(b) for b in ds["basin"].values]

        if requested_basins:
            req_set = set(str(b) for b in requested_basins)
            valid = [b for b in avail_basins if b in req_set]
            if not valid:
                lower_map = {b.lower(): b for b in avail_basins}
                valid = [lower_map[b.lower()] for b in requested_basins if b.lower() in lower_map]
            if valid:
                ds = ds.sel(basin=valid)
                avail_basins = [str(b) for b in ds["basin"].values]

        data = {"Basin ID": avail_basins}

        # Area
        if "area" in ds:
            data["area"] = ds["area"].values
        elif "area_shp" in ds:
            data["area"] = ds["area_shp"].values
        else:
            data["area"] = np.nan

        # Coordinates
        data["lat"] = ds["gauge_lat"].values if "gauge_lat" in ds else np.nan
        data["lon"] = ds["gauge_lon"].values if "gauge_lon" in ds else np.nan
        data["country"] = [str(x) for x in ds["country"].values] if "country" in ds else "Unknown"
        data["gauge_name"] = [str(x) for x in ds["gauge_name"].values] if "gauge_name" in ds else avail_basins

        # Mean Precipitation (mm/day)
        if "p_mean" in ds:
            data["p_mean"] = ds["p_mean"].values
        elif "precip_mean" in ds:
            data["p_mean"] = ds["precip_mean"].values
        else:
            data["p_mean"] = np.nan

        # Aridity Index (P / PET)
        if "unep_aridity_index" in ds:
            data["unep_aridity_index"] = ds["unep_aridity_index"].values
        elif "aridity" in ds:
            data["unep_aridity_index"] = ds["aridity"].values
        elif "p_mean" in ds and "pet_mean" in ds:
            p = ds["p_mean"].values
            pet = np.where(ds["pet_mean"].values > 0, ds["pet_mean"].values, np.nan)
            data["unep_aridity_index"] = p / pet
        else:
            data["unep_aridity_index"] = np.nan

        # Snow Fraction
        if "frac_snow" in ds:
            data["frac_snow"] = ds["frac_snow"].values
        elif "fraction_snow" in ds:
            data["frac_snow"] = ds["fraction_snow"].values
        elif "snw_pc_syr" in ds:
            data["frac_snow"] = ds["snw_pc_syr"].values / 100.0
        else:
            data["frac_snow"] = np.nan

        # Slope & Elevation
        if "slp_dg_sav" in ds:
            data["slp_dg_sav"] = ds["slp_dg_sav"].values
        elif "gradient_mean" in ds:
            data["slp_dg_sav"] = ds["gradient_mean"].values
        else:
            data["slp_dg_sav"] = np.nan

        if "ele_mt_sav" in ds:
            data["ele_mt_sav"] = ds["ele_mt_sav"].values
        elif "elv_mean" in ds:
            data["ele_mt_sav"] = ds["elv_mean"].values
        else:
            data["ele_mt_sav"] = np.nan

        return pd.DataFrame(data)

    except Exception as e:
        print(f"[ERROR] Failed to extract physical attributes from zarr: {e}")
        return pd.DataFrame()


def load_legacy_csv_metrics(
    data_dir: str,
    filter_basins: Optional[List[str] | set] = None,
) -> pd.DataFrame:
    """Ingests evaluation metrics from legacy per-config test_metrics*.csv files."""
    csv_files = glob.glob(os.path.join(str(data_dir), "**/test_metrics*.csv"), recursive=True)
    if not csv_files:
        print(f"[WARNING] No legacy test_metrics*.csv files found in {data_dir}.")
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        parts = Path(f).parts
        cfg_name = None
        for i, p in enumerate(parts):
            if p == "test" and i > 0:
                candidate = parts[i - 1]
                if re.match(r"^shard_\d+$", candidate) and i >= 2:
                    cfg_name = parts[i - 2]
                else:
                    cfg_name = candidate
                break
        if not cfg_name:
            cfg_name = os.path.basename(os.path.dirname(f))

        # Standardize legacy config IDs for mode validation
        cfg_lower = cfg_name.lower()
        if "baseline" in cfg_lower:
            cfg_id = "Baseline"
        elif "mf2lstm" in cfg_lower:
            cfg_id = cfg_name
        elif any(k in cfg_lower for k in ("both_", "dyn_", "stat_", "all_", "triple_", "dualdyn_", "_fc_")) and not cfg_lower.startswith("embedded_"):
            cfg_id = f"embedded_{cfg_name}"
        else:
            cfg_id = cfg_name

        try:
            df = pd.read_csv(f)
            df["Config ID"] = cfg_id
            df["_raw_cfg_folder"] = cfg_name
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read CSV {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    lead_cols = [c for c in combined.columns if "_lead" in c and "mean" not in c]
    if lead_cols and "Lead Time (Days)" not in combined.columns:
        combined["basin"] = combined["basin"].astype(str)
        combined["Config ID"] = combined["Config ID"].astype(str)
        col_nse0 = "NSE_lead0_in_window"
        col_kge0 = "KGE_lead0_in_window"
        nse0_vals = combined[col_nse0] if col_nse0 in combined.columns else np.nan
        kge0_vals = combined[col_kge0] if col_kge0 in combined.columns else np.nan
        lead_dfs = []
        for lead in range(1, 8):
            col_nse = f"NSE_lead{lead}"
            col_kge = f"KGE_lead{lead}"
            col_r = f"Pearson-r_lead{lead}"
            col_alpha = f"Alpha-NSE_lead{lead}"
            col_bkge = f"Beta-KGE_lead{lead}"
            col_bnse = f"Beta-NSE_lead{lead}"
            sub = pd.DataFrame({
                "Basin ID": combined["basin"],
                "Config ID": combined["Config ID"],
                "Lead Time (Days)": lead,
                "DA NSE": combined[col_nse] if col_nse in combined.columns else np.nan,
                "DA KGE": combined[col_kge] if col_kge in combined.columns else np.nan,
                "DA Pearson-r": combined[col_r] if col_r in combined.columns else np.nan,
                "DA Alpha-NSE": combined[col_alpha] if col_alpha in combined.columns else np.nan,
                "DA Beta-KGE": combined[col_bkge] if col_bkge in combined.columns else np.nan,
                "DA Beta-NSE": combined[col_bnse] if col_bnse in combined.columns else np.nan,
                "DA NSE (t+0 in-window)": nse0_vals,
                "DA KGE (t+0 in-window)": kge0_vals,
            })
            lead_dfs.append(sub)
        df_long = pd.concat(lead_dfs, ignore_index=True)
    else:
        df_long = combined

    df_long = _standardize_columns(df_long)
    if {"Basin ID", "Config ID", "Lead Time (Days)"}.issubset(df_long.columns):
        df_long = df_long.drop_duplicates(subset=["Basin ID", "Config ID", "Lead Time (Days)"])

    # Compute base values & deltas
    base_df = df_long[df_long["Config ID"] == "Baseline"].copy()
    metric_names = ("NSE", "KGE", "Pearson-r", "Alpha-NSE", "Beta-KGE", "Beta-NSE",
                    "NSE (t+0 in-window)", "KGE (t+0 in-window)")
    if not base_df.empty:
        avail_m = [m for m in metric_names if f"DA {m}" in base_df.columns]
        rename_map = {f"DA {m}": f"Base {m}" for m in avail_m}
        base_df = base_df.rename(columns=rename_map)
        keep_cols = ["Basin ID", "Lead Time (Days)"] + [f"Base {m}" for m in avail_m]
        base_df = base_df[keep_cols].drop_duplicates(subset=["Basin ID", "Lead Time (Days)"])
        # Drop any pre-existing Base columns before merging
        drop_pre = [f"Base {m}" for m in avail_m if f"Base {m}" in df_long.columns]
        df_long = df_long[df_long["Config ID"] != "Baseline"].drop(columns=drop_pre, errors="ignore").merge(
            base_df, on=["Basin ID", "Lead Time (Days)"], how="left"
        )
        for m in avail_m:
            df_long[f"{m} Delta"] = df_long[f"DA {m}"] - df_long[f"Base {m}"]
    else:
        print("[WARNING] No 'Baseline' configuration found in local CSVs; defaulting Base NSE to 0.0 for provisional delta ranking.")
        df_long["Base NSE"] = 0.0
        df_long["Base KGE"] = 0.0
        df_long["NSE Delta"] = df_long["DA NSE"]
        df_long["KGE Delta"] = df_long["DA KGE"]

    df_long = _ensure_skill_score_columns(df_long)

    if filter_basins:
        b_set = {str(b).lower() for b in filter_basins}
        df_long = df_long[df_long["Basin ID"].astype(str).str.lower().isin(b_set)]

    print(f"[LEGACY CSV] Ingested {len(df_long):,} evaluation metrics across {df_long['Config ID'].nunique()} configs.")
    return df_long


def load_legacy_zarr_timeseries(
    data_dir: str,
    filter_configs: Optional[List[str]] = None,
    filter_basins: Optional[List[str] | set] = None,
) -> pd.DataFrame:
    """Extracts forecast timeseries from legacy per-config test_results*.zarr stores."""
    zarr_files = glob.glob(os.path.join(str(data_dir), "**/test_results*.zarr"), recursive=True)
    if not zarr_files:
        print(f"[WARNING] No legacy test_results*.zarr stores found in {data_dir}.")
        return pd.DataFrame()

    filter_cfg_set = set(filter_configs) if filter_configs else None
    filter_basin_set = {str(b).lower() for b in filter_basins} if filter_basins else None

    dfs = []
    for z_path in zarr_files:
        parts = Path(z_path).parts
        cfg_name = None
        for i, p in enumerate(parts):
            if p == "test" and i > 0:
                candidate = parts[i - 1]
                if re.match(r"^shard_\d+$", candidate) and i >= 2:
                    cfg_name = parts[i - 2]
                else:
                    cfg_name = candidate
                break
        if not cfg_name:
            cfg_name = os.path.basename(os.path.dirname(z_path))

        cfg_lower = cfg_name.lower()
        if "baseline" in cfg_lower:
            cfg_id = "Baseline"
        elif "mf2lstm" in cfg_lower:
            cfg_id = cfg_name
        elif any(k in cfg_lower for k in ("both_", "dyn_", "stat_", "all_", "triple_", "dualdyn_", "_fc_")) and not cfg_lower.startswith("embedded_"):
            cfg_id = f"embedded_{cfg_name}"
        else:
            cfg_id = cfg_name

        if filter_cfg_set:
            match_found = False
            # 1. Exact match check against raw names
            if cfg_id in filter_cfg_set:
                match_found = True
            elif cfg_name in filter_cfg_set:
                cfg_id = cfg_name
                match_found = True
            else:
                # 2. Check normalized form stripped of default _stat1e-06
                cfg_id_stripped = re.sub(r'_stat1e-0?6$', '', cfg_id)
                cfg_name_stripped = re.sub(r'_stat1e-0?6$', '', cfg_name)
                if cfg_id_stripped in filter_cfg_set:
                    cfg_id = cfg_id_stripped
                    match_found = True
                elif cfg_name_stripped in filter_cfg_set:
                    cfg_id = cfg_name_stripped
                    match_found = True
                else:
                    # Also check if any entry in filter_cfg_set matches normalized (ignoring embedded_ prefix)
                    for f_cfg in filter_cfg_set:
                        f_norm = re.sub(r'_stat1e-0?6$', '', f_cfg).replace("embedded_", "")
                        if f_norm and f_norm == cfg_id_stripped.replace("embedded_", ""):
                            cfg_id = f_cfg
                            match_found = True
                            break

            # 3. Token fallback check if not yet matched
            if not match_found:
                for tgt in filter_cfg_set:
                    if "Per-Basin" in tgt or "Best" in tgt or "Oracle" in tgt:
                        match_found = True
                        break
                    tgt_tokens = set(re.split(r'[^a-zA-Z0-9]', tgt.lower()))
                    cfg_tokens = set(re.split(r'[^a-zA-Z0-9]', cfg_name.lower()))
                    if 'baseline' in tgt_tokens and 'baseline' in cfg_tokens:
                        match_found = True
                        break
                    if ('both' in tgt_tokens and 'both' in cfg_tokens) or ('cell' in tgt_tokens and 'cell' in cfg_tokens) or ('dyn' in tgt_tokens and 'dyn' in cfg_tokens) or ('stat' in tgt_tokens and 'stat' in cfg_tokens) or ('all' in tgt_tokens and 'all' in cfg_tokens):
                        window_tokens = {'w365', 'w180', 'w90', 'w60', 'w30', 'w14', 'w7', 'w3'}
                        if tgt_tokens & cfg_tokens & window_tokens:
                            match_found = True
                            break
            if not match_found:
                continue

        try:
            ds = xr.open_zarr(z_path, consolidated=True)
            obs_var = "streamflow_obs" if "streamflow_obs" in ds else "q_obs"
            sim_var = "streamflow_sim" if "streamflow_sim" in ds else "q_sim"

            if obs_var not in ds or sim_var not in ds:
                continue

            avail_basins = [str(b) for b in ds["basin"].values]
            if filter_basin_set:
                valid_b = [b for b in avail_basins if str(b).lower() in filter_basin_set or str(b) in filter_basin_set]
                if not valid_b:
                    continue
                ds = ds.sel(basin=valid_b)
                avail_basins = [str(b) for b in ds["basin"].values]

            dates = pd.to_datetime(ds["date"].values)
            ts_coords = [int(x) for x in ds["time_step"].values] if "time_step" in ds.coords or "time_step" in ds else list(range(1, 8))
            sim_vals = ds[sim_var].values  # (basin, freq, date, time_step[, samples])
            obs_vals = ds[obs_var].values

            # Reduce ensemble samples with the MEAN (matches mixture_mean point-estimate metrics).
            if sim_vals.ndim == 5:
                sim_vals = reduce_samples(sim_vals[:, 0])  # -> (basin, date, time_step)
            elif sim_vals.ndim == 4:
                sim_vals = sim_vals[:, 0, :, :]

            if obs_vals.ndim == 5:
                obs_vals = obs_vals[:, 0, :, :, 0]
            elif obs_vals.ndim == 4:
                obs_vals = obs_vals[:, 0, :, :]
            elif obs_vals.ndim == 3:
                obs_vals = obs_vals[:, 0, :, np.newaxis]

            # Vectorised conversion (one frame per lead instead of one per basin x lead).
            block = block_to_frame(avail_basins, dates.values, ts_coords, sim_vals, obs_vals, cfg_id)
            if not block.empty:
                dfs.append(block)
        except Exception as e:
            print(f"[WARNING] Could not load legacy zarr {z_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Attach baseline timeseries as q_base if available
    base_ts = combined[combined["Config ID"] == "Baseline"].copy()
    if not base_ts.empty:
        base_ts = base_ts.rename(columns={"q_da": "q_base"})
        base_ts = base_ts[["Valid Date", "Basin ID", "Lead Time (Days)", "q_base"]].drop_duplicates()
        da_ts = combined[combined["Config ID"] != "Baseline"]
        combined = da_ts.merge(base_ts, on=["Valid Date", "Basin ID", "Lead Time (Days)"], how="left")
    else:
        combined["q_base"] = combined["q_obs"]

    print(f"[LEGACY ZARR] Loaded {len(combined):,} daily forecast records across {combined['Config ID'].nunique()} models.")
    return _standardize_columns(combined)
