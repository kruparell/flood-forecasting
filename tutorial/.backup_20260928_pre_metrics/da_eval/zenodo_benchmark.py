"""Benchmark comparison utilities against the 2024 Zenodo Dual-LSTM global paper baseline.

Provides dual-key crosswalk (direct GRDC ID + Caravans V2 nat_id matching), ingestion of per-gauge
CSV hydrograph metrics, multi-leadtime benchmark comparison matrices, and diagnostic plot suites.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from da_eval.benchmarks import get_global_best_config
from da_eval.ingestion import detect_lead_times

# Module-level cache for GRDC crosswalk
_GRDC_CROSSWALK_CACHE: Dict[str, Dict[str, Tuple[str, str]]] = {}


def build_grdc_crosswalk_map(
    attr_zarr_path: Optional[str | Path] = "/usr/local/google/home/kruparell/Caravans_V2/attributes.zarr",
) -> Dict[str, Tuple[str, str]]:
    """Builds a mapping from Caravans/DA Basin ID -> (GRDC_ID, match_type).

    Supports:
    1. Direct match: 'GRDC_XXXXXXX' -> ('GRDC_XXXXXXX', 'Direct GRDC')
    2. National ID crosswalk via attributes.zarr['nat_id']:
       'camels_01031500', 'camelsbr_15800000', 'hysets_05280000', etc. -> ('GRDC_XXXXXXX', 'Crosswalk (nat_id)')
    """
    cache_key = str(attr_zarr_path)
    if cache_key in _GRDC_CROSSWALK_CACHE:
        return _GRDC_CROSSWALK_CACHE[cache_key]

    crosswalk: Dict[str, Tuple[str, str]] = {}

    if attr_zarr_path and os.path.exists(str(attr_zarr_path)):
        try:
            ds = xr.open_zarr(str(attr_zarr_path), consolidated=True)
            if "nat_id" in ds:
                df_attr = ds[["nat_id"]].to_dataframe()

                # 1. Direct GRDC basins and build nat_id -> GRDC_ID lookup
                nid_to_grdc: Dict[str, str] = {}
                for idx, row in df_attr.iterrows():
                    idx_str = str(idx)
                    if idx_str.startswith("GRDC_"):
                        crosswalk[idx_str] = (idx_str, "Direct GRDC")
                        crosswalk[idx_str.lower()] = (idx_str, "Direct GRDC")
                        nid = str(row["nat_id"]).strip()
                        if nid:
                            nid_to_grdc[nid] = idx_str
                            nid_to_grdc[nid.lstrip("0")] = idx_str

                # 2. Cross-match non-GRDC basins (camels*, hysets*, lamah*, etc.) via nat_id
                for idx in df_attr.index:
                    idx_str = str(idx)
                    if not idx_str.startswith("GRDC_") and "_" in idx_str:
                        nid_part = idx_str.split("_", 1)[1].strip()
                        grdc_match = nid_to_grdc.get(nid_part) or nid_to_grdc.get(nid_part.lstrip("0"))
                        if grdc_match:
                            crosswalk[idx_str] = (grdc_match, "Crosswalk (nat_id)")
                            crosswalk[idx_str.lower()] = (grdc_match, "Crosswalk (nat_id)")
        except Exception as e:
            print(f"[WARNING] Could not build nat_id crosswalk from {attr_zarr_path}: {e}")

    _GRDC_CROSSWALK_CACHE[cache_key] = crosswalk
    return crosswalk


def _read_single_zenodo_csv(
    args: Tuple[str, str, str, str]
) -> Optional[pd.DataFrame]:
    """Reads a single Zenodo gauge CSV and formats rows across lead times."""
    basin_id, grdc_id, match_type, csv_path = args
    try:
        df_csv = pd.read_csv(csv_path, index_col=0)
        if df_csv.empty:
            return None

        # Columns in Zenodo CSV are lead times '0', '1', ..., '7'
        lead_cols = [c for c in df_csv.columns if str(c).strip().isdigit()]
        if not lead_cols:
            return None

        records = []
        for l_col in lead_cols:
            lead_int = int(str(l_col).strip())
            row_dict = {
                "Basin ID": basin_id,
                "GRDC ID": grdc_id,
                "Match Type": match_type,
                "Lead Time (Days)": lead_int,
            }
            for metric_name in ["NSE", "KGE", "log-NSE", "RMSE", "MSE", "Pearson-r", "Peak-MAPE", "FLV", "FHV", "FMS"]:
                if metric_name in df_csv.index:
                    val = df_csv.loc[metric_name, l_col]
                    row_dict[f"Zenodo {metric_name}"] = float(val) if pd.notna(val) else np.nan
            records.append(row_dict)

        return pd.DataFrame(records)
    except Exception:
        return None


def load_zenodo_gauge_metrics(
    zenodo_dir: str = "/usr/local/google/home/kruparell/zenodo_2024_paper/metrics/hydrograph_metrics/per_gauge/google/2014/dual_lstm/full_run",
    requested_basins: Optional[List[str] | set] = None,
    attr_zarr_path: Optional[str | Path] = "/usr/local/google/home/kruparell/Caravans_V2/attributes.zarr",
) -> pd.DataFrame:
    """Loads Zenodo 2024 Dual-LSTM hydrograph metrics for all overlapping basins.

    Resolves both direct GRDC_* IDs and CAMELS/HYSETS/Caravans IDs via nat_id crosswalk.
    """
    if not os.path.exists(zenodo_dir):
        print(f"[WARNING] Zenodo metrics directory not found: {zenodo_dir}")
        return pd.DataFrame()

    crosswalk = build_grdc_crosswalk_map(attr_zarr_path)

    # Index available CSV files in zenodo_dir
    available_csvs = {
        f.replace(".csv", "").upper(): os.path.join(zenodo_dir, f)
        for f in os.listdir(zenodo_dir)
        if f.endswith(".csv")
    }

    tasks = []
    if requested_basins is not None:
        for b in requested_basins:
            b_str = str(b)
            # 1. Check direct GRDC match
            if b_str.upper() in available_csvs:
                tasks.append((b_str, b_str.upper(), "Direct GRDC", available_csvs[b_str.upper()]))
            # 2. Check crosswalk map
            elif b_str in crosswalk or b_str.lower() in crosswalk:
                grdc_id, m_type = crosswalk.get(b_str) or crosswalk.get(b_str.lower())
                if grdc_id.upper() in available_csvs:
                    tasks.append((b_str, grdc_id.upper(), m_type, available_csvs[grdc_id.upper()]))
    else:
        # Load all available Zenodo files
        for grdc_id, fpath in available_csvs.items():
            tasks.append((grdc_id, grdc_id, "Direct GRDC", fpath))

    if not tasks:
        print("[WARNING] No overlapping gauges found between active session basins and Zenodo directory.")
        return pd.DataFrame()

    dfs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(_read_single_zenodo_csv, tasks):
            if res is not None and not res.empty:
                dfs.append(res)

    if not dfs:
        return pd.DataFrame()

    df_zenodo = pd.concat(dfs, ignore_index=True)
    return df_zenodo


def load_pretrained_metrics(
    metrics_path: Optional[str | Path] = None,
    requested_basins: Optional[List[str] | set] = None,
) -> pd.DataFrame:
    """Loads precalculated benchmark metrics from pretrained model test_metrics.csv / metrics.csv.

    Resolves both direct file paths, directory paths, and standard fallback locations:
    - /usr/local/google/home/kruparell/openhydronets_next/pretrained-models/google-floodhub-settings-55-epochs/test/test_metrics.csv
    - /usr/local/google/home/kruparell/openhydronets_next/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs/test/model_epoch085/test_metrics.csv
    - /cns/jn-d/home/floods/hydro_model/work/kruparell/pretrained-models/google-floodhub-settings-55-epochs/test/test_metrics.csv

    Returns:
        pd.DataFrame with columns ['Basin ID', 'Pretrained NSE', 'Pretrained KGE', ...]
    """
    default_candidates = [
        Path("/usr/local/google/home/kruparell/openhydronets_next/pretrained-models/google-floodhub-settings-55-epochs/test/test_metrics.csv"),
        Path("/usr/local/google/home/kruparell/openhydronets_next/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs/test/model_epoch085/test_metrics.csv"),
        Path("/cns/jn-d/home/floods/hydro_model/work/kruparell/pretrained-models/google-floodhub-settings-55-epochs/test/test_metrics.csv"),
    ]
    resolved_path = None
    if metrics_path is not None:
        p = Path(metrics_path)
        if p.is_file():
            resolved_path = p
        elif p.is_dir():
            for sub in [
                p / "test" / "test_metrics.csv",
                p / "test_metrics.csv",
                p / "metrics.csv",
            ]:
                if sub.is_file():
                    resolved_path = sub
                    break
            if resolved_path is None:
                found = list(p.glob("**/test_metrics.csv")) + list(p.glob("**/metrics.csv"))
                if found:
                    resolved_path = found[0]
    else:
        for cand in default_candidates:
            if cand.is_file():
                resolved_path = cand
                break

    if resolved_path is None or not resolved_path.exists():
        print(f"[WARNING] Pretrained metrics file not found: {metrics_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(resolved_path)
        basin_col = next((c for c in df.columns if c.lower() in ["basin", "basin_id", "basin id"]), None)
        if not basin_col:
            print(f"[WARNING] No basin identifier column found in {resolved_path}")
            return pd.DataFrame()

        df = df.rename(columns={basin_col: "Basin ID"})
        df["Basin ID"] = df["Basin ID"].astype(str)

        rename_map = {}
        for col in ["NSE", "KGE", "log-NSE", "RMSE", "MSE", "Pearson-r", "Alpha", "Beta", "Peak-MAPE", "FLV", "FHV", "FMS"]:
            if col in df.columns:
                rename_map[col] = f"Pretrained {col}"
            elif col.lower() in [c.lower() for c in df.columns]:
                actual = next(c for c in df.columns if c.lower() == col.lower())
                rename_map[actual] = f"Pretrained {col}"

        df = df.rename(columns=rename_map)

        if requested_basins is not None:
            req_set = set(str(b) for b in requested_basins)
            df_filtered = df[df["Basin ID"].isin(req_set)]
            if df_filtered.empty:
                req_lower = {b.lower(): b for b in req_set}
                df["Basin ID"] = df["Basin ID"].apply(lambda x: req_lower.get(x.lower(), x))
                df = df[df["Basin ID"].isin(req_set)]
            else:
                df = df_filtered

        return df
    except Exception as e:
        print(f"[WARNING] Failed to load pretrained metrics from {resolved_path}: {e}")
        return pd.DataFrame()


def build_zenodo_comparison_table(
    df_eval: pd.DataFrame,
    df_zenodo: pd.DataFrame,
    lead_times: Optional[List[int]] = None,
    metric: str = "NSE",
    df_pretrained: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Constructs a multi-leadtime benchmark comparison table strictly on overlapping catchments.

    Compares:
    1. Zenodo 2024 Dual-LSTM Baseline
    2. Pretrained Flood Hub Model (metrics.csv) [Optional]
    3. Baseline Open-Loop (Unassimilated OpenHydroNets)
    4. Global Best DA Configuration
    5. Per-Basin Best DA (Oracle)
    """
    if df_eval.empty or df_zenodo.empty:
        return pd.DataFrame()

    if lead_times is None:
        eval_leads = detect_lead_times(df_eval)
        zen_leads = sorted(df_zenodo["Lead Time (Days)"].dropna().unique().astype(int).tolist())
        lead_times = sorted(list(set(eval_leads).intersection(set(zen_leads))))
        if not lead_times:
            lead_times = [1, 2, 3, 4, 5, 6, 7]

    zen_col = f"Zenodo {metric}"
    da_col = f"DA {metric}" if f"DA {metric}" in df_eval.columns else "DA NSE"
    base_col = f"Base {metric}" if f"Base {metric}" in df_eval.columns else "Base NSE"

    # Filter strictly to overlapping basins
    overlap_basins = set(df_zenodo["Basin ID"].unique()).intersection(set(df_eval["Basin ID"].unique()))
    if not overlap_basins:
        return pd.DataFrame()

    sub_eval = df_eval[df_eval["Basin ID"].isin(overlap_basins)].copy()
    sub_zen = df_zenodo[df_zenodo["Basin ID"].isin(overlap_basins)].copy()
    sub_pt = (
        df_pretrained[df_pretrained["Basin ID"].isin(overlap_basins)].copy()
        if df_pretrained is not None and not df_pretrained.empty
        else pd.DataFrame()
    )

    # Identify Global Best Config
    ref_lead = 1 if 1 in lead_times else lead_times[0]
    best_cfg = get_global_best_config(sub_eval, lead_time=ref_lead)

    model_rows = [
        ("Zenodo 2024 Dual-LSTM (Paper Baseline)", "ZENODO"),
    ]
    if not sub_pt.empty:
        model_rows.append(("Pretrained Flood Hub Model (metrics.csv)", "PRETRAINED"))
    model_rows.extend([
        ("OpenHydroNets Baseline (Open-Loop)", "BASE"),
        (f"Global Best DA ({best_cfg})", best_cfg),
        ("Per-Basin Best DA (Oracle)", "Per-Basin Best DA"),
    ])

    rows = []
    for label, cfg_key in model_rows:
        num_gauges = len(overlap_basins)
        if cfg_key == "PRETRAINED":
            pt_col = f"Pretrained {metric}"
            if pt_col in sub_pt.columns:
                num_gauges = len(sub_pt.dropna(subset=[pt_col])["Basin ID"].unique())

        row_data = {"Benchmark Model": label, "Overlapping Gauges": num_gauges}

        for lt in lead_times:
            lt_tag = f"t+{lt}"
            zen_lt = sub_zen[sub_zen["Lead Time (Days)"] == lt].set_index("Basin ID")[zen_col]

            if cfg_key == "ZENODO":
                med_val = zen_lt.median()
                row_data[f"Median {metric} ({lt_tag})"] = med_val
                row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = 0.0
                row_data[f"Win Rate vs Zenodo ({lt_tag})"] = np.nan
            elif cfg_key == "PRETRAINED":
                pt_col = f"Pretrained {metric}"
                if pt_col in sub_pt.columns:
                    if "Lead Time (Days)" in sub_pt.columns:
                        pt_sub_lt = sub_pt[sub_pt["Lead Time (Days)"] == lt]
                    elif lt == 1:
                        pt_sub_lt = sub_pt
                    else:
                        pt_sub_lt = pd.DataFrame()

                    if not pt_sub_lt.empty:
                        pt_lt = pt_sub_lt.drop_duplicates("Basin ID").set_index("Basin ID")[pt_col].dropna()
                        common = pt_lt.index.intersection(zen_lt.index)
                        if len(common) > 0:
                            diff = pt_lt.loc[common] - zen_lt.loc[common]
                            row_data[f"Median {metric} ({lt_tag})"] = pt_lt.loc[common].median()
                            row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = diff.median()
                            row_data[f"Win Rate vs Zenodo ({lt_tag})"] = (diff > 0).mean() * 100.0
                        else:
                            row_data[f"Median {metric} ({lt_tag})"] = np.nan
                            row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = np.nan
                            row_data[f"Win Rate vs Zenodo ({lt_tag})"] = np.nan
                    else:
                        row_data[f"Median {metric} ({lt_tag})"] = np.nan
                        row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = np.nan
                        row_data[f"Win Rate vs Zenodo ({lt_tag})"] = np.nan
                else:
                    row_data[f"Median {metric} ({lt_tag})"] = np.nan
                    row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = np.nan
                    row_data[f"Win Rate vs Zenodo ({lt_tag})"] = np.nan
            elif cfg_key == "BASE":
                eval_lt = sub_eval[sub_eval["Lead Time (Days)"] == lt].drop_duplicates("Basin ID").set_index("Basin ID")[base_col]
                common = eval_lt.index.intersection(zen_lt.index)
                diff = eval_lt.loc[common] - zen_lt.loc[common]
                row_data[f"Median {metric} ({lt_tag})"] = eval_lt.loc[common].median()
                row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = diff.median()
                row_data[f"Win Rate vs Zenodo ({lt_tag})"] = (diff > 0).mean() * 100.0
            else:
                eval_lt = sub_eval[(sub_eval["Config ID"] == cfg_key) & (sub_eval["Lead Time (Days)"] == lt)].drop_duplicates("Basin ID").set_index("Basin ID")[da_col]
                common = eval_lt.index.intersection(zen_lt.index)
                if len(common) > 0:
                    diff = eval_lt.loc[common] - zen_lt.loc[common]
                    row_data[f"Median {metric} ({lt_tag})"] = eval_lt.loc[common].median()
                    row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = diff.median()
                    row_data[f"Win Rate vs Zenodo ({lt_tag})"] = (diff > 0).mean() * 100.0
                else:
                    row_data[f"Median {metric} ({lt_tag})"] = np.nan
                    row_data[f"Δ{metric} vs Zenodo ({lt_tag})"] = np.nan
                    row_data[f"Win Rate vs Zenodo ({lt_tag})"] = np.nan

        rows.append(row_data)

    return pd.DataFrame(rows)


def style_zenodo_comparison_table(df_table: pd.DataFrame, metric: str = "NSE"):
    """Applies publication styling to the Zenodo benchmark comparison table."""
    if df_table.empty:
        return df_table

    metric_cols = [c for c in df_table.columns if c.startswith(f"Median {metric}")]
    delta_cols = [c for c in df_table.columns if c.startswith(f"Δ{metric}")]
    win_cols = [c for c in df_table.columns if c.startswith("Win Rate")]

    format_dict = {"Overlapping Gauges": "{:,}"}
    for c in metric_cols:
        format_dict[c] = "{:.3f}"
    for c in delta_cols:
        format_dict[c] = "{:+.3f}"
    for c in win_cols:
        format_dict[c] = "{:.1f}%"

    styler = (
        df_table.style.format(format_dict, na_rep="-")
        .background_gradient(subset=metric_cols, cmap="YlGnBu", vmin=0.3, vmax=0.85)
        .background_gradient(subset=delta_cols, cmap="RdYlGn", vmin=-0.15, vmax=0.15)
        .set_caption(f"Multi-Horizon Benchmark Comparison vs. Zenodo 2024 Dual-LSTM Baseline ({metric})")
    )
    return styler


def build_zenodo_per_gauge_table(
    df_eval: pd.DataFrame,
    df_zenodo: pd.DataFrame,
    lead_time: int = 1,
    metric: str = "NSE",
    top_n: int = 25,
    sort_by: str = "DA vs Zenodo",
    df_pretrained: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Constructs a per-gauge side-by-side comparison table at a specified lead time."""
    if df_eval.empty or df_zenodo.empty:
        return pd.DataFrame()

    zen_col = f"Zenodo {metric}"
    da_col = f"DA {metric}" if f"DA {metric}" in df_eval.columns else "DA NSE"
    base_col = f"Base {metric}" if f"Base {metric}" in df_eval.columns else "Base NSE"

    best_cfg = get_global_best_config(df_eval, lead_time=lead_time)

    zen_sub = df_zenodo[df_zenodo["Lead Time (Days)"] == lead_time][["Basin ID", "GRDC ID", "Match Type", zen_col]].drop_duplicates("Basin ID")
    base_sub = df_eval[df_eval["Lead Time (Days)"] == lead_time][["Basin ID", base_col]].drop_duplicates("Basin ID")
    global_sub = df_eval[(df_eval["Config ID"] == best_cfg) & (df_eval["Lead Time (Days)"] == lead_time)][["Basin ID", da_col]].drop_duplicates("Basin ID").rename(columns={da_col: f"Global Best DA {metric}"})
    oracle_sub = df_eval[(df_eval["Config ID"] == "Per-Basin Best DA") & (df_eval["Lead Time (Days)"] == lead_time)][["Basin ID", da_col]].drop_duplicates("Basin ID").rename(columns={da_col: f"Per-Basin Best DA {metric}"})

    merged = zen_sub.merge(base_sub, on="Basin ID", how="inner")
    if df_pretrained is not None and not df_pretrained.empty:
        pt_col = f"Pretrained {metric}"
        if pt_col in df_pretrained.columns:
            if "Lead Time (Days)" in df_pretrained.columns:
                pt_sub = df_pretrained[df_pretrained["Lead Time (Days)"] == lead_time][["Basin ID", pt_col]].drop_duplicates("Basin ID")
            else:
                pt_sub = df_pretrained[["Basin ID", pt_col]].drop_duplicates("Basin ID")
            merged = merged.merge(pt_sub, on="Basin ID", how="left")
            merged = merged.rename(columns={pt_col: f"Pretrained FloodHub {metric}"})

    merged = merged.merge(global_sub, on="Basin ID", how="left")
    merged = merged.merge(oracle_sub, on="Basin ID", how="left")

    merged = merged.rename(columns={
        base_col: f"Open-Loop Base {metric}",
        zen_col: f"Zenodo 2024 Dual-LSTM {metric}",
    })

    da_ref = f"Per-Basin Best DA {metric}" if f"Per-Basin Best DA {metric}" in merged.columns else f"Global Best DA {metric}"
    merged[f"Δ{metric} (DA vs Base)"] = merged[da_ref] - merged[f"Open-Loop Base {metric}"]
    merged[f"Δ{metric} (DA vs Zenodo)"] = merged[da_ref] - merged[f"Zenodo 2024 Dual-LSTM {metric}"]
    if f"Pretrained FloodHub {metric}" in merged.columns:
        merged[f"Δ{metric} (DA vs Pretrained)"] = merged[da_ref] - merged[f"Pretrained FloodHub {metric}"]

    if sort_by == "DA vs Zenodo":
        merged = merged.sort_values(f"Δ{metric} (DA vs Zenodo)", ascending=False)
    elif sort_by == "DA vs Base":
        merged = merged.sort_values(f"Δ{metric} (DA vs Base)", ascending=False)
    elif sort_by == "DA vs Pretrained" and f"Δ{metric} (DA vs Pretrained)" in merged.columns:
        merged = merged.sort_values(f"Δ{metric} (DA vs Pretrained)", ascending=False)
    elif sort_by == "Zenodo":
        merged = merged.sort_values(f"Zenodo 2024 Dual-LSTM {metric}", ascending=False)
    elif sort_by == "Pretrained" and f"Pretrained FloodHub {metric}" in merged.columns:
        merged = merged.sort_values(f"Pretrained FloodHub {metric}", ascending=False)
    else:
        merged = merged.sort_values(da_ref, ascending=False)

    return merged.head(top_n).reset_index(drop=True)


def style_zenodo_per_gauge_table(df_gauges: pd.DataFrame, metric: str = "NSE"):
    """Styles the per-gauge side-by-side comparison table."""
    if df_gauges.empty:
        return df_gauges

    metric_cols = [c for c in df_gauges.columns if metric in c and not c.startswith("Δ")]
    delta_cols = [c for c in df_gauges.columns if c.startswith("Δ")]

    format_dict = {}
    for c in metric_cols:
        format_dict[c] = "{:.3f}"
    for c in delta_cols:
        format_dict[c] = "{:+.3f}"

    styler = (
        df_gauges.style.format(format_dict, na_rep="-")
        .background_gradient(subset=metric_cols, cmap="YlGnBu", vmin=0.0, vmax=0.9)
        .background_gradient(subset=delta_cols, cmap="RdYlGn", vmin=-0.3, vmax=0.3)
        .set_caption(f"Per-Gauge Comparison at Lead t+1 ({metric}): DA Models vs. Zenodo 2024 Dual-LSTM")
    )
    return styler



def plot_zenodo_comparison_suite(
    df_eval: pd.DataFrame,
    df_zenodo: pd.DataFrame,
    metric: str = "NSE",
    lead_time_focus: int = 1,
    df_pretrained: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Renders a 4-panel comparative diagnostic suite against the 2024 Zenodo Dual-LSTM baseline.

    - Panel A: Multi-model Empirical CDF at Lead t+1 (Zenodo vs Pretrained vs Open-Loop vs Global Best DA vs Per-Basin Best DA)
    - Panel B: Median Forecast Horizon Decay (Leads t+1 .. t+7) across models
    - Panel C: Per-Gauge Scatter Plot (DA NSE vs Zenodo Dual-LSTM NSE) with Win-Rate quadrant
    - Panel D: Distribution of Per-Gauge ΔNSE (DA minus Zenodo Dual-LSTM) across horizons
    """
    if df_eval.empty or df_zenodo.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient overlapping data for Zenodo comparison.", ha="center", va="center")
        return fig

    zen_col = f"Zenodo {metric}"
    da_col = f"DA {metric}" if f"DA {metric}" in df_eval.columns else "DA NSE"
    base_col = f"Base {metric}" if f"Base {metric}" in df_eval.columns else "Base NSE"

    overlap_basins = set(df_zenodo["Basin ID"].unique()).intersection(set(df_eval["Basin ID"].unique()))
    sub_eval = df_eval[df_eval["Basin ID"].isin(overlap_basins)].copy()
    sub_zen = df_zenodo[df_zenodo["Basin ID"].isin(overlap_basins)].copy()
    sub_pt = (
        df_pretrained[df_pretrained["Basin ID"].isin(overlap_basins)].copy()
        if df_pretrained is not None and not df_pretrained.empty
        else pd.DataFrame()
    )

    eval_leads = detect_lead_times(sub_eval)
    zen_leads = sorted(sub_zen["Lead Time (Days)"].dropna().unique().astype(int).tolist())
    lead_times = sorted(list(set(eval_leads).intersection(set(zen_leads))))
    if not lead_times:
        lead_times = [1, 2, 3, 4, 5, 6, 7]

    best_cfg = get_global_best_config(sub_eval, lead_time=lead_time_focus)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel A: ECDF Comparison at Lead Time Focus (t+1)
    # -------------------------------------------------------------
    ax_ecdf = axes[0, 0]
    zen_l1 = sub_zen[sub_zen["Lead Time (Days)"] == lead_time_focus].set_index("Basin ID")[zen_col].dropna()
    base_l1 = sub_eval[sub_eval["Lead Time (Days)"] == lead_time_focus].drop_duplicates("Basin ID").set_index("Basin ID")[base_col].dropna()
    global_da_l1 = sub_eval[(sub_eval["Config ID"] == best_cfg) & (sub_eval["Lead Time (Days)"] == lead_time_focus)].drop_duplicates("Basin ID").set_index("Basin ID")[da_col].dropna()
    oracle_da_l1 = sub_eval[(sub_eval["Config ID"] == "Per-Basin Best DA") & (sub_eval["Lead Time (Days)"] == lead_time_focus)].drop_duplicates("Basin ID").set_index("Basin ID")[da_col].dropna()

    curves = [
        ("Zenodo 2024 Dual-LSTM", zen_l1, "#d95f02", "--", 2.2),
    ]
    if not sub_pt.empty:
        pt_col = f"Pretrained {metric}"
        if pt_col in sub_pt.columns:
            if "Lead Time (Days)" in sub_pt.columns:
                pt_sub_l1 = sub_pt[sub_pt["Lead Time (Days)"] == lead_time_focus]
            else:
                pt_sub_l1 = sub_pt
            pt_l1 = pt_sub_l1.drop_duplicates("Basin ID").set_index("Basin ID")[pt_col].dropna()
            if not pt_l1.empty:
                curves.append(("Pretrained Flood Hub Model", pt_l1, "#33a02c", ":", 2.2))

    curves.extend([
        ("OpenHydroNets Baseline", base_l1, "#7570b3", "-.", 2.0),
        (f"Global Best DA ({best_cfg[:18]})", global_da_l1, "#1b9e77", "-", 2.4),
        ("Per-Basin Best DA (Oracle)", oracle_da_l1, "#e7298a", "-", 2.6),
    ])

    for label, series, color, ls, lw in curves:
        if not series.empty:
            vals = np.sort(series.values)
            vals_clipped = np.clip(vals, -0.5, 1.0)
            y_ecdf = np.arange(1, len(vals_clipped) + 1) / len(vals_clipped)
            med_v = np.median(vals)
            ax_ecdf.plot(vals_clipped, y_ecdf, label=f"{label} (Med: {med_v:.3f})", color=color, linestyle=ls, linewidth=lw)

    ax_ecdf.set_title(f"A. Cumulative Distribution Function ({metric} at Lead t+{lead_time_focus})", fontsize=12, fontweight="bold")
    ax_ecdf.set_xlabel(f"{metric} (Clipped at -0.5)", fontsize=10)
    ax_ecdf.set_ylabel("Cumulative Probability (ECDF)", fontsize=10)
    ax_ecdf.set_xlim(-0.5, 1.0)
    ax_ecdf.grid(True, linestyle=":", alpha=0.6)
    ax_ecdf.legend(loc="upper left", fontsize=8.5, frameon=True)

    # -------------------------------------------------------------
    # Panel B: Multi-Horizon Skill Decay (Leads t+1 .. t+7)
    # -------------------------------------------------------------
    ax_decay = axes[0, 1]
    horizon_models = {
        "Zenodo 2024 Dual-LSTM": ("#d95f02", "o", "--"),
    }
    pt_col = f"Pretrained {metric}"
    if not sub_pt.empty and pt_col in sub_pt.columns:
        horizon_models["Pretrained Flood Hub Model"] = ("#33a02c", "v", ":")

    horizon_models.update({
        "OpenHydroNets Baseline": ("#7570b3", "s", "-."),
        "Global Best DA": ("#1b9e77", "^", "-"),
        "Per-Basin Best DA": ("#e7298a", "D", "-"),
    })

    for model_name, (color, marker, ls) in horizon_models.items():
        medians = []
        for lt in lead_times:
            if model_name == "Zenodo 2024 Dual-LSTM":
                s = sub_zen[sub_zen["Lead Time (Days)"] == lt][zen_col]
            elif model_name == "Pretrained Flood Hub Model":
                if "Lead Time (Days)" in sub_pt.columns:
                    s = sub_pt[sub_pt["Lead Time (Days)"] == lt][pt_col]
                elif lt == 1:
                    s = sub_pt[pt_col]
                else:
                    s = pd.Series(dtype=float)
            elif model_name == "OpenHydroNets Baseline":
                s = sub_eval[sub_eval["Lead Time (Days)"] == lt].drop_duplicates("Basin ID")[base_col]
            elif model_name == "Global Best DA":
                s = sub_eval[(sub_eval["Config ID"] == best_cfg) & (sub_eval["Lead Time (Days)"] == lt)].drop_duplicates("Basin ID")[da_col]
            else:
                s = sub_eval[(sub_eval["Config ID"] == "Per-Basin Best DA") & (sub_eval["Lead Time (Days)"] == lt)].drop_duplicates("Basin ID")[da_col]
            medians.append(s.median() if not s.empty and not s.dropna().empty else np.nan)

        ax_decay.plot(lead_times, medians, label=model_name, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=6)

    ax_decay.set_title(f"B. Forecast Horizon Skill Decay (Median {metric} across {len(overlap_basins):,} Gauges)", fontsize=12, fontweight="bold")
    ax_decay.set_xlabel("Forecast Lead Time (Days)", fontsize=10)
    ax_decay.set_ylabel(f"Median {metric}", fontsize=10)
    ax_decay.set_xticks(lead_times)
    ax_decay.set_xticklabels([f"t+{lt}" for lt in lead_times])
    ax_decay.grid(True, linestyle=":", alpha=0.6)
    ax_decay.legend(loc="lower left", fontsize=9, frameon=True)

    # -------------------------------------------------------------
    # Panel C: Gauge-Level Scatter Plot (DA vs Zenodo at Lead t+1)
    # -------------------------------------------------------------
    ax_scat = axes[1, 0]
    common_idx = oracle_da_l1.index.intersection(zen_l1.index)
    if len(common_idx) > 0:
        x_zen = np.clip(zen_l1.loc[common_idx].values, -0.5, 1.0)
        y_da = np.clip(oracle_da_l1.loc[common_idx].values, -0.5, 1.0)
        delta_vals = y_da - x_zen
        win_rate = (delta_vals > 0).mean() * 100.0

        sc = ax_scat.scatter(x_zen, y_da, c=delta_vals, cmap="RdYlGn", vmin=-0.3, vmax=0.3, alpha=0.8, edgecolors="k", linewidth=0.3, s=40)
        plt.colorbar(sc, ax=ax_scat, label=f"Δ{metric} (DA - Zenodo)")
        ax_scat.plot([-0.5, 1.0], [-0.5, 1.0], "k--", linewidth=1.2, label="1:1 Parity Line")
        ax_scat.set_title(f"C. Per-Gauge Parity (Lead t+{lead_time_focus}) | DA Win Rate: {win_rate:.1f}%", fontsize=12, fontweight="bold")
        ax_scat.set_xlabel(f"Zenodo 2024 Dual-LSTM {metric}", fontsize=10)
        ax_scat.set_ylabel(f"Per-Basin Best DA {metric}", fontsize=10)
        ax_scat.set_xlim(-0.5, 1.0)
        ax_scat.set_ylim(-0.5, 1.0)
        ax_scat.grid(True, linestyle=":", alpha=0.6)
        ax_scat.legend(loc="upper left", fontsize=9)

    # -------------------------------------------------------------
    # Panel D: Distribution of Per-Gauge ΔNSE Across Horizons
    # -------------------------------------------------------------
    ax_hist = axes[1, 1]
    horizons_to_plot = [lt for lt in [1, 3, 7] if lt in lead_times] or [lead_times[0]]
    colors_hist = ["#1b9e77", "#7570b3", "#e7298a"]

    for lt, col_h in zip(horizons_to_plot, colors_hist):
        zen_s = sub_zen[sub_zen["Lead Time (Days)"] == lt].set_index("Basin ID")[zen_col]
        da_s = sub_eval[(sub_eval["Config ID"] == "Per-Basin Best DA") & (sub_eval["Lead Time (Days)"] == lt)].drop_duplicates("Basin ID").set_index("Basin ID")[da_col]
        c_idx = da_s.index.intersection(zen_s.index)
        if len(c_idx) > 0:
            diffs = np.clip((da_s.loc[c_idx] - zen_s.loc[c_idx]).values, -0.5, 0.5)
            med_d = np.median(diffs)
            ax_hist.hist(diffs, bins=30, alpha=0.45, color=col_h, label=f"Lead t+{lt} (Med Δ: {med_d:+.3f})", density=True)

    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="Zero Difference")
    ax_hist.set_title(f"D. Distribution of Per-Gauge Skill Gain (Δ{metric} = DA - Zenodo)", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel(f"Δ{metric} (Clipped to [-0.5, +0.5])", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        f"Data Assimilation vs. 2024 Zenodo Dual-LSTM Paper Benchmark ({len(overlap_basins):,} Overlapping Catchments)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    return fig
