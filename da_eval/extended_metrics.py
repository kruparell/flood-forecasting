"""Extended DA evaluation metrics, statistical significance, KGE decomposition, overfitting diagnostics,
persistence/autocorrelation analysis (across Observed, Baseline, DA, and PP Post-Processing models),
flood-event peak timing/magnitude, exceedance precision/recall/CSI, and multi-year Leave-One-Year-Out
(5-yr -> 1-yr) cross-validation.

Key Design Principles:
  1. Per-Leadtime Default with Leadtimes as Columns: All tables calculate metrics per leadtime by default
     and format leadtimes (``t+1``, ``t+2``, ..., ``t+7``) as the table columns.
  2. Notebook-Wide Switches: Strictly respects ``da_eval.selection.get_active()`` for reference models and
     ``da_eval.units`` for displaying Skill Scores (SS) as percentages (%) or raw fractions.
  3. Interactive Exploration: Supports ``interactive=True``, ``lead_time=1..7`` sliders,
     ``show=["global_da", "per_basin_da", "global_rho", "per_basin_rho"]`` model toggles, and
     catchment-type / climate-regime filtering.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from da_eval import units
from da_eval import palette
from da_eval.ar1_metrics import DEFAULT_RHOS
from da_eval.benchmarks import get_global_best_config
from da_eval.ingestion import (
    _ensure_skill_score_columns,
    detect_lead_times,
    is_per_basin_rho_config,
    is_rho_config,
    parse_da_config_id,
)
from da_eval.lazy_timeseries import ar1_config_name, ar1_from_baseline, parse_ar1_config
from da_eval.selection import PER_BASIN_DA, PER_BASIN_RHO, SelectionPolicy, get_active
from da_eval.tables import DEFAULT_HORIZON_WINDOWS

YEARLY_STATS_CACHE = "yearly_sufficient_stats_v1.parquet"
PERSISTENCE_CACHE_V1 = "persistence_autocorr_v1.parquet"
PERSISTENCE_CACHE = "persistence_autocorr_v2.parquet"
EVENT_PEAK_CACHE = "event_peak_metrics_v1.parquet"

DEFAULT_SHOW_MODELS = ("global_da", "per_basin_da", "global_rho", "per_basin_rho")

SCHEME_LABELS = {
    "global_da": "Global-Best DA",
    "per_basin_da": "Per-Basin DA (Oracle)",
    "global_rho": "Global-Best PP",
    "per_basin_rho": "Per-Basin PP",
}

CATCHMENT_FILTER_OPTIONS = [
    ("All Catchments", "all"),
    ("Arid / Semi-Arid (P/PET < 0.50)", "arid"),
    ("Sub-Humid (P/PET 0.50–0.80)", "subhumid"),
    ("Humid (P/PET ≥ 0.80)", "humid"),
    ("Köppen B — Arid / Steppe", "kg_B"),
    ("Köppen A — Tropical", "kg_A"),
    ("Köppen C — Temperate", "kg_C"),
    ("Köppen D — Continental / Cold", "kg_D"),
    ("Köppen E — Polar / Alpine", "kg_E"),
    ("Flashy / Low Memory (Q1 ρ_e)", "q1_mem"),
    ("Baseflow / High Memory (Q4 ρ_e)", "q4_mem"),
    ("Low Baseline NSE (< 0.50)", "low_base"),
    ("Mid Baseline NSE (0.50–0.70)", "mid_base"),
    ("High Baseline NSE (≥ 0.70)", "high_base"),
    ("Snow-Dominant (Snow Frac ≥ 0.30)", "snow"),
    ("Small Catchments (< 500 km²)", "small_area"),
    ("Large Catchments (> 2,500 km²)", "large_area"),
]

AUTOCORR_MODE_SPECS = {
    "flow_lagL": (
        "Flow Autocorr Lag-L",
        "Lag-L Streamflow Autocorrelation $\\rho_Q(L) = \\mathrm{Corr}(Q(t+L), Q(t))$",
        "Lag-L Flow Autocorr ρ_Q(L)",
    ),
    "issue_persist": (
        "Issue-to-Lead Flow Corr",
        "Issue-Date Flow Persistence $\\mathrm{Corr}(\\hat{Q}(t+L\\mid t), Q_{\\mathrm{obs}}(t))$",
        "Issue-Date Corr(Q̂(t+L|t), Q_obs(t))",
    ),
    "flow_lag1": (
        "Flow Autocorr Lag-1",
        "Lag-1 Hydrograph Smoothness $\\rho_Q(1) = \\mathrm{Corr}(\\hat{Q}(t+1, L), \\hat{Q}(t, L))$",
        "Lag-1 Flow Smoothness ρ_Q(1)",
    ),
    "res_lagL": (
        "Residual Autocorr Lag-L",
        "Lag-L Residual Error Autocorrelation $\\rho_e(L)$ & Assumed PP $\\rho^L$",
        "Lag-L Error Autocorr ρ_e(L)",
    ),
    "res_lag1": (
        "Residual Autocorr Lag-1",
        "Lag-1 Residual Error Autocorrelation $\\rho_e(1)$ at Lead $t+L$",
        "Lag-1 Error Autocorr ρ_e(1)",
    ),
}


# =============================================================================
# Helper utilities: reference schemes, lead windows, and catchment filtering
# =============================================================================
def _per_lead_windows(df: pd.DataFrame, lead_times: Optional[Sequence[int]] = None) -> Dict[str, Tuple[int, ...]]:
    """Default per-leadtime column mapping ``{'t+1': (1,), 't+2': (2,), ...}``."""
    leads = list(lead_times) if lead_times is not None else detect_lead_times(df)
    leads = [int(lt) for lt in leads if int(lt) >= 1]
    return {f"t+{lt}": (lt,) for lt in leads}


def _resolve_windows(
    df: pd.DataFrame,
    windows: Optional[Dict[str, Sequence[int]] | str] = None,
    lead_times: Optional[Sequence[int]] = None,
) -> Dict[str, Tuple[int, ...]]:
    """Resolves ``windows`` into a dict of ``{col_label: (leads...)}``.
    Defaults to per-leadtime columns ``t+1 .. t+7``. Pass ``windows="grouped"`` for Short/Medium/Full windows.
    """
    if windows is None or windows == "per_lead":
        return _per_lead_windows(df, lead_times=lead_times)
    if isinstance(windows, str) and windows.lower() in ("grouped", "windows", "horizon_windows"):
        return {k: tuple(v) for k, v in DEFAULT_HORIZON_WINDOWS.items()}
    return {str(k): tuple(v) for k, v in windows.items()}


def _resolve_scheme_configs(df_eval: pd.DataFrame, show: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Returns canonical reference scheme mapping from the active notebook-wide selection."""
    sel = get_active()
    cfgs = set(df_eval["Config ID"].dropna().astype(str))
    real_da = [c for c in cfgs if not is_rho_config(c) and "per-basin" not in c.lower() and "baseline" not in c.lower() and c != "Observed"]
    fixed_rho = [c for c in cfgs if is_rho_config(c) and not is_per_basin_rho_config(c)]
    g_da = sel.global_da if (sel is not None and sel.global_da in cfgs) else (
        get_global_best_config(df_eval) if real_da else None
    )
    g_rho = sel.global_rho if (sel is not None and sel.global_rho in cfgs) else (
        get_global_best_config(df_eval[df_eval["Config ID"].isin(fixed_rho)], exclude_rho=False) if fixed_rho else None
    )
    wanted = list(DEFAULT_SHOW_MODELS if show is None else show)
    out = {}
    for k in wanted:
        if k == "global_da" and g_da and g_da in cfgs:
            out["global_da"] = g_da
        elif k == "per_basin_da" and PER_BASIN_DA in cfgs:
            out["per_basin_da"] = PER_BASIN_DA
        elif k == "global_rho" and g_rho and g_rho in cfgs:
            out["global_rho"] = g_rho
        elif k == "per_basin_rho" and PER_BASIN_RHO in cfgs:
            out["per_basin_rho"] = PER_BASIN_RHO
    return out


def filter_basins_by_catchment_type(
    df_eval: pd.DataFrame,
    catchment_filter: str = "all",
    df_meta: Optional[pd.DataFrame] = None,
    df_koppen: Optional[pd.DataFrame] = None,
    df_pers: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[set], str]:
    """Returns ``(allowed_basin_set_or_None, human_readable_label)`` for ``catchment_filter``."""
    f = (catchment_filter or "all").strip()
    if not f or f.lower() == "all":
        return None, "All Catchments"

    label_map = dict((v, k) for k, v in CATCHMENT_FILTER_OPTIONS)
    desc = label_map.get(f, f)
    all_basins = set(df_eval["Basin ID"].dropna().astype(str).unique())

    if f in ("low_base", "mid_base", "high_base"):
        sub1 = df_eval[df_eval["Lead Time (Days)"] == 1].drop_duplicates("Basin ID")
        if "Base NSE" not in sub1.columns or sub1.empty:
            return all_basins, desc
        b_nse = sub1.set_index("Basin ID")["Base NSE"].dropna()
        if f == "low_base":
            return set(b_nse[b_nse < 0.50].index.astype(str)), desc
        if f == "mid_base":
            return set(b_nse[(b_nse >= 0.50) & (b_nse < 0.70)].index.astype(str)), desc
        return set(b_nse[b_nse >= 0.70].index.astype(str)), desc

    if f.startswith("kg_") and df_koppen is not None and not df_koppen.empty:
        grp = f.split("_", 1)[1].upper()
        matched = df_koppen[df_koppen["kg_group"].astype(str).str.upper() == grp]["Basin ID"].astype(str)
        return set(matched) & all_basins, desc

    if f in ("q1_mem", "q4_mem") and df_pers is not None and not df_pers.empty:
        p1 = df_pers[(df_pers["Config ID"].isin(["Observed", "Baseline"])) & (df_pers["Lead Time (Days)"] == 1)]
        col = "Residual Autocorr Lag-L" if "Residual Autocorr Lag-L" in p1.columns else "Residual Autocorrelation"
        if col in p1.columns:
            s = p1.drop_duplicates("Basin ID").set_index("Basin ID")[col].dropna()
            if len(s) >= 4:
                q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
                if f == "q1_mem":
                    return set(s[s <= q25].index.astype(str)) & all_basins, desc
                return set(s[s >= q75].index.astype(str)) & all_basins, desc

    if df_meta is not None and not df_meta.empty and "Basin ID" in df_meta.columns:
        meta = df_meta.drop_duplicates("Basin ID").set_index("Basin ID")
        if f in ("arid", "subhumid", "humid") and "unep_aridity_index" in meta.columns:
            ai = meta["unep_aridity_index"].dropna()
            if f == "arid":
                return set(ai[ai < 0.50].index.astype(str)) & all_basins, desc
            if f == "subhumid":
                return set(ai[(ai >= 0.50) & (ai < 0.80)].index.astype(str)) & all_basins, desc
            return set(ai[ai >= 0.80].index.astype(str)) & all_basins, desc
        if f == "snow" and "frac_snow" in meta.columns:
            sn = meta["frac_snow"].dropna()
            return set(sn[sn >= 0.30].index.astype(str)) & all_basins, desc
        if f == "rain" and "frac_snow" in meta.columns:
            sn = meta["frac_snow"].dropna()
            return set(sn[sn < 0.10].index.astype(str)) & all_basins, desc
        if f == "small_area" and "area" in meta.columns:
            ar = meta["area"].dropna()
            return set(ar[ar < 500.0].index.astype(str)) & all_basins, desc
        if f == "large_area" and "area" in meta.columns:
            ar = meta["area"].dropna()
            return set(ar[ar > 2500.0].index.astype(str)) & all_basins, desc

    return None, desc


def _basin_window_means(frame: pd.DataFrame, col: str, leads: Sequence[int]) -> pd.Series:
    """Per basin, arithmetic mean of ``col`` across ``leads`` (fast path for single lead)."""
    leads = list(leads)
    if frame.empty or col not in frame.columns:
        return pd.Series(dtype=float)
    if len(leads) == 1:
        sub = frame[frame["Lead Time (Days)"] == leads[0]].dropna(subset=[col])
        return sub.drop_duplicates("Basin ID").set_index("Basin ID")[col].astype(float)
    sub = frame[frame["Lead Time (Days)"].isin(leads)]
    if sub.empty:
        return pd.Series(dtype=float)
    w = sub.pivot_table(index="Basin ID", columns="Lead Time (Days)", values=col, aggfunc="first").reindex(columns=leads)
    return w[w.notna().all(axis=1)].mean(axis=1)


def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _wilcoxon_p(x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test on finite paired differences -> (stat, p_value)."""
    d = np.asarray(x, dtype=float) if y is None else (np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-12]
    if nz.size < 5:
        return np.nan, np.nan
    try:
        res = sp_stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, np.nan


def _holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down multiple-testing adjustment."""
    arr = np.asarray(pvals, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(arr))[0]
    m = len(finite_idx)
    if m == 0:
        return out.tolist()
    order = finite_idx[np.argsort(arr[finite_idx])]
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, arr[idx] * (m - rank))
        running_max = max(running_max, adj)
        out[idx] = running_max
    return out.tolist()


# =============================================================================
# 1. Paired Wilcoxon Signed-Rank Significance Table (Leadtimes as Columns)
# =============================================================================
def build_wilcoxon_significance_table(
    df_eval: pd.DataFrame,
    windows: Optional[Dict[str, Sequence[int]] | str] = None,
    metric_col: str = "NSE Skill Score",
    common_basins: bool = True,
    show: Optional[Sequence[str]] = None,
    stat_view: str = "summary",
    lead_times: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank significance & win-rate table with Leadtimes (``t+1..t+7``) as COLUMNS.

    By default ``windows=None`` computes statistics per leadtime (``t+1``, ``t+2``, ..., ``t+7`` as columns).
    Pass ``windows="grouped"`` to switch columns to Short-Range / Medium-Range / Full Horizon.
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    win_map = _resolve_windows(df_eval, windows=windows, lead_times=lead_times)
    refs = _resolve_scheme_configs(df_eval, show=show)
    col_names = list(win_map.keys())
    is_ss_or_delta = ("Skill" in metric_col) or ("Delta" in metric_col)

    # Compute raw cell stats across all (scheme, window_col) first so Holm-Bonferroni corrects over the full grid
    grid_records: List[Dict[str, Any]] = []
    n_basins_per_col: Dict[str, int] = {}

    for wname, leads in win_map.items():
        series_map = {
            k: _basin_window_means(df_eval[df_eval["Config ID"] == cid], metric_col, leads)
            for k, cid in refs.items()
        }
        if common_basins and series_map:
            common = sorted(set.intersection(*(set(s.index) for s in series_map.values() if not s.empty)))
            series_map = {k: s.reindex(common) for k, s in series_map.items()}
            n_basins_per_col[wname] = len(common)
        else:
            n_basins_per_col[wname] = max((len(s.dropna()) for s in series_map.values()), default=0)

        for k in refs:
            s = series_map[k].dropna()
            vals = s.to_numpy(float)
            n = len(vals)
            med = float(np.median(vals)) if n else np.nan
            win_base = float((vals > 0).mean() * 100.0) if n else np.nan
            harm_rate = float((vals < -0.01).mean() * 100.0) if n else np.nan
            _, p_base = _wilcoxon_p(vals)

            ar1_key = "global_rho" if k == "global_da" else ("per_basin_rho" if k == "per_basin_da" else None)
            if ar1_key and ar1_key in series_map:
                paired = pd.concat([series_map[k].rename("da"), series_map[ar1_key].rename("ar1")], axis=1).dropna()
                diff_ar1 = (paired["da"] - paired["ar1"]).to_numpy(float)
                med_diff_ar1 = float(np.median(diff_ar1)) if len(diff_ar1) else np.nan
                win_vs_ar1 = float((diff_ar1 > 0).mean() * 100.0) if len(diff_ar1) else np.nan
                _, p_vs_ar1 = _wilcoxon_p(paired["da"].to_numpy(float), paired["ar1"].to_numpy(float))
            else:
                med_diff_ar1, win_vs_ar1, p_vs_ar1 = np.nan, np.nan, np.nan

            grid_records.append({
                "col": wname,
                "skey": k,
                "scheme": SCHEME_LABELS.get(k, k),
                "cfg": refs[k],
                "n": n,
                "med": med,
                "win_base": win_base,
                "harm": harm_rate,
                "p_base": p_base,
                "med_diff_ar1": med_diff_ar1,
                "win_ar1": win_vs_ar1,
                "p_ar1": p_vs_ar1,
            })

    if not grid_records:
        return pd.DataFrame()

    gdf = pd.DataFrame(grid_records)
    gdf["holm_p_base"] = _holm_bonferroni(gdf["p_base"].tolist())
    gdf["holm_p_ar1"] = _holm_bonferroni(gdf["p_ar1"].tolist())

    def _fmt_val(v: float) -> str:
        if not np.isfinite(v):
            return "–"
        return units.fmt_ss(v) if "Skill" in metric_col else f"{v:+.3f}"

    # Pivot into rows = (Scheme, Statistic), columns = Leadtimes (t+1 .. t+7)
    all_stat_rows = [
        ("Median (with Sig vs Base)", "med_sig"),
        ("Median", "med"),
        ("% Improved (>0 vs Base)", "win_base"),
        ("% Harmed (<-1% vs Base)", "harm"),
        ("Holm p-value (vs Base)", "p_base_str"),
        ("Median Δ (DA − PP)", "diff_ar1_sig"),
        ("% Win (DA vs PP)", "win_ar1"),
        ("Holm p-value (DA vs PP)", "p_ar1_str"),
    ]
    if stat_view == "median":
        wanted_stats = ["med_sig", "win_base"]
    elif stat_view == "win_base":
        wanted_stats = ["win_base", "harm"]
    elif stat_view == "vs_ar1":
        wanted_stats = ["diff_ar1_sig", "win_ar1", "p_ar1_str"]
    elif stat_view == "pvalues":
        wanted_stats = ["med", "p_base_str", "diff_ar1_sig", "p_ar1_str"]
    elif stat_view == "all":
        wanted_stats = [k for _, k in all_stat_rows]
    else:  # "summary" / "compact" default
        wanted_stats = ["med_sig", "win_base", "harm", "diff_ar1_sig", "win_ar1"]

    table_rows = []
    for skey in refs:
        slabel = SCHEME_LABELS.get(skey, skey)
        sub_s = gdf[gdf["skey"] == skey].set_index("col")
        n_rep = int(sub_s["n"].max()) if not sub_s.empty else 0
        is_da = skey in ("global_da", "per_basin_da")

        for stat_label, stat_code in all_stat_rows:
            if stat_code not in wanted_stats:
                continue
            if stat_code in ("diff_ar1_sig", "win_ar1", "p_ar1_str") and not is_da:
                continue
            row_dict: Dict[str, Any] = {
                "Scheme": slabel,
                "Statistic": stat_label,
                "N Basins": n_rep,
                "_stat_code": stat_code,
            }
            for wname in col_names:
                if wname not in sub_s.index:
                    row_dict[wname] = np.nan
                    continue
                r = sub_s.loc[wname]
                if stat_code == "med_sig":
                    stars = _sig_stars(float(r["holm_p_base"]))
                    row_dict[wname] = f"{_fmt_val(float(r['med']))} ({stars})" if np.isfinite(r["med"]) else "–"
                elif stat_code == "med":
                    row_dict[wname] = float(r["med"])
                elif stat_code == "win_base":
                    row_dict[wname] = float(r["win_base"])
                elif stat_code == "harm":
                    row_dict[wname] = float(r["harm"])
                elif stat_code == "p_base_str":
                    p = float(r["holm_p_base"])
                    row_dict[wname] = f"{p:.2e} ({_sig_stars(p)})" if np.isfinite(p) else "–"
                elif stat_code == "diff_ar1_sig":
                    d_val = float(r["med_diff_ar1"])
                    p_a = float(r["holm_p_ar1"])
                    row_dict[wname] = f"{_fmt_val(d_val)} ({_sig_stars(p_a)})" if np.isfinite(d_val) else "–"
                elif stat_code == "win_ar1":
                    row_dict[wname] = float(r["win_ar1"])
                elif stat_code == "p_ar1_str":
                    p_a = float(r["holm_p_ar1"])
                    row_dict[wname] = f"{p_a:.2e} ({_sig_stars(p_a)})" if np.isfinite(p_a) else "–"
            table_rows.append(row_dict)

    out = pd.DataFrame(table_rows)
    sel = get_active()
    out.attrs["selection"] = sel.label() if sel is not None else "default t+1"
    out.attrs["metric_col"] = metric_col
    out.attrs["col_names"] = col_names
    out.attrs["raw_grid"] = gdf
    return out


def style_wilcoxon_significance_table(df: pd.DataFrame) -> Any:
    if df.empty:
        return df
    col_names = df.attrs.get("col_names", [c for c in df.columns if c.startswith("t+") or "Range" in c or "Horizon" in c])
    metric_col = df.attrs.get("metric_col", "NSE Skill Score")
    is_ss = "Skill" in metric_col

    disp = df.drop(columns=["_stat_code"], errors="ignore").copy()
    for c in col_names:
        if c in disp.columns:
            disp[c] = disp[c].astype(object)
    for idx, r in df.iterrows():
        scode = r.get("_stat_code", "")
        for c in col_names:
            val = r[c]
            if isinstance(val, (int, float, np.floating)):
                if not np.isfinite(val):
                    disp.at[idx, c] = "–"
                elif scode in ("win_base", "harm", "win_ar1"):
                    disp.at[idx, c] = f"{float(val):.1f}%"
                elif scode == "med":
                    disp.at[idx, c] = units.fmt_ss(float(val)) if is_ss else f"{float(val):+.3f}"

    sty = disp.style.format({"N Basins": "{:,d}"}, na_rep="–").hide(axis="index")
    sty = sty.set_properties(subset=["Scheme", "Statistic"], **{"text-align": "left", "font-weight": "bold"})
    sty = sty.set_properties(subset=col_names, **{"text-align": "right", "padding": "4px 10px"})
    sty = sty.set_caption(
        f"Paired Two-Sided Wilcoxon Signed-Rank Significance Across Lead Times ({metric_col}) | "
        f"Holm–Bonferroni adjusted: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05 | References {df.attrs.get('selection', '')}"
    )
    return sty


# =============================================================================
# 2. KGE & NSE Component Decomposition (Leadtimes as Columns)
# =============================================================================
KGE_COMPONENTS = (
    ("KGE", "KGE", 1.0, "KGE Skill Score"),
    ("NSE", "NSE", 1.0, "NSE Skill Score"),
    ("Pearson-r", "Correlation r", 1.0, "Pearson-r Skill Score"),
    ("Alpha-NSE", "Variability Ratio α (σ_sim/σ_obs)", 1.0, "Alpha-NSE Skill Score"),
    ("Beta-KGE", "Volume Bias Ratio β_KGE (μ_sim/μ_obs)", 1.0, "Beta-KGE Skill Score"),
    ("Beta-NSE", "Normalized Bias β_NSE ((μ_sim−μ_obs)/σ_obs)", 0.0, "Beta-NSE Skill Score"),
)


def build_kge_decomposition_table(
    df_eval: pd.DataFrame,
    windows: Optional[Dict[str, Sequence[int]] | str] = None,
    stat: str = "median",
    common_basins: bool = True,
    show: Optional[Sequence[str]] = None,
    value_mode: str = "both",  # "both", "raw", or "ss"
    component_filter: str = "all",
    lead_times: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """KGE & NSE Component Decomposition table with Leadtimes (``t+1..t+7``) as COLUMNS.

    Reports BOTH raw median M (direction of under/over-prediction) and Skill Score SS_M (%)
    for KGE, NSE, Pearson-r, Alpha-NSE, Beta-KGE, and Beta-NSE across lead times.
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    win_map = _resolve_windows(df_eval, windows=windows, lead_times=lead_times)
    col_names = list(win_map.keys())
    refs = _resolve_scheme_configs(df_eval, show=show)
    cfgs = set(df_eval["Config ID"].dropna().astype(str))
    real_da = [c for c in cfgs if not is_rho_config(c) and "per-basin" not in c.lower() and "baseline" not in c.lower()]
    base_src = df_eval[df_eval["Config ID"].isin(real_da)].drop_duplicates(["Basin ID", "Lead Time (Days)"])
    agg_fn = np.median if stat == "median" else np.mean

    # Determine common basins across all leads
    all_leads = sorted({lt for lts in win_map.values() for lt in lts})
    common = None
    if common_basins and all_leads:
        b_sets = []
        w_b = base_src.pivot_table(index="Basin ID", columns="Lead Time (Days)", values="Base NSE").reindex(columns=all_leads).dropna()
        if not w_b.empty:
            b_sets.append(set(w_b.index))
        for cid in refs.values():
            w_c = df_eval[df_eval["Config ID"] == cid].pivot_table(index="Basin ID", columns="Lead Time (Days)", values="DA NSE").reindex(columns=all_leads).dropna()
            if not w_c.empty:
                b_sets.append(set(w_c.index))
        if b_sets:
            common = set.intersection(*b_sets)

    scheme_order = [("baseline", "Baseline (Open-Loop)", None)] + [
        (k, SCHEME_LABELS[k], refs[k]) for k in refs
    ]

    comps = [c for c in KGE_COMPONENTS if component_filter in ("all", c[0])]
    rows = []
    for m_key, m_label, opt_val, ss_col in comps:
        if value_mode in ("both", "raw"):
            for skey, slabel, cid in scheme_order:
                src = base_src if skey == "baseline" else df_eval[df_eval["Config ID"] == cid]
                if common is not None:
                    src = src[src["Basin ID"].isin(common)]
                raw_col = f"Base {m_key}" if skey == "baseline" else f"DA {m_key}"
                row: Dict[str, Any] = {
                    "Component": f"{m_key} (opt={opt_val:g})",
                    "Scheme": slabel,
                    "Metric Type": f"Raw {stat.capitalize()}",
                    "N Basins": len(common) if common is not None else int(src["Basin ID"].nunique()),
                    "_is_ss": False,
                }
                for wname, leads in win_map.items():
                    s_raw = _basin_window_means(src, raw_col, leads)
                    row[wname] = float(agg_fn(s_raw)) if len(s_raw) else np.nan
                rows.append(row)

        if value_mode in ("both", "ss"):
            for skey, slabel, cid in scheme_order:
                if skey == "baseline":
                    continue
                src = df_eval[df_eval["Config ID"] == cid]
                if common is not None:
                    src = src[src["Basin ID"].isin(common)]
                row = {
                    "Component": f"{m_key} (opt={opt_val:g})",
                    "Scheme": slabel,
                    "Metric Type": units.ss_label("Skill Score SS"),
                    "N Basins": len(common) if common is not None else int(src["Basin ID"].nunique()),
                    "_is_ss": True,
                }
                for wname, leads in win_map.items():
                    s_ss = _basin_window_means(src, ss_col, leads)
                    row[wname] = float(agg_fn(s_ss)) if len(s_ss) else np.nan
                rows.append(row)

    out = pd.DataFrame(rows)
    sel = get_active()
    out.attrs["selection"] = sel.label() if sel is not None else "default t+1"
    out.attrs["stat"] = stat
    out.attrs["col_names"] = col_names
    return out


def style_kge_decomposition_table(df: pd.DataFrame) -> Any:
    if df.empty:
        return df
    col_names = df.attrs.get("col_names", [c for c in df.columns if c.startswith("t+") or "Range" in c or "Horizon" in c])
    disp = df.drop(columns=["_is_ss"], errors="ignore").copy()
    for c in col_names:
        if c in disp.columns:
            disp[c] = disp[c].astype(object)
    for idx, r in df.iterrows():
        is_ss = bool(r.get("_is_ss", False))
        for c in col_names:
            val = r[c]
            if pd.isna(val) or not np.isfinite(float(val)):
                disp.at[idx, c] = "–"
            elif is_ss:
                disp.at[idx, c] = units.fmt_ss(float(val))
            else:
                disp.at[idx, c] = f"{float(val):.3f}"

    sty = disp.style.format({"N Basins": "{:,d}"}, na_rep="–").hide(axis="index")
    sty = sty.set_properties(subset=["Component", "Scheme", "Metric Type"], **{"text-align": "left"})
    sty = sty.set_properties(subset=col_names, **{"text-align": "right", "padding": "4px 10px"})
    sty = sty.set_caption(
        f"KGE & NSE Component Decomposition Across Lead Times: Raw {df.attrs.get('stat', 'median')} values "
        f"(ideal: r=1, α=1, β_KGE=1, β_NSE=0) and distance-to-optimum Skill Scores "
        f"SS_M = 1 − |M_DA − M*| / |M_base − M*| | References {df.attrs.get('selection', '')}"
    )
    return sty


def plot_kge_decomposition(
    df_eval: pd.DataFrame,
    lead_times: Optional[Sequence[int]] = None,
    lead_time_focus: int = 1,
    component_focus: str = "all",
    show: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (18, 8.8),
) -> plt.Figure:
    """Publication figure of KGE/NSE components across leads t+1..t+7:
      - When ``component_focus == 'all'``: 2x4 grid (Top row: Raw median values with vertical marker at
        ``lead_time_focus``; Bottom row: Skill Score SS_M %).
      - When ``component_focus`` is a specific component (e.g. ``'Alpha-NSE'`` or ``'Pearson-r'``):
        2-panel view showing Panel A: Across-basin ECDF at ``lead_time_focus`` (Raw & Skill Score) and
        Panel B: Horizon trajectory across ``t+1..t+7``.
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    leads = [int(lt) for lt in (lead_times if lead_times is not None else detect_lead_times(df_eval)) if int(lt) >= 1]
    refs = _resolve_scheme_configs(df_eval, show=show)
    cfgs = set(df_eval["Config ID"].dropna().astype(str))
    real_da = [c for c in cfgs if not is_rho_config(c) and "per-basin" not in c.lower() and "baseline" not in c.lower()]
    base_src = df_eval[df_eval["Config ID"].isin(real_da)].drop_duplicates(["Basin ID", "Lead Time (Days)"])

    common_sets = []
    for cid in refs.values():
        w = df_eval[df_eval["Config ID"] == cid].pivot_table(index="Basin ID", columns="Lead Time (Days)", values="DA NSE")
        w = w.reindex(columns=leads).dropna()
        if not w.empty:
            common_sets.append(set(w.index))
    common = set.intersection(*common_sets) if common_sets else set(df_eval["Basin ID"].unique())

    all_styles = [
        ("baseline", "Open-Loop Baseline", palette.color("baseline"), palette.ls("baseline"), "o"),
        ("global_da", "Global-Best DA", palette.color("global_da"), palette.ls("global_da"), "s"),
        ("per_basin_da", "Per-Basin DA (Oracle)", palette.color("per_basin_da"), palette.ls("per_basin_da"), "D"),
        ("global_rho", "Global-Best PP", palette.color("global_rho"), palette.ls("global_rho"), "^"),
        ("per_basin_rho", "Per-Basin PP", palette.color("per_basin_rho"), palette.ls("per_basin_rho"), "v"),
    ]
    styles = [s for s in all_styles if s[0] == "baseline" or s[0] in refs]

    comp_lookup = {c[0]: c for c in KGE_COMPONENTS}
    if component_focus in comp_lookup:
        m_key, m_title, opt_val, ss_col = comp_lookup[component_focus]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.4))
        lt_f = lead_time_focus if lead_time_focus in leads else leads[0]

        # Panel A: Raw ECDF at lead_time_focus
        ax1.axvline(opt_val, color="#80868b", lw=1.2, ls=":", label=f"Optimum ({opt_val:g})")
        for skey, slabel, color, ls, mk in styles:
            src = base_src if skey == "baseline" else df_eval[df_eval["Config ID"] == refs[skey]]
            raw_col = f"Base {m_key}" if skey == "baseline" else f"DA {m_key}"
            vals = np.sort(src[(src["Basin ID"].isin(common)) & (src["Lead Time (Days)"] == lt_f)][raw_col].dropna().to_numpy(float))
            if len(vals):
                q_lo, q_hi = np.quantile(vals, [0.02, 0.98])
                ax1.plot(vals, np.linspace(0, 1, len(vals)), label=f"{slabel} (med={np.median(vals):.3f})", color=color, ls=ls, lw=2.3)
        ax1.set_title(f"A. Raw {m_key} ECDF at Lead t+{lt_f}", fontsize=11.5, fontweight="bold")
        ax1.set_xlabel(f"Per-Basin {m_title} (Lead t+{lt_f})", fontsize=10.5)
        ax1.set_ylabel("Fraction of Basins (ECDF)", fontsize=10.5)
        ax1.set_xlim(-0.2 if opt_val == 1.0 else -0.6, 1.4 if opt_val == 1.0 else 0.6)
        ax1.legend(loc="best", fontsize=8.8)
        ax1.grid(True, alpha=0.3)

        # Panel B: Raw trajectory across leads t+1..t+7
        ax2.axhline(opt_val, color="#80868b", lw=1.1, ls=":")
        ax2.axvline(lt_f, color="#dadce0", lw=6.0, zorder=0)
        for skey, slabel, color, ls, mk in styles:
            src = base_src if skey == "baseline" else df_eval[df_eval["Config ID"] == refs[skey]]
            src = src[src["Basin ID"].isin(common)]
            raw_col = f"Base {m_key}" if skey == "baseline" else f"DA {m_key}"
            meds = [float(src.loc[src["Lead Time (Days)"] == lt, raw_col].median()) for lt in leads]
            ax2.plot(leads, meds, label=slabel, color=color, ls=ls, marker=mk, lw=2.2, ms=5.5)
        ax2.set_xticks(leads)
        ax2.set_xticklabels([f"t+{l}" for l in leads])
        ax2.set_title(f"B. Raw Median {m_key} Across Leads", fontsize=11.5, fontweight="bold")
        ax2.set_xlabel("Forecast Lead Time (Days)", fontsize=10.5)
        ax2.set_ylabel(f"Median {m_key}", fontsize=10.5)
        ax2.grid(True, alpha=0.3)

        # Panel C: Skill Score trajectory across leads t+1..t+7
        ax3.axhline(0.0, color="#80868b", lw=1.1, ls="--")
        ax3.axvline(lt_f, color="#dadce0", lw=6.0, zorder=0)
        for skey, slabel, color, ls, mk in styles:
            if skey == "baseline":
                continue
            src = df_eval[(df_eval["Config ID"] == refs[skey]) & (df_eval["Basin ID"].isin(common))]
            meds = [float(src.loc[src["Lead Time (Days)"] == lt, ss_col].median()) for lt in leads]
            ax3.plot(leads, meds, label=slabel, color=color, ls=ls, marker=mk, lw=2.2, ms=5.5)
        units.ss_axis(ax3, "y")
        ax3.set_xticks(leads)
        ax3.set_xticklabels([f"t+{l}" for l in leads])
        ax3.set_title(f"C. {units.ss_label(f'SS ({m_key})')} Across Leads", fontsize=11.5, fontweight="bold")
        ax3.set_xlabel("Forecast Lead Time (Days)", fontsize=10.5)
        ax3.set_ylabel(units.ss_label("Skill Score"), fontsize=10.5)
        ax3.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    panels = [
        ("Pearson-r", "Correlation $r$", 1.0, "Pearson-r Skill Score"),
        ("Alpha-NSE", "Variability Ratio $\\alpha = \\sigma_{\\mathrm{sim}}/\\sigma_{\\mathrm{obs}}$", 1.0, "Alpha-NSE Skill Score"),
        ("Beta-KGE", "Volume Bias $\\beta_{\\mathrm{KGE}} = \\mu_{\\mathrm{sim}}/\\mu_{\\mathrm{obs}}$", 1.0, "Beta-KGE Skill Score"),
        ("KGE", "Overall KGE", 1.0, "KGE Skill Score"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    for col_idx, (m_key, m_title, opt_val, ss_col) in enumerate(panels):
        ax_raw = axes[0, col_idx]
        ax_ss = axes[1, col_idx]
        ax_raw.axhline(opt_val, color="#80868b", lw=1.1, ls=":", zorder=1, label="Optimum (1.0)" if col_idx == 0 else None)
        ax_ss.axhline(0.0, color="#80868b", lw=1.1, ls="--", zorder=1)
        if lead_time_focus in leads:
            ax_raw.axvline(lead_time_focus, color="#e8eaed", lw=5.0, zorder=0)
            ax_ss.axvline(lead_time_focus, color="#e8eaed", lw=5.0, zorder=0)

        for skey, slabel, color, ls, marker in styles:
            src = base_src if skey == "baseline" else df_eval[df_eval["Config ID"] == refs[skey]]
            src = src[src["Basin ID"].isin(common)]
            raw_col = f"Base {m_key}" if skey == "baseline" else f"DA {m_key}"
            if raw_col not in src.columns or src[raw_col].dropna().empty:
                continue
            med_raw = [float(src.loc[src["Lead Time (Days)"] == lt, raw_col].median()) for lt in leads]
            ax_raw.plot(leads, med_raw, label=slabel, color=color, ls=ls, marker=marker, lw=2.2, ms=5.5)

            if skey != "baseline" and ss_col in src.columns and not src[ss_col].dropna().empty:
                med_ss = [float(src.loc[src["Lead Time (Days)"] == lt, ss_col].median()) for lt in leads]
                ax_ss.plot(leads, med_ss, label=slabel, color=color, ls=ls, marker=marker, lw=2.2, ms=5.5)

        ax_raw.set_title(m_title, fontsize=12, fontweight="bold")
        ax_raw.set_ylabel("Raw Median Value" if col_idx == 0 else "", fontsize=11)
        ax_raw.grid(True, alpha=0.3)

        units.ss_axis(ax_ss, "y")
        ax_ss.set_title(units.ss_label(f"SS ({m_key})"), fontsize=11, fontweight="bold")
        ax_ss.set_ylabel(units.ss_label("Skill Score") if col_idx == 0 else "", fontsize=11)
        ax_ss.set_xlabel("Forecast Lead Time (Days)", fontsize=11)
        ax_ss.set_xticks(leads)
        ax_ss.set_xticklabels([f"t+{l}" for l in leads])
        ax_ss.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    sel = get_active()
    sel_txt = sel.label() if sel is not None else "default t+1"
    fig.suptitle(
        f"KGE Component Decomposition Across Lead Times: Raw Medians & Distance-to-Optimum Skill Scores "
        f"(N = {len(common):,} basins | {sel_txt})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# =============================================================================
# 3. Assimilation Overfitting vs. Forecast Generalization (Leadtimes as Columns)
# =============================================================================
def build_assimilation_overfitting_table(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    common_basins: bool = True,
) -> pd.DataFrame:
    """Compares in-window optimization fit (``t+0 (In-Win)``) against out-of-sample forecast skill
    across per-leadtime columns ``t+1``, ``t+2``, ..., ``t+7``, plus Overfit Drop, Retention Ratio,
    and ``% Basins Overfit`` at ``lead_time`` (``t+L``).
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    ss0_col = "NSE Skill Score (t+0 in-window)"
    if ss0_col not in df_eval.columns or df_eval[ss0_col].dropna().empty:
        return pd.DataFrame()

    leads = [int(lt) for lt in detect_lead_times(df_eval) if int(lt) >= 1]
    ref_lt = int(lead_time) if int(lead_time) in leads else leads[0]

    cfgs = sorted(df_eval["Config ID"].dropna().astype(str).unique())
    da_cfgs = [c for c in cfgs if not is_rho_config(c) and "baseline" not in c.lower() and "per-basin" not in c.lower()]
    eval_cfgs = da_cfgs + ([PER_BASIN_DA] if PER_BASIN_DA in cfgs else [])

    if common_basins and da_cfgs:
        b_sets = []
        for c in da_cfgs:
            sub = df_eval[(df_eval["Config ID"] == c) & (df_eval["Lead Time (Days)"] == ref_lt)]
            b_sets.append(set(sub.dropna(subset=[ss0_col, "NSE Skill Score"])["Basin ID"]))
        common = set.intersection(*b_sets) if b_sets else set()
    else:
        common = set(df_eval["Basin ID"].dropna().unique())

    sel = get_active()
    g_da = sel.global_da if sel is not None else None
    rows = []

    for cid in eval_cfgs:
        sub = df_eval[df_eval["Config ID"] == cid]
        if common:
            sub = sub[sub["Basin ID"].isin(common)]
        if sub.empty:
            continue
        hp = parse_da_config_id(cid)
        s_t0 = sub[sub["Lead Time (Days)"] == 1].set_index("Basin ID")[ss0_col].dropna()
        piv = sub.pivot_table(index="Basin ID", columns="Lead Time (Days)", values="NSE Skill Score", aggfunc="first").reindex(columns=leads)
        paired = piv.join(s_t0.rename("t0"), how="inner").dropna(subset=["t0", ref_lt])
        if paired.empty:
            continue

        pos_t0 = paired[paired["t0"] > 0.01]
        ret_lt = float((pos_t0[ref_lt] / pos_t0["t0"]).median()) if len(pos_t0) else np.nan
        overfit_harm_lt = float(((paired["t0"] > 0.05) & (paired[ref_lt] < -0.01)).mean() * 100.0)

        row: Dict[str, Any] = {
            "Config ID": f"★ {cid}" if cid == g_da else cid,
            "Target": hp.get("Target", "-"),
            "Window": hp.get("Window (Days)", np.nan),
            "LR": hp.get("Learning Rate", np.nan),
            "N Basins": len(paired),
            "t+0 (In-Win)": float(paired["t0"].median()),
        }
        for lt in leads:
            row[f"t+{lt}"] = float(paired[lt].median()) if lt in paired.columns else np.nan
        row[f"Overfit Drop (t+0→t+{ref_lt})"] = float((paired["t0"] - paired[ref_lt]).median())
        row[f"Retention (t+{ref_lt}/t+0)"] = ret_lt
        row[f"% Overfit (t+0>5% & t+{ref_lt}<-1%)"] = overfit_harm_lt
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_col = f"t+{ref_lt}"
    if not out.empty and sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out.attrs["selection"] = sel.label() if sel is not None else "default t+1"
    out.attrs["lead_time"] = ref_lt
    out.attrs["leads"] = leads
    return out


def style_assimilation_overfitting_table(df: pd.DataFrame) -> Any:
    if df.empty:
        return df
    ref_lt = df.attrs.get("lead_time", 1)
    leads = df.attrs.get("leads", [1, 2, 3, 4, 5, 6, 7])
    ss_cols = ["t+0 (In-Win)"] + [f"t+{lt}" for lt in leads] + [f"Overfit Drop (t+0→t+{ref_lt})"]
    fmt: Dict[str, Any] = {
        "Window": lambda v: "-" if pd.isna(v) else f"{int(v)}",
        "LR": lambda v: "-" if pd.isna(v) else f"{v:g}",
        f"Retention (t+{ref_lt}/t+0)": "{:.2f}×",
        f"% Overfit (t+0>5% & t+{ref_lt}<-1%)": "{:.1f}%",
        "N Basins": "{:,d}",
    }
    for c in ss_cols:
        if c in df.columns:
            fmt[c] = units.ss_formatter()
    sty = df.style.format(fmt, na_rep="–").hide(axis="index")
    sty = sty.set_caption(
        f"Assimilation Overfitting vs. Out-of-Sample Generalization Across Lead Times (t+0 In-Window vs. t+1..t+7) | "
        f"Diagnostic columns evaluated at Lead t+{ref_lt} | ★ = Global-Best DA ({df.attrs.get('selection', '')})"
    )
    return sty


def plot_assimilation_overfitting(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    figsize: Tuple[float, float] = (15.5, 6.0),
) -> plt.Figure:
    """2-panel figure diagnosing assimilation overfitting vs forecast generalization:
      Panel A: In-window SS(t+0) vs Out-of-Sample Forecast SS(t+L) at ``lead_time`` across DA configurations.
      Panel B: Complete skill trajectory from In-Window (t+0) through Forecast Leads t+1..t+7.
    """
    tbl = build_assimilation_overfitting_table(df_eval, lead_time=lead_time)
    if tbl.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No t+0 in-window metrics available", ha="center", transform=ax.transAxes)
        return fig

    ref_lt = tbl.attrs.get("lead_time", lead_time)
    y_col = f"t+{ref_lt}"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sweep_tbl = tbl[~tbl["Config ID"].str.contains("Per-Basin", na=False)].copy()
    target_colors = {
        "Both Embeddings": "#1a73e8",
        "All Embeddings": "#9334e6",
        "Dual Dynamic Embeddings": "#e8710a",
        "Dynamic Embeddings": "#c5221f",
        "Static Embeddings": "#1e8e3e",
    }
    for tgt, grp in sweep_tbl.groupby("Target"):
        c = target_colors.get(str(tgt), "#5f6368")
        ax1.scatter(
            grp["t+0 (In-Win)"], grp[y_col],
            s=95, color=c, alpha=0.88, edgecolors="white", linewidth=0.9, label=str(tgt), zorder=3,
        )
    sorted_pts = sweep_tbl.sort_values(y_col, ascending=True).reset_index(drop=True)
    for pt_idx, r in sorted_pts.iterrows():
        cid_clean = str(r["Config ID"]).replace("★ ", "").replace("embedded_", "")
        parts = cid_clean.split("_")
        short = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else cid_clean[:12]
        if "★" in str(r["Config ID"]):
            short = f"★ {short}"
        side = -1 if (pt_idx % 2 == 1) else 1
        dx = 7 * side
        dy = -5 if (pt_idx % 3 == 0) else (4 if (pt_idx % 3 == 1) else 0)
        ha = "left" if dx >= 0 else "right"
        ax1.annotate(
            short, (r["t+0 (In-Win)"], r[y_col]),
            xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8.5, color="#202124",
        )

    ax1.axhline(0.0, color="#80868b", ls="--", lw=1.1)
    ax1.axvline(0.0, color="#80868b", ls="--", lw=1.1)
    units.ss_axis(ax1, "both")
    ax1.set_xlabel(units.ss_label("In-Window Assimilation Skill SS_NSE (t+0)"), fontsize=11)
    ax1.set_ylabel(units.ss_label(f"Out-of-Sample Forecast Skill SS_NSE (Lead t+{ref_lt})"), fontsize=11)
    ax1.set_title(f"A. In-Window Fit (t+0) vs. Forecast Generalization (Lead t+{ref_lt})", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", frameon=True, fontsize=9.2, title="Optimized Target")
    ax1.grid(True, alpha=0.3)

    leads = tbl.attrs.get("leads", [1, 2, 3, 4, 5, 6, 7])
    x_ticks = [0] + list(leads)
    sel = get_active()
    g_da = sel.global_da if sel is not None else None

    ax2.axvline(ref_lt, color="#e8eaed", lw=6.0, zorder=0)
    for _, r in sweep_tbl.iterrows():
        cid = str(r["Config ID"]).replace("★ ", "")
        tgt = str(r["Target"])
        vals = [float(r["t+0 (In-Win)"])] + [float(r[f"t+{lt}"]) for lt in leads]
        c = target_colors.get(tgt, "#5f6368")
        is_gb = (cid == g_da)
        lw = 2.8 if is_gb else 1.4
        alpha = 1.0 if is_gb else 0.55
        lbl = f"Global-Best DA ({cid[:18]}..)" if is_gb else None
        ax2.plot(x_ticks, vals, color=c, lw=lw, alpha=alpha, marker="o", ms=5 if is_gb else 3.5, label=lbl)

    pb_rows = tbl[tbl["Config ID"].str.contains("Per-Basin", na=False)]
    if not pb_rows.empty:
        r_pb = pb_rows.iloc[0]
        vals_pb = [float(r_pb["t+0 (In-Win)"])] + [float(r_pb[f"t+{lt}"]) for lt in leads]
        ax2.plot(x_ticks, vals_pb, color=palette.color("per_basin_da"), lw=2.6, ls="--", marker="D", ms=5.5, label="Per-Basin DA (Oracle)")

    ax2.axhline(0.0, color="#80868b", ls="--", lw=1.1)
    ax2.axvline(0.5, color="#d93025", ls=":", lw=1.3, label="Assimilation | Forecast Boundary")
    units.ss_axis(ax2, "y")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(["t+0\n(In-Win)"] + [f"t+{l}" for l in leads])
    ax2.set_xlabel("Horizon Step (In-Window t+0 → Causal Forecast t+1..t+7)", fontsize=11)
    ax2.set_ylabel(units.ss_label("Median NSE Skill Score"), fontsize=11)
    ax2.set_title("B. Skill Retention Trajectory Across Forecast Horizon", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9.2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# 4. Offline Zarr Timeseries Cache Builder (Yearly CV, Model Autocorr, Peaks/Exceedance)
# =============================================================================
def _decode_zarr_dates(g) -> pd.DatetimeIndex:
    u = dict(g["date"].attrs).get("units", "days since 1970-01-01")
    raw = g["date"][:]
    origin = pd.Timestamp(u.split("since", 1)[1].strip()) if "since" in u else pd.Timestamp("1970-01-01")
    step = u.split("since", 1)[0].strip() if "since" in u else "days"
    unit = {"days": "D", "hours": "h", "seconds": "s", "minutes": "min"}.get(step, "D")
    return pd.DatetimeIndex(origin + pd.to_timedelta(raw.astype("int64"), unit=unit))


def _fast_corr_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=1)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma = a0.sum(axis=1) / n
        mb = b0.sum(axis=1) / n
        va = (a0 * a0).sum(axis=1) / n - ma * ma
        vb = (b0 * b0).sum(axis=1) / n - mb * mb
        cov = (a0 * b0).sum(axis=1) / n - ma * mb
        denom = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
        return np.where((n >= 10) & (denom > 1e-12), cov / denom, np.nan)


def _fast_corr_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(axis=1)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ma = a0.sum(axis=1) / n
        mb = b0.sum(axis=1) / n
        va = (a0 * a0).sum(axis=1) / n - ma * ma
        vb = (b0 * b0).sum(axis=1) / n - mb * mb
        cov = (a0 * b0).sum(axis=1) / n - ma * mb
        denom = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
        return np.where((n >= 10) & (denom > 1e-12), cov / denom, np.nan)


def _compute_model_autocorr_for_shard(
    cfg_id: str,
    basins: List[str],
    sim: np.ndarray,
    obs: np.ndarray,
    nse_pers: Optional[np.ndarray] = None,
    kge_pers: Optional[np.ndarray] = None,
    nse_base: Optional[np.ndarray] = None,
    ss_pers: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Computes per-basin, per-leadtime streamflow and residual autocorrelations for ``cfg_id``."""
    n_b = len(basins)
    leads = [1, 2, 3, 4, 5, 6, 7]
    sim_l = sim[:, :, 1:8]
    obs_l = obs[:, :, 1:8]
    obs_0_3d = np.repeat(obs[:, :, 0:1], 7, axis=2)
    sim_0_3d = np.repeat(sim[:, :, 0:1], 7, axis=2)

    is_obs = (cfg_id == "Observed")
    series_l = obs_l if is_obs else sim_l
    err_l = obs_l - sim_l

    flow_lag1 = _fast_corr_3d(series_l[:, :-1, :], series_l[:, 1:, :])
    issue_to_lead = _fast_corr_3d(series_l, obs_0_3d)
    nowcast_to_lead = issue_to_lead if is_obs else _fast_corr_3d(series_l, sim_0_3d)
    res_lag1 = _fast_corr_3d(err_l[:, :-1, :], err_l[:, 1:, :])

    flow_lagL = np.full((n_b, 7), np.nan, dtype=np.float64)
    res_lagL = np.full((n_b, 7), np.nan, dtype=np.float64)
    err_0 = obs[:, :, 0] - sim[:, :, 0]
    for k_idx, L in enumerate(leads):
        flow_lagL[:, k_idx] = _fast_corr_2d(series_l[:, :-L, k_idx], series_l[:, L:, k_idx])
        if is_obs:
            res_lagL[:, k_idx] = _fast_corr_2d(err_l[:, :, k_idx], err_0)
        else:
            res_lagL[:, k_idx] = _fast_corr_2d(err_l[:, :-L, k_idx], err_l[:, L:, k_idx])

    basins_rep = np.repeat(np.asarray(basins, dtype=object), 7)
    leads_tile = np.tile(np.asarray(leads, dtype=np.int32), n_b)
    out = pd.DataFrame({
        "Basin ID": basins_rep,
        "Config ID": cfg_id,
        "Lead Time (Days)": leads_tile,
        "Flow Autocorr Lag-L": flow_lagL.reshape(-1).astype(np.float32),
        "Flow Autocorr Lag-1": flow_lag1.reshape(-1).astype(np.float32),
        "Issue-to-Lead Flow Corr": issue_to_lead.reshape(-1).astype(np.float32),
        "Nowcast-to-Lead Flow Corr": nowcast_to_lead.reshape(-1).astype(np.float32),
        "Residual Autocorr Lag-L": res_lagL.reshape(-1).astype(np.float32),
        "Residual Autocorr Lag-1": res_lag1.reshape(-1).astype(np.float32),
    })
    if nse_pers is not None:
        out["Persistence NSE"] = nse_pers.reshape(-1).astype(np.float32)
        out["Persistence KGE"] = kge_pers.reshape(-1).astype(np.float32)
        out["Base NSE"] = nse_base.reshape(-1).astype(np.float32)
        out["Persistence NSE Skill Score"] = ss_pers.reshape(-1).astype(np.float32)
        out["Obs Autocorrelation"] = flow_lagL.reshape(-1).astype(np.float32)
        out["Residual Autocorrelation"] = res_lagL.reshape(-1).astype(np.float32)
    return out


def _detect_basin_peaks(obs_0: np.ndarray, q90: float, min_sep: int = 5, half_win: int = 2, min_idx: int = 9, max_margin: int = 8) -> np.ndarray:
    d_len = len(obs_0)
    if not np.isfinite(q90) or q90 <= 0 or d_len <= min_idx + max_margin:
        return np.empty(0, dtype=np.int64)
    cand = np.where(np.isfinite(obs_0) & (obs_0 >= q90))[0]
    cand = cand[(cand >= min_idx) & (cand < d_len - max_margin)]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    valid_peaks = []
    for idx in cand:
        win = obs_0[idx - half_win: idx + half_win + 1]
        if np.isfinite(win).all() and obs_0[idx] >= np.max(win):
            valid_peaks.append(int(idx))
    if not valid_peaks:
        return np.empty(0, dtype=np.int64)
    peaks = np.asarray(valid_peaks, dtype=np.int64)
    order = np.argsort(-obs_0[peaks])
    selected = []
    for p in peaks[order]:
        if all(abs(int(p) - s) >= min_sep for s in selected):
            selected.append(int(p))
    return np.asarray(sorted(selected), dtype=np.int64)


def _compute_config_event_and_yearly_for_shard(
    cfg_id: str,
    basins: List[str],
    sim: np.ndarray,
    obs: np.ndarray,
    years_arr: np.ndarray,
    unique_years: List[int],
    q90: np.ndarray,
    q95: np.ndarray,
    q98: np.ndarray,
    basin_peaks: List[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized computation of (yearly_sufficient_stats_df, event_peak_df) for one config on one shard."""
    n_b, _, _ = sim.shape
    leads = [1, 2, 3, 4, 5, 6, 7]
    sim_l = sim[:, :, 1:8]
    obs_l = obs[:, :, 1:8]
    valid = np.isfinite(sim_l) & np.isfinite(obs_l)

    yearly_frames = []
    basins_rep = np.repeat(np.asarray(basins, dtype=object), len(leads))
    leads_tile = np.tile(np.asarray(leads, dtype=np.int64), n_b)
    for y in unique_years:
        ymask = (years_arr == y)
        m_y = valid[:, ymask, :]
        o_y = np.where(m_y, obs_l[:, ymask, :], 0.0)
        s_y = np.where(m_y, sim_l[:, ymask, :], 0.0)
        n_cnt = m_y.sum(axis=1).reshape(-1)
        so = o_y.sum(axis=1).reshape(-1)
        soo = (o_y * o_y).sum(axis=1).reshape(-1)
        ss = s_y.sum(axis=1).reshape(-1)
        sss = (s_y * s_y).sum(axis=1).reshape(-1)
        sos = (o_y * s_y).sum(axis=1).reshape(-1)
        yearly_frames.append(pd.DataFrame({
            "Basin ID": basins_rep,
            "Config ID": cfg_id,
            "Year": int(y),
            "Lead Time (Days)": leads_tile,
            "n": n_cnt.astype(np.int32),
            "so": so.astype(np.float64),
            "soo": soo.astype(np.float64),
            "ss": ss.astype(np.float64),
            "sss": sss.astype(np.float64),
            "sos": sos.astype(np.float64),
        }))
    df_yearly = pd.concat(yearly_frames, ignore_index=True)

    q90_3d = q90[:, None, None]
    q95_3d = q95[:, None, None]
    q98_3d = q98[:, None, None]

    obs_e90 = valid & (obs_l >= q90_3d) & (q90_3d > 0)
    sim_e90 = valid & (sim_l >= q90_3d) & (q90_3d > 0)
    tp90 = (obs_e90 & sim_e90).sum(axis=1)
    fp90 = (~obs_e90 & sim_e90).sum(axis=1)
    fn90 = (obs_e90 & ~sim_e90).sum(axis=1)

    obs_e95 = valid & (obs_l >= q95_3d) & (q95_3d > 0)
    sim_e95 = valid & (sim_l >= q95_3d) & (q95_3d > 0)
    tp95 = (obs_e95 & sim_e95).sum(axis=1)
    fp95 = (~obs_e95 & sim_e95).sum(axis=1)
    fn95 = (obs_e95 & ~sim_e95).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        prec90 = np.where(tp90 + fp90 > 0, tp90 / (tp90 + fp90), np.nan)
        rec90 = np.where(tp90 + fn90 > 0, tp90 / (tp90 + fn90), np.nan)
        far90 = np.where(tp90 + fp90 > 0, fp90 / (tp90 + fp90), np.nan)
        csi90 = np.where(tp90 + fp90 + fn90 > 0, tp90 / (tp90 + fp90 + fn90), np.nan)
        f1_90 = np.where(2 * tp90 + fp90 + fn90 > 0, 2 * tp90 / (2 * tp90 + fp90 + fn90), np.nan)
        zero_alarm90 = (tp90 + fp90 == 0) & (tp90 + fn90 > 0)

        prec95 = np.where(tp95 + fp95 > 0, tp95 / (tp95 + fp95), np.nan)
        rec95 = np.where(tp95 + fn95 > 0, tp95 / (tp95 + fn95), np.nan)
        far95 = np.where(tp95 + fp95 > 0, fp95 / (tp95 + fp95), np.nan)
        csi95 = np.where(tp95 + fp95 + fn95 > 0, tp95 / (tp95 + fp95 + fn95), np.nan)
        f1_95 = np.where(2 * tp95 + fp95 + fn95 > 0, 2 * tp95 / (2 * tp95 + fp95 + fn95), np.nan)
        zero_alarm95 = (tp95 + fp95 == 0) & (tp95 + fn95 > 0)

        m98 = valid & (obs_l >= q98_3d) & (q98_3d > 0)
        sum_o98 = np.where(m98, obs_l, 0.0).sum(axis=1)
        sum_diff98 = np.where(m98, sim_l - obs_l, 0.0).sum(axis=1)
        fhv = np.where(sum_o98 > 0, sum_diff98 / sum_o98, np.nan)
        fhv_abs = np.abs(fhv)

    n_peaks_arr = np.zeros((n_b, 7), dtype=np.int32)
    pte_signed = np.full((n_b, 7), np.nan, dtype=np.float64)
    pte_abs = np.full((n_b, 7), np.nan, dtype=np.float64)
    exact_pct = np.full((n_b, 7), np.nan, dtype=np.float64)
    pte_phase_signed = np.full((n_b, 7), np.nan, dtype=np.float64)
    pte_phase_abs = np.full((n_b, 7), np.nan, dtype=np.float64)
    pfe_signed = np.full((n_b, 7), np.nan, dtype=np.float64)
    pfe_abs = np.full((n_b, 7), np.nan, dtype=np.float64)

    offsets_w2 = np.arange(-2, 3, dtype=np.int64)
    offsets_w7 = np.arange(-2, 8, dtype=np.int64)

    for b_idx in range(n_b):
        T_peaks = basin_peaks[b_idx]
        if T_peaks.size == 0:
            continue
        obs_peak_vals = obs[b_idx, T_peaks, 0]
        for k_idx, lead in enumerate(leads):
            tp_idx = T_peaks - lead
            win2_idx = tp_idx[:, None] + offsets_w2[None, :]
            sim_w2 = sim_l[b_idx, win2_idx, k_idx]
            ok2 = np.isfinite(sim_w2).all(axis=1) & np.isfinite(obs_peak_vals) & (obs_peak_vals > 0)
            if ok2.any():
                sw2 = sim_w2[ok2]
                op = obs_peak_vals[ok2]
                argmax2 = np.argmax(sw2, axis=1)
                dt2 = offsets_w2[argmax2].astype(np.float64)
                sp2 = sw2[np.arange(len(op)), argmax2]
                rel_err = (sp2 - op) / op

                n_peaks_arr[b_idx, k_idx] = int(ok2.sum())
                pte_signed[b_idx, k_idx] = float(np.mean(dt2))
                pte_abs[b_idx, k_idx] = float(np.mean(np.abs(dt2)))
                exact_pct[b_idx, k_idx] = float(np.mean(dt2 == 0) * 100.0)
                pfe_signed[b_idx, k_idx] = float(np.median(rel_err))
                pfe_abs[b_idx, k_idx] = float(np.median(np.abs(rel_err)))

            win7_idx = tp_idx[:, None] + offsets_w7[None, :]
            sim_w7 = sim_l[b_idx, win7_idx, k_idx]
            ok7 = np.isfinite(sim_w7).all(axis=1)
            if ok7.any():
                sw7 = sim_w7[ok7]
                dt7 = offsets_w7[np.argmax(sw7, axis=1)].astype(np.float64)
                pte_phase_signed[b_idx, k_idx] = float(np.mean(dt7))
                pte_phase_abs[b_idx, k_idx] = float(np.mean(np.abs(dt7)))

    df_event = pd.DataFrame({
        "Basin ID": basins_rep,
        "Config ID": cfg_id,
        "Lead Time (Days)": leads_tile,
        "TP_Q90": tp90.reshape(-1).astype(np.int32),
        "FP_Q90": fp90.reshape(-1).astype(np.int32),
        "FN_Q90": fn90.reshape(-1).astype(np.int32),
        "Precision (Q90)": prec90.reshape(-1).astype(np.float32),
        "Recall (Q90)": rec90.reshape(-1).astype(np.float32),
        "FAR (Q90)": far90.reshape(-1).astype(np.float32),
        "CSI (Q90)": csi90.reshape(-1).astype(np.float32),
        "F1 (Q90)": f1_90.reshape(-1).astype(np.float32),
        "ZeroAlarm (Q90)": zero_alarm90.reshape(-1),
        "TP_Q95": tp95.reshape(-1).astype(np.int32),
        "FP_Q95": fp95.reshape(-1).astype(np.int32),
        "FN_Q95": fn95.reshape(-1).astype(np.int32),
        "Precision (Q95)": prec95.reshape(-1).astype(np.float32),
        "Recall (Q95)": rec95.reshape(-1).astype(np.float32),
        "FAR (Q95)": far95.reshape(-1).astype(np.float32),
        "CSI (Q95)": csi95.reshape(-1).astype(np.float32),
        "F1 (Q95)": f1_95.reshape(-1).astype(np.float32),
        "ZeroAlarm (Q95)": zero_alarm95.reshape(-1),
        "N Peaks": n_peaks_arr.reshape(-1),
        "PTE Signed (Days)": pte_signed.reshape(-1).astype(np.float32),
        "PTE Abs (Days)": pte_abs.reshape(-1).astype(np.float32),
        "Exact Peak Timing (%)": exact_pct.reshape(-1).astype(np.float32),
        "PTE Phase Signed (Days)": pte_phase_signed.reshape(-1).astype(np.float32),
        "PTE Phase Abs (Days)": pte_phase_abs.reshape(-1).astype(np.float32),
        "PFE Signed": pfe_signed.reshape(-1).astype(np.float32),
        "PFE Abs": pfe_abs.reshape(-1).astype(np.float32),
        "FHV": fhv.reshape(-1).astype(np.float32),
        "FHV Abs": fhv_abs.reshape(-1).astype(np.float32),
    })
    return df_yearly, df_event


def build_or_load_zarr_caches(
    data_dir: str | Path,
    rhos: Sequence[float] = DEFAULT_RHOS,
    baseline_cfg: str = "E_baseline_0da",
    overwrite: bool = False,
    workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Builds (or loads from ``<data_dir>/_ts_cache/``) the three timeseries caches:
      - ``yearly_stats``: per-(Basin ID, Config ID, Year, Lead) additive sufficient statistics
      - ``persistence``: per-(Basin ID, Config ID, Lead) Persistence NSE/KGE and model streamflow/residual autocorrelations
      - ``event_peaks``: per-(Basin ID, Config ID, Lead) Q90/Q95 Precision/Recall/CSI/F1, PTE, PFE, and FHV.
    """
    import zarr

    data_dir = Path(data_dir)
    cache_dir = data_dir / "_ts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p_yearly = cache_dir / YEARLY_STATS_CACHE
    p_pers = cache_dir / PERSISTENCE_CACHE
    p_pers_v1 = cache_dir / PERSISTENCE_CACHE_V1
    p_event = cache_dir / EVENT_PEAK_CACHE

    if not overwrite and p_yearly.exists() and p_event.exists() and (p_pers.exists() or p_pers_v1.exists()):
        pers_path = p_pers if p_pers.exists() else p_pers_v1
        return {
            "yearly_stats": pd.read_parquet(p_yearly),
            "persistence": pd.read_parquet(pers_path),
            "event_peaks": pd.read_parquet(p_event),
        }

    base_stores = sorted(glob.glob(str(data_dir / baseline_cfg / "shard_*" / "test" / "model_epoch*" / "test_results*.zarr")))
    if not base_stores:
        base_stores = sorted(glob.glob(str(data_dir / baseline_cfg / "test" / "model_epoch*" / "test_results*.zarr")))
    if not base_stores:
        print(f"[EXTENDED METRICS] No Baseline Zarr stores found in {data_dir / baseline_cfg}.")
        return {"yearly_stats": pd.DataFrame(), "persistence": pd.DataFrame(), "event_peaks": pd.DataFrame()}

    all_cfg_dirs = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name != baseline_cfg and not d.name.startswith("_") and not is_rho_config(d.name)
        and glob.glob(str(d / "**" / "test_results*.zarr"), recursive=True)
    ])
    print(f"[EXTENDED METRICS] Computing yearly CV, model autocorrelation, and peak/exceedance caches "
          f"across {len(base_stores)} shards x (Observed + Baseline + {len(all_cfg_dirs)} DA + {len(rhos)} PP configs)...")

    yearly_all, pers_all, event_all = [], [], []

    for b_store in base_stores:
        rel_dir = os.path.relpath(os.path.dirname(b_store), str(data_dir / baseline_cfg))
        g_base = zarr.open_consolidated(b_store, mode="r")
        basins = [str(b) for b in g_base["basin"][:]]
        ts = [int(t) for t in g_base["time_step"][:]]
        dates = _decode_zarr_dates(g_base)
        years_arr = dates.year.to_numpy(np.int32)
        unique_years = sorted(int(y) for y in np.unique(years_arr))

        sim_base = np.asarray(g_base["streamflow_sim"][:, 0, :, :, 0], dtype=np.float64)
        obs = np.asarray(g_base["streamflow_obs"][:, 0, :, :], dtype=np.float64)

        with np.errstate(all="ignore"):
            q90, q95, q98 = np.nanquantile(obs[:, :, 0], [0.90, 0.95, 0.98], axis=1)
        basin_peaks = [_detect_basin_peaks(obs[i, :, 0], float(q90[i])) for i in range(len(basins))]

        # Persistence NSE & KGE
        obs_l = obs[:, :, 1:8]
        pers_l = np.repeat(obs[:, :, 0:1], 7, axis=2)
        sim_b_l = sim_base[:, :, 1:8]
        m_p = np.isfinite(obs_l) & np.isfinite(pers_l)
        n_p = m_p.sum(axis=1)
        o_p = np.where(m_p, obs_l, 0.0)
        s_p = np.where(m_p, pers_l, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_o = o_p.sum(axis=1) / n_p
            mu_s = s_p.sum(axis=1) / n_p
            sst = (o_p * o_p).sum(axis=1) - n_p * (mu_o ** 2)
            sse = np.where(m_p, (obs_l - pers_l) ** 2, 0.0).sum(axis=1)
            nse_pers = np.where((n_p >= 2) & (sst > 0), 1.0 - sse / sst, np.nan)
            var_o = sst / n_p
            var_s = ((s_p * s_p).sum(axis=1) / n_p) - (mu_s ** 2)
            cov_os = ((o_p * s_p).sum(axis=1) / n_p) - (mu_o * mu_s)
            std_o = np.sqrt(np.maximum(var_o, 0.0))
            std_s = np.sqrt(np.maximum(var_s, 0.0))
            rho_q = np.where((n_p >= 5) & (std_o > 0) & (std_s > 0), cov_os / (std_o * std_s), np.nan)
            alpha_p = np.where((n_p >= 2) & (std_o > 0), std_s / std_o, np.nan)
            beta_p = np.where((n_p >= 2) & (np.abs(mu_o) > 0), mu_s / mu_o, np.nan)
            kge_pers = 1.0 - np.sqrt((rho_q - 1.0) ** 2 + (alpha_p - 1.0) ** 2 + (beta_p - 1.0) ** 2)

            m_b = np.isfinite(obs_l) & np.isfinite(sim_b_l)
            n_b_cnt = m_b.sum(axis=1)
            o_b = np.where(m_b, obs_l, 0.0)
            mu_ob = o_b.sum(axis=1) / n_b_cnt
            sst_b = (o_b * o_b).sum(axis=1) - n_b_cnt * (mu_ob ** 2)
            sse_b = np.where(m_b, (obs_l - sim_b_l) ** 2, 0.0).sum(axis=1)
            nse_base = np.where((n_b_cnt >= 2) & (sst_b > 0), 1.0 - sse_b / sst_b, np.nan)
            ss_pers = np.where(np.abs(1.0 - nse_base) > 1e-9, (nse_pers - nse_base) / (1.0 - nse_base), np.nan)

        pers_all.append(_compute_model_autocorr_for_shard("Observed", basins, sim_base, obs, nse_pers, kge_pers, nse_base, ss_pers))
        pers_all.append(_compute_model_autocorr_for_shard("Baseline", basins, sim_base, obs))

        y_base, e_base = _compute_config_event_and_yearly_for_shard(
            "Baseline", basins, sim_base, obs, years_arr, unique_years, q90, q95, q98, basin_peaks
        )
        yearly_all.append(y_base)
        event_all.append(e_base)

        for rho in rhos:
            sim_ar1 = ar1_from_baseline(sim_base, obs, ts, rho)
            cname = ar1_config_name(rho)
            pers_all.append(_compute_model_autocorr_for_shard(cname, basins, sim_ar1, obs))
            y_ar1, e_ar1 = _compute_config_event_and_yearly_for_shard(
                cname, basins, sim_ar1, obs, years_arr, unique_years, q90, q95, q98, basin_peaks
            )
            yearly_all.append(y_ar1)
            event_all.append(e_ar1)

        def _proc_da(cfg_name: str):
            z_cands = glob.glob(str(data_dir / cfg_name / rel_dir / "test_results*.zarr"))
            if not z_cands:
                return None
            g_da = zarr.open_consolidated(z_cands[0], mode="r")
            da_basins = [str(b) for b in g_da["basin"][:]]
            sim_da = np.asarray(g_da["streamflow_sim"][:, 0, :, :, 0], dtype=np.float64)
            if da_basins != basins:
                idx_map = {b: i for i, b in enumerate(da_basins)}
                perm = [idx_map[b] for b in basins if b in idx_map]
                if len(perm) != len(basins):
                    return None
                sim_da = sim_da[perm]
            p_da = _compute_model_autocorr_for_shard(cfg_name, basins, sim_da, obs)
            y_da, e_da = _compute_config_event_and_yearly_for_shard(
                cfg_name, basins, sim_da, obs, years_arr, unique_years, q90, q95, q98, basin_peaks
            )
            return p_da, y_da, e_da

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(all_cfg_dirs)))) as ex:
            for res in ex.map(_proc_da, all_cfg_dirs):
                if res is not None:
                    pers_all.append(res[0])
                    yearly_all.append(res[1])
                    event_all.append(res[2])

    df_yearly = pd.concat(yearly_all, ignore_index=True)
    df_pers = pd.concat(pers_all, ignore_index=True)
    df_event = pd.concat(event_all, ignore_index=True)

    base_ev = df_event[df_event["Config ID"] == "Baseline"].copy()
    unit_opt_metrics = ("Precision (Q90)", "Recall (Q90)", "CSI (Q90)", "F1 (Q90)",
                        "Precision (Q95)", "Recall (Q95)", "CSI (Q95)", "F1 (Q95)")
    zero_opt_metrics = ("PTE Abs (Days)", "PTE Phase Abs (Days)", "PFE Abs", "FHV Abs", "FAR (Q90)", "FAR (Q95)")
    raw_passthrough = ("PTE Signed (Days)", "PTE Phase Signed (Days)", "PFE Signed", "FHV", "Exact Peak Timing (%)")

    all_m = unit_opt_metrics + zero_opt_metrics + raw_passthrough
    base_keep = ["Basin ID", "Lead Time (Days)"] + [f"Base {m}" for m in all_m]
    base_ev = base_ev.rename(columns={m: f"Base {m}" for m in all_m})[base_keep].drop_duplicates(["Basin ID", "Lead Time (Days)"])
    df_event = df_event.merge(base_ev, on=["Basin ID", "Lead Time (Days)"], how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        for m in unit_opt_metrics:
            denom = 1.0 - df_event[f"Base {m}"].astype(np.float64)
            ss = np.where(np.abs(denom) > 1e-9, (df_event[m].astype(np.float64) - df_event[f"Base {m}"].astype(np.float64)) / denom, np.nan)
            df_event[f"{m} Skill Score"] = ss.astype(np.float32)
        for m in zero_opt_metrics:
            b_err = np.abs(df_event[f"Base {m}"].astype(np.float64))
            d_err = np.abs(df_event[m].astype(np.float64))
            ss = np.where(b_err > 1e-9, 1.0 - d_err / b_err, np.nan)
            df_event[f"{m} Skill Score"] = ss.astype(np.float32)

    df_yearly.to_parquet(p_yearly, index=False)
    df_pers.to_parquet(p_pers, index=False)
    df_event.to_parquet(p_event, index=False)
    return {"yearly_stats": df_yearly, "persistence": df_pers, "event_peaks": df_event}


# =============================================================================
# 5. Persistence & Model Streamflow / Residual Autocorrelation Suite
# =============================================================================
def _attach_per_basin_autocorr_rows(df_pers: pd.DataFrame) -> pd.DataFrame:
    """Synthesizes ``Per-Basin Best DA`` and ``AR1_Per_Basin_Best_Rho`` rows in ``df_pers``
    and attaches ``Assumed PP Autocorr`` = rho^L for PP post-processing models.
    """
    if df_pers.empty:
        return df_pers
    if "Config ID" not in df_pers.columns:
        df = df_pers.copy()
        df["Config ID"] = "Observed"
        return df

    sel = get_active()
    df = df_pers[~df_pers["Config ID"].isin([PER_BASIN_DA, PER_BASIN_RHO])].copy()

    # Compute Assumed PP Autocorr = rho^L on fixed-rho rows before synthesizing Per-Basin PP
    if "Assumed PP Autocorr" not in df.columns:
        assumed = np.full(len(df), np.nan, dtype=np.float32)
        for cid in df["Config ID"].unique():
            rho_val = parse_ar1_config(str(cid))
            if isinstance(rho_val, float):
                mask = (df["Config ID"] == cid).to_numpy()
                lts = df.loc[mask, "Lead Time (Days)"].to_numpy(np.float32)
                assumed[mask] = np.power(float(rho_val), lts)
        df["Assumed PP Autocorr"] = assumed

    if sel is None:
        return df

    def _synth(mapping: Dict[str, str], name: str) -> pd.DataFrame:
        if not mapping:
            return df.iloc[0:0]
        m = pd.DataFrame({"Basin ID": list(mapping.keys()), "_sel": list(mapping.values())})
        sub = df[df["Config ID"].isin(set(mapping.values()))].merge(m, on="Basin ID")
        sub = sub[sub["Config ID"] == sub["_sel"]].drop(columns=["_sel"])
        sub["Config ID"] = name
        return sub

    extra = [_synth(sel.per_basin_da, PER_BASIN_DA), _synth(sel.per_basin_rho, PER_BASIN_RHO)]
    extra = [e for e in extra if not e.empty]
    return pd.concat([df] + extra, ignore_index=True) if extra else df


def build_persistence_autocorr_table(
    df_eval: pd.DataFrame,
    df_pers: pd.DataFrame,
    lead_times: Optional[Sequence[int]] = None,
    autocorr_type: str = "flow_lagL",
    show: Optional[Sequence[str]] = None,
    strata_view: str = "both",  # "both", "skill", or "autocorr"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the two Persistence & Model Autocorrelation tables with Leadtimes (``t+1..t+7``) as COLUMNS:

      1. ``df_pers_leads``: Columns = ``t+1 .. t+7``.
         Rows =
           - Streamflow Autocorrelation ρ_Q(L) of Observed, Baseline, Global DA, Per-Basin DA, Global PP, Per-Basin PP
           - Issue-Date Flow Persistence Corr(Q̂(t+L|t), Q_obs(t)) of Observed, Baseline, DA, and PP
           - Residual (Error) Autocorrelation ρ_e(L) of Baseline, Assumed PP ρ^L, DA, and PP
           - Forecast Performance (Persistence NSE, Baseline NSE, and SS_NSE of DA and PP models)
      2. ``df_pers_strata``: Stratified by Lag-1 Open-Loop Residual Autocorrelation Quartiles ρ_e(1)
         (``Q1: Low Memory (Flashy)`` .. ``Q4: High Memory (Baseflow)``), with Columns = ``t+1 .. t+7``.
         Shows BOTH the Autocorrelation of Observed, Baseline, DA, and Post-Processing models AND their SS_NSE across leads!
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    df_p = _attach_per_basin_autocorr_rows(df_pers)
    leads = [int(lt) for lt in (lead_times if lead_times is not None else detect_lead_times(df_eval)) if int(lt) >= 1]
    col_names = [f"t+{lt}" for lt in leads]
    refs = _resolve_scheme_configs(df_eval, show=show)

    obs_sub = df_p[df_p["Config ID"] == "Observed"]
    p_valid = obs_sub.pivot_table(index="Basin ID", columns="Lead Time (Days)", values="Persistence NSE").reindex(columns=leads).dropna()
    common = set(p_valid.index)
    for cid in refs.values():
        w = df_eval[df_eval["Config ID"] == cid].pivot_table(index="Basin ID", columns="Lead Time (Days)", values="DA NSE").reindex(columns=leads).dropna()
        if not w.empty:
            common &= set(w.index)

    p_sub = df_p[df_p["Basin ID"].isin(common)]
    ev_sub = df_eval[df_eval["Basin ID"].isin(common)]
    n_common = len(common)

    model_specs = [
        ("Observed", "Observed Streamflow", "Observed"),
        ("baseline", "Baseline (Open-Loop)", "Baseline"),
    ] + [(k, SCHEME_LABELS[k], refs[k]) for k in refs]

    def _med_across_leads(frame: pd.DataFrame, col: str) -> Dict[str, float]:
        if frame.empty or col not in frame.columns:
            return {f"t+{lt}": np.nan for lt in leads}
        piv = frame.groupby("Lead Time (Days)")[col].median()
        return {f"t+{lt}": float(piv.get(lt, np.nan)) for lt in leads}

    lead_rows: List[Dict[str, Any]] = []

    # Block 1: Lag-L Streamflow Autocorrelation rho_Q(L)
    for skey, slabel, cid in model_specs:
        sub_m = p_sub[p_sub["Config ID"] == cid]
        col = "Flow Autocorr Lag-L" if "Flow Autocorr Lag-L" in sub_m.columns else "Obs Autocorrelation"
        lead_rows.append({
            "Category": "1. Streamflow Autocorrelation ρ_Q(L)",
            "Model / Metric": slabel,
            "N Basins": n_common,
            "_fmt": "raw",
            **_med_across_leads(sub_m, col),
        })

    # Block 2: Issue-to-Horizon Flow Persistence Corr(Q_hat(t+L|t), Q_obs(t))
    if "Issue-to-Lead Flow Corr" in p_sub.columns:
        for skey, slabel, cid in model_specs:
            sub_m = p_sub[p_sub["Config ID"] == cid]
            lead_rows.append({
                "Category": "2. Issue-Date Flow Persistence Corr(Q̂(t+L|t), Q_obs(t))",
                "Model / Metric": slabel,
                "N Basins": n_common,
                "_fmt": "raw",
                **_med_across_leads(sub_m, "Issue-to-Lead Flow Corr"),
            })

    # Block 3: Residual (Error) Autocorrelation rho_e(L) & Assumed PP Decay rho^L
    obs_m = p_sub[p_sub["Config ID"] == "Observed"]
    res_col_obs = "Residual Autocorr Lag-L" if "Residual Autocorr Lag-L" in obs_m.columns else "Residual Autocorrelation"
    lead_rows.append({
        "Category": "3. Residual Error Autocorrelation ρ_e(L)",
        "Model / Metric": "Open-Loop Issue-to-Lead Error Corr(e_t+L, e_t)",
        "N Basins": n_common,
        "_fmt": "raw",
        **_med_across_leads(obs_m, res_col_obs),
    })
    for k_ar in ("global_rho", "per_basin_rho"):
        if k_ar in refs:
            sub_ar = p_sub[p_sub["Config ID"] == refs[k_ar]]
            if "Assumed PP Autocorr" in sub_ar.columns:
                lead_rows.append({
                    "Category": "3. Residual Error Autocorrelation ρ_e(L)",
                    "Model / Metric": f"{SCHEME_LABELS[k_ar]} Assumed Decay (ρ^L)",
                    "N Basins": n_common,
                    "_fmt": "raw",
                    **_med_across_leads(sub_ar, "Assumed PP Autocorr"),
                })
    for skey, slabel, cid in model_specs:
        if skey == "Observed":
            continue
        sub_m = p_sub[p_sub["Config ID"] == cid]
        if "Residual Autocorr Lag-L" in sub_m.columns:
            lead_rows.append({
                "Category": "3. Residual Error Autocorrelation ρ_e(L)",
                "Model / Metric": f"{slabel} Remaining Error ρ_e(L)",
                "N Basins": n_common,
                "_fmt": "raw",
                **_med_across_leads(sub_m, "Residual Autocorr Lag-L"),
            })

    # Block 4: Forecast NSE & Skill Score SS_NSE
    lead_rows.append({
        "Category": "4. Forecast Skill (NSE & SS_NSE)",
        "Model / Metric": "Naive Persistence Q(t) (Median NSE)",
        "N Basins": n_common,
        "_fmt": "raw",
        **_med_across_leads(obs_m, "Persistence NSE"),
    })
    lead_rows.append({
        "Category": "4. Forecast Skill (NSE & SS_NSE)",
        "Model / Metric": "Baseline Open-Loop (Median NSE)",
        "N Basins": n_common,
        "_fmt": "raw",
        **_med_across_leads(obs_m, "Base NSE"),
    })
    for skey in refs:
        sub_ev = ev_sub[ev_sub["Config ID"] == refs[skey]]
        lead_rows.append({
            "Category": "4. Forecast Skill (NSE & SS_NSE)",
            "Model / Metric": f"{SCHEME_LABELS[skey]} ({units.ss_label('SS_NSE')})",
            "N Basins": n_common,
            "_fmt": "ss",
            **_med_across_leads(sub_ev, "NSE Skill Score"),
        })

    lead_table = pd.DataFrame(lead_rows)
    lead_table.attrs["n_basins"] = n_common
    lead_table.attrs["col_names"] = col_names

    # --- Table 2: Stratified by Lag-1 Open-Loop Residual Autocorrelation Quartiles rho_e(1), Leadtimes as Columns ---
    ac_col, _, ac_short = AUTOCORR_MODE_SPECS.get(autocorr_type, AUTOCORR_MODE_SPECS["flow_lagL"])
    p1_obs = obs_m[obs_m["Lead Time (Days)"] == 1].set_index("Basin ID")
    res1_col = "Residual Autocorr Lag-L" if "Residual Autocorr Lag-L" in p1_obs.columns else "Residual Autocorrelation"
    obs1_col = "Flow Autocorr Lag-L" if "Flow Autocorr Lag-L" in p1_obs.columns else "Obs Autocorrelation"
    p1 = p1_obs[[obs1_col, res1_col]].rename(columns={obs1_col: "obs1", res1_col: "res1"}).dropna()
    p1["Autocorr Bin"] = pd.qcut(
        p1["res1"], q=4,
        labels=["Q1: Low Memory (Flashy)", "Q2: Moderate-Low", "Q3: Moderate-High", "Q4: High Memory (Baseflow)"],
    )

    strata_rows: List[Dict[str, Any]] = []
    for bname, grp_b in p1.groupby("Autocorr Bin", observed=False):
        b_ids = set(grp_b.index)
        med_re1 = float(grp_b["res1"].median())
        med_rq1 = float(grp_b["obs1"].median())
        n_q = len(b_ids)

        # 1. Skill Score rows in this quartile across t+1..t+7
        if strata_view in ("both", "skill"):
            for skey in refs:
                sub_ev = ev_sub[(ev_sub["Config ID"] == refs[skey]) & (ev_sub["Basin ID"].isin(b_ids))]
                strata_rows.append({
                    "Residual Memory Quartile": str(bname),
                    "Median ρ_e(1)": med_re1,
                    "Median ρ_Q(1)": med_rq1,
                    "Model": SCHEME_LABELS[skey],
                    "Metric": units.ss_label("SS_NSE"),
                    "N Basins": n_q,
                    "_fmt": "ss",
                    **_med_across_leads(sub_ev, "NSE Skill Score"),
                })

        # 2. Model Autocorrelation rows in this quartile across t+1..t+7
        if strata_view in ("both", "autocorr"):
            for skey, slabel, cid in model_specs:
                if ac_col == "Residual Autocorr Lag-1" and skey == "Observed":
                    continue
                sub_m = p_sub[(p_sub["Config ID"] == cid) & (p_sub["Basin ID"].isin(b_ids))]
                if ac_col in sub_m.columns:
                    m_name = "Open-Loop Issue-to-Lead Error Corr" if (ac_col == "Residual Autocorr Lag-L" and skey == "Observed") else slabel
                    strata_rows.append({
                        "Residual Memory Quartile": str(bname),
                        "Median ρ_e(1)": med_re1,
                        "Median ρ_Q(1)": med_rq1,
                        "Model": m_name,
                        "Metric": ac_short,
                        "N Basins": n_q,
                        "_fmt": "raw",
                        **_med_across_leads(sub_m, ac_col),
                    })
                if ac_col == "Residual Autocorr Lag-L" and skey in ("global_rho", "per_basin_rho") and "Assumed PP Autocorr" in sub_m.columns:
                    strata_rows.append({
                        "Residual Memory Quartile": str(bname),
                        "Median ρ_e(1)": med_re1,
                        "Median ρ_Q(1)": med_rq1,
                        "Model": f"{slabel} (Assumed ρ^L)",
                        "Metric": "Assumed ρ^L",
                        "N Basins": n_q,
                        "_fmt": "raw",
                        **_med_across_leads(sub_m, "Assumed PP Autocorr"),
                    })

    strata_table = pd.DataFrame(strata_rows)
    strata_table.attrs["col_names"] = col_names
    strata_table.attrs["autocorr_type"] = autocorr_type
    return lead_table, strata_table


def style_persistence_autocorr_tables(lead_df: pd.DataFrame, strata_df: pd.DataFrame) -> Tuple[Any, Any]:
    """Formats ``lead_df`` and ``strata_df`` (both with ``t+1..t+7`` as columns) using ``units.fmt_ss``."""
    col_names = lead_df.attrs.get("col_names", [c for c in lead_df.columns if c.startswith("t+")])
    disp_lead = lead_df.drop(columns=["_fmt"], errors="ignore").copy()
    for c in col_names:
        if c in disp_lead.columns:
            disp_lead[c] = disp_lead[c].astype(object)
    for idx, r in lead_df.iterrows():
        is_ss = (r.get("_fmt") == "ss")
        for c in col_names:
            val = r[c]
            if pd.isna(val) or not np.isfinite(float(val)):
                disp_lead.at[idx, c] = "–"
            elif is_ss:
                disp_lead.at[idx, c] = units.fmt_ss(float(val))
            else:
                disp_lead.at[idx, c] = f"{float(val):.3f}"

    sty_lead = (
        disp_lead.style.format({"N Basins": "{:,d}"}, na_rep="–")
        .hide(axis="index")
        .set_properties(subset=["Category", "Model / Metric"], **{"text-align": "left"})
        .set_properties(subset=col_names, **{"text-align": "right", "padding": "4px 10px"})
        .set_caption(
            f"Model Autocorrelation (Observed, Baseline, DA & Post-Processing PP) and Persistence Skill Across Lead Times "
            f"(N = {lead_df.attrs.get('n_basins', 0):,} common basins)"
        )
    )

    s_cols = strata_df.attrs.get("col_names", [c for c in strata_df.columns if c.startswith("t+")])
    disp_strata = strata_df.drop(columns=["_fmt"], errors="ignore").copy()
    for c in s_cols:
        if c in disp_strata.columns:
            disp_strata[c] = disp_strata[c].astype(object)
    for idx, r in strata_df.iterrows():
        is_ss = (r.get("_fmt") == "ss")
        for c in s_cols:
            val = r[c]
            if pd.isna(val) or not np.isfinite(float(val)):
                disp_strata.at[idx, c] = "–"
            elif is_ss:
                disp_strata.at[idx, c] = units.fmt_ss(float(val))
            else:
                disp_strata.at[idx, c] = f"{float(val):.3f}"

    sty_strata = (
        disp_strata.style.format({"Median ρ_e(1)": "{:.3f}", "Median ρ_Q(1)": "{:.3f}", "N Basins": "{:,d}"}, na_rep="–")
        .hide(axis="index")
        .set_properties(subset=["Residual Memory Quartile", "Model", "Metric"], **{"text-align": "left"})
        .set_properties(subset=s_cols, **{"text-align": "right", "padding": "4px 10px"})
        .set_caption(
            "DA vs. Post-Processing Forecast Skill (SS_NSE) & Model Autocorrelation Stratified by Lag-1 Open-Loop Residual Autocorrelation Quartiles ρ_e(1)"
        )
    )
    return sty_lead, sty_strata


def plot_persistence_and_autocorrelation(
    df_eval: pd.DataFrame,
    df_pers: pd.DataFrame,
    lead_time: int = 1,
    autocorr_type: str = "flow_lagL",
    show: Optional[Sequence[str]] = None,
    catchment_label: str = "All Catchments",
    figsize: Tuple[float, float] = (17.5, 10.2),
) -> plt.Figure:
    """2x2 Diagnostic Figure exploring Model Autocorrelation (Observed, Baseline, DA, Post-Processing PP)
    and Forecast Skill across Lead Times and Residual Memory Quartiles:

      - Panel A (Top-Left): Autocorrelation trajectory across leads ``t+1..t+7`` for Observed, Baseline,
        DA models, and Post-Processing PP models (plus Assumed PP ρ^L when viewing residual autocorrelation),
        with a vertical highlight at ``lead_time`` (``t+L``).
      - Panel B (Top-Right): Across-basin ECDF of Autocorrelation at the selected ``lead_time`` (``t+L``)
        comparing Observed, Baseline, DA, and Post-Processing models (showing over-smoothing / excess autocorrelation).
      - Panel C (Bottom-Left): Model vs. Observed Autocorrelation at ``lead_time`` (``t+L``) stratified by
        Lag-1 Residual Autocorrelation Quartiles (``Q1: Low Memory`` .. ``Q4: High Memory``).
      - Panel D (Bottom-Right): Forecast Skill Score ``SS_NSE`` at ``lead_time`` (``t+L``) stratified by
        the same Lag-1 Residual Autocorrelation Quartiles (``Q1`` .. ``Q4``).
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    df_p = _attach_per_basin_autocorr_rows(df_pers)
    leads = [int(lt) for lt in detect_lead_times(df_eval) if int(lt) >= 1]
    ref_lt = int(lead_time) if int(lead_time) in leads else leads[0]
    refs = _resolve_scheme_configs(df_eval, show=show)
    ac_col, ac_title, ac_short = AUTOCORR_MODE_SPECS.get(autocorr_type, AUTOCORR_MODE_SPECS["flow_lagL"])

    obs_sub = df_p[df_p["Config ID"] == "Observed"]
    p_valid = obs_sub.pivot_table(index="Basin ID", columns="Lead Time (Days)", values="Persistence NSE").reindex(columns=leads).dropna()
    common = set(p_valid.index)
    for cid in refs.values():
        w = df_eval[df_eval["Config ID"] == cid].pivot_table(index="Basin ID", columns="Lead Time (Days)", values="DA NSE").reindex(columns=leads).dropna()
        if not w.empty:
            common &= set(w.index)

    p_sub = df_p[df_p["Basin ID"].isin(common)]
    ev_sub = df_eval[df_eval["Basin ID"].isin(common)]
    n_b = len(common)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    curve_styles = [
        ("Observed", "Observed Streamflow", "Observed", "#202124", "--", "o"),
        ("baseline", "Open-Loop Baseline", "Baseline", palette.color("baseline"), palette.ls("baseline"), "o"),
        ("global_da", "Global-Best DA", refs.get("global_da"), palette.color("global_da"), palette.ls("global_da"), "s"),
        ("per_basin_da", "Per-Basin DA (Oracle)", refs.get("per_basin_da"), palette.color("per_basin_da"), palette.ls("per_basin_da"), "D"),
        ("global_rho", "Global-Best PP", refs.get("global_rho"), palette.color("global_rho"), palette.ls("global_rho"), "^"),
        ("per_basin_rho", "Per-Basin PP", refs.get("per_basin_rho"), palette.color("per_basin_rho"), palette.ls("per_basin_rho"), "v"),
    ]
    curve_styles = [c for c in curve_styles if c[2] is not None]

    # --- Panel A: Autocorrelation trajectory across leads t+1..t+7 ---
    ax1.axvline(ref_lt, color="#e8eaed", lw=7.0, zorder=0, label=f"Selected Lead t+{ref_lt}")
    if ac_col.startswith("Residual"):
        ax1.axhline(0.0, color="#80868b", lw=1.0, ls=":")
    for skey, slabel, cid, color, ls, mk in curve_styles:
        if ac_col == "Residual Autocorr Lag-1" and skey == "Observed":
            continue
        sub_m = p_sub[p_sub["Config ID"] == cid]
        if ac_col not in sub_m.columns or sub_m.empty:
            continue
        lbl = "Open-Loop Issue-to-Lead Error Corr" if (skey == "Observed" and ac_col == "Residual Autocorr Lag-L") else slabel
        meds = [float(sub_m.loc[sub_m["Lead Time (Days)"] == lt, ac_col].median()) for lt in leads]
        ax1.plot(leads, meds, label=lbl, color=color, ls=ls, marker=mk, lw=2.5 if skey == "Observed" else 2.2, ms=5.5)

    if ac_col == "Residual Autocorr Lag-L":
        for k_ar, l_ar, col_ar in (("global_rho", "Assumed Global PP $\\rho^L$", palette.color("global_rho")),
                                   ("per_basin_rho", "Assumed Per-Basin PP $\\rho_b^L$", palette.color("per_basin_rho"))):
            if k_ar in refs:
                sub_ar = p_sub[p_sub["Config ID"] == refs[k_ar]]
                if "Assumed PP Autocorr" in sub_ar.columns:
                    assumed_meds = [float(sub_ar.loc[sub_ar["Lead Time (Days)"] == lt, "Assumed PP Autocorr"].median()) for lt in leads]
                    ax1.plot(leads, assumed_meds, label=l_ar, color=col_ar, ls=":", lw=2.0, marker="x", ms=5.0)

    ax1.set_xticks(leads)
    ax1.set_xticklabels([f"t+{l}" for l in leads])
    ax1.set_xlabel("Forecast Lead Time (Days)", fontsize=11)
    ax1.set_ylabel(f"Median {ac_short}", fontsize=11)
    ax1.set_title(f"A. {ac_title}\nAcross Leads t+1..t+7 [{catchment_label}, N={n_b:,}]", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="best", frameon=True, fontsize=8.8)
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Across-Basin ECDF of Autocorrelation at Selected Lead t+L ---
    for skey, slabel, cid, color, ls, mk in curve_styles:
        if ac_col == "Residual Autocorr Lag-1" and skey == "Observed":
            continue
        sub_m = p_sub[(p_sub["Config ID"] == cid) & (p_sub["Lead Time (Days)"] == ref_lt)]
        if ac_col not in sub_m.columns or sub_m.empty:
            continue
        vals = np.sort(sub_m[ac_col].dropna().to_numpy(float))
        if len(vals):
            base_lbl = "Open-Loop Error Corr" if (skey == "Observed" and ac_col == "Residual Autocorr Lag-L") else slabel
            lbl = f"{base_lbl} (med={np.median(vals):.3f})"
            ax2.plot(vals, np.linspace(0, 1, len(vals)), label=lbl, color=color, ls=ls, lw=2.5 if skey == "Observed" else 2.2)
    if ac_col == "Residual Autocorr Lag-L" and "per_basin_rho" in refs:
        sub_ar = p_sub[(p_sub["Config ID"] == refs["per_basin_rho"]) & (p_sub["Lead Time (Days)"] == ref_lt)]
        if "Assumed PP Autocorr" in sub_ar.columns:
            vals_ar = np.sort(sub_ar["Assumed PP Autocorr"].dropna().to_numpy(float))
            if len(vals_ar):
                ax2.plot(vals_ar, np.linspace(0, 1, len(vals_ar)), label=f"Assumed Per-Basin PP $\\rho_b^L$ (med={np.median(vals_ar):.3f})", color=palette.color("per_basin_rho"), ls=":", lw=2.0)
    ax2.axhline(0.5, color="#9aa0a6", lw=0.9, ls=":")
    if ac_col.startswith("Residual"):
        ax2.axvline(0.0, color="#80868b", lw=1.0, ls="--")
    ax2.set_xlabel(f"Per-Basin {ac_short} at Lead t+{ref_lt}", fontsize=11)
    ax2.set_ylabel("Fraction of Basins (ECDF)", fontsize=11)
    ax2.set_title(f"B. Across-Basin Autocorrelation Distribution at Lead t+{ref_lt}\n(Shift Right of Observed = Model Over-Autocorrelation)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # --- Stratify by Lag-1 Open-Loop Residual Autocorrelation Quartiles rho_e(1) for Panels C & D ---
    obs_l1 = p_sub[(p_sub["Config ID"] == "Observed") & (p_sub["Lead Time (Days)"] == 1)].set_index("Basin ID")
    res1_col = "Residual Autocorr Lag-L" if "Residual Autocorr Lag-L" in obs_l1.columns else "Residual Autocorrelation"
    s_res1 = obs_l1[res1_col].dropna()
    if len(s_res1) >= 8:
        q_labels = ["Q1: Flashy", "Q2: Mod-Low", "Q3: Mod-High", "Q4: Baseflow"]
        bins = pd.qcut(s_res1, q=4, labels=q_labels, duplicates="drop")
        x_idx = np.arange(len(bins.cat.categories))
        x_tick_labels = []
        for cat in bins.cat.categories:
            b_cat = s_res1[bins == cat]
            x_tick_labels.append(f"{cat}\n(med $\\rho_e$={b_cat.median():.2f})")

        # Panel C: Model vs Observed Autocorrelation in each Quartile at Lead t+L
        bar_models_c = []
        for skey, slabel, cid, color, ls, mk in curve_styles:
            if ac_col == "Residual Autocorr Lag-1" and skey == "Observed":
                continue
            lbl_c = "Open-Loop Error Corr" if (skey == "Observed" and ac_col == "Residual Autocorr Lag-L") else slabel
            bar_models_c.append((skey, lbl_c, cid, color, ac_col, 1.0, None))
        if ac_col == "Residual Autocorr Lag-L" and "per_basin_rho" in refs:
            bar_models_c.append(("assumed_ar1", "Assumed Per-Basin PP $\\rho_b^L$", refs["per_basin_rho"], "#795548", "Assumed PP Autocorr", 0.85, "//"))

        n_mc = max(1, len(bar_models_c))
        w_c = min(0.13, 0.80 / n_mc)
        for m_idx, (skey, slabel, cid, color, col_use, alpha_v, h_pat) in enumerate(bar_models_c):
            sub_m = p_sub[(p_sub["Config ID"] == cid) & (p_sub["Lead Time (Days)"] == ref_lt)].set_index("Basin ID")
            med_q = []
            for cat in bins.cat.categories:
                b_ids = bins[bins == cat].index
                vals_q = sub_m.reindex(b_ids)[col_use].dropna() if col_use in sub_m.columns else pd.Series(dtype=float)
                med_q.append(float(vals_q.median()) if len(vals_q) else np.nan)
            offset = (m_idx - (n_mc - 1) / 2.0) * w_c
            ax3.bar(x_idx + offset, med_q, width=w_c, color=color, label=slabel, alpha=alpha_v * 0.9, edgecolor="white", lw=0.6, hatch=h_pat)

        if ac_col.startswith("Residual"):
            ax3.axhline(0.0, color="#80868b", lw=1.0, ls="--")
        ax3.set_xticks(x_idx)
        ax3.set_xticklabels(x_tick_labels, fontsize=9.5)
        ax3.set_xlabel("Lag-1 Open-Loop Residual Autocorrelation Quartile $\\rho_e(1)$", fontsize=11)
        ax3.set_ylabel(f"Median {ac_short} (Lead t+{ref_lt})", fontsize=11)
        ax3.set_title(f"C. Model vs. Observed Autocorrelation by Memory Regime (Lead t+{ref_lt})", fontsize=11.5, fontweight="bold")
        ax3.legend(loc="best", frameon=True, fontsize=8.2, ncol=2)
        ax3.grid(True, alpha=0.3, axis="y")

        # Panel D: Forecast Skill Score SS_NSE in each Quartile at Lead t+L
        da_ar_models = [c for c in curve_styles if c[0] in refs]
        n_md = max(1, len(da_ar_models))
        w_d = min(0.18, 0.75 / n_md)
        for m_idx, (skey, slabel, cid, color, _, _) in enumerate(da_ar_models):
            sub_ev = ev_sub[(ev_sub["Config ID"] == cid) & (ev_sub["Lead Time (Days)"] == ref_lt)].set_index("Basin ID")
            med_ss_q = []
            for cat in bins.cat.categories:
                b_ids = bins[bins == cat].index
                vals_q = sub_ev.reindex(b_ids)["NSE Skill Score"].dropna()
                med_ss_q.append(float(vals_q.median()) if len(vals_q) else np.nan)
            offset = (m_idx - (n_md - 1) / 2.0) * w_d
            ax4.bar(x_idx + offset, med_ss_q, width=w_d, color=color, label=slabel, alpha=0.9, edgecolor="white", lw=0.6)

        ax4.axhline(0.0, color="#80868b", lw=1.0, ls="--")
        units.ss_axis(ax4, "y")
        ax4.set_xticks(x_idx)
        ax4.set_xticklabels(x_tick_labels, fontsize=9.5)
        ax4.set_xlabel("Lag-1 Open-Loop Residual Autocorrelation Quartile $\\rho_e(1)$", fontsize=11)
        ax4.set_ylabel(units.ss_label(f"Median SS_NSE (Lead t+{ref_lt})"), fontsize=11)
        ax4.set_title(f"D. Forecast Skill (SS_NSE) by Basin Memory Regime (Lead t+{ref_lt})", fontsize=11.5, fontweight="bold")
        ax4.legend(loc="upper left", frameon=True, fontsize=8.8)
        ax4.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


# =============================================================================
# 6. High-Flow Exceedance (Precision/Recall/CSI) & Flood Peak Timing/Magnitude (Leadtimes as Columns)
# =============================================================================
def _attach_per_basin_event_rows(df_event: pd.DataFrame) -> pd.DataFrame:
    """Synthesizes Per-Basin Best DA and AR1_Per_Basin_Best_Rho rows in ``df_event`` using the active selection."""
    sel = get_active()
    df = df_event[~df_event["Config ID"].isin([PER_BASIN_DA, PER_BASIN_RHO])].copy()
    if sel is None:
        return df

    def _synth(mapping: Dict[str, str], name: str) -> pd.DataFrame:
        if not mapping:
            return df.iloc[0:0]
        m = pd.DataFrame({"Basin ID": list(mapping.keys()), "_sel": list(mapping.values())})
        sub = df[df["Config ID"].isin(set(mapping.values()))].merge(m, on="Basin ID")
        sub = sub[sub["Config ID"] == sub["_sel"]].drop(columns=["_sel"])
        sub["Config ID"] = name
        return sub

    extra = [_synth(sel.per_basin_da, PER_BASIN_DA), _synth(sel.per_basin_rho, PER_BASIN_RHO)]
    extra = [e for e in extra if not e.empty]
    return pd.concat([df] + extra, ignore_index=True) if extra else df


def build_event_and_peak_summary_table(
    df_event: pd.DataFrame,
    windows: Optional[Dict[str, Sequence[int]] | str] = None,
    show: Optional[Sequence[str]] = None,
    threshold: str = "both",      # "Q90", "Q95", or "both"
    value_mode: str = "both",     # "both", "raw", or "ss"
    lead_times: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds two summary tables with Leadtimes (``t+1..t+7``) as COLUMNS by default:
      1. ``exceedance_table``: High-Flow Exceedance Precision, Recall (POD), CSI, F1, and Zero-Alarm % at Q90 and Q95.
      2. ``peak_table``: Flood Event Peak Timing Error (Signed PTE, Absolute |PTE| in days, Exact 0-day %,
         Extended Phase-Lag PTE) and Peak Flow Error (Signed PFE, Absolute |PFE|, Top-2% FHV) + Skill Scores.
    """
    if df_event.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = _attach_per_basin_event_rows(df_event)
    win_map = _resolve_windows(df, windows=windows, lead_times=lead_times)
    col_names = list(win_map.keys())
    refs = _resolve_scheme_configs(df, show=show)
    scheme_order = [("baseline", "Baseline (Open-Loop)", "Baseline")] + [
        (k, SCHEME_LABELS[k], refs[k]) for k in refs
    ]

    all_leads = sorted({lt for lts in win_map.values() for lt in lts})
    csi_sets, pte_sets = [], []
    for _, _, cid in scheme_order:
        sub = df[df["Config ID"] == cid]
        s_csi = _basin_window_means(sub, "CSI (Q90)", all_leads)
        s_pte = _basin_window_means(sub, "PTE Abs (Days)", all_leads)
        if not s_csi.empty:
            csi_sets.append(set(s_csi.index))
        if not s_pte.empty:
            pte_sets.append(set(s_pte.index))
    common_csi = set.intersection(*csi_sets) if csi_sets else set()
    common_pte = set.intersection(*pte_sets) if pte_sets else set()

    # --- 1. High-Flow Exceedance Table (t+1..t+7 as columns) ---
    exc_rows: List[Dict[str, Any]] = []
    q_levels = ["Q90", "Q95"] if threshold.lower() == "both" else [threshold.upper()]
    exc_metrics = [
        ("Precision", "Precision ({q})", True, "raw_3"),
        ("Recall / POD", "Recall ({q})", True, "raw_3"),
        ("CSI (Threat Score)", "CSI ({q})", True, "raw_3"),
        ("F1 Score", "F1 ({q})", True, "raw_3"),
        ("Zero-Alarm Basins (%)", "ZeroAlarm ({q})", False, "pct"),
    ]
    for q in q_levels:
        for m_label, col_tmpl, has_ss, fmt_kind in exc_metrics:
            col_name = col_tmpl.format(q=q)
            if value_mode in ("both", "raw"):
                for skey, slabel, cid in scheme_order:
                    sub_c = df[(df["Config ID"] == cid) & (df["Basin ID"].isin(common_csi))] if common_csi else df[df["Config ID"] == cid]
                    row: Dict[str, Any] = {
                        "Threshold": q,
                        "Metric": m_label,
                        "Scheme": slabel,
                        "Type": "Rate (%)" if fmt_kind == "pct" else "Raw Median",
                        "N Basins": len(common_csi),
                        "_fmt": fmt_kind,
                    }
                    for wname, leads in win_map.items():
                        if fmt_kind == "pct":
                            sub_w = sub_c[sub_c["Lead Time (Days)"].isin(leads)]
                            row[wname] = float(sub_w[col_name].mean() * 100.0) if col_name in sub_w else np.nan
                        else:
                            s = _basin_window_means(sub_c, col_name, leads)
                            row[wname] = float(s.median()) if len(s) else np.nan
                    exc_rows.append(row)

            if has_ss and value_mode in ("both", "ss"):
                ss_col = f"{col_name} Skill Score"
                for skey, slabel, cid in scheme_order:
                    if skey == "baseline":
                        continue
                    sub_c = df[(df["Config ID"] == cid) & (df["Basin ID"].isin(common_csi))] if common_csi else df[df["Config ID"] == cid]
                    row = {
                        "Threshold": q,
                        "Metric": m_label,
                        "Scheme": slabel,
                        "Type": units.ss_label("Skill Score SS"),
                        "N Basins": len(common_csi),
                        "_fmt": "ss",
                    }
                    for wname, leads in win_map.items():
                        s = _basin_window_means(sub_c, ss_col, leads)
                        row[wname] = float(s.median()) if len(s) else np.nan
                    exc_rows.append(row)

    # --- 2. Flood Event Peak Timing & Magnitude Table (t+1..t+7 as columns) ---
    peak_rows: List[Dict[str, Any]] = []
    peak_specs = [
        ("PTE Signed (±2d Window)", "PTE Signed (Days)", "mean", False, "days_signed"),
        ("PTE |Error| (±2d Window)", "PTE Abs (Days)", "mean", True, "days_abs"),
        ("Exact 0-Day Peak Timing (%)", "Exact Peak Timing (%)", "mean", False, "pct"),
        ("Phase-Lag PTE Signed ([-2,+7]d)", "PTE Phase Signed (Days)", "mean", False, "days_signed"),
        ("Phase-Lag PTE |Error| ([-2,+7]d)", "PTE Phase Abs (Days)", "mean", True, "days_abs"),
        ("PFE Signed (Rel. Peak Bias)", "PFE Signed", "median", False, "raw_signed"),
        ("PFE |Error| (Rel. Peak Error)", "PFE Abs", "median", True, "raw_3"),
        ("FHV Top-2% Volume Bias", "FHV", "median", False, "raw_signed"),
        ("FHV |Bias| Top-2% Volume Error", "FHV Abs", "median", True, "raw_3"),
    ]
    for m_label, col_name, agg_kind, has_ss, fmt_kind in peak_specs:
        if value_mode in ("both", "raw"):
            for skey, slabel, cid in scheme_order:
                sub_p = df[(df["Config ID"] == cid) & (df["Basin ID"].isin(common_pte))] if common_pte else df[df["Config ID"] == cid]
                row = {
                    "Metric": m_label,
                    "Scheme": slabel,
                    "Type": f"Raw {agg_kind.capitalize()}",
                    "N Basins": len(common_pte),
                    "_fmt": fmt_kind,
                }
                for wname, leads in win_map.items():
                    s = _basin_window_means(sub_p, col_name, leads)
                    row[wname] = float(s.mean() if agg_kind == "mean" else s.median()) if len(s) else np.nan
                peak_rows.append(row)

        if has_ss and value_mode in ("both", "ss"):
            ss_col = f"{col_name} Skill Score"
            for skey, slabel, cid in scheme_order:
                if skey == "baseline":
                    continue
                sub_p = df[(df["Config ID"] == cid) & (df["Basin ID"].isin(common_pte))] if common_pte else df[df["Config ID"] == cid]
                row = {
                    "Metric": m_label,
                    "Scheme": slabel,
                    "Type": units.ss_label("Skill Score SS"),
                    "N Basins": len(common_pte),
                    "_fmt": "ss",
                }
                for wname, leads in win_map.items():
                    s = _basin_window_means(sub_p, ss_col, leads)
                    row[wname] = float(s.median()) if len(s) else np.nan
                peak_rows.append(row)

    exc_df = pd.DataFrame(exc_rows)
    peak_df = pd.DataFrame(peak_rows)
    exc_df.attrs["col_names"] = col_names
    peak_df.attrs["col_names"] = col_names
    return exc_df, peak_df


def style_event_and_peak_tables(exc_df: pd.DataFrame, peak_df: pd.DataFrame) -> Tuple[Any, Any]:
    """Formats High-Flow Exceedance (Q90/Q95) and Flood Peak Timing/Magnitude tables with ``t+1..t+7`` as columns."""
    def _format_df(df: pd.DataFrame, left_cols: List[str], caption: str) -> Any:
        if df.empty:
            return df
        col_names = df.attrs.get("col_names", [c for c in df.columns if c.startswith("t+") or "Range" in c or "Horizon" in c])
        disp = df.drop(columns=["_fmt"], errors="ignore").copy()
        for c in col_names:
            if c in disp.columns:
                disp[c] = disp[c].astype(object)
        for idx, r in df.iterrows():
            fkind = r.get("_fmt", "raw_3")
            for c in col_names:
                val = r[c]
                if pd.isna(val) or not np.isfinite(float(val)):
                    disp.at[idx, c] = "–"
                elif fkind == "ss":
                    disp.at[idx, c] = units.fmt_ss(float(val))
                elif fkind == "pct":
                    disp.at[idx, c] = f"{float(val):.1f}%"
                elif fkind == "days_signed":
                    disp.at[idx, c] = f"{float(val):+.2f} d"
                elif fkind == "days_abs":
                    disp.at[idx, c] = f"{float(val):.2f} d"
                elif fkind == "raw_signed":
                    disp.at[idx, c] = f"{float(val):+.3f}"
                else:
                    disp.at[idx, c] = f"{float(val):.3f}"
        return (
            disp.style.format({"N Basins": "{:,d}"}, na_rep="–")
            .hide(axis="index")
            .set_properties(subset=[c for c in left_cols if c in disp.columns], **{"text-align": "left"})
            .set_properties(subset=col_names, **{"text-align": "right", "padding": "4px 10px"})
            .set_caption(caption)
        )

    sty_exc = _format_df(
        exc_df,
        ["Threshold", "Metric", "Scheme", "Type"],
        "High-Flow Exceedance Categorical Skill Across Lead Times (Q90 & Q95): Precision, Recall (POD), CSI, F1, Skill Scores SS_M, and % Zero-Alarm Basins",
    )
    sty_peak = _format_df(
        peak_df,
        ["Metric", "Scheme", "Type"],
        "Flood Event Peak Timing Error (PTE in days: local ±2d window & extended [-2,+7]d phase-lag window), Peak Flow Error (PFE), and Top-2% Volume Bias (FHV) Across Lead Times",
    )
    return sty_exc, sty_peak


def plot_event_and_peak_metrics(
    df_event: pd.DataFrame,
    lead_time_focus: int = 1,
    threshold: str = "Q95",
    show: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (18.0, 9.2),
) -> plt.Figure:
    """2x4 publication figure of Flood Event Timing, Peak Magnitude, and High-Flow Exceedance across leads t+1..t+7."""
    df = _attach_per_basin_event_rows(df_event)
    leads = sorted(int(l) for l in df["Lead Time (Days)"].dropna().unique() if int(l) >= 1)
    refs = _resolve_scheme_configs(df, show=show)
    q_tag = "Q90" if str(threshold).upper() == "Q90" else "Q95"

    styles = [
        ("baseline", "Open-Loop Baseline", "Baseline", palette.color("baseline"), palette.ls("baseline"), "o"),
        ("global_da", "Global-Best DA", refs.get("global_da"), palette.color("global_da"), palette.ls("global_da"), "s"),
        ("per_basin_da", "Per-Basin DA (Oracle)", refs.get("per_basin_da"), palette.color("per_basin_da"), palette.ls("per_basin_da"), "D"),
        ("global_rho", "Global-Best PP", refs.get("global_rho"), palette.color("global_rho"), palette.ls("global_rho"), "^"),
        ("per_basin_rho", "Per-Basin PP", refs.get("per_basin_rho"), palette.color("per_basin_rho"), palette.ls("per_basin_rho"), "v"),
    ]
    styles = [s for s in styles if s[2] is not None]

    common = set(df["Basin ID"].unique())
    for _, _, cid, _, _, _ in styles:
        b = set(df[df["Config ID"] == cid]["Basin ID"].unique())
        if b:
            common &= b
    df = df[df["Basin ID"].isin(common)]

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    panels = [
        (axes[0, 0], "PTE Signed (Days)", "mean", "A. Signed Peak Timing Bias (±2d Window)", "Timing Error (Days, + = Late)", 0.0, False),
        (axes[0, 1], "PTE Phase Signed (Days)", "mean", "B. Phase-Lag Timing Bias ([-2,+7]d Window)", "Phase Lag (Days, + = Echo Lag)", 0.0, False),
        (axes[0, 2], "Exact Peak Timing (%)", "mean", "C. Exact 0-Day Peak Timing Rate", "% Flood Peaks at Exact Day", None, False),
        (axes[0, 3], "PFE Abs", "median", "D. Relative Peak Magnitude Error |PFE|", "Median |Q_sim − Q_obs| / Q_obs", None, False),
        (axes[1, 0], f"Recall ({q_tag})", "median", f"E. High-Flow Recall / POD ({q_tag})", "Median Recall (TP / (TP + FN))", None, False),
        (axes[1, 1], f"Precision ({q_tag})", "median", f"F. High-Flow Precision ({q_tag})", "Median Precision (TP / (TP + FP))", None, False),
        (axes[1, 2], f"CSI ({q_tag})", "median", f"G. High-Flow Threat Score / CSI ({q_tag})", "Median CSI (TP / (TP + FP + FN))", None, False),
        (axes[1, 3], f"CSI ({q_tag}) Skill Score", "median", f"H. High-Flow CSI Skill Score ({q_tag})", units.ss_label("CSI Skill Score"), 0.0, True),
    ]

    for ax, col, agg_kind, title, ylabel, ref_line, is_ss in panels:
        if ref_line is not None:
            ax.axhline(ref_line, color="#80868b", ls="--", lw=1.1)
        if lead_time_focus in leads:
            ax.axvline(lead_time_focus, color="#e8eaed", lw=5.0, zorder=0)
        for skey, slabel, cid, color, ls, mk in styles:
            if is_ss and skey == "baseline":
                continue
            sub = df[df["Config ID"] == cid]
            vals = []
            for lt in leads:
                s = sub.loc[sub["Lead Time (Days)"] == lt, col].dropna()
                vals.append(float(s.mean() if agg_kind == "mean" else s.median()) if len(s) else np.nan)
            ax.plot(leads, vals, label=slabel, color=color, ls=ls, marker=mk, lw=2.2, ms=5.5)
        if is_ss:
            units.ss_axis(ax, "y")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xticks(leads)
        ax.set_xticklabels([f"t+{l}" for l in leads])
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("Forecast Lead Time (Days)", fontsize=10.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    sel = get_active()
    sel_txt = sel.label() if sel is not None else "default t+1"
    fig.suptitle(
        f"Flood Event Peak Timing, Peak Magnitude & High-Flow Exceedance ({q_tag}) Across Leads "
        f"(N = {len(common):,} basins | {sel_txt})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), frameon=False, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# =============================================================================
# 7. Multi-Year Leave-One-Year-Out (5-Yr -> 1-Yr) Temporal Cross-Validation (Leadtimes as Columns)
# =============================================================================
def _metrics_from_sums(df_sums: pd.DataFrame) -> pd.DataFrame:
    out = df_sums.copy()
    n = out["n"].to_numpy(np.float64)
    so = out["so"].to_numpy(np.float64)
    soo = out["soo"].to_numpy(np.float64)
    ss = out["ss"].to_numpy(np.float64)
    sss = out["sss"].to_numpy(np.float64)
    sos = out["sos"].to_numpy(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        sst = soo - (so * so) / np.maximum(n, 1.0)
        sse = soo + sss - 2.0 * sos
        nse = np.where((n >= 10) & (sst > 0), 1.0 - sse / sst, np.nan)

        mu_o = so / n
        mu_s = ss / n
        var_o = (soo / n) - mu_o ** 2
        var_s = (sss / n) - mu_s ** 2
        cov_os = (sos / n) - mu_o * mu_s
        std_o = np.sqrt(np.maximum(var_o, 0.0))
        std_s = np.sqrt(np.maximum(var_s, 0.0))
        r = np.where((n >= 10) & (std_o > 0) & (std_s > 0), cov_os / (std_o * std_s), np.nan)
        alpha = np.where((n >= 10) & (std_o > 0), std_s / std_o, np.nan)
        beta = np.where((n >= 10) & (np.abs(mu_o) > 0), mu_s / mu_o, np.nan)
        kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    out["DA NSE"] = nse
    out["DA KGE"] = kge
    return out


def _attach_base_and_ss(df_metrics: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    base = df_metrics[df_metrics["Config ID"] == "Baseline"][group_keys + ["DA NSE", "DA KGE"]].rename(
        columns={"DA NSE": "Base NSE", "DA KGE": "Base KGE"}
    ).drop_duplicates(group_keys)
    merged = df_metrics[df_metrics["Config ID"] != "Baseline"].merge(base, on=group_keys, how="left")
    merged["NSE Delta"] = merged["DA NSE"] - merged["Base NSE"]
    merged["KGE Delta"] = merged["DA KGE"] - merged["Base KGE"]
    return _ensure_skill_score_columns(merged)


def compute_temporal_cv(
    df_yearly_stats: pd.DataFrame,
    policy: Optional[SelectionPolicy] = None,
    cv_years: Sequence[int] = (2017, 2018, 2019, 2020, 2021, 2022),
    require_complete_years: bool = True,
    min_days_per_year: int = 30,
    lead_time: int = 1,
    windows: Optional[Dict[str, Sequence[int]] | str] = None,
    stat_view: str = "median",  # "median", "q10", "harm", "all"
) -> Dict[str, Any]:
    """Leave-One-Year-Out (5-Year Train -> 1-Year Out-of-Sample Test) Cross-Validation across ``cv_years``
    with Leadtimes (``t+1..t+7``) as COLUMNS by default across all summary and calibration-ladder tables!
    """
    sel = get_active()
    policy = policy or (sel.policy if sel is not None else SelectionPolicy(leads=(4, 5, 6, 7)))
    cv_years = sorted(int(y) for y in cv_years)
    df_y = df_yearly_stats[df_yearly_stats["Year"].isin(cv_years)].copy()
    if df_y.empty:
        return {}

    if require_complete_years:
        b_valid = df_y[
            (df_y["Config ID"] == "Baseline")
            & (df_y["Lead Time (Days)"] == 1)
            & (df_y["n"] >= min_days_per_year)
        ]
        yr_counts = b_valid.groupby("Basin ID")["Year"].nunique()
        complete_basins = set(yr_counts[yr_counts == len(cv_years)].index)
        if complete_basins:
            df_y = df_y[df_y["Basin ID"].isin(complete_basins)].copy()

    stat_cols = ["n", "so", "soo", "ss", "sss", "sos"]
    all_cfgs = sorted(df_y["Config ID"].unique())
    da_cfgs = [c for c in all_cfgs if c != "Baseline" and not is_rho_config(c)]
    rho_cfgs = [c for c in all_cfgs if is_rho_config(c) and not is_per_basin_rho_config(c)]

    full_sums = df_y.groupby(["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False)[stat_cols].sum()
    full_metrics = _attach_base_and_ss(_metrics_from_sums(full_sums), ["Basin ID", "Lead Time (Days)"])
    leads = sorted(int(l) for l in full_metrics["Lead Time (Days)"].dropna().unique() if int(l) >= 1)
    ref_lt = int(lead_time) if int(lead_time) in leads else leads[0]
    win_map = _resolve_windows(full_metrics, windows=windows, lead_times=leads)
    col_names = list(win_map.keys())

    def _select_from_metrics(df_m: pd.DataFrame, pool: List[str]) -> Tuple[Optional[str], Dict[str, str]]:
        sub = df_m[df_m["Config ID"].isin(pool) & df_m["Lead Time (Days)"].isin(policy.leads)]
        if sub.empty:
            return None, {}
        wide = sub.pivot_table(index=["Config ID", "Basin ID"], columns="Lead Time (Days)", values=policy.metric, aggfunc="first")
        wide = wide.reindex(columns=list(policy.leads))
        wide = wide[wide.notna().all(axis=1)]
        sc = (wide.sum(axis=1) if policy.agg == "sum" else wide.mean(axis=1)).rename("score").reset_index()
        if sc.empty:
            return None, {}
        n_cfg = sc.groupby("Basin ID")["Config ID"].nunique()
        common_b = n_cfg[n_cfg == sc["Config ID"].nunique()].index
        sc_g = sc[sc["Basin ID"].isin(common_b)] if len(common_b) else sc
        rank = sc_g.groupby("Config ID")["score"].median().sort_values(ascending=False)
        gb = str(rank.index[0]) if not rank.empty else None
        pb_rows = sc.loc[sc.groupby("Basin ID")["score"].idxmax()]
        pb_map = dict(zip(pb_rows["Basin ID"].astype(str), pb_rows["Config ID"].astype(str)))
        return gb, pb_map

    insample_gb_da, insample_pb_da = _select_from_metrics(full_metrics, da_cfgs)
    insample_gb_rho, insample_pb_rho = _select_from_metrics(full_metrics, rho_cfgs)

    fold_records = []
    heldout_selected_sums = []
    pb_da_choices_per_fold: Dict[int, Dict[str, str]] = {}
    gb_da_per_fold: Dict[int, str] = {}
    gb_rho_per_fold: Dict[int, str] = {}

    per_year_metrics = _attach_base_and_ss(_metrics_from_sums(df_y), ["Basin ID", "Year", "Lead Time (Days)"])

    for test_yr in cv_years:
        train_df = df_y[df_y["Year"] != test_yr]
        test_df = df_y[df_y["Year"] == test_yr]

        train_sums = train_df.groupby(["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False)[stat_cols].sum()
        train_m = _attach_base_and_ss(_metrics_from_sums(train_sums), ["Basin ID", "Lead Time (Days)"])

        gb_da_cv, pb_da_cv = _select_from_metrics(train_m, da_cfgs)
        gb_rho_cv, pb_rho_cv = _select_from_metrics(train_m, rho_cfgs)
        gb_da_per_fold[test_yr] = gb_da_cv or ""
        gb_rho_per_fold[test_yr] = gb_rho_cv or ""
        pb_da_choices_per_fold[test_yr] = pb_da_cv

        b_test = test_df[test_df["Config ID"] == "Baseline"].copy()
        heldout_selected_sums.append(b_test)

        if gb_da_cv:
            gda_test = test_df[test_df["Config ID"] == gb_da_cv].copy()
            gda_test["Config ID"] = "Global-Best DA (5-yr CV)"
            heldout_selected_sums.append(gda_test)
        if gb_rho_cv:
            grho_test = test_df[test_df["Config ID"] == gb_rho_cv].copy()
            grho_test["Config ID"] = "Global-Best PP (5-yr CV)"
            heldout_selected_sums.append(grho_test)
        if pb_da_cv:
            m_da = pd.DataFrame({"Basin ID": list(pb_da_cv.keys()), "_sel": list(pb_da_cv.values())})
            pbda_test = test_df[test_df["Config ID"].isin(set(pb_da_cv.values()))].merge(m_da, on="Basin ID")
            pbda_test = pbda_test[pbda_test["Config ID"] == pbda_test["_sel"]].drop(columns=["_sel"])
            pbda_test["Config ID"] = "Per-Basin Best DA (5-yr CV)"
            heldout_selected_sums.append(pbda_test)
        if pb_rho_cv:
            m_rho = pd.DataFrame({"Basin ID": list(pb_rho_cv.keys()), "_sel": list(pb_rho_cv.values())})
            pbrho_test = test_df[test_df["Config ID"].isin(set(pb_rho_cv.values()))].merge(m_rho, on="Basin ID")
            pbrho_test = pbrho_test[pbrho_test["Config ID"] == pbrho_test["_sel"]].drop(columns=["_sel"])
            pbrho_test["Config ID"] = "Per-Basin PP (5-yr CV)"
            heldout_selected_sums.append(pbrho_test)

        ym = per_year_metrics[per_year_metrics["Year"] == test_yr]

        def _yr_lead_score(mapping_or_cfg, lt_target: int):
            if isinstance(mapping_or_cfg, str):
                sub = ym[ym["Config ID"] == mapping_or_cfg]
            else:
                m_df = pd.DataFrame({"Basin ID": list(mapping_or_cfg.keys()), "_sel": list(mapping_or_cfg.values())})
                sub = ym.merge(m_df, on="Basin ID")
                sub = sub[sub["Config ID"] == sub["_sel"]]
            return _basin_window_means(sub, "NSE Skill Score", (lt_target,))

        s_gda_in = _yr_lead_score(insample_gb_da, ref_lt) if insample_gb_da else pd.Series(dtype=float)
        s_gda_cv = _yr_lead_score(gb_da_cv, ref_lt) if gb_da_cv else pd.Series(dtype=float)
        s_pbda_in = _yr_lead_score(insample_pb_da, ref_lt)
        s_pbda_cv = _yr_lead_score(pb_da_cv, ref_lt)
        s_grho_cv = _yr_lead_score(gb_rho_cv, ref_lt) if gb_rho_cv else pd.Series(dtype=float)
        s_pbrho_cv = _yr_lead_score(pb_rho_cv, ref_lt)

        common_yr = set.intersection(*(set(s.index) for s in (s_gda_in, s_gda_cv, s_pbda_in, s_pbda_cv, s_grho_cv, s_pbrho_cv) if not s.empty))
        agree_pct = float(np.mean([pb_da_cv.get(b) == insample_pb_da.get(b) for b in common_yr]) * 100.0) if common_yr else np.nan

        fold_row: Dict[str, Any] = {
            "Held-Out Test Year": test_yr,
            "Train Years": f"{min(set(cv_years) - {test_yr})}–{max(set(cv_years) - {test_yr})} ({len(cv_years)-1} yrs)",
            "% Basins Matching 6-Yr Winner": agree_pct,
            "N Basins": len(common_yr),
            "Global DA (In-Sample) SS": float(s_gda_in.reindex(common_yr).median()) if common_yr else np.nan,
            "Global DA (5-yr CV) SS": float(s_gda_cv.reindex(common_yr).median()) if common_yr else np.nan,
            "Global PP (5-yr CV) SS": float(s_grho_cv.reindex(common_yr).median()) if common_yr else np.nan,
            "Per-Basin DA (In-Sample Oracle) SS": float(s_pbda_in.reindex(common_yr).median()) if common_yr else np.nan,
            "Per-Basin DA (5-yr CV) SS": float(s_pbda_cv.reindex(common_yr).median()) if common_yr else np.nan,
            "Per-Basin PP (5-yr CV) SS": float(s_pbrho_cv.reindex(common_yr).median()) if common_yr else np.nan,
        }
        # Also attach per-leadtime Per-Basin DA (5-yr CV) across t+1..t+7 so leadtimes are columns!
        for lt in leads:
            s_lt = _yr_lead_score(pb_da_cv, lt)
            fold_row[f"t+{lt}"] = float(s_lt.reindex(common_yr).median()) if common_yr else np.nan
        fold_records.append(fold_row)

    stitched_sums = pd.concat(heldout_selected_sums, ignore_index=True).groupby(
        ["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False
    )[stat_cols].sum()
    stitched_cv_metrics = _attach_base_and_ss(_metrics_from_sums(stitched_sums), ["Basin ID", "Lead Time (Days)"])

    def _synth_full(mapping: Dict[str, str], name: str) -> pd.DataFrame:
        m_df = pd.DataFrame({"Basin ID": list(mapping.keys()), "_sel": list(mapping.values())})
        sub = full_metrics.merge(m_df, on="Basin ID")
        sub = sub[sub["Config ID"] == sub["_sel"]].drop(columns=["_sel"])
        sub["Config ID"] = name
        return sub

    insample_rows = []
    if insample_gb_da:
        g1 = full_metrics[full_metrics["Config ID"] == insample_gb_da].copy()
        g1["Config ID"] = "Global-Best DA (In-Sample)"
        insample_rows.append(g1)
    if insample_gb_rho:
        g2 = full_metrics[full_metrics["Config ID"] == insample_gb_rho].copy()
        g2["Config ID"] = "Global-Best PP (In-Sample)"
        insample_rows.append(g2)
    if insample_pb_da:
        insample_rows.append(_synth_full(insample_pb_da, "Per-Basin Best DA (In-Sample Oracle)"))
    if insample_pb_rho:
        insample_rows.append(_synth_full(insample_pb_rho, "Per-Basin PP (In-Sample Oracle)"))

    combined_cv = pd.concat([stitched_cv_metrics] + insample_rows, ignore_index=True)

    scheme_order = [
        "Global-Best DA (In-Sample)",
        "Global-Best DA (5-yr CV)",
        "Global-Best PP (In-Sample)",
        "Global-Best PP (5-yr CV)",
        "Per-Basin Best DA (In-Sample Oracle)",
        "Per-Basin Best DA (5-yr CV)",
        "Per-Basin PP (In-Sample Oracle)",
        "Per-Basin PP (5-yr CV)",
    ]

    # --- Summary Table with Leadtimes (t+1..t+7) as COLUMNS! ---
    base_sub = full_metrics.drop_duplicates(["Basin ID", "Lead Time (Days)"])
    common_b_sets = []
    for lt in leads:
        s_b = _basin_window_means(base_sub, "Base NSE", (lt,))
        if not s_b.empty:
            common_b_sets.append(set(s_b.index))
    common_b = set.intersection(*common_b_sets) if common_b_sets else set()

    summary_rows: List[Dict[str, Any]] = []
    # Row 1: Baseline Open-Loop Median NSE across t+1..t+7
    base_row: Dict[str, Any] = {
        "Scheme": "Baseline Open-Loop",
        "Statistic": "Median NSE",
        "N Basins": len(common_b),
        "_fmt": "base",
    }
    for wname, wleads in win_map.items():
        vals = _basin_window_means(base_sub, "Base NSE", wleads).reindex(common_b).dropna().to_numpy(float)
        base_row[wname] = float(np.median(vals)) if len(vals) else np.nan
    summary_rows.append(base_row)

    stat_choices = [("Median SS", "median"), ("Q10 (Downside Risk)", "q10"), ("% Harmed (<-1%)", "harm")]
    if stat_view != "all":
        stat_choices = [sc for sc in stat_choices if sc[1] == stat_view] or [stat_choices[0]]

    for sc_name in scheme_order:
        sub_sc = combined_cv[combined_cv["Config ID"] == sc_name]
        for st_label, st_code in stat_choices:
            row_sc: Dict[str, Any] = {
                "Scheme": sc_name,
                "Statistic": units.ss_label(st_label) if st_code in ("median", "q10") else st_label,
                "N Basins": len(common_b),
                "_fmt": "pct" if st_code == "harm" else "ss",
            }
            for wname, wleads in win_map.items():
                vals = _basin_window_means(sub_sc, "NSE Skill Score", wleads).reindex(common_b).dropna().to_numpy(float)
                if not len(vals):
                    row_sc[wname] = np.nan
                elif st_code == "median":
                    row_sc[wname] = float(np.median(vals))
                elif st_code == "q10":
                    row_sc[wname] = float(np.quantile(vals, 0.10))
                elif st_code == "harm":
                    row_sc[wname] = float((vals < -0.01).mean() * 100.0)
            summary_rows.append(row_sc)

    # Add Wilcoxon p-value rows across t+1..t+7 comparing Per-Basin CV vs Global CV and vs PP CV
    p_gb_row: Dict[str, Any] = {
        "Scheme": "Per-Basin Best DA (5-yr CV) vs. Global-Best DA (5-yr CV)",
        "Statistic": "Wilcoxon p-value",
        "N Basins": len(common_b),
        "_fmt": "str",
    }
    p_ar1_row: Dict[str, Any] = {
        "Scheme": "Per-Basin Best DA (5-yr CV) vs. Per-Basin PP (5-yr CV)",
        "Statistic": "Wilcoxon p-value",
        "N Basins": len(common_b),
        "_fmt": "str",
    }
    for wname, wleads in win_map.items():
        pb_s = _basin_window_means(combined_cv[combined_cv["Config ID"] == "Per-Basin Best DA (5-yr CV)"], "NSE Skill Score", wleads).reindex(common_b)
        gb_s = _basin_window_means(combined_cv[combined_cv["Config ID"] == "Global-Best DA (5-yr CV)"], "NSE Skill Score", wleads).reindex(common_b)
        ar_s = _basin_window_means(combined_cv[combined_cv["Config ID"] == "Per-Basin PP (5-yr CV)"], "NSE Skill Score", wleads).reindex(common_b)
        _, p1 = _wilcoxon_p(pb_s.to_numpy(float), gb_s.to_numpy(float))
        _, p2 = _wilcoxon_p(pb_s.to_numpy(float), ar_s.to_numpy(float))
        p_gb_row[wname] = f"{p1:.2e} ({_sig_stars(p1)})" if np.isfinite(p1) else "–"
        p_ar1_row[wname] = f"{p2:.2e} ({_sig_stars(p2)})" if np.isfinite(p2) else "–"
    summary_rows.extend([p_gb_row, p_ar1_row])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.attrs["col_names"] = col_names

    fold_df = pd.DataFrame(fold_records)
    fold_df.attrs["col_names"] = [f"t+{lt}" for lt in leads]
    fold_df.attrs["lead_time"] = ref_lt

    # --- C. Calibration Training-Length Ladder (K = 1..5 Training Years) with Leadtimes (t+1..t+7) as COLUMNS! ---
    oracle_sub = combined_cv[combined_cv["Config ID"] == "Per-Basin Best DA (In-Sample Oracle)"]
    oracle_by_col = {
        wname: float(_basin_window_means(oracle_sub, "NSE Skill Score", wleads).reindex(common_b).median())
        for wname, wleads in win_map.items()
    }

    ladder_rows: List[Dict[str, Any]] = []
    ladder_plot_records: List[Dict[str, Any]] = []

    for k_train in range(1, len(cv_years)):
        held_pb, held_gb, matches = [], [], []
        for test_yr in cv_years:
            other_yrs = sorted([y for y in cv_years if y != test_yr], key=lambda y: (abs(y - test_yr), y))[:k_train]
            tr_sums = df_y[df_y["Year"].isin(other_yrs)].groupby(["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False)[stat_cols].sum()
            tr_m = _attach_base_and_ss(_metrics_from_sums(tr_sums), ["Basin ID", "Lead Time (Days)"])
            gb_k, pb_k = _select_from_metrics(tr_m, da_cfgs)
            if pb_k and insample_pb_da:
                matches.append(float(np.mean([pb_k.get(b) == insample_pb_da.get(b) for b in insample_pb_da]) * 100.0))
            te = df_y[df_y["Year"] == test_yr]
            b_te = te[te["Config ID"] == "Baseline"].copy()
            if gb_k:
                g_te = te[te["Config ID"] == gb_k].copy()
                g_te["Config ID"] = "GB_K"
                held_gb.append(pd.concat([b_te, g_te], ignore_index=True))
            if pb_k:
                m_df = pd.DataFrame({"Basin ID": list(pb_k.keys()), "_sel": list(pb_k.values())})
                p_te = te.merge(m_df, on="Basin ID").query("`Config ID` == _sel").drop(columns=["_sel"]).copy()
                p_te["Config ID"] = "PB_K"
                held_pb.append(pd.concat([b_te, p_te], ignore_index=True))

        st_pb = _attach_base_and_ss(_metrics_from_sums(pd.concat(held_pb, ignore_index=True).groupby(["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False)[stat_cols].sum()), ["Basin ID", "Lead Time (Days)"])
        st_gb = _attach_base_and_ss(_metrics_from_sums(pd.concat(held_gb, ignore_index=True).groupby(["Basin ID", "Config ID", "Lead Time (Days)"], as_index=False)[stat_cols].sum()), ["Basin ID", "Lead Time (Days)"])

        match_pct = float(np.mean(matches)) if matches else np.nan
        h_label = f"{k_train} Yr{'s' if k_train > 1 else ''} Train → 1 Yr Test"

        row_pb: Dict[str, Any] = {
            "Calibration Horizon": h_label,
            "% Match 6-Yr Oracle": match_pct,
            "Scheme / Metric": "Per-Basin DA CV (Out-of-Sample SS)",
            "N Basins": len(common_b),
            "_fmt": "ss",
        }
        row_gb: Dict[str, Any] = {
            "Calibration Horizon": h_label,
            "% Match 6-Yr Oracle": match_pct,
            "Scheme / Metric": "Global DA CV (Out-of-Sample SS)",
            "N Basins": len(common_b),
            "_fmt": "ss",
        }
        row_ret: Dict[str, Any] = {
            "Calibration Horizon": h_label,
            "% Match 6-Yr Oracle": match_pct,
            "Scheme / Metric": "Oracle Retention (% of 6-Yr Oracle)",
            "N Basins": len(common_b),
            "_fmt": "pct",
        }
        for wname, wleads in win_map.items():
            s_pb = _basin_window_means(st_pb, "NSE Skill Score", wleads).reindex(common_b)
            s_gb = _basin_window_means(st_gb, "NSE Skill Score", wleads).reindex(common_b)
            pb_m = float(s_pb.median()) if len(s_pb.dropna()) else np.nan
            gb_m = float(s_gb.median()) if len(s_gb.dropna()) else np.nan
            or_m = oracle_by_col.get(wname, np.nan)
            row_pb[wname] = pb_m
            row_gb[wname] = gb_m
            row_ret[wname] = (pb_m / or_m * 100.0) if (np.isfinite(or_m) and or_m > 1e-6) else np.nan

        ladder_rows.extend([row_pb, row_gb, row_ret])

        for lt in leads:
            s_pb_lt = _basin_window_means(st_pb, "NSE Skill Score", (lt,)).reindex(common_b)
            s_gb_lt = _basin_window_means(st_gb, "NSE Skill Score", (lt,)).reindex(common_b)
            ladder_plot_records.append({
                "K Train Years": k_train,
                "Lead Time (Days)": lt,
                "% Matching 6-Yr Oracle Winner": match_pct,
                "Per-Basin DA CV SS": float(s_pb_lt.median()),
                "Global DA CV SS": float(s_gb_lt.median()),
                "Oracle SS": float(_basin_window_means(oracle_sub, "NSE Skill Score", (lt,)).reindex(common_b).median()),
            })

    ladder_df = pd.DataFrame(ladder_rows)
    ladder_df.attrs["col_names"] = col_names

    all_b = sorted(set.intersection(*(set(d.keys()) for d in pb_da_choices_per_fold.values()))) if pb_da_choices_per_fold else []
    n_distinct = [len({pb_da_choices_per_fold[y][b] for y in cv_years}) for b in all_b]
    stability = {
        "n_basins": len(all_b),
        "pct_1_unique_cfg_all_6_folds": float(np.mean([k == 1 for k in n_distinct]) * 100.0) if n_distinct else np.nan,
        "pct_le_2_unique_cfgs": float(np.mean([k <= 2 for k in n_distinct]) * 100.0) if n_distinct else np.nan,
        "mean_unique_cfgs_across_6_folds": float(np.mean(n_distinct)) if n_distinct else np.nan,
        "global_da_unique_winners": len(set(gb_da_per_fold.values())),
        "mean_oracle_winner_match_pct": float(fold_df["% Basins Matching 6-Yr Winner"].mean()) if not fold_df.empty else np.nan,
    }

    return {
        "summary_table": summary_df,
        "fold_table": fold_df,
        "calibration_ladder_table": ladder_df,
        "ladder_plot_df": pd.DataFrame(ladder_plot_records),
        "combined_cv_metrics": combined_cv,
        "stability": stability,
        "policy": policy,
        "cv_years": cv_years,
        "lead_time": ref_lt,
    }


def style_temporal_cv_tables(
    summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    ladder_df: Optional[pd.DataFrame] = None,
) -> Tuple[Any, ...]:
    """Formats Leave-One-Year-Out CV summary, calibration ladder, and fold tables with ``t+1..t+7`` as columns."""
    def _style_lead_cols(df: pd.DataFrame, left_cols: List[str], fmt_extra: Dict[str, str], caption: str) -> Any:
        if df is None or df.empty:
            return df
        col_names = df.attrs.get("col_names", [c for c in df.columns if c.startswith("t+") or "Range" in c or "Horizon" in c])
        disp = df.drop(columns=["_fmt"], errors="ignore").copy()
        for c in col_names:
            if c in disp.columns:
                disp[c] = disp[c].astype(object)
        for idx, r in df.iterrows():
            fkind = r.get("_fmt", "ss")
            for c in col_names:
                val = r[c]
                if isinstance(val, str):
                    disp.at[idx, c] = val
                elif pd.isna(val) or not np.isfinite(float(val)):
                    disp.at[idx, c] = "–"
                elif fkind == "base":
                    disp.at[idx, c] = f"{float(val):.3f}"
                elif fkind == "pct":
                    disp.at[idx, c] = f"{float(val):.1f}%"
                else:
                    disp.at[idx, c] = units.fmt_ss(float(val))
        return (
            disp.style.format(fmt_extra, na_rep="–")
            .hide(axis="index")
            .set_properties(subset=[c for c in left_cols if c in disp.columns], **{"text-align": "left"})
            .set_properties(subset=col_names, **{"text-align": "right", "padding": "4px 10px"})
            .set_caption(caption)
        )

    sty_sum = _style_lead_cols(
        summary_df,
        ["Scheme", "Statistic"],
        {"N Basins": "{:,d}"},
        "6-Year Leave-One-Year-Out Cross-Validation (5-Year Train → 1-Year Out-of-Sample Test) Across Lead Times: Stitched Out-of-Sample Skill vs. 6-Year In-Sample Oracle",
    )

    f_cols = fold_df.attrs.get("col_names", [c for c in fold_df.columns if c.startswith("t+")])
    ref_lt = fold_df.attrs.get("lead_time", 1)
    fmt_fold: Dict[str, Any] = {
        "% Basins Matching 6-Yr Winner": "{:.1f}%",
        "N Basins": "{:,d}",
        "Global DA (In-Sample) SS": units.ss_formatter(),
        "Global DA (5-yr CV) SS": units.ss_formatter(),
        "Global PP (5-yr CV) SS": units.ss_formatter(),
        "Per-Basin DA (In-Sample Oracle) SS": units.ss_formatter(),
        "Per-Basin DA (5-yr CV) SS": units.ss_formatter(),
        "Per-Basin PP (5-yr CV) SS": units.ss_formatter(),
    }
    for c in f_cols:
        if c in fold_df.columns:
            fmt_fold[c] = units.ss_formatter()
    sty_fold = (
        fold_df.style.format(fmt_fold, na_rep="–")
        .hide(axis="index")
        .set_caption(
            f"Year-by-Year Fold Evaluation (5-Year Training Selection Applied to Each Held-Out Test Year | "
            f"Scheme Columns at Lead t+{ref_lt} + Per-Basin DA CV Across t+1..t+7)"
        )
    )

    if ladder_df is not None and not ladder_df.empty:
        sty_lad = _style_lead_cols(
            ladder_df,
            ["Calibration Horizon", "Scheme / Metric"],
            {"% Match 6-Yr Oracle": "{:.1f}%", "N Basins": "{:,d}"},
            "Calibration Training-Length Sensitivity (K = 1..5 Training Years → 1 Held-Out Test Year) Across Lead Times (t+1..t+7)",
        )
        return sty_sum, sty_lad, sty_fold

    return sty_sum, sty_fold


def plot_temporal_cv_results(
    cv_res: Dict[str, Any],
    lead_time: Optional[int] = None,
    show: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (17.5, 5.8),
) -> plt.Figure:
    """3-panel figure of 5-Year -> 1-Year Out-of-Sample Cross-Validation:
      Panel A: Lead-time skill trajectory (t+1..t+7) comparing In-Sample vs Out-of-Sample (5-yr CV).
      Panel B: Training-History Calibration Curve (K = 1..5 Training Years -> 1 Held-Out Test Year) at ``lead_time``.
      Panel C: Year-by-year held-out test fold performance (2017..2022) at ``lead_time``.
    """
    if not cv_res or "combined_cv_metrics" not in cv_res:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No multi-year CV data available", ha="center", transform=ax.transAxes)
        return fig

    comb = cv_res["combined_cv_metrics"]
    fold_tbl = cv_res["fold_table"]
    lad_plot = cv_res.get("ladder_plot_df", pd.DataFrame())
    stab = cv_res.get("stability", {})
    leads = sorted(int(l) for l in comb["Lead Time (Days)"].dropna().unique() if int(l) >= 1)
    ref_lt = int(lead_time) if (lead_time is not None and int(lead_time) in leads) else int(cv_res.get("lead_time", leads[0]))
    n_b = stab.get("n_basins", 0)
    wanted = set(DEFAULT_SHOW_MODELS if show is None else show)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Panel A: Per-lead median SS_NSE (Stitched 6-year Out-of-Sample vs In-Sample)
    ax1.axvline(ref_lt, color="#e8eaed", lw=6.0, zorder=0)
    curve_specs = [
        ("per_basin_da", "Per-Basin Best DA (In-Sample Oracle)", "Per-Basin DA (In-Sample Oracle)", palette.color("per_basin_da"), "--", "D"),
        ("per_basin_da", "Per-Basin Best DA (5-yr CV)", "Per-Basin DA (5-yr Out-of-Sample CV)", palette.color("per_basin_da"), "-", "D"),
        ("global_da", "Global-Best DA (In-Sample)", "Global DA (In-Sample)", palette.color("global_da"), "--", "s"),
        ("global_da", "Global-Best DA (5-yr CV)", "Global DA (5-yr Out-of-Sample CV)", palette.color("global_da"), "-", "s"),
        ("global_rho", "Global-Best PP (5-yr CV)", "Global PP (5-yr CV)", palette.color("global_rho"), "-", "^"),
        ("per_basin_rho", "Per-Basin PP (5-yr CV)", "Per-Basin PP (5-yr CV)", palette.color("per_basin_rho"), ":", "v"),
    ]
    for skey, cid, lbl, col, ls, mk in curve_specs:
        if skey not in wanted:
            continue
        sub = comb[comb["Config ID"] == cid]
        if sub.empty:
            continue
        meds = [float(sub.loc[sub["Lead Time (Days)"] == lt, "NSE Skill Score"].median()) for lt in leads]
        ax1.plot(leads, meds, label=lbl, color=col, ls=ls, marker=mk, lw=2.2, ms=5.2)
    ax1.axhline(0.0, color="#80868b", ls="--", lw=1.1)
    units.ss_axis(ax1, "y")
    ax1.set_xticks(leads)
    ax1.set_xticklabels([f"t+{l}" for l in leads])
    ax1.set_xlabel("Forecast Lead Time (Days)", fontsize=11)
    ax1.set_ylabel(units.ss_label("Median NSE Skill Score"), fontsize=11)
    ax1.set_title(f"A. In-Sample vs. 5-Yr CV Across Leads (N = {n_b:,})", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel B: Calibration Training-Length Curve at Lead t+L
    if not lad_plot.empty:
        sub_l = lad_plot[lad_plot["Lead Time (Days)"] == ref_lt].sort_values("K Train Years")
        if not sub_l.empty:
            k_vals = sub_l["K Train Years"].to_numpy(int)
            or_lt = float(sub_l["Oracle SS"].iloc[0])
            ax2.axhline(or_lt, color=palette.color("per_basin_da"), ls="--", lw=1.6, label=f"Per-Basin DA (6-Yr Oracle @ t+{ref_lt})")
            ax2.plot(k_vals, sub_l["Per-Basin DA CV SS"], color=palette.color("per_basin_da"), ls="-", marker="D", lw=2.3, ms=6, label=f"Per-Basin DA CV (Lead t+{ref_lt})")
            ax2.plot(k_vals, sub_l["Global DA CV SS"], color=palette.color("global_da"), ls="-", marker="s", lw=2.1, ms=5.5, label=f"Global DA CV (Lead t+{ref_lt})")
            units.ss_axis(ax2, "y")
            ax2.set_xticks(k_vals)
            ax2.set_xticklabels([f"{k} Yr{'s' if k > 1 else ''}\n({m:.0f}% match)" for k, m in zip(k_vals, sub_l["% Matching 6-Yr Oracle Winner"])], fontsize=9.2)
            ax2.set_xlabel("Calibration Training Window Length K (and % Oracle Winner Match)", fontsize=10.5)
            ax2.set_ylabel(units.ss_label(f"Out-of-Sample Median SS_NSE (Lead t+{ref_lt})"), fontsize=11)
            ax2.set_title(f"B. Out-of-Sample Skill vs. Calibration Years (Lead t+{ref_lt})", fontsize=11.5, fontweight="bold")
            ax2.legend(loc="best", frameon=True, fontsize=8.8)
            ax2.grid(True, alpha=0.3)

    # Panel C: Per-fold held-out year skill at Lead t+L
    yrs = fold_tbl["Held-Out Test Year"].to_numpy(int)
    if "per_basin_da" in wanted:
        ax3.plot(yrs, fold_tbl["Per-Basin DA (In-Sample Oracle) SS"], color=palette.color("per_basin_da"), ls="--", marker="D", lw=2.0, label="Per-Basin DA (In-Sample Oracle)")
        ax3.plot(yrs, fold_tbl["Per-Basin DA (5-yr CV) SS"], color=palette.color("per_basin_da"), ls="-", marker="D", lw=2.3, label="Per-Basin DA (5-yr CV Out-of-Sample)")
    if "global_da" in wanted:
        ax3.plot(yrs, fold_tbl["Global DA (5-yr CV) SS"], color=palette.color("global_da"), ls="-", marker="s", lw=2.2, label="Global DA (5-yr CV)")
    if "global_rho" in wanted:
        ax3.plot(yrs, fold_tbl["Global PP (5-yr CV) SS"], color=palette.color("global_rho"), ls="-", marker="^", lw=2.0, label="Global PP (5-yr CV)")
    if "per_basin_rho" in wanted:
        ax3.plot(yrs, fold_tbl["Per-Basin PP (5-yr CV) SS"], color=palette.color("per_basin_rho"), ls=":", marker="v", lw=2.0, label="Per-Basin PP (5-yr CV)")
    ax3.axhline(0.0, color="#80868b", ls="--", lw=1.0)
    units.ss_axis(ax3, "y")
    ax3.set_xticks(yrs)
    ax3.set_xlabel("Held-Out Out-of-Sample Test Year", fontsize=11)
    ax3.set_ylabel(units.ss_label(f"Held-Out Year Median SS_NSE (Lead t+{ref_lt})"), fontsize=11)
    ax3.set_title(
        f"C. Year-by-Year Held-Out Test Skill at Lead t+{ref_lt} ({stab.get('mean_oracle_winner_match_pct', 0):.0f}% Match)",
        fontsize=11.5, fontweight="bold",
    )
    ax3.legend(loc="best", frameon=True, fontsize=8.6)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
