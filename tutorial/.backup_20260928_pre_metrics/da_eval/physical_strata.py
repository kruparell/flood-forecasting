"""Physical catchment characteristics stratification engine across 6 core dimensions:

1. Aridity Index (UNEP classification)
2. Baseline Performance Quality Tier
3. Seasonal Flow Regimes (High-Flow Wet Quarter vs Low-Flow Dry Quarter)
4. Catchment Scale (Area in km²)
5. Snow Fraction (Rain-fed vs Snow-melt)
6. Terrain Slope (Flat Lowlands vs Steep Relief)
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from da_eval import units


def classify_aridity_regimes(aridity_series: pd.Series) -> pd.Series:
    """Classifies catchments by UNEP Aridity Index (P / PET)."""
    bins = [-np.inf, 0.2, 0.5, 0.65, np.inf]
    labels = ["Arid (<0.2)", "Semi-Arid (0.2-0.5)", "Dry Sub-Humid (0.5-0.65)", "Humid (>=0.65)"]
    return pd.cut(aridity_series, bins=bins, labels=labels)


def classify_baseline_tiers(base_nse_series: pd.Series) -> pd.Series:
    """Partitions catchments into Baseline Quality Tiers (Error Remedy vs Operational Default)."""
    bins = [-np.inf, 0.0, 0.6, np.inf]
    labels = ["Poor (<0.0)", "Moderate (0.0-0.6)", "Strong (>=0.6)"]
    return pd.cut(base_nse_series, bins=bins, labels=labels)


def classify_catchment_area(area_series: pd.Series) -> pd.Series:
    """Classifies catchments by drainage area (km²)."""
    bins = [-np.inf, 500.0, 2500.0, np.inf]
    labels = ["Small (<500 km²)", "Medium (500-2500 km²)", "Large (>2500 km²)"]
    return pd.cut(area_series, bins=bins, labels=labels)


def classify_snow_fraction(snow_series: pd.Series) -> pd.Series:
    """Classifies catchments by cryosphere precipitation fraction (frac_snow)."""
    bins = [-np.inf, 0.10, 0.30, np.inf]
    labels = ["Rain-fed (<0.10)", "Mixed (0.10-0.30)", "Snow-melt (>=0.30)"]
    return pd.cut(snow_series, bins=bins, labels=labels)


def classify_terrain_slope(slope_series: pd.Series) -> pd.Series:
    """Classifies catchments by mean terrain slope (degrees)."""
    bins = [-np.inf, 2.0, 8.0, np.inf]
    labels = ["Flat (<2°)", "Rolling (2°-8°)", "Steep (>=8°)"]
    return pd.cut(slope_series, bins=bins, labels=labels)


def classify_precipitation_regimes(p_mean_series: pd.Series) -> pd.Series:
    """Classifies catchments by mean precipitation (mm/day)."""
    bins = [-np.inf, 1.5, 3.5, np.inf]
    labels = ["Low (<1.5 mm/d)", "Moderate (1.5-3.5 mm/d)", "High (>3.5 mm/d)"]
    return pd.cut(p_mean_series, bins=bins, labels=labels)


def classify_seasonal_flow_regimes(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Identifies climatological 3-month High-Flow (wet) and Low-Flow (dry) quarters per basin."""
    if df_ts.empty or "q_obs" not in df_ts.columns:
        return pd.DataFrame()

    sub = df_ts.dropna(subset=["q_obs"]).copy()
    if "Valid Date" not in sub.columns:
        return pd.DataFrame()

    sub["Month"] = sub["Valid Date"].dt.month
    monthly_mean = sub.groupby(["Basin ID", "Month"])["q_obs"].mean().reset_index()

    quarters = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }

    records = []
    for basin, grp in monthly_mean.groupby("Basin ID"):
        m_dict = grp.set_index("Month")["q_obs"].to_dict()
        q_means = {q: np.mean([m_dict.get(m, 0.0) for m in months]) for q, months in quarters.items()}
        high_q = max(q_means, key=q_means.get)
        low_q = min(q_means, key=q_means.get)
        records.append({"Basin ID": basin, "High Flow Quarter": high_q, "Low Flow Quarter": low_q})

    return pd.DataFrame(records)


def compute_consolidated_physical_strata(
    df_eval: pd.DataFrame,
    df_ts: Optional[pd.DataFrame] = None,
    df_meta: Optional[pd.DataFrame] = None,
    lead_time: int = 1,
    config_id: Optional[str] = None,
) -> pd.DataFrame:
    """Builds unified effect-size and reliability statistics across all 6 physical dimensions."""
    if df_eval.empty:
        return pd.DataFrame()

    from da_eval.ingestion import _ensure_skill_score_columns
    df_eval = _ensure_skill_score_columns(df_eval)

    sub = df_eval[df_eval["Lead Time (Days)"] == lead_time].copy()
    if sub.empty:
        sub = df_eval.copy()

    score_col = "NSE Skill Score" if "NSE Skill Score" in sub.columns else "NSE Delta"

    # Select configuration
    if config_id:
        df_target = sub[sub["Config ID"] == config_id].copy()
    elif (sub["Config ID"] == "Per-Basin Best DA").any():
        df_target = sub[sub["Config ID"] == "Per-Basin Best DA"].copy()
    else:
        best_cfg = sub.groupby("Config ID")[score_col].median().idxmax()
        df_target = sub[sub["Config ID"] == best_cfg].copy()

    df_target = df_target.drop_duplicates(subset=["Basin ID"])

    # Merge metadata if available
    if df_meta is not None and not df_meta.empty:
        df_target = pd.merge(df_target, df_meta, on="Basin ID", how="left")

    summary_rows = []

    def _calc_metrics(df_grp: pd.DataFrame, dim_name: str, cat_name: str):
        n = len(df_grp)
        if n == 0:
            return
        base_nse = df_grp["Base NSE"].median() if "Base NSE" in df_grp else np.nan
        da_nse = df_grp["DA NSE"].median() if "DA NSE" in df_grp else np.nan
        delta_nse = df_grp["NSE Delta"].median() if "NSE Delta" in df_grp else (da_nse - base_nse)
        skill_nse = df_grp["NSE Skill Score"].median() if "NSE Skill Score" in df_grp else delta_nse
        active_series = df_grp["NSE Skill Score"] if "NSE Skill Score" in df_grp else df_grp["NSE Delta"]
        q25 = active_series.quantile(0.25) if not active_series.empty else skill_nse
        q75 = active_series.quantile(0.75) if not active_series.empty else skill_nse

        pct_imp = (active_series > 0.01).mean() * 100.0 if not active_series.empty else np.nan
        pct_deg = (active_series < -0.01).mean() * 100.0 if not active_series.empty else np.nan

        summary_rows.append({
            "Dimension": dim_name,
            "Sub-Category": str(cat_name),
            "N Basins": int(n),
            "Median Base NSE": float(base_nse),
            "Median DA NSE": float(da_nse),
            "Median NSE Skill Score": float(skill_nse),
            "Median ΔNSE": float(delta_nse),
            "IQR Lower (25%)": float(q25),
            "IQR Upper (75%)": float(q75),
            "% Improved (SS>0.01)": float(pct_imp),
            "% Degraded (SS<-0.01)": float(pct_deg),
        })

    # 1. Baseline Quality Tier
    if "Base NSE" in df_target.columns:
        df_target["dim_base_tier"] = classify_baseline_tiers(df_target["Base NSE"])
        for cat, grp in df_target.groupby("dim_base_tier", observed=False):
            _calc_metrics(grp, "Baseline Tier", cat)

    # 2. Aridity Index
    if "unep_aridity_index" in df_target.columns and df_target["unep_aridity_index"].notna().any():
        df_target["dim_aridity"] = classify_aridity_regimes(df_target["unep_aridity_index"])
        for cat, grp in df_target.groupby("dim_aridity", observed=False):
            _calc_metrics(grp, "Aridity Index", cat)

    # 3. Catchment Scale (Area)
    if "area" in df_target.columns and df_target["area"].notna().any():
        df_target["dim_area"] = classify_catchment_area(df_target["area"])
        for cat, grp in df_target.groupby("dim_area", observed=False):
            _calc_metrics(grp, "Catchment Area", cat)

    # 4. Snow Fraction
    if "frac_snow" in df_target.columns and df_target["frac_snow"].notna().any():
        df_target["dim_snow"] = classify_snow_fraction(df_target["frac_snow"])
        for cat, grp in df_target.groupby("dim_snow", observed=False):
            _calc_metrics(grp, "Snow Fraction", cat)

    # 5. Terrain Slope
    if "slp_dg_sav" in df_target.columns and df_target["slp_dg_sav"].notna().any():
        df_target["dim_slope"] = classify_terrain_slope(df_target["slp_dg_sav"])
        for cat, grp in df_target.groupby("dim_slope", observed=False):
            _calc_metrics(grp, "Terrain Slope", cat)

    # 6. Seasonal Flow Regime (from timeseries if available)
    if df_ts is not None and not df_ts.empty and "q_obs" in df_ts.columns and "q_da" in df_ts.columns:
        df_regimes = classify_seasonal_flow_regimes(df_ts)
        if not df_regimes.empty:
            quarter_map = {
                12: "DJF", 1: "DJF", 2: "DJF",
                3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA",
                9: "SON", 10: "SON", 11: "SON",
            }
            ts_sub = df_ts[df_ts["Lead Time (Days)"] == lead_time].copy()
            if not ts_sub.empty:
                ts_sub["Quarter"] = ts_sub["Valid Date"].dt.month.map(quarter_map)
                ts_merged = pd.merge(ts_sub, df_regimes, on="Basin ID", how="inner")

                # High Flow Slice
                high_slice = ts_merged[ts_merged["Quarter"] == ts_merged["High Flow Quarter"]]
                # Low Flow Slice
                low_slice = ts_merged[ts_merged["Quarter"] == ts_merged["Low Flow Quarter"]]

                # Compute slice-level NSE deltas per basin
                def _calc_slice_deltas(s_df):
                    res = []
                    for b, g in s_df.groupby("Basin ID"):
                        if len(g) >= 10:
                            obs = g["q_obs"].values
                            base = g["q_base"].values
                            da = g["q_da"].values
                            var_obs = np.var(obs)
                            if var_obs > 1e-6:
                                nse_b = 1.0 - (np.mean((obs - base) ** 2) / var_obs)
                                nse_d = 1.0 - (np.mean((obs - da) ** 2) / var_obs)
                                denom = max(1.0 - nse_b, 1e-6)
                                ss = (nse_d - nse_b) / denom
                                res.append({"Basin ID": b, "Base NSE": nse_b, "DA NSE": nse_d, "NSE Delta": nse_d - nse_b, "NSE Skill Score": ss})
                    return pd.DataFrame(res)

                df_high = _calc_slice_deltas(high_slice)
                df_low = _calc_slice_deltas(low_slice)

    # 7. Average Precipitation
    if "p_mean" in df_target.columns and df_target["p_mean"].notna().any():
        df_target["dim_precip"] = classify_precipitation_regimes(df_target["p_mean"])
        for cat, grp in df_target.groupby("dim_precip", observed=False):
            _calc_metrics(grp, "Average Precipitation", cat)

    return pd.DataFrame(summary_rows)


def compute_basin_epoch_spreads(df_eval: pd.DataFrame, lead_time: int = 1) -> pd.DataFrame:
    """Computes epoch sensitivity spread (max DA NSE - min DA NSE across epochs) holding backbone constant."""
    sub = df_eval[df_eval["Lead Time (Days)"] == lead_time].copy()
    if sub.empty or "Config ID" not in sub.columns:
        return pd.DataFrame()

    def _parse_ep(cid):
        m = re.search(r"_ep(\d+)", str(cid))
        return int(m.group(1)) if m else np.nan

    def _parse_bb(cid):
        return re.sub(r"_ep\d+", "", str(cid))

    # Parse once per unique config (not once per row).
    uniq = pd.Series(sub["Config ID"].astype(str).unique())
    ep_map = dict(zip(uniq, uniq.map(_parse_ep)))
    bb_map = dict(zip(uniq, uniq.map(_parse_bb)))
    sub["Epoch"] = sub["Config ID"].astype(str).map(ep_map)
    sub["Backbone"] = sub["Config ID"].astype(str).map(bb_map)

    sub_ep = sub.dropna(subset=["Epoch"]).copy()
    if sub_ep.empty:
        return pd.DataFrame()

    # Vectorised equivalent of the per-basin loop:
    # 1) best backbone per basin = backbone with the highest max DA NSE (ties -> first backbone in sort order).
    bb_max = sub_ep.groupby(["Basin ID", "Backbone"], sort=True)["DA NSE"].max().dropna().reset_index()
    if bb_max.empty:
        return pd.DataFrame()
    bb_max["_neg"] = -bb_max["DA NSE"]
    best_bb = bb_max.sort_values(["Basin ID", "_neg", "Backbone"], kind="mergesort").drop_duplicates("Basin ID")[["Basin ID", "Backbone"]]
    grp = sub_ep.merge(best_bb, on=["Basin ID", "Backbone"]).sort_values(["Basin ID", "Epoch"], kind="mergesort")

    # 2) spread / epoch range within the best backbone.
    agg = grp.groupby("Basin ID").agg(
        n=("Epoch", "size"), nse_min=("DA NSE", "min"), nse_max=("DA NSE", "max"),
        ep_min=("Epoch", "min"), ep_max=("Epoch", "max"), bb=("Backbone", "first"))
    best_rows = grp.dropna(subset=["DA NSE"]).loc[lambda d: d.groupby("Basin ID")["DA NSE"].idxmax()].set_index("Basin ID")["Epoch"]
    first_ep = grp.groupby("Basin ID")["Epoch"].first()
    epoch_best = best_rows.reindex(agg.index).fillna(first_ep)

    out = pd.DataFrame({
        "Basin ID": agg.index,
        "Best Backbone": agg["bb"].values,
        "Min Epoch": agg["ep_min"].astype(int).values,
        "Max Epoch": agg["ep_max"].astype(int).values,
        "Epoch Spread (ΔNSE)": np.where(agg["n"].values >= 2, (agg["nse_max"] - agg["nse_min"]).values, 0.0).astype(float),
        "Epoch Best": epoch_best.astype(int).values,
    })
    return out.reset_index(drop=True)


GAP_COL = "PB−GB SS Gap"


def compute_gb_pb_gap(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    gb_cfg: Optional[str] = None,
    selection_lead: int = 1,
) -> pd.DataFrame:
    """Per-basin skill gap between Per-Basin Best DA and the Global Best DA config at ``lead_time``.

    Returns columns ``Basin ID``, ``Global Best SS``, ``Per-Basin SS``, ``PB−GB SS Gap`` (Per-Basin minus
    Global Best NSE skill score) and ``Per-Basin Config`` (the config chosen for the basin at
    ``selection_lead``, mirroring ``Per-Basin Best DA`` synthesis; AR(1)/rho configs excluded).
    A large gap flags basins where the globally chosen config is clearly sub-optimal.
    """
    from da_eval.benchmarks import get_global_best_config
    from da_eval.ingestion import _ensure_skill_score_columns, is_rho_config

    df_eval = _ensure_skill_score_columns(df_eval)
    col = "NSE Skill Score" if "NSE Skill Score" in df_eval.columns else "NSE Delta"
    gb_cfg = gb_cfg or get_global_best_config(df_eval, lead_time=lead_time)
    at_lead = df_eval[df_eval["Lead Time (Days)"] == lead_time]
    gb = at_lead[at_lead["Config ID"] == gb_cfg].drop_duplicates("Basin ID").set_index("Basin ID")[col]
    pb = at_lead[at_lead["Config ID"] == "Per-Basin Best DA"].drop_duplicates("Basin ID").set_index("Basin ID")[col]
    out = pd.DataFrame({"Global Best SS": gb, "Per-Basin SS": pb})
    out[GAP_COL] = out["Per-Basin SS"] - out["Global Best SS"]

    # Which config Per-Basin Best DA picked (argmax at the selection lead among DA configs).
    sel = df_eval[df_eval["Lead Time (Days)"] == selection_lead]
    cfgs = sel["Config ID"].astype(str)
    sel = sel[~cfgs.str.contains("Per-Basin|Baseline", na=False)]
    uniq = sel["Config ID"].astype(str).unique()
    rho = {c for c in uniq if is_rho_config(c)}
    sel = sel[~sel["Config ID"].isin(rho)]
    if not sel.empty:
        pick = sel.sort_values(col, ascending=False).drop_duplicates("Basin ID").set_index("Basin ID")["Config ID"]
        out["Per-Basin Config"] = pick.reindex(out.index)
    out.index.name = "Basin ID"
    return out.reset_index()



def find_candidate_basins(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    df_ts: Optional[pd.DataFrame] = None,
    category: Optional[str] = None,
    sub_category: Optional[str] = None,
    lead_time: int = 1,
    config_id: Optional[str] = None,
    min_delta: Optional[float] = None,
    max_delta: Optional[float] = None,
    min_base_nse: Optional[float] = None,
    max_base_nse: Optional[float] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    min_epoch_spread: Optional[float] = None,
    sort_by: str = "skill_desc",  # 'skill_desc', 'skill_asc', 'gap_desc', 'gap_asc', 'epoch_spread_desc'
    top_n: int = 10,
    min_gap: Optional[float] = None,
    max_gap: Optional[float] = None,
    gb_cfg: Optional[str] = None,
) -> pd.DataFrame:
    """Discovers and ranks candidate catchments meeting specified physical and performance criteria.

    ``min_gap`` / ``max_gap`` filter on the Per-Basin minus Global Best DA skill-score gap
    (``PB−GB SS Gap``); ``sort_by='gap_desc'`` ranks basins where the per-basin choice helps most.
    """
    from da_eval.ingestion import _ensure_skill_score_columns
    df_eval = _ensure_skill_score_columns(df_eval)

    sub = df_eval[df_eval["Lead Time (Days)"] == lead_time].copy()
    if sub.empty:
        return pd.DataFrame()

    score_col = "NSE Skill Score" if "NSE Skill Score" in sub.columns else "NSE Delta"

    if config_id:
        sub = sub[sub["Config ID"] == config_id].copy()
    elif (sub["Config ID"] == "Per-Basin Best DA").any():
        sub = sub[sub["Config ID"] == "Per-Basin Best DA"].copy()
    else:
        best_cfg = sub.groupby("Config ID")[score_col].median().idxmax()
        sub = sub[sub["Config ID"] == best_cfg].copy()

    # Merge metadata if available
    if df_meta is not None and not df_meta.empty:
        cols_to_merge = [c for c in df_meta.columns if c not in sub.columns or c == "Basin ID"]
        sub = pd.merge(sub, df_meta[cols_to_merge], on="Basin ID", how="left")

    # Add classifications
    if "unep_aridity_index" in sub.columns:
        sub["Aridity Regime"] = classify_aridity_regimes(sub["unep_aridity_index"])
    if "Base NSE" in sub.columns:
        sub["Baseline Tier"] = classify_baseline_tiers(sub["Base NSE"])
    if "area" in sub.columns:
        sub["Catchment Scale"] = classify_catchment_area(sub["area"])
    if "p_mean" in sub.columns:
        sub["Precipitation Regime"] = classify_precipitation_regimes(sub["p_mean"])

    # Global Best vs Per-Basin DA skill gap
    if gb_cfg is None and config_id and "Per-Basin" not in str(config_id):
        gb_cfg = config_id
    gap_df = compute_gb_pb_gap(df_eval, lead_time=lead_time, gb_cfg=gb_cfg)
    if not gap_df.empty:
        sub = pd.merge(sub, gap_df, on="Basin ID", how="left")

    # Epoch spreads only on request (not meaningful for most sweeps)
    if min_epoch_spread is not None or sort_by == "epoch_spread_desc":
        ep_df = compute_basin_epoch_spreads(df_eval, lead_time=lead_time)
        if not ep_df.empty:
            sub = pd.merge(sub, ep_df, on="Basin ID", how="left")
        else:
            sub["Epoch Spread (ΔNSE)"] = 0.0

    # Apply filters
    filtered = sub.copy()
    if category:
        cat_lower = category.lower()
        if "arid" in cat_lower and sub_category:
            filtered = filtered[filtered["Aridity Regime"].astype(str).str.lower().str.contains(sub_category.lower())]
        elif ("tier" in cat_lower or "base" in cat_lower or "initial" in cat_lower) and sub_category:
            filtered = filtered[filtered["Baseline Tier"].astype(str).str.lower().str.contains(sub_category.lower())]
        elif ("area" in cat_lower or "scale" in cat_lower) and sub_category:
            filtered = filtered[filtered["Catchment Scale"].astype(str).str.lower().str.contains(sub_category.lower())]
        elif ("precip" in cat_lower or "rain" in cat_lower) and sub_category:
            filtered = filtered[filtered["Precipitation Regime"].astype(str).str.lower().str.contains(sub_category.lower())]

    if min_delta is not None and score_col in filtered:
        filtered = filtered[filtered[score_col] >= min_delta]
    if max_delta is not None and score_col in filtered:
        filtered = filtered[filtered[score_col] <= max_delta]
    if min_base_nse is not None and "Base NSE" in filtered:
        filtered = filtered[filtered["Base NSE"] >= min_base_nse]
    if max_base_nse is not None and "Base NSE" in filtered:
        filtered = filtered[filtered["Base NSE"] <= max_base_nse]
    if min_area is not None and "area" in filtered:
        filtered = filtered[filtered["area"] >= min_area]
    if max_area is not None and "area" in filtered:
        filtered = filtered[filtered["area"] <= max_area]
    if min_epoch_spread is not None and "Epoch Spread (ΔNSE)" in filtered:
        filtered = filtered[filtered["Epoch Spread (ΔNSE)"] >= min_epoch_spread]
    if min_gap is not None and GAP_COL in filtered:
        filtered = filtered[filtered[GAP_COL] >= min_gap]
    if max_gap is not None and GAP_COL in filtered:
        filtered = filtered[filtered[GAP_COL] <= max_gap]

    # Sort
    if sort_by in ("skill_desc", "delta_desc"):
        filtered = filtered.sort_values(score_col, ascending=False)
    elif sort_by in ("skill_asc", "delta_asc"):
        filtered = filtered.sort_values(score_col, ascending=True)
    elif sort_by in ("gap_desc", "gap_asc") and GAP_COL in filtered:
        filtered = filtered.sort_values(GAP_COL, ascending=(sort_by == "gap_asc"))
    elif sort_by == "epoch_spread_desc" and "Epoch Spread (ΔNSE)" in filtered:
        filtered = filtered.sort_values("Epoch Spread (ΔNSE)", ascending=False)
    else:
        filtered = filtered.sort_values(score_col, ascending=False)

    cols = [
        "Basin ID",
        "gauge_name",
        "country",
        "Baseline Tier",
        "Aridity Regime",
        "Catchment Scale",
        "Precipitation Regime",
        "area",
        "unep_aridity_index",
        "p_mean",
        "Base NSE",
        "DA NSE",
        "NSE Skill Score",
        "NSE Delta",
        "Base KGE",
        "DA KGE",
        "KGE Skill Score",
        "Global Best SS",
        "Per-Basin SS",
        GAP_COL,
        "Per-Basin Config",
        "Epoch Spread (ΔNSE)",
        "Best Backbone",
    ]
    avail_cols = [c for c in cols if c in filtered.columns]
    res = filtered[avail_cols].head(top_n).copy()
    if "gauge_name" in res.columns:
        res = res.rename(columns={"gauge_name": "Basin Name"})
    return res


def get_category_candidates(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    df_ts: Optional[pd.DataFrame] = None,
    lead_time: int = 1,
    config_id: Optional[str] = None,
    top_per_category: int = 2,
) -> pd.DataFrame:
    """Discovers exemplary contrasting candidate catchments across the 5 core physical dimensions:

    1. Initial NSE (Baseline Tier: Poor, Moderate, Strong)
    2. Seasonal Flow (High-Flow Wet vs Low-Flow Dry)
    3. Aridity Regimes (Arid, Semi-Arid, Dry Sub-Humid, Humid)
    4. Catchment Scale (Small, Medium, Large)
    5. Average Precipitation (Low, Moderate, High)
    """
    from da_eval.ingestion import _ensure_skill_score_columns
    df_eval = _ensure_skill_score_columns(df_eval)

    sub = df_eval[df_eval["Lead Time (Days)"] == lead_time].copy()
    if sub.empty:
        return pd.DataFrame()

    score_col = "NSE Skill Score" if "NSE Skill Score" in sub.columns else "NSE Delta"

    if config_id:
        sub = sub[sub["Config ID"] == config_id].copy()
    elif (sub["Config ID"] == "Per-Basin Best DA").any():
        sub = sub[sub["Config ID"] == "Per-Basin Best DA"].copy()
    else:
        best_cfg = sub.groupby("Config ID")[score_col].median().idxmax()
        sub = sub[sub["Config ID"] == best_cfg].copy()

    if df_meta is not None and not df_meta.empty:
        cols_to_merge = [c for c in df_meta.columns if c not in sub.columns or c == "Basin ID"]
        sub = pd.merge(sub, df_meta[cols_to_merge], on="Basin ID", how="left")

    # Add classifications
    if "unep_aridity_index" in sub.columns:
        sub["Aridity Regime"] = classify_aridity_regimes(sub["unep_aridity_index"])
    if "Base NSE" in sub.columns:
        sub["Baseline Tier"] = classify_baseline_tiers(sub["Base NSE"])
    if "area" in sub.columns:
        sub["Catchment Scale"] = classify_catchment_area(sub["area"])
    if "p_mean" in sub.columns:
        sub["Precipitation Regime"] = classify_precipitation_regimes(sub["p_mean"])

    gap_df = compute_gb_pb_gap(
        df_eval, lead_time=lead_time,
        gb_cfg=config_id if config_id and "Per-Basin" not in str(config_id) else None)
    if not gap_df.empty:
        sub = pd.merge(sub, gap_df, on="Basin ID", how="left")

    records = []

    def _row_dict(dim: str, sub_cat: str, r: pd.Series) -> dict:
        return {
            "Dimension": dim,
            "Sub-Category": sub_cat,
            "Basin ID": r["Basin ID"],
            "Basin Name": r.get("gauge_name", r["Basin ID"]),
            "Country": r.get("country", "Unknown"),
            "Area (km²)": r.get("area", np.nan),
            "Aridity": r.get("unep_aridity_index", np.nan),
            "Precip (mm/d)": r.get("p_mean", np.nan),
            "Base NSE": r.get("Base NSE", np.nan),
            "DA NSE": r.get("DA NSE", np.nan),
            "NSE Skill Score": r.get("NSE Skill Score", np.nan),
            "ΔNSE": r.get("NSE Delta", np.nan),
            "Base KGE": r.get("Base KGE", np.nan),
            "DA KGE": r.get("DA KGE", np.nan),
            "Per-Basin SS": r.get("Per-Basin SS", np.nan),
            "PB−GB Gap": r.get(GAP_COL, np.nan),
        }

    # 1. Baseline Tier
    if "Baseline Tier" in sub.columns:
        for tier in ["Poor (<0.0)", "Moderate (0.0-0.6)", "Strong (>=0.6)"]:
            t_grp = sub[sub["Baseline Tier"] == tier].sort_values(score_col, ascending=False)
            for _, r in t_grp.head(top_per_category).iterrows():
                records.append(_row_dict("Initial NSE", tier, r))

    # 2. Aridity Regime
    if "Aridity Regime" in sub.columns:
        for arid in ["Arid (<0.2)", "Semi-Arid (0.2-0.5)", "Dry Sub-Humid (0.5-0.65)", "Humid (>=0.65)"]:
            a_grp = sub[sub["Aridity Regime"] == arid].sort_values(score_col, ascending=False)
            for _, r in a_grp.head(top_per_category).iterrows():
                records.append(_row_dict("Aridity", arid, r))

    # 3. Catchment Scale
    if "Catchment Scale" in sub.columns:
        for scale in ["Small (<500 km²)", "Medium (500-2500 km²)", "Large (>2500 km²)"]:
            s_grp = sub[sub["Catchment Scale"] == scale].sort_values(score_col, ascending=False)
            for _, r in s_grp.head(top_per_category).iterrows():
                records.append(_row_dict("Catchment Area", scale, r))

    # 4. Average Precipitation
    if "Precipitation Regime" in sub.columns:
        for p_reg in ["Low (<1.5 mm/d)", "Moderate (1.5-3.5 mm/d)", "High (>3.5 mm/d)"]:
            p_grp = sub[sub["Precipitation Regime"] == p_reg].sort_values(score_col, ascending=False)
            for _, r in p_grp.head(top_per_category).iterrows():
                records.append(_row_dict("Average Precipitation", p_reg, r))

    # 5. Seasonal Flow (High vs Low Flow)
    if df_ts is not None and not df_ts.empty and "q_obs" in df_ts.columns and "q_da" in df_ts.columns:
        df_regimes = classify_seasonal_flow_regimes(df_ts)
        if not df_regimes.empty:
            quarter_map = {
                12: "DJF", 1: "DJF", 2: "DJF",
                3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA",
                9: "SON", 10: "SON", 11: "SON",
            }
            ts_sub = df_ts[df_ts["Lead Time (Days)"] == lead_time].copy()
            if not ts_sub.empty:
                ts_sub["Quarter"] = ts_sub["Valid Date"].dt.month.map(quarter_map)
                ts_merged = pd.merge(ts_sub, df_regimes, on="Basin ID", how="inner")
                high_slice = ts_merged[ts_merged["Quarter"] == ts_merged["High Flow Quarter"]]
                low_slice = ts_merged[ts_merged["Quarter"] == ts_merged["Low Flow Quarter"]]

                def _slice_summary(s_df, label):
                    res = []
                    for b, g in s_df.groupby("Basin ID"):
                        if len(g) >= 10:
                            obs = g["q_obs"].values
                            base = g["q_base"].values
                            da = g["q_da"].values
                            var_obs = np.var(obs)
                            if var_obs > 1e-6:
                                nse_b = 1.0 - (np.mean((obs - base) ** 2) / var_obs)
                                nse_d = 1.0 - (np.mean((obs - da) ** 2) / var_obs)
                                denom = max(1.0 - nse_b, 1e-6)
                                ss = (nse_d - nse_b) / denom
                                res.append({"Basin ID": b, "Base NSE": nse_b, "DA NSE": nse_d, "NSE Skill Score": ss, "NSE Delta": nse_d - nse_b})
                    if res:
                        df_res = pd.DataFrame(res).sort_values("NSE Skill Score", ascending=False)
                        top_b = df_res.head(top_per_category)
                        for _, r in top_b.iterrows():
                            b_id = r["Basin ID"]
                            meta_row = sub[sub["Basin ID"] == b_id]
                            g_name = meta_row["gauge_name"].iloc[0] if not meta_row.empty and "gauge_name" in meta_row else b_id
                            cntry = meta_row["country"].iloc[0] if not meta_row.empty and "country" in meta_row else "Unknown"
                            ar = meta_row["area"].iloc[0] if not meta_row.empty and "area" in meta_row else np.nan
                            ai = meta_row["unep_aridity_index"].iloc[0] if not meta_row.empty and "unep_aridity_index" in meta_row else np.nan
                            pm = meta_row["p_mean"].iloc[0] if not meta_row.empty and "p_mean" in meta_row else np.nan
                            pb_ss = meta_row["Per-Basin SS"].iloc[0] if not meta_row.empty and "Per-Basin SS" in meta_row else np.nan
                            gap_v = meta_row[GAP_COL].iloc[0] if not meta_row.empty and GAP_COL in meta_row else np.nan
                            records.append({
                                "Dimension": "High vs Low Flow",
                                "Sub-Category": label,
                                "Basin ID": b_id,
                                "Basin Name": g_name,
                                "Country": cntry,
                                "Area (km²)": ar,
                                "Aridity": ai,
                                "Precip (mm/d)": pm,
                                "Base NSE": r["Base NSE"],
                                "DA NSE": r["DA NSE"],
                                "NSE Skill Score": r["NSE Skill Score"],
                                "ΔNSE": r["NSE Delta"],
                                "Base KGE": np.nan,
                                "DA KGE": np.nan,
                                "Per-Basin SS": pb_ss,
                                "PB−GB Gap": gap_v,
                            })

                _slice_summary(high_slice, "High-Flow Quarter")
                _slice_summary(low_slice, "Low-Flow Quarter")

    return pd.DataFrame(records)


def style_candidate_gauges_table(df: pd.DataFrame):
    """Styles the candidate gauges table with colored deltas/skill scores and bold labels."""
    if df.empty:
        return df

    def _color_delta(val):
        if pd.isna(val):
            return ""
        if val > 0.05:
            return "color: #137333; font-weight: bold;"  # dark green
        elif val >= 0.0:
            return "color: #174ea6; font-weight: bold;"  # blue
        else:
            return "color: #c5221f; font-weight: bold;"  # red

    fmt = {
        "Area (km²)": "{:,.1f}",
        "Aridity": "{:.2f}",
        "Precip (mm/d)": "{:.2f}",
        "Base NSE": "{:.3f}",
        "DA NSE": "{:.3f}",
        "NSE Skill Score": units.ss_formatter(),
        "ΔNSE": "{:+.3f}",
        "Base KGE": "{:.3f}",
        "DA KGE": "{:.3f}",
        "KGE Skill Score": "{:+.3f}",
        "Global Best SS": units.ss_formatter(),
        "Per-Basin SS": units.ss_formatter(),
        "PB−GB Gap": units.ss_formatter(),
        GAP_COL: units.ss_formatter(),
    }
    actual_fmt = {k: v for k, v in fmt.items() if k in df.columns}
    target_subsets = [c for c in ["NSE Skill Score", "ΔNSE"] if c in df.columns]

    styler = (
        df.style
        .format(actual_fmt, na_rep="—")
        .applymap(_color_delta, subset=target_subsets if target_subsets else None)
        .set_properties(**{"text-align": "center", "font-size": "11px", "padding": "4px 8px"})
        .set_properties(subset=["Basin ID", "Basin Name"], **{"text-align": "left", "font-weight": "bold"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#1a73e8"), ("color", "white"), ("font-weight", "bold"), ("text-align", "center"), ("padding", "6px 10px")]},
            {"selector": "tr:hover", "props": [("background-color", "#f1f3f4")]},
        ])
    )
    return styler


# ==============================================================================
# Dynamic Catchment Stratification & Stratified Leaderboard Suite
# ==============================================================================

def filter_cohort_by_attributes(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    base_nse_range: Optional[Tuple[float, float]] = None,
    aridity_range: Optional[Tuple[float, float]] = None,
    p_mean_range: Optional[Tuple[float, float]] = None,
    area_range: Optional[Tuple[float, float]] = None,
    snow_range: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filters evaluation dataset by arbitrary physical attribute ranges (Aridity, Precip, Base NSE, Area, Snow).

    Returns:
        (df_eval_filtered, cohort_info_dict)
    """
    if df_eval.empty:
        return df_eval, {"n_basins": 0, "total_basins": 0, "pct_total": 0.0, "desc": "No data"}

    total_basins = int(df_eval["Basin ID"].nunique())
    work_df = df_eval.copy()

    # Merge metadata if needed
    meta_cols = ["unep_aridity_index", "p_mean", "area", "frac_snow", "slp_dg_sav"]
    if df_meta is not None and not df_meta.empty:
        missing_cols = [c for c in meta_cols if c in df_meta.columns and c not in work_df.columns]
        if missing_cols:
            merge_cols = ["Basin ID"] + missing_cols
            work_df = pd.merge(work_df, df_meta[merge_cols].drop_duplicates("Basin ID"), on="Basin ID", how="left")

    active_filters = []
    basin_mask = pd.Series(True, index=work_df["Basin ID"].unique())

    # Build basin-level attribute table for clean filtering
    basin_attr_df = work_df.drop_duplicates("Basin ID").set_index("Basin ID")

    # 1. Base NSE filter
    if base_nse_range is not None and "Base NSE" in basin_attr_df.columns:
        b_min, b_max = float(base_nse_range[0]), float(base_nse_range[1])
        cond = (basin_attr_df["Base NSE"] >= b_min) & (basin_attr_df["Base NSE"] <= b_max)
        basin_mask = basin_mask & cond
        active_filters.append(f"Base NSE ∈ [{b_min:.2f}, {b_max:.2f}]")

    # 2. Aridity filter (unep_aridity_index = P / PET)
    if aridity_range is not None and "unep_aridity_index" in basin_attr_df.columns:
        a_min, a_max = float(aridity_range[0]), float(aridity_range[1])
        cond = (basin_attr_df["unep_aridity_index"] >= a_min) & (basin_attr_df["unep_aridity_index"] <= a_max)
        basin_mask = basin_mask & cond
        active_filters.append(f"Aridity ∈ [{a_min:.2f}, {a_max:.2f}]")

    # 3. Mean Precipitation filter (mm/day)
    if p_mean_range is not None and "p_mean" in basin_attr_df.columns:
        p_min, p_max = float(p_mean_range[0]), float(p_mean_range[1])
        cond = (basin_attr_df["p_mean"] >= p_min) & (basin_attr_df["p_mean"] <= p_max)
        basin_mask = basin_mask & cond
        active_filters.append(f"Precip ∈ [{p_min:.1f}, {p_max:.1f}] mm/d")

    # 4. Drainage Area filter (km²)
    if area_range is not None and "area" in basin_attr_df.columns:
        ar_min, ar_max = float(area_range[0]), float(area_range[1])
        cond = (basin_attr_df["area"] >= ar_min) & (basin_attr_df["area"] <= ar_max)
        basin_mask = basin_mask & cond
        active_filters.append(f"Area ∈ [{ar_min:,.0f}, {ar_max:,.0f}] km²")

    # 5. Snow Fraction filter
    if snow_range is not None and "frac_snow" in basin_attr_df.columns:
        s_min, s_max = float(snow_range[0]), float(snow_range[1])
        cond = (basin_attr_df["frac_snow"] >= s_min) & (basin_attr_df["frac_snow"] <= s_max)
        basin_mask = basin_mask & cond
        active_filters.append(f"Snow Frac ∈ [{s_min:.2f}, {s_max:.2f}]")

    matched_basins = set(basin_attr_df[basin_mask].index)
    filtered_df = work_df[work_df["Basin ID"].isin(matched_basins)].copy()

    n_matched = len(matched_basins)
    pct = (n_matched / max(1, total_basins)) * 100.0
    desc = " & ".join(active_filters) if active_filters else "All Basins (Unfiltered)"

    # Compute cohort summary statistics
    matched_attrs = basin_attr_df.loc[list(matched_basins)] if n_matched > 0 else pd.DataFrame()
    med_base = float(matched_attrs["Base NSE"].median()) if "Base NSE" in matched_attrs and not matched_attrs.empty else np.nan
    med_arid = float(matched_attrs["unep_aridity_index"].median()) if "unep_aridity_index" in matched_attrs and not matched_attrs.empty else np.nan
    med_precip = float(matched_attrs["p_mean"].median()) if "p_mean" in matched_attrs and not matched_attrs.empty else np.nan

    cohort_info = {
        "n_basins": n_matched,
        "total_basins": total_basins,
        "pct_total": pct,
        "desc": desc,
        "median_base_nse": med_base,
        "median_aridity": med_arid,
        "median_p_mean": med_precip,
        "matched_basins": matched_basins,
    }
    return filtered_df, cohort_info


def build_stratified_leaderboard_table(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    lead_time: int = 1,
    metric: str = "skill_nse",
    base_nse_range: Optional[Tuple[float, float]] = None,
    aridity_range: Optional[Tuple[float, float]] = None,
    p_mean_range: Optional[Tuple[float, float]] = None,
    area_range: Optional[Tuple[float, float]] = None,
    snow_range: Optional[Tuple[float, float]] = None,
    top_n: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ranks all configurations strictly on a physically-filtered catchment stratum.

    Returns:
        (leaderboard_df, cohort_info_dict)
    """
    filtered_df, cohort_info = filter_cohort_by_attributes(
        df_eval=df_eval,
        df_meta=df_meta,
        base_nse_range=base_nse_range,
        aridity_range=aridity_range,
        p_mean_range=p_mean_range,
        area_range=area_range,
        snow_range=snow_range,
    )

    if filtered_df.empty or cohort_info["n_basins"] == 0:
        return pd.DataFrame(), cohort_info

    sub = filtered_df[filtered_df["Lead Time (Days)"] == lead_time].copy()
    if sub.empty:
        sub = filtered_df.copy()

    from da_eval.ingestion import _ensure_skill_score_columns
    sub = _ensure_skill_score_columns(sub)

    is_kge = "kge" in metric.lower()
    is_skill = "skill" in metric.lower()
    base_col = "Base KGE" if is_kge else "Base NSE"
    da_col = "DA KGE" if is_kge else "DA NSE"
    delta_col = "KGE Delta" if is_kge else "NSE Delta"
    skill_col = "KGE Skill Score" if is_kge else "NSE Skill Score"
    sort_col = skill_col if is_skill else delta_col
    metric_label = "KGE" if is_kge else "NSE"

    # Separate sweep configurations from oracle/synthetic benchmarks
    sweep_sub = sub[
        (~sub["Config ID"].str.contains("Per-Basin", na=False))
        & (~sub["Config ID"].str.contains("Baseline", na=False))
    ].copy()

    records = []
    for cfg, grp in sweep_sub.groupby("Config ID"):
        n_pts = len(grp)
        if n_pts == 0:
            continue
        base_med = grp[base_col].median() if base_col in grp else np.nan
        da_med = grp[da_col].median() if da_col in grp else np.nan
        delta_vals = grp[delta_col].dropna().values if delta_col in grp else np.array([])
        skill_vals = grp[skill_col].dropna().values if skill_col in grp else np.array([])
        active_vals = skill_vals if is_skill else delta_vals

        if len(delta_vals) > 0:
            delta_med = np.median(delta_vals)
        else:
            delta_med = np.nan

        if len(skill_vals) > 0:
            skill_med = np.median(skill_vals)
        else:
            skill_med = np.nan

        if len(active_vals) > 0:
            active_med = np.median(active_vals)
            q25 = np.percentile(active_vals, 25)
            q75 = np.percentile(active_vals, 75)
            win_rate = (active_vals > 0.01).mean() * 100.0
            deg_rate = (active_vals < -0.01).mean() * 100.0
        else:
            active_med, q25, q75, win_rate, deg_rate = np.nan, np.nan, np.nan, np.nan, np.nan

        iqr_label = f"Skill {metric_label} IQR [25%..75%]" if is_skill else f"Δ{metric_label} IQR [25%..75%]"
        records.append({
            "Config ID": cfg,
            f"Median Base {metric_label}": float(base_med),
            f"Median DA {metric_label}": float(da_med),
            f"Median Δ{metric_label}": float(delta_med),
            f"Median {metric_label} Skill": float(skill_med),
            iqr_label: (f"[{units.fmt_ss(q25)} .. {units.fmt_ss(q75)}]" if (is_skill and not is_kge)
                        else f"[{q25:+.3f} .. {q75:+.3f}]") if np.isfinite(q25) else "—",
            "Win Rate (Δ>0.01)": float(win_rate),
            "Degraded (Δ<-0.01)": float(deg_rate),
            "Basins": int(n_pts),
            "_sort_key": float(active_med) if np.isfinite(active_med) else -999.0,
        })

    if not records:
        return pd.DataFrame(), cohort_info

    df_lb = pd.DataFrame(records).sort_values("_sort_key", ascending=False).reset_index(drop=True)
    df_lb.insert(0, "Rank", range(1, len(df_lb) + 1))
    df_lb["Config ID"] = df_lb.apply(
        lambda r: f"🥇 {r['Config ID']}" if r["Rank"] == 1 else (f"🥈 {r['Config ID']}" if r["Rank"] == 2 else (f"🥉 {r['Config ID']}" if r["Rank"] == 3 else r["Config ID"])),
        axis=1,
    )
    df_lb = df_lb.drop(columns=["_sort_key"])

    if top_n is not None and top_n > 0:
        df_lb = df_lb.head(top_n)

    return df_lb, cohort_info

