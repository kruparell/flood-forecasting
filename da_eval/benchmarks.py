"""Benchmarking utilities: Per-Basin Oracle synthesis, Global Best selection, and ANOVA variance decomposition."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from da_eval.ingestion import parse_da_config_id, _ensure_skill_score_columns, is_rho_config
from da_eval.selection import apply_selection, get_active, per_basin_timeseries

_WARNED = set()


def _warn_missing(cfg) -> None:
    if cfg not in _WARNED:
        _WARNED.add(cfg)
        print(f"[SELECTION] Global-Best {cfg!r} is not among the configs passed here; falling back to a local pick.")


def build_per_basin_optimal_eval(
    df_eval: pd.DataFrame,
    lead_time_ref: int = 1,
    metric_col: str = "NSE Skill Score",
    exclude_rho: bool = True,
) -> pd.DataFrame:
    """Synthesizes 'Per-Basin Best DA' benchmark by assigning each basin its optimal configuration

    (determined by maximizing performance at lead_time_ref, typically Lead 1 Day).
    PP post-processing (rho) configs are excluded by default: they are not DA.
    When a notebook-wide selection is active (``DAAnalysisSession.set_selection``) its per-basin choice is
    used instead and ``lead_time_ref`` / ``metric_col`` are ignored.
    """
    sel = get_active()
    if sel is not None and not df_eval.empty:
        return apply_selection(df_eval, sel)
    if df_eval.empty or (df_eval["Config ID"] == "Per-Basin Best DA").any():
        return df_eval

    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"

    # Filter out baseline or previous synthetic configs
    raw_eval = df_eval[
        (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ].copy()
    if exclude_rho:
        raw_eval = raw_eval[~raw_eval["Config ID"].apply(is_rho_config)]

    if raw_eval.empty:
        return df_eval

    # Identify best config per basin at reference lead time
    ref_records = raw_eval[raw_eval["Lead Time (Days)"] == lead_time_ref]
    if ref_records.empty:
        ref_records = raw_eval.copy()

    best_cfgs = (
        ref_records.sort_values(metric_col, ascending=False)
        .drop_duplicates(subset=["Basin ID"])[["Basin ID", "Config ID"]]
        .rename(columns={"Config ID": "Optimal_Config_ID"})
    )

    # Pull records across all lead times for the identified optimal configs
    merged = pd.merge(
        raw_eval,
        best_cfgs,
        left_on=["Basin ID", "Config ID"],
        right_on=["Basin ID", "Optimal_Config_ID"],
    )
    merged = merged.drop(columns=["Optimal_Config_ID"])
    merged["Config ID"] = "Per-Basin Best DA"

    combined = pd.concat([df_eval, merged], ignore_index=True)
    return combined


def build_per_basin_optimal_timeseries(
    df_ts: pd.DataFrame,
    df_eval: pd.DataFrame,
    lead_time_ref: int = 1,
    metric_col: str = "NSE Skill Score",
    exclude_rho: bool = True,
) -> pd.DataFrame:
    """Synthesizes 'Per-Basin Best DA' continuous forecast timeseries by matching each catchment

    to its highest performing configuration (the active notebook-wide selection when set).
    """
    sel = get_active()
    if sel is not None and not df_ts.empty:
        return per_basin_timeseries(df_ts, sel)
    if df_ts.empty or (df_ts["Config ID"] == "Per-Basin Best DA").any():
        return df_ts

    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"

    raw_eval = df_eval[
        (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ]
    if exclude_rho:
        raw_eval = raw_eval[~raw_eval["Config ID"].apply(is_rho_config)]
    ref_records = raw_eval[raw_eval["Lead Time (Days)"] == lead_time_ref]
    if ref_records.empty:
        ref_records = raw_eval

    best_cfgs = (
        ref_records.sort_values(metric_col, ascending=False)
        .drop_duplicates(subset=["Basin ID"])[["Basin ID", "Config ID"]]
        .rename(columns={"Config ID": "Optimal_Config_ID"})
    )

    merged = pd.merge(
        df_ts,
        best_cfgs,
        left_on=["Basin ID", "Config ID"],
        right_on=["Basin ID", "Optimal_Config_ID"],
    )
    merged = merged.drop(columns=["Optimal_Config_ID"])
    merged["Config ID"] = "Per-Basin Best DA"

    combined = pd.concat([df_ts, merged], ignore_index=True)
    return combined


def get_global_best_config(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
    exclude_rho: bool = True,
) -> str:
    """Identifies the single best DA configuration across all basins (highest median score).

    PP post-processing (rho) configs are excluded by default (use ``exclude_rho=False`` to include).

    When a notebook-wide selection is active (``DAAnalysisSession.set_selection``) this returns its
    Global-Best DA (or Global-Best PP when ``df_eval`` only holds PP configs) and ignores
    ``lead_time`` / ``metric_col``, so every plot shows the same model.
    """
    if df_eval is None or df_eval.empty or "Lead Time (Days)" not in df_eval.columns or "Config ID" not in df_eval.columns:
        return ""
    sel = get_active()
    if sel is not None:
        present = set(df_eval["Config ID"].dropna().astype(str))
        pool = [c for c in present if "per-basin" not in c.lower() and "baseline" not in c.lower()]
        only_rho = bool(pool) and all(is_rho_config(c) for c in pool)
        pick = sel.global_rho if (only_rho and not exclude_rho) else sel.global_da
        if pick and pick in present:
            return pick
        _warn_missing(pick)
    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"
    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ]
    if sub.empty:
        sub = df_eval[
            (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
            & (~df_eval["Config ID"].str.contains("Baseline", na=False))
        ]
    if exclude_rho:
        da_only = sub[~sub["Config ID"].apply(is_rho_config)]
        if not da_only.empty:
            sub = da_only
    if sub.empty:
        return str(df_eval["Config ID"].iloc[0])

    ranking = sub.groupby("Config ID")[metric_col].median().dropna()
    if ranking.empty:
        return str(sub["Config ID"].iloc[0])
    return str(ranking.idxmax())


def compute_anova_variance_decomposition(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
) -> pd.DataFrame:
    """Computes Type-II ANOVA decomposition of variance explained (%) across hyperparameter knobs."""
    if df_eval.empty or "Lead Time (Days)" not in df_eval.columns or "Config ID" not in df_eval.columns:
        return pd.DataFrame({
            "Factor": ["Residual / Basin Variation"],
            "Sum Sq": [1.0],
            "Variance Explained (%)": [100.0],
        })

    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"

    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ].copy()

    if sub.empty:
        sub = df_eval.copy()

    # Parse hyperparameters
    parsed_meta = sub["Config ID"].apply(parse_da_config_id)
    for col in parsed_meta.columns:
        sub[col] = parsed_meta[col]

    # Select parameters with more than 1 unique valid level
    factors = []
    for cand in ["Window (Days)", "Learning Rate", "Epochs", "BG Weight", "Target"]:
        if cand in sub.columns and sub[cand].dropna().nunique() > 1:
            factors.append(cand)

    if not factors:
        return pd.DataFrame({
            "Factor": ["Residual / Basin Variation"],
            "Sum Sq": [1.0],
            "Variance Explained (%)": [100.0],
        })

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Sanitize factor column names for patsy formula
        rename_map = {f: f.replace(" ", "_").replace("(", "").replace(")", "") for f in factors}
        sub_model = sub.rename(columns=rename_map).copy()
        clean_factors = [rename_map[f] for f in factors]

        formula = f"Q('{metric_col}') ~ " + " + ".join([f"C(Q('{cf}'))" for cf in clean_factors])
        model = ols(formula, data=sub_model).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        ss = anova_table["sum_sq"]
        total_ss = ss.sum()

        results = []
        for cf, orig in zip(clean_factors, factors):
            key = f"C(Q('{cf}'))"
            if key in anova_table.index:
                factor_ss = anova_table.loc[key, "sum_sq"]
                pct = (factor_ss / total_ss) * 100.0
                results.append({"Factor": orig, "Sum Sq": factor_ss, "Variance Explained (%)": pct})

        residual_ss = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 0.0
        results.append({
            "Factor": "Catchment Heterogeneity (Residual)",
            "Sum Sq": residual_ss,
            "Variance Explained (%)": (residual_ss / total_ss) * 100.0,
        })

        df_res = pd.DataFrame(results).sort_values("Variance Explained (%)", ascending=False)
        return df_res

    except Exception as e:
        # Fallback to sum of squares grouping
        total_var = sub[metric_col].var() * len(sub)
        results = []
        for f in factors:
            grp_means = sub.groupby(f)[metric_col].transform("mean")
            ss_factor = ((grp_means - sub[metric_col].mean()) ** 2).sum()
            pct = min(100.0, (ss_factor / max(1e-6, total_var)) * 100.0)
            results.append({"Factor": f, "Sum Sq": ss_factor, "Variance Explained (%)": pct})
        df_res = pd.DataFrame(results).sort_values("Variance Explained (%)", ascending=False)
        return df_res
