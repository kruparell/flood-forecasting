"""Publication-ready static visualization suite for DA Evaluation.

Features:
- Dual-Panel ECDF Suite:
  1. Fixed Horizon (Lead 1): All Sweep Configurations + Benchmarks
  2. Multi-Horizon Progression: Baseline, Global Best, and Per-Basin Best across all leadtimes
- Skill & Memory Persistence Decay curves
- Unified 2x3 Physical Characteristics Diverging Bar Chart with text offset above error bars
"""

import glob
import os
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

from da_eval.benchmarks import get_global_best_config
from da_eval.selection import get_active
from da_eval.ingestion import detect_lead_times
from da_eval.ingestion import is_rho_config as _is_rho
from da_eval import units

_SS_COLS = ("NSE Skill Score", "Median NSE Skill Score")


def _is_ss(col: Optional[str]) -> bool:
    """True when ``col`` is the NSE skill score (displayed via ``units``); KGE SS / ΔNSE / NSE are untouched."""
    return col in _SS_COLS


def _fmt_val(v, is_ss: bool, sign: bool = True) -> str:
    """Annotation text for one metric value: ``units.fmt_ss`` for SS_NSE, else the legacy 3-dp format."""
    if is_ss:
        return units.fmt_ss(v, sign=sign)
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def _zero_ref_label(short_lbl: str, is_ss: bool) -> str:
    """Legend label of the zero (open-loop) reference line on delta / skill-score axes."""
    if "Skill" not in short_lbl:
        return "Baseline Reference (Δ = 0.000)"
    zero = units.fmt_ss(0.0, sign=False) if is_ss else "0.000"
    return f"Baseline Reference (SS = {zero})"


def _finite_or_default(values, reducer, default):
    """Reduces values ignoring NaN/Inf, returning default when nothing is finite."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return reducer(finite) if finite else default


def _resolve_metric_spec(
    metric: Optional[str] = None,
    metric_col: str = "DA NSE",
    base_metric_col: str = "Base NSE",
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[bool, str, Optional[str], str, str, str, Tuple[float, float]]:
    """Resolves metric mode ('NSE', 'delta_nse', 'skill_nse', 'KGE', 'delta_kge', 'skill_kge') into column and axis specs."""
    m_norm = (metric or "").strip().lower().replace(" ", "_")
    if not m_norm:
        if "skill" in metric_col.lower():
            m_norm = "skill_kge" if "kge" in metric_col.lower() else "skill_nse"
        elif "delta" in metric_col.lower():
            m_norm = "delta_kge" if "kge" in metric_col.lower() else "delta_nse"
        elif "kge" in metric_col.lower():
            m_norm = "kge"
        else:
            m_norm = "nse"

    if m_norm in ("skill_nse", "nse_skill", "nse_skill_score", "skill_score", "ss_nse", "skill"):
        is_delta = True
        m_col = "NSE Skill Score"
        b_col = None
        rank_col = "NSE Skill Score"
        short_lbl = "NSE Skill Score"
        axis_lbl = "Per-Gauge NSE Skill Score [1 − (1 − DA NSE) / (1 − Base NSE)]"
        resolved_xlim = xlim if xlim is not None else (-0.5, 0.5)
    elif m_norm in ("skill_kge", "kge_skill", "kge_skill_score", "ss_kge"):
        is_delta = True
        m_col = "KGE Skill Score"
        b_col = None
        rank_col = "KGE Skill Score"
        short_lbl = "KGE Skill Score"
        axis_lbl = "Per-Gauge KGE Skill Score [1 − (1 − DA KGE) / (1 − Base KGE)]"
        resolved_xlim = xlim if xlim is not None else (-0.5, 0.5)
    elif m_norm in ("delta_nse", "dnse", "Δnse", "nse_delta"):
        is_delta = True
        m_col = "NSE Delta"
        b_col = None
        rank_col = "NSE Delta"
        short_lbl = "ΔNSE"
        axis_lbl = "Per-Gauge ΔNSE (DA − Baseline)"
        resolved_xlim = xlim if xlim is not None else (-0.5, 0.5)
    elif m_norm in ("delta_kge", "dkge", "Δkge", "kge_delta"):
        is_delta = True
        m_col = "KGE Delta"
        b_col = None
        rank_col = "KGE Delta"
        short_lbl = "ΔKGE"
        axis_lbl = "Per-Gauge ΔKGE (DA − Baseline)"
        resolved_xlim = xlim if xlim is not None else (-0.5, 0.5)
    elif m_norm in ("kge", "da_kge", "abs_kge", "absolute_kge"):
        is_delta = False
        m_col = "DA KGE"
        b_col = "Base KGE"
        rank_col = "KGE Delta"
        short_lbl = "KGE"
        axis_lbl = "Catchment Performance (KGE)"
        resolved_xlim = xlim if xlim is not None else (-1.0, 1.0)
    else:
        is_delta = False
        m_col = metric_col if metric is None else "DA NSE"
        b_col = base_metric_col if metric is None else "Base NSE"
        rank_col = "NSE Delta"
        short_lbl = "NSE"
        axis_lbl = f"Catchment Performance ({m_col})"
        resolved_xlim = xlim if xlim is not None else (-1.0, 1.0)

    return is_delta, m_col, b_col, rank_col, short_lbl, axis_lbl, resolved_xlim


ECDF_MODELS = ("global_da", "per_basin_da", "global_rho", "per_basin_rho")
ECDF_MODEL_LABELS = {
    "global_da": "Global-Best DA",
    "per_basin_da": "Per-Basin DA",
    "global_rho": "Global-Best AR(1)",
    "per_basin_rho": "Per-Basin AR(1)",
}
DEFAULT_ECDF_SHOW = ("global_da", "per_basin_da")


def resolve_ecdf_models(work_df: pd.DataFrame, ref_lead: int, rank_col: str, global_best_cfg: str,
                        show=DEFAULT_ECDF_SHOW) -> List[dict]:
    """Benchmark curves for the ECDF / decay plots, in draw order.

    ``show`` picks from ``ECDF_MODELS``. All reference models follow the notebook-wide selection
    (``session.set_selection``); without one, the fallback ranks at ``ref_lead``.
    """
    show = list(DEFAULT_ECDF_SHOW if show is None else show)
    cfgs = work_df["Config ID"].dropna().unique()
    out = []
    for key in ECDF_MODELS:
        if key not in show:
            continue
        if key == "global_da" and global_best_cfg:
            out.append(dict(key=key, cfg=global_best_cfg, long=f"Global Best DA ({global_best_cfg})",
                            short="Global Best", color="#1a73e8", ls="-", z=5))
        elif key == "per_basin_da" and "Per-Basin Best DA" in cfgs:
            out.append(dict(key=key, cfg="Per-Basin Best DA", long="Per-Basin Best DA (Oracle)",
                            short="Oracle", color="#0d47a1", ls="--", z=6))
        elif key == "global_rho":
            fixed = [c for c in cfgs if _is_rho(c) and "per_basin" not in c.lower() and "per-basin" not in c.lower()]
            sub = work_df[work_df["Config ID"].isin(fixed)]
            col = rank_col if rank_col in sub.columns else "NSE Delta"
            cfg = get_global_best_config(sub, lead_time=ref_lead, metric_col=col, exclude_rho=False) if fixed else ""
            if cfg:
                rho = cfg.split("rho")[-1]
                out.append(dict(key=key, cfg=cfg, long=f"Global-Best AR(1) (ρ={rho})",
                                short=f"AR(1) ρ={rho}", color="#e8710a", ls="-", z=4.6))
        elif key == "per_basin_rho":
            pb = [c for c in cfgs if _is_rho(c) and ("per_basin" in c.lower() or "per-basin" in c.lower())]
            if pb:
                out.append(dict(key=key, cfg=pb[0], long="Per-Basin AR(1) (best ρ per gauge)",
                                short="Per-Basin AR(1)", color="#b3261e", ls="--", z=4.8))
    return out


def plot_ecdf_suite(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    progression_leads: Optional[List[int]] = None,
    metric: Optional[str] = None,
    metric_col: str = "DA NSE",
    base_metric_col: str = "Base NSE",
    figsize: Tuple[int, int] = (18, 6.5),
    xlim: Optional[Tuple[float, float]] = None,
    show=DEFAULT_ECDF_SHOW,
    show_sweep: bool = True,
) -> plt.Figure:
    """Renders 2-Panel Empirical Cumulative Distribution Function (ECDF) Suite:

    Supports Absolute Metrics ('NSE', 'KGE'), Per-Gauge Gains ('delta_nse', 'delta_kge'),
    and Skill Scores ('skill_nse', 'skill_kge'):
    - Panel A: Fixed Lead Time (All Configs vs Baseline, Global Best, Per-Basin Best)
    - Panel B: Multi-Horizon Progression across Key Horizons (e.g. t+1 and t+7)
    """
    from da_eval.benchmarks import build_per_basin_optimal_eval
    from da_eval.ingestion import _ensure_skill_score_columns

    if df_eval.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No evaluation records available", ha="center", va="center")
        return fig

    is_delta, m_col, b_col, rank_col, short_lbl, axis_lbl, resolved_xlim = _resolve_metric_spec(
        metric=metric, metric_col=metric_col, base_metric_col=base_metric_col, xlim=xlim
    )

    lead_times = detect_lead_times(df_eval)
    ref_lead = lead_time if lead_time in lead_times else lead_times[0]

    work_df = _ensure_skill_score_columns(df_eval.copy())
    if rank_col in work_df.columns and rank_col != "NSE Delta":
        base_only = work_df[work_df["Config ID"] != "Per-Basin Best DA"].copy()
        work_df = build_per_basin_optimal_eval(base_only, lead_time_ref=ref_lead, metric_col=rank_col)

    global_best_cfg = get_global_best_config(work_df, lead_time=ref_lead, metric_col=rank_col if rank_col in work_df.columns else "NSE Delta")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -------------------------------------------------------------------------
    # Panel A: Fixed Lead Time (All Configs vs Benchmarks)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    sub_lead = work_df[work_df["Lead Time (Days)"] == ref_lead].copy()
    if sub_lead.empty:
        sub_lead = work_df.copy()

    # 1. Sweep Configs (Thin semi-transparent curves; DA configs only — AR(1) post-processors are benchmarks)
    configs = [c for c in sub_lead["Config ID"].unique()
               if "Per-Basin" not in c and "Baseline" not in c and c != global_best_cfg and not _is_rho(c)]
    if show_sweep and configs and m_col in sub_lead.columns:
        cmap = cm.get_cmap("tab20", max(1, len(configs)))
        for idx, cfg in enumerate(configs):
            vals = np.sort(sub_lead[sub_lead["Config ID"] == cfg][m_col].dropna().values)
            if len(vals) > 0:
                y = np.linspace(0, 1, len(vals))
                ax1.plot(vals, y, color=cmap(idx), alpha=0.32, linewidth=1.0)
        ax1.plot([], [], color="#80868b", alpha=0.6, linewidth=1.2, label=f"Sweep Configs ({len(configs)} variations)")

    is_ss = _is_ss(m_col)
    ref_zero_lbl = _zero_ref_label(short_lbl, is_ss)

    # 2. Baseline Open-Loop (Curve for Absolute, Vertical Zero Line for Delta/Skill)
    if is_delta:
        ax1.axvline(0.0, color="#3c4043", linewidth=2.2, linestyle="--", label=ref_zero_lbl, zorder=4)
        ax1.axhline(0.5, color="#9aa0a6", linewidth=0.9, linestyle=":", alpha=0.7)
    elif b_col and b_col in sub_lead.columns:
        base_vals = np.sort(sub_lead.drop_duplicates("Basin ID")[b_col].dropna().values)
        if len(base_vals) > 0:
            y_base = np.linspace(0, 1, len(base_vals))
            ax1.plot(base_vals, y_base, label=f"Baseline Open-Loop (Med: {np.median(base_vals):.3f})", color="#3c4043", linewidth=2.8, linestyle="-")

    # 3. Benchmark models (Global / Per-Basin DA and AR(1) post-processors), as selected in ``show``.
    models = resolve_ecdf_models(work_df, ref_lead, rank_col, global_best_cfg, show)
    if m_col in sub_lead.columns:
        for mdl in models:
            vals = np.sort(sub_lead[sub_lead["Config ID"] == mdl["cfg"]][m_col].dropna().values)
            if len(vals) == 0:
                continue
            y = np.linspace(0, 1, len(vals))
            if is_delta:
                lbl = f"{mdl['long']} (Med: {_fmt_val(np.median(vals), is_ss)} | Win: {(vals > 0).mean() * 100.0:.1f}%)"
            else:
                lbl = f"{mdl['long']} (Med: {np.median(vals):.3f})"
            ax1.plot(vals, y, label=lbl, color=mdl["color"], linewidth=3.0, linestyle=mdl["ls"], zorder=mdl["z"])

    ax1.set_title(f"A. Fixed Horizon (Lead t+{ref_lead}) — Per-Gauge {short_lbl} CDF", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel(units.ss_label(axis_lbl) if is_ss else axis_lbl, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Gauge Fraction (ECDF)", fontsize=11, fontweight="bold")
    ax1.set_xlim(*resolved_xlim)
    if is_ss:
        units.ss_axis(ax1, "x")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.4 if len(models) > 2 else 9.2)

    # -------------------------------------------------------------------------
    # Panel B: Multi-Horizon Progression across All Horizons (e.g. t+1 .. t+7)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    if progression_leads is None:
        progression_leads = list(lead_times)
    progression_leads = sorted(list(set(progression_leads)))
    n_prog = len(progression_leads)

    def _lead_alpha(idx: int) -> float:
        if n_prog <= 1:
            return 1.0
        # Progressive fade: 1.0 at earliest lead (t+1) down to 0.22 at latest lead (t+7)
        return float(1.0 - 0.78 * (idx / (n_prog - 1)))

    def _lead_lw(idx: int, plt_lead: int, base_lw: float = 2.6) -> float:
        if plt_lead == ref_lead:
            return base_lw + 0.4
        if n_prog <= 1:
            return base_lw
        return float(base_lw - 0.9 * (idx / (n_prog - 1)))

    if is_delta:
        ax2.axvline(0.0, color="#3c4043", linewidth=2.2, linestyle="--", label=ref_zero_lbl, zorder=4)
        ax2.axhline(0.5, color="#9aa0a6", linewidth=0.9, linestyle=":", alpha=0.7)
    else:
        # 1. Baseline Open-Loop across all progression leads with progressive alpha
        for b_idx, plt_lead in enumerate(progression_leads):
            base_sub = work_df[work_df["Lead Time (Days)"] == plt_lead]
            if b_col and b_col in base_sub.columns:
                base_vals = np.sort(base_sub.drop_duplicates("Basin ID")[b_col].dropna().values)
                if len(base_vals) > 0:
                    y_b = np.linspace(0, 1, len(base_vals))
                    a_val = _lead_alpha(b_idx)
                    lw_val = _lead_lw(b_idx, plt_lead, base_lw=2.4)
                    lbl_b = f"Baseline (t+{plt_lead}) (Med: {np.median(base_vals):.3f})"
                    ax2.plot(
                        base_vals,
                        y_b,
                        label=lbl_b,
                        color="#3c4043",
                        alpha=a_val,
                        linewidth=lw_val,
                        linestyle="-",
                        zorder=4,
                    )

    # 2. Selected benchmark models across all progression leads (progressive alpha: darker -> lighter)
    if m_col in work_df.columns:
        for m_idx, mdl in enumerate(models):
            for l_idx, plt_lead in enumerate(progression_leads):
                l_sub = work_df[work_df["Lead Time (Days)"] == plt_lead]
                vals = np.sort(l_sub[l_sub["Config ID"] == mdl["cfg"]][m_col].dropna().values)
                if len(vals) == 0:
                    continue
                if is_delta:
                    lbl = f"{mdl['short']} (t+{plt_lead}) (Med: {_fmt_val(np.median(vals), is_ss)} | Win: {(vals > 0).mean() * 100.0:.1f}%)"
                else:
                    lbl = f"{mdl['short']} (t+{plt_lead}) (Med: {np.median(vals):.3f})"
                ax2.plot(
                    vals,
                    np.linspace(0, 1, len(vals)),
                    label=lbl,
                    color=mdl["color"],
                    alpha=_lead_alpha(l_idx),
                    linewidth=_lead_lw(l_idx, plt_lead, base_lw=2.6),
                    linestyle=mdl["ls"],
                    zorder=mdl["z"] + (n_prog - l_idx) * 0.05,
                )

    if n_prog > 2 and progression_leads == list(range(progression_leads[0], progression_leads[-1] + 1)):
        prog_title = f"t+{progression_leads[0]} to t+{progression_leads[-1]}"
    else:
        prog_title = ", ".join([f"t+{l}" for l in progression_leads])
    ax2.set_title(f"B. Multi-Horizon Progression ({prog_title}) — Per-Gauge {short_lbl} CDF", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel(units.ss_label(axis_lbl) if is_ss else axis_lbl, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Gauge Fraction (ECDF)", fontsize=11, fontweight="bold")
    ax2.set_xlim(*resolved_xlim)
    if is_ss:
        units.ss_axis(ax2, "x")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    _h, _l = ax2.get_legend_handles_labels()
    _lead_items = [(h, l) for h, l in zip(_h, _l) if "(t+" in l]
    _other_items = [(h, l) for h, l in zip(_h, _l) if "(t+" not in l]
    _groups = list(dict.fromkeys(l.split(" (t+")[0] for _, l in _lead_items))
    _ncol = 1 if n_prog <= 3 else max(1, len(_groups))
    if _ncol > 1:
        # Column-major fill: pad each model's column to the same height so leads line up across models.
        _rows = max(sum(1 for _, l in _lead_items if l.startswith(g + " (t+")) for g in _groups) + len(_other_items)
        from matplotlib.lines import Line2D
        _blank = Line2D([], [], alpha=0)
        _hh, _ll = [], []
        for gi, g in enumerate(_groups):
            head = _other_items if gi == 0 else [(_blank, "")] * len(_other_items)
            col = head + [(h, l) for h, l in _lead_items if l.startswith(g + " (t+")]
            col += [(_blank, "")] * (_rows - len(col))
            _hh += [h for h, _ in col]
            _ll += [l for _, l in col]
        _h, _l = _hh, _ll
    if _ncol >= 3:
        fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=True, fontsize=7.6,
                   labelspacing=0.3, ncol=_ncol, columnspacing=1.2, handlelength=1.8,
                   title="Panel B", title_fontsize=8)
    else:
        ax2.legend(_h, _l, loc="upper left", frameon=True, fontsize=8.4 if _ncol == 1 else 7.6, labelspacing=0.32,
                   ncol=_ncol, columnspacing=0.9, handlelength=1.8)

    plt.tight_layout()
    return fig


def plot_leadtime_decay(
    df_eval: pd.DataFrame,
    lead_times: Optional[List[int]] = None,
    metric: Optional[str] = None,
    metric_col: str = "DA NSE",
    base_metric_col: str = "Base NSE",
    figsize: Tuple[int, int] = (10, 4.8),
    show=DEFAULT_ECDF_SHOW,
) -> plt.Figure:
    """Renders forecast skill and memory persistence curves across forecast horizons."""
    from da_eval.benchmarks import build_per_basin_optimal_eval
    from da_eval.ingestion import _ensure_skill_score_columns

    if df_eval.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center")
        return fig

    is_delta, m_col, b_col, rank_col, short_lbl, _, _ = _resolve_metric_spec(
        metric=metric, metric_col=metric_col, base_metric_col=base_metric_col
    )

    if lead_times is None:
        lead_times = detect_lead_times(df_eval)

    fig, ax = plt.subplots(figsize=figsize)

    ref_lead = 1 if 1 in lead_times else lead_times[0]
    work_df = _ensure_skill_score_columns(df_eval.copy())
    if rank_col in work_df.columns and rank_col != "NSE Delta":
        base_only = work_df[work_df["Config ID"] != "Per-Basin Best DA"].copy()
        work_df = build_per_basin_optimal_eval(base_only, lead_time_ref=ref_lead, metric_col=rank_col)

    best_cfg = get_global_best_config(work_df, lead_time=ref_lead, metric_col=rank_col if rank_col in work_df.columns else "NSE Delta")
    models = resolve_ecdf_models(work_df, ref_lead, rank_col, best_cfg, show)

    base_medians = []
    series = {m["key"]: {"med": [], "q25": [], "q75": []} for m in models}

    for lt in lead_times:
        lt_sub = work_df[work_df["Lead Time (Days)"] == lt]
        # Baseline
        if is_delta:
            base_medians.append(0.0)
        else:
            base_medians.append(lt_sub.drop_duplicates("Basin ID")[b_col].median() if (b_col and b_col in lt_sub) else np.nan)
        for mdl in models:
            v = lt_sub[lt_sub["Config ID"] == mdl["cfg"]][m_col].dropna() if m_col in lt_sub else pd.Series(dtype=float)
            series[mdl["key"]]["med"].append(v.median() if not v.empty else np.nan)
            series[mdl["key"]]["q25"].append(v.quantile(0.25) if not v.empty else np.nan)
            series[mdl["key"]]["q75"].append(v.quantile(0.75) if not v.empty else np.nan)

    x_labels = [f"t+{lt}" for lt in lead_times]
    x_pos = np.arange(len(lead_times))

    # Plot Curves
    if is_delta:
        ax.axhline(0.0, color="#5f6368", linewidth=2.0, linestyle="--", label=("Open-loop baseline (SS = 0%)" if (_is_ss(m_col) and units.ss_percent())
                          else "Baseline Reference (Δ = 0.0)"))
    else:
        ax.plot(x_pos, base_medians, marker="o", color="#5f6368", linewidth=2.2, label="Baseline Open-Loop")
    markers = {"global_da": "s", "per_basin_da": "^", "global_rho": "D", "per_basin_rho": "v"}
    for mdl in models:
        s = series[mdl["key"]]
        if all(np.isnan(v) for v in s["med"]):
            continue
        ax.plot(x_pos, s["med"], marker=markers[mdl["key"]], color=mdl["color"], linewidth=2.6, linestyle=mdl["ls"],
                label=mdl["long"])
        if mdl["key"] == "global_da":
            ax.fill_between(x_pos, s["q25"], s["q75"], color=mdl["color"], alpha=0.15, label="Global Best IQR (25th–75th)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11, fontweight="bold")
    is_ss = _is_ss(m_col)
    ax.set_ylabel(units.ss_label(f"Median {short_lbl}") if is_ss else f"Median {short_lbl}", fontsize=11, fontweight="bold")
    if is_ss:
        units.ss_axis(ax, "y")
    ax.set_title(f"Data Assimilation Horizon Persistence & Skill Decay ({short_lbl})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=10)

    plt.tight_layout()
    return fig


def plot_basin_characteristics_efficacy(
    df_strata: pd.DataFrame,
    df_strata_compare: Optional[pd.DataFrame] = None,
    label_primary: str = "Global Best DA",
    label_compare: str = "Per-Basin Best DA (Oracle)",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10),
) -> plt.Figure:
    """Renders the Unified 2x3 Horizontal Diverging Effect-Size Bar Chart grid across 6 dimensions.

    If df_strata_compare is provided, renders grouped paired horizontal bars comparing both models.
    """
    dimensions = ["Aridity Index", "Baseline Tier", "Seasonal Flow", "Catchment Area", "Snow Fraction", "Terrain Slope"]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes_flat = axes.flatten()

    is_paired = df_strata_compare is not None and not df_strata_compare.empty
    med_col = "Median NSE Skill Score" if "Median NSE Skill Score" in df_strata.columns else "Median ΔNSE"
    axis_lbl = "Median NSE Skill Score (with [25th, 75th] IQR)" if med_col == "Median NSE Skill Score" else "Median ΔNSE (with [25th, 75th] IQR)"
    is_ss = _is_ss(med_col)
    if is_ss:
        axis_lbl = units.ss_label(axis_lbl)

    for idx, dim in enumerate(dimensions):
        ax = axes_flat[idx]
        sub = df_strata[df_strata["Dimension"] == dim].copy()

        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {dim}", ha="center", va="center", fontsize=11)
            ax.set_title(dim, fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        cats = sub["Sub-Category"].tolist()
        deltas = sub[med_col].tolist()
        q25 = sub["IQR Lower (25%)"].tolist()
        q75 = sub["IQR Upper (75%)"].tolist()

        y_pos = np.arange(len(cats))

        if not is_paired:
            colors = ["#1a73e8" if d >= 0 else "#ea4335" for d in deltas]
            xerr_lower = [max(0.0, d - lo) for d, lo in zip(deltas, q25)]
            xerr_upper = [max(0.0, hi - d) for d, hi in zip(deltas, q75)]
            xerr = [xerr_lower, xerr_upper]

            bars = ax.barh(y_pos, deltas, xerr=xerr, color=colors, edgecolor="black", alpha=0.85, capsize=4, height=0.5)
            for bar, delta, n_b in zip(bars, deltas, sub["N Basins"]):
                offset = 0.01 if delta >= 0 else -0.01
                ha = "left" if delta >= 0 else "right"
                y_text = bar.get_y() + bar.get_height() + 0.04
                ax.text(delta + offset, y_text, f"{_fmt_val(delta, is_ss)} (N={n_b:,})", va="bottom", ha=ha, fontsize=8.5, fontweight="bold", color="#202124")
            ax.set_ylim(-0.6, len(cats) + 0.15)
        else:
            sub2 = df_strata_compare[df_strata_compare["Dimension"] == dim].copy()
            sub2 = sub2.set_index("Sub-Category").reindex(cats).reset_index()
            med_col2 = "Median NSE Skill Score" if "Median NSE Skill Score" in sub2.columns else "Median ΔNSE"
            deltas2 = sub2[med_col2].fillna(0.0).tolist()
            q25_2 = sub2["IQR Lower (25%)"].fillna(0.0).tolist()
            q75_2 = sub2["IQR Upper (75%)"].fillna(0.0).tolist()

            c1 = ["#1a73e8" if d >= 0 else "#ea4335" for d in deltas]
            c2 = ["#0d47a1" if d >= 0 else "#b71c1c" for d in deltas2]

            xerr1 = [[max(0.0, d - lo) for d, lo in zip(deltas, q25)], [max(0.0, hi - d) for d, hi in zip(deltas, q75)]]
            xerr2 = [[max(0.0, d - lo) for d, lo in zip(deltas2, q25_2)], [max(0.0, hi - d) for d, hi in zip(deltas2, q75_2)]]

            bars1 = ax.barh(y_pos - 0.17, deltas, xerr=xerr1, height=0.30, color=c1, edgecolor="black", alpha=0.9, capsize=3, label=label_primary if idx == 0 else "")
            bars2 = ax.barh(y_pos + 0.17, deltas2, xerr=xerr2, height=0.30, color=c2, edgecolor="black", hatch="//", alpha=0.8, capsize=3, label=label_compare if idx == 0 else "")

            for b1, v1 in zip(bars1, deltas):
                ha = "left" if v1 >= 0 else "right"
                off = 0.01 if v1 >= 0 else -0.01
                ax.text(v1 + off, b1.get_y() + b1.get_height() / 2, _fmt_val(v1, is_ss), va="center", ha=ha, fontsize=8, fontweight="bold", color="#202124")
            for b2, v2 in zip(bars2, deltas2):
                ha = "left" if v2 >= 0 else "right"
                off = 0.01 if v2 >= 0 else -0.01
                ax.text(v2 + off, b2.get_y() + b2.get_height() / 2, _fmt_val(v2, is_ss), va="center", ha=ha, fontsize=8, fontweight="bold", color="#0d47a1")

            ax.set_ylim(-0.6, len(cats) + 0.15)
            if idx == 0:
                ax.legend(loc="lower right", frameon=True, fontsize=9.5)

        ax.axvline(0.0, color="black", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=10, fontweight="bold")
        ax.set_xlabel(axis_lbl, fontsize=10)
        ax.set_title(f"{idx+1}. {dim}", fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Ensure the zero-axis reference is clearly visible and not pinned to the left edge
        x_min, x_max = ax.get_xlim()
        safe_x_max = x_max if np.isfinite(x_max) else 0.1
        safe_x_min = x_min if np.isfinite(x_min) else -0.1
        pad_neg = max(0.06, 0.15 * max(abs(safe_x_max), 0.1))
        new_x_min = min(safe_x_min, -pad_neg)
        new_x_max = max(safe_x_max, pad_neg * 0.5)
        ax.set_xlim(new_x_min, new_x_max * 1.10)
        if is_ss:
            units.ss_axis(ax, "x")

    suptitle = title or ("Physical Characteristics: Global Best DA vs. Per-Basin Best DA (Oracle) [Lead 1 Day]" if is_paired else "Physical Catchment Characteristics Impact on Data Assimilation Skill Score (Lead 1 Day)")
    plt.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_hyperparameter_effects(
    df_eval: pd.DataFrame,
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
    figsize: Tuple[int, int] = (18, 9),
) -> plt.Figure:
    """Renders non-parametric Hyperparameter Main Effects & Rank Contribution Grid:

    - Panel A: Hyperparameter Rank Variance Contribution (%) [Friedman Rank Effect]
    - Panels B+: Marginal Main Effect curves showing Median Skill Score and Catchment Win Rate (%) per level.
    """
    from da_eval.ingestion import parse_da_config_id, _ensure_skill_score_columns
    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"
    is_ss = _is_ss(metric_col)
    y_lbl = units.ss_label(f"Median {metric_col}") if is_ss else f"Median {metric_col}"

    # Exclude synthetic / benchmark / baseline rows (case-insensitive)
    synthetic_patterns = ["per-basin", "baseline", "global best", "oracle", "tuned per basin"]
    mask_exclude = df_eval["Config ID"].astype(str).str.lower().apply(
        lambda s: any(p in s for p in synthetic_patterns)
    )
    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~mask_exclude)
    ].copy()

    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No hyperparameter sweep data available", ha="center", va="center")
        return fig

    # Parse hyperparameters
    parsed = sub["Config ID"].apply(parse_da_config_id)
    for col in parsed.columns:
        sub[col] = parsed[col]

    # Identify factors with > 1 level
    candidate_factors = ["Target", "Loss", "Window (Days)", "Learning Rate", "Epochs", "BG Weight", "Stat Weight"]
    active_factors = [f for f in candidate_factors if f in sub.columns and sub[f].dropna().nunique() > 1]

    if not active_factors:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Single configuration evaluated\n(No multi-level hyperparameter sweep)", ha="center", va="center")
        return fig

    # Compute per-basin rank (1 = worst, K = best) to eliminate hydrological scale bias,
    # filtering to finite metric values first.
    finite_mask = np.isfinite(sub[metric_col])
    sub["Rank"] = np.nan
    if finite_mask.any():
        sub.loc[finite_mask, "Rank"] = (
            sub[finite_mask]
            .groupby("Basin ID")[metric_col]
            .rank(ascending=True)
        )

    # 1. Non-parametric Rank Variance Contribution
    var_contributions = []
    overall_mean_rank = sub["Rank"].mean()
    for f in active_factors:
        f_sub = sub.dropna(subset=[f, "Rank"])
        if len(f_sub) < 2 or f_sub[f].nunique() < 2:
            var_contributions.append(0.0)
            continue
        # Variance of group means
        group_means = f_sub.groupby(f)["Rank"].mean()
        group_counts = f_sub.groupby(f)["Rank"].count()
        diffs = group_means - overall_mean_rank
        terms = group_counts * (diffs**2)
        valid_terms = terms.dropna()
        between_var = float(valid_terms.sum() / len(f_sub)) if len(f_sub) > 0 else 0.0
        var_contributions.append(between_var if np.isfinite(between_var) else 0.0)

    sum_var = sum(v for v in var_contributions if np.isfinite(v) and v > 0)
    if sum_var > 0:
        pct_contributions = [
            (v / sum_var * 100.0) if (np.isfinite(v) and v > 0) else 0.0
            for v in var_contributions
        ]
    else:
        pct_contributions = [0.0 for _ in var_contributions]

    n_panels = len(active_factors) + 1
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten()

    # -------------------------------------------------------------
    # Panel 0: Relative Hyperparameter Contribution (%)
    # -------------------------------------------------------------
    ax0 = axes_flat[0]
    palette = sns.color_palette("Blues_r", len(active_factors))
    bars0 = ax0.barh(active_factors, pct_contributions, color=palette, edgecolor="black", alpha=0.85, height=0.55)
    ax0.set_xlabel("Relative Rank Contribution (%)", fontsize=10, fontweight="bold")
    ax0.set_title(f"A. Hyperparameter Influence (Lead {lead_time})", fontsize=11, fontweight="bold", pad=8)
    ax0.invert_yaxis()
    ax0.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars0, pct_contributions):
        w = bar.get_width()
        ax0.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9.5, fontweight="bold")
    max_pct = _finite_or_default(pct_contributions, max, 0.0)
    ax0.set_xlim(0, max(100.0, max_pct + 15))

    # -------------------------------------------------------------
    # Panels 1+: Marginal Effects & Win Rate per Factor
    # -------------------------------------------------------------
    for idx, factor in enumerate(active_factors):
        ax = axes_flat[idx + 1]
        levels = sorted(sub[factor].dropna().unique().tolist(), key=lambda x: (str(type(x)), x))

        medians = []
        q25 = []
        q75 = []
        win_rates = []
        skipped_levels = set()

        # Calculate best level per basin on finite metric values
        sub_clean = sub[np.isfinite(sub[metric_col])].dropna(subset=[factor])
        if not sub_clean.empty:
            best_per_basin = sub_clean.sort_values(metric_col, ascending=False).drop_duplicates(subset=["Basin ID"])
            total_basins = best_per_basin["Basin ID"].nunique()
        else:
            best_per_basin = pd.DataFrame()
            total_basins = 0

        for lev in levels:
            lev_sub = sub[sub[factor] == lev]
            vals = lev_sub[metric_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                skipped_levels.add(lev)
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
                win_rates.append(0.0)
            else:
                m = float(np.median(vals))
                medians.append(m)
                q25.append(float(np.percentile(vals, 25)))
                q75.append(float(np.percentile(vals, 75)))

                wins = (best_per_basin[factor] == lev).sum() if not best_per_basin.empty else 0
                win_rates.append((wins / total_basins * 100.0) if total_basins > 0 else 0.0)

        # If all levels lack finite data, show a clear message on the panel
        if len(skipped_levels) == len(levels):
            ax.text(0.5, 0.5, f"No finite data for {factor}\n(All configurations diverged/NaN)", ha="center", va="center", fontsize=9.5, transform=ax.transAxes, color="#5f6368")
            ax.set_title(f"{chr(66 + idx)}. Marginal Effect: {factor}", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(factor, fontsize=10, fontweight="bold")
            ax.set_ylabel(y_lbl, fontsize=9)
            continue

        # 1. Compute robust y-limits from well-behaved levels using an IQR-based fence
        pool = np.array([v for v in (medians + q25 + q75) if v is not None and np.isfinite(v)])
        if pool.size > 0:
            p25, p75 = np.percentile(pool, [25, 75])
            iqr = p75 - p25
            lo_fence = p25 - 3.0 * iqr
            hi_fence = p75 + 3.0 * iqr
            inliers = pool[(pool >= lo_fence) & (pool <= hi_fence)]
            y_min_raw = float(np.min(inliers)) if inliers.size > 0 else float(np.min(pool))
            y_max_raw = float(np.max(inliers)) if inliers.size > 0 else float(np.max(pool))
        else:
            y_min_raw, y_max_raw = -0.1, 0.1

        # 2. Enforce a hydrologically sensible floor and ensure valid range
        y_min = max(y_min_raw, -1.0)
        y_max = max(y_max_raw, 0.1)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = -0.1, 0.1
        margin = max(0.03, (y_max - y_min) * 0.25)
        final_y_min = y_min - margin * 0.5
        final_y_max = y_max + margin * 1.5
        ax.set_ylim(final_y_min, final_y_max)

        # 3. Clip off-scale points and whiskers to the axis, and mark them
        x_pos = np.arange(len(levels))
        plot_medians = []
        yerr_lower = []
        yerr_upper = []
        has_offscale = False

        for i, (lev, m, lo, hi) in enumerate(zip(levels, medians, q25, q75)):
            if lev in skipped_levels or not np.isfinite(m):
                plot_medians.append(np.nan)
                yerr_lower.append(0.0)
                yerr_upper.append(0.0)
                continue

            if m < final_y_min:
                has_offscale = True
                plot_medians.append(np.nan)
                yerr_lower.append(0.0)
                yerr_upper.append(0.0)
                bound_y = final_y_min + (final_y_max - final_y_min) * 0.035
                ax.plot(x_pos[i], bound_y, marker="v", color="#d93025", markersize=9, zorder=5)
            elif m > final_y_max:
                has_offscale = True
                plot_medians.append(np.nan)
                yerr_lower.append(0.0)
                yerr_upper.append(0.0)
                bound_y = final_y_max - (final_y_max - final_y_min) * 0.035
                ax.plot(x_pos[i], bound_y, marker="^", color="#d93025", markersize=9, zorder=5)
            else:
                plot_medians.append(m)
                if np.isfinite(lo) and lo < final_y_min:
                    has_offscale = True
                    yerr_lower.append(max(0.0, m - final_y_min))
                    ax.plot(x_pos[i], final_y_min + (final_y_max - final_y_min) * 0.02, marker="v", color="#d93025", markersize=6, zorder=4)
                else:
                    yerr_lower.append(max(0.0, m - lo) if np.isfinite(lo) else 0.0)

                if np.isfinite(hi) and hi > final_y_max:
                    has_offscale = True
                    yerr_upper.append(max(0.0, final_y_max - m))
                    ax.plot(x_pos[i], final_y_max - (final_y_max - final_y_min) * 0.02, marker="^", color="#d93025", markersize=6, zorder=4)
                else:
                    yerr_upper.append(max(0.0, hi - m) if np.isfinite(hi) else 0.0)

        ax.errorbar(x_pos, plot_medians, yerr=[yerr_lower, yerr_upper], fmt="o-", color="#1a73e8", ecolor="#5f6368", elinewidth=1.5, capsize=4, markersize=7, linewidth=2.0)
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5, linewidth=1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(l) for l in levels], fontsize=9, fontweight="bold")
        ax.set_xlabel(factor, fontsize=10, fontweight="bold")
        ax.set_ylabel(y_lbl, fontsize=9)
        if is_ss:
            units.ss_axis(ax, "y")
        ax.set_title(f"{chr(66 + idx)}. Marginal Effect: {factor}", fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # 4. Annotate win rate and median above each point, with off-scale indicators
        for xp, m, wr, hi, lev in zip(x_pos, medians, win_rates, q75, levels):
            if lev in skipped_levels or not np.isfinite(m):
                no_data_y = 0.0 + (final_y_max - final_y_min) * 0.05
                ax.text(
                    xp, no_data_y, "No Data",
                    ha="center", va="bottom", fontsize=8.5, color="#5f6368", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#dadce0", alpha=0.85, linewidth=0.8)
                )
                continue
            if m < final_y_min:
                text_y = final_y_min + (final_y_max - final_y_min) * 0.06
                ax.text(xp, text_y, f"Win: {wr:.0f}%\n{_fmt_val(m, is_ss)} ↓", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#d93025")
            elif m > final_y_max:
                text_y = final_y_max - (final_y_max - final_y_min) * 0.12
                ax.text(xp, text_y, f"Win: {wr:.0f}%\n{_fmt_val(m, is_ss)} ↑", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#d93025")
            else:
                top_y = hi if np.isfinite(hi) else m
                text_y = min(top_y + 0.012, final_y_max - (final_y_max - final_y_min) * 0.12)
                ax.text(xp, text_y, f"Win: {wr:.0f}%\n{_fmt_val(m, is_ss)}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#202124")

        # 5. Concise note on any panel containing off-scale levels, collision-aware placement
        if has_offscale:
            offscale_x_positions = [
                x_pos[i] for i, (lev, m, lo, hi) in enumerate(zip(levels, medians, q25, q75))
                if lev not in skipped_levels and (m < final_y_min or m > final_y_max or (np.isfinite(lo) and lo < final_y_min))
            ]
            mean_offscale_x = np.mean(offscale_x_positions) if offscale_x_positions else 0.0
            mid_x = (len(levels) - 1) / 2.0
            if mean_offscale_x <= mid_x:
                badge_x = 0.97
                badge_ha = "right"
            else:
                badge_x = 0.03
                badge_ha = "left"

            ax.text(
                badge_x, 0.05,
                "▼ = diverged, value off-scale",
                transform=ax.transAxes,
                ha=badge_ha,
                fontsize=8,
                fontstyle="italic",
                color="#d93025",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fce8e6", edgecolor="#d93025", alpha=0.9),
                zorder=6
            )

    # Hide any unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("Hyperparameter Main Effects & Relative Contribution (Lead 1 Day)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def _render_bar_panel(ax, df_plot, top_n, lead_time, total_basins, med_prefix="Global Med SS", is_ss=False):
    y_pos = np.arange(len(df_plot))
    bars = ax.barh(y_pos, df_plot["Basins Won"], color=df_plot["Color"], edgecolor="black", alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Label"], fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Number of Catchments where Configuration is Optimal (#1)", fontsize=11, fontweight="bold")
    ax.set_title(f"A. Top {top_n} Configurations by Optimal Catchment Count (Lead {lead_time} Day)", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        w = bar.get_width()
        delta_str = f" | {med_prefix}: {_fmt_val(row['Med Delta'], is_ss)}" if pd.notna(row["Med Delta"]) else ""
        ax.text(
            w + total_basins * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['Basins Won']:,} ({row['Share']:.1f}%){delta_str}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#202124",
        )

    max_won = _finite_or_default(df_plot["Basins Won"], max, 1.0)
    ax.set_xlim(0, max(1.0, max_won * 1.35))


def _render_donut_panel(ax, top_subset, other_count, other_n_cfgs, total_basins, n_total_cfgs):
    pie_labels = [f"Rank {i+1}" for i in range(len(top_subset))]
    pie_vals = top_subset["Basins Won"].tolist()
    colors = list(sns.color_palette("tab10", len(top_subset)))

    if other_count > 0:
        pie_labels.append(f"Other ({other_n_cfgs})")
        pie_vals.append(other_count)
        colors.append("#bdc1c6")

    wedges, _, autotexts = ax.pie(
        pie_vals,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 3.5 else "",
        startangle=90,
        pctdistance=0.75,
        colors=colors,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(8.5)
        autotext.set_fontweight("bold")

    ax.text(
        0,
        0,
        f"Total Catchments\nN = {total_basins:,}\n({n_total_cfgs} Configs)",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#202124",
    )
    ax.set_title("B. Catchment Optimal Configuration Allocation", fontsize=12, fontweight="bold", pad=10)

    legend_labels = [f"{lbl}: {val:,} ({val/total_basins*100:.1f}%)" for lbl, val in zip(pie_labels, pie_vals)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=9)


def plot_top_n_optimal_basin_distribution(
    df_eval: pd.DataFrame,
    top_n: int = 10,
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
    plot_type: str = "combined",
    figsize: Optional[Tuple[int, int]] = None,
    filter_top_n_global: Optional[int] = None,
) -> plt.Figure:
    """Renders distribution of optimal configurations across catchments:

    Visualizes how many basins each top configuration wins (is the optimal per-basin model),
    highlighting the Globally Best DA configuration to support Globally Best vs. Per-Basin Oracle comparisons.

    - plot_type="combined": 2-panel figure with Horizontal Bar Chart (Panel A) and Donut Chart (Panel B).
    - plot_type="bar": Standalone Horizontal Bar Chart.
    - plot_type="donut" or "pie": Standalone Donut Chart.
    """
    from da_eval.ingestion import _ensure_skill_score_columns
    if df_eval.empty:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center")
        return fig

    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"
    med_prefix = "Global Med SS" if "Skill" in metric_col else "Global Med Δ"

    # Filter out synthetic benchmarks
    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ].copy()

    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.text(0.5, 0.5, f"No evaluation data for Lead Time {lead_time}", ha="center", va="center")
        return fig

    if filter_top_n_global and get_active() is None:  # legacy only: would redefine Per-Basin DA
        top_global_cfgs = (
            sub.groupby("Config ID")[metric_col]
            .median()
            .sort_values(ascending=False)
            .head(filter_top_n_global)
            .index.tolist()
        )
        sub = sub[sub["Config ID"].isin(top_global_cfgs)].copy()

    global_best_cfg = get_global_best_config(df_eval, lead_time=lead_time, metric_col=metric_col)
    sel = get_active()
    if sel is not None:  # notebook-wide Per-Basin DA definition (lead_time only sets the displayed medians)
        best_per_basin = pd.DataFrame({"Basin ID": list(sel.per_basin_da), "Config ID": list(sel.per_basin_da.values())})
    else:
        sub_clean = sub.dropna(subset=[metric_col])
        best_per_basin = (
            sub_clean.sort_values(metric_col, ascending=False)
            .drop_duplicates(subset=["Basin ID"])[["Basin ID", "Config ID", metric_col]]
        )
    total_basins = best_per_basin["Basin ID"].nunique()

    win_counts = best_per_basin["Config ID"].value_counts().reset_index()
    win_counts.columns = ["Config ID", "Basins Won"]
    win_counts["Win Share (%)"] = (win_counts["Basins Won"] / total_basins) * 100.0
    win_counts["Cumulative Basins"] = win_counts["Basins Won"].cumsum()
    win_counts["Cumulative Share (%)"] = (win_counts["Cumulative Basins"] / total_basins) * 100.0

    global_med_delta = sub.groupby("Config ID")[metric_col].median().rename("Global Med ΔNSE")
    win_counts = win_counts.merge(global_med_delta, on="Config ID", how="left")
    win_counts["Is Global Best"] = win_counts["Config ID"] == global_best_cfg

    top_subset = win_counts.head(top_n).copy()
    other_count = win_counts["Basins Won"].iloc[top_n:].sum() if len(win_counts) > top_n else 0
    other_n_cfgs = len(win_counts) - top_n

    plot_rows = []
    for _, r in top_subset.iterrows():
        is_gb = r["Is Global Best"]
        cid = str(r["Config ID"])
        plot_rows.append({
            "Config ID": cid,
            "Label": f"{cid} ★ [GLOBAL BEST]" if is_gb else cid,
            "Basins Won": r["Basins Won"],
            "Share": r["Win Share (%)"],
            "Med Delta": r["Global Med ΔNSE"],
            "Is Global Best": is_gb,
            "Color": "#1a73e8" if is_gb else "#4285f4",
        })

    if other_count > 0:
        plot_rows.append({
            "Config ID": "Other",
            "Label": f"Other Configurations ({other_n_cfgs} remaining)",
            "Basins Won": other_count,
            "Share": (other_count / total_basins) * 100.0,
            "Med Delta": np.nan,
            "Is Global Best": False,
            "Color": "#bdc1c6",
        })

    df_plot = pd.DataFrame(plot_rows)

    if plot_type == "bar":
        fig, ax1 = plt.subplots(figsize=figsize or (12, max(5, top_n * 0.55)))
        _render_bar_panel(ax1, df_plot, top_n, lead_time, total_basins, med_prefix=med_prefix, is_ss=_is_ss(metric_col))
    elif plot_type in ["donut", "pie"]:
        fig, ax2 = plt.subplots(figsize=figsize or (8, 7))
        _render_donut_panel(ax2, top_subset, other_count, other_n_cfgs, total_basins, len(win_counts))
    else:  # "combined"
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize or (18, max(6.5, top_n * 0.55)), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )
        _render_bar_panel(ax1, df_plot, top_n, lead_time, total_basins, med_prefix=med_prefix, is_ss=_is_ss(metric_col))
        _render_donut_panel(ax2, top_subset, other_count, other_n_cfgs, total_basins, len(win_counts))

    plt.tight_layout()
    return fig


CASE_STUDY_ROLES = ("Baseline", "Global-Best DA", "Per-Basin DA", "Global-Best Rho", "Per-Basin Rho")

_CASE_STUDY_ROLE_ALIASES = {
    "baseline": "Baseline", "open-loop": "Baseline", "baseline open-loop": "Baseline",
    "global-best da": "Global-Best DA", "global best da": "Global-Best DA", "global best": "Global-Best DA",
    "global-best": "Global-Best DA", "gb": "Global-Best DA",
    "per-basin da": "Per-Basin DA", "per-basin best da": "Per-Basin DA", "per basin da": "Per-Basin DA",
    "per-basin": "Per-Basin DA", "oracle": "Per-Basin DA", "basin best da": "Per-Basin DA", "pb": "Per-Basin DA",
    "global-best rho": "Global-Best Rho", "global best rho": "Global-Best Rho",
    "global-best ar(1)": "Global-Best Rho", "global-best ar1": "Global-Best Rho",
    "per-basin rho": "Per-Basin Rho", "per basin rho": "Per-Basin Rho",
    "per-basin ar(1)": "Per-Basin Rho", "per-basin ar1": "Per-Basin Rho",
}


def split_case_study_models(models) -> Tuple[set, List[str]]:
    """Splits a ``models`` selection into (role names, explicit Config IDs).

    Roles (case-insensitive, see ``CASE_STUDY_ROLES``) resolve per basin/lead; anything else is treated as a
    Config ID. Numbers are interpreted as AR(1) rho values (``0.7`` -> ``AR1_postprocess_rho0.7``).
    """
    roles, cfgs = set(), []
    for m in models or []:
        if isinstance(m, (int, float)) and not isinstance(m, bool):
            cfgs.append(f"AR1_postprocess_rho{float(m):g}")
            continue
        key = str(m).strip()
        role = _CASE_STUDY_ROLE_ALIASES.get(key.lower())
        if role:
            roles.add(role)
        elif key and key not in cfgs:
            cfgs.append(key)
    return roles, cfgs


def plot_basin_epoch_case_study(
    basin_id: str,
    df_eval: pd.DataFrame,
    df_ts: Optional[pd.DataFrame] = None,
    df_meta: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    year: int = 2017,
    date_range: Optional[Tuple[str, str]] = None,
    lead_times: Tuple[int, int] = (1, 7),
    target_backbone: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 9.5),
    show_rho: bool = True,
    extra_configs: Optional[List[str]] = None,
    models: Optional[List] = None,
    precip_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """Renders a publication-ready 2-subplot Basin Case Study hydrograph.

    With ``show_rho`` the Global-Best Rho and Per-Basin Rho AR(1) references are drawn when present in ``df_ts``.
    ``extra_configs`` (e.g. ``["AR1_postprocess_rho0.7"]``) are drawn first among the extra "Model:" lines.

    ``models`` (optional) selects exactly which lines are drawn besides Observed flow, e.g.
    ``["Baseline", "Global-Best DA", "Per-Basin Rho", "S49_05_...", 0.7]``. Roles are listed in
    ``CASE_STUDY_ROLES``; other entries are Config IDs (numbers = AR(1) rho). When given, ``show_rho`` and
    the automatic extra "Model:" lines are ignored. ``None`` keeps the default set.

    Subplot 1 (Top): Lead Time 1 Day (t+1) - Epoch Convergence
    Subplot 2 (Bottom): Lead Time 7 Days (t+7) - Skill Persistence Across Epochs
    Includes continuous NSE, NSE Skill Score (SS), and KGE metrics directly in legend.
    """
    import re
    import glob
    from da_eval.ingestion import _ensure_skill_score_columns

    df_eval = _ensure_skill_score_columns(df_eval)

    # 1. Retrieve timeseries records for this basin
    b_ts = pd.DataFrame()
    if df_ts is not None and not df_ts.empty and "Basin ID" in df_ts.columns:
        b_ts = df_ts[df_ts["Basin ID"] == basin_id].copy()

    if b_ts.empty or b_ts["Config ID"].nunique() <= 1:
        if data_dir and os.path.exists(data_dir):
            patterns = [
                os.path.join(data_dir, f"timeseries_{basin_id}.parquet"),
                os.path.join(data_dir, f"*timeseries*_{basin_id}.parquet"),
                os.path.join(data_dir, "**", f"timeseries_{basin_id}.parquet"),
            ]
            for pat in patterns:
                matches = glob.glob(pat, recursive=True)
                if matches:
                    try:
                        b_ts = pd.read_parquet(matches[0])
                        break
                    except Exception:
                        pass

    if b_ts.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No timeseries data found for catchment {basin_id}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # 2. Extract Basin Metadata
    b_eval = df_eval[df_eval["Basin ID"] == basin_id].copy()
    meta_row = pd.Series()
    if df_meta is not None and not df_meta.empty and "Basin ID" in df_meta.columns:
        m_sub = df_meta[df_meta["Basin ID"] == basin_id]
        if not m_sub.empty:
            meta_row = m_sub.iloc[0]

    basin_name = str(meta_row.get("gauge_name", meta_row.get("Basin Name", basin_id)))
    country = str(meta_row.get("country", "Unknown"))
    area = meta_row.get("area", np.nan)
    aridity = meta_row.get("unep_aridity_index", np.nan)
    p_mean = meta_row.get("p_mean", np.nan)

    # 3. Determine best backbone configuration and its swept epochs
    def _parse_ep(cid):
        m = re.search(r"_ep(\d+)", str(cid))
        return int(m.group(1)) if m else np.nan

    def _parse_bb(cid):
        return re.sub(r"_ep\d+", "", str(cid))

    b_eval["Epoch"] = b_eval["Config ID"].apply(_parse_ep)
    b_eval["Backbone"] = b_eval["Config ID"].apply(_parse_bb)

    lead1_eval = b_eval[b_eval["Lead Time (Days)"] == lead_times[0]].copy()
    sub_ep = lead1_eval.dropna(subset=["Epoch"])

    if target_backbone:
        bb = target_backbone
    elif not sub_ep.empty:
        bb = sub_ep.groupby("Backbone")["DA NSE"].max().idxmax()
    else:
        bb = b_eval["Config ID"].dropna().iloc[0] if not b_eval.empty else "DA Model"

    # Gather available configs for this backbone sorted by epoch
    bb_eval = b_eval[b_eval["Backbone"] == bb].sort_values("Epoch")
    available_epochs = sorted(bb_eval["Epoch"].dropna().unique())

    # Map epochs to config IDs
    epoch_cfg_map = {}
    for ep in available_epochs:
        match = bb_eval[bb_eval["Epoch"] == ep]
        if not match.empty:
            epoch_cfg_map[int(ep)] = match["Config ID"].iloc[0]

    # 4. Filter timeseries to requested time window
    b_ts["Valid Date"] = pd.to_datetime(b_ts["Valid Date"])
    b_ts = b_ts.sort_values("Valid Date")

    if date_range:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        ts_win = b_ts[(b_ts["Valid Date"] >= start_d) & (b_ts["Valid Date"] <= end_d)].copy()
    else:
        ts_win = b_ts[b_ts["Valid Date"].dt.year == year].copy()
        if ts_win.empty:
            ts_win = b_ts.copy()

    # 5. Render 2 Vertically Stacked Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes = [ax1, ax2]
    palette = ["#4fc3f7", "#1e88e5", "#1565c0", "#0d47a1", "#051c48"]

    # Model selection: None = default set (all roles + up to 2 automatic extra lines).
    if models is None:
        sel_roles = set(CASE_STUDY_ROLES) if show_rho else set(CASE_STUDY_ROLES) - {"Global-Best Rho", "Per-Basin Rho"}
        sel_cfgs, auto_extras = list(extra_configs or []), True
    else:
        sel_roles, sel_cfgs = split_case_study_models(models)
        auto_extras = False

    for idx, (ax, lt) in enumerate(zip(axes, lead_times)):
        lt_ts = ts_win[ts_win["Lead Time (Days)"] == lt].copy()
        lt_eval = b_eval[b_eval["Lead Time (Days)"] == lt].copy()

        # Baseline metrics
        base_row = lt_eval[lt_eval["Config ID"].str.contains("Baseline|Per-Basin", na=False)]
        if base_row.empty and not lt_eval.empty:
            base_row = lt_eval.iloc[[0]]
        base_nse = base_row["Base NSE"].iloc[0] if not base_row.empty and "Base NSE" in base_row else np.nan
        base_kge = base_row["Base KGE"].iloc[0] if not base_row.empty and "Base KGE" in base_row else np.nan

        # Render Observed Streamflow
        ref_slice = lt_ts.drop_duplicates(subset=["Valid Date"])
        if not ref_slice.empty and "q_obs" in ref_slice.columns:
            ax.plot(ref_slice["Valid Date"], ref_slice["q_obs"], color="black", linewidth=2.4, label="Observed Streamflow (q_obs)", zorder=6)

        # Render Open-Loop Baseline
        if "Baseline" in sel_roles and not ref_slice.empty and "q_base" in ref_slice.columns:
            b_lbl = f"Baseline Open-Loop (NSE: {base_nse:.2f}, KGE: {base_kge:.2f})" if pd.notna(base_nse) else "Baseline Open-Loop"
            ax.plot(ref_slice["Valid Date"], ref_slice["q_base"], color="#80868b", linestyle="--", linewidth=1.8, label=b_lbl, zorder=5)

        # 1. Render Global Best DA
        # Global Best among configs that actually have a hydrograph for this basin (DA-only by default).
        ts_cfgs = set(b_ts["Config ID"].dropna().unique()) - {"Per-Basin Best DA", "Baseline"}
        _sel = get_active()
        if _sel is not None:  # notebook-wide selection: the same Global-Best at every lead
            gb_cfg = _sel.global_da
        else:
            gb_cfg = get_global_best_config(df_eval[df_eval["Config ID"].isin(ts_cfgs)], lead_time=lt) if ts_cfgs else ""
            if not gb_cfg:
                gb_cfg = get_global_best_config(df_eval, lead_time=lt)
        plotted = {"Per-Basin Best DA", "Baseline"}

        def _metrics(cid, series):
            """(NSE, SS, KGE) from the eval table, falling back to the plotted series itself."""
            ev = lt_eval[lt_eval["Config ID"] == cid]
            if not ev.empty and pd.notna(ev["DA NSE"].iloc[0] if "DA NSE" in ev else np.nan):
                return (ev["DA NSE"].iloc[0],
                        ev["NSE Skill Score"].iloc[0] if "NSE Skill Score" in ev else np.nan,
                        ev["DA KGE"].iloc[0] if "DA KGE" in ev else np.nan)
            full = b_ts[(b_ts["Config ID"] == cid) & (b_ts["Lead Time (Days)"] == lt)]
            o, s = full["q_obs"].to_numpy(float), full["q_da"].to_numpy(float)
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() < 10 or np.var(o[m]) == 0:
                return np.nan, np.nan, np.nan
            o, s = o[m], s[m]
            nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
            r = np.corrcoef(o, s)[0, 1]
            kge = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
            ss = 1 - (1 - nse) / (1 - base_nse) if pd.notna(base_nse) and base_nse < 1 else np.nan
            return nse, ss, kge

        def _label(prefix, cid, series, with_kge=True):
            nse, ss, kge = _metrics(cid, series)
            ss_str = f", SS: {units.fmt_ss(ss, decimals=None if units.ss_percent() else 2)}" if pd.notna(ss) else ""
            kge_str = f", KGE: {kge:.2f}" if with_kge and pd.notna(kge) else ""
            return f"{prefix} (NSE: {nse:.2f}{ss_str}{kge_str})"

        if gb_cfg and "Global-Best DA" in sel_roles:
            gb_data = lt_ts[lt_ts["Config ID"] == gb_cfg]
            if not gb_data.empty and "q_da" in gb_data.columns:
                ax.plot(gb_data["Valid Date"], gb_data["q_da"], color="#1a73e8", linewidth=2.2, linestyle="-",
                        label=_label(f"Global Best ({gb_cfg[:22]})", gb_cfg, gb_data), zorder=8)
                plotted.add(gb_cfg)

        # 2. Render Basin Best DA (Oracle or optimal for this basin)
        if "Per-Basin DA" in sel_roles:
            pb_data = lt_ts[lt_ts["Config ID"] == "Per-Basin Best DA"]
            if not pb_data.empty and "q_da" in pb_data.columns:
                ax.plot(pb_data["Valid Date"], pb_data["q_da"], color="#188038", linewidth=2.0, linestyle="-.",
                        label=_label("Basin Best DA (Oracle)", "Per-Basin Best DA", pb_data), zorder=9)

        # 3. AR(1) error-correction references (synthesised from Baseline + observed flow)
        rho_roles = sel_roles & {"Global-Best Rho", "Per-Basin Rho"}
        if rho_roles:
            rho_fixed = [c for c in ts_cfgs if _is_rho(c) and "per_basin" not in c.lower()]
            if _sel is not None:
                gbr_cfg = _sel.global_rho or ""
            else:
                gbr_cfg = get_global_best_config(df_eval[df_eval["Config ID"].isin(rho_fixed)], lead_time=lt,
                                                 exclude_rho=False) if rho_fixed else ""
            pbr_cfg = next((c for c in ts_cfgs if _is_rho(c) and "per_basin" in c.lower()), "")
            for cid, name, color, ls in ((gbr_cfg, "Global-Best Rho", "#e37400", (0, (5, 2))),
                                         (pbr_cfg, "Per-Basin Rho", "#c5221f", (0, (1, 1)))):
                if name not in rho_roles:
                    continue
                r_data = lt_ts[lt_ts["Config ID"] == cid] if cid else pd.DataFrame()
                if r_data.empty:
                    continue
                tag = cid.replace("AR1_postprocess_", "") if "per_basin" not in cid.lower() else ""
                prefix = f"{name} ({tag})" if tag else name
                ax.plot(r_data["Valid Date"], r_data["q_da"], color=color, linewidth=1.6, linestyle=ls,
                        label=_label(prefix, cid, r_data), zorder=7)
                plotted.add(cid)

        # 4. Explicitly requested configs, then (default mode only) up to 2 other configs present in df_ts
        present = list(lt_ts["Config ID"].dropna().unique())
        pinned = [c for c in sel_cfgs if c in present and c not in plotted]
        missing = [c for c in sel_cfgs if c not in present]
        if missing and idx == 0:
            print(f"[INFO] No hydrograph for {basin_id}: {missing}")
        if auto_extras:
            rest = [c for c in present if c not in plotted and c not in pinned and (show_rho or not _is_rho(c))]
            other_cfgs = (pinned + rest)[:max(2, len(pinned))]
        else:
            other_cfgs = pinned
        other_palette = ["#9334e6", "#00897b", "#795548", "#d81b60", "#f9ab00", "#5f6368"]
        for o_idx, o_cid in enumerate(other_cfgs):
            o_data = lt_ts[lt_ts["Config ID"] == o_cid]
            if not o_data.empty and "q_da" in o_data.columns:
                o_name = o_cid.replace("AR1_postprocess_", "AR(1) ") if _is_rho(o_cid) else o_cid[:22]
                ax.plot(o_data["Valid Date"], o_data["q_da"], color=other_palette[o_idx % len(other_palette)], linewidth=1.5,
                        linestyle=":", label=_label(f"Model: {o_name}", o_cid, o_data, with_kge=False), zorder=7)


        panel_tag = "a" if idx == 0 else "b"
        horizon_tag = f"Lead Time {lt} Day (t+{lt})" if lt == 1 else f"Lead Time {lt} Days (t+{lt})"
        context_tag = "Near-Term Forecast Alignment" if lt == 1 else "Horizon Persistence Dynamics"
        ax.set_title(f"({panel_tag}) {horizon_tag} — {context_tag}", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Streamflow (mm/day)", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor="#ffffff", framealpha=0.92)

        # Optional precipitation hyetograph (inverted, top of panel) per provider.
        if precip_df is not None and not precip_df.empty and not ref_slice.empty:
            from da_eval.forcings import PRECIP_COLORS, PRECIP_PRODUCTS
            p_lt = precip_df[precip_df["Lead Time (Days)"] == lt]
            d0, d1 = ref_slice["Valid Date"].min(), ref_slice["Valid Date"].max()
            p_lt = p_lt[(p_lt["Valid Date"] >= d0) & (p_lt["Valid Date"] <= d1)]
            p_cols = [c for c in precip_df.columns if c in PRECIP_PRODUCTS and p_lt[c].notna().any()]
            if p_cols:
                axp = ax.twinx()
                for c in p_cols:
                    is_fc = PRECIP_PRODUCTS[c][2]
                    lbl = f"{c} fcst (lead {lt})" if is_fc else f"{c} (obs)"
                    axp.step(p_lt["Valid Date"], p_lt[c], where="mid", color=PRECIP_COLORS.get(c, "#555"),
                             linewidth=1.3 if is_fc else 1.0, linestyle="-" if is_fc else "--", alpha=0.85, label=lbl)
                pmax = float(np.nanmax(p_lt[p_cols].to_numpy())) or 1.0
                axp.set_ylim(pmax * 2.8, 0)  # inverted; occupies roughly the top third
                axp.set_ylabel("Precip (mm/day)", fontsize=9.5, color="#1565c0")
                axp.tick_params(axis="y", labelsize=8.5, colors="#1565c0")
                axp.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9, title="Precipitation",
                           title_fontsize=8.5)
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, lo + (hi - lo) * 1.45)  # headroom so hydrograph peaks sit below the hyetograph

    # 6. Suptitle and Appendix Signpost Footnote
    area_str = f"{area:,.1f} km²" if pd.notna(area) else "N/A"
    arid_str = f"{aridity:.2f}" if pd.notna(aridity) else "N/A"
    prec_str = f"{p_mean:.1f} mm/d" if pd.notna(p_mean) else "N/A"

    title_lines = (
        f"Basin Case Study: {basin_name} [{basin_id}] ({country})\n"
        f"Drainage Area: {area_str} | Aridity Index (P/PET): {arid_str} | Mean Precip: {prec_str} | Backbone: {bb}"
    )
    plt.suptitle(title_lines, fontsize=13, fontweight="bold", y=1.03)

    footnote = (
        "*Signpost: While hydrographs report continuous daily NSE, NSE Skill Score (SS), and KGE metrics, comprehensive multi-lead metric decomposition, "
        "β-bias ratio, and peak flood error tables are compiled in Appendix Table A2."
    )
    fig.text(0.5, -0.015, footnote, ha="center", fontsize=9.5, style="italic", color="#5f6368")

    plt.tight_layout()
    return fig


# ==============================================================================
# Stratified Catchment ECDF & Horizon Decay Suite
# ==============================================================================

def plot_stratified_ecdf_suite(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    lead_time: int = 1,
    metric: str = "skill_nse",
    top_n_configs: int = 5,
    base_nse_range: Optional[Tuple[float, float]] = None,
    aridity_range: Optional[Tuple[float, float]] = None,
    p_mean_range: Optional[Tuple[float, float]] = None,
    area_range: Optional[Tuple[float, float]] = None,
    snow_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (18, 6.5),
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, pd.DataFrame, dict]:
    """Renders 2-Panel Stratified Catchment ECDF and Memory Decay Suite.

    Filters evaluation records strictly to catchments meeting the specified physical
    criteria (Aridity, Precip, Base NSE, Area, Snow Fraction), ranks the configurations
    specifically for that stratum, and plots:
      - Panel A: Empirical CDF of the candidate models on the filtered catchment cohort.
      - Panel B: Lead Time Horizon Decay (t+1..t+7) for the top configs on this stratum.
    """
    from da_eval.physical_strata import filter_cohort_by_attributes, build_stratified_leaderboard_table
    from da_eval.benchmarks import build_per_basin_optimal_eval, get_global_best_config
    from da_eval.ingestion import _ensure_skill_score_columns

    df_eval = _ensure_skill_score_columns(df_eval.copy())

    filtered_df, cohort_info = filter_cohort_by_attributes(
        df_eval=df_eval,
        df_meta=df_meta,
        base_nse_range=base_nse_range,
        aridity_range=aridity_range,
        p_mean_range=p_mean_range,
        area_range=area_range,
        snow_range=snow_range,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if filtered_df.empty or cohort_info["n_basins"] == 0:
        fig.suptitle(f"No Catchments Match Active Stratum Filter: {cohort_info['desc']}", fontsize=13, fontweight="bold")
        for ax in axes:
            ax.text(0.5, 0.5, "0 matching catchments in stratum", ha="center", va="center", fontsize=12)
        return fig, pd.DataFrame(), cohort_info

    df_lb, _ = build_stratified_leaderboard_table(
        df_eval=df_eval,
        df_meta=df_meta,
        lead_time=lead_time,
        metric=metric,
        base_nse_range=base_nse_range,
        aridity_range=aridity_range,
        p_mean_range=p_mean_range,
        area_range=area_range,
        snow_range=snow_range,
    )

    is_delta, m_col, b_col, rank_col, short_lbl, axis_lbl, resolved_xlim = _resolve_metric_spec(
        metric=metric, xlim=xlim
    )
    lead_times = detect_lead_times(filtered_df)
    ref_lead = lead_time if lead_time in lead_times else lead_times[0]
    is_ss = _is_ss(m_col)
    ref_zero_lbl = _zero_ref_label(short_lbl, is_ss)

    # Extract clean config IDs without medals
    top_clean_cfgs = []
    if not df_lb.empty:
        for c in df_lb["Config ID"].tolist():
            clean = c.replace("🥇 ", "").replace("🥈 ", "").replace("🥉 ", "").strip()
            if clean not in top_clean_cfgs:
                top_clean_cfgs.append(clean)
    top_sel_cfgs = top_clean_cfgs[:top_n_configs]

    # Global best across entire dataset for comparison
    global_best_cfg = get_global_best_config(df_eval, lead_time=ref_lead, metric_col=rank_col if rank_col in df_eval.columns else "NSE Delta")

    # -------------------------------------------------------------------------
    # Panel A: Stratified Empirical CDF (Lead ref_lead)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    sub_lead = filtered_df[filtered_df["Lead Time (Days)"] == ref_lead].copy()
    if sub_lead.empty:
        sub_lead = filtered_df.copy()

    # Distinct palette for top configs
    palette = ["#1a73e8", "#d93025", "#188038", "#e37400", "#9334e6", "#12b5cb", "#f29900", "#5f6368"]

    # 1. Baseline Open-Loop
    if is_delta:
        ax1.axvline(0.0, color="#3c4043", linewidth=2.0, linestyle="--", label=ref_zero_lbl, zorder=4)
        ax1.axhline(0.5, color="#9aa0a6", linewidth=0.8, linestyle=":", alpha=0.7)
    elif b_col and b_col in sub_lead.columns:
        base_vals = np.sort(sub_lead.drop_duplicates("Basin ID")[b_col].dropna().values)
        if len(base_vals) > 0:
            y_base = np.linspace(0, 1, len(base_vals))
            ax1.plot(base_vals, y_base, label=f"Baseline (Med: {np.median(base_vals):.3f})", color="#3c4043", linewidth=2.5, linestyle="-")

    # 2. Top-N Configs for this specific stratum
    for idx, cfg in enumerate(top_sel_cfgs):
        cfg_data = sub_lead[sub_lead["Config ID"] == cfg]
        if not cfg_data.empty and m_col in cfg_data.columns:
            vals = np.sort(cfg_data[m_col].dropna().values)
            if len(vals) > 0:
                y = np.linspace(0, 1, len(vals))
                color = palette[idx % len(palette)]
                med = float(np.median(vals))
                tag = f"#{idx+1} {cfg}"
                if is_delta:
                    win = (vals > 0.01).mean() * 100.0
                    lbl = f"{tag} (Med: {_fmt_val(med, is_ss)} | Win: {win:.1f}%)"
                else:
                    lbl = f"{tag} (Med: {med:.3f})"
                linewidth = 3.0 if idx == 0 else 2.0
                ax1.plot(vals, y, color=color, linewidth=linewidth, label=lbl, zorder=6 - idx)

    # 3. Global Best comparison (if not already plotted in top-N)
    if global_best_cfg not in top_sel_cfgs:
        g_data = sub_lead[sub_lead["Config ID"] == global_best_cfg]
        if not g_data.empty and m_col in g_data.columns:
            g_vals = np.sort(g_data[m_col].dropna().values)
            if len(g_vals) > 0:
                y_g = np.linspace(0, 1, len(g_vals))
                lbl_g = f"Global Best ({global_best_cfg}) [Med: {_fmt_val(np.median(g_vals), is_ss)}]" if is_delta else f"Global Best ({global_best_cfg}) [Med: {np.median(g_vals):.3f}]"
                ax1.plot(g_vals, y_g, color="#70757a", linewidth=2.0, linestyle=":", label=lbl_g, zorder=5)

    # 4. Per-Basin Best Oracle for this stratum
    if (sub_lead["Config ID"] == "Per-Basin Best DA").any() and m_col in sub_lead.columns:
        pb_vals = np.sort(sub_lead[sub_lead["Config ID"] == "Per-Basin Best DA"][m_col].dropna().values)
        if len(pb_vals) > 0:
            y_pb = np.linspace(0, 1, len(pb_vals))
            lbl_pb = f"Per-Basin Best Oracle (Med: {_fmt_val(np.median(pb_vals), is_ss)})" if is_delta else f"Per-Basin Best Oracle (Med: {np.median(pb_vals):.3f})"
            ax1.plot(pb_vals, y_pb, color="#0d47a1", linewidth=2.5, linestyle="--", label=lbl_pb, zorder=7)

    ax1.set_title(f"(A) Catchment ECDF (Lead t+{ref_lead})", fontsize=12, fontweight="bold", pad=8)
    ax1.set_xlabel(units.ss_label(axis_lbl) if is_ss else axis_lbl, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative Catchment Fraction", fontsize=11, fontweight="bold")
    ax1.set_xlim(*resolved_xlim)
    if is_ss:
        units.ss_axis(ax1, "x")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, fontsize=9.0, framealpha=0.92)

    # -------------------------------------------------------------------------
    # Panel B: Stratified Horizon Memory Decay (t+1..t+7)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    plot_cfgs = top_sel_cfgs[:4]
    if global_best_cfg not in plot_cfgs:
        plot_cfgs.append(global_best_cfg)

    # Baseline trajectory
    if not is_delta and b_col and b_col in filtered_df.columns:
        base_decay = [
            float(filtered_df[filtered_df["Lead Time (Days)"] == lt].drop_duplicates("Basin ID")[b_col].median())
            for lt in lead_times
        ]
        ax2.plot(lead_times, base_decay, color="#3c4043", marker="o", linewidth=2.5, label="Baseline Open-Loop", zorder=3)
    elif is_delta:
        ax2.axhline(0.0, color="#3c4043", linewidth=1.8, linestyle="--", label=ref_zero_lbl, zorder=3)

    for idx, cfg in enumerate(plot_cfgs):
        c_sub = filtered_df[filtered_df["Config ID"] == cfg]
        if not c_sub.empty and m_col in c_sub.columns:
            decay_pts = []
            for lt in lead_times:
                lt_v = c_sub[c_sub["Lead Time (Days)"] == lt][m_col].dropna()
                decay_pts.append(float(lt_v.median()) if len(lt_v) else np.nan)
            if any(np.isfinite(decay_pts)):
                color = palette[idx % len(palette)] if cfg in top_sel_cfgs else "#70757a"
                style = "-" if cfg in top_sel_cfgs else ":"
                marker = "s" if idx == 0 else "o"
                lbl = f"#{idx+1} {cfg}" if cfg in top_sel_cfgs else f"Global Best ({cfg})"
                ax2.plot(lead_times, decay_pts, color=color, linestyle=style, marker=marker, linewidth=2.2, label=lbl, zorder=5 + idx)

    # Oracle trajectory
    if (filtered_df["Config ID"] == "Per-Basin Best DA").any() and m_col in filtered_df.columns:
        pb_decay = [
            float(filtered_df[(filtered_df["Lead Time (Days)"] == lt) & (filtered_df["Config ID"] == "Per-Basin Best DA")][m_col].median())
            for lt in lead_times
        ]
        ax2.plot(lead_times, pb_decay, color="#0d47a1", marker="^", linewidth=2.2, linestyle="--", label="Per-Basin Best Oracle", zorder=8)

    ax2.set_title(f"(B) Horizon Persistence Decay Curve (Leads 1..{lead_times[-1]})", fontsize=12, fontweight="bold", pad=8)
    ax2.set_xlabel("Forecast Lead Time (Days)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(units.ss_label(f"Median {short_lbl} Metric") if is_ss else f"Median {short_lbl} Metric", fontsize=11, fontweight="bold")
    if is_ss:
        units.ss_axis(ax2, "y")
    ax2.set_xticks(lead_times)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.0, framealpha=0.92)

    # Suptitle summarizing active stratum filter
    med_base_str = f"{cohort_info['median_base_nse']:.2f}" if np.isfinite(cohort_info['median_base_nse']) else "N/A"
    med_arid_str = f"{cohort_info['median_aridity']:.2f}" if np.isfinite(cohort_info['median_aridity']) else "N/A"
    med_p_str = f"{cohort_info['median_p_mean']:.1f} mm/d" if np.isfinite(cohort_info['median_p_mean']) else "N/A"

    suptitle_text = (
        f"Catchment Stratum ECDF & Forecast Horizon Dynamics — Filter: {cohort_info['desc']}\n"
        f"Cohort: {cohort_info['n_basins']:,} / {cohort_info['total_basins']:,} catchments ({cohort_info['pct_total']:.1f}%) | "
        f"Median Base NSE: {med_base_str} | Median Aridity (P/PET): {med_arid_str} | Median Precip: {med_p_str}"
    )
    fig.suptitle(suptitle_text, fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout()

    return fig, df_lb, cohort_info

