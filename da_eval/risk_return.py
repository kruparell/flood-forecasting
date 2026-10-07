"""Risk–return (Pareto) view of DA configurations: typical gain vs downside risk across basins.

Each DA configuration becomes one point. Per basin, its score is the chosen metric (skill score by default)
aggregated over a user-chosen set of lead times (mean or sum). Across the basins common to every plotted
configuration it is then summarised by a *return* statistic (Y, e.g. median) and a *risk* statistic
(X, e.g. 10th percentile), both oriented so that up/right is better.

Public API:
  * ``build_risk_return_table``: one row per config (hyperparameters, stats, win rate, N, Pareto flag).
  * ``plot_risk_return``: grey cloud of all configs, highlighted slice coloured/marked by hyperparameters,
    Pareto frontier, reference models (Baseline, Global-Best DA, Per-Basin DA, PP post-processors).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections.abc import Mapping

from da_eval import units
from da_eval import palette
from da_eval.benchmarks import get_global_best_config
from da_eval.selection import PER_BASIN_DA, PER_BASIN_RHO, get_active
from da_eval.ingestion import _ensure_skill_score_columns, is_rho_config, parse_da_config_id

METRICS = {
    "skill": ("NSE Skill Score", "NSE skill score"),
    "delta": ("NSE Delta", "ΔNSE (DA − baseline)"),
    "nse": ("DA NSE", "NSE"),
    "kge_skill": ("KGE Skill Score", "KGE skill score"),
    "kge": ("DA KGE", "KGE"),
    "r_skill": ("Pearson-r Skill Score", "Pearson-r skill score"),
    "alpha_skill": ("Alpha-NSE Skill Score", "Alpha-NSE skill score"),
    "beta_kge_skill": ("Beta-KGE Skill Score", "Beta-KGE skill score"),
    "beta_nse_skill": ("Beta-NSE Skill Score", "Beta-NSE skill score"),
    "t0_skill": ("NSE Skill Score (t+0 in-window)", "In-window t+0 NSE skill score"),
}

# key: (label, higher_is_better)
STATS = {
    "median": ("50th percentile", True),
    "mean": ("Mean", True),
    "q05": ("5th percentile", True),
    "q10": ("10th percentile", True),
    "q25": ("25th percentile", True),
    "cvar10": ("Mean of worst 10% (CVaR)", True),
    "pct_improved": ("% basins improved (> 0)", True),
    "pct_harm": ("% basins harmed", False),
}
DEFAULT_RISK = "q10"
DEFAULT_RETURN = "median"

HP_PATTERNS = {
    "Window (Days)": (r"_w(\d+)", float),
    "Learning Rate": (r"_(?:lr|lrd)([0-9.]+(?:[eE][+-]?\d+)?)", float),
    "LR Ratio": (r"_r([0-9][0-9.]*)(?=_|$)", float),
    "Static LR": (r"_lrs([0-9.]+(?:[eE][+-]?\d+)?)", float),
    "BG Weight": (r"_(?:bg|bgd)([0-9.]+(?:[eE][+-]?\d+)?)", float),
    "Tolerance": (r"_tol([A-Za-z0-9.]+)", str),
    "Epochs": (r"_ep(\d+)", float),
    "Sweep Group": (r"^S\d+([A-Z])_", str),
}
HP_SHORT = {"Window (Days)": "w", "Learning Rate": "lr", "LR Ratio": "r", "Static LR": "lrs", "BG Weight": "bg",
            "Tolerance": "tol", "Epochs": "ep", "Sweep Group": "grp", "Target": "", "Loss": ""}

# Legend wording for the split (colour / shape) legend.
LEGEND_HEADERS = {
    "Learning Rate": "Learning rate (dynamic, lr_dyn)",
    "LR Ratio": "LR ratio r = lr_static / lr_dyn",
}


def _legend_value(col: str, v) -> str:
    """Legend text for one colour / shape value (LR ratio explains how fast the static embedding moves)."""
    if col == "Target":
        return TARGET_SHORT.get(str(v), str(v))
    txt = f"{HP_SHORT.get(col) or col} = {_fmt_val(v)}"
    if col == "LR Ratio":
        try:
            r = float(v)
        except (TypeError, ValueError):
            return txt
        if np.isclose(r, 1.0):
            txt += "  (static = dynamic)"
        elif r < 1:
            txt += f"  (static {1 / r:.3g}× lower)"
        else:
            txt += f"  (static {r:.3g}× higher)"
    return txt


_REF_SHAPES = {
    "baseline": dict(label="Open-loop baseline", marker="X", s=140),
    "global_da": dict(label="Global-Best DA", marker="*", s=320),
    "per_basin_da": dict(label="Per-Basin DA (oracle)", marker="D", s=110),
    "global_rho": dict(label="Global-Best PP", marker="P", s=150),
    "per_basin_rho": dict(label="Per-Basin PP", marker="h", s=150),
}


class _RefStyles(Mapping):
    """Reference-model scatter styles; colours come from the notebook-wide ``palette`` at draw time."""

    def __getitem__(self, key):
        return dict(_REF_SHAPES[key], color=palette.color(key))

    def __iter__(self):
        return iter(_REF_SHAPES)

    def __len__(self):
        return len(_REF_SHAPES)


REF_STYLES = _RefStyles()
DEFAULT_REFS = ("baseline", "global_da", "per_basin_da", "global_rho", "per_basin_rho")


def _is_sweep_cfg(cid: str) -> bool:
    c = str(cid)
    return not (is_rho_config(c) or "baseline" in c.lower() or "per-basin" in c.lower())


def parse_hyperparameters(config_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for cid in config_ids:
        row = {"Config ID": cid}
        for name, (pat, cast) in HP_PATTERNS.items():
            m = re.search(pat, str(cid))
            row[name] = cast(m.group(1)) if m else (np.nan if cast is float else None)
        if np.isnan(row.get("Learning Rate", np.nan)) and not np.isnan(row.get("Static LR", np.nan)):
            row["Learning Rate"] = row["Static LR"]
            row["Static LR"] = np.nan
        try:
            p = parse_da_config_id(cid)
            row["Target"] = p.get("Target")
            row["Loss"] = p.get("Loss")
        except Exception:
            row["Target"], row["Loss"] = None, None
        rows.append(row)
    return pd.DataFrame(rows)


def varying_hyperparameters(hp: pd.DataFrame) -> List[str]:
    cols = [c for c in hp.columns if c != "Config ID"]
    return [c for c in cols if hp[c].dropna().astype(str).nunique() > 1]


def _stats(v: np.ndarray, harm_threshold: float) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: np.nan for k in STATS} | {"N": 0}
    worst = np.sort(v)[: max(1, int(np.ceil(0.1 * v.size)))]
    return {
        "median": float(np.median(v)), "mean": float(np.mean(v)),
        "q05": float(np.quantile(v, 0.05)), "q10": float(np.quantile(v, 0.10)), "q25": float(np.quantile(v, 0.25)),
        "cvar10": float(worst.mean()), "pct_improved": float((v > 0).mean() * 100),
        "pct_harm": float((v < harm_threshold).mean() * 100), "N": int(v.size),
    }


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-dominated points when maximising both x and y."""
    ok = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros(len(x), dtype=bool)
    idx = np.where(ok)[0]
    for i in idx:
        dominated = np.any((x[idx] >= x[i]) & (y[idx] >= y[i]) & ((x[idx] > x[i]) | (y[idx] > y[i])))
        mask[i] = not dominated
    return mask


def resolve_reference_configs(df_eval: pd.DataFrame) -> Dict[str, str]:
    """Reference models from the notebook-wide selection (``session.set_selection``).

    Global-Best DA / PP come from the active selection; Per-Basin DA / PP are the synthetic rows it
    builds (``Per-Basin Best DA`` / ``AR1_Per_Basin_Best_Rho``). Without an active selection the library
    falls back to median skill score at t+1.
    """
    cfgs = set(df_eval["Config ID"].dropna().unique())
    refs = {}
    gb = get_global_best_config(df_eval, lead_time=1)
    if gb:
        refs["global_da"] = gb
    if PER_BASIN_DA in cfgs:
        refs["per_basin_da"] = PER_BASIN_DA
    fixed = [c for c in cfgs if is_rho_config(c) and "per_basin" not in c.lower() and "per-basin" not in c.lower()]
    if fixed:
        gbr = get_global_best_config(df_eval[df_eval["Config ID"].isin(fixed)], lead_time=1, exclude_rho=False)
        if gbr:
            refs["global_rho"] = gbr
    if PER_BASIN_RHO in cfgs:
        refs["per_basin_rho"] = PER_BASIN_RHO
    return refs


def build_risk_return_table(
    df_eval: pd.DataFrame,
    leads: Sequence[int] = (4, 5, 6, 7),
    metric: str = "skill",
    agg: str = "mean",
    harm_threshold: float = -0.01,
    basins: Optional[Iterable[str]] = None,
    min_coverage: float = 0.9,
) -> Tuple[pd.DataFrame, dict]:
    """Per-config risk/return statistics on the common basin set.

    Per basin: the metric aggregated (``agg`` = 'mean' or 'sum') over ``leads``; a basin counts only if every
    selected lead is finite. Configs covering < ``min_coverage`` of the best-covered config's basins are
    dropped (reported in ``info['dropped']``) so a few failed shards don't shrink the common set.
    Returns (table, info) where info has the common basins, N, dropped configs and reference mapping.
    Reference models follow the notebook-wide selection (see :func:`resolve_reference_configs`).
    """
    m_col, _ = METRICS.get(metric, METRICS["skill"])
    df = _ensure_skill_score_columns(df_eval) if m_col not in df_eval.columns else df_eval
    leads = sorted({int(l) for l in leads})
    refs = resolve_reference_configs(df)
    sweep = [c for c in df["Config ID"].dropna().unique() if _is_sweep_cfg(c)]
    keep_cfgs = set(sweep) | set(refs.values())
    d = df[df["Config ID"].isin(keep_cfgs) & df["Lead Time (Days)"].isin(leads)]
    if basins is not None:
        d = d[d["Basin ID"].astype(str).isin({str(b) for b in basins})]
    d = d[["Config ID", "Basin ID", "Lead Time (Days)", m_col] + (["Base NSE"] if metric == "nse" else [])]
    d = d.drop_duplicates(["Config ID", "Basin ID", "Lead Time (Days)"])

    def _score(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        wide = frame.pivot_table(index=["Config ID", "Basin ID"], columns="Lead Time (Days)", values=col, aggfunc="first")
        wide = wide.reindex(columns=leads)
        full = wide.notna().all(axis=1)
        s = wide[full].sum(axis=1) if agg == "sum" else wide[full].mean(axis=1)
        return s.rename("score").reset_index()

    scores = _score(d, m_col)
    cover = scores.groupby("Config ID")["Basin ID"].nunique()
    if cover.empty:
        return pd.DataFrame(), {"n_basins": 0, "dropped": [], "refs": refs, "leads": leads}
    thresh = min_coverage * cover.max()
    dropped = sorted(cover[cover < thresh].index.tolist())
    kept = cover[cover >= thresh].index
    common = None
    for cfg in kept:
        b = set(scores.loc[scores["Config ID"] == cfg, "Basin ID"])
        common = b if common is None else common & b
    common = common or set()
    scores = scores[scores["Config ID"].isin(kept) & scores["Basin ID"].isin(common)]

    rows = []
    for cfg, g in scores.groupby("Config ID"):
        rows.append({"Config ID": cfg} | _stats(g["score"].to_numpy(float), harm_threshold))
    if metric == "nse":  # the baseline point is the open-loop NSE itself
        base = _score(d.rename(columns={m_col: "_m"}), "Base NSE")
        base = base[base["Basin ID"].isin(common)].drop_duplicates("Basin ID")
        rows.append({"Config ID": "__baseline__"} | _stats(base["score"].to_numpy(float), harm_threshold))
    else:  # skill / delta are 0 for the baseline by construction
        rows.append({"Config ID": "__baseline__"} | _stats(np.zeros(len(common)), harm_threshold)
                    | {"pct_harm": 0.0, "pct_improved": 0.0})
    tbl = pd.DataFrame(rows)
    role = {v: k for k, v in refs.items()}
    role["__baseline__"] = "baseline"
    tbl["Role"] = tbl["Config ID"].map(lambda c: role.get(c, "sweep"))
    tbl["Is Sweep"] = tbl["Config ID"].map(_is_sweep_cfg) & (tbl["Config ID"] != "__baseline__")
    hp = parse_hyperparameters(tbl.loc[tbl["Is Sweep"], "Config ID"])
    tbl = tbl.merge(hp, on="Config ID", how="left")
    sel = get_active()
    info = {"n_basins": len(common), "basins": sorted(common), "dropped": dropped, "refs": refs,
            "leads": leads, "metric": metric, "agg": agg, "harm_threshold": harm_threshold,
            "selection": sel.label() if sel else "selected on median NSE skill score, t+1"}
    return tbl, info


TARGET_SHORT = {"Dynamic Embeddings": "H", "Both Embeddings": "S+H", "All Embeddings": "S+H+F",
                "Static Embeddings": "S"}


def short_label(row: pd.Series, hp_cols: Sequence[str]) -> str:
    parts = []
    for c in hp_cols:
        v = row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if c == "Target":
            parts.append(TARGET_SHORT.get(str(v), str(v)))
            continue
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        parts.append(f"{HP_SHORT.get(c, c)}{v}" if HP_SHORT.get(c, c) else str(v))
    return " ".join(parts)


def _label_cols(frame: pd.DataFrame, hp_cols: Sequence[str]) -> List[str]:
    """Hyperparameters that differ within ``frame`` (falls back to all varying ones)."""
    cols = [c for c in hp_cols if c in frame.columns and frame[c].astype(str).nunique() > 1]
    return cols or list(hp_cols)


def _fmt_val(v) -> str:
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return f"{v:g}" if isinstance(v, float) else str(v)


def plot_risk_return(
    tbl: pd.DataFrame,
    info: dict,
    risk_stat: str = DEFAULT_RISK,
    return_stat: str = DEFAULT_RETURN,
    color_by: Optional[str] = "Learning Rate",
    marker_by: Optional[str] = None,
    label_by: Optional[str] = None,
    fixed: Optional[Dict[str, object]] = None,
    show_refs: Sequence[str] = DEFAULT_REFS,
    show_frontier: bool = True,
    label_frontier: int = 5,
    annotate: Sequence[str] = (),
    figsize: Tuple[float, float] = (11, 7.5),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    inset_refs: Sequence[str] = (),
    inset_loc: str = "auto",
    inset_size: Tuple[float, float] = (0.34, 0.34),
    legend: bool = True,
    inset_style: str = "callout",
    legend_style: str = "split",
    frontier_style: str = "step",
    frontier_shade: bool = False,
    dim_dominated: Optional[float] = None,
    frontier_extend: bool = True,
    frontier_color: str = "#202124",
    callout_loc: str = "margin",
) -> plt.Figure:
    """Risk (X) vs return (Y). ``fixed`` = {hyperparameter: value} picks the highlighted slice.

    ``inset_refs``: reference keys (e.g. ``("per_basin_da",)``) that, when they fall outside the main view, do not
    stretch the axes (inside the view they are drawn normally). ``inset_style='callout'`` (default) pins the marker
    to the axes edge with an arrow pointing towards its true position and a small box giving its true
    coordinates; ``'overview'`` draws a shrunken whole-plot overview inset instead (``inset_loc``/``inset_size``).
    ``legend_style='split'`` (default) gives one legend block for colours and one for marker shapes instead of
    one entry per colour x shape combination (``'combined'``).
    Frontier emphasis: ``frontier_style`` 'step' (staircase = exact non-dominated boundary) | 'line' (straight
    connectors) | 'legacy' (old thin red dashed line under the points); ``frontier_shade`` lightly shades the
    region dominated by the frontier; ``dim_dominated`` = alpha for coloured configs NOT on the frontier (None =
    no dimming), so frontier configs keep full colour; ``frontier_color`` = line/ring colour; ``frontier_extend``
    continues the frontier to the axes edges (horizontal from the best-return config towards the risky edge,
    vertical from the safest config towards the low-return edge). ``callout_loc``: where the off-scale coordinate
    box goes - 'margin' (right of the axes) or 'inside' (emptiest inside corner; used for multi-panel figures).
    Set ``label_by=None`` and ``label_frontier=0`` for a clean figure without in-plot text.
    """
    if tbl is None or tbl.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No configurations with the selected leads", ha="center", transform=ax.transAxes)
        return fig
    fig = ax.figure if ax is not None else plt.subplots(figsize=figsize)[1].figure
    ax = ax or fig.axes[0]
    sel_lbl = info.get("selection", "")
    x_lbl, x_hib = STATS[risk_stat]
    y_lbl, y_hib = STATS[return_stat]
    sgn_x = 1 if x_hib else -1
    sgn_y = 1 if y_hib else -1
    # NSE skill-score axes follow the notebook-wide display unit (``units.set_ss_percent``); % basins stats don't.
    x_ss = info.get("metric") == "skill" and risk_stat not in ("pct_improved", "pct_harm")
    y_ss = info.get("metric") == "skill" and return_stat not in ("pct_improved", "pct_harm")
    fmt_x = units.fmt_ss if x_ss else (lambda v: f"{v:+.3f}")
    fmt_y = units.fmt_ss if y_ss else (lambda v: f"{v:+.3f}")

    sweep = tbl[tbl["Is Sweep"]].copy()
    hp_cols = varying_hyperparameters(sweep[[c for c in sweep.columns if c in HP_PATTERNS or c in ("Target", "Loss", "Config ID")]])

    pm_all = pd.Series(False, index=sweep.index)
    if show_frontier and len(sweep) > 1:
        pm_all[:] = _pareto_mask(sgn_x * sweep[risk_stat].to_numpy(float), sgn_y * sweep[return_stat].to_numpy(float))
    legacy = frontier_style == "legacy"
    dim = None if (legacy or not show_frontier) else dim_dominated

    # Layer 1: all configs (grey). With colour/shape highlighting, the legend entry describes only the configs
    # left grey, i.e. outside the ``fixed`` slice or missing the colour / shape hyperparameter.
    grey_lbl = f"All DA configs (n={len(sweep)})"
    if fixed or color_by or marker_by:
        keep = pd.Series(True, index=sweep.index)
        for k, v in {k: v for k, v in (fixed or {}).items() if v not in (None, "any", "")}.items():
            if k in sweep.columns:
                keep &= sweep[k].map(_fmt_val) == _fmt_val(v)
        missing = []
        for col in (color_by, marker_by):
            if col in sweep.columns:
                miss = sweep[col].isna()
                if (miss & keep).any():
                    missing.append(col)
                keep &= ~miss
        grey = sweep[~keep]
        if grey.empty:
            grey_lbl = "_nolegend_"
        else:
            tg = grey["Target"].map(lambda t: TARGET_SHORT.get(str(t), str(t))).unique() if "Target" in grey else []
            why = f"no {' / '.join(missing)}" if missing else "outside slice"
            who = f"{tg[0]} only, " if len(tg) == 1 and missing else ""
            grey_lbl = f"{who}{why} (n={len(grey)})"
    ax.scatter(sweep[risk_stat], sweep[return_stat], s=42, color="#cccccc", alpha=0.6, edgecolors="none", zorder=2,
               label=grey_lbl)

    # Layer 2: highlighted slice
    split_handles = []
    sl = sweep
    fixed = {k: v for k, v in (fixed or {}).items() if v not in (None, "any", "")}
    for k, v in fixed.items():
        if k in sl.columns:
            sl = sl[sl[k].map(_fmt_val) == _fmt_val(v)]
    highlight = bool(fixed) or bool(color_by) or bool(marker_by)
    if highlight and not sl.empty:
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
        mk_vals = sorted(sl[marker_by].dropna().unique(), key=lambda z: (str(type(z)), z)) if marker_by in sl.columns else [None]
        if color_by in sl.columns and sl[color_by].notna().any():
            c_vals = sorted(sl[color_by].dropna().unique(), key=lambda z: (str(type(z)), z))
            colors = palette.value_colors(c_vals)  # ordered viridis for numeric values (colour-blind safe)
        else:
            c_vals, colors = [None], {None: palette.QUALITATIVE[0]}
        for cv in c_vals:
            for j, mv in enumerate(mk_vals):
                g = sl
                if cv is not None:
                    g = g[g[color_by] == cv]
                if mv is not None:
                    g = g[g[marker_by] == mv]
                if g.empty:
                    continue
                parts = []
                if cv is not None:
                    parts.append(f"{HP_SHORT.get(color_by) or color_by}={_fmt_val(cv)}")
                if mv is not None:
                    parts.append(f"{HP_SHORT.get(marker_by) or marker_by}={_fmt_val(mv)}")
                split = legend_style == "split"
                lab = "_nolegend_" if split else ", ".join(parts) + f" (n={len(g)})"
                on_f = pm_all.reindex(g.index).fillna(False).to_numpy(bool)
                for sub, a, z in ((g[~on_f], 0.95 if dim is None else dim, 4), (g[on_f], 0.95, 6)):
                    if sub.empty:
                        continue
                    ax.scatter(sub[risk_stat], sub[return_stat], s=70, color=colors[cv],
                               marker=markers[j % len(markers)], edgecolors="#202124", linewidths=0.5, alpha=a,
                               zorder=z, label=lab)
                    lab = "_nolegend_"
        if legend_style == "split":
            from matplotlib.lines import Line2D
            _nm = _legend_value
            blank = Line2D([], [], ls="", marker="")
            if c_vals != [None]:
                split_handles.append((blank, f"$\\bf{{Colour:}}$ {LEGEND_HEADERS.get(color_by, color_by)}"))
                for cv in c_vals:
                    split_handles.append((Line2D([], [], ls="", marker="o", ms=8, mfc=colors[cv], mec="#202124",
                                                 mew=0.5), _nm(color_by, cv)))
            if mk_vals != [None]:
                split_handles.append((blank, f"$\\bf{{Shape:}}$ {LEGEND_HEADERS.get(marker_by, marker_by)}"))
                for j, mv in enumerate(mk_vals):
                    split_handles.append((Line2D([], [], ls="", marker=markers[j % len(markers)], ms=8, mfc="#bdc1c6",
                                                 mec="#202124", mew=0.7), _nm(marker_by, mv)))
        if label_by in sl.columns:
            for _, r in sl.iterrows():
                ax.annotate(f"{HP_SHORT.get(label_by) or label_by}{_fmt_val(r[label_by])}",
                            (r[risk_stat], r[return_stat]), xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color="#3c4043", zorder=5)

    # Layer 3: Pareto frontier over all sweep configs
    front = pd.DataFrame()
    if show_frontier and len(sweep) > 1:
        front = sweep[pm_all.to_numpy(bool)].sort_values(risk_stat)
        f_lbl = f"Pareto frontier ({len(front)} configs)"
        if legacy:
            ax.plot(front[risk_stat], front[return_stat], color="#d93025", lw=1.4, ls="--", zorder=3, label=f_lbl)
            ax.scatter(front[risk_stat], front[return_stat], s=110, facecolors="none", edgecolors="#d93025",
                       linewidths=1.4, zorder=5)
        else:
            import matplotlib.patheffects as pe
            halo = [pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()]
            ax.plot(front[risk_stat], front[return_stat], color=frontier_color, lw=2.2, zorder=5.5,
                    drawstyle="steps-pre" if (frontier_style == "step" and sgn_x * sgn_y > 0) else
                    ("steps-post" if frontier_style == "step" else "default"),
                    solid_joinstyle="miter", path_effects=halo, label=f_lbl)
            ax.scatter(front[risk_stat], front[return_stat], s=150, facecolors="none", edgecolors=frontier_color,
                       linewidths=1.6, zorder=6.5)
        top = front.sort_values(return_stat, ascending=not y_hib).head(label_frontier)
        f_cols = _label_cols(front, hp_cols)
        for i, (_, r) in enumerate(top.iterrows()):
            ax.annotate(short_label(r, f_cols), (r[risk_stat], r[return_stat]), xytext=(7, 6 if i % 2 else -11),
                        textcoords="offset points", fontsize=7.5, color="#a50e0e", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for cid in annotate or ():
        r = tbl[tbl["Config ID"] == cid]
        if not r.empty:
            r = r.iloc[0]
            ax.annotate(short_label(r, hp_cols) or cid, (r[risk_stat], r[return_stat]), xytext=(8, 8),
                        textcoords="offset points", fontsize=8.5, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#5f6368", lw=0.7), zorder=7)

    # Layer 4: reference models (those in ``inset_refs`` only go in the inset)
    inset_refs = [k for k in (inset_refs or ()) if k in REF_STYLES]
    ref_pts = {}
    for key in list(dict.fromkeys(list(show_refs) + inset_refs)):
        r = tbl[tbl["Role"] == key]
        if r.empty:
            continue
        st = REF_STYLES[key]
        r = r.iloc[0]
        lbl = st["label"]
        if key == "global_rho":
            lbl += f" (ρ={str(r['Config ID']).split('rho')[-1]})"
        ref_pts[key] = (r[risk_stat], r[return_stat], lbl)
        if key in inset_refs:
            continue  # decided below, once the main view is known
        ax.scatter([r[risk_stat]], [r[return_stat]], marker=st["marker"], color=st["color"], s=st["s"],
                   edgecolors="black", linewidths=0.8, zorder=8, label=lbl)
    ax.autoscale_view()
    (vx0, vx1), (vy0, vy1) = ax.get_xlim(), ax.get_ylim()
    outside = []
    for key in inset_refs:
        if key not in ref_pts:
            continue
        x, y, lbl = ref_pts[key]
        st = REF_STYLES[key]
        if vx0 <= x <= vx1 and vy0 <= y <= vy1:
            ax.scatter([x], [y], marker=st["marker"], color=st["color"], s=st["s"], edgecolors="black",
                       linewidths=0.8, zorder=8, label=lbl)
        else:
            outside.append(key)
            if inset_style == "overview":
                ax.scatter([], [], marker=st["marker"], color=st["color"], s=st["s"], edgecolors="black",
                           linewidths=0.8, label=lbl + " (inset)")
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    if frontier_extend and not legacy and not front.empty:
        import matplotlib.patheffects as pe
        halo = [pe.Stroke(linewidth=4.6, foreground="white"), pe.Normal()]
        best_y = front.loc[front[return_stat].idxmax() if y_hib else front[return_stat].idxmin()]
        best_x = front.loc[front[risk_stat].idxmax() if x_hib else front[risk_stat].idxmin()]
        edge_x = vx0 if x_hib else vx1
        edge_y = vy0 if y_hib else vy1
        for xs, ys in (([edge_x, best_y[risk_stat]], [best_y[return_stat]] * 2),
                       ([best_x[risk_stat]] * 2, [best_x[return_stat], edge_y])):
            ax.plot(xs, ys, color=frontier_color, lw=2.2, zorder=5.5, path_effects=halo, solid_capstyle="butt")
        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
    if frontier_shade and not legacy and not front.empty and x_hib and y_hib:
        # region dominated by the frontier (something on the frontier is both safer and better)
        fx = np.r_[vx0, front[risk_stat].to_numpy(float)]
        fy = np.r_[front[return_stat].iloc[0], front[return_stat].to_numpy(float)]
        ax.fill_between(fx, fy, vy0, step="pre", color=frontier_color, alpha=0.06, lw=0, zorder=1,
                        label="Dominated by frontier")

    if info.get("metric") in ("skill", "delta"):
        if risk_stat not in ("pct_improved", "pct_harm"):
            ax.axvline(0, color="#9aa0a6", lw=0.8, zorder=1)
        if return_stat not in ("pct_improved", "pct_harm"):
            ax.axhline(0, color="#9aa0a6", lw=0.8, zorder=1)

    m_lbl = METRICS.get(info.get("metric", "skill"), METRICS["skill"])[1]
    leads = info.get("leads", [])
    lead_s = ("t+" + ",".join(str(l) for l in leads)) if len(leads) < 4 or leads != list(range(leads[0], leads[-1] + 1)) \
        else f"t+{leads[0]}–{leads[-1]}"
    score_s = f"{info.get('agg', 'mean')} {m_lbl} over {lead_s}" if len(leads) > 1 else f"{m_lbl} at {lead_s}"
    arrow_x = "→ safer" if x_hib else "← safer"
    x_score = units.ss_label("per-basin score") if x_ss else "per-basin score"
    y_score = units.ss_label("per-basin score") if y_ss else "per-basin score"
    ax.set_xlabel(f"Risk: {x_lbl} of {x_score}  ({arrow_x})", fontsize=10.5)
    ax.set_ylabel(f"Return: {y_lbl} of {y_score}  (↑ better)", fontsize=10.5)
    if x_ss:
        units.ss_axis(ax, "x")
    if y_ss:
        units.ss_axis(ax, "y")
    fixed_s = ", ".join(f"{HP_SHORT.get(k) or k}={_fmt_val(v)}" for k, v in fixed.items())
    ax.set_title(title or f"Risk–return of DA configurations — score = {score_s}\n"
                          f"N = {info.get('n_basins', 0):,} common basins | references {sel_lbl}"
                          + (f" | slice: {fixed_s}" if fixed_s else ""), fontsize=11.5, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    callouts = []
    if outside and inset_style == "callout":
        for i, key in enumerate(outside):
            x, y, lbl = ref_pts[key]
            pts = None
            if callout_loc == "inside":
                x0_, x1_ = ax.get_xlim()
                y0_, y1_ = ax.get_ylim()
                pts = list(zip((sweep[risk_stat].to_numpy(float) - x0_) / (x1_ - x0_),
                               (sweep[return_stat].to_numpy(float) - y0_) / (y1_ - y0_)))
                for ln in ax.get_lines():  # frontier (incl. edge extensions): densely sampled so boxes avoid it
                    if ln.get_color() != frontier_color:
                        continue
                    lx = (np.asarray(ln.get_xdata(), float) - x0_) / (x1_ - x0_)
                    ly = (np.asarray(ln.get_ydata(), float) - y0_) / (y1_ - y0_)
                    for k in range(len(lx) - 1):
                        tt = np.linspace(0, 1, 25)
                        pts += list(zip(lx[k] + tt * (lx[k + 1] - lx[k]), ly[k] + tt * (ly[k + 1] - ly[k])))
            callouts.append(_draw_offscale_callout(ax, x, y, lbl, REF_STYLES[key], x_lbl, y_lbl, slot=i, pts=pts,
                                                   fmt_x=fmt_x, fmt_y=fmt_y))
    h_auto, l_auto = ax.get_legend_handles_labels()
    ax._rr_legend = dict(split=list(split_handles), other=list(zip(h_auto, l_auto)))  # for shared legends
    if legend:
        h, l = h_auto, l_auto
        if split_handles:
            from matplotlib.lines import Line2D
            gap = (Line2D([], [], ls="", marker=""), " ")
            pairs = split_handles + [gap] + list(zip(h, l))
            h, l = [p[0] for p in pairs], [p[1] for p in pairs]
        callouts = [c for c in callouts if pts_is_margin(c)]
        ax.legend(h, l, loc="upper left" if callouts else "center left",
                  bbox_to_anchor=(1.01, 1.0 - 0.14 * len(callouts) - 0.03) if callouts else (1.01, 0.5),
                  fontsize=9, frameon=True,
                  handletextpad=0.4, labelspacing=0.45)
    if outside and inset_style != "callout":
        ins = _draw_overview_inset(ax, sweep, front, {k: v[:2] for k, v in ref_pts.items()}, risk_stat, return_stat,
                                   inset_loc, inset_size, zero_lines=info.get("metric") in ("skill", "delta"))
        if x_ss:
            units.ss_axis(ins, "x")
        if y_ss:
            units.ss_axis(ins, "y")
    if ax.figure.get_layout_engine() is None:
        fig.tight_layout()
    fig._risk_return_frontier = front  # handy for callers
    return fig


def pts_is_margin(ann) -> bool:
    """True if a callout annotation sits in the right-hand margin (so the legend must move below it)."""
    return getattr(ann, "_rr_margin", False)


def _draw_offscale_callout(ax, x, y, label, st, x_name, y_name, slot=0, pts=None, fmt_x=None, fmt_y=None):
    """Pin an off-scale reference onto the axes border (arrow = direction of its true position) and write its
    true coordinates in a small box in the right-hand margin, so nothing covers the plotted configurations.
    ``fmt_x`` / ``fmt_y``: value -> text for the coordinates (default ``+0.123``)."""
    fmt_x = fmt_x or (lambda v: f"{v:+.3f}")
    fmt_y = fmt_y or (lambda v: f"{v:+.3f}")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)
    cx, cy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
    dx = 0 if fx == cx else (1 if fx > cx else -1)
    dy = 0 if fy == cy else (1 if fy > cy else -1)
    tr = ax.transAxes
    label = label[:-1] + ", off-scale)" if label.endswith(")") else label + " (off-scale)"
    ax.scatter([cx], [cy], transform=tr, marker=st["marker"], color=st["color"], s=st["s"], edgecolors="black",
               linewidths=0.8, zorder=10, clip_on=False, label=label)
    ax.annotate("", xy=(cx + 0.045 * dx, cy + 0.045 * dy), xytext=(cx + 0.015 * dx, cy + 0.015 * dy),
                xycoords=tr, textcoords=tr, zorder=10, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=1.5, mutation_scale=13))
    if dy > 0:
        ax.set_title(ax.get_title(), fontsize=ax.title.get_fontsize(), fontweight=ax.title.get_fontweight(), pad=16)
    arrow = {(1, 1): "↗", (1, 0): "→", (1, -1): "↘", (0, 1): "↑", (0, -1): "↓", (-1, 1): "↖", (-1, 0): "←",
             (-1, -1): "↙"}.get((dx, dy), "")
    text = f"{arrow} {label}\n{x_name}: {fmt_x(x)}\n{y_name}: {fmt_y(y)}"
    box = dict(boxstyle="round,pad=0.35", fc="white", ec=st["color"], lw=1.0)
    if pts is None:  # right-hand margin
        ann = ax.annotate(text, (1.02, 1.0 - 0.14 * slot), xycoords=tr, ha="left", va="top", fontsize=9,
                          color="#202124", zorder=10, annotation_clip=False, bbox=box)
        ann._rr_margin = True
        return ann
    # inside: emptiest corner (box ~0.42 x 0.17 axes fraction), avoiding the pinned marker
    w, h = 0.42, 0.17
    corners = {"upper left": (0.02, 0.98), "upper right": (0.98, 0.98), "lower left": (0.02, 0.02),
               "lower right": (0.98, 0.02)}
    def _box(c):
        ax_, ay_ = corners[c]
        bx0 = ax_ if "left" in c else ax_ - w
        by0 = ay_ - h if "upper" in c else ay_
        return bx0, by0
    def _n(c):
        bx0, by0 = _box(c)
        inside = [bx0 - 0.02 <= a <= bx0 + w + 0.02 and by0 - 0.02 <= b <= by0 + h + 0.02 for a, b in pts + [(cx, cy)]]
        return sum(inside) + 100 * inside[-1]
    c = min(corners, key=_n)
    ha = "left" if "left" in c else "right"
    va = "top" if "upper" in c else "bottom"
    return ax.annotate(text, corners[c], xycoords=tr, ha=ha, va=va, fontsize=9, color="#202124", zorder=10,
                       bbox=box)


def _draw_overview_inset(ax, sweep, front, ref_pts, xcol, ycol, loc, size, zero_lines=True):
    """Small overview axes: all configs (grey), frontier, reference markers and a box for the main view."""
    import matplotlib.patches as mpatches
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = size
    corners = {"upper left": (0.02, 0.98 - h), "upper right": (0.98 - w, 0.98 - h),
               "lower left": (0.02, 0.02), "lower right": (0.98 - w, 0.02)}
    if loc not in corners:  # auto: corner covering the fewest plotted points (in axes fractions)
        fx = (sweep[xcol].to_numpy(float) - x0) / (x1 - x0)
        fy = (sweep[ycol].to_numpy(float) - y0) / (y1 - y0)
        pts = list(zip(fx, fy)) + [((p[0] - x0) / (x1 - x0), (p[1] - y0) / (y1 - y0)) for p in ref_pts.values()]
        def _n(c):
            cx, cy = corners[c]
            return sum(cx - 0.03 <= a <= cx + w + 0.03 and cy - 0.03 <= b <= cy + h + 0.03 for a, b in pts)
        loc = min(corners, key=_n)
    pos = corners[loc]
    ins = ax.inset_axes([pos[0], pos[1], w, h])
    ins.scatter(sweep[xcol], sweep[ycol], s=8, color="#9aa0a6", alpha=0.7, edgecolors="none", zorder=2)
    if front is not None and not front.empty:
        ins.plot(front[xcol], front[ycol], color="#d93025", lw=0.9, ls="--", zorder=3)
    for key, (x, y) in ref_pts.items():
        st = REF_STYLES[key]
        ins.scatter([x], [y], marker=st["marker"], color=st["color"], s=st["s"] * 0.35, edgecolors="black",
                    linewidths=0.5, zorder=5)
    if zero_lines:
        ins.axhline(0, color="#9aa0a6", lw=0.6, zorder=1)
        ins.axvline(0, color="#9aa0a6", lw=0.6, zorder=1)
    ins.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#202124", lw=0.8, ls=":", zorder=6))
    xs = [x0, x1] + [p[0] for p in ref_pts.values()]
    ys = [y0, y1] + [p[1] for p in ref_pts.values()]
    px, py = 0.08 * (max(xs) - min(xs)), 0.08 * (max(ys) - min(ys))
    ins.set_xlim(min(xs) - px, max(xs) + px)
    ins.set_ylim(min(ys) - py, max(ys) + py)
    ins.tick_params(labelsize=6.5, length=2, pad=1, labelleft=False, labelright=True)
    ins.set_facecolor("white")
    ins.set_title("Overview (dotted box = main view)", fontsize=7.5, pad=2)
    for sp in ins.spines.values():
        sp.set_edgecolor("#5f6368")
    return ins


def plot_risk_return_panels(
    panels: Sequence[Tuple[pd.DataFrame, dict, str]],
    figsize: Tuple[float, float] = (17, 7.2),
    legend_fontsize: float = 10,
    sharey: bool = False,
    **kwargs,
) -> plt.Figure:
    """Side-by-side risk-return panels (e.g. t+1-3 | t+4-7) with ONE shared legend below the panels.

    ``panels``: list of ``(tbl, info, title)`` from ``session.get_risk_return_table(...)``. ``kwargs`` go to
    :func:`plot_risk_return` for every panel (same colour/shape encoding, so the legend is valid for all).
    The legend has one column per block: colours | shapes | references & frontier. Off-scale coordinate boxes
    are placed inside each panel (emptiest corner).
    """
    from matplotlib.lines import Line2D
    kwargs = {**kwargs, "legend": False, "callout_loc": "inside"}
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=sharey, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (tbl, info, title) in zip(axes, panels):
        plot_risk_return(tbl, info, title=title, ax=ax, **kwargs)
    for ax in axes:  # align titles (an upward off-scale marker adds title padding in its own panel)
        ax.set_title(ax.get_title(), fontsize=ax.title.get_fontsize(), fontweight=ax.title.get_fontweight(), pad=16)
    # merge legend entries across panels (per-panel counts / off-scale tags removed so labels match)
    import re as _re
    def _clean(lbl):
        lbl = lbl.replace(", off-scale)", ")").replace(" (off-scale)", "")
        return _re.sub(r"^Pareto frontier \(\d+ configs\)$", "Pareto frontier", lbl)
    split, other, seen = [], [], set()
    for ax in axes:
        d = getattr(ax, "_rr_legend", {})
        if not split and d.get("split"):
            split = d["split"]
        for h, l in d.get("other", []):
            l = _clean(l)
            if l not in seen and not l.startswith("_"):
                seen.add(l)
                other.append((h, l))
    blank = (Line2D([], [], ls="", marker=""), " ")
    blocks, cur = [], []
    for hl in split:  # split = [header, entries..., header, entries...]
        if hl[1].startswith("$\\bf") and cur:
            blocks.append(cur)
            cur = []
        cur.append(hl)
    if cur:
        blocks.append(cur)
    blocks.append(other)
    n = max(len(b) for b in blocks)
    items = [hl for b in blocks for hl in (b + [blank] * (n - len(b)))]  # column-major fill => 1 block/column
    fig.legend([h for h, _ in items], [l for _, l in items], loc="outside lower center", ncol=len(blocks),
               fontsize=legend_fontsize, frameon=True, handletextpad=0.4, columnspacing=2.5)
    return fig


def frontier_table(tbl: pd.DataFrame, risk_stat: str = DEFAULT_RISK, return_stat: str = DEFAULT_RETURN) -> pd.DataFrame:
    """Pareto-optimal sweep configs with their stats and varying hyperparameters, best return first."""
    sweep = tbl[tbl["Is Sweep"]].copy()
    if sweep.empty:
        return sweep
    sx = 1 if STATS[risk_stat][1] else -1
    sy = 1 if STATS[return_stat][1] else -1
    pm = _pareto_mask(sx * sweep[risk_stat].to_numpy(float), sy * sweep[return_stat].to_numpy(float))
    hp_cols = varying_hyperparameters(sweep[[c for c in sweep.columns if c in HP_PATTERNS or c in ("Target", "Loss", "Config ID")]])
    cols = ["Config ID"] + hp_cols + [return_stat, risk_stat, "pct_improved", "pct_harm", "N"]
    cols = list(dict.fromkeys(cols))
    return sweep[pm].sort_values(return_stat, ascending=not STATS[return_stat][1])[cols].reset_index(drop=True)


SS_STATS = ("median", "mean", "q05", "q10", "q25", "cvar10")  # stats in the metric's own units (not % basins)


def style_frontier_table(ft: pd.DataFrame, metric: str = "skill"):
    """Styled :func:`frontier_table` (display only). Stats in native units get 4 decimals; for ``metric='skill'``
    the NSE skill-score stats follow the notebook-wide display unit (``units.set_ss_percent``)."""
    fmt = {c: "{:.4f}" for c in ft.columns if ft[c].dtype.kind == "f" and c not in HP_PATTERNS}
    ss_cols = [c for c in fmt if c in SS_STATS] if (metric == "skill" and units.ss_percent()) else []
    for c in ss_cols:
        fmt[c] = units.ss_formatter()
    sty = ft.style.format(fmt).hide(axis="index")
    if ss_cols:
        sty = sty.format_index(lambda c: units.ss_label(c) if c in ss_cols else c, axis=1)
    return sty
