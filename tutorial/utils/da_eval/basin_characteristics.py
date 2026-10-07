"""Per-characteristic DA efficacy: skill score by catchment-characteristic bin on ONE common gauge set.

Replaces the all-in-one physical-strata table for the notebook viewer. Differences from
``physical_strata.compute_consolidated_physical_strata``:

* Reference models (Global-Best DA, Per-Basin DA, Global-Best / Per-Basin PP) come from the notebook-wide
  selection via ``session.get_reference_skill`` -- never re-ranked here.
* Every characteristic uses the same gauges (those with a finite score for all reference models at every
  lead), so N and "% improved / harmed" are comparable across characteristics.
* Scores can be a single lead or a window (per-gauge mean over the leads, as in the selection).
* Bins with fewer than ``min_n`` gauges are merged into their smaller neighbour (labels are regenerated
  from the merged edges, so the range shown is always the real one).
* HydroATLAS ``slp_dg_sav`` is stored in tenths of a degree; it is converted to degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from da_eval import palette, units

ALL_MODELS = ("Global-Best DA", "Per-Basin DA", "Global-Best PP", "Per-Basin PP")
HARM_THRESHOLD = 0.01


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    column: str                      # column in the per-gauge frame
    edges: Tuple[float, ...]         # bin edges incl. -inf / inf
    names: Tuple[str, ...]           # one per bin
    unit: str = ""
    fmt: Callable[[float], str] = lambda v: f"{v:g}"


def _fmt_area(v: float) -> str:
    return f"{v:,.0f}"


DIMENSIONS: Dict[str, Dimension] = {d.key: d for d in (
    Dimension("baseline", "Baseline NSE", "Base NSE", (-np.inf, 0.0, 0.6, np.inf),
              ("Poor", "Moderate", "Strong")),
    Dimension("aridity", "Aridity index (P/PET)", "unep_aridity_index", (-np.inf, 0.2, 0.5, 0.65, np.inf),
              ("Arid", "Semi-arid", "Dry sub-humid", "Humid")),
    Dimension("area", "Catchment area", "area", (-np.inf, 500.0, 2500.0, np.inf),
              ("Small", "Medium", "Large"), unit=" km²", fmt=_fmt_area),
    Dimension("snow", "Snow fraction", "frac_snow", (-np.inf, 0.10, 0.30, np.inf),
              ("Rain-fed", "Mixed", "Snow-melt")),
    Dimension("slope", "Terrain slope", "slope_deg", (-np.inf, 2.0, 8.0, np.inf),
              ("Flat", "Rolling", "Steep"), unit="°"),
    Dimension("precip", "Mean precipitation", "p_mean", (-np.inf, 1.5, 3.5, np.inf),
              ("Low", "Moderate", "High"), unit=" mm/d"),
)}


def resolve_dimension(name: str) -> Dimension:
    """'Aridity', 'aridity index', 'area', 'Baseline NSE', ... -> Dimension."""
    n = str(name).lower().strip()
    for d in DIMENSIONS.values():
        if n in (d.key, d.label.lower()) or d.label.lower().startswith(n) or n.startswith(d.key):
            return d
    raise KeyError(f"Unknown characteristic {name!r}; choose from {[d.label for d in DIMENSIONS.values()]}")


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def gauge_frame(session, leads: Sequence[int], models: Sequence[str] = ALL_MODELS) -> Tuple[pd.DataFrame, dict]:
    """Per-gauge frame on the common gauge set: one SS column per model + ``Base NSE`` + attributes.

    Window scores are the per-gauge mean over ``leads`` (skill score and Baseline NSE alike).
    Returns ``(frame, model_labels)``; models absent from the run are dropped.
    """
    leads = [int(l) for l in leads]
    _, harm, labels = session.get_reference_skill("SS", list(models))
    wide_nse, _, _ = session.get_reference_skill("NSE", list(models), include_baseline=True, include_persistence=False)
    common = next(iter(harm.values())).index
    out = pd.DataFrame(index=common)
    for m, w in harm.items():
        out[m] = w.reindex(columns=leads).mean(axis=1)
    if "Baseline" in wide_nse:
        out["Base NSE"] = wide_nse["Baseline"].reindex(index=common, columns=leads).mean(axis=1)
    meta = getattr(session, "df_meta", pd.DataFrame())
    if meta is not None and not meta.empty:
        meta = meta.drop_duplicates("Basin ID").set_index("Basin ID")
        for c in ("unep_aridity_index", "area", "frac_snow", "p_mean", "slp_dg_sav"):
            if c in meta:
                out[c] = meta[c].reindex(common).astype(float)
        if "slp_dg_sav" in out:
            out["slope_deg"] = out["slp_dg_sav"] / 10.0  # HydroATLAS: tenths of a degree
    out = out.dropna(subset=[m for m in harm])
    out.index.name = "Basin ID"
    return out, {m: labels.get(m, m) for m in harm}


def _range_text(lo: float, hi: float, d: Dimension) -> str:
    if np.isinf(lo):
        return f"<{d.fmt(hi)}{d.unit}"
    if np.isinf(hi):
        return f"≥{d.fmt(lo)}{d.unit}"
    return f"{d.fmt(lo)}–{d.fmt(hi)}{d.unit}"


def assign_bins(values: pd.Series, d: Dimension, min_n: int = 20) -> pd.Series:
    """Ordered categorical of bin labels; bins with < ``min_n`` gauges are merged into their smaller neighbour."""
    edges, names = list(d.edges), [[n] for n in d.names]
    v = values.astype(float)
    while len(names) > 1:
        counts = pd.cut(v, bins=edges, right=False).value_counts(sort=False).to_numpy()
        small = [i for i, c in enumerate(counts) if c < min_n]
        if not small:
            break
        i = min(small, key=lambda k: counts[k])
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(names)]
        j = min(nbrs, key=lambda k: counts[k])
        a, b = sorted((i, j))
        names[a] = names[a] + names[b]
        del names[b]
        del edges[b]  # edge between bins a and b
    labels = [f"{' / '.join(n)} ({_range_text(edges[k], edges[k + 1], d)})" for k, n in enumerate(names)]
    cat = pd.cut(v, bins=edges, labels=labels, right=False)
    return cat


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def _kruskal_p(groups: List[np.ndarray]) -> float:
    groups = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() >= 2]
    if len(groups) < 2:
        return np.nan
    try:
        from scipy.stats import kruskal
        return float(kruskal(*groups).pvalue)
    except Exception:  # pylint: disable=broad-except
        return np.nan


def characteristic_table(df: pd.DataFrame, dim, models: Sequence[str], min_n: int = 20,
                         harm: float = HARM_THRESHOLD) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Rows = bins. Columns: N, median Baseline NSE, then per model median SS [IQR], % improved, % harmed.

    Returns ``(table, {model: Kruskal-Wallis p across bins})``. Column index is a 2-level MultiIndex
    (model / statistic) so the styler can group headers.
    """
    d = resolve_dimension(dim) if not isinstance(dim, Dimension) else dim
    if d.column not in df or df[d.column].notna().sum() == 0:
        return pd.DataFrame(), {}
    sub = df.dropna(subset=[d.column]).copy()
    sub["_bin"] = assign_bins(sub[d.column], d, min_n=min_n)
    g = sub.groupby("_bin", observed=True)
    cols = {("", "N"): g.size()}
    if "Base NSE" in sub:
        cols[("", "Median Baseline NSE")] = g["Base NSE"].median()
    pvals = {}
    for m in models:
        if m not in sub:
            continue
        cols[(m, "Median")] = g[m].median()
        cols[(m, "Q25")] = g[m].quantile(0.25)
        cols[(m, "Q75")] = g[m].quantile(0.75)
        cols[(m, "% improved")] = g[m].apply(lambda s: (s > harm).mean() * 100)
        cols[(m, "% harmed")] = g[m].apply(lambda s: (s < -harm).mean() * 100)
        pvals[m] = _kruskal_p([grp[m].to_numpy() for _, grp in g])
    tbl = pd.DataFrame(cols)
    tbl.columns = pd.MultiIndex.from_tuples(tbl.columns)
    tbl.index = tbl.index.astype(str)
    tbl.index.name = d.label
    return tbl, pvals


def overview_table(df: pd.DataFrame, models: Sequence[str], min_n: int = 20) -> pd.DataFrame:
    """One row per characteristic: per model the most / least helped bin, the gap between them and p."""
    rows = []
    for d in DIMENSIONS.values():
        tbl, pvals = characteristic_table(df, d, models, min_n=min_n)
        if tbl.empty:
            continue
        row = {("", "Characteristic"): d.label, ("", "Bins"): len(tbl), ("", "N"): int(tbl[("", "N")].sum())}
        for m in models:
            if (m, "Median") not in tbl:
                continue
            med = tbl[(m, "Median")]
            row[(m, "Most helped")] = f"{med.idxmax().split(' (')[0]} ({units.fmt_ss(med.max())})"
            row[(m, "Least helped")] = f"{med.idxmin().split(' (')[0]} ({units.fmt_ss(med.min())})"
            row[(m, "Gap")] = med.max() - med.min()
            row[(m, "p")] = pvals.get(m, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out.set_index(("", "Characteristic")).rename_axis("Characteristic")


def _p_text(p: float) -> str:
    if p != p:
        return "–"
    stars = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
    return f"{p:.1e} ({stars})" if p < 1e-3 else f"{p:.3f} ({stars})"


_HEADER_STYLE = [
    {"selector": "th", "props": [("background-color", "#e8eaed"), ("color", "#202124"), ("font-weight", "bold"),
                                 ("text-align", "center"), ("padding", "4px 10px")]},
    {"selector": "td", "props": [("padding", "4px 10px"), ("text-align", "right")]},
]


def style_characteristic_table(tbl: pd.DataFrame, pvals: Optional[Dict[str, float]] = None):
    """Compact styled table: median SS with [Q25, Q75] in one cell, % improved / harmed, p in the caption."""
    if tbl.empty:
        return tbl
    show = pd.DataFrame(index=tbl.index)
    show[("", "N")] = tbl[("", "N")].map("{:,d}".format)
    if ("", "Median Baseline NSE") in tbl:
        show[("", "Baseline NSE")] = tbl[("", "Median Baseline NSE")].map("{:.2f}".format)
    models = [m for m in dict.fromkeys(c[0] for c in tbl.columns) if m]
    for m in models:
        show[(m, units.ss_label("Median SS [IQR]"))] = [
            f"{units.fmt_ss(a)} [{units.fmt_ss(b, sign=False)}, {units.fmt_ss(c, sign=False)}]"
            for a, b, c in zip(tbl[(m, "Median")], tbl[(m, "Q25")], tbl[(m, "Q75")])]
        show[(m, "% improved")] = tbl[(m, "% improved")].map("{:.0f}%".format)
        show[(m, "% harmed")] = tbl[(m, "% harmed")].map("{:.0f}%".format)
    show.columns = pd.MultiIndex.from_tuples(show.columns)
    sty = show.style.set_table_styles(_HEADER_STYLE).format_index(escape="html", axis=0)
    for m in models:
        c = (m, "% harmed")
        worst = tbl[(m, "% harmed")]
        sty = sty.apply(lambda col, w=worst: ["color: #c5221f; font-weight: bold;" if v == w.max() and w.max() > 0
                                              else "" for v in w], subset=[c])
    if pvals:
        cap = "Kruskal–Wallis p (bins differ): " + "; ".join(f"{m}: {_p_text(p)}" for m, p in pvals.items())
        sty = sty.set_caption(cap).set_table_styles(
            _HEADER_STYLE + [{"selector": "caption", "props": [("caption-side", "bottom"), ("color", "#5f6368"),
                                                               ("font-size", "0.9em"), ("text-align", "left")]}])
    return sty


def style_overview_table(tbl: pd.DataFrame):
    if tbl.empty:
        return tbl
    fmt = {c: units.ss_formatter() for c in tbl.columns if c[1] == "Gap"}
    fmt.update({c: _p_text for c in tbl.columns if c[1] == "p"})
    fmt.update({c: "{:,d}" for c in tbl.columns if c[1] in ("N", "Bins")})
    sty = tbl.style.format(fmt, na_rep="–", escape="html").set_table_styles(_HEADER_STYLE)
    gaps = [c for c in tbl.columns if c[1] == "Gap"]
    if gaps:
        sty = sty.background_gradient(cmap="Purples", subset=gaps, vmin=0)
    return sty


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def plot_characteristic(df: pd.DataFrame, dim, models: Sequence[str], min_n: int = 20, ax=None,
                        whis=(10, 90), title: Optional[str] = None, legend: bool = True):
    """Grouped box plots of per-gauge skill score per bin (one colour per model, whiskers = Q10–Q90)."""
    d = resolve_dimension(dim) if not isinstance(dim, Dimension) else dim
    sub = df.dropna(subset=[d.column]).copy()
    sub["_bin"] = assign_bins(sub[d.column], d, min_n=min_n)
    bins = [b for b in sub["_bin"].cat.categories if (sub["_bin"] == b).any()]
    models = [m for m in models if m in sub]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(bins) + 2), 4.6), layout="constrained")
    width = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        data = [sub.loc[sub["_bin"] == b, m].dropna().to_numpy() for b in bins]
        pos = np.arange(len(bins)) + (k - (len(models) - 1) / 2) * width
        col = palette.color(palette.key_for(m) or "global_da")
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, whis=whis, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.6))
        for patch in bp["boxes"]:
            patch.set(facecolor=col, alpha=0.75, edgecolor="black", lw=0.8)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color="black", lw=0.8)
        ax.plot([], [], marker="s", ls="", ms=10, color=col, alpha=0.75, label=m)
    ax.axhline(0, color="#5f6368", lw=1, ls="--", zorder=0)
    ns = sub["_bin"].value_counts()
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{b.split(' (')[0]}\n({b.split(' (', 1)[1]}\nN = {ns[b]:,}" for b in bins], fontsize=10)
    ax.set_ylabel(units.ss_label("NSE skill score"), fontsize=11)
    units.ss_axis(ax, "y")
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.set_title(title or d.label, fontsize=12, fontweight="bold")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(models), frameon=False, fontsize=10)
    return ax.figure


def plot_characteristics_grid(df: pd.DataFrame, models: Sequence[str], min_n: int = 20, ncols: int = 3,
                              suptitle: Optional[str] = None, figsize=None):
    """All characteristics as small multiples (static / paper version) with one shared legend."""
    dims = [d for d in DIMENSIONS.values() if d.column in df and df[d.column].notna().any()]
    nrows = int(np.ceil(len(dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6.0 * ncols, 4.6 * nrows),
                             layout="constrained", sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, d in zip(axes, dims):
        plot_characteristic(df, d, models, min_n=min_n, ax=ax, legend=False)
    for k, ax in enumerate(axes):
        if k >= len(dims):
            ax.set_visible(False)
        elif k % ncols:
            ax.set_ylabel("")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), frameon=False, fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig


def plot_characteristics_heatmap(df: pd.DataFrame, models: Sequence[str] = ("Global-Best DA", "Per-Basin DA"),
                                 color_by: Optional[str] = None, min_n: int = 20, stat: str = "median",
                                 title: Optional[str] = None, figsize=None, cmap: str = "RdBu", vmax=None):
    """Grid: rows = characteristics, columns = classes ordered low -> high; cell = 'Global / Per-Basin' score.

    The cell colour is ``color_by`` (default: first model) on a diverging scale centred on 0 (red = harm);
    each cell also names its class and gives N. ``stat``: 'median' or 'mean' of the per-gauge skill score.
    Rows with fewer classes leave the right-hand cells empty.
    """
    from matplotlib.colors import TwoSlopeNorm
    models = [m for m in models if m in df]
    color_by = color_by if color_by in models else models[0]
    rows = []
    for d in DIMENSIONS.values():
        tbl, _ = characteristic_table(df, d, models, min_n=min_n)
        if tbl.empty:
            continue
        sub = df.dropna(subset=[d.column]).copy()
        sub["_bin"] = assign_bins(sub[d.column], d, min_n=min_n)
        g = sub.groupby("_bin", observed=True)
        vals = {m: (g[m].median() if stat == "median" else g[m].mean()) for m in models}
        cells = [{"name": b, "n": int(tbl.loc[b, ("", "N")]), **{m: float(vals[m][b]) for m in models}}
                 for b in tbl.index]
        rows.append((d.label, cells))
    ncols = max(len(c) for _, c in rows)
    grid = np.full((len(rows), ncols), np.nan)
    for i, (_, cells) in enumerate(rows):
        for j, c in enumerate(cells):
            grid[i, j] = c[color_by]
    vm = vmax if vmax is not None else np.nanmax(np.abs(grid))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vm, vmax=vm)
    fig, ax = plt.subplots(figsize=figsize or (3.1 * ncols + 2.6, 1.05 * len(rows) + 1.6), layout="constrained")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    cm = plt.get_cmap(cmap)
    for i, (_, cells) in enumerate(rows):
        for j, c in enumerate(cells):
            rgba = cm(norm(c[color_by]))
            dark = (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]) < 0.5
            tc = "white" if dark else "#202124"
            name, rng = (c["name"].split(" (", 1) + [""])[:2]
            ax.text(j, i - 0.30, f"{name} ({rng}" if rng else name, ha="center", va="center", fontsize=8.5, color=tc)
            ax.text(j, i + 0.02, " / ".join(units.fmt_ss(c[m]) for m in models), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc)
            ax.text(j, i + 0.32, f"N = {c['n']:,}", ha="center", va="center", fontsize=8, color=tc, alpha=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, fontweight="bold")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(["Lowest"] + [""] * (ncols - 2) + ["Highest"] if ncols > 1 else [""], fontsize=10)
    ax.set_xlabel("Class (ordered low → high value of the characteristic)", fontsize=10)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, ncols), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label(units.ss_label(f"{stat.capitalize()} NSE skill score — {color_by}"), fontsize=10)
    units.ss_colorbar(cb)
    ax.set_title(title or f"{stat.capitalize()} NSE skill score per class: " + " / ".join(models),
                 fontsize=12, fontweight="bold")
    return fig
