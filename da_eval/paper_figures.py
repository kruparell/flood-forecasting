"""Publication figures for multi-lead Data Assimilation skill.

All figures compare a small set of *reference models* on a common gauge set (gauges with a valid
NSE skill score for every model at every lead):

* ``Global-Best DA``   - one DA config for all leads (highest mean-over-leads median skill score;
                         the same rule as the Top-N table, i.e. a deployable choice).
* ``Per-Basin DA``     - per gauge, best DA config at the reference lead (oracle).
* ``Global-Best PP``- one fixed-rho PP error-correction config, same ranking rule.
* ``Per-Basin PP``  - ``AR1_Per_Basin_Best_Rho`` (or per-gauge best fixed rho at the reference lead).
* ``Baseline``         - open-loop model (absolute metrics only; its skill score is 0 by definition).
* ``Persistence``      - naive forecast q(T+L) = q_obs(T), computed from observed flow (absolute NSE only).

Metrics: absolute ``DA NSE`` / ``DA KGE`` (shows the Baseline's own skill and what DA adds on top) or the
relative ``NSE Skill Score`` (improvement over Baseline). "% harmed" is always skill-score based.

Figures
-------
* ``plot_skill_decay``        - median + IQR per lead, with a "% gauges harmed" panel.
* ``plot_harm_benefit_bars``  - diverging stacked bars of harm/benefit categories per lead.
* ``plot_skill_cdf_panels``   - CDF small multiples at selected leads (per-panel x-range).
* ``plot_skill_violins``      - clipped violins at selected leads (best for 2-3 leads).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from collections.abc import Mapping

from da_eval import units
from da_eval import palette
from da_eval.ingestion import _ensure_skill_score_columns, detect_lead_times, is_per_basin_rho_config, is_rho_config
from da_eval.tables import _per_basin_best_records
from da_eval.selection import PER_BASIN_DA, get_active

SS_COL = "NSE Skill Score"
DEFAULT_MODELS = ("Global-Best DA", "Per-Basin DA", "Global-Best PP")
ABSOLUTE_MODELS = ("Persistence", "Baseline") + DEFAULT_MODELS
BASE_COL = {"DA NSE": "Base NSE", "DA KGE": "Base KGE"}
METRIC_LABEL = {"DA NSE": "NSE", "DA KGE": "KGE", SS_COL: "NSE skill score"}
class _LiveModelStyle(Mapping):
    """{label: dict(ls=...)} from the notebook-wide ``palette`` (read at draw time)."""

    def __getitem__(self, k):
        key = palette.key_for(k)
        if key is None:
            raise KeyError(k)
        return dict(ls=palette.ls(key))

    def __iter__(self):
        return iter(palette.by_label())

    def __len__(self):
        return len(palette.by_label())


# Label-keyed live views of the notebook-wide palette (``session.set_palette``).
MODEL_STYLE = _LiveModelStyle()
MODEL_COLORS = palette.by_label()
HARM_COLORS = ["#b2182b", "#ef8a62", "#e0e0e0", "#67a9cf", "#2166ac"]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def build_reference_skill(
    df_eval: pd.DataFrame,
    models: Sequence[str] = DEFAULT_MODELS,
    lead_times: Optional[Sequence[int]] = None,
    metric_col: str = SS_COL,
    obs_df: Optional[pd.DataFrame] = None,
    select_col: str = SS_COL,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Returns ``({model: wide (gauge x lead) frame of metric_col}, {model: description})`` on a common gauge set.

    Reference models follow the notebook-wide selection (``session.set_selection``), so they are identical
    whichever metric is plotted (without an active selection: ``select_col`` at t+1 / mean of lead medians). ``Baseline`` uses ``Base NSE`` / ``Base KGE``; ``Persistence`` needs ``obs_df``
    (Valid Date, Basin ID, q_obs) and supports NSE only.
    """
    df_eval = _ensure_skill_score_columns(df_eval)
    leads = list(lead_times) if lead_times is not None else detect_lead_times(df_eval)
    ref_lead = 1 if 1 in leads else leads[0]
    sel = get_active()
    ev = df_eval[df_eval["Lead Time (Days)"].isin(leads)]
    cfgs = [c for c in ev["Config ID"].dropna().unique() if "Per-Basin" not in c and "baseline" not in c.lower()]
    da = [c for c in cfgs if not is_rho_config(c)]
    rho_pb = [c for c in cfgs if is_per_basin_rho_config(c)]
    rho_fixed = [c for c in cfgs if is_rho_config(c) and c not in rho_pb]
    score = ev[ev["Config ID"].isin(da + rho_fixed)].groupby(["Config ID", "Lead Time (Days)"])[select_col].median()
    score = score.groupby(level=0).mean().dropna()

    records: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, str] = {}
    for m in models:
        if m == "Global-Best DA" and da:
            c = sel.global_da if (sel and sel.global_da in da) else score.reindex(da).idxmax()
            records[m], labels[m] = ev[ev["Config ID"] == c], c
        elif m == "Per-Basin DA" and da:
            if sel is not None and (ev["Config ID"] == PER_BASIN_DA).any():
                records[m], labels[m] = ev[ev["Config ID"] == PER_BASIN_DA], f"best DA per gauge ({sel.policy.label()})"
            else:
                records[m], labels[m] = _per_basin_best_records(ev, da, select_col, ref_lead), f"best DA per gauge at t+{ref_lead}"
        elif m == "Global-Best PP" and rho_fixed:
            c = sel.global_rho if (sel and sel.global_rho in rho_fixed) else score.reindex(rho_fixed).idxmax()
            records[m], labels[m] = ev[ev["Config ID"] == c], c
        elif m == "Per-Basin PP" and (rho_pb or rho_fixed):
            if rho_pb:
                records[m], labels[m] = ev[ev["Config ID"] == rho_pb[0]], rho_pb[0]
            else:
                records[m] = _per_basin_best_records(ev, rho_fixed, select_col, ref_lead)
                labels[m] = f"best fixed rho per gauge at t+{ref_lead}"
        elif m == "Baseline":
            base = ev[ev["Config ID"].isin(da or cfgs)].drop_duplicates(["Basin ID", "Lead Time (Days)"]).copy()
            base[metric_col] = base[BASE_COL[metric_col]] if metric_col in BASE_COL else 0.0
            records[m], labels[m] = base, "open-loop model"
        elif m == "Persistence":
            if obs_df is None or metric_col != "DA NSE":
                continue  # persistence only defined for absolute NSE with observations available
            records[m], labels[m] = persistence_nse(obs_df, leads), "q(T+L) = q_obs(T)"
        elif m not in MODEL_COLORS:
            raise ValueError(f"Unknown model {m!r}; choose from {list(MODEL_COLORS)}")

    wide = {k: v.pivot_table(index="Basin ID", columns="Lead Time (Days)", values=metric_col if k != "Persistence" else "DA NSE")
            for k, v in records.items()}
    wide = {k: w.reindex(columns=leads) for k, w in wide.items()}
    common = sorted(set.intersection(*[set(w.dropna().index) for w in wide.values()])) if wide else []
    return {k: w.loc[common] for k, w in wide.items()}, labels


def persistence_nse(obs_df: pd.DataFrame, lead_times: Sequence[int], min_pairs: int = 30) -> pd.DataFrame:
    """Per-gauge NSE of the naive persistence forecast q(T+L) = q_obs(T), on the valid-date axis.

    ``obs_df``: Valid Date, Basin ID, q_obs (one row per gauge-day, any single lead). Returns long records with
    ``DA NSE`` so it plugs into ``build_reference_skill``.
    """
    o = obs_df[["Basin ID", "Valid Date", "q_obs"]].dropna(subset=["Valid Date"]).drop_duplicates(["Basin ID", "Valid Date"])
    wide = o.pivot(index="Valid Date", columns="Basin ID", values="q_obs").sort_index()
    wide = wide.asfreq("D")
    obs = wide.to_numpy(dtype=float)
    rows = []
    for lt in lead_times:
        pred = np.full_like(obs, np.nan)
        pred[lt:] = obs[:-lt]
        m = np.isfinite(obs) & np.isfinite(pred)
        cnt = m.sum(0)
        o_m = np.where(m, obs, np.nan)
        mean = np.nanmean(o_m, axis=0)
        sse = np.nansum(np.where(m, (pred - obs) ** 2, np.nan), axis=0)
        sst = np.nansum((o_m - mean) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            nse = np.where((cnt >= min_pairs) & (sst > 0), 1 - sse / sst, np.nan)
        rows.append(pd.DataFrame({"Basin ID": wide.columns.astype(str), "Lead Time (Days)": lt, "DA NSE": nse}))
    return pd.concat(rows, ignore_index=True)


def _n(wide: Dict[str, pd.DataFrame]) -> int:
    return len(next(iter(wide.values()))) if wide else 0


def _thr(t: float) -> str:
    """Compact skill-score threshold text (0.01 -> '1%' in percent mode, '0.01' otherwise)."""
    return f"{t * 100:g}%" if units.ss_percent() else f"{t:g}"


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_skill_decay(
    wide: Dict[str, pd.DataFrame],
    harm_thresholds: Sequence[float] = (0.01, 0.05),
    show_harm_panel: bool = True,
    figsize: Tuple[float, float] = (6.5, 6.0),
    title: Optional[str] = None,
    metric_label: str = "NSE skill score",
    harm_wide: Optional[Dict[str, pd.DataFrame]] = None,
) -> plt.Figure:
    """Median + IQR (offset error bars) per lead; optional lower panel with % gauges harmed (SS < -thr).

    ``wide`` holds the plotted metric (absolute NSE/KGE or skill score). ``harm_wide`` holds skill scores
    for the harm panel (defaults to ``wide``, i.e. when ``wide`` already is a skill score). Models absent
    from ``harm_wide`` (Baseline, Persistence) get no harm line.
    """
    harm_wide = harm_wide if harm_wide is not None else wide
    leads = list(next(iter(wide.values())).columns)
    if show_harm_panel:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    else:
        fig, a1 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.65))
        a2 = None
    k_n = len(wide)
    for i, (k, w) in enumerate(wide.items()):
        col = MODEL_COLORS.get(k, None)
        x = np.array(leads) + (i - (k_n - 1) / 2) * 0.12
        med, q1, q3 = w.median(), w.quantile(0.25), w.quantile(0.75)
        eb = a1.errorbar(x, med, yerr=[med - q1, q3 - med], fmt="o", color=col, ms=5, capsize=3, lw=1.4, label=k)
        a1.plot(x, med, color=col, lw=1.8, **MODEL_STYLE.get(k, dict(ls="-")))
        if a2 is not None and k in harm_wide and k not in ("Baseline", "Persistence"):
            hw = harm_wide[k]
            for thr, ls in zip(harm_thresholds, ["-", "--", ":"]):
                a2.plot(leads, 100 * (hw < -abs(thr)).mean(), marker="o", ls=ls, color=col, ms=4 if ls == "-" else 3,
                        lw=1.6 if ls == "-" else 1.0, alpha=1 if ls == "-" else 0.75)
    is_skill = "skill" in metric_label.lower()
    if is_skill:
        a1.axhline(0, color="k", lw=0.8, ls="--")
        units.ss_axis(a1, "y")
    a1.set_ylabel(f"{units.ss_label(metric_label) if is_skill else metric_label}\n(median, IQR)")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title(title or f"Skill decay with lead time (N = {_n(wide):,} gauges)", fontsize=11, fontweight="bold")
    a1.grid(alpha=0.3, ls=":")
    last = a1
    if a2 is not None:
        a2.set_ylabel("% gauges harmed\n(vs Baseline)")
        a2.legend(handles=[Line2D([], [], color="gray", ls=ls, label=f"SS < −{_thr(abs(t))}")
                           for t, ls in zip(harm_thresholds, ["-", "--", ":"])], frameon=False, fontsize=8)
        a2.grid(alpha=0.3, ls=":")
        last = a2
    last.set_xlabel("Lead time (days)")
    last.set_xticks(leads)
    fig.tight_layout()
    return fig


def plot_harm_benefit_bars(
    wide: Dict[str, pd.DataFrame],
    bin_thresholds: Tuple[float, float] = (0.01, 0.05),
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Share of gauges harmed / improved by lead time",
) -> plt.Figure:
    """One panel per model; per lead a diverging bar of 5 skill-score categories (neutral centred on 0)."""
    t1, t2 = sorted(abs(t) for t in bin_thresholds)
    edges = [-np.inf, -t2, -t1, t1, t2, np.inf]
    s1, s2 = _thr(t1), _thr(t2)
    labels = [f"SS < −{s2}", f"−{s2}…−{s1}", f"±{s1} (neutral)", f"{s1}…{s2}", f"SS > {s2}"]
    leads = list(next(iter(wide.values())).columns)
    n = _n(wide)
    fig, axes = plt.subplots(1, len(wide), figsize=figsize or (4 * len(wide), 0.4 * len(leads) + 1.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (k, w) in zip(axes, wide.items()):
        for j, lt in enumerate(leads):
            frac = 100 * np.histogram(w[lt].values, bins=edges)[0] / max(n, 1)
            harmed, improved, neutral = frac[0] + frac[1], frac[3] + frac[4], frac[2]
            left = -(harmed + neutral / 2)
            for c in range(5):
                ax.barh(j, frac[c], left=left, color=HARM_COLORS[c], edgecolor="white", lw=0.5, height=0.75)
                left += frac[c]
            ax.text(-(harmed + neutral / 2) - 1, j, f"{harmed:.0f}%", ha="right", va="center", fontsize=8, color=HARM_COLORS[0])
            ax.text(improved + neutral / 2 + 1, j, f"{improved:.0f}%", ha="left", va="center", fontsize=8, color=HARM_COLORS[4])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(k, fontsize=10.5, fontweight="bold", color=MODEL_COLORS.get(k, "k"))
        ax.set_xlim(-60, 110)
        ax.set_xticks([-50, 0, 50, 100])
        ax.set_xticklabels(["50", "0", "50", "100"])
        ax.set_xlabel("% of gauges")
    axes[0].set_yticks(range(len(leads)))
    axes[0].set_yticklabels([f"t+{lt}" for lt in leads])
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(HARM_COLORS, labels)], loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle(f"{title} (N = {n:,} gauges)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    return fig


def _auto_xlim(values: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> Tuple[float, float]:
    lo, hi = np.nanquantile(values, [lo_q, hi_q])
    lo, hi = min(lo, -0.02), max(hi, 0.02)
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_skill_cdf_panels(
    wide: Dict[str, pd.DataFrame],
    leads: Sequence[int] = (1, 3, 7),
    harm_threshold: float = 0.01,
    xlim: Optional[Tuple[float, float]] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    metric_label: str = "NSE skill score",
    harm_wide: Optional[Dict[str, pd.DataFrame]] = None,
) -> plt.Figure:
    """ECDF small multiples at ``leads``; each panel gets its own x-range unless ``xlim`` is given.

    For a skill score the harm region (SS < -thr) is shaded. For absolute NSE/KGE the x-range is clipped at
    the lower 5% (long negative tails) and "% harmed" in the legend comes from ``harm_wide`` when given.
    """
    is_skill = "skill" in metric_label.lower()
    leads = [lt for lt in leads if lt in next(iter(wide.values())).columns]
    ncols = ncols or len(leads)
    nrows = int(np.ceil(len(leads) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4 * ncols, 3.4 * nrows + 0.6), sharey=True, squeeze=False)
    flat = axes.ravel()
    shared_abs = None
    if not is_skill and xlim is None:
        # Absolute metrics: one shared x-range across panels (comparable), lower 5% of the forecast models
        # (Persistence excluded so its long negative tail does not squash the axis).
        ref = [w[leads].values.ravel() for k, w in wide.items() if k != "Persistence"] or [w[leads].values.ravel() for w in wide.values()]
        shared_abs = (max(np.nanquantile(np.concatenate(ref), 0.05), -1.0) - 0.05, 1.0)
    for ax, lt in zip(flat, leads):
        pooled = np.concatenate([w[lt].values for k, w in wide.items() if k != "Persistence"] or [w[lt].values for w in wide.values()])
        xl = xlim or (_auto_xlim(pooled) if is_skill else shared_abs)
        for k, w in wide.items():
            v = np.sort(w[lt].values)
            st = MODEL_STYLE.get(k, dict(ls="-"))
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=MODEL_COLORS.get(k), lw=2, **st)
            lbl = f"med {units.fmt_ss(np.median(v))}" if is_skill else f"med {np.median(v):.2f}"
            hw = (wide if is_skill else (harm_wide or {})).get(k)
            if hw is not None and k not in ("Baseline", "Persistence"):
                lbl += f" | harmed {100 * (hw[lt].values < -abs(harm_threshold)).mean():.0f}%"
            ax.plot([], [], color=MODEL_COLORS.get(k), lw=2, label=lbl, **st)
        if is_skill:
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.axvspan(xl[0], -abs(harm_threshold), color=HARM_COLORS[0], alpha=0.06, lw=0)
        ax.set_xlim(*xl)
        ax.set_title(f"Lead t+{lt}", fontsize=11, fontweight="bold")
        ax.set_xlabel(units.ss_label(metric_label) if is_skill else metric_label)
        if is_skill:
            units.ss_axis(ax, "x")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.2)
    for ax in flat[len(leads):]:
        ax.axis("off")
    for r in range(nrows):
        axes[r, 0].set_ylabel("Cumulative fraction of gauges")
    fig.legend(handles=[Line2D([], [], color=MODEL_COLORS.get(k), lw=2.5, label=k, **MODEL_STYLE.get(k, dict(ls="-")))
                        for k in wide], loc="lower center", ncol=len(wide), frameon=False, fontsize=9.5)
    sub = f"; shaded: SS < −{_thr(abs(harm_threshold))}" if is_skill else ""
    fig.suptitle(f"Per-gauge {metric_label} CDFs (N = {_n(wide):,} gauges{sub})", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def plot_skill_violins(
    wide: Dict[str, pd.DataFrame],
    leads: Sequence[int] = (1, 3, 7),
    clip: Tuple[float, float] = (-0.5, 1.0),
    figsize: Optional[Tuple[float, float]] = None,
    metric_label: str = "NSE skill score",
) -> plt.Figure:
    """Clipped violins (with IQR bar and median) per lead and model; share of gauges below the clip is annotated."""
    leads = [lt for lt in leads if lt in next(iter(wide.values())).columns]
    lo, hi = clip
    k_n = len(wide)
    width = 0.8 / k_n
    fig, ax = plt.subplots(figsize=figsize or (max(5.0, 1.9 * len(leads) * k_n / 3 + 2), 4.2))
    xs = np.arange(len(leads))
    for i, (k, w) in enumerate(wide.items()):
        col = MODEL_COLORS.get(k)
        pos = xs + (i - (k_n - 1) / 2) * width
        data = [np.clip(w[lt].values, lo, hi) for lt in leads]
        vp = ax.violinplot(data, positions=pos, widths=width * 0.95, showextrema=False, bw_method=0.15)
        for b in vp["bodies"]:
            b.set_facecolor(col)
            b.set_edgecolor(col)
            b.set_alpha(0.35)
        ax.vlines(pos, [np.percentile(d, 25) for d in data], [np.percentile(d, 75) for d in data], color=col, lw=3)
        ax.scatter(pos, [np.median(d) for d in data], color="white", edgecolor=col, zorder=3, s=20)
        for p, lt in zip(pos, leads):
            below = 100 * (w[lt] < lo).mean()
            if below >= 0.5:
                ax.text(p, lo - 0.03, f"{below:.0f}%", ha="center", va="top", fontsize=7, color=col)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(lo - 0.1, hi)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"t+{lt}" for lt in leads])
    is_skill = "skill" in metric_label.lower()
    if is_skill:
        units.ss_axis(ax, "y")
    ax.set_ylabel(units.ss_label(metric_label) if is_skill else metric_label)
    clip_txt = f"[{_thr(lo)}, {_thr(hi)}]" if is_skill else f"[{lo:g}, {hi:g}]"
    ax.set_title(f"Per-gauge {metric_label} distribution (N = {_n(wide):,}; clipped to {clip_txt}, % below annotated)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(handles=[Patch(color=MODEL_COLORS.get(k), alpha=0.5, label=k) for k in wide], frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    return fig
