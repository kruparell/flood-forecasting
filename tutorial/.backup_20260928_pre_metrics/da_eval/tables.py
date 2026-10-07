"""Table generation and styling: Top-N Benchmark Matrix and Consolidated Physical Stratification Matrix."""

import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from da_eval import units
from da_eval.ingestion import (
    _ensure_skill_score_columns,
    detect_lead_times,
    is_per_basin_rho_config,
    is_rho_config,
    parse_da_config_id,
)
from da_eval.benchmarks import get_global_best_config
from da_eval.selection import get_active


_is_rho_config = is_rho_config
_is_per_basin_rho_config = is_per_basin_rho_config


def _ss_threshold_header(col: str) -> str:
    """Display-only header for SS thresholds: '% SS<-0.01 (t+1)' -> '% SS<-1% (t+1)', '(Q0.05 > -0.10)' ->
    '(Q0.05 > -10%)' in percent mode. The underlying column name and computation stay in native units."""
    if not units.ss_percent():
        return col
    return re.sub(r"(SS\s*[<>]\s*|Q0\.05\s*>\s*|\(<\s*)(-?)(\d*\.\d+|\d+)(?!\d*%)",
                  lambda m: f"{m.group(1)}{m.group(2)}{float(m.group(3)) * 100:g}%", col)


def _relabel_columns(styler: Any, mapping: Dict[Any, str]) -> Any:
    """Display-only column-header renames (data column names are unchanged)."""
    mapping = {k: v for k, v in mapping.items() if v != k}
    if not mapping:
        return styler
    return styler.format_index(lambda c: mapping.get(c, c), axis=1)


def _per_basin_best_records(df: pd.DataFrame, candidates: List[str], metric_col: str, ref_lead: int) -> pd.DataFrame:
    """Per basin, picks the candidate config with the best ``metric_col`` at ``ref_lead`` and returns
    that config's records across all leads (same selection rule as ``build_per_basin_optimal_eval``).

    With an active notebook-wide selection its per-basin choice is used (restricted to ``candidates``)."""
    sub = df[df["Config ID"].isin(candidates)]
    sel = get_active()
    if sel is not None:
        cand = set(candidates)
        pool = sel.per_basin_rho if all(is_rho_config(c) for c in cand) else sel.per_basin_da
        mapping = {b_: c for b_, c in pool.items() if c in cand}
        if mapping:
            best = pd.DataFrame({"Basin ID": list(mapping), "Config ID": list(mapping.values())})
            return sub.assign(_b=sub["Basin ID"].astype(str)).merge(
                best.rename(columns={"Basin ID": "_b"}), on=["_b", "Config ID"]).drop(columns="_b")
    ref = sub[sub["Lead Time (Days)"] == ref_lead].dropna(subset=[metric_col])
    if ref.empty:
        return sub.iloc[0:0]
    best = ref.sort_values(metric_col, ascending=False).drop_duplicates(subset=["Basin ID"])[["Basin ID", "Config ID"]]
    return sub.merge(best, on=["Basin ID", "Config ID"])


def build_top_n_benchmark_matrix(
    df_eval: pd.DataFrame,
    top_n: int = 10,
    lead_times: Optional[List[int]] = None,
    metric_col: str = "DA NSE",
    delta_col: str = "NSE Skill Score",
    lower_quantile: Optional[float] = None,
    harm_thresholds: Optional[Sequence[float]] = (0.01,),
) -> pd.DataFrame:
    """Constructs multi-leadtime benchmark matrix table across horizons (supporting dynamic t+0).

    ``lower_quantile`` (default 0.05) adds, per lead, the across-basin lower-tail skill score
    ``SS_NSE Q0.05 (t+L)`` plus its mean over leads: the downside-risk counterpart of the median
    (higher is better; negative means the worst 5% of basins are degraded). ``None`` (default) disables it.

    ``harm_thresholds`` (default ``(0.01,)``) adds, per lead, ``% SS<-0.01 (t+L)``: the share of basins
    (with a valid skill score at that lead) where the config does harm, i.e. skill score below ``-thr``,
    plus its mean over leads. Lower is better. Pass e.g. ``(0.01, 0.05)`` for several thresholds or ``None``.

    Row order:
      1. Baseline (Open-Loop)
      2. Global-Best DA   - the notebook-wide selection's Global-Best DA (fallback: highest mean-across-leads
                            median skill score when no selection is active)
      3. Global-Best Rho  - the selection's Global-Best AR(1) (same fallback)
      4. Per-Basin DA     - per basin, the selection's best *DA* config (fallback: best at Lead 1)
      5. Per-Basin Rho    - ``AR1_Per_Basin_Best_Rho`` if present, else per-basin best fixed rho at Lead 1
      6+. Top-N individual configs (DA and fixed-rho AR1), ranked by the mean over leads of the
          per-lead median skill score (column ``SS_NSE (mean t+a..t+b)``).
    Columns: Hyperparameters | Median Metric per Leadtime | Median Skill Score per Leadtime | ranking score.
    """
    if df_eval.empty:
        return pd.DataFrame()

    df_eval = _ensure_skill_score_columns(df_eval)
    if delta_col not in df_eval.columns:
        delta_col = "NSE Delta"

    is_skill = "skill" in delta_col.lower()
    col_prefix = "SS_NSE" if is_skill else "ΔNSE"

    if lead_times is None:
        lead_times = detect_lead_times(df_eval)
    lead_times = list(lead_times)
    ref_lead = 1 if 1 in lead_times else lead_times[0]
    score_col = f"{col_prefix} (mean t+{lead_times[0]}..t+{lead_times[-1]})"
    q_tag = f"Q{lower_quantile:g}" if lower_quantile is not None else None
    q_score_col = f"{col_prefix} {q_tag} (mean t+{lead_times[0]}..t+{lead_times[-1]})" if q_tag else None
    harm_thresholds = [abs(float(t)) for t in (harm_thresholds or [])]
    harm_sym = "SS" if is_skill else "ΔNSE"
    harm_tags = [f"% {harm_sym}<-{t:g}" for t in harm_thresholds]

    df_leads = df_eval[df_eval["Lead Time (Days)"].isin(lead_times)]
    cfg_ids = pd.Series(df_leads["Config ID"].dropna().unique()).astype(str)
    synthetic = cfg_ids.str.contains("Per-Basin", na=False) | cfg_ids.str.contains("Baseline", case=False, na=False)
    real_cfgs = cfg_ids[~synthetic].tolist()
    per_basin_rho_cfgs = [c for c in real_cfgs if _is_per_basin_rho_config(c)]
    fixed_rho_cfgs = [c for c in real_cfgs if _is_rho_config(c) and c not in per_basin_rho_cfgs]
    da_cfgs = [c for c in real_cfgs if not _is_rho_config(c)]

    def _row(label: str, sub: pd.DataFrame, hp_cfg: Optional[str], target: Optional[str] = None) -> dict:
        row = {"Configuration": label}
        if hp_cfg is not None and target is None:
            hp = parse_da_config_id(hp_cfg)
            row.update({"Window": hp["Window (Days)"], "LR": hp["Learning Rate"], "Epochs": hp["Epochs"],
                        "BG Reg": hp["BG Weight"], "Target": hp["Target"],
                        "Loss": hp["Loss"] if pd.notna(hp.get("Loss", np.nan)) else "-"})
        else:
            row.update({"Window": "-", "LR": "-", "Epochs": "-", "BG Reg": "-", "Target": target or "-", "Loss": "-"})
        per_lead, per_lead_q = [], []
        per_lead_harm = {h: [] for h in harm_tags}
        for lt in lead_times:
            lt_sub = sub[sub["Lead Time (Days)"] == lt]
            tag = f"t+{lt}"
            if lt_sub.empty:
                row[f"NSE ({tag})"] = np.nan
                row[f"{col_prefix} ({tag})"] = np.nan
                if q_tag:
                    row[f"{col_prefix} {q_tag} ({tag})"] = np.nan
                for h in harm_tags:
                    row[f"{h} ({tag})"] = np.nan
            else:
                row[f"NSE ({tag})"] = lt_sub[metric_col].median()
                row[f"{col_prefix} ({tag})"] = lt_sub[delta_col].median()
                per_lead.append(row[f"{col_prefix} ({tag})"])
                if q_tag:
                    row[f"{col_prefix} {q_tag} ({tag})"] = lt_sub[delta_col].quantile(lower_quantile)
                    per_lead_q.append(row[f"{col_prefix} {q_tag} ({tag})"])
                valid = lt_sub[delta_col].dropna()
                for h, thr in zip(harm_tags, harm_thresholds):
                    pct = 100.0 * float((valid < -thr).mean()) if len(valid) else np.nan
                    row[f"{h} ({tag})"] = pct
                    per_lead_harm[h].append(pct)
        row[score_col] = float(np.nanmean(per_lead)) if per_lead else np.nan
        if q_score_col:
            row[q_score_col] = float(np.nanmean(per_lead_q)) if per_lead_q else np.nan
        for h in harm_tags:
            vals = per_lead_harm[h]
            row[f"{h} (mean t+{lead_times[0]}..t+{lead_times[-1]})"] = float(np.nanmean(vals)) if vals else np.nan
        return row

    # Rank individual configs by the mean over leads of the per-lead median skill score.
    per_lead_median = df_leads[df_leads["Config ID"].isin(da_cfgs + fixed_rho_cfgs)].groupby(
        ["Config ID", "Lead Time (Days)"])[delta_col].median()
    cfg_score = per_lead_median.groupby(level="Config ID").mean().dropna().sort_values(ascending=False)
    da_ranked = [c for c in cfg_score.index if c in da_cfgs]
    rho_ranked = [c for c in cfg_score.index if c in fixed_rho_cfgs]

    # Global-Best rows follow the notebook-wide selection when one is active (the Top-N rows below keep the
    # table's own ranking by mean-over-leads median skill).
    sel = get_active()
    g_da = sel.global_da if sel is not None and sel.global_da in da_cfgs else (da_ranked[0] if da_ranked else None)
    g_rho = sel.global_rho if sel is not None and sel.global_rho in fixed_rho_cfgs else (rho_ranked[0] if rho_ranked else None)

    rows = []
    # 1. Baseline
    if "Base NSE" in df_leads.columns and not df_leads.empty:
        base_ref = df_leads[df_leads["Config ID"].isin(da_cfgs)] if da_cfgs else df_leads
        base = base_ref.drop_duplicates(subset=["Basin ID", "Lead Time (Days)"]).copy()
        base[metric_col] = base["Base NSE"]
        base[delta_col] = 0.0
        rows.append(_row("Baseline (Open-Loop)", base, None, target="Baseline"))
    # 2. Global-Best DA
    if g_da:
        rows.append(_row(f"Global-Best DA ({g_da})", df_leads[df_leads["Config ID"] == g_da], g_da))
    # 3. Global-Best Rho
    if g_rho:
        rows.append(_row(f"Global-Best Rho ({g_rho})", df_leads[df_leads["Config ID"] == g_rho], None,
                         target="AR1 Post-process"))
    # 4. Per-Basin DA (DA configs only)
    if da_cfgs:
        rows.append(_row("Per-Basin DA", _per_basin_best_records(df_leads, da_cfgs, delta_col, ref_lead), None,
                         target="Tuned Per Basin"))
    # 5. Per-Basin Rho
    if per_basin_rho_cfgs:
        rows.append(_row(f"Per-Basin Rho ({per_basin_rho_cfgs[0]})",
                         df_leads[df_leads["Config ID"] == per_basin_rho_cfgs[0]], None, target="AR1 Per-Basin Rho"))
    elif fixed_rho_cfgs:
        rows.append(_row("Per-Basin Rho", _per_basin_best_records(df_leads, fixed_rho_cfgs, delta_col, ref_lead), None,
                         target="AR1 Per-Basin Rho"))
    # 6+. Individual configs
    for cfg in cfg_score.index[:top_n]:
        is_rho = cfg in fixed_rho_cfgs
        rows.append(_row(cfg, df_leads[df_leads["Config ID"] == cfg], None if is_rho else cfg,
                         target="AR1 Post-process" if is_rho else None))

    df_matrix = pd.DataFrame(rows)
    ordered = ["Configuration", "Window", "LR", "Epochs", "BG Reg", "Loss", "Target"]
    df_matrix = df_matrix[[c for c in ordered if c in df_matrix.columns] + [c for c in df_matrix.columns if c not in ordered]]
    return df_matrix


DEFAULT_HORIZON_WINDOWS = {
    "Short-Range (Days 1–3)": (1, 2, 3),
    "Medium-Range (Days 4–7)": (4, 5, 6, 7),
    "Full Horizon (Days 1–7)": (1, 2, 3, 4, 5, 6, 7),
}
SS_HTML = "SS<sub>NSE</sub>"
DEFAULT_HORIZON_LABELS = {
    "baseline": "Baseline Open-Loop (Median NSE)",
    "global_da": f"Global Default DA ({SS_HTML})",
    "global_rho": f"Global Post-Processing ({SS_HTML})",
    "per_basin_da": f"Per-Basin Best DA ({SS_HTML})",
    "per_basin_rho": f"Per-Basin Post-Processing ({SS_HTML})",
}


def build_horizon_summary_table(
    df_eval: pd.DataFrame,
    windows: Optional[dict] = None,
    stat: str = "median",
    metric_col: str = "NSE Skill Score",
    labels: Optional[dict] = None,
    common_basins: bool = True,
) -> pd.DataFrame:
    """Compact paper table: rows = reference schemes, columns = lead windows.

    Baseline row = open-loop NSE; every other row = NSE skill score vs that baseline. Per window and row:
    each basin's value is its mean over the window's leads (basins need all leads finite), then ``stat``
    ('median' | 'mean') across basins - the same per-basin score as the risk-return figure. With
    ``common_basins`` each window uses only basins valid for every row, so rows are directly comparable.
    Global/Per-Basin rows follow the notebook-wide selection. ``df.attrs['n_basins']`` = N per window.
    """
    from da_eval.selection import PER_BASIN_DA, PER_BASIN_RHO
    windows = windows or DEFAULT_HORIZON_WINDOWS
    labels = {**DEFAULT_HORIZON_LABELS, **(labels or {})}
    df_eval = _ensure_skill_score_columns(df_eval)
    sel = get_active()
    cfgs = set(df_eval["Config ID"].dropna().astype(str))
    real_da = [c for c in cfgs if not is_rho_config(c) and "per-basin" not in c.lower() and "baseline" not in c.lower()]
    g_da = sel.global_da if sel is not None else get_global_best_config(df_eval)
    g_rho = sel.global_rho if sel is not None else None
    rows = {"global_da": (g_da, metric_col), "global_rho": (g_rho, metric_col),
            "per_basin_da": (PER_BASIN_DA, metric_col), "per_basin_rho": (PER_BASIN_RHO, metric_col)}
    rows = {k: v for k, v in rows.items() if v[0] in cfgs}
    base_src = df_eval[df_eval["Config ID"].isin(real_da)].drop_duplicates(["Basin ID", "Lead Time (Days)"])
    agg = np.median if stat == "median" else np.mean

    def _basin_scores(frame, col, leads):
        w = frame[frame["Lead Time (Days)"].isin(leads)].pivot_table(
            index="Basin ID", columns="Lead Time (Days)", values=col, aggfunc="first").reindex(columns=list(leads))
        return w[w.notna().all(axis=1)].mean(axis=1)

    out, n_basins = {}, {}
    for wname, leads in windows.items():
        leads = tuple(leads)
        per_row = {"baseline": _basin_scores(base_src, "Base NSE", leads)}
        for k, (cid, col) in rows.items():
            per_row[k] = _basin_scores(df_eval[df_eval["Config ID"] == cid], col, leads)
        if common_basins:
            common = set.intersection(*(set(v.index) for v in per_row.values())) if per_row else set()
            per_row = {k: v[v.index.isin(common)] for k, v in per_row.items()}
            n_basins[wname] = len(common)
        else:
            n_basins[wname] = {k: len(v) for k, v in per_row.items()}
        out[wname] = {labels[k]: (float(agg(v)) if len(v) else np.nan) for k, v in per_row.items()}
    order = [labels[k] for k in ("baseline", "global_da", "global_rho", "per_basin_da", "per_basin_rho")]
    table = pd.DataFrame(out).reindex([o for o in order if o in pd.DataFrame(out).index])
    table.index.name = "Assimilation Scheme"
    table.attrs.update(n_basins=n_basins, stat=stat, windows={k: tuple(v) for k, v in windows.items()},
                       selection=sel.label() if sel is not None else "legacy ranking",
                       configs={labels[k]: v[0] for k, v in rows.items()})
    return table


def style_horizon_summary_table(table: pd.DataFrame, decimals: Optional[int] = None, bold_best: bool = True,
                                baseline_in_header: bool = True, percent: Optional[bool] = None) -> Any:
    """Style the horizon summary table.

    ``baseline_in_header``: the Baseline row moves into the column headings ("Days 1–3 / NSE_base = 0.614") and
    the remaining rows are skill scores only. ``percent``: skill scores shown as % (SS_NSE x 100, i.e. the share
    of the open-loop's remaining error variance, 1 - NSE_base, that the scheme removes; +9.9%).
    ``percent=None`` follows the notebook-wide switch (``units.set_ss_percent``).
    Best skill per column in bold. ``decimals`` defaults to 1 (percent) or 3.
    """
    if table.empty:
        return table
    percent = units.ss_percent() if percent is None else bool(percent)
    decimals = (1 if percent else 3) if decimals is None else decimals
    base = table.index[0]
    skill_rows = [r for r in table.index if r != base]
    unit = "%" if percent else ""
    scale = 100.0 if percent else 1.0
    fmt_base = lambda v: "–" if pd.isna(v) else f"{v:.3f}"
    fmt_skill = lambda v: "–" if pd.isna(v) else f"{v * scale:+.{decimals}f}{unit}"
    if baseline_in_header:
        disp = table.loc[skill_rows].copy()
        disp.columns = [f"{c}<br><span style='font-weight:normal'>NSE<sub>base</sub> = {fmt_base(table.loc[base, c])}"
                        f"</span>" for c in table.columns]
        disp.index = [r.replace(f" ({SS_HTML})", "") for r in disp.index]
        disp.index.name = f"Assimilation Scheme — {SS_HTML}" + (" (%)" if percent else "")
        sty = disp.style.format(fmt_skill)
        rows_for_bold = list(disp.index)
    else:
        disp = table
        sty = table.style.format(fmt_base, subset=pd.IndexSlice[[base], :]).format(
            fmt_skill, subset=pd.IndexSlice[skill_rows, :])
        sty = sty.set_properties(subset=pd.IndexSlice[[base], :], **{"border-bottom": "1.5px solid #5f6368"})
        rows_for_bold = skill_rows
    if bold_best and rows_for_bold:
        def _bold(col):
            best = col[rows_for_bold].max()
            return ["font-weight: bold" if (r in rows_for_bold and col[r] == best) else "" for r in col.index]
        sty = sty.apply(_bold, axis=0)
    sty = sty.set_table_styles([{"selector": "th.col_heading", "props": "text-align: center;"}], overwrite=False)
    n = table.attrs.get("n_basins", {})
    n_s = ", ".join(f"{k.split(' (')[0]}: N={v}" for k, v in n.items() if isinstance(v, int))
    stat = table.attrs.get("stat", "median")
    what = (f"{SS_HTML} = (NSE − NSE<sub>base</sub>) / (1 − NSE<sub>base</sub>)"
            + (" × 100%: share of the open-loop's remaining error removed" if percent else ""))
    sty = sty.set_caption(f"{what}. {stat.capitalize()} across basins of each basin's mean over the window's leads | "
                          f"references {table.attrs.get('selection', '')}" + (f" | {n_s}" if n_s else ""))
    return sty


def style_top_n_table(df_matrix: pd.DataFrame) -> Any:
    """Applies clean styling, colorblind blue/red skill/delta shading, and bold column champions."""
    if df_matrix.empty:
        return df_matrix

    delta_cols = [c for c in df_matrix.columns if c.startswith("ΔNSE") or c.startswith("SS_NSE")]
    ss_cols = [c for c in delta_cols if c.startswith("SS_NSE")]
    metric_cols = [c for c in df_matrix.columns if c.startswith("NSE")]
    harm_cols = [c for c in df_matrix.columns if c.startswith("% ")]

    format_dict = {}
    for c in metric_cols:
        format_dict[c] = "{:.3f}"
    for c in delta_cols:
        format_dict[c] = units.ss_formatter() if c in ss_cols else "{:+.3f}"
    for c in harm_cols:
        format_dict[c] = "{:.1f}%"
    not_baseline = ~df_matrix["Configuration"].astype(str).str.contains("Baseline", case=False, na=False) \
        if "Configuration" in df_matrix.columns else pd.Series(True, index=df_matrix.index)

    def highlight_champions(col):
        if col.name in metric_cols or col.name in delta_cols:
            numeric_col = pd.to_numeric(col, errors="coerce")
            is_max = numeric_col == numeric_col.max()
            return ["font-weight: bold; text-decoration: underline;" if v else "" for v in is_max]
        if col.name in harm_cols:
            # Lower harm is better; Baseline is trivially 0% so it is excluded from the champion.
            numeric_col = pd.to_numeric(col, errors="coerce")
            best = numeric_col[not_baseline].min()
            return ["font-weight: bold; text-decoration: underline;" if (v == best and nb) else ""
                    for v, nb in zip(numeric_col, not_baseline)]
        return ["" for _ in col]

    def color_harm(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if np.isnan(v) or v <= 0:
            return "color: #5f6368;"
        # Red intensity scales with the share of harmed basins (saturates at 30%).
        a = min(v / 30.0, 1.0)
        return f"background-color: rgba(197, 34, 31, {0.08 + 0.45 * a:.2f}); color: #3c0d0c;"

    def color_deltas(val):
        try:
            v = float(val)
            if v > 0.005:
                # Soft blue
                return "background-color: #e8f0fe; color: #174ea6; font-weight: bold;"
            elif v < -0.005:
                # Soft red
                return "background-color: #fce8e6; color: #c5221f; font-weight: bold;"
            else:
                return "color: #5f6368;"
        except Exception:
            return ""

    styler = df_matrix.style.format(format_dict, na_rep="-")
    styler = styler.apply(highlight_champions, axis=0)

    if hasattr(styler, "map"):
        styler = styler.map(color_deltas, subset=delta_cols)
    else:
        styler = styler.applymap(color_deltas, subset=delta_cols)
    if harm_cols:
        styler = styler.map(color_harm, subset=harm_cols) if hasattr(styler, "map") else styler.applymap(color_harm, subset=harm_cols)

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("background-color", "#f1f3f4"), ("color", "#202124"), ("font-size", "12px"), ("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-size", "12px"), ("padding", "6px 10px")]},
    ])
    # SS_NSE headers carry the display unit; harm thresholds shown in % (columns / computation unchanged).
    headers = {c: c.replace("SS_NSE", units.ss_label("SS_NSE"), 1) for c in ss_cols}
    headers.update({c: _ss_threshold_header(c) for c in harm_cols})
    return _relabel_columns(styler, headers)


def style_physical_characteristics_table(df_strata: pd.DataFrame) -> Any:
    """Formats and styles the 6-dimension physical characteristics summary table."""
    if df_strata.empty:
        return df_strata

    # IQR columns hold the skill score whenever it is available (see compute_consolidated_physical_strata).
    iqr_ss = "Median NSE Skill Score" in df_strata.columns
    format_dict = {
        "N Basins": "{:,d}",
        "Median Base NSE": "{:.3f}",
        "Median DA NSE": "{:.3f}",
        "Median NSE Skill Score": units.ss_formatter(),
        "Median ΔNSE": "{:+.3f}",
        "IQR Lower (25%)": units.ss_formatter() if iqr_ss else "{:+.3f}",
        "IQR Upper (75%)": units.ss_formatter() if iqr_ss else "{:+.3f}",
        "% Improved (SS>0.01)": "{:.1f}%",
        "% Degraded (SS<-0.01)": "{:.1f}%",
        "% Improved (Δ>0.01)": "{:.1f}%",
        "% Degraded (Δ<-0.01)": "{:.1f}%",
    }
    actual_fmt = {k: v for k, v in format_dict.items() if k in df_strata.columns}

    def color_delta_col(val):
        try:
            v = float(val)
            if v > 0.02:
                return "background-color: #e8f0fe; color: #174ea6; font-weight: bold;"
            elif v < -0.02:
                return "background-color: #fce8e6; color: #c5221f; font-weight: bold;"
            return ""
        except Exception:
            return ""

    styler = df_strata.style.format(actual_fmt, na_rep="-")

    target_subsets = [c for c in ["Median NSE Skill Score", "Median ΔNSE"] if c in df_strata.columns]
    if target_subsets:
        if hasattr(styler, "map"):
            styler = styler.map(color_delta_col, subset=target_subsets)
        else:
            styler = styler.applymap(color_delta_col, subset=target_subsets)

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("background-color", "#e8eaed"), ("color", "#202124"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("padding", "6px 12px")]},
    ])
    headers = {c: _ss_threshold_header(c) for c in df_strata.columns if isinstance(c, str) and "SS" in c}
    if "Median NSE Skill Score" in df_strata.columns:
        headers["Median NSE Skill Score"] = units.ss_label("Median NSE Skill Score")
    return _relabel_columns(styler, headers)


def build_top_n_optimal_basin_distribution_table(
    df_eval: pd.DataFrame,
    top_n: int = 10,
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
    filter_top_n_global: Optional[int] = None,
) -> pd.DataFrame:
    """Constructs a table showing the top N configurations ranked by how many basins

    for which they are the optimal (best) configuration, comparing them against the
    Globally Best DA configuration.
    """
    if df_eval.empty:
        return pd.DataFrame()

    df_eval = _ensure_skill_score_columns(df_eval)
    if metric_col not in df_eval.columns:
        metric_col = "NSE Delta"

    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

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

    # Identify optimal configuration per basin
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

    # Count wins
    win_counts = best_per_basin["Config ID"].value_counts().reset_index()
    win_counts.columns = ["Config ID", "Basins Won"]
    win_counts["Win Share (%)"] = (win_counts["Basins Won"] / total_basins) * 100.0
    win_counts["Cumulative Basins"] = win_counts["Basins Won"].cumsum()
    win_counts["Cumulative Share (%)"] = (win_counts["Cumulative Basins"] / total_basins) * 100.0

    med_col_label = "Global Med SS_NSE" if "Skill" in metric_col else "Global Med ΔNSE"
    global_med_delta = sub.groupby("Config ID")[metric_col].median().rename(med_col_label)
    da_metric = "DA NSE" if "DA NSE" in sub.columns else metric_col
    global_med_score = sub.groupby("Config ID")[da_metric].median().rename("Global Med NSE")
    global_rank = global_med_delta.rank(ascending=False, method="min").astype(int).rename("Global Rank")

    win_counts = win_counts.merge(global_med_delta, on="Config ID", how="left")
    win_counts = win_counts.merge(global_med_score, on="Config ID", how="left")
    win_counts = win_counts.merge(global_rank, on="Config ID", how="left")
    win_counts["Is Global Best"] = win_counts["Config ID"] == global_best_cfg

    top_df = win_counts.head(top_n).copy()
    top_df.insert(0, "Rank", range(1, len(top_df) + 1))

    # Parse hyperparameters
    parsed = top_df["Config ID"].apply(parse_da_config_id)
    top_df["Window"] = parsed["Window (Days)"]
    top_df["LR"] = parsed["Learning Rate"]
    top_df["Epochs"] = parsed["Epochs"]
    top_df["BG Reg"] = parsed["BG Weight"]
    top_df["Target"] = parsed["Target"]

    cols = [
        "Rank",
        "Config ID",
        "Is Global Best",
        "Basins Won",
        "Win Share (%)",
        "Cumulative Basins",
        "Cumulative Share (%)",
        med_col_label,
        "Global Med NSE",
        "Global Rank",
        "Window",
        "LR",
        "Epochs",
        "BG Reg",
        "Target",
    ]
    cols = [c for c in cols if c in top_df.columns]
    return top_df[cols]


def style_top_n_optimal_basin_distribution_table(df_table: pd.DataFrame) -> Any:
    """Applies styling to optimal basin distribution table with Global Best highlight and delta shading."""
    if df_table.empty:
        return df_table

    format_dict = {
        "Basins Won": "{:,d}",
        "Win Share (%)": "{:.1f}%",
        "Cumulative Basins": "{:,d}",
        "Cumulative Share (%)": "{:.1f}%",
        "Global Med SS_NSE": units.ss_formatter(),
        "Global Med ΔNSE": "{:+.3f}",
        "Global Med NSE": "{:.3f}",
        "Global Rank": "{:d}",
    }
    actual_fmt = {k: v for k, v in format_dict.items() if k in df_table.columns}

    def highlight_global_best(row):
        if row.get("Is Global Best", False):
            return ["background-color: #e8f0fe; font-weight: bold; color: #174ea6;" for _ in row]
        return ["" for _ in row]

    styler = df_table.style.format(actual_fmt, na_rep="-")
    styler = styler.apply(highlight_global_best, axis=1)

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("background-color", "#f1f3f4"), ("color", "#202124"), ("font-size", "12px"), ("text-align", "center"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-size", "12px"), ("padding", "6px 10px")]},
    ])
    return _relabel_columns(styler, {c: units.ss_label(c) for c in ("Global Med SS_NSE",) if c in df_table.columns})


def build_hyperparameter_risk_table(
    df_eval: pd.DataFrame,
    group_by: str = "Learning Rate",
    lead_time: int = 1,
    metric_col: str = "NSE Skill Score",
) -> pd.DataFrame:
    """Computes hyperparameter distribution and risk profiles (Q0.05, tail severity, win rates).

    Args:
        df_eval: Evaluation dataframe with Config ID and metrics.
        group_by: Parameter to stratify by ('Learning Rate', 'Window (Days)', 'BG Weight', 'Tolerance').
        lead_time: Forecast horizon in days (e.g. 1).
        metric_col: Target metric ('NSE Skill Score', 'NSE Delta', 'KGE Skill Score', 'KGE Delta').
    """
    import re
    if df_eval.empty:
        return pd.DataFrame()

    df_eval = _ensure_skill_score_columns(df_eval)
    sub = df_eval[
        (df_eval["Lead Time (Days)"] == lead_time)
        & (~df_eval["Config ID"].str.contains("Per-Basin", na=False))
        & (~df_eval["Config ID"].str.contains("Baseline", na=False))
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    if metric_col not in sub.columns:
        metric_col = "NSE Skill Score" if "NSE Skill Score" in sub.columns else "NSE Delta"

    # Regex parameter extraction
    def _parse(cid):
        w_m = re.search(r"_w(\d+)", cid)
        lr_m = re.search(r"_(?:lr|lrd)([0-9.eE+-]+)", cid)
        bg_m = re.search(r"_(?:bg|bgd)([0-9.eE+-]+)", cid)
        tol_m = re.search(r"_tol([A-Za-z0-9\.]+)", cid)
        stat_m = re.search(r"_(?:stat|bgs)([0-9.eE+-]+)", cid)
        ratio_m = re.search(r"_r([0-9][0-9.]*)(?=_|$)", cid)
        ep_m = re.search(r"_ep(\d+)", cid)
        return {
            "Window (Days)": int(w_m.group(1)) if w_m else np.nan,
            "Learning Rate": float(lr_m.group(1)) if lr_m else np.nan,
            "BG Weight": float(bg_m.group(1)) if bg_m else np.nan,
            "Stat Weight": float(stat_m.group(1)) if stat_m else (float(bg_m.group(1)) if bg_m else np.nan),
            "Tolerance": str(tol_m.group(1)) if tol_m else "None",
            "LR Ratio": float(ratio_m.group(1)) if ratio_m else np.nan,
            "Epochs": int(ep_m.group(1)) if ep_m else np.nan,
        }

    parsed = [_parse(c) for c in sub["Config ID"]]
    for k in ["Window (Days)", "Learning Rate", "BG Weight", "Stat Weight", "Tolerance", "LR Ratio", "Epochs"]:
        sub[k] = [p[k] for p in parsed]

    # Map aliases
    col_map = {
        "lr": "Learning Rate",
        "learning_rate": "Learning Rate",
        "w": "Window (Days)",
        "window": "Window (Days)",
        "bg": "BG Weight",
        "bg_reg": "BG Weight",
        "bg_weight": "BG Weight",
        "stat_weight": "Stat Weight",
        "tol": "Tolerance",
        "tolerance": "Tolerance",
    }
    norm_group = col_map.get(str(group_by).lower(), group_by)

    # Per-config distributions across basins
    cfg_stats = sub.groupby(["Config ID", norm_group])[metric_col].agg(
        q05=lambda x: x.quantile(0.05),
        q25=lambda x: x.quantile(0.25),
        median="median",
        q75=lambda x: x.quantile(0.75),
        q95=lambda x: x.quantile(0.95),
        win_rate=lambda x: (x > 0.0).mean() * 100.0,
        severe_degraded=lambda x: (x < -0.10).mean() * 100.0,
    ).reset_index()

    # Stratified summary across configurations
    summary = cfg_stats.groupby(norm_group).agg(
        n_configs=("Config ID", "count"),
        mean_q05=("q05", "mean"),
        mean_q25=("q25", "mean"),
        mean_median=("median", "mean"),
        mean_q75=("q75", "mean"),
        mean_q95=("q95", "mean"),
        safe_configs=("q05", lambda x: f"{(x > -0.10).sum()}/{len(x)} ({(x > -0.10).mean()*100:.0f}%)"),
        mean_win_rate=("win_rate", "mean"),
        mean_severe_deg=("severe_degraded", "mean"),
    ).reset_index()

    summary.columns = [
        norm_group,
        "Configs",
        "Mean Q0.05 (Downside Tail)",
        "Mean Q0.25",
        "Mean Median",
        "Mean Q0.75",
        "Mean Q0.95 (Upside Tail)",
        "Safe Configs (Q0.05 > -0.10)",
        "Mean Win Rate (%)",
        "Severe Degraded (< -0.10) (%)",
    ]
    summary.attrs["metric_col"] = metric_col
    return summary


_RISK_SS_COLS = ("Mean Q0.05 (Downside Tail)", "Mean Q0.25", "Mean Median", "Mean Q0.75", "Mean Q0.95 (Upside Tail)")


def style_hyperparameter_risk_table(df_table: pd.DataFrame, metric_col: Optional[str] = None) -> Any:
    """Applies clean styling, color shading for safe configs, and formatted percentiles.

    ``metric_col`` (default: ``df_table.attrs['metric_col']`` set by ``build_hyperparameter_risk_table``): for
    'NSE Skill Score' the quantile columns follow the notebook-wide SS display unit (``units.set_ss_percent``).
    """
    if df_table.empty:
        return df_table
    metric_col = metric_col or df_table.attrs.get("metric_col")
    is_ss = metric_col == "NSE Skill Score"
    q_fmt = units.ss_formatter() if is_ss else "{:+.3f}"

    fmt_dict = {
        "Configs": "{:d}",
        "Mean Q0.05 (Downside Tail)": q_fmt,
        "Mean Q0.25": q_fmt,
        "Mean Median": q_fmt,
        "Mean Q0.75": q_fmt,
        "Mean Q0.95 (Upside Tail)": q_fmt,
        "Mean Win Rate (%)": "{:.1f}%",
        "Severe Degraded (< -0.10) (%)": "{:.1f}%",
    }
    actual_fmt = {k: v for k, v in fmt_dict.items() if k in df_table.columns}

    def _color_tail(val):
        try:
            v = float(val)
            if v > -0.05:
                return "background-color: #e6f4ea; color: #137333; font-weight: bold;"
            elif v < -0.15:
                return "background-color: #fce8e6; color: #c5221f; font-weight: bold;"
            else:
                return "background-color: #fef7e0; color: #b06000;"
        except Exception:
            return ""

    def _color_median(val):
        try:
            v = float(val)
            if v >= 0.12:
                return "background-color: #e8f0fe; color: #174ea6; font-weight: bold;"
            elif v >= 0.08:
                return "color: #174ea6;"
            return "color: #5f6368;"
        except Exception:
            return ""

    styler = df_table.style.format(actual_fmt, na_rep="—")
    if "Mean Q0.05 (Downside Tail)" in df_table.columns:
        styler = styler.applymap(_color_tail, subset=["Mean Q0.05 (Downside Tail)"])
    if "Mean Median" in df_table.columns:
        styler = styler.applymap(_color_median, subset=["Mean Median"])

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("background-color", "#1a73e8"), ("color", "white"), ("font-size", "12px"), ("text-align", "center"), ("font-weight", "bold"), ("padding", "6px 10px")]},
        {"selector": "td", "props": [("text-align", "center"), ("font-size", "12px"), ("padding", "6px 12px")]},
        {"selector": "tr:hover", "props": [("background-color", "#f8f9fa")]},
    ])
    if not is_ss:
        return styler
    headers = {c: units.ss_label(c) for c in _RISK_SS_COLS if c in df_table.columns}
    headers.update({c: _ss_threshold_header(c) for c in ("Safe Configs (Q0.05 > -0.10)", "Severe Degraded (< -0.10) (%)")
                    if c in df_table.columns})
    return _relabel_columns(styler, headers)
