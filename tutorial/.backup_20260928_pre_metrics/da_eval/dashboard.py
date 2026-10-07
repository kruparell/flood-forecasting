"""Interactive diagnostic dashboard supporting both Bokeh Clickable Map (TapTool) and ipywidgets.

Features:
- Dynamic Catchment Info Card (Morphology, Climate Indices, Baseline vs DA Metrics)
- Bokeh Map with Natural Earth country boundaries underlay & TapTool point-clicking
- Continuous Hydrograph showing NSE and KGE directly in the legend for each model
- Basin-specific Leadtime Skill Persistence Decay curve
"""

import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import geopandas as gpd
from IPython.display import HTML, display
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from da_eval.benchmarks import get_global_best_config
from da_eval.selection import get_active
from da_eval.ingestion import _ensure_skill_score_columns, detect_lead_times, is_per_basin_rho_config, is_rho_config
from da_eval import koppen as kg
from da_eval import units

KG_COLS = ["kg_code", "kg_symbol", "kg_group", "kg_group_name", "kg_frac", "kg_method"]


def _parse_kg_filter(kg_filter) -> Tuple[set, set]:
    """'all' / None -> no filter; 'B' -> Arid group; 'B, Cfb' -> Arid group plus class Cfb."""
    if kg_filter is None:
        return set(), set()
    toks = kg_filter if isinstance(kg_filter, (list, tuple, set)) else re.split(r"[,\s]+", str(kg_filter))
    toks = [str(t).strip() for t in toks if str(t).strip() and str(t).strip().lower() != "all"]
    groups = {t.upper() for t in toks if len(t) == 1}
    syms = {t[0].upper() + t[1:] for t in toks if len(t) > 1}
    return groups, syms


def _kg_filter_label(kg_filter) -> str:
    groups, syms = _parse_kg_filter(kg_filter)
    parts = [f"{g} {kg.KG_GROUPS[g][0]}" for g in sorted(groups) if g in kg.KG_GROUPS] + sorted(syms)
    return ", ".join(parts)


def _attach_koppen(df: pd.DataFrame, koppen_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Adds per-basin Köppen-Geiger columns (computing + caching them from Caravan polygons if needed)."""
    if df is None or df.empty or "kg_code" in df.columns:
        return df
    if koppen_df is None:
        ll = df[["Basin ID", "lat", "lon"]] if {"lat", "lon"} <= set(df.columns) else None
        koppen_df = kg.compute_basin_koppen(df["Basin ID"].dropna().astype(str).unique(), lat_lon=ll, verbose=False)
    cols = ["Basin ID"] + [c for c in KG_COLS if c in koppen_df.columns]
    return pd.merge(df, koppen_df[cols].drop_duplicates("Basin ID"), on="Basin ID", how="left")


def _apply_kg_filter(df: pd.DataFrame, kg_filter) -> pd.DataFrame:
    groups, syms = _parse_kg_filter(kg_filter)
    if not (groups or syms) or df.empty or "kg_group" not in df.columns:
        return df
    return df[df["kg_group"].isin(groups) | df["kg_symbol"].isin(syms)]


def _draw_kg_layer(ax, df_points: pd.DataFrame, koppen_mode: str, kg_filter, alpha: float, world_gdf) -> None:
    groups, syms = _parse_kg_filter(kg_filter)
    shade_groups = groups if (groups and not syms) else None
    present = df_points["kg_code"].dropna().unique() if "kg_code" in df_points.columns else None
    kg.draw_koppen_underlay(ax, mode=koppen_mode, alpha=alpha, groups=shade_groups,
                            present_codes=present, world_gdf=world_gdf)

# Natural Earth country boundaries shapefile path
COUNTRY_SHP_PATHS = [
    "/usr/local/google/home/kruparell/flood-forecasting/data/countries/ne_110m_admin_0_countries.shp",
    "/usr/local/google/home/kruparell/Caravans_V2/ne_110m_admin_0_countries.shp",
]

# 5-category discrete diverging palette:
# [Strong Loss (<-t2), Moderate Loss (-t2..-t1), Neutral (-t1..+t1), Moderate Gain (+t1..+t2), Strong Gain (>+t2)]
DISCRETE_5BIN_COLORS = [
    "#b2182b",  # Deep crimson (strong degradation)
    "#f4a582",  # Coral / soft red (moderate degradation)
    "#d5d8dc",  # Distinct warm slate grey (neutral within +-t1)
    "#67a9cf",  # Sky blue (moderate improvement)
    "#2166ac",  # Deep navy blue (strong improvement)
]

REGION_EXTENTS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "north_america": ((-130.0, -60.0), (23.0, 58.0)),
    "south_america": ((-82.0, -33.0), (-56.0, 12.0)),
    "europe": ((-12.0, 32.0), (35.0, 64.0)),
    "australia": ((112.0, 155.0), (-44.0, -10.0)),
}


def _make_diverging_norm(vmin: float, vmax: float, vcenter: float = 0.0):
    """Creates a diverging color normalization centered at vcenter when vmin < vcenter < vmax,
    falling back safely to linear Normalize if the user-chosen range does not straddle vcenter.
    """
    lo = float(vmin) if np.isfinite(vmin) else -0.2
    hi = float(vmax) if np.isfinite(vmax) else 0.2
    if hi <= lo:
        hi = lo + 1e-3
    if lo < vcenter < hi:
        return TwoSlopeNorm(vmin=lo, vcenter=vcenter, vmax=hi)
    return Normalize(vmin=lo, vmax=hi)


def _parse_bin_thresholds(
    bin_thresholds=None,
    default_pair: Tuple[float, float] = (0.01, 0.05),
) -> Tuple[float, float, float, float]:
    """Parses user-supplied bin thresholds into 4 strictly increasing interior edges (e0, e1, e2, e3)
    defining 5 categories: (< e0), [e0, e1), [e1, e2], (e2, e3], (> e3).

    Accepts:
      - None -> (-0.05, -0.01, 0.01, 0.05)
      - "0.01, 0.05" or (0.01, 0.05) -> (-0.05, -0.01, +0.01, +0.05)
      - "-0.08, -0.02, 0.02, 0.08" or 4-tuple -> (-0.08, -0.02, 0.02, 0.08)
    """
    t1_def, t2_def = sorted([abs(float(default_pair[0])), abs(float(default_pair[1]))])
    if t2_def <= t1_def:
        t2_def = t1_def + 0.04
    fallback = (-t2_def, -t1_def, t1_def, t2_def)

    if bin_thresholds is None:
        return fallback

    nums: List[float] = []
    if isinstance(bin_thresholds, str):
        parts = [p.strip() for p in re.split(r"[,;\s]+", bin_thresholds.strip()) if p.strip()]
        for p in parts:
            try:
                v = float(p)
                if np.isfinite(v):
                    nums.append(v)
            except ValueError:
                pass
    elif isinstance(bin_thresholds, (list, tuple, np.ndarray)):
        for p in bin_thresholds:
            try:
                v = float(p)
                if np.isfinite(v):
                    nums.append(v)
            except (ValueError, TypeError):
                pass
    elif isinstance(bin_thresholds, (int, float)) and np.isfinite(float(bin_thresholds)):
        v = abs(float(bin_thresholds))
        if v > 0:
            nums = [v, v * 5.0]

    if len(nums) == 2:
        a, b = sorted([abs(nums[0]), abs(nums[1])])
        if a <= 0:
            a = 0.01
        if b <= a:
            b = a * 2.0
        return (-b, -a, a, b)
    elif len(nums) >= 4:
        s = sorted(nums[:4])
        for i in range(1, 4):
            if s[i] <= s[i - 1]:
                s[i] = s[i - 1] + 1e-4
        return (s[0], s[1], s[2], s[3])
    elif len(nums) == 1 and abs(nums[0]) > 0:
        a = abs(nums[0])
        return (-5.0 * a, -a, a, 5.0 * a)

    return fallback


def _classify_into_5_bins(
    values: np.ndarray,
    edges: Tuple[float, float, float, float],
    ss_units: bool = False,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Assigns each finite value to bin index 0..4 and formats 5 category labels with counts/pct.

    ``ss_units=True`` marks the values as NSE skill scores: labels are then shown in the notebook-wide
    SS display units (``units.fmt_ss``, e.g. '< -5%', '-5% to -1%', '±1%'); bin edges are unchanged.
    """
    e0, e1, e2, e3 = edges
    arr = np.asarray(values, dtype=float)
    bin_idx = np.full(len(arr), 2, dtype=int)
    finite = np.isfinite(arr)
    v = arr[finite]
    cats = np.full(len(v), 2, dtype=int)
    cats[v < e0] = 0
    cats[(v >= e0) & (v < e1)] = 1
    cats[(v >= e1) & (v <= e2)] = 2
    cats[(v > e2) & (v <= e3)] = 3
    cats[v > e3] = 4
    bin_idx[finite] = cats

    n_valid = max(1, int(finite.sum()))
    counts = [int((cats == k).sum()) if finite.any() else 0 for k in range(5)]
    pcts = [c / n_valid * 100.0 for c in counts]

    if ss_units and units.ss_percent():
        # Fewest decimals that represent every edge exactly in % (0.01 -> '1%', 0.005 -> '0.5%').
        dec = next((d for d in range(4) if all(np.isclose(round(abs(e) * 100, d), abs(e) * 100)
                                               for e in edges)), 3)
        _fmt = lambda val: units.fmt_ss(val, decimals=dec, sign=True)  # noqa: E731
        sep = " to "
        mid_mag = units.fmt_ss(abs(e2), decimals=dec, sign=False)
    else:
        def _fmt(val: float) -> str:
            return f"{val:+.2g}" if abs(val) >= 0.001 else f"{val:+.3f}"
        sep = ".."
        mid_mag = f"{abs(e2):.2g}"

    if np.isclose(abs(e1), abs(e2)) and e1 < 0 < e2:
        mid_str = f"±{mid_mag}"
    else:
        mid_str = f"{_fmt(e1)}{sep}{_fmt(e2)}"

    base_labels = [
        f"< {_fmt(e0)}",
        f"{_fmt(e0)}{sep}{_fmt(e1)}",
        mid_str,
        f"{_fmt(e2)}{sep}{_fmt(e3)}",
        f"> {_fmt(e3)}",
    ]
    labels_with_pct = [f"{lbl}\n({pct:.0f}%)" for lbl, pct in zip(base_labels, pcts)]
    return bin_idx, labels_with_pct, counts


def _render_map_scatter_and_colorbar(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    cbar_label: str,
    color_mode: str = "binned",
    bin_thresholds=None,
    vmin: float = -0.2,
    vmax: float = 0.2,
    marker_size: float = 10.0,
    is_diverging: bool = True,
    cbar_shrink: float = 0.78,
    cbar_pad: float = 0.05,
    font_scale: float = 1.0,
    cbar_fraction: float = 0.15,
    ss_units: bool = False,
):
    """Draws catchment markers with magnitude-based Z-ordering and either a 5-category discrete
    colorbar (default for ΔNSE / Skill Score) or a continuous colorbar.

    ``ss_units=True`` (NSE skill score only) shows the colourbar label / ticks / bin labels in the
    notebook-wide SS display units (``da_eval.units``); data, ``vmin``/``vmax`` and bin edges are unchanged.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    vals = np.asarray(values, dtype=float)
    s_base = max(1.0, float(marker_size))

    if len(vals) == 0 or np.isnan(vals).all():
        ax.scatter(
            lons, lats, color="#1a73e8", s=s_base, alpha=0.85,
            edgecolors="#0d47a1", linewidth=0.35, zorder=3
        )
        return

    if ss_units:
        cbar_label = units.ss_label(cbar_label)
    use_binned = str(color_mode).strip().lower() in ("binned", "discrete", "5bin", "categories")
    if use_binned:
        edges = _parse_bin_thresholds(
            bin_thresholds,
            default_pair=(0.01, 0.05) if is_diverging else (0.3, 0.6),
        )
        if not is_diverging and bin_thresholds is None:
            edges = (0.0, 0.3, 0.5, 0.7)
        bin_idx, tick_labels, _ = _classify_into_5_bins(vals, edges, ss_units=ss_units)
        cmap_5 = ListedColormap(DISCRETE_5BIN_COLORS)
        norm_5 = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_5.N)

        # Draw in 3 z-order tiers so neutral (±0.01) never obscures moderate/strong gain or loss
        tiers = [
            (bin_idx == 2, 0.72 * s_base, 0.75, "#7f8c8d", 0.28, 3),                  # Neutral bottom
            ((bin_idx == 1) | (bin_idx == 3), 0.95 * s_base, 0.88, "#3c4043", 0.32, 4),  # Moderate mid
            ((bin_idx == 0) | (bin_idx == 4), 1.12 * s_base, 0.95, "#202124", 0.40, 5),  # Strong top
        ]
        for mask, s_tier, alpha_tier, edge_c, lw_tier, z_tier in tiers:
            if np.any(mask):
                c_hex = [DISCRETE_5BIN_COLORS[i] for i in bin_idx[mask]]
                ax.scatter(
                    lons[mask], lats[mask],
                    c=c_hex,
                    s=s_tier,
                    alpha=alpha_tier,
                    edgecolors=edge_c,
                    linewidth=lw_tier,
                    zorder=z_tier,
                )

        sm = cm.ScalarMappable(cmap=cmap_5, norm=norm_5)
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="horizontal", pad=cbar_pad, shrink=cbar_shrink, fraction=cbar_fraction,
            ticks=[0.5, 1.5, 2.5, 3.5, 4.5],
        )
        cbar.ax.set_xticklabels(tick_labels, fontsize=8.5 * font_scale, fontweight="bold")
        cbar.set_label(cbar_label, fontsize=9.5 * font_scale, fontweight="bold")
    else:
        # Sort ascending by |val| (for diverging) or val so highest-magnitude points render on top
        finite_mask = np.isfinite(vals)
        order = np.argsort(np.abs(vals[finite_mask]) if is_diverging else vals[finite_mask])
        f_lons = lons[finite_mask][order]
        f_lats = lats[finite_mask][order]
        f_vals = vals[finite_mask][order]

        if is_diverging:
            norm = _make_diverging_norm(vmin, vmax, vcenter=0.0)
            sc = ax.scatter(
                f_lons, f_lats, c=f_vals, cmap="RdBu", norm=norm,
                s=s_base, alpha=0.9, edgecolors="#202124", linewidth=0.25, zorder=4
            )
        else:
            lo = float(vmin) if np.isfinite(vmin) else 0.0
            hi = float(vmax) if np.isfinite(vmax) else 1.0
            if hi <= lo:
                hi = lo + 1e-3
            sc = ax.scatter(
                f_lons, f_lats, c=f_vals, cmap="viridis", vmin=lo, vmax=hi,
                s=s_base, alpha=0.9, edgecolors="#202124", linewidth=0.25, zorder=4
            )
        cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=cbar_pad, shrink=cbar_shrink,
                            fraction=cbar_fraction)
        if ss_units:
            units.ss_colorbar(cbar)
        cbar.ax.tick_params(labelsize=8.5 * font_scale)
        cbar.set_label(cbar_label, fontsize=9.5 * font_scale, fontweight="bold")
    return cbar


def _apply_region_extent(ax: plt.Axes, lons: np.ndarray, lats: np.ndarray, region: str = "global") -> None:
    """Applies either a named continental zoom preset or a tightly cropped global gauge extent."""
    reg_key = str(region).strip().lower()
    if reg_key in REGION_EXTENTS:
        xlim, ylim = REGION_EXTENTS[reg_key]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return
    if len(lons) > 0:
        lon_min = max(-135.0, min(-125.0, float(np.nanmin(lons)) - 5.0))
        lon_max = min(162.0, max(152.0, float(np.nanmax(lons)) + 5.0))
        lat_min = max(-57.0, min(-52.0, float(np.nanmin(lats)) - 4.0))
        lat_max = min(68.0, max(62.0, float(np.nanmax(lats)) + 4.0))
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)


def get_complete_streamflow_basins(
    df_ts: pd.DataFrame,
    year: Optional[int] = None,
    lead_time: Optional[int] = None,
) -> set:
    """Returns the set of Basin IDs that have 100% complete (non-missing, finite) observed
    streamflow (q_obs) across all expected dates in the specified plotting period (year).
    """
    if df_ts is None or df_ts.empty or "Basin ID" not in df_ts.columns or "q_obs" not in df_ts.columns:
        return set()

    sub = df_ts
    if year is not None and "Valid Date" in sub.columns:
        yr_sub = sub[pd.to_datetime(sub["Valid Date"]).dt.year == int(year)]
        if not yr_sub.empty:
            sub = yr_sub

    if lead_time is not None and "Lead Time (Days)" in sub.columns:
        lt_sub = sub[sub["Lead Time (Days)"] == int(lead_time)]
        if not lt_sub.empty:
            sub = lt_sub

    if sub.empty:
        return set()

    if "Valid Date" in sub.columns:
        dedup = sub.drop_duplicates(subset=["Basin ID", "Valid Date"])
        expected_days = int(dedup["Valid Date"].nunique())
        if expected_days <= 0:
            return set()
        valid_mask = dedup["q_obs"].notna() & np.isfinite(dedup["q_obs"].astype(float))
        total_day_counts = dedup.groupby("Basin ID")["Valid Date"].nunique()
        valid_day_counts = (dedup.loc[valid_mask].groupby("Basin ID")["Valid Date"].nunique()
                            .reindex(total_day_counts.index, fill_value=0))
        complete_ids = valid_day_counts[
            (valid_day_counts >= expected_days) & (valid_day_counts == total_day_counts)
        ].index
        return {str(b) for b in complete_ids}
    else:
        valid_mask = sub["q_obs"].notna() & np.isfinite(sub["q_obs"].astype(float))
        any_na_basins = set(sub.loc[~valid_mask, "Basin ID"].dropna().astype(str).unique())
        all_basins = set(sub["Basin ID"].dropna().astype(str).unique())
        return all_basins - any_na_basins


class InteractiveDADashboard:
    """Linked interactive dashboard for exploring catchment-level DA dynamics."""

    def __init__(
        self,
        df_eval: pd.DataFrame,
        df_ts: pd.DataFrame,
        df_meta: Optional[pd.DataFrame] = None,
        default_year: int = 2019,
        default_vmin: float = -0.2,
        default_vmax: float = 0.2,
        default_filter_complete_streamflow: bool = False,
        default_color_by: str = "skill_nse",
        default_color_mode: str = "binned",
        default_bin_thresholds="0.01, 0.05",
        default_marker_size: float = 10.0,
        default_region: str = "global",
        ts_loader: Optional[Callable[[str], pd.DataFrame]] = None,
        ts_configs: Optional[List[str]] = None,
        ts_basins: Optional[List[str]] = None,
        obs_df: Optional[pd.DataFrame] = None,
        obs_loader: Optional[Callable[[], pd.DataFrame]] = None,
        default_koppen: str = "off",
        default_koppen_filter="all",
        default_koppen_alpha: float = 0.4,
        koppen_df: Optional[pd.DataFrame] = None,
    ):
        """Args (lazy mode): ``ts_loader(basin)`` returns that basin's timeseries on demand;
        ``ts_configs`` / ``ts_basins`` list what the lazy store can serve; ``obs_df`` / ``obs_loader``
        provide observed-flow-only data for the complete-streamflow filter.

        Köppen-Geiger (Beck et al. 2023): ``default_koppen`` in {'off', 'groups', 'classes'} sets the map
        underlay; ``default_koppen_filter`` ('all', a group letter like 'B', or classes like 'BSk, Cfb')
        restricts the plotted catchments; ``koppen_df`` optionally supplies precomputed basin classes.
        """
        self.df_eval = _ensure_skill_score_columns(df_eval.copy())
        self.df_ts = df_ts.copy()
        self.ts_loader = ts_loader
        self._ts_loaded_basins: set = set(self.df_ts["Basin ID"].astype(str).unique()) if (not self.df_ts.empty and "Basin ID" in self.df_ts.columns) else set()
        self._obs_df = obs_df
        self._obs_loader = obs_loader
        self.df_meta = df_meta.copy() if df_meta is not None else pd.DataFrame()
        self.default_year = default_year
        self.default_vmin = default_vmin
        self.default_vmax = default_vmax
        self.default_filter_complete_streamflow = default_filter_complete_streamflow
        self.default_color_by = default_color_by
        self.default_color_mode = default_color_mode
        self.default_bin_thresholds = default_bin_thresholds
        self.default_marker_size = float(default_marker_size)
        self.default_region = default_region
        self.lead_times = detect_lead_times(self.df_eval)
        self.ref_lead = 1 if 1 in self.lead_times else self.lead_times[0]
        self.global_best_cfg = get_global_best_config(self.df_eval, lead_time=self.ref_lead)

        # Load world boundaries shapefile once
        self.world_gdf = None
        for p in COUNTRY_SHP_PATHS:
            if os.path.exists(p):
                try:
                    self.world_gdf = gpd.read_file(p)
                    break
                except Exception:
                    pass

        # Merge metadata into eval for quick lookup
        if not self.df_meta.empty and "Basin ID" in self.df_meta.columns:
            missing_meta = [c for c in self.df_meta.columns if c not in self.df_eval.columns or c == "Basin ID"]
            if len(missing_meta) > 1:
                self.df_eval = pd.merge(self.df_eval, self.df_meta[missing_meta].drop_duplicates("Basin ID"), on="Basin ID", how="left")

        self.default_koppen = str(default_koppen or "off").lower()
        self.default_koppen_filter = default_koppen_filter
        self.default_koppen_alpha = float(default_koppen_alpha)
        try:
            self.df_eval = _attach_koppen(self.df_eval, koppen_df)
        except Exception as e:
            print(f"[koppen] Could not attach Köppen-Geiger classes: {e}")

        # Determine all available basins strictly with valid, non-null time-series data
        if ts_basins is not None:  # Lazy mode: everything the store can serve
            ts_set = {str(b).lower() for b in ts_basins}
            raw_basins = [str(b) for b in self.df_eval["Basin ID"].dropna().unique() if str(b).lower() in ts_set]
        elif not self.df_ts.empty and "Basin ID" in self.df_ts.columns:
            ts_cols = [c for c in ["q_obs", "q_da", "q_base"] if c in self.df_ts.columns]
            if ts_cols:
                valid_ts_mask = self.df_ts[ts_cols].notna().any(axis=1)
                valid_ts_basins = set(self.df_ts.loc[valid_ts_mask, "Basin ID"].dropna().astype(str).unique())
            else:
                valid_ts_basins = set(self.df_ts["Basin ID"].dropna().astype(str).unique())

            eval_basins = [str(b) for b in self.df_eval["Basin ID"].dropna().unique()]
            lower_ts_map = {str(b).lower(): str(b) for b in valid_ts_basins}
            raw_basins = [b for b in eval_basins if b.lower() in lower_ts_map]
            if not raw_basins:
                raw_basins = list(valid_ts_basins)
        else:
            raw_basins = [str(b) for b in self.df_eval["Basin ID"].dropna().unique()]

        self.available_basins = sorted(list(set(raw_basins)), key=lambda s: s.lower())
        self._complete_basins_cache: Dict[Tuple[Optional[int], Optional[int]], List[str]] = {}

        # Metric and timeseries frames disagree on configuration naming (see _normalize_cfg_name),
        # so precompute a normalized lookup to translate evaluation config IDs into timeseries ones.
        self._ts_cfg_lookup: Dict[str, str] = {}
        if not self.df_ts.empty and "Config ID" in self.df_ts.columns:
            for cfg in self.df_ts["Config ID"].dropna().unique():
                self._ts_cfg_lookup.setdefault(self._normalize_cfg_name(cfg), str(cfg))
        for cfg in (ts_configs or []):
            self._ts_cfg_lookup.setdefault(self._normalize_cfg_name(cfg), str(cfg))

        # Timeseries ingestion frequently covers only a subset of the swept configurations, so the
        # headline global best may have no hydrograph. Fall back to the best-ranked config that does.
        self.global_best_plot_cfg = self._resolve_global_best_with_timeseries()
        # AR(1) references (generated on the fly from Baseline + obs in lazy mode).
        self.global_best_rho_cfg, self.per_basin_rho_cfg = self._resolve_rho_cfgs()

        # Expose a pre-filtered default map slice for quick external verification
        init_lead_df = self.df_eval[self.df_eval["Lead Time (Days)"] == self.ref_lead].copy()
        if self.default_filter_complete_streamflow:
            c_set = set(self.get_complete_basins_for_year(self.default_year, self.ref_lead))
            init_lead_df = init_lead_df[init_lead_df["Basin ID"].astype(str).isin(c_set)].copy()
        self.df_map = init_lead_df.drop_duplicates(subset=["Basin ID"])

    def _obs_source(self) -> pd.DataFrame:
        """Observed-flow frame for completeness checks (lazy: obs-only cached frame; else df_ts)."""
        if self._obs_df is None and self._obs_loader is not None:
            self._obs_df = self._obs_loader()
        return self._obs_df if self._obs_df is not None else self.df_ts

    def _ensure_basin_ts(self, basin_id: str) -> None:
        """Lazy mode: fetch one basin's timeseries on first use (~0.05 s)."""
        if self.ts_loader is None or not basin_id or basin_id in self._ts_loaded_basins:
            return
        new = self.ts_loader(basin_id)
        self._ts_loaded_basins.add(basin_id)
        if new is not None and not new.empty:
            self.df_ts = pd.concat([self.df_ts, new], ignore_index=True) if not self.df_ts.empty else new.copy()

    def get_complete_basins_for_year(self, year: Optional[int], lead_time: Optional[int] = None) -> List[str]:
        """Returns sorted list of available basins that have zero missing q_obs in the specified year."""
        key = (year, lead_time)
        if key not in self._complete_basins_cache:
            src = self._obs_source()
            # The obs-only frame holds a single lead; completeness is lead-independent for q_obs.
            lt_arg = lead_time if (src is self.df_ts) else None
            complete_set = get_complete_streamflow_basins(src, year=year, lead_time=lt_arg)
            lower_complete = {b.lower() for b in complete_set}
            filtered = [b for b in self.available_basins if b in complete_set or b.lower() in lower_complete]
            self._complete_basins_cache[key] = filtered
        return self._complete_basins_cache[key]

    def _resolve_global_best_with_timeseries(self) -> Optional[str]:
        """Returns the highest-ranked configuration that also has an ingested forecast timeseries."""
        if self._resolve_ts_config(self.global_best_cfg):
            return self.global_best_cfg
        if get_active() is not None:  # never swap in a different model under the Global-Best label
            print(f"[SELECTION] Global-Best {self.global_best_cfg!r} has no forecast timeseries; hydrograph omitted.")
            return None
        if self.df_eval.empty or "DA NSE" not in self.df_eval.columns:
            return None

        ref = self.df_eval[
            (self.df_eval["Lead Time (Days)"] == self.ref_lead)
            & (self.df_eval["Config ID"] != "Per-Basin Best DA")
        ]
        if ref.empty:
            return None

        ref = ref[~ref["Config ID"].apply(is_rho_config)]
        ranked = ref.groupby("Config ID")["DA NSE"].mean().sort_values(ascending=False)
        for cfg in ranked.index:
            if self._resolve_ts_config(cfg):
                return str(cfg)
        return None

    def _resolve_rho_cfgs(self) -> Tuple[Optional[str], Optional[str]]:
        """(Global-Best fixed-rho config, Per-Basin rho config) that have hydrographs; None when absent."""
        if self.df_eval.empty or "Config ID" not in self.df_eval.columns:
            return None, None
        cfgs = [c for c in self.df_eval["Config ID"].dropna().unique() if is_rho_config(c) and self._resolve_ts_config(c)]
        fixed = [c for c in cfgs if not is_per_basin_rho_config(c)]
        gbr = get_global_best_config(self.df_eval[self.df_eval["Config ID"].isin(fixed)], lead_time=self.ref_lead,
                                     exclude_rho=False) if fixed else None
        pbr = next((c for c in cfgs if is_per_basin_rho_config(c)), None)
        return (gbr or None), pbr

    @staticmethod
    def _normalize_cfg_name(cfg_id: Optional[str]) -> str:
        """Normalizes a configuration identifier so evaluation and timeseries frames can be matched."""
        if cfg_id is None:
            return ""
        name = str(cfg_id).strip().lower()
        name = re.sub(r"_stat[0-9e.+-]+$", "", name)
        if name.startswith("embedded_"):
            name = name[len("embedded_"):]
        return name

    def _resolve_ts_config(self, cfg_id: Optional[str]) -> Optional[str]:
        """Maps an evaluation configuration ID onto the equivalent ID present in the timeseries frame."""
        if not cfg_id:
            return None
        if not self.df_ts.empty and "Config ID" in self.df_ts.columns:
            if (self.df_ts["Config ID"] == cfg_id).any():
                return cfg_id
        return self._ts_cfg_lookup.get(self._normalize_cfg_name(cfg_id))

    def _basin_best_config(self, basin_id: str, lead_time: int) -> Optional[str]:
        """The catchment's Per-Basin DA config (notebook-wide selection; same at every lead) if it has a timeseries."""
        sel = get_active()
        if sel is not None:
            cfg = sel.per_basin_da.get(str(basin_id))
            return str(cfg) if cfg and self._resolve_ts_config(cfg) else None
        if self.df_eval.empty or "DA NSE" not in self.df_eval.columns:
            return None
        sub = self.df_eval[
            (self.df_eval["Basin ID"] == basin_id)
            & (self.df_eval["Lead Time (Days)"] == lead_time)
            & (self.df_eval["Config ID"] != "Per-Basin Best DA")
        ].dropna(subset=["DA NSE"])
        sub = sub[~sub["Config ID"].apply(is_rho_config)]
        if sub.empty:
            return None

        for cfg in sub.sort_values("DA NSE", ascending=False)["Config ID"]:
            if self._resolve_ts_config(cfg):
                return str(cfg)
        return None

    def _select_metric_row(self, df: pd.DataFrame, prefer_cfg: Optional[str] = None) -> pd.DataFrame:
        """Selects a single representative metric row that actually carries populated DA metrics."""
        if df.empty:
            return df

        for cfg in [c for c in [prefer_cfg, "Per-Basin Best DA", self.global_best_cfg] if c]:
            sub = df[df["Config ID"] == cfg]
            if sub.empty:
                continue
            sub_valid = sub.dropna(subset=["DA NSE"]) if "DA NSE" in sub.columns else sub
            return sub_valid.head(1) if not sub_valid.empty else sub.head(1)

        if "DA NSE" in df.columns:
            non_null = df.dropna(subset=["DA NSE"])
            if not non_null.empty:
                return non_null.head(1)
        return df.head(1)

    @staticmethod
    def _first_valid_metric(df: pd.DataFrame, column: str) -> float:
        """Returns the first non-null value of `column`, tolerating configurations with null metrics."""
        if df.empty or column not in df.columns:
            return np.nan
        valid = df[column].dropna()
        return valid.iloc[0] if not valid.empty else np.nan

    def _build_hydrograph_series(self, basin_id: str, year: int, lead_time: int) -> Tuple[dict, Dict[str, Optional[str]]]:
        """Builds the fixed hydrograph series for one catchment at one forecast horizon."""
        empty = dict(date=[], q_obs=[], q_base=[], q_global=[], q_per_basin=[], q_rho_global=[], q_rho_per_basin=[])
        no_cfgs: Dict[str, Optional[str]] = {"q_global": None, "q_per_basin": None, "q_rho_global": None, "q_rho_per_basin": None}
        self._ensure_basin_ts(basin_id)
        if self.df_ts.empty or "Basin ID" not in self.df_ts.columns:
            return empty, no_cfgs

        sub = self.df_ts[self.df_ts["Basin ID"] == basin_id]
        if sub.empty or "Valid Date" not in sub.columns:
            return empty, no_cfgs

        yr_sub = sub[sub["Valid Date"].dt.year == year]
        if yr_sub.empty:
            yr_sub = sub

        lead_sub = yr_sub[yr_sub["Lead Time (Days)"] == lead_time]
        if lead_sub.empty:
            lead_sub = yr_sub[yr_sub["Lead Time (Days)"] == self.ref_lead]
        if lead_sub.empty:
            return empty, no_cfgs

        shared = lead_sub.drop_duplicates(subset=["Valid Date"]).sort_values("Valid Date")
        dates = shared["Valid Date"]

        def _cfg_series(cfg_id: Optional[str]) -> np.ndarray:
            resolved = self._resolve_ts_config(cfg_id)
            if not resolved or "Config ID" not in lead_sub.columns or "q_da" not in lead_sub.columns:
                return np.full(len(dates), np.nan)
            cfg = lead_sub[lead_sub["Config ID"] == resolved]
            if cfg.empty:
                return np.full(len(dates), np.nan)
            return cfg.drop_duplicates(subset=["Valid Date"]).set_index("Valid Date")["q_da"].reindex(dates).values

        per_basin_cfg: Optional[str] = "Per-Basin Best DA"
        q_per_basin = _cfg_series(per_basin_cfg)
        if np.isnan(q_per_basin).all():
            per_basin_cfg = self._basin_best_config(basin_id, lead_time)
            q_per_basin = _cfg_series(per_basin_cfg)
            if np.isnan(q_per_basin).all():
                per_basin_cfg = None

        q_global = _cfg_series(self.global_best_plot_cfg)
        global_cfg = self.global_best_plot_cfg if not np.isnan(q_global).all() else None
        q_rho_global = _cfg_series(self.global_best_rho_cfg)
        rho_global_cfg = self.global_best_rho_cfg if not np.isnan(q_rho_global).all() else None
        q_rho_pb = _cfg_series(self.per_basin_rho_cfg)
        rho_pb_cfg = self.per_basin_rho_cfg if not np.isnan(q_rho_pb).all() else None

        data = dict(
            date=dates.values,
            q_obs=shared["q_obs"].values if "q_obs" in shared.columns else np.full(len(dates), np.nan),
            q_base=shared["q_base"].values if "q_base" in shared.columns else np.full(len(dates), np.nan),
            q_global=q_global,
            q_per_basin=q_per_basin,
            q_rho_global=q_rho_global,
            q_rho_per_basin=q_rho_pb,
        )
        return data, {"q_global": global_cfg, "q_per_basin": per_basin_cfg,
                      "q_rho_global": rho_global_cfg, "q_rho_per_basin": rho_pb_cfg}

    def _render_info_card_html(self, basin_id: str, lead_time: Optional[int] = None) -> str:
        """Constructs styled HTML diagnostic card for the selected catchment and forecast horizon."""
        ref_l = lead_time if lead_time is not None else self.ref_lead
        b_eval = self.df_eval[self.df_eval["Basin ID"] == basin_id]
        b_lead = b_eval[b_eval["Lead Time (Days)"] == ref_l]
        if b_lead.empty and not b_eval.empty:
            b_lead = b_eval[b_eval["Lead Time (Days)"] == self.ref_lead]

        area = "N/A"
        elev = "N/A"
        slope = "N/A"
        aridity = "N/A"
        snow = "N/A"
        country = "N/A"
        lat = "N/A"
        lon = "N/A"

        if not b_lead.empty:
            row = self._select_metric_row(b_lead).iloc[0]
            area = f"{row['area']:,.1f} km²" if "area" in row and pd.notna(row["area"]) else "N/A"
            elev = f"{row['ele_mt_sav']:.0f} m" if "ele_mt_sav" in row and pd.notna(row["ele_mt_sav"]) else "N/A"
            slope = f"{row['slp_dg_sav']:.1f}°" if "slp_dg_sav" in row and pd.notna(row["slp_dg_sav"]) else "N/A"
            aridity = f"{row['unep_aridity_index']:.2f}" if "unep_aridity_index" in row and pd.notna(row["unep_aridity_index"]) else "N/A"
            snow = f"{row['frac_snow']:.2f}" if "frac_snow" in row and pd.notna(row["frac_snow"]) else "N/A"
            country = str(row.get("country", "Unknown"))
            lat = f"{row['lat']:.2f}°" if "lat" in row and pd.notna(row["lat"]) else "N/A"
            lon = f"{row['lon']:.2f}°" if "lon" in row and pd.notna(row["lon"]) else "N/A"

            base_nse = row.get("Base NSE", np.nan)
            if pd.isna(base_nse):
                base_nse = self._first_valid_metric(b_lead, "Base NSE")
            base_kge = row.get("Base KGE", np.nan)
            if pd.isna(base_kge):
                base_kge = self._first_valid_metric(b_lead, "Base KGE")
            da_nse = row.get("DA NSE", np.nan)
            delta_nse = row.get("NSE Delta", np.nan)
            skill_nse = row.get("NSE Skill Score", np.nan)
            if pd.isna(skill_nse) and pd.notna(da_nse) and pd.notna(base_nse) and abs(1.0 - base_nse) > 1e-9:
                skill_nse = 1.0 - (1.0 - da_nse) / (1.0 - base_nse)
            da_kge = row.get("DA KGE", np.nan)
            best_cfg = str(row.get("Config ID", "Unknown"))
        else:
            base_nse, da_nse, delta_nse, skill_nse, base_kge, da_kge, best_cfg = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "N/A"

        delta_color = "#174ea6" if pd.notna(delta_nse) and delta_nse >= 0 else "#c5221f"
        skill_str = f", Skill=<b>{units.fmt_ss(skill_nse)}</b>" if pd.notna(skill_nse) else ""

        kg_str = "N/A"
        if "kg_code" in b_eval.columns and not b_eval.empty:
            kg_row = b_eval.iloc[0]
            code = kg_row.get("kg_code")
            if pd.notna(code) and int(code) in kg.KG_CLASSES:
                sym, desc, _ = kg.KG_CLASSES[int(code)]
                frac = kg_row.get("kg_frac")
                frac_s = f", {frac * 100:.0f}% of area" if pd.notna(frac) else ""
                kg_str = f"{sym}</b> ({desc}{frac_s})<b>"

        html = f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 12px 18px; margin-bottom: 12px; font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e8eaed; padding-bottom: 8px; margin-bottom: 10px;">
                <span style="font-size: 16px; font-weight: bold; color: #202124;">Catchment: <span style="color: #1a73e8;">{basin_id}</span> ({country})</span>
                <span style="font-size: 12px; color: #5f6368;">Coordinates: {lat}, {lon}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1.3fr; gap: 15px; font-size: 12px;">
                <div>
                    <b style="color: #3c4043;">Physical Morphology:</b><br/>
                    • Area: <b>{area}</b><br/>
                    • Elevation: <b>{elev}</b><br/>
                    • Slope: <b>{slope}</b>
                </div>
                <div>
                    <b style="color: #3c4043;">Hydro-Climatic Regime:</b><br/>
                    • Aridity Index (P/PET): <b>{aridity}</b><br/>
                    • Snow Fraction: <b>{snow}</b><br/>
                    • Köppen-Geiger: <b>{kg_str}</b>
                </div>
                <div style="background-color: #ffffff; padding: 8px 12px; border-radius: 6px; border: 1px solid #e8eaed;">
                    <b style="color: #3c4043;">Operational Benchmark (Lead {ref_l} Day{'s' if ref_l > 1 else ''}):</b><br/>
                    • Baseline: NSE=<b>{base_nse:.3f}</b>, KGE=<b>{base_kge:.3f}</b><br/>
                    • DA Model: NSE=<b>{da_nse:.3f}</b> (<span style="color: {delta_color}; font-weight: bold;">{delta_nse:+.3f}</span>{skill_str}), KGE=<b>{da_kge:.3f}</b><br/>
                    • Best Config: <code style="background-color: #f1f3f4; padding: 2px 4px; border-radius: 4px;">{best_cfg}</code>
                </div>
            </div>
        </div>
        """
        return html

    def _render_plots(
        self,
        basin_id: str,
        year: int,
        show_star: bool = True,
        lead_time: Optional[int] = None,
        map_config: str = "global_best",
        color_by: str = "delta",
        vmin: float = -0.2,
        vmax: float = 0.2,
        filter_complete_streamflow: bool = False,
        color_mode: str = "binned",
        bin_thresholds="0.01, 0.05",
        marker_size: float = 10.0,
        region: str = "global",
        koppen: Optional[str] = None,
        koppen_filter=None,
        koppen_alpha: Optional[float] = None,
    ) -> plt.Figure:
        """Renders linked 3-panel figure with countries underlay, 5-bin discrete or continuous map
        scale, and KGE/NSE in the hydrograph legend.
        """
        koppen = self.default_koppen if koppen is None else str(koppen).lower()
        koppen_filter = self.default_koppen_filter if koppen_filter is None else koppen_filter
        koppen_alpha = self.default_koppen_alpha if koppen_alpha is None else float(koppen_alpha)
        curr_lead = lead_time if lead_time is not None and lead_time in self.lead_times else self.ref_lead
        fig = plt.figure(figsize=(20, 9.2))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.18], height_ratios=[1.28, 0.72], hspace=0.28, wspace=0.16)

        ax_map = fig.add_subplot(gs[:, 0])
        ax_hydro = fig.add_subplot(gs[0, 1])
        ax_decay = fig.add_subplot(gs[1, 1])

        # -------------------------------------------------------------
        # 1. Geographic Map (Left Panel) with Country Boundaries
        # -------------------------------------------------------------
        if self.world_gdf is not None:
            self.world_gdf.plot(ax=ax_map, color="#f3f5f7", edgecolor="#bdc1c6", linewidth=0.65, zorder=1)

        target_cfg = "Per-Basin Best DA" if map_config == "per_basin_best" else self.global_best_cfg
        cfg_title_label = "Per-Basin Oracle" if map_config == "per_basin_best" else "Global Best DA"

        c_norm_key = str(color_by).strip().lower()
        is_skill_mode = c_norm_key in ("skill_nse", "nse_skill", "nse_skill_score", "skill_score", "ss_nse", "skill")

        lead_map_data = self.df_eval[self.df_eval["Lead Time (Days)"] == curr_lead].copy()
        n_map_basins = 0
        if not lead_map_data.empty and "lat" in lead_map_data.columns and "lon" in lead_map_data.columns:
            best_lead_data = lead_map_data[lead_map_data["Config ID"] == target_cfg]
            if best_lead_data.empty:
                best_lead_data = lead_map_data[lead_map_data["Config ID"] == "Per-Basin Best DA"]
            if best_lead_data.empty:
                best_lead_data = lead_map_data.dropna(subset=["NSE Delta"])
            if best_lead_data.empty:
                best_lead_data = lead_map_data
            valid_coords = best_lead_data.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Basin ID"])
            if self.available_basins:
                valid_coords = valid_coords[valid_coords["Basin ID"].astype(str).isin(self.available_basins)]
            if filter_complete_streamflow:
                complete_b = set(self.get_complete_basins_for_year(year, curr_lead))
                valid_coords = valid_coords[valid_coords["Basin ID"].astype(str).isin(complete_b)]
            valid_coords = _apply_kg_filter(valid_coords, koppen_filter)
            _draw_kg_layer(ax_map, valid_coords, koppen, koppen_filter, koppen_alpha, self.world_gdf)

            # If a continental region zoom is active, report N inside that region
            reg_key = str(region).strip().lower()
            if reg_key in REGION_EXTENTS:
                (rx0, rx1), (ry0, ry1) = REGION_EXTENTS[reg_key]
                in_box = (
                    (valid_coords["lon"] >= rx0) & (valid_coords["lon"] <= rx1)
                    & (valid_coords["lat"] >= ry0) & (valid_coords["lat"] <= ry1)
                )
                n_map_basins = int(in_box.sum())
            else:
                n_map_basins = len(valid_coords)

            lons = valid_coords["lon"].values
            lats = valid_coords["lat"].values

            if c_norm_key in ("baseline", "da"):
                col_name = "Base NSE" if c_norm_key == "baseline" else "DA NSE"
                cbar_lbl = "Open-Loop Baseline NSE" if c_norm_key == "baseline" else f"Lead t+{curr_lead} {cfg_title_label} NSE"
                vals = valid_coords[col_name].values if col_name in valid_coords.columns else np.full(len(valid_coords), np.nan)
                _render_map_scatter_and_colorbar(
                    ax_map, lons, lats, vals,
                    cbar_label=cbar_lbl,
                    color_mode=color_mode,
                    bin_thresholds=bin_thresholds,
                    vmin=vmin, vmax=vmax,
                    marker_size=marker_size,
                    is_diverging=False,
                    cbar_shrink=0.82, cbar_pad=_cbar_pad_frac(ax_map, 1.0, True),
                )
            elif is_skill_mode:
                skills = valid_coords["NSE Skill Score"].values if "NSE Skill Score" in valid_coords.columns else np.full(len(valid_coords), np.nan)
                _render_map_scatter_and_colorbar(
                    ax_map, lons, lats, skills,
                    cbar_label=f"Lead t+{curr_lead} NSE Skill Score [{cfg_title_label}]",
                    color_mode=color_mode,
                    bin_thresholds=bin_thresholds,
                    vmin=vmin, vmax=vmax,
                    marker_size=marker_size,
                    is_diverging=True,
                    cbar_shrink=0.82, cbar_pad=_cbar_pad_frac(ax_map, 1.0, True),
                    ss_units=True,
                )
            else:
                deltas = valid_coords["NSE Delta"].values
                _render_map_scatter_and_colorbar(
                    ax_map, lons, lats, deltas,
                    cbar_label=f"Lead t+{curr_lead} ΔNSE [{cfg_title_label}]",
                    color_mode=color_mode,
                    bin_thresholds=bin_thresholds,
                    vmin=vmin, vmax=vmax,
                    marker_size=marker_size,
                    is_diverging=True,
                    cbar_shrink=0.82, cbar_pad=_cbar_pad_frac(ax_map, 1.0, True),
                )

            # Highlight selected basin with a prominent star on top of all layers
            if show_star:
                sel_b = valid_coords[valid_coords["Basin ID"] == basin_id]
                if not sel_b.empty:
                    b_lat = sel_b["lat"].iloc[0]
                    b_lon = sel_b["lon"].iloc[0]
                    ax_map.scatter(
                        [b_lon], [b_lat],
                        color="#fbbc04", edgecolors="black",
                        s=max(220.0, marker_size * 22.0), marker="*",
                        linewidth=1.5, zorder=20, label=f"Selected ({basin_id})"
                    )
                    ax_map.legend(loc="upper right", frameon=True, fontsize=9.5, framealpha=0.92)

            _apply_region_extent(ax_map, lons, lats, region=region)

        filter_tag = f" [Complete q_obs {year}]" if filter_complete_streamflow else ""
        kg_lbl = _kg_filter_label(koppen_filter)
        if kg_lbl:
            filter_tag += f" [KG: {kg_lbl}]"
        ax_map.set_title(f"Catchment Map [{cfg_title_label}] (Lead t+{curr_lead}, N={n_map_basins:,}){filter_tag}", fontsize=12, fontweight="bold", pad=8)
        ax_map.set_xlabel("Longitude", fontsize=10)
        ax_map.set_ylabel("Latitude", fontsize=10)
        ax_map.grid(True, linestyle="--", alpha=0.35)

        # -------------------------------------------------------------
        # 2. 1-Year Rolling Hydrograph with Basin ID, KGE & NSE in Title/Legend
        # -------------------------------------------------------------
        b_eval = self.df_eval[self.df_eval["Basin ID"] == basin_id].copy()
        b_lead = b_eval[b_eval["Lead Time (Days)"] == curr_lead]
        if b_lead.empty and not b_eval.empty:
            b_lead = b_eval[b_eval["Lead Time (Days)"] == self.ref_lead]

        base_nse = self._first_valid_metric(b_lead, "Base NSE")
        base_kge = self._first_valid_metric(b_lead, "Base KGE")

        series, series_cfgs = self._build_hydrograph_series(basin_id, year, curr_lead)
        if len(series["date"]) > 0:
            dates = series["date"]

            # 1. Observed streamflow
            ax_hydro.plot(dates, series["q_obs"], color="black", linewidth=1.8, label="Observed Streamflow (q_obs)")

            # 2. Baseline Open-Loop with NSE & KGE in Legend
            base_lbl = f"Baseline Open-Loop (NSE: {base_nse:.2f}, KGE: {base_kge:.2f})" if pd.notna(base_nse) else "Baseline Open-Loop"
            ax_hydro.plot(dates, series["q_base"], color="#80868b", linestyle="--", linewidth=1.5, label=base_lbl)

            # 3. Deterministic DA benchmarks at the selected lead time: Global Best DA and Per-Basin Best DA
            da_specs = [
                ("q_global", "Global Best DA", "#f9ab00"),
                ("q_per_basin", "Per-Basin Best DA", "#1a73e8"),
                ("q_rho_global", "Global-Best Rho", "#e37400"),
                ("q_rho_per_basin", "Per-Basin Rho", "#c5221f"),
            ]
            for series_key, cfg_label, color in da_specs:
                values = series[series_key]
                cfg_id = series_cfgs.get(series_key)
                if cfg_id is None or len(values) == 0 or np.isnan(values).all():
                    continue

                cfg_row = self._select_metric_row(b_lead, prefer_cfg=cfg_id)
                cfg_nse = cfg_row["DA NSE"].iloc[0] if not cfg_row.empty and "DA NSE" in cfg_row else np.nan
                cfg_kge = cfg_row["DA KGE"].iloc[0] if not cfg_row.empty and "DA KGE" in cfg_row else np.nan
                lbl = f"{cfg_label} t+{curr_lead} (NSE: {cfg_nse:.2f}, KGE: {cfg_kge:.2f})" if pd.notna(cfg_nse) else f"{cfg_label} t+{curr_lead}"
                ls = (0, (4, 2)) if "Rho" in cfg_label else "-"
                ax_hydro.plot(dates, values, color=color, linewidth=1.6 if "Rho" in cfg_label else 2.0, linestyle=ls, label=lbl)

            ax_hydro.set_title(f"Daily Forecast Hydrograph — {basin_id} (Year {year}, Lead t+{curr_lead})", fontsize=12, fontweight="bold", pad=8)
            ax_hydro.set_ylabel("Streamflow (mm/day)", fontsize=10)
            ax_hydro.grid(True, linestyle="--", alpha=0.4)
            ax_hydro.legend(loc="upper right", frameon=True, fontsize=9)
        else:
            ax_hydro.text(0.5, 0.5, f"No timeseries records for {basin_id}", ha="center", va="center")
            ax_hydro.axis("off")

        # -------------------------------------------------------------
        # 3. Basin-Specific Leadtime Skill Persistence (Bottom Right)
        # -------------------------------------------------------------
        if not b_eval.empty:
            b_cfg = b_eval[b_eval["Config ID"] == "Per-Basin Best DA"]
            if b_cfg.empty:
                b_cfg = b_eval[b_eval["Config ID"] == self.global_best_cfg]
            if b_cfg.empty:
                b_cfg = b_eval

            lt_vals_int = []
            lt_pts = []
            base_pts = []
            da_pts = []
            bar_pts = []
            bar_col_name = "NSE Skill Score" if is_skill_mode else "NSE Delta"
            bar_lbl_name = "NSE Skill Score" if is_skill_mode else "ΔNSE"

            for lt in self.lead_times:
                row = b_cfg[b_cfg["Lead Time (Days)"] == lt]
                if not row.empty:
                    lt_vals_int.append(int(lt))
                    lt_pts.append(f"t+{lt}")
                    base_pts.append(row["Base NSE"].iloc[0] if "Base NSE" in row else np.nan)
                    da_pts.append(row["DA NSE"].iloc[0] if "DA NSE" in row else np.nan)
                    bar_pts.append(row[bar_col_name].iloc[0] if bar_col_name in row else np.nan)

            x_idx = np.arange(len(lt_pts))
            edges_5 = _parse_bin_thresholds(bin_thresholds)
            bar_bins, _, _ = _classify_into_5_bins(np.asarray(bar_pts, dtype=float), edges_5)
            bar_colors = [DISCRETE_5BIN_COLORS[idx] for idx in bar_bins]

            # Highlight active horizon (curr_lead) so all 3 panels are visually linked
            if curr_lead in lt_vals_int:
                active_i = lt_vals_int.index(curr_lead)
                ax_decay.axvspan(active_i - 0.34, active_i + 0.34, color="#e8f0fe", alpha=0.8, zorder=1, label=f"Active Horizon (t+{curr_lead})")

            ax_decay.plot(x_idx, base_pts, marker="o", color="#5f6368", linewidth=2.0, zorder=4, label="Baseline NSE")
            ax_decay.plot(x_idx, da_pts, marker="s", color="#1a73e8", linewidth=2.2, zorder=5, label="DA NSE")
            ax_decay.bar(x_idx, bar_pts, width=0.36, alpha=0.75, color=bar_colors, edgecolor="#3c4043", linewidth=0.5, zorder=3, label=bar_lbl_name)

            ax_decay.set_xticks(x_idx)
            ax_decay.set_xticklabels(lt_pts, fontsize=10, fontweight="bold")
            ax_decay.set_xlabel("Forecast Horizon", fontsize=10)
            ax_decay.set_ylabel(f"NSE / {bar_lbl_name}", fontsize=10)
            first_lt = lt_pts[0] if lt_pts else "t+1"
            last_lt = lt_pts[-1] if lt_pts else "t+7"
            ax_decay.set_title(f"Basin Skill Persistence Across Horizons ({first_lt} to {last_lt}) — {basin_id}", fontsize=11, fontweight="bold", pad=6)
            ax_decay.axhline(0.0, color="black", linestyle="--", alpha=0.4)
            ax_decay.grid(True, linestyle="--", alpha=0.4)
            ax_decay.legend(loc="best", frameon=True, fontsize=8.5, ncol=2)
        else:
            ax_decay.text(0.5, 0.5, "No evaluation metrics for basin", ha="center", va="center")
            ax_decay.axis("off")

        return fig

    def make_bokeh_app(self, default_year: Optional[int] = None):
        """Constructs an interactive Bokeh application where clicking map points updates hydrographs."""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, TapTool, Div, Select
        from bokeh.layouts import column, row

        yr = default_year if default_year is not None else self.default_year

        def bkapp(doc):
            lead1 = self.df_eval[self.df_eval["Lead Time (Days)"] == self.ref_lead].copy()
            best_lead_data = lead1[lead1["Config ID"] == "Per-Basin Best DA"]
            if best_lead_data.empty:
                best_lead_data = lead1.dropna(subset=["NSE Delta"])
            if best_lead_data.empty:
                best_lead_data = lead1
            valid_coords = best_lead_data.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Basin ID"])
            if self.available_basins:
                valid_coords = valid_coords[valid_coords["Basin ID"].astype(str).isin(self.available_basins)]

            if self.default_filter_complete_streamflow:
                complete_b = set(self.get_complete_basins_for_year(yr, self.ref_lead))
                valid_coords = valid_coords[valid_coords["Basin ID"].astype(str).isin(complete_b)]

            # 5-category discrete colors for Bokeh points
            edges_5 = _parse_bin_thresholds(self.default_bin_thresholds)
            metric_col_bk = "NSE Skill Score" if self.default_color_by == "skill_nse" else "NSE Delta"
            bin_idx, _, _ = _classify_into_5_bins(
                valid_coords[metric_col_bk].values if metric_col_bk in valid_coords.columns else np.zeros(len(valid_coords)),
                edges_5,
            )
            colors = [DISCRETE_5BIN_COLORS[i] for i in bin_idx]

            map_source = ColumnDataSource(data=dict(
                lon=valid_coords["lon"].values,
                lat=valid_coords["lat"].values,
                basin=valid_coords["Basin ID"].values,
                delta=[f"{d:+.3f}" if pd.notna(d) else "N/A" for d in valid_coords["NSE Delta"]],
                skill=[units.fmt_ss(s) if pd.notna(s) else "N/A" for s in valid_coords.get("NSE Skill Score", pd.Series(np.nan, index=valid_coords.index))],
                base_nse=[f"{n:.3f}" if pd.notna(n) else "N/A" for n in valid_coords["Base NSE"]],
                da_nse=[f"{n:.3f}" if pd.notna(n) else "N/A" for n in valid_coords["DA NSE"]],
                country=valid_coords.get("country", ["Unknown"]*len(valid_coords)),
                color=colors,
            ))

            active_basins = self.get_complete_basins_for_year(yr, self.ref_lead) if self.default_filter_complete_streamflow else self.available_basins
            if not active_basins:
                active_basins = self.available_basins
            init_basin = active_basins[0]
            init_row = valid_coords[valid_coords["Basin ID"] == init_basin]
            init_lon = init_row["lon"].iloc[0] if not init_row.empty else 0.0
            init_lat = init_row["lat"].iloc[0] if not init_row.empty else 0.0

            star_source = ColumnDataSource(data=dict(lon=[init_lon], lat=[init_lat]))

            p_map = figure(
                title="Catchment Map (Click any point to view hydrograph)",
                width=540,
                height=420,
                tools="pan,wheel_zoom,box_zoom,tap,reset",
                match_aspect=True,
            )

            if self.world_gdf is not None:
                xs, ys = [], []
                for geom in self.world_gdf.geometry:
                    if geom.geom_type == "Polygon":
                        x, y = geom.exterior.xy
                        xs.append(list(x))
                        ys.append(list(y))
                    elif geom.geom_type == "MultiPolygon":
                        for poly in geom.geoms:
                            x, y = poly.exterior.xy
                            xs.append(list(x))
                            ys.append(list(y))
                p_map.patches(xs, ys, fill_color="#f1f3f4", line_color="#bdc1c6", line_width=0.6)

            bk_dot_size = max(3.0, min(18.0, float(self.default_marker_size) * 0.65))
            renderer = p_map.scatter("lon", "lat", source=map_source, size=bk_dot_size, color="color", fill_alpha=0.88, line_color="#3c4043", line_width=0.3)
            p_map.scatter("lon", "lat", source=star_source, size=20, marker="star", color="#fbbc04", line_color="black", line_width=1.5)

            hover = HoverTool(renderers=[renderer], tooltips=[
                ("Catchment", "@basin"),
                ("Country", "@country"),
                ("ΔNSE", "@delta"),
                ("NSE Skill Score", "@skill"),
                ("Base NSE", "@base_nse"),
                ("DA NSE", "@da_nse"),
            ])
            p_map.add_tools(hover)
            p_map.xaxis.axis_label = "Longitude"
            p_map.yaxis.axis_label = "Latitude"

            info_div = Div(text=self._render_info_card_html(init_basin), width=1050)

            ts_source = ColumnDataSource(data=self._build_hydrograph_series(init_basin, yr, self.ref_lead)[0])

            p_hydro = figure(title=f"Daily Forecast Hydrograph — {init_basin} (Year {yr})", width=530, height=420, x_axis_type="datetime", tools="pan,wheel_zoom,box_zoom,reset")
            p_hydro.line("date", "q_obs", source=ts_source, color="black", line_width=2.0, legend_label="Observed (q_obs)")
            p_hydro.line("date", "q_base", source=ts_source, color="#80868b", line_dash="dashed", line_width=1.8, legend_label="Baseline Open-Loop")
            p_hydro.line("date", "q_global", source=ts_source, color="#f9ab00", line_width=2.0, legend_label="Global Best DA")
            p_hydro.line("date", "q_per_basin", source=ts_source, color="#1a73e8", line_width=2.0, legend_label="Per-Basin Best DA")
            if self.global_best_rho_cfg:
                p_hydro.line("date", "q_rho_global", source=ts_source, color="#e37400", line_width=1.6, line_dash="dashed",
                             legend_label=f"Global-Best Rho ({self.global_best_rho_cfg.replace('AR1_postprocess_', '')})")
            if self.per_basin_rho_cfg:
                p_hydro.line("date", "q_rho_per_basin", source=ts_source, color="#c5221f", line_width=1.6, line_dash="dotted",
                             legend_label="Per-Basin Rho")
            p_hydro.yaxis.axis_label = "Streamflow (mm/day)"
            p_hydro.legend.location = "top_right"
            p_hydro.legend.click_policy = "hide"

            basin_select = Select(title="Catchment:", value=init_basin, options=active_basins, width=350)
            lead_select = Select(title="Leadtime:", value=str(self.ref_lead), options=[str(l) for l in self.lead_times], width=130)

            def update_basin_view(selected_basin):
                sel_lead = int(lead_select.value) if lead_select.value.isdigit() else self.ref_lead
                info_div.text = self._render_info_card_html(selected_basin, lead_time=sel_lead)
                b_row = valid_coords[valid_coords["Basin ID"] == selected_basin]
                if not b_row.empty:
                    star_source.data = dict(lon=[b_row["lon"].iloc[0]], lat=[b_row["lat"].iloc[0]])
                ts_source.data = self._build_hydrograph_series(selected_basin, yr, sel_lead)[0]
                p_hydro.title.text = f"Daily Forecast Hydrograph — {selected_basin} (Year {yr}, Lead t+{sel_lead})"

            def on_map_tap(attr, old, new):
                if new:
                    idx = new[0]
                    clicked_b = map_source.data["basin"][idx]
                    basin_select.value = clicked_b
                    update_basin_view(clicked_b)

            def on_dropdown_change(attr, old, new):
                if new:
                    update_basin_view(basin_select.value)

            map_source.selected.on_change("indices", on_map_tap)
            basin_select.on_change("value", on_dropdown_change)
            lead_select.on_change("value", on_dropdown_change)

            controls_row = row(basin_select, lead_select)
            layout = column(controls_row, info_div, row(p_map, p_hydro))
            doc.add_root(layout)

        return bkapp

    def launch(
        self,
        interactive_map: bool = True,
        year: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        filter_complete_streamflow: Optional[bool] = None,
        color_by: Optional[str] = None,
        color_mode: Optional[str] = None,
        bin_thresholds=None,
        marker_size: Optional[float] = None,
        region: Optional[str] = None,
        koppen: Optional[str] = None,
        koppen_filter=None,
        koppen_alpha: Optional[float] = None,
        font_scale: float = 1.5,
        view_mode: str = "inspector_global",
    ) -> None:
        """Launches dashboard. If interactive_map=True, launches Bokeh Clickable Map with TapTool."""
        yr = year if year is not None else self.default_year

        if interactive_map:
            try:
                from bokeh.io import output_notebook, show
                output_notebook()
                bkapp = self.make_bokeh_app(default_year=yr)
                show(bkapp)
                return
            except Exception as e:
                print(f"[INFO] Interactive Bokeh server launch fell back to ipywidgets: {e}")

        self.launch_ipywidgets(
            year=yr,
            vmin=vmin if vmin is not None else self.default_vmin,
            vmax=vmax if vmax is not None else self.default_vmax,
            filter_complete_streamflow=(
                filter_complete_streamflow
                if filter_complete_streamflow is not None
                else self.default_filter_complete_streamflow
            ),
            color_by=color_by if color_by is not None else self.default_color_by,
            color_mode=color_mode if color_mode is not None else self.default_color_mode,
            bin_thresholds=bin_thresholds if bin_thresholds is not None else self.default_bin_thresholds,
            marker_size=marker_size if marker_size is not None else self.default_marker_size,
            region=region if region is not None else self.default_region,
            koppen=koppen,
            koppen_filter=koppen_filter,
            koppen_alpha=koppen_alpha,
            font_scale=font_scale,
            view_mode=view_mode,
        )

    def launch_ipywidgets(
        self,
        year: Optional[int] = None,
        show_star: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        filter_complete_streamflow: Optional[bool] = None,
        color_by: Optional[str] = None,
        color_mode: Optional[str] = None,
        bin_thresholds=None,
        marker_size: Optional[float] = None,
        region: Optional[str] = None,
        koppen: Optional[str] = None,
        koppen_filter=None,
        koppen_alpha: Optional[float] = None,
        font_scale: float = 1.5,
        view_mode: str = "inspector_global",
    ) -> None:
        """Launches standard ipywidgets application in Jupyter Notebook.

        ``view_mode``: 'inspector_global', 'inspector_oracle', 'map_compare', 'map_pub_global', 'map_pub_oracle'
        or 'map_grid' (paper 2x2: Global vs Per-Basin DA at two leads). ``font_scale`` sizes text on the
        publication / paper maps.
        """
        import ipywidgets as widgets
        init_koppen = str(koppen if koppen is not None else self.default_koppen).lower()
        if init_koppen not in kg.KOPPEN_MODES:
            init_koppen = "off"
        init_kg_filter = koppen_filter if koppen_filter is not None else self.default_koppen_filter
        init_kg_alpha = float(koppen_alpha if koppen_alpha is not None else self.default_koppen_alpha)
        yr = year if year is not None else self.default_year
        init_vmin = float(vmin if vmin is not None else self.default_vmin)
        init_vmax = float(vmax if vmax is not None else self.default_vmax)
        init_filter_q = bool(
            filter_complete_streamflow
            if filter_complete_streamflow is not None
            else self.default_filter_complete_streamflow
        )
        init_color_by = str(color_by if color_by is not None else self.default_color_by).strip().lower()
        if init_color_by in ("nse_skill", "nse_skill_score", "skill_score", "ss_nse", "skill"):
            init_color_by = "skill_nse"
        elif init_color_by not in ("delta", "skill_nse", "da", "baseline"):
            init_color_by = "delta"

        init_color_mode = str(color_mode if color_mode is not None else self.default_color_mode).strip().lower()
        init_color_mode = "continuous" if init_color_mode == "continuous" else "binned"

        raw_bins = bin_thresholds if bin_thresholds is not None else self.default_bin_thresholds
        if isinstance(raw_bins, (list, tuple, np.ndarray)):
            init_bins_str = ", ".join(str(x) for x in raw_bins)
        else:
            init_bins_str = str(raw_bins) if raw_bins is not None else "0.01, 0.05"

        init_marker_size = float(marker_size if marker_size is not None else self.default_marker_size)
        init_region = str(region if region is not None else self.default_region).strip().lower()
        if init_region not in ("global", "north_america", "south_america", "europe", "australia"):
            init_region = "global"

        if not self.available_basins:
            print("[WARNING] No basins available to render interactive dashboard.")
            return

        view_mode_dropdown = widgets.Dropdown(
            options=[
                ("Inspector: Global Best DA Map + Hydrograph", "inspector_global"),
                ("Inspector: Per-Basin Oracle Map + Hydrograph", "inspector_oracle"),
                ("Comparison Map: Global Best vs Oracle (2-Panel)", "map_compare"),
                ("Publication Map: Full-Width Global Best DA", "map_pub_global"),
                ("Publication Map: Full-Width Per-Basin Oracle", "map_pub_oracle"),
                ("Paper 2×2: Global vs Per-Basin DA × two leads", "map_grid"),
            ],
            value=view_mode if view_mode in ("inspector_global", "inspector_oracle", "map_compare", "map_pub_global",
                                             "map_pub_oracle", "map_grid") else "inspector_global",
            description="Mode:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px"),
        )

        color_dropdown = widgets.Dropdown(
            options=[
                ("ΔNSE (DA - Baseline)", "delta"),
                ("NSE Skill Score [1 - (1-DA)/(1-Base)]", "skill_nse"),
                ("DA NSE (Absolute)", "da"),
                ("Open-Loop Baseline NSE", "baseline"),
            ],
            value=init_color_by,
            description="Metric:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="290px"),
        )

        scale_mode_dropdown = widgets.Dropdown(
            options=[
                ("5-Bin Discrete", "binned"),
                ("Continuous", "continuous"),
            ],
            value=init_color_mode,
            description="Scale:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="175px"),
        )

        bins_input = widgets.Text(
            value=init_bins_str,
            placeholder="e.g. 0.01, 0.05",
            description="Bin Thresholds (±):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="225px"),
        )

        vmin_input = widgets.FloatText(
            value=init_vmin,
            step=0.05,
            description="Min:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="115px"),
        )

        vmax_input = widgets.FloatText(
            value=init_vmax,
            step=0.05,
            description="Max:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="115px"),
        )

        dot_size_slider = widgets.FloatSlider(
            value=init_marker_size,
            min=2.0,
            max=40.0,
            step=1.0,
            description="Circle Size:",
            continuous_update=False,
            readout_format=".0f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="220px"),
        )

        region_dropdown = widgets.Dropdown(
            options=[
                ("Global (Cropped)", "global"),
                ("North America (CONUS+CA)", "north_america"),
                ("South America", "south_america"),
                ("Europe", "europe"),
                ("Australia", "australia"),
            ],
            value=init_region,
            description="Region:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="225px"),
        )

        years = [yr, 2017, 2018, 2019, 2020]
        yr_src = self.df_ts if (not self.df_ts.empty or self.ts_loader is None) else self._obs_source()
        if not yr_src.empty and "Valid Date" in yr_src.columns:
            avail_years = sorted(pd.to_datetime(yr_src["Valid Date"]).dt.year.dropna().unique().tolist())
            if avail_years:
                years = avail_years

        init_yr_val = yr if yr in years else years[0]

        year_dropdown = widgets.Dropdown(
            options=years,
            value=init_yr_val,
            description="Year:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="125px"),
        )

        lead_dropdown = widgets.Dropdown(
            options=[(f"Lead +{l}d (t+{l})", l) for l in self.lead_times],
            value=self.ref_lead,
            description="Horizon:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="165px"),
        )

        star_checkbox = widgets.Checkbox(
            value=show_star,
            description="Show Star (★)",
            indent=False,
            layout=widgets.Layout(width="115px"),
        )

        complete_q_checkbox = widgets.Checkbox(
            value=init_filter_q,
            description="No Missing Streamflow (q_obs) in Period",
            indent=False,
            layout=widgets.Layout(width="265px"),
        )

        init_pool = (
            self.get_complete_basins_for_year(init_yr_val, self.ref_lead)
            if init_filter_q
            else self.available_basins
        )
        if not init_pool:
            init_pool = self.available_basins

        basin_dropdown = widgets.Dropdown(
            options=init_pool,
            value=init_pool[0],
            description="Catchment:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px"),
        )

        search_box = widgets.Text(
            placeholder="Filter (e.g. 12101500)...",
            description="Search:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="210px"),
        )

        koppen_dropdown = widgets.Dropdown(
            options=[("Off", "off"), ("Main groups (A–E)", "groups"), ("All 30 classes", "classes")],
            value=init_koppen,
            description="Köppen-Geiger:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="245px"),
        )
        kg_filter_opts = [("All climates", "all")]
        if "kg_group" in self.df_eval.columns:
            kg_b = self.df_eval.drop_duplicates("Basin ID")
            kg_b = kg_b[kg_b["Basin ID"].astype(str).isin(self.available_basins)]
            g_counts = kg_b["kg_group"].value_counts()
            s_counts = kg_b["kg_symbol"].value_counts()
            for g, (name, _c) in kg.KG_GROUPS.items():
                kg_filter_opts.append((f"{g} – {name} (n={int(g_counts.get(g, 0))})", g))
            for code, (sym, desc, _rgb) in kg.KG_CLASSES.items():
                if s_counts.get(sym, 0):
                    kg_filter_opts.append((f"   {sym} – {desc} (n={int(s_counts[sym])})", sym))
        init_kg_val = init_kg_filter if isinstance(init_kg_filter, str) else ", ".join(init_kg_filter)
        init_kg_val = init_kg_val.strip() or "all"
        if init_kg_val.lower() == "all":
            init_kg_val = "all"
        elif init_kg_val not in [v for _, v in kg_filter_opts]:
            kg_filter_opts.append((init_kg_val, init_kg_val))
        kg_filter_dropdown = widgets.Dropdown(
            options=kg_filter_opts,
            value=init_kg_val,
            description="Climate filter:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px"),
        )
        kg_alpha_slider = widgets.FloatSlider(
            value=init_kg_alpha, min=0.1, max=1.0, step=0.05,
            description="KG opacity:",
            continuous_update=False, readout_format=".2f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="220px"),
        )

        font_scale_slider = widgets.FloatSlider(
            value=float(font_scale), min=0.8, max=2.5, step=0.1,
            description="Text size (maps):",
            continuous_update=False, readout_format=".1f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="250px"),
        )
        leads_avail = list(self.lead_times)
        grid_opts = [(f"t+{a} / t+{b}", f"{a},{b}") for a, b in [(1, 7), (1, 3), (1, 5), (3, 7), (1, 2)]
                     if a in leads_avail and b in leads_avail] or [(f"t+{leads_avail[0]} / t+{leads_avail[-1]}",
                                                                     f"{leads_avail[0]},{leads_avail[-1]}")]
        grid_leads_dropdown = widgets.Dropdown(
            options=grid_opts, value=grid_opts[0][1], description="2×2 rows:",
            style={"description_width": "initial"}, layout=widgets.Layout(width="180px"),
        )

        info_card_output = widgets.Output()
        plot_output = widgets.Output()

        def _get_active_basin_pool() -> List[str]:
            cur_yr = int(year_dropdown.value)
            cur_l = int(lead_dropdown.value)
            if complete_q_checkbox.value:
                pool = self.get_complete_basins_for_year(cur_yr, cur_l)
                if not pool:
                    pool = self.available_basins
            else:
                pool = self.available_basins
            query = search_box.value.strip().lower()
            if query:
                q_filtered = [b for b in pool if query in b.lower()]
                if q_filtered:
                    pool = q_filtered
            k_groups, k_syms = _parse_kg_filter(kg_filter_dropdown.value)
            if (k_groups or k_syms) and "kg_group" in self.df_eval.columns:
                kb = self.df_eval.drop_duplicates("Basin ID")
                ok = set(_apply_kg_filter(kb, kg_filter_dropdown.value)["Basin ID"].astype(str))
                k_filtered = [b for b in pool if b in ok]
                if k_filtered:
                    pool = k_filtered
            return pool

        def _sync_basin_dropdown(*args):
            pool = _get_active_basin_pool()
            if pool:
                cur_val = basin_dropdown.value
                basin_dropdown.options = pool
                if cur_val not in pool:
                    basin_dropdown.value = pool[0]

        def update_view(*args):
            _sync_basin_dropdown()
            mode = view_mode_dropdown.value
            c_by = color_dropdown.value
            c_mode = scale_mode_dropdown.value
            b_thresh = bins_input.value
            m_size = float(dot_size_slider.value)
            reg = region_dropdown.value
            basin = basin_dropdown.value
            cur_yr = int(year_dropdown.value)
            star_val = bool(star_checkbox.value)
            l_val = int(lead_dropdown.value)
            cur_vmin = float(vmin_input.value)
            cur_vmax = float(vmax_input.value)
            filt_q = bool(complete_q_checkbox.value)
            kgkw = dict(koppen=koppen_dropdown.value, koppen_filter=kg_filter_dropdown.value,
                        koppen_alpha=float(kg_alpha_slider.value))
            pubkw = dict(font_scale=float(font_scale_slider.value))

            info_card_output.clear_output()
            with info_card_output:
                if mode.startswith("inspector_"):
                    display(HTML(self._render_info_card_html(basin, lead_time=l_val)))

            plot_output.clear_output(wait=True)
            with plot_output:
                if mode == "inspector_global":
                    fig = self._render_plots(
                        basin, cur_yr, show_star=star_val, lead_time=l_val,
                        map_config="global_best", color_by=c_by,
                        vmin=cur_vmin, vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        color_mode=c_mode, bin_thresholds=b_thresh,
                        marker_size=m_size, region=reg, **kgkw,
                    )
                elif mode == "inspector_oracle":
                    fig = self._render_plots(
                        basin, cur_yr, show_star=star_val, lead_time=l_val,
                        map_config="per_basin_best", color_by=c_by,
                        vmin=cur_vmin, vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        color_mode=c_mode, bin_thresholds=b_thresh,
                        marker_size=m_size, region=reg, **kgkw,
                    )
                elif mode == "map_compare":
                    fig = plot_catchment_map(
                        self.df_eval,
                        df_meta=self.df_meta,
                        color_by=c_by,
                        show_star=star_val,
                        star_basin=basin,
                        lead_time=l_val,
                        world_gdf=self.world_gdf,
                        config_mode="both",
                        vmin=cur_vmin,
                        vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        year=cur_yr,
                        df_ts=self._obs_source() if filt_q else self.df_ts,
                        color_mode=c_mode,
                        bin_thresholds=b_thresh,
                        marker_size=m_size,
                        region=reg,
                        **kgkw,
                        **pubkw,
                    )
                elif mode == "map_pub_oracle":
                    fig = plot_catchment_map(
                        self.df_eval,
                        df_meta=self.df_meta,
                        color_by=c_by,
                        show_star=star_val,
                        star_basin=basin,
                        lead_time=l_val,
                        world_gdf=self.world_gdf,
                        config_mode="per_basin_best",
                        vmin=cur_vmin,
                        vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        year=cur_yr,
                        df_ts=self._obs_source() if filt_q else self.df_ts,
                        color_mode=c_mode,
                        bin_thresholds=b_thresh,
                        marker_size=m_size,
                        region=reg,
                        **kgkw,
                        **pubkw,
                    )
                elif mode == "map_grid":
                    fig = plot_catchment_map(
                        self.df_eval,
                        df_meta=self.df_meta,
                        color_by=c_by,
                        show_star=star_val,
                        star_basin=basin,
                        world_gdf=self.world_gdf,
                        config_mode="grid",
                        grid_leads=tuple(int(x) for x in grid_leads_dropdown.value.split(",")),
                        vmin=cur_vmin,
                        vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        year=cur_yr,
                        df_ts=self._obs_source() if filt_q else self.df_ts,
                        color_mode=c_mode,
                        bin_thresholds=b_thresh,
                        marker_size=m_size,
                        region=reg,
                        **kgkw,
                        **pubkw,
                    )
                else:
                    fig = plot_catchment_map(
                        self.df_eval,
                        df_meta=self.df_meta,
                        color_by=c_by,
                        show_star=star_val,
                        star_basin=basin,
                        lead_time=l_val,
                        world_gdf=self.world_gdf,
                        config_mode="global_best",
                        vmin=cur_vmin,
                        vmax=cur_vmax,
                        filter_complete_streamflow=filt_q,
                        year=cur_yr,
                        df_ts=self._obs_source() if filt_q else self.df_ts,
                        color_mode=c_mode,
                        bin_thresholds=b_thresh,
                        marker_size=m_size,
                        region=reg,
                        **kgkw,
                        **pubkw,
                    )
                display(fig)
                plt.close(fig)

        for w in (
            view_mode_dropdown,
            color_dropdown,
            scale_mode_dropdown,
            bins_input,
            vmin_input,
            vmax_input,
            dot_size_slider,
            region_dropdown,
            basin_dropdown,
            year_dropdown,
            lead_dropdown,
            star_checkbox,
            complete_q_checkbox,
            search_box,
            koppen_dropdown,
            kg_filter_dropdown,
            kg_alpha_slider,
            font_scale_slider,
            grid_leads_dropdown,
        ):
            w.observe(update_view, names="value")

        update_view()

        row1 = widgets.HBox(
            [view_mode_dropdown, color_dropdown, scale_mode_dropdown, bins_input, vmin_input, vmax_input],
            layout=widgets.Layout(margin="0 0 6px 0", align_items="center")
        )
        row2 = widgets.HBox(
            [dot_size_slider, region_dropdown, year_dropdown, lead_dropdown, star_checkbox, complete_q_checkbox],
            layout=widgets.Layout(margin="0 0 6px 0", align_items="center")
        )
        row3 = widgets.HBox(
            [search_box, basin_dropdown, koppen_dropdown, kg_filter_dropdown, kg_alpha_slider],
            layout=widgets.Layout(margin="0 0 10px 0", align_items="center")
        )
        row4 = widgets.HBox(
            [widgets.HTML("<b style='margin-right:6px'>Publication / paper maps:</b>"), font_scale_slider,
             grid_leads_dropdown],
            layout=widgets.Layout(margin="0 0 10px 0", align_items="center")
        )
        dashboard = widgets.VBox([row1, row2, row3, row4, info_card_output, plot_output])
        display(dashboard)


def _deg_formatter(axis: str):
    from matplotlib.ticker import FuncFormatter

    def _f(v, _pos):
        if abs(v) < 1e-9:
            return "0°" if axis == "lon" else "EQ"
        hemi = ("E" if v > 0 else "W") if axis == "lon" else ("N" if v > 0 else "S")
        return f"{abs(v):.0f}°{hemi}"
    return FuncFormatter(_f)


def _cbar_pad_frac(ax: plt.Axes, font_scale: float, has_xlabel: bool) -> float:
    """Colourbar pad (fraction of axes height) that clears the x tick labels (+ x label)."""
    pts = 10.0 * font_scale + 6.0 + ((11.0 * font_scale + 4.0) if has_xlabel else 0.0)
    ax_h_in = max(0.5, ax.get_position().height * ax.figure.get_figheight())
    return float(np.clip(pts / 72.0 / ax_h_in, 0.04, 0.35))


def plot_catchment_map(
    df_eval: pd.DataFrame,
    df_meta: Optional[pd.DataFrame] = None,
    color_by: str = "delta",
    show_star: bool = False,
    star_basin: Optional[str] = None,
    lead_time: int = 1,
    world_gdf: Optional[gpd.GeoDataFrame] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    vmin: float = -0.2,
    vmax: float = 0.2,
    config_mode: str = "global_best",
    filter_complete_streamflow: bool = False,
    year: Optional[int] = None,
    df_ts: Optional[pd.DataFrame] = None,
    color_mode: str = "binned",
    bin_thresholds="0.01, 0.05",
    marker_size: float = 12.0,
    region: str = "global",
    koppen: str = "off",
    koppen_filter="all",
    koppen_alpha: float = 0.4,
    koppen_df: Optional[pd.DataFrame] = None,
    font_scale: float = 1.5,
    grid_leads: Sequence[int] = (1, 7),
    panel_letters: bool = True,
) -> plt.Figure:
    """Renders a publication-ready catchment map.

    Args:
        df_eval: Evaluation DataFrame with metrics and lat/lon.
        df_meta: Optional metadata DataFrame containing lat/lon.
        color_by: 'delta' (colored by Lead ΔNSE),
                  'skill_nse' (colored by NSE Skill Score [1 - (1-DA)/(1-Base)]),
                  'baseline' (colored by Base NSE),
                  'da' (colored by DA NSE),
                  or 'uniform' (clean uniform blue dots for publication station maps).
        show_star: If True, stars a specific basin.
        star_basin: Specific basin ID to star if show_star is True.
        lead_time: Forecast horizon for metric display (default 1; ignored for config_mode='grid').
        world_gdf: Optional preloaded world GeoDataFrame.
        figsize: Figure size (default depends on the layout).
        title: Custom title (single panel) / figure title (multi-panel). '' disables the figure title.
        vmin: Lower bound for continuous colorbar scale (default -0.2).
        vmax: Upper bound for continuous colorbar scale (default 0.2).
        config_mode: 'global_best' (default), 'per_basin_best' (Oracle), 'both' (1x2 side-by-side) or
            'grid' (2x2: rows = ``grid_leads``, columns = Global-Best DA | Per-Basin DA).
        filter_complete_streamflow: If True, filters displayed basins to those with zero missing q_obs in `year`.
        year: Evaluation year for complete streamflow filtering.
        df_ts: Forecast timeseries DataFrame used when filter_complete_streamflow=True.
        color_mode: 'binned' (default 5-category discrete color map) or 'continuous'.
        bin_thresholds: Positive thresholds pair (e.g. '0.01, 0.05' or (0.01, 0.05)) or 4 explicit edges.
        marker_size: Circle marker size (`s`, default 12.0).
        region: Geographic zoom preset ('global', 'north_america', 'south_america', 'europe', 'australia').
        koppen: Köppen-Geiger underlay (Beck et al. 2023, 1991–2020): 'off', 'groups' (A–E) or 'classes' (30).
        koppen_filter: 'all', a group letter ('B' = Arid) or classes ('BSk, Cfb'); restricts plotted catchments.
        koppen_alpha: Underlay opacity.
        koppen_df: Optional precomputed basin classes (``koppen.compute_basin_koppen``).
        font_scale: Multiplier on all text (1.0 = screen size; 1.5 default for papers).
        grid_leads: Row lead times for config_mode='grid' (default (1, 7)).
        Global-Best / Per-Basin DA follow the notebook-wide selection (``session.set_selection``); the same
            config is shown at every lead.
        panel_letters: Prefix multi-panel titles with (a), (b), ...
    """
    fs = float(font_scale)
    if world_gdf is None:
        for p in COUNTRY_SHP_PATHS:
            if os.path.exists(p):
                try:
                    world_gdf = gpd.read_file(p)
                    break
                except Exception:
                    pass

    mode = str(config_mode).lower()
    is_grid = mode in ("grid", "2x2")
    is_pair = mode in ("both", "compare")
    work_eval = _ensure_skill_score_columns(df_eval.copy())
    if df_meta is not None and not df_meta.empty and "lat" not in work_eval.columns:
        work_eval = pd.merge(work_eval, df_meta, on="Basin ID", how="left")
    koppen = str(koppen or "off").lower()
    if koppen != "off" or _kg_filter_label(koppen_filter):
        try:
            work_eval = _attach_koppen(work_eval, koppen_df)
        except Exception as e:
            print(f"[koppen] Could not attach Köppen-Geiger classes: {e}")
        work_eval = _apply_kg_filter(work_eval, koppen_filter)

    def _sub_for(lead: int) -> pd.DataFrame:
        sub = work_eval[work_eval["Lead Time (Days)"] == lead]
        if filter_complete_streamflow and df_ts is not None and not df_ts.empty:
            complete_set = get_complete_streamflow_basins(df_ts, year=year, lead_time=lead)
            lower_complete = {b.lower() for b in complete_set}
            sub = sub[sub["Basin ID"].astype(str).isin(complete_set)
                      | sub["Basin ID"].astype(str).str.lower().isin(lower_complete)]
        return sub

    gb_cache: Dict[int, Optional[str]] = {}

    def _gb_for(lead: int) -> Optional[str]:
        # The active notebook-wide selection makes this lead-independent (fallback: pick at t+1).
        if not gb_cache:
            gb_cache[0] = get_global_best_config(work_eval, lead_time=1)
        return gb_cache[0]

    c_norm_key = str(color_by).strip().lower()
    is_skill_mode = c_norm_key in ("skill_nse", "nse_skill", "nse_skill_score", "skill_score", "ss_nse", "skill")
    multi = is_grid or is_pair
    present_codes: set = set()

    def _draw_panel(ax, mode_key: str, lead: int, letter: str = ""):
        sub = _sub_for(lead)
        gb = _gb_for(lead)
        if world_gdf is not None:
            world_gdf.plot(ax=ax, color="#f3f5f7", edgecolor="#bdc1c6", linewidth=0.65, zorder=1)
        target_cfg = "Per-Basin Best DA" if mode_key == "per_basin_best" else gb
        if multi:
            cfg_tag = "Per-Basin DA" if mode_key == "per_basin_best" else "Global-Best DA"
        else:
            cfg_tag = "Per-Basin Best DA (Oracle)" if mode_key == "per_basin_best" else f"Global Best DA ({gb})"

        coords = pd.DataFrame()
        lons = lats = np.array([])
        if not sub.empty and "lat" in sub.columns and "lon" in sub.columns:
            best_sub = sub[sub["Config ID"] == target_cfg]
            if best_sub.empty:
                best_sub = sub[sub["Config ID"] == "Per-Basin Best DA"]
            if best_sub.empty:
                best_sub = sub.dropna(subset=["NSE Delta"])
            if best_sub.empty:
                best_sub = sub
            coords = best_sub.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["Basin ID"])
            lons = coords["lon"].values
            lats = coords["lat"].values
            if "kg_code" in coords.columns:
                present_codes.update(int(c) for c in coords["kg_code"].dropna().unique())
            groups, syms = _parse_kg_filter(koppen_filter)
            kg.draw_koppen_underlay(ax, mode=koppen, alpha=koppen_alpha,
                                    groups=groups if (groups and not syms) else None,
                                    present_codes=coords["kg_code"].dropna().unique() if "kg_code" in coords.columns else None,
                                    legend=not multi, world_gdf=world_gdf, font_scale=fs)

            _apply_region_extent(ax, lons, lats, region=region)
            ax.set_aspect("equal", adjustable="box")
            pad = 0.02  # constrained layout measures the pad from the tick labels / x label
            reg_key = str(region).strip().lower()
            if reg_key in REGION_EXTENTS:  # stats / N for the basins inside the zoomed view only
                (rx0, rx1), (ry0, ry1) = REGION_EXTENTS[reg_key]
                coords = coords[(coords["lon"] >= rx0) & (coords["lon"] <= rx1)
                                & (coords["lat"] >= ry0) & (coords["lat"] <= ry1)]
                lons, lats = coords["lon"].values, coords["lat"].values
            shrink = 0.72 if multi else 0.68
            lead_tag = f"t+{lead}"
            cb_kw = dict(color_mode=color_mode, bin_thresholds=bin_thresholds, vmin=vmin, vmax=vmax,
                         marker_size=marker_size, cbar_shrink=shrink, cbar_pad=pad, font_scale=fs,
                         cbar_fraction=0.075 if multi else 0.1)
            if c_norm_key == "uniform":
                ax.scatter(lons, lats, color="#1a73e8", s=marker_size, alpha=0.85, edgecolors="#0d47a1",
                           linewidth=0.4, zorder=3, label=f"Catchments (N={len(coords)})")
                ax.legend(loc="lower right" if koppen != "off" else "lower left", frameon=True, fontsize=10 * fs)
            elif c_norm_key in ("baseline", "da"):
                col_name = "Base NSE" if c_norm_key == "baseline" else "DA NSE"
                label_name = f"{lead_tag} open-loop baseline NSE" if c_norm_key == "baseline" else f"{lead_tag} DA NSE"
                vals = coords[col_name].values if col_name in coords.columns else np.full(len(coords), np.nan)
                _render_map_scatter_and_colorbar(ax, lons, lats, vals, cbar_label=label_name, is_diverging=False, **cb_kw)
            elif is_skill_mode:
                vals = coords["NSE Skill Score"].values if "NSE Skill Score" in coords.columns else np.full(len(coords), np.nan)
                lbl = f"{lead_tag} NSE skill score" if multi else f"Lead {lead_tag} NSE Skill Score [1 − (1−DA)/(1−Base)]"
                _render_map_scatter_and_colorbar(ax, lons, lats, vals, cbar_label=lbl, is_diverging=True,
                                                 ss_units=True, **cb_kw)
            else:
                lbl = f"{lead_tag} ΔNSE (DA − baseline)"
                _render_map_scatter_and_colorbar(ax, lons, lats, coords["NSE Delta"].values, cbar_label=lbl,
                                                 is_diverging=True, **cb_kw)

            if show_star and star_basin:
                sel_b = coords[coords["Basin ID"] == star_basin]
                if not sel_b.empty:
                    ax.scatter([sel_b["lon"].iloc[0]], [sel_b["lat"].iloc[0]], color="#fbbc04", edgecolors="black",
                               s=max(240.0, marker_size * 22.0), marker="*", linewidth=1.5, zorder=20,
                               label=f"Catchment {star_basin}")
                    ax.legend(loc="upper right", frameon=True, fontsize=9 * fs)

        n_count = len(coords)
        if c_norm_key != "uniform" and not coords.empty:  # evaluable basins only (NaN metrics are not drawn)
            n_col = {"baseline": "Base NSE", "da": "DA NSE"}.get(c_norm_key, "NSE Skill Score" if is_skill_mode else "NSE Delta")
            if n_col in coords.columns:
                n_count = int(coords[n_col].notna().sum())
        stat_str = ""
        if not coords.empty:
            stat_col = "NSE Skill Score" if is_skill_mode else "NSE Delta"
            stat_lbl = "median SS" if is_skill_mode else "median ΔNSE"
            if stat_col in coords.columns:
                valid_d = coords[stat_col].dropna()
                if not valid_d.empty:
                    med_str = units.fmt_ss(valid_d.median()) if is_skill_mode else f"{valid_d.median():+.3f}"
                    stat_str = f"{stat_lbl} {med_str}, {(valid_d > 0).mean() * 100:.0f}% > 0"
        filter_tag = f" [Complete q_obs {year}]" if (filter_complete_streamflow and year) else (" [Complete q_obs]" if filter_complete_streamflow else "")
        if _kg_filter_label(koppen_filter):
            filter_tag += f" [KG: {_kg_filter_label(koppen_filter)}]"
        if multi:
            head = f"{letter + ' ' if (panel_letters and letter) else ''}{cfg_tag}, t+{lead}"
            panel_title = f"{head}\nN={n_count:,}" + (f" | {stat_str}" if stat_str else "") + filter_tag
            ax.set_title(panel_title, fontsize=11.5 * fs, fontweight="bold", pad=6, loc="left")
        else:
            panel_title = title if title else f"{cfg_tag} (N={n_count:,}" + (f" | {stat_str}" if stat_str else "") + f"){filter_tag}"
            ax.set_title(panel_title, fontsize=12 * fs, fontweight="bold", pad=10)
            ax.set_xlabel("Longitude", fontsize=11 * fs)
            ax.set_ylabel("Latitude", fontsize=11 * fs)
        ax.xaxis.set_major_formatter(_deg_formatter("lon"))
        ax.yaxis.set_major_formatter(_deg_formatter("lat"))
        ax.tick_params(labelsize=10 * fs)
        ax.grid(True, linestyle="--", alpha=0.35)

    letters = "abcdefgh"
    if is_grid:
        leads = list(grid_leads)[:2] if len(grid_leads) >= 2 else [grid_leads[0], grid_leads[0]]
        fig, axes = plt.subplots(2, 2, figsize=figsize or (19, 13.5), layout="constrained")
        k = 0
        for r, lead in enumerate(leads):
            for c, mk in enumerate(("global_best", "per_basin_best")):
                _draw_panel(axes[r, c], mk, int(lead), f"({letters[k]})")
                k += 1
    elif is_pair:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (21, 7.5), layout="constrained")
        _draw_panel(axes[0], "global_best", lead_time, "(a)")
        _draw_panel(axes[1], "per_basin_best", lead_time, "(b)")
    else:
        fig, ax = plt.subplots(figsize=figsize or (15, 8.2), layout="constrained")
        _draw_panel(ax, mode, lead_time)

    if multi and koppen in ("groups", "classes"):
        groups, syms = _parse_kg_filter(koppen_filter)
        handles = kg.kg_legend_handles(koppen, groups if (groups and not syms) else None,
                                       sorted(present_codes) if koppen == "classes" else None, koppen_alpha)
        if handles:
            ncol = len(handles) if len(handles) <= 10 else int(np.ceil(len(handles) / 2))
            fig.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=10.5 * fs, frameon=False,
                       title="Köppen-Geiger climate (Beck et al., 2023)", title_fontsize=11 * fs,
                       handlelength=1.6, columnspacing=1.4)
    if multi and title != "":
        gb_note = _gb_for(grid_leads[0] if is_grid else lead_time)
        sel = get_active()
        fig.suptitle(title or f"Global-Best DA = {gb_note} ({sel.label() if sel else 'selected at t+1'})",
                     fontsize=10 * fs, color="#3c4043")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.04, wspace=0.03)
    fig.canvas.draw()
    return fig
