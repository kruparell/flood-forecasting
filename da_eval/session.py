"""High-level DAAnalysisSession façade uniting all analytical stages into clean, single-line calls."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = lambda x: x
    display = print
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from da_eval.benchmarks import (
    build_per_basin_optimal_eval,
    build_per_basin_optimal_timeseries,
    get_global_best_config,
)
from da_eval.dashboard import InteractiveDADashboard, plot_catchment_map
from da_eval.lazy_timeseries import PER_BASIN_RHO, LazyZarrTimeseries, ar1_config_name, parse_ar1_config
from da_eval.forcings import (
    DEFAULT_SUBSET as DEFAULT_PRECIP_SUBSET,
    PRECIP_PRODUCTS,
    PrecipForcingStore,
    cache_precip_subset,
    normalize_precip_names,
)
from da_eval.ingestion import (
    detect_lead_times,
    is_rho_config,
    load_basin_attributes,
    load_valid_parquets,
    sync_cns_shards_locally,
)
from da_eval.physical_strata import (
    GAP_COL,
    build_stratified_leaderboard_table,
    compute_consolidated_physical_strata,
    compute_gb_pb_gap,
    filter_cohort_by_attributes,
    find_candidate_basins,
    get_category_candidates,
    style_candidate_gauges_table,
)
from da_eval.static_plots import (
    plot_basin_characteristics_efficacy,
    plot_basin_epoch_case_study,
    CASE_STUDY_ROLES,
    split_case_study_models,
    plot_ecdf_suite,
    format_ecdf_median_table,
    DEFAULT_ECDF_SHOW,
    ECDF_MODEL_LABELS,
    ECDF_MODELS,
    plot_hyperparameter_effects,
    plot_leadtime_decay,
    plot_stratified_ecdf_suite,
    plot_top_n_optimal_basin_distribution,
)
from da_eval import paper_figures as _pf
from da_eval import units
from da_eval import palette
from da_eval import basin_characteristics as _bc
from da_eval.tables import (
    build_hyperparameter_risk_table,
    build_top_n_benchmark_matrix,
    build_top_n_optimal_basin_distribution_table,
    style_hyperparameter_risk_table,
    style_physical_characteristics_table,
    style_top_n_optimal_basin_distribution_table,
    style_top_n_table,
)
from da_eval.zenodo_benchmark import (
    build_zenodo_comparison_table,
    load_pretrained_metrics,
    load_zenodo_gauge_metrics,
    plot_zenodo_comparison_suite,
    style_zenodo_comparison_table,
)



class DAAnalysisSession:
    """Standardized orchestrator for Data Assimilation Evaluation across Cell-State, Precip, and Embedding modes."""

    def __init__(
        self,
        mode: str,  # "cell_state", "precipitation", or "embedding"
        data_dir: str,
        attr_zarr_path: Optional[str | Path] = None,
        cns_dir: Optional[str] = None,
        filter_basins: Optional[List[str] | set | str | Path] = None,
        max_shards: Optional[int] = None,
        max_basins: Optional[int] = None,
        load_timeseries: bool = False,
        selection: Optional["SelectionPolicy | dict"] = None,
    ):
        """``selection``: how Global-Best / Per-Basin reference models are chosen for the WHOLE session
        (a :class:`da_eval.selection.SelectionPolicy` or its keyword dict; default median NSE skill score at
        t+1). Change it later with :meth:`set_selection`; no plot chooses its own reference models."""
        self.mode = mode.lower()
        self.data_dir = data_dir
        self.attr_zarr_path = attr_zarr_path
        self.cns_dir = cns_dir
        self.filter_basins = filter_basins
        self.max_shards = max_shards
        self.max_basins = max_basins
        self.load_timeseries_flag = load_timeseries

        # Mode-specific fallbacks
        mode_fallbacks = {
            "cell_state": [
                "/usr/local/google/home/kruparell/da-paper/notebooks/02_results/data_281271878",
                "/usr/local/google/home/kruparell/da-paper/notebooks/02_results/data_281279189",
                "/usr/local/google/home/kruparell/da-paper/data/eval_staging/data_281245292",
            ],
            "precipitation": [
                "/usr/local/google/home/kruparell/da-paper/notebooks/02_results/data_281290836",
                "/usr/local/google/home/kruparell/da-paper/data/eval_staging/data_281257427",
            ],
            "embedding": [
                "/tmp/da_embedded_cache",
            ],
        }

        # Fallback staging check if local directory does not exist, is empty, or lacks baseline/parquets
        if not str(self.data_dir).startswith("/cns/"):
            has_parquets = os.path.exists(self.data_dir) and any(Path(self.data_dir).glob("*.parquet"))
            has_baseline_csv = os.path.exists(self.data_dir) and any(
                "baseline" in str(p).lower() for p in Path(self.data_dir).glob("**/test_metrics*.csv")
            )
            if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir) or (not has_parquets and not has_baseline_csv):
                inferred_cns = self.cns_dir or f"/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/{os.path.basename(str(self.data_dir).rstrip('/'))}"
                print(f"[INFO] Local directory '{self.data_dir}' is incomplete or missing baseline. Syncing from CNS ({inferred_cns})...")
                sync_cns_shards_locally(inferred_cns, str(self.data_dir), max_shards=self.max_shards, sync_timeseries=self.load_timeseries_flag)
            if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
                for fb in mode_fallbacks.get(self.mode, []):
                    if os.path.exists(fb) and os.listdir(fb):
                        print(f"[INFO] Using staged local directory for {self.mode}: {fb}")
                        self.data_dir = fb
                        break

        # Parse filter_basins upfront to allow pushdown filtering during shard ingestion
        allowed_basins = None
        if self.filter_basins is not None:
            if isinstance(self.filter_basins, (str, Path)):
                if os.path.exists(str(self.filter_basins)):
                    with open(str(self.filter_basins), "r") as f:
                        allowed_basins = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
                else:
                    allowed_basins = {str(self.filter_basins)}
            else:
                allowed_basins = set(self.filter_basins)

        if self.cns_dir:
            sync_cns_shards_locally(self.cns_dir, self.data_dir, max_shards=self.max_shards, sync_timeseries=self.load_timeseries_flag)

        print(f"Initializing DA Analysis Session (Mode: {self.mode.upper()})...")
        if self.max_basins or self.max_shards:
            print(f"[SUBSET] Quick exploration mode active: max_shards={self.max_shards}, max_basins={self.max_basins}")

        self.df_eval = load_valid_parquets(
            self.data_dir,
            "detailed_basin_parameter_eval",
            filter_basins=allowed_basins,
            max_shards=self.max_shards,
            max_basins=self.max_basins,
        )

        if self.df_eval.empty:
            print(f"[WARNING] No evaluation parquet files loaded from '{self.data_dir}'.")
            self.lead_times = [1, 2, 3, 4, 5, 6, 7]
            self.df_meta = pd.DataFrame()
            self.df_strata = pd.DataFrame()
            self.df_ts = pd.DataFrame()
            self.selection = None
            return

        # Optional secondary basin filtering confirmation
        if allowed_basins is not None:
            orig_cnt = self.df_eval["Basin ID"].nunique()
            eval_basins = self.df_eval["Basin ID"].astype(str)
            lower_map = {b.lower(): b for b in allowed_basins}
            mask = eval_basins.isin(allowed_basins) | eval_basins.str.lower().isin(lower_map)
            self.df_eval = self.df_eval[mask].reset_index(drop=True)
            print(f"[FILTER] Filtered basins to requested subset: {self.df_eval['Basin ID'].nunique():,} / {orig_cnt:,} catchments.")

            if self.df_eval.empty:
                print(f"[WARNING] No evaluation records match the provided filter_basins.")
                self.lead_times = [1, 2, 3, 4, 5, 6, 7]
                self.df_meta = pd.DataFrame()
                self.df_strata = pd.DataFrame()
                self.df_ts = pd.DataFrame()
                self.selection = None
                return

        # Cross-contamination validation check
        sample_cfgs = self.df_eval["Config ID"].astype(str).tolist()
        if self.mode == "cell_state":
            invalid = [c for c in sample_cfgs[:50] if "embedded_" in c or "precip" in c]
            if invalid:
                raise ValueError(f"Cell-state session loaded foreign configurations: {invalid[:3]}. Check data_dir path.")
        elif self.mode == "precipitation":
            invalid = [c for c in sample_cfgs[:50] if "precip" not in c and "Baseline" not in c and "Per-Basin" not in c and "ar1" not in c.lower()]
            if invalid:
                raise ValueError(f"Precipitation session loaded foreign configurations: {invalid[:3]}. Check data_dir path.")
        elif self.mode == "embedding":
            valid_tokens = (
                # Legacy / Stage <=36 naming
                "embedded_", "both_", "dyn_", "stat_", "all_", "triple_",
                "dualdyn_", "_fc_", "baseline", "per-basin", "ar1", "mf2lstm",
                # Stage 37 factorial naming: SA_w90_lr0.01_bg5, DD_w14_lr0.02_bg10,
                # TR_w30_unleashed, AB_TM2_tol0.02
                "sa_w", "dd_w", "tr_w", "ab_",
                # Stage 38-53 sweep naming: S_w90_..., D_w30_..., T_w7_..., S41_..., S45_..., W2_..., W2B_...
                "s_w", "d_w", "t_w", "epc_", "s40_", "s41_", "s42_",
                "s43_", "s44_", "s45_", "s48_", "s49_", "s50_", "s51_", "s52_", "s53_",
                "w2_", "w2b_", "_hs_w", "_h_w", "_s_w", "_t_w", "ar1",
            )
            invalid = [c for c in sample_cfgs[:50] if not any(tok in c.lower() for tok in valid_tokens)]
            if invalid:
                raise ValueError(f"Embedding session loaded foreign configurations: {invalid[:3]}. Check data_dir path.")

        self.lead_times = detect_lead_times(self.df_eval)
        has_lead_zero = 0 in self.lead_times
        print(f"Detected Horizons: {len(self.lead_times)} lead times ({'Includes Nowcast t+0' if has_lead_zero else 'Leads 1..7'})")

        print(f"Synthesizing Per-Basin Best DA benchmarks across {self.df_eval['Basin ID'].nunique():,} catchments...")
        self.set_selection(selection, _init=True)

        # Optimize timeseries loading: lazy by default, only loaded if explicitly requested
        ref_lead = 1 if 1 in self.lead_times else self.lead_times[0]
        if self.load_timeseries_flag:
            best_cfg = get_global_best_config(self.df_eval, lead_time=ref_lead)
            target_configs = [best_cfg, "Baseline", "Per-Basin Best DA"]
            eval_basins = set(self.df_eval["Basin ID"].dropna().unique())

            print(f"Loading forecast timeseries for top benchmark models ({best_cfg}) across {len(eval_basins):,} catchments...")
            self.df_ts = load_valid_parquets(
                self.data_dir,
                "timeseries_forecasts",
                filter_configs=target_configs,
                filter_basins=eval_basins,
                max_shards=self.max_shards,
            )

            if not self.df_ts.empty:
                self.df_ts = build_per_basin_optimal_timeseries(self.df_ts, self.df_eval)
        else:
            self.df_ts = pd.DataFrame()

        self.df_pretrained = pd.DataFrame()

        # Ingest physical attributes
        basins = self.df_eval["Basin ID"].dropna().unique().tolist()
        print("Extracting physical morphology and climate attributes from Caravans V2 zarr...")
        self.df_meta = load_basin_attributes(self.attr_zarr_path, requested_basins=basins)

        # Compute physical stratifications
        print("Computing effect-size statistics across 6 physical dimensions...")
        self.df_strata = compute_consolidated_physical_strata(
            self.df_eval,
            df_ts=self.df_ts,
            df_meta=self.df_meta,
            lead_time=ref_lead,
        )
        print("Session initialized successfully.")

    # -------------------------------------------------------------
    # Notebook-wide reference-model selection
    # -------------------------------------------------------------
    def set_selection(self, policy=None, _init: bool = False, **kwargs) -> "Selection":
        """Defines what "Global-Best" and "Per-Basin Best" mean for every plot, table and dashboard.

        ``policy``: a :class:`da_eval.selection.SelectionPolicy`, a dict of its fields, or None (default:
        median NSE skill score at t+1); ``kwargs`` override fields, e.g.
        ``set_selection(metric="NSE Skill Score", leads=(1,2,3,4,5,6,7), agg="mean", global_stat="median")``.
        Rebuilds the synthetic ``Per-Basin Best DA`` / ``AR1_Per_Basin_Best_Rho`` rows and clears derived caches.
        """
        from da_eval import selection as S
        if isinstance(policy, dict):
            policy = S.SelectionPolicy(**policy)
        policy = S.policy_from(policy, **kwargs)
        if not set(policy.leads) <= set(self.lead_times):
            raise ValueError(f"Selection leads {policy.leads} not all in {self.lead_times}")
        sel = S.resolve_selection(self.df_eval, policy)
        self.selection = sel
        S.set_active(sel)
        self.df_eval = S.apply_selection(self.df_eval, sel)
        if not _init:
            for attr in ("_rr_cache", "_ts_gb_cache", "_ts_gbr_cache", "_ts_pb_cache", "koppen_table",
                         "config_affinity", "_lazy_loaded"):
                self.__dict__.pop(attr, None)
            if getattr(self, "_ts_store", None) is not None:
                if self._ts_store.supports_ar1:
                    self._ts_store.set_basin_rho(self._ar1_basin_rho())
                self.df_ts = pd.DataFrame()  # lazily re-read with the new reference models
            elif not self.df_ts.empty:
                self.df_ts = S.per_basin_timeseries(self.df_ts, sel)
            if not self.df_meta.empty:
                ref = sel.policy.leads[0]
                self.df_strata = compute_consolidated_physical_strata(self.df_eval, df_ts=self.df_ts,
                                                                      df_meta=self.df_meta, lead_time=ref)
        rho = f" | Global-Best PP = {sel.global_rho}" if sel.global_rho else ""
        print(f"[SELECTION] {sel.label()}: Global-Best DA = {sel.global_da}{rho} "
              f"(ranked on {sel.n_basins:,} common basins)")
        return sel

    @property
    def global_best(self) -> str:
        """Global-Best DA Config ID under the session selection."""
        return self.selection.global_da

    @staticmethod
    def set_ss_percent(on: bool = True) -> None:
        """Notebook-wide display unit for the NSE skill score: % (True, default) or fraction (False).

        Presentation only (tables, axes, labels); data, thresholds and arguments keep native units.
        """
        units.set_ss_percent(on)

    @staticmethod
    def set_palette(name: str = "cvd_safe") -> None:
        """Notebook-wide reference-model colours / line styles: ``"cvd_safe"`` (default, colour-blind safe) or
        ``"legacy"`` (earlier Google colours). Every da_eval plot reads it at draw time (see ``da_eval.palette``)."""
        palette.set_palette(name)

    def selection_summary(self, top_n: int = 15):
        """Ranking behind the session selection (DA and PP pools) with the per-basin winner shares."""
        sel = getattr(self, "selection", None)
        if sel is None:
            print("[WARNING] No reference-model selection: no DA evaluation metrics were loaded for this experiment "
                  f"({self.data_dir}). Check that DA configs (not only E_baseline_0da) are staged locally / on CNS.")
            return
        display(HTML(f"<b>Reference models {sel.label()}</b> — Global-Best DA: <code>{sel.global_da}</code>"
                     + (f"; Global-Best PP: <code>{sel.global_rho}</code>" if sel.global_rho else "")
                     + f"; ranked on {sel.n_basins:,} common basins. Per-Basin models pick each basin's best "
                       "config on the same score (in-sample oracle)."))
        r = sel.ranking.groupby("Pool", sort=False).head(top_n)
        num = {c: "{:.3f}" for c in r.columns if r[c].dtype.kind == "f"}
        # the ranking statistic is an NSE skill score unless it is a share of basins
        ss_col = sel.policy.global_stat if (sel.policy.metric == "NSE Skill Score" and units.ss_percent()
                                            and sel.policy.global_stat != "pct_improved") else None
        if ss_col in num:
            num[ss_col] = units.ss_formatter(sign=False)
        sty = (r.style.format(num).hide(axis="index")
               .apply(lambda row: ["font-weight:bold; background:#e8f0fe" if row["Global-Best"] else "" for _ in row], axis=1))
        if ss_col in num:
            sty = sty.format_index(lambda c: units.ss_label(c) if c == ss_col else c, axis=1)
        display(sty)
        return None

    def load_timeseries(
        self,
        basins: Optional[List[str] | set] = None,
        configs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Loads forecast timeseries on-demand for specific basins or benchmark configs."""
        ref_lead = 1 if 1 in self.lead_times else self.lead_times[0]
        best_cfg = get_global_best_config(self.df_eval, lead_time=ref_lead)

        if configs:
            target_configs = configs
        else:
            raw_eval = self.df_eval[
                (~self.df_eval["Config ID"].str.contains("Per-Basin", na=False))
                & (~self.df_eval["Config ID"].str.contains("Baseline", na=False))
            ]
            all_unique_cfgs = raw_eval["Config ID"].dropna().unique().tolist()
            if len(all_unique_cfgs) <= 12:
                target_configs = list(set(all_unique_cfgs + [best_cfg, "Baseline"]))
            else:
                opt_cfgs = raw_eval[raw_eval["Lead Time (Days)"] == ref_lead].sort_values("NSE Delta", ascending=False).drop_duplicates(subset=["Basin ID"])["Config ID"].dropna().unique().tolist()
                target_configs = list(set(opt_cfgs + [best_cfg, "Baseline"]))

        target_basins = set(basins) if basins else set(self.df_eval["Basin ID"].dropna().unique())

        has_local_ts = (
            os.path.exists(self.data_dir)
            and (
                any(Path(self.data_dir).glob("**/*timeseries*.parquet"))
                or any(Path(self.data_dir).glob("**/test_results*.zarr"))
            )
        )
        if not has_local_ts:
            inferred_cns = self.cns_dir or f"/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results/{os.path.basename(str(self.data_dir).rstrip('/'))}"
            sync_cns_shards_locally(inferred_cns, str(self.data_dir), max_shards=self.max_shards, sync_timeseries=True)

        print(f"Loading forecast timeseries for {len(target_configs)} models across {len(target_basins):,} catchments...")
        df_ts = load_valid_parquets(
            self.data_dir,
            "timeseries_forecasts",
            filter_configs=target_configs,
            filter_basins=target_basins,
            max_shards=self.max_shards,
        )
        if not df_ts.empty:
            df_ts = build_per_basin_optimal_timeseries(df_ts, self.df_eval, lead_time_ref=ref_lead)
            self.df_strata = compute_consolidated_physical_strata(
                self.df_eval,
                df_ts=df_ts,
                df_meta=self.df_meta,
                lead_time=ref_lead,
            )
        self.df_ts = df_ts
        return self.df_ts

    # -------------------------------------------------------------
    # Lazy (on-demand) timeseries access
    # -------------------------------------------------------------
    @property
    def ts_store(self) -> Optional[LazyZarrTimeseries]:
        """Lazy per-basin zarr reader (``None`` when the experiment ships timeseries parquets instead)."""
        if not hasattr(self, "_ts_store"):
            self._ts_store = None
            if self.data_dir and os.path.isdir(str(self.data_dir)):
                has_parquet_ts = any(Path(self.data_dir).glob("*timeseries*.parquet")) or any(
                    Path(self.data_dir).glob("timeseries_forecasts/*.parquet"))
                if not has_parquet_ts and LazyZarrTimeseries.has_stores(str(self.data_dir)):
                    self._ts_store = LazyZarrTimeseries(self.data_dir)
                    if self._ts_store.supports_ar1:
                        self._ts_store.set_basin_rho(self._ar1_basin_rho())
        return self._ts_store

    def _ar1_basin_rho(self) -> Dict[str, float]:
        """Per-basin rho behind ``AR1_Per_Basin_Best_Rho`` (from the session selection; legacy: best lead-1 NSE)."""
        sel = getattr(self, "selection", None)
        if sel is not None and sel.per_basin_rho:
            return {b: r for b, c in sel.per_basin_rho.items() if isinstance(r := parse_ar1_config(c), float)}
        ev = self.df_eval
        fixed = {c: r for c in ev["Config ID"].dropna().unique() if isinstance(r := parse_ar1_config(c), float)}
        if not fixed:
            return {}
        ref = 1 if 1 in self.lead_times else self.lead_times[0]
        sub = ev[(ev["Lead Time (Days)"] == ref) & ev["Config ID"].isin(list(fixed))][["Basin ID", "Config ID", "DA NSE"]].dropna()
        best = sub.sort_values("DA NSE", ascending=False).drop_duplicates("Basin ID")
        return {str(b): fixed[c] for b, c in zip(best["Basin ID"], best["Config ID"])}

    def _ts_configs(self) -> List[str]:
        """Evaluation Config IDs that have a forecast timeseries in the lazy store (PP configs are synthesised)."""
        store = self.ts_store
        if store is None:
            return []
        if not hasattr(self, "_ts_cfg_cache"):
            cfgs = [c for c in self.df_eval["Config ID"].dropna().unique() if "Per-Basin" not in str(c)]
            self._ts_cfg_cache = [c for c in cfgs if store.resolve_config(c) and store.resolve_config(c) != "Baseline"]
        return self._ts_cfg_cache

    def _ts_global_best(self, lead_time: int) -> Optional[str]:
        """Global-Best DA config of the session selection (lead-independent); warns if it has no timeseries."""
        sel = getattr(self, "selection", None)
        if sel is not None:
            if self.ts_store is not None and sel.global_da not in self._ts_configs():
                print(f"[SELECTION] Global-Best {sel.global_da!r} has no forecast timeseries in the store.")
                return None
            return sel.global_da
        if not hasattr(self, "_ts_gb_cache"):
            self._ts_gb_cache = {}
        if lead_time not in self._ts_gb_cache:
            cfgs = self._ts_configs()
            src = self.df_eval[self.df_eval["Config ID"].isin(cfgs)] if cfgs else self.df_eval
            self._ts_gb_cache[lead_time] = get_global_best_config(src, lead_time=lead_time)
        return self._ts_gb_cache[lead_time]

    def _ts_global_best_rho(self, lead_time: int) -> Optional[str]:
        """Global-Best fixed-rho PP config of the session selection; None if no PP metrics."""
        sel = getattr(self, "selection", None)
        if sel is not None:
            return sel.global_rho if sel.global_rho in self._ts_configs() else None
        if not hasattr(self, "_ts_gbr_cache"):
            self._ts_gbr_cache = {}
        if lead_time not in self._ts_gbr_cache:
            cfgs = [c for c in self._ts_configs() if isinstance(parse_ar1_config(c), float)]
            self._ts_gbr_cache[lead_time] = (get_global_best_config(self.df_eval[self.df_eval["Config ID"].isin(cfgs)],
                                                                    lead_time=lead_time, exclude_rho=False) or None) if cfgs else None
        return self._ts_gbr_cache[lead_time]

    def _ts_per_basin_rho_cfg(self) -> Optional[str]:
        cfgs = [c for c in self._ts_configs() if parse_ar1_config(c) == PER_BASIN_RHO]
        return cfgs[0] if cfgs else None

    def _ts_basin_best(self, basin_id: str, lead_time: int) -> Optional[str]:
        """Per-Basin DA config of the session selection (lead-independent; no PP)."""
        sel = getattr(self, "selection", None)
        if sel is not None:
            cfg = sel.per_basin_da.get(str(basin_id))
            return cfg if cfg in self._ts_configs() else None
        if not hasattr(self, "_ts_pb_cache"):
            self._ts_pb_cache = {}
        if lead_time not in self._ts_pb_cache:
            da_cfgs = [c for c in self._ts_configs() if not is_rho_config(c)]
            sub = self.df_eval[(self.df_eval["Lead Time (Days)"] == lead_time)
                               & (self.df_eval["Config ID"].isin(da_cfgs))].dropna(subset=["NSE Skill Score"])
            best = sub.sort_values("NSE Skill Score", ascending=False).drop_duplicates("Basin ID")
            self._ts_pb_cache[lead_time] = dict(zip(best["Basin ID"].astype(str), best["Config ID"].astype(str)))
        return self._ts_pb_cache[lead_time].get(str(basin_id))

    def ensure_timeseries(self, basins, extra_configs: Optional[List[str]] = None) -> pd.DataFrame:
        """Loads forecast timeseries for ``basins`` on demand (~0.05 s per basin and config) and appends them to ``df_ts``.

        Per basin this reads Baseline, the Global Best DA config for each lead, the basin's own best DA
        config (stored as ``Per-Basin Best DA``), the Global-Best PP and Per-Basin PP PP series
        (synthesised from the Baseline store), and any ``extra_configs``. ``extra_configs`` may include
        arbitrary PP series such as ``"AR1_rho0.7"``. Returns the ``df_ts`` rows for ``basins``.
        No-op for parquet-backed experiments.
        """
        basins = [str(b) for b in ([basins] if isinstance(basins, str) else basins)]
        store = self.ts_store
        if store is None:
            if self.df_ts.empty:
                self.load_timeseries(basins=basins)
            return self.df_ts[self.df_ts["Basin ID"].isin(basins)] if not self.df_ts.empty else self.df_ts
        if not hasattr(self, "_lazy_loaded"):
            self._lazy_loaded = set()  # (basin, config) pairs already in df_ts
        ref_lead = 1 if 1 in self.lead_times else self.lead_times[0]
        default_cfgs = {self._ts_global_best(lt) for lt in (self.lead_times if getattr(self, "selection", None) is None
                                                          else [ref_lead])}
        default_cfgs |= {self._ts_global_best_rho(ref_lead), self._ts_per_basin_rho_cfg()}
        default_cfgs.discard(None)
        extras = [c for c in (extra_configs or []) if c]
        new_frames = []
        for b in basins:
            pb_cfg = self._ts_basin_best(b, ref_lead)
            wanted = list(dict.fromkeys(list(default_cfgs) + extras))
            todo = [c for c in wanted if (b, c) not in self._lazy_loaded]
            need_pb = pb_cfg and (b, "Per-Basin Best DA") not in self._lazy_loaded
            if not todo and not need_pb:
                continue
            f = store.get_basin(b, todo + ([pb_cfg] if need_pb else []))
            if not f.empty and need_pb:
                pb = f[f["Config ID"] == pb_cfg].copy()
                pb["Config ID"] = "Per-Basin Best DA"
                if pb_cfg not in todo:
                    f = f[f["Config ID"] != pb_cfg]  # keep only the Per-Basin Best DA copy (avoids duplicate lines)
                f = pd.concat([f, pb], ignore_index=True)
            if not f.empty:
                new_frames.append(f)
            self._lazy_loaded.update((b, c) for c in todo)
            if need_pb:
                self._lazy_loaded.add((b, "Per-Basin Best DA"))
        if new_frames:
            self.df_ts = pd.concat([self.df_ts] + new_frames, ignore_index=True) if not self.df_ts.empty else pd.concat(new_frames, ignore_index=True)
        return self.df_ts[self.df_ts["Basin ID"].isin(basins)] if not self.df_ts.empty else self.df_ts

    def get_ar1_timeseries(self, basin_id: str, rho) -> pd.DataFrame:
        """PP-corrected Baseline hydrograph for one basin, generated on the fly for any ``rho``.

        ``rho`` may be a float (e.g. ``0.7``) or ``"per_basin"`` (the basin's best fixed rho at lead 1).
        The series is also appended to ``df_ts`` (Config ID ``AR1_postprocess_rho<x>``) so plots and the
        dashboard can show it. Model: ``q = max(0, q_base(L) + rho**L * (q_obs(T) - q_base(t+0)))``.
        """
        cfg = "AR1_Per_Basin_Best_Rho" if rho == PER_BASIN_RHO else ar1_config_name(float(rho))
        df = self.ensure_timeseries([basin_id], extra_configs=[cfg])
        return df[df["Config ID"] == cfg] if not df.empty else df

    def get_obs_timeseries(self, lead_time: Optional[int] = None) -> pd.DataFrame:
        """Observed streamflow only for every basin at one lead (cached parquet; ~3 s first time)."""
        lt = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        if self.ts_store is not None:
            return self.ts_store.obs_frame(lt)
        if self.df_ts.empty:
            self.load_timeseries()
        return self.df_ts

    def get_strata_timeseries(self, lead_time: Optional[int] = None) -> pd.DataFrame:
        """Baseline + Global Best DA at one lead for every basin (cached parquet; ~15 s first time).

        Used by the seasonal High/Low-flow strata and candidate finders instead of full timeseries.
        """
        lt = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        if self.ts_store is None:
            return self.df_ts
        gb = self._ts_global_best(lt)
        return self.ts_store.lead_frame(gb, lt) if gb else pd.DataFrame()

    def _candidate_ts(self) -> pd.DataFrame:
        """Timeseries frame for strata / candidate finders (lazy store -> compact cached lead frame)."""
        return self.get_strata_timeseries() if self.ts_store is not None else self.df_ts

    # -------------------------------------------------------------
    # Stage 1: Top-N Benchmark Table
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Paper figures: multi-lead skill (absolute NSE/KGE + harm vs Baseline)
    # -------------------------------------------------------------
    _METRIC_COLS = {"NSE": "DA NSE", "KGE": "DA KGE", "SS": "NSE Skill Score"}

    def get_reference_skill(self, metric: str = "NSE", models: Optional[List[str]] = None,
                            include_baseline: bool = True, include_persistence: bool = True):
        """Returns ``(wide, harm_wide, labels)`` for the paper figures on one common gauge set.

        ``wide``: {model: gauge x lead frame of ``metric``} ('NSE' / 'KGE' absolute, or 'SS' skill score).
        For absolute metrics ``Baseline`` (and ``Persistence``, NSE only, from observed flow) are added so the
        figure shows the Baseline's own skill. ``harm_wide``: skill scores of the DA/PP models on the same
        gauges (for "% harmed"). Models are always selected by skill score, whatever is plotted.
        """
        metric = metric.upper()
        mcol = self._METRIC_COLS[metric]
        models = list(models or _pf.DEFAULT_MODELS)
        key = (metric, tuple(models), include_baseline, include_persistence)
        cache = self.__dict__.setdefault("_ref_skill_cache", {})
        if key in cache:
            return cache[key]
        abs_models = list(models)
        obs = None
        if metric != "SS":
            if include_baseline:
                abs_models = ["Baseline"] + abs_models
            if include_persistence and metric == "NSE":
                try:
                    obs = self.get_obs_timeseries()
                    abs_models = ["Persistence"] + abs_models
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[WARNING] Persistence reference unavailable ({e}).")
        wide, labels = _pf.build_reference_skill(self.df_eval, abs_models, self.lead_times, metric_col=mcol, obs_df=obs)
        harm, _ = _pf.build_reference_skill(self.df_eval, models, self.lead_times, metric_col="NSE Skill Score")
        common = sorted(set(next(iter(wide.values())).index) & set(next(iter(harm.values())).index))
        wide = {k: w.loc[common] for k, w in wide.items()}
        harm = {k: w.loc[common] for k, w in harm.items()}
        cache[key] = (wide, harm, labels)
        return cache[key]

    def _metric_label(self, metric: str) -> str:
        return {"NSE": "NSE", "KGE": "KGE", "SS": "NSE skill score"}[metric.upper()]

    def plot_skill_decay(self, metric: str = "NSE", models: Optional[List[str]] = None,
                         include_baseline: bool = True, include_persistence: bool = True,
                         harm_thresholds=(0.01, 0.05), show_harm_panel: bool = True, **kw) -> plt.Figure:
        """Median + IQR per lead (absolute NSE by default, with Baseline & Persistence) + "% gauges harmed" panel."""
        wide, harm, labels = self.get_reference_skill(metric, models, include_baseline, include_persistence)
        fig = _pf.plot_skill_decay(wide, harm_thresholds=harm_thresholds, show_harm_panel=show_harm_panel,
                                   metric_label=self._metric_label(metric), harm_wide=harm, **kw)
        self._print_model_labels(labels)
        plt.show()
        return fig

    def plot_harm_benefit_bars(self, models: Optional[List[str]] = None, bin_thresholds=(0.01, 0.05), **kw) -> plt.Figure:
        """Diverging stacked bars of skill-score categories (harm vs benefit relative to Baseline) per lead."""
        _, harm, labels = self.get_reference_skill("SS", models)
        fig = _pf.plot_harm_benefit_bars(harm, bin_thresholds=bin_thresholds, **kw)
        self._print_model_labels(labels)
        plt.show()
        return fig

    def plot_skill_cdf_panels(self, leads=(1, 3, 7), metric: str = "NSE", models: Optional[List[str]] = None,
                              include_baseline: bool = True, include_persistence: bool = True, **kw) -> plt.Figure:
        """CDF small multiples at ``leads`` (absolute NSE by default so the Baseline's skill is visible)."""
        wide, harm, labels = self.get_reference_skill(metric, models, include_baseline, include_persistence)
        fig = _pf.plot_skill_cdf_panels(wide, leads=leads, metric_label=self._metric_label(metric), harm_wide=harm, **kw)
        self._print_model_labels(labels)
        plt.show()
        return fig

    def plot_skill_violins(self, leads=(1, 3, 7), metric: str = "NSE", models: Optional[List[str]] = None,
                           include_baseline: bool = True, clip=(-0.5, 1.0), **kw) -> plt.Figure:
        """Clipped violins per lead (best with 2-3 leads)."""
        wide, _, labels = self.get_reference_skill(metric, models, include_baseline, include_persistence=False)
        fig = _pf.plot_skill_violins(wide, leads=leads, clip=clip, metric_label=self._metric_label(metric), **kw)
        self._print_model_labels(labels)
        plt.show()
        return fig

    def show_paper_skill_figures(self, main_leads=(1, 3, 7), supplement: bool = True, metric: str = "NSE",
                                 models: Optional[List[str]] = None):
        """Main-text multi-lead figures (+ supplementary leads) in one call."""
        print("=== Main text ===")
        self.plot_skill_cdf_panels(leads=main_leads, metric=metric, models=models)
        self.plot_skill_decay(metric=metric, models=models)
        self.plot_harm_benefit_bars(models=models)
        if supplement:
            print("=== Supplement ===")
            rest = [lt for lt in self.lead_times if lt not in main_leads]
            if rest:
                self.plot_skill_cdf_panels(leads=rest, metric=metric, models=models, ncols=min(4, len(rest)))
            self.plot_skill_cdf_panels(leads=self.lead_times, metric="SS", models=models, ncols=4)
            self.plot_skill_violins(leads=main_leads, metric="SS", models=models)

    @staticmethod
    def _print_model_labels(labels: dict):
        print(" | ".join(f"{k}: {v}" for k, v in labels.items()))

    def show_top_n_table(self, top_n: int = 10, harm_thresholds=(0.01,), lower_quantile: Optional[float] = None):
        """Renders Top-N Benchmark Matrix with column bolding and blue/red shading.

        ``harm_thresholds`` adds per-lead "% of basins harmed" columns (``% SS<-0.01 (t+L)``: share of basins whose
        skill score is below ``-thr``), e.g. ``(0.01, 0.05)``; ``None`` hides them. ``lower_quantile`` (e.g. 0.05)
        optionally adds lower-tail skill-score columns (``SS_NSE Q0.05 (t+L)``).
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot display Top-N table.")
            return
        matrix = build_top_n_benchmark_matrix(self.df_eval, top_n=top_n, lead_times=self.lead_times,
                                              lower_quantile=lower_quantile, harm_thresholds=harm_thresholds)
        if matrix.empty:
            print("[WARNING] Benchmark matrix is empty.")
            return
        styled = style_top_n_table(matrix)
        display(styled)
        return styled

    def show_horizon_summary_table(self, windows: Optional[dict] = None, stat: str = "median",
                                   labels: Optional[dict] = None, decimals: Optional[int] = None,
                                   common_basins: bool = True, baseline_in_header: bool = True,
                                   percent: Optional[bool] = None):
        """Compact paper table: Baseline NSE + skill score of the four reference schemes per lead window.

        ``windows`` = {column title: leads}, default Short (1-3) / Medium (4-7) / Full (1-7). ``labels`` overrides
        row names by key ('baseline', 'global_da', 'global_rho', 'per_basin_da', 'per_basin_rho').
        ``baseline_in_header`` puts the open-loop NSE under each column heading; ``percent`` shows SS_NSE in %
        (None = follow ``session.set_ss_percent``).
        The raw numbers are kept in ``session.horizon_table`` (``.attrs`` has N basins and config IDs).
        """
        from da_eval.tables import build_horizon_summary_table, style_horizon_summary_table
        if windows is None:
            windows = {k: v for k, v in __import__("da_eval.tables", fromlist=["x"]).DEFAULT_HORIZON_WINDOWS.items()
                       if set(v) <= set(self.lead_times)}
        self.horizon_table = build_horizon_summary_table(self.df_eval, windows=windows, stat=stat, labels=labels,
                                                         common_basins=common_basins)
        styled = style_horizon_summary_table(self.horizon_table, decimals=decimals,
                                             baseline_in_header=baseline_in_header, percent=percent)
        display(styled)
        return styled

    # -------------------------------------------------------------
    # Stage 2: Hyperparameter Main Effects & Rank Contribution
    # -------------------------------------------------------------
    def plot_hyperparameter_effects(self, lead_time: Optional[int] = None):
        """Renders Hyperparameter Rank Variance Contribution (%) & Marginal Effects Grid with Win Rates."""
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot hyperparameter effects.")
            return
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        fig = plot_hyperparameter_effects(self.df_eval, lead_time=ref_lead)
        plt.show()

    def get_hyperparameter_risk_table(
        self,
        group_by: str = "Learning Rate",
        lead_time: Optional[int] = None,
        metric: str = "NSE Skill Score",
    ) -> pd.DataFrame:
        """Computes hyperparameter risk profiles and tail distribution tables."""
        if self.df_eval.empty:
            return pd.DataFrame()
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        return build_hyperparameter_risk_table(
            df_eval=self.df_eval,
            group_by=group_by,
            lead_time=ref_lead,
            metric_col=metric,
        )

    def show_hyperparameter_risk_suite(
        self,
        default_group: str = "Learning Rate",
        default_lead: Optional[int] = None,
        default_metric: str = "NSE Skill Score",
    ):
        """Interactive widget suite to inspect hyperparameter distributions and downside risk (Q0.05)."""
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty.")
            return
        ref_lead = default_lead if default_lead is not None else (1 if 1 in self.lead_times else self.lead_times[0])

        try:
            import ipywidgets as widgets
        except ImportError:
            tbl = self.get_hyperparameter_risk_table(group_by=default_group, lead_time=ref_lead, metric=default_metric)
            styler = style_hyperparameter_risk_table(tbl)
            display(styler)
            return

        _hp_options = [
            ("Learning Rate (LR)", "Learning Rate", r"_(?:lr|lrd)([0-9.eE+-]+)"),
            ("BG Regularization (bg)", "BG Weight", r"_(?:bg|bgd)([0-9.eE+-]+)"),
            ("Assimilation Window (w)", "Window (Days)", r"_w(\d+)"),
            ("Early Stopping Tol", "Tolerance", r"_tol([A-Za-z0-9\.]+)"),
            ("LR Ratio (stat/dyn)", "LR Ratio", r"_r([0-9][0-9.]*)(?=_|$)"),
            ("Epochs", "Epochs", r"_ep(\d+)"),
        ]
        _cids = [c for c in self.df_eval["Config ID"].unique() if not is_rho_config(c) and "Baseline" not in str(c)
                 and "Per-Basin" not in str(c)]

        def _n_levels(pat):
            return len({(m.group(1) if (m := re.search(pat, str(c))) else None) for c in _cids})

        opts = [(lbl, col) for lbl, col, pat in _hp_options if _n_levels(pat) > 1 or col == default_group]
        param_toggle = widgets.ToggleButtons(
            options=opts,
            value=default_group if default_group in [o[1] for o in opts] else opts[0][1],
            description="Hyperparameter:",
            button_style="info",
            style={"description_width": "initial", "button_width": "160px"},
        )

        lead_drop = widgets.Dropdown(
            options=[(f"Lead t+{lt} Day{'s' if lt != 1 else ''}", lt) for lt in self.lead_times],
            value=ref_lead,
            description="Forecast Horizon:",
            layout=widgets.Layout(width="220px"),
            style={"description_width": "initial"},
        )

        metric_drop = widgets.Dropdown(
            options=[
                ("NSE Skill Score (SS)", "NSE Skill Score"),
                ("ΔNSE (DA − Base)", "NSE Delta"),
                ("KGE Skill Score", "KGE Skill Score"),
                ("ΔKGE (DA − Base)", "KGE Delta"),
            ],
            value=default_metric,
            description="Metric:",
            layout=widgets.Layout(width="220px"),
            style={"description_width": "initial"},
        )

        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_group = param_toggle.value
                sel_lead = lead_drop.value
                sel_metric = metric_drop.value

                tbl = self.get_hyperparameter_risk_table(
                    group_by=sel_group,
                    lead_time=sel_lead,
                    metric=sel_metric,
                )
                if not tbl.empty:
                    header = (
                        f"<div style='margin-bottom:8px;padding:8px 12px;background:#f8f9fa;border-left:4px solid #1a73e8;border-radius:4px;'>"
                        f"<b>Hyperparameter Downside Risk & Distribution Profile (Lead t+{sel_lead})</b> &mdash; Stratified by <code>{sel_group}</code><br>"
                        f"<span style='color:#555;font-size:12px;'>"
                        f"Identifies bifurcation boundaries, downside risk (Q0.05), safe configuration fractions, and upside potential (Q0.75, Q0.95)."
                        f"</span></div>"
                    )
                    display(HTML(header))
                    styler = style_hyperparameter_risk_table(tbl)
                    display(styler)

        for w in (param_toggle, lead_drop, metric_drop):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([param_toggle]),
            widgets.HBox([lead_drop, metric_drop], layout=widgets.Layout(margin="4px 0 6px 0")),
        ])
        display(controls, out)
        _render()

    # -------------------------------------------------------------
    # Stage 6c: Risk–Return (Pareto) view of DA configurations
    # -------------------------------------------------------------
    def get_risk_return_table(self, leads=(4, 5, 6, 7), metric: str = "skill", agg: str = "mean",
                              harm_threshold: float = -0.01, koppen_filter="all"):
        """Per-config risk/return stats on the common basin set (see ``da_eval.risk_return``). Cached.

        Reference models (Global-Best / Per-Basin DA and PP) follow the session selection.
        """
        from da_eval import risk_return as rr
        from da_eval.dashboard import _apply_kg_filter, _parse_kg_filter
        key = (tuple(sorted(int(l) for l in leads)), metric, agg, float(harm_threshold), str(koppen_filter))
        cache = self.__dict__.setdefault("_rr_cache", {})
        if key not in cache:
            basins = None
            g, s = _parse_kg_filter(koppen_filter)
            if g or s:
                basins = _apply_kg_filter(self.get_basin_koppen(), koppen_filter)["Basin ID"].tolist()
            cache[key] = rr.build_risk_return_table(self.df_eval, leads=key[0], metric=metric, agg=agg,
                                                    harm_threshold=harm_threshold, basins=basins)
        return cache[key]

    def plot_risk_return(
        self,
        leads=(4, 5, 6, 7),
        metric: str = "skill",
        agg: str = "mean",
        risk_stat: str = "q10",
        return_stat: str = "median",
        color_by: Optional[str] = "Learning Rate",
        marker_by: Optional[str] = None,
        label_by: Optional[str] = None,
        fixed: Optional[Dict[str, object]] = None,
        show_refs=("baseline", "global_da", "per_basin_da", "global_rho", "per_basin_rho"),
        show_frontier: bool = True,
        label_frontier: int = 5,
        harm_threshold: float = -0.01,
        koppen_filter="all",
        annotate=(),
        interactive: bool = False,
    ):
        """Risk–return (Pareto) plot: one point per DA config; Y = typical per-basin gain, X = downside risk.

        Per-basin score = ``metric`` ('skill', 'delta' or 'nse') aggregated (``agg`` 'mean'/'sum') over the
        chosen ``leads`` (any subset, e.g. (7,), (4,5,6,7), (1,3,7)), on basins common to all configs.
        ``risk_stat`` / ``return_stat`` in: median, mean, q05, q10, q25, cvar10, pct_improved, pct_harm.
        ``color_by`` / ``marker_by`` / ``label_by``: any varying hyperparameter (Window (Days), Learning Rate,
        LR Ratio, BG Weight, Tolerance, Epochs, Sweep Group, Target, ...). ``fixed`` = {hyperparameter: value}
        highlights a controlled slice (other configs stay as the grey cloud). ``koppen_filter`` restricts basins.
        Reference markers follow the session selection (``session.set_selection``).
        Returns the frontier table (static mode).
        """
        from da_eval import risk_return as rr
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty.")
            return None

        def _draw(leads_, metric_, agg_, risk_, ret_, color_, marker_, label_, fixed_, refs_, front_, nlab_, harm_, kg_):
            tbl, info = self.get_risk_return_table(leads_, metric_, agg_, harm_, kg_)
            fig = rr.plot_risk_return(tbl, info, risk_stat=risk_, return_stat=ret_, color_by=color_,
                                      marker_by=marker_, label_by=label_, fixed=fixed_, show_refs=refs_,
                                      show_frontier=front_, label_frontier=nlab_, annotate=annotate)
            plt.show()
            plt.close(fig)
            ft = rr.frontier_table(tbl, risk_, ret_) if not tbl.empty else pd.DataFrame()
            if info.get("dropped"):
                print(f"[INFO] Dropped {len(info['dropped'])} configs with <90% basin coverage: {info['dropped'][:5]}...")
            return ft

        if not interactive:
            ft = _draw(leads, metric, agg, risk_stat, return_stat, color_by, marker_by, label_by, fixed,
                       show_refs, show_frontier, label_frontier, harm_threshold, koppen_filter)
            return ft

        import ipywidgets as widgets
        tbl0, _ = self.get_risk_return_table(leads, metric, agg, harm_threshold, koppen_filter)
        sweep0 = tbl0[tbl0["Is Sweep"]] if not tbl0.empty else pd.DataFrame()
        hp_all = [c for c in list(rr.HP_PATTERNS) + ["Target", "Loss"] if c in sweep0.columns]
        hp_vary = rr.varying_hyperparameters(sweep0[["Config ID"] + hp_all]) if not sweep0.empty else []
        hp_opts = [("(none)", None)] + [(c, c) for c in hp_vary]
        lay = lambda w: widgets.Layout(width=w)  # noqa: E731
        sty = {"description_width": "initial"}

        init_leads = {int(l) for l in leads}
        w_leads = [widgets.ToggleButton(value=lt in init_leads, description=f"t+{lt}", layout=lay("52px"))
                   for lt in self.lead_times]
        w_preset = widgets.Dropdown(options=[("Custom", "custom"), ("All", "all"), ("t+1", "1"), ("t+7", "7"),
                                             ("t+1–3", "1,2,3"), ("t+4–7", "4,5,6,7"), ("t+1,3,7", "1,3,7")],
                                    value="custom", description="Leads:", layout=lay("165px"), style=sty)
        w_agg = widgets.Dropdown(options=[("Mean over leads", "mean"), ("Sum over leads", "sum")], value=agg,
                                 layout=lay("150px"))
        w_metric = widgets.Dropdown(options=[("NSE skill score", "skill"), ("ΔNSE", "delta"), ("NSE (absolute)", "nse")],
                                    value=metric, description="Metric:", layout=lay("210px"), style=sty)
        stat_opts = [(lbl, k) for k, (lbl, _) in rr.STATS.items()]
        w_risk = widgets.Dropdown(options=stat_opts, value=risk_stat, description="X (risk):", layout=lay("270px"), style=sty)
        w_ret = widgets.Dropdown(options=stat_opts, value=return_stat, description="Y (return):", layout=lay("270px"), style=sty)
        w_harm = widgets.Dropdown(options=[("harm < −0.01", -0.01), ("harm < −0.05", -0.05), ("harm < 0", 0.0)],
                                  value=harm_threshold, layout=lay("130px"))
        w_color = widgets.Dropdown(options=hp_opts, value=color_by if color_by in hp_vary else None,
                                   description="Colour by:", layout=lay("230px"), style=sty)
        w_marker = widgets.Dropdown(options=hp_opts, value=marker_by if marker_by in hp_vary else None,
                                    description="Marker by:", layout=lay("230px"), style=sty)
        w_label = widgets.Dropdown(options=hp_opts, value=label_by if label_by in hp_vary else None,
                                   description="Label by:", layout=lay("230px"), style=sty)
        init_fixed = fixed or {}
        w_fixed = {}
        for c in hp_vary:
            vals = sorted(sweep0[c].dropna().unique(), key=lambda z: (str(type(z)), z))
            opts = [("any", "any")] + [(rr._fmt_val(v), rr._fmt_val(v)) for v in vals]
            iv = rr._fmt_val(init_fixed[c]) if c in init_fixed else "any"
            w_fixed[c] = widgets.Dropdown(options=opts, value=iv if iv in [o[1] for o in opts] else "any",
                                          description=f"{rr.HP_SHORT.get(c) or c}:", layout=lay("150px"), style=sty)
        ref_keys = [k for k in rr.DEFAULT_REFS if k == "baseline" or not tbl0.empty and (tbl0["Role"] == k).any()]
        w_refs = {k: widgets.ToggleButton(value=k in show_refs, description=rr.REF_STYLES[k]["label"],
                                          layout=lay("165px"), style={"text_color": rr.REF_STYLES[k]["color"]})
                  for k in ref_keys}
        w_front = widgets.Checkbox(value=show_frontier, description="Pareto frontier", indent=False, layout=lay("130px"))
        w_nlab = widgets.IntSlider(value=label_frontier, min=0, max=15, description="Label top:", layout=lay("220px"),
                                   style=sty, continuous_update=False)
        kg_opts = [("All climates", "all")] + [(f"{g} – {n}", g) for g, (n, _) in __import__(
            "da_eval.koppen", fromlist=["KG_GROUPS"]).KG_GROUPS.items()]
        w_kg = widgets.Dropdown(options=kg_opts, value=koppen_filter if koppen_filter in [o[1] for o in kg_opts] else "all",
                                description="Basins:", layout=lay("200px"), style=sty)
        out = widgets.Output()
        state = {"preset": False}

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel = [lt for lt, b in zip(self.lead_times, w_leads) if b.value] or [self.lead_times[-1]]
                fx = {c: w.value for c, w in w_fixed.items() if w.value != "any"}
                ft = _draw(sel, w_metric.value, w_agg.value, w_risk.value, w_ret.value, w_color.value,
                           w_marker.value, w_label.value, fx, [k for k, b in w_refs.items() if b.value],
                           w_front.value, w_nlab.value, w_harm.value, w_kg.value)
                if w_front.value and ft is not None and not ft.empty:
                    display(HTML("<b>Pareto-optimal configurations</b> (best return first)"))
                    display(rr.style_frontier_table(ft, metric=w_metric.value))

        def _apply_preset(ch):
            if ch["new"] == "custom":
                return
            wanted = set(self.lead_times) if ch["new"] == "all" else {int(x) for x in ch["new"].split(",")}
            state["preset"] = True
            try:
                for lt, b in zip(self.lead_times, w_leads):
                    b.value = lt in wanted
            finally:
                state["preset"] = False
            _render()

        def _on_lead(*_):
            if state["preset"]:
                return
            w_preset.unobserve(_apply_preset, names="value")
            w_preset.value = "custom"
            w_preset.observe(_apply_preset, names="value")
            _render()

        w_preset.observe(_apply_preset, names="value")
        for b in w_leads:
            b.observe(_on_lead, names="value")
        for w in (w_agg, w_metric, w_risk, w_ret, w_harm, w_color, w_marker, w_label, w_front, w_nlab, w_kg,
                  *w_fixed.values(), *w_refs.values()):
            w.observe(_render, names="value")

        box = lambda ws, m="0 0 4px 0": widgets.HBox(ws, layout=widgets.Layout(margin=m, flex_flow="row wrap"))  # noqa: E731
        controls = widgets.VBox([
            box([w_preset, *w_leads, w_agg]),
            box([w_metric, w_risk, w_ret, w_harm, w_kg]),
            box([w_color, w_marker, w_label, w_front, w_nlab]),
            box([widgets.HTML("<b style='margin-right:6px'>Hold fixed:</b>"), *w_fixed.values()]),
            box([widgets.HTML("<b style='margin-right:6px'>References:</b>"), *w_refs.values()], "0 0 8px 0"),
        ])
        display(controls, out)
        _render()
        return None

    # -------------------------------------------------------------
    # Stage 3: Cumulative Distribution Functions (Dual-Panel CDF Suite)
    # -------------------------------------------------------------
    def plot_ecdfs(
        self,
        lead_time: Optional[int] = None,
        progression_leads: Optional[List[int]] = None,
        metric: str = "skill_nse",
        xlim: Optional[Tuple[float, float]] = None,
        interactive: bool = False,
        show_decay: bool = True,
        show: Optional[List[str]] = None,
        show_sweep: bool = True,
        leads: Optional[List[int]] = None,
        show_b: Optional[List[str]] = None,
        lead_style: str = "marker",
        median_table: bool = True,
        xlim_b: Optional[Tuple[float, float]] = None,
        font_scale: float = 1.15,
        kind: str = "cdf",
        panels: str = "ab",
    ):
        """Renders the 2-Panel CDF Suite (and optional Horizon Decay curve):

        Args:
            lead_time: Primary forecast horizon (defaults to Lead 1).
            progression_leads: Horizons shown in Panel B (defaults to all available lead times), e.g. ``[1, 3, 7]``.
                    With interactive=True this seeds per-lead toggle buttons (plus presets) below the controls.
            metric: 'skill_nse' (Per-Gauge NSE Skill Score), 'delta_nse' (Per-Gauge ΔNSE),
                    'delta_kge' (Per-Gauge ΔKGE), 'NSE' (Absolute NSE), or 'KGE' (Absolute KGE).
            xlim: Custom x-axis limits (auto-scales to (-0.5, 0.5) for deltas and (-1.0, 1.0) for skill/absolutes).
            interactive: If True, renders an ipywidgets toggle bar to switch between
                         NSE Skill Score, ΔNSE, ΔKGE, Absolute NSE, and Absolute KGE in a single cell.
            show_decay: When interactive=True, includes a toggleable Horizon Persistence Decay curve.
            show: Benchmark models to draw, any of ``"global_da"``, ``"per_basin_da"``, ``"global_rho"``
                  (best fixed-rho PP post-processor) and ``"per_basin_rho"`` (best rho per gauge).
                  Defaults to the two DA models; interactive=True adds a toggle button per model.
            show_sweep: Draw the thin per-config sweep curves in Panel A.
            leads: Alias of ``progression_leads``.
            show_b: Models in Panel B (default: all of ``show`` on SS / delta metrics; DA models only on
                    absolute NSE / KGE, where the baseline curves are drawn instead).
            lead_style: ``"marker"`` (colour + line style = model, marker = lead) or ``"alpha"`` (old look).
            median_table: Show the per-model / per-lead median (and win-rate) table under the figure.
            xlim_b: Panel B x-limits (default: -25%..+50% for SS / delta, -0.25..1 for absolute metrics).
            font_scale: Multiplier on all figure font sizes.
            kind: ``"cdf"`` (default) or ``"box"`` (same data as box-and-whisker plots: (a) one box per model,
                  (b) boxes per model per lead; box = IQR, whiskers = Q10-Q90). The viewer has a toggle for it.
            panels: ``"ab"`` (default) or ``"b"`` = only the multi-day panel (one figure, all selected leads).
        """
        if leads is not None:
            progression_leads = leads
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot CDFs.")
            return
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])

        if interactive:
            return self.launch_cdf_viewer(
                default_metric=metric,
                default_lead=ref_lead,
                progression_leads=progression_leads,
                show_decay=show_decay,
                show=show,
                show_sweep=show_sweep,
                show_b=show_b,
                lead_style=lead_style,
                median_table=median_table,
                xlim_b=xlim_b,
                font_scale=font_scale,
                kind=kind,
                panels=panels,
            )

        fig = plot_ecdf_suite(
            self.df_eval,
            lead_time=ref_lead,
            progression_leads=progression_leads,
            metric=metric,
            xlim=xlim,
            show=show,
            show_sweep=show_sweep,
            show_b=show_b,
            lead_style=lead_style,
            xlim_b=xlim_b,
            font_scale=font_scale,
            kind=kind,
            panels=panels,
        )
        display(fig)
        plt.close(fig)
        self._show_ecdf_medians(fig, median_table)

    @staticmethod
    def _show_ecdf_medians(fig, median_table: bool = True):
        med = getattr(fig, "_ecdf_medians", None)
        if median_table and med is not None and not med.empty:
            display(format_ecdf_median_table(med))

    def launch_cdf_viewer(
        self,
        default_metric: str = "skill_nse",
        default_lead: Optional[int] = None,
        progression_leads: Optional[List[int]] = None,
        show_decay: bool = True,
        show: Optional[List[str]] = None,
        show_sweep: bool = True,
        show_b: Optional[List[str]] = None,
        lead_style: str = "marker",
        median_table: bool = True,
        xlim_b: Optional[Tuple[float, float]] = None,
        font_scale: float = 1.15,
        kind: str = "cdf",
        panels: str = "ab",
    ):
        """Renders an interactive single-cell CDF & Horizon Decay viewer with metric toggles (NSE Skill Score, ΔNSE, ΔKGE, NSE, KGE)."""
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot launch CDF viewer.")
            return

        ref_lead = default_lead if default_lead is not None else (1 if 1 in self.lead_times else self.lead_times[0])

        try:
            import ipywidgets as widgets
        except ImportError:
            self.plot_ecdfs(lead_time=ref_lead, progression_leads=progression_leads, metric=default_metric, interactive=False,
                            kind=kind, panels=panels)
            if show_decay:
                self.plot_leadtime_decay(metric=default_metric)
            return

        metric_options = [
            ("NSE Skill Score (SS)", "skill_nse"),
            ("ΔNSE (Per-Gauge Gain)", "delta_nse"),
            ("ΔKGE (Per-Gauge Gain)", "delta_kge"),
            ("Absolute NSE", "NSE"),
            ("Absolute KGE", "KGE"),
        ]
        m_in = str(default_metric).strip()
        if "skill" in m_in.lower():
            norm_default = "skill_nse"
        elif m_in in ("delta_nse", "delta_kge", "NSE", "KGE"):
            norm_default = m_in
        else:
            norm_default = "skill_nse"

        w_metric = widgets.ToggleButtons(
            options=metric_options,
            value=norm_default,
            description="Metric:",
            button_style="info",
            style={"description_width": "initial", "button_width": "155px"},
        )
        w_lead = widgets.Dropdown(
            options=[(f"Lead t+{lt} Day{'s' if lt != 1 else ''}", lt) for lt in self.lead_times],
            value=ref_lead,
            description="Panel A Horizon:",
            layout=widgets.Layout(width="210px"),
            style={"description_width": "initial"},
        )
        w_xlim = widgets.Dropdown(
            options=[
                ("Auto Scale", "auto"),
                ("[-0.2, +0.2]", "0.2"),
                ("[-0.3, +0.3]", "0.3"),
                ("[-0.5, +0.5]", "0.5"),
                ("[-1.0, +1.0]", "1.0"),
            ],
            value="auto",
            description="X-Axis Range:",
            layout=widgets.Layout(width="200px"),
            style={"description_width": "initial"},
        )
        w_kind = widgets.ToggleButtons(
            options=[("CDF", "cdf"), ("Box plot", "box")],
            value="box" if str(kind).lower().startswith("box") else "cdf",
            description="Plot:",
            style={"description_width": "initial", "button_width": "90px"},
        )
        w_panels = widgets.ToggleButtons(
            options=[("Multi-day only", "b"), ("Both panels", "ab")],
            value="b" if str(panels).lower().strip() in ("b", "multi", "multi_day", "leads") else "ab",
            description="Panels:",
            style={"description_width": "initial", "button_width": "120px"},
        )
        w_decay = widgets.Checkbox(
            value=show_decay,
            description="Show Horizon Decay Curve (t+1..t+7)",
            indent=False,
            layout=widgets.Layout(width="260px"),
        )

        # Model toggles: which benchmark curves to draw (DA vs PP post-processing, global vs per-basin).
        init_show = set(DEFAULT_ECDF_SHOW if show is None else show)
        has_rho = self.df_eval["Config ID"].apply(is_rho_config).any()
        _model_keys = [k for k in ECDF_MODELS if has_rho or "rho" not in k]
        _model_colors = {k: palette.color(k) for k in ECDF_MODELS}
        w_models = {
            k: widgets.ToggleButton(
                value=k in init_show,
                description=ECDF_MODEL_LABELS[k],
                layout=widgets.Layout(width="150px"),
                style={"text_color": _model_colors[k], "font_weight": "bold"},
                tooltip=f"Show {ECDF_MODEL_LABELS[k]} in Panel A, Panel B and the decay curve",
            )
            for k in _model_keys
        }
        w_sweep = widgets.Checkbox(value=show_sweep, description="Sweep configs (Panel A)", indent=False,
                                   layout=widgets.Layout(width="190px"))

        def _sel_models():
            return [k for k, b in w_models.items() if b.value]

        # Panel B horizon toggles (one button per lead; seeded from progression_leads, default = all leads).
        init_prog = set(progression_leads) if progression_leads else set(self.lead_times)
        w_prog = [
            widgets.ToggleButton(
                value=lt in init_prog,
                description=f"t+{lt}",
                button_style="",
                layout=widgets.Layout(width="52px"),
                tooltip=f"Show lead t+{lt} in the multi-horizon CDF",
            )
            for lt in self.lead_times
        ]
        w_preset = widgets.Dropdown(
            options=[("Custom", "custom"), ("All", "all"), ("1, 3, 7", "1,3,7"), ("1, 3, 5, 7", "1,3,5,7"), ("1, 7", "1,7")],
            value="custom",
            description="Preset:",
            layout=widgets.Layout(width="160px"),
            style={"description_width": "initial"},
        )

        # A fresh Output per render (swapped into this box) instead of clear_output(wait=True) on one Output:
        # some front-ends (VS Code) do not honour the deferred clear, so every re-render appended another copy.
        out = widgets.VBox()

        def _sel_prog_leads():
            leads = [lt for lt, b in zip(self.lead_times, w_prog) if b.value]
            return leads or [w_lead.value]

        def _render(*_):
            plt.close("all")
            fresh = widgets.Output()
            out.children = (fresh,)
            with fresh:
                sel_metric = w_metric.value
                sel_lead = w_lead.value
                sel_xlim_str = w_xlim.value
                if sel_xlim_str == "auto":
                    sel_xlim = None
                else:
                    bound = float(sel_xlim_str)
                    sel_xlim = (-bound, bound) if ("delta" in sel_metric or "skill" in sel_metric) else (-bound, 1.0)

                fig1 = plot_ecdf_suite(
                    self.df_eval,
                    lead_time=sel_lead,
                    progression_leads=_sel_prog_leads(),
                    metric=sel_metric,
                    xlim=sel_xlim,
                    show=_sel_models(),
                    show_sweep=w_sweep.value,
                    show_b=None if show_b is None else [k for k in show_b if k in _sel_models()],
                    lead_style=lead_style,
                    xlim_b=xlim_b if sel_xlim is None else sel_xlim,
                    font_scale=font_scale,
                    kind=w_kind.value,
                    panels=w_panels.value,
                )
                display(fig1)
                plt.close(fig1)
                self._show_ecdf_medians(fig1, median_table)

                if w_decay.value:
                    fig2 = plot_leadtime_decay(
                        self.df_eval,
                        lead_times=self.lead_times,
                        metric=sel_metric,
                        show=_sel_models(),
                    )
                    display(fig2)
                    plt.close(fig2)

        state = {"applying_preset": False}

        def _apply_preset(change):
            if change["new"] == "custom":
                return
            wanted = set(self.lead_times) if change["new"] == "all" else {int(x) for x in change["new"].split(",")}
            state["applying_preset"] = True
            try:
                for lt, b in zip(self.lead_times, w_prog):
                    b.value = lt in wanted
            finally:
                state["applying_preset"] = False
            _render()

        def _on_toggle(*_):
            if state["applying_preset"]:
                return
            w_preset.unobserve(_apply_preset, names="value")
            w_preset.value = "custom"
            w_preset.observe(_apply_preset, names="value")
            _render()

        for w in (w_metric, w_kind, w_panels, w_lead, w_xlim, w_decay, w_sweep, *w_models.values()):
            w.observe(_render, names="value")
        for b in w_prog:
            b.observe(_on_toggle, names="value")
        w_preset.observe(_apply_preset, names="value")

        controls = widgets.VBox([
            w_metric,
            widgets.HBox([w_kind, w_panels, w_decay], layout=widgets.Layout(margin="6px 0 4px 0", align_items="center")),
            widgets.HBox([w_lead, w_xlim], layout=widgets.Layout(margin="0 0 4px 0", align_items="center")),
            widgets.HBox([widgets.Label("Panel B Horizons:")] + w_prog + [w_preset],
                         layout=widgets.Layout(margin="0 0 4px 0", align_items="center")),
            widgets.HBox([widgets.Label("Models:")] + list(w_models.values()) + [w_sweep],
                         layout=widgets.Layout(margin="0 0 4px 0", align_items="center")),
        ])
        display(controls, out)
        _render()

    # -------------------------------------------------------------
    # Stage 3b: Stratified Catchment ECDF & Basin-Type Leaderboards
    # -------------------------------------------------------------
    def plot_stratified_cdfs(
        self,
        lead_time: int = 1,
        metric: str = "skill_nse",
        top_n: int = 5,
        base_nse_range: Optional[Tuple[float, float]] = None,
        aridity_range: Optional[Tuple[float, float]] = None,
        p_mean_range: Optional[Tuple[float, float]] = None,
        area_range: Optional[Tuple[float, float]] = None,
        snow_range: Optional[Tuple[float, float]] = None,
        interactive: bool = True,
        figsize: Tuple[int, int] = (18, 6.5),
    ):
        """Plots Empirical CDF and Horizon Persistence Decay strictly for catchments meeting physical criteria.

        Enables testing whether specific configurations excel on particular basin types (e.g. Arid, Low Base NSE, High Precip).
        Also outputs a ranked leaderboard highlighting the #1 winning configuration on that specific catchment stratum.

        Args:
            lead_time: Forecast horizon in days (e.g. 1, 4, 7).
            metric: 'skill_nse', 'delta_nse', 'delta_kge', 'NSE', or 'KGE'.
            top_n: Number of top configs to plot on the CDF.
            base_nse_range: (min_base_nse, max_base_nse) tuple.
            aridity_range: (min_aridity, max_aridity) tuple (unep_aridity_index = P / PET).
            p_mean_range: (min_p, max_p) tuple in mm/day.
            area_range: (min_area, max_area) tuple in km².
            snow_range: (min_snow, max_snow) tuple fraction [0..1].
            interactive: If True, renders interactive widgets for real-time attribute filtering.
            figsize: Figure dimensions (width, height).
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot stratified CDFs.")
            return

        if interactive:
            self.launch_stratified_ecdf_viewer(
                default_lead=lead_time,
                default_metric=metric,
                default_top_n=top_n,
                init_base_range=base_nse_range,
                init_aridity_range=aridity_range,
                init_precip_range=p_mean_range,
                figsize=figsize,
            )
            return

        fig, df_lb, cohort_info = plot_stratified_ecdf_suite(
            df_eval=self.df_eval,
            df_meta=self.df_meta,
            lead_time=lead_time,
            metric=metric,
            top_n_configs=top_n,
            base_nse_range=base_nse_range,
            aridity_range=aridity_range,
            p_mean_range=p_mean_range,
            area_range=area_range,
            snow_range=snow_range,
            figsize=figsize,
        )
        if not df_lb.empty:
            print(f"\n--- Stratified Catchment Leaderboard: {cohort_info['desc']} ---")
            print(f"Cohort Size: {cohort_info['n_basins']:,} / {cohort_info['total_basins']:,} catchments ({cohort_info['pct_total']:.1f}%)")
            if "Median NSE Skill" in df_lb.columns and units.ss_percent():
                display(df_lb.style.format({"Median NSE Skill": units.ss_formatter()}).format_index(
                    lambda c: units.ss_label(c) if c == "Median NSE Skill" else c, axis=1))
            else:
                display(df_lb)
        plt.show()
        plt.close(fig)

    def launch_stratified_ecdf_viewer(
        self,
        default_lead: int = 1,
        default_metric: str = "skill_nse",
        default_top_n: int = 5,
        init_base_range: Optional[Tuple[float, float]] = None,
        init_aridity_range: Optional[Tuple[float, float]] = None,
        init_precip_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (18, 6.5),
    ):
        """Launches an interactive widget control bar to filter CDFs and leaderboards by catchment attributes."""
        try:
            import ipywidgets as widgets
        except ImportError:
            self.plot_stratified_cdfs(lead_time=default_lead, metric=default_metric, interactive=False, figsize=figsize)
            return

        ref_lead = default_lead if default_lead in self.lead_times else (1 if 1 in self.lead_times else self.lead_times[0])

        w_preset = widgets.Dropdown(
            options=[
                ("All Catchments (No Filter)", "all"),
                ("Low Baseline NSE (< 0.50)", "low_base"),
                ("Mid Baseline NSE (0.50 - 0.70)", "mid_base"),
                ("High Baseline NSE (>= 0.70)", "high_base"),
                ("Arid / Semi-Arid (P/PET < 0.50)", "arid"),
                ("Sub-Humid (P/PET 0.50 - 0.80)", "subhumid"),
                ("Humid (P/PET >= 0.80)", "humid"),
                ("Low Precipitation (< 1.5 mm/d)", "low_p"),
                ("Moderate Precipitation (1.5 - 3.5 mm/d)", "mod_p"),
                ("High Precipitation (> 3.5 mm/d)", "high_p"),
                ("Small Catchments (< 500 km²)", "small_area"),
                ("Large Catchments (> 2,500 km²)", "large_area"),
                ("Snow-melt Dominant (Snow Frac >= 0.30)", "snow"),
                ("Custom Range Sliders", "custom"),
            ],
            value="all",
            description="Catchment Stratum Preset:",
            style={"description_width": "180px"},
            layout=widgets.Layout(width="420px"),
        )

        m_strat_in = str(default_metric).strip()
        if "skill" in m_strat_in.lower():
            norm_strat_metric = "skill_nse"
        elif m_strat_in in ("delta_nse", "delta_kge", "NSE", "KGE"):
            norm_strat_metric = m_strat_in
        else:
            norm_strat_metric = "skill_nse"

        w_metric = widgets.ToggleButtons(
            options=[
                ("NSE Skill Score (SS)", "skill_nse"),
                ("Per-Gauge ΔNSE", "delta_nse"),
                ("Per-Gauge ΔKGE", "delta_kge"),
                ("Absolute NSE", "NSE"),
                ("Absolute KGE", "KGE"),
            ],
            value=norm_strat_metric,
            description="Metric:",
            button_style="info",
            style={"description_width": "initial", "button_width": "145px"},
        )

        w_lead = widgets.Dropdown(
            options=[(f"Lead t+{lt}", lt) for lt in self.lead_times],
            value=ref_lead,
            description="Horizon:",
            layout=widgets.Layout(width="180px"),
            style={"description_width": "initial"},
        )

        w_top_n = widgets.Dropdown(
            options=[("Top 3", 3), ("Top 5", 5), ("Top 8", 8), ("All Configs", 99)],
            value=default_top_n if default_top_n in (3, 5, 8, 99) else 5,
            description="Top N Configs:",
            layout=widgets.Layout(width="180px"),
            style={"description_width": "initial"},
        )

        w_base_nse = widgets.FloatRangeSlider(
            value=init_base_range if init_base_range else (-1.0, 1.0),
            min=-1.0,
            max=1.0,
            step=0.05,
            description="Base NSE:",
            readout_format=".2f",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="340px"),
        )

        w_aridity = widgets.FloatRangeSlider(
            value=init_aridity_range if init_aridity_range else (0.0, 4.0),
            min=0.0,
            max=4.0,
            step=0.05,
            description="Aridity (P/PET):",
            readout_format=".2f",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="340px"),
        )

        w_precip = widgets.FloatRangeSlider(
            value=init_precip_range if init_precip_range else (0.0, 15.0),
            min=0.0,
            max=15.0,
            step=0.2,
            description="Precip (mm/d):",
            readout_format=".1f",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="340px"),
        )

        out = widgets.Output()

        def _on_preset_change(change):
            preset = change.get("new")
            if preset == "all":
                w_base_nse.value = (-1.0, 1.0)
                w_aridity.value = (0.0, 4.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "low_base":
                w_base_nse.value = (-1.0, 0.50)
                w_aridity.value = (0.0, 4.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "mid_base":
                w_base_nse.value = (0.50, 0.70)
                w_aridity.value = (0.0, 4.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "high_base":
                w_base_nse.value = (0.70, 1.0)
                w_aridity.value = (0.0, 4.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "arid":
                w_aridity.value = (0.0, 0.50)
                w_base_nse.value = (-1.0, 1.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "subhumid":
                w_aridity.value = (0.50, 0.80)
                w_base_nse.value = (-1.0, 1.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "humid":
                w_aridity.value = (0.80, 4.0)
                w_base_nse.value = (-1.0, 1.0)
                w_precip.value = (0.0, 15.0)
            elif preset == "low_p":
                w_precip.value = (0.0, 1.5)
                w_aridity.value = (0.0, 4.0)
                w_base_nse.value = (-1.0, 1.0)
            elif preset == "mod_p":
                w_precip.value = (1.5, 3.5)
                w_aridity.value = (0.0, 4.0)
                w_base_nse.value = (-1.0, 1.0)
            elif preset == "high_p":
                w_precip.value = (3.5, 15.0)
                w_aridity.value = (0.0, 4.0)
                w_base_nse.value = (-1.0, 1.0)

        w_preset.observe(_on_preset_change, names="value")

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_lead = int(w_lead.value)
                sel_metric = str(w_metric.value)
                sel_top_n = int(w_top_n.value)

                # Read range bounds, treating extremes as unbounded
                base_rng = w_base_nse.value if (w_base_nse.value[0] > -0.99 or w_base_nse.value[1] < 0.99) else None
                arid_rng = w_aridity.value if (w_aridity.value[0] > 0.01 or w_aridity.value[1] < 3.99) else None
                prec_rng = w_precip.value if (w_precip.value[0] > 0.05 or w_precip.value[1] < 14.95) else None

                fig, df_lb, cohort_info = plot_stratified_ecdf_suite(
                    df_eval=self.df_eval,
                    df_meta=self.df_meta,
                    lead_time=sel_lead,
                    metric=sel_metric,
                    top_n_configs=sel_top_n,
                    base_nse_range=base_rng,
                    aridity_range=arid_rng,
                    p_mean_range=prec_rng,
                    figsize=figsize,
                )

                if not df_lb.empty:
                    header_html = (
                        f"<div style='margin-bottom:8px;padding:8px 12px;background:#f8f9fa;border-left:4px solid #1a73e8;border-radius:4px;'>"
                        f"<b>Catchment Stratum Leaderboard (Lead t+{sel_lead})</b> &mdash; "
                        f"<code>{cohort_info['desc']}</code><br>"
                        f"<span style='color:#555;font-size:12px;'>"
                        f"Active Cohort: <b>{cohort_info['n_basins']:,}</b> / {cohort_info['total_basins']:,} catchments ({cohort_info['pct_total']:.1f}%) | "
                        f"Median Base NSE: <b>{cohort_info['median_base_nse']:.3f}</b> | "
                        f"Median Aridity: <b>{cohort_info['median_aridity']:.2f}</b> | "
                        f"Median Precip: <b>{cohort_info['median_p_mean']:.1f} mm/d</b></span>"
                        f"</div>"
                    )
                    display(HTML(header_html))

                    # Style table cleanly
                    def _color_delta(val):
                        try:
                            v = float(val)
                            if v > 0.01:
                                return "color: #137333; font-weight: bold; background-color: #e6f4ea;"
                            elif v < -0.01:
                                return "color: #c5221f; font-weight: bold; background-color: #fce8e6;"
                            return ""
                        except Exception:
                            return ""

                    fmt_dict = {
                        "Median Base NSE": "{:.3f}",
                        "Median DA NSE": "{:.3f}",
                        "Median ΔNSE": "{:+.3f}",
                        "Median NSE Skill": units.ss_formatter(),
                        "Median Base KGE": "{:.3f}",
                        "Median DA KGE": "{:.3f}",
                        "Median ΔKGE": "{:+.3f}",
                        "Median KGE Skill": "{:+.3f}",
                        "Win Rate (Δ>0.01)": "{:.1f}%",
                        "Degraded (Δ<-0.01)": "{:.1f}%",
                        "Basins": "{:,d}",
                    }
                    avail_fmt = {k: v for k, v in fmt_dict.items() if k in df_lb.columns}
                    color_cols = [c for c in ["Median ΔNSE", "Median NSE Skill", "Median ΔKGE", "Median KGE Skill"] if c in df_lb.columns]
                    styler = (
                        df_lb.style
                        .format(avail_fmt, na_rep="—")
                        .applymap(_color_delta, subset=color_cols if color_cols else None)
                        .set_properties(**{"text-align": "center", "font-size": "12px", "padding": "5px 10px"})
                        .set_properties(subset=["Config ID"], **{"text-align": "left", "font-weight": "bold"})
                        .set_table_styles([
                            {"selector": "th", "props": [("background-color", "#1a73e8"), ("color", "white"), ("font-weight", "bold"), ("text-align", "center"), ("padding", "6px 10px")]},
                            {"selector": "tr:hover", "props": [("background-color", "#f8f9fa")]},
                        ])
                    )
                    if "Median NSE Skill" in df_lb.columns and units.ss_percent():
                        styler = styler.format_index(lambda c: units.ss_label(c) if c == "Median NSE Skill" else c, axis=1)
                    display(styler)
                    print()

                plt.show()
                plt.close(fig)

        for w in (w_preset, w_metric, w_lead, w_top_n, w_base_nse, w_aridity, w_precip):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HTML("<h4>Interactive Stratified Catchment ECDF & Leaderboard Suite</h4>"
                         "<p style='font-size:12px;color:#555;margin-top:-6px;'>"
                         "Filter by Baseline Performance Tier, Aridity (P/PET), or Mean Precipitation to discover which "
                         "DA configuration performs best for each catchment morphology.</p>"),
            widgets.HBox([w_preset, w_lead, w_top_n]),
            w_metric,
            widgets.HBox([w_base_nse, w_aridity, w_precip], layout=widgets.Layout(margin="6px 0 4px 0", align_items="center")),
        ])
        display(controls, out)
        _render()

    # -------------------------------------------------------------
    # Stage 4: Leadtime Memory Decay
    # -------------------------------------------------------------
    def plot_leadtime_decay(
        self,
        lead_times: Optional[List[int]] = None,
        metric: str = "NSE",
        show: Optional[List[str]] = None,
    ):
        """Plots persistence decay curve across forecast horizons for 'NSE', 'KGE', 'delta_nse', or 'delta_kge'.

        ``show`` selects the benchmark models (see ``plot_ecdfs``).
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot leadtime decay.")
            return
        lts = lead_times if lead_times is not None else self.lead_times
        fig = plot_leadtime_decay(self.df_eval, lead_times=lts, metric=metric, show=show)
        plt.show()

    # -------------------------------------------------------------
    # Stage 5: Basin Physical Characteristics Efficacy
    # -------------------------------------------------------------
    def get_physical_strata(
        self,
        config: str = "global_best",
        lead_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Computes or retrieves physical stratification statistics table.

        Args:
            config: 'global_best' (default, single deployable best model),
                    'per_basin_best' (oracle upper bound across all configs),
                    or an explicit configuration ID string.
            lead_time: Forecast horizon in days (defaults to lead 1).
        """
        if self.df_eval.empty:
            return pd.DataFrame()
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        if config == "global_best":
            cfg_id = get_global_best_config(self.df_eval, lead_time=ref_lead)
        elif config == "per_basin_best":
            cfg_id = "Per-Basin Best DA"
        else:
            cfg_id = config

        return compute_consolidated_physical_strata(
            self.df_eval,
            df_ts=self.df_ts,
            df_meta=self.df_meta,
            lead_time=ref_lead,
            config_id=cfg_id,
        )

    # Catchment-characteristic efficacy (one common gauge set, notebook-wide reference models)
    _BC_CONFIG_MODELS = {
        "both": ("Global-Best DA", "Per-Basin DA"), "compare": ("Global-Best DA", "Per-Basin DA"),
        "all": _bc.ALL_MODELS, "global_best": ("Global-Best DA",), "per_basin_best": ("Per-Basin DA",),
    }

    def _bc_leads(self, leads=None, lead_time: Optional[int] = None) -> Tuple[int, ...]:
        if leads is not None:
            return tuple(int(l) for l in np.atleast_1d(leads))
        if lead_time is not None:
            return (int(lead_time),)
        return tuple(self.selection.policy.leads)

    def _bc_lead_text(self, leads) -> str:
        leads = tuple(leads)
        if len(leads) == 1:
            return f"Day {leads[0]}"
        contiguous = list(leads) == list(range(leads[0], leads[-1] + 1))
        return f"Days {leads[0]}–{leads[-1]} (mean)" if contiguous else "Days " + ", ".join(map(str, leads)) + " (mean)"

    def get_basin_characteristics_frame(self, leads=None) -> pd.DataFrame:
        """Per-gauge skill score of every reference model (mean over ``leads``; default = selection leads),
        Baseline NSE and catchment attributes, on one common gauge set."""
        leads = self._bc_leads(leads)
        cache = self.__dict__.setdefault("_bc_cache", {})
        key = (leads, self.selection.global_da, self.selection.global_rho)
        if key not in cache:
            cache[key] = _bc.gauge_frame(self, leads)
        return cache[key][0]

    def basin_characteristic_table(self, dimension: str, leads=None, models: Optional[Sequence[str]] = None,
                                   min_n: int = 20, style: bool = True):
        """Small per-bin table for one characteristic (e.g. 'Aridity', 'Area', 'Baseline NSE')."""
        df = self.get_basin_characteristics_frame(leads)
        tbl, p = _bc.characteristic_table(df, dimension, list(models or self._BC_CONFIG_MODELS["both"]), min_n=min_n)
        return _bc.style_characteristic_table(tbl, p) if style else (tbl, p)

    def plot_basin_characteristics_grid(self, leads=None, models: Sequence[str] = ("Global-Best DA", "Per-Basin DA"),
                                        color_by: Optional[str] = None, stat: str = "median", min_n: int = 20,
                                        title: Optional[str] = None, figsize=None, vmax=None):
        """Heat-map grid: rows = characteristics, columns = classes (low -> high), cell = 'Global / Per-Basin'
        skill score, colour = ``color_by`` (default first model). Leads default to the notebook SELECTION."""
        leads = self._bc_leads(leads)
        df = self.get_basin_characteristics_frame(leads)
        fig = _bc.plot_characteristics_heatmap(df, models, color_by=color_by, min_n=min_n, stat=stat, vmax=vmax,
                                               title=title or f"{stat.capitalize()} NSE skill score per class — "
                                               f"{self._bc_lead_text(leads)} ({len(df):,} gauges)", figsize=figsize)
        plt.show()
        return fig

    def plot_basin_characteristics_efficacy(
        self,
        dimension: Optional[str] = None,
        leads=None,
        models: Optional[Sequence[str]] = None,
        interactive: bool = False,
        min_n: int = 20,
        show_table: bool = True,
        config: Optional[str] = None,
        lead_time: Optional[int] = None,
        default_table: Optional[str] = None,  # accepted for backwards compatibility (ignored)
        title: Optional[str] = None,
        figsize=None,
    ):
        """Skill score by catchment characteristic, one characteristic at a time.

        Args:
            dimension: 'Baseline NSE', 'Aridity', 'Area', 'Snow', 'Slope', 'Precip', or None / 'overview' for
                the class heat-map grid (first two models) + one-row-per-characteristic summary table.
            leads: lead(s) to score; a window is the per-gauge mean. Default: the notebook SELECTION leads.
            models: subset of 'Global-Best DA', 'Per-Basin DA', 'Global-Best PP', 'Per-Basin PP'
                (default: the two DA models). Models always come from the notebook-wide selection.
            interactive: toggle bar (characteristic / lead / models) in one cell.
            min_n: bins with fewer gauges are merged into their smaller neighbour.
            config / lead_time: legacy aliases ('both' -> both DA models; lead_time -> leads=(lead_time,)).
        The pre-2026-10 all-in-one version is kept as ``plot_basin_characteristics_efficacy_legacy``.
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot basin characteristics efficacy.")
            return
        if models is None:
            models = self._BC_CONFIG_MODELS.get(str(config or "both").lower(), ("Global-Best DA", "Per-Basin DA"))
        leads = self._bc_leads(leads, lead_time)
        if interactive:
            return self._basin_characteristics_viewer(dimension, leads, list(models), min_n)
        return self._render_basin_characteristics(dimension, leads, list(models), min_n, show_table, title, figsize)

    def _render_basin_characteristics(self, dimension, leads, models, min_n, show_table=True, title=None,
                                      figsize=None):
        df = self.get_basin_characteristics_frame(leads)
        lead_txt = self._bc_lead_text(leads)
        head = f"{lead_txt} · {len(df):,} gauges with a score for every reference model"
        if dimension is None or str(dimension).lower() == "overview":
            display(HTML(f"<b>Which characteristics matter?</b> {head}. Gap = difference in median skill score "
                         "between the most and least helped bin; p = Kruskal–Wallis test that bins differ."))
            fig = _bc.plot_characteristics_heatmap(df, models[:2], min_n=min_n,
                                                   title=title or f"Median NSE skill score per class — {lead_txt}",
                                                   figsize=figsize)
            plt.show()
            display(_bc.style_overview_table(_bc.overview_table(df, models, min_n=min_n)))
            return fig
        d = _bc.resolve_dimension(dimension)
        fig = _bc.plot_characteristic(df, d, models, min_n=min_n, title=title or f"{d.label} — {lead_txt}")
        if figsize:
            fig.set_size_inches(*figsize)
        plt.show()
        if show_table:
            tbl, p = _bc.characteristic_table(df, d, models, min_n=min_n)
            display(_bc.style_characteristic_table(tbl, p))
        return fig

    def _basin_characteristics_viewer(self, dimension, leads, models, min_n):
        try:
            import ipywidgets as widgets
        except ImportError:
            return self._render_basin_characteristics(dimension, leads, models, min_n)
        dims = [("Overview", "overview")] + [(d.label.split(" (")[0], d.key) for d in _bc.DIMENSIONS.values()]
        init_dim = "overview" if dimension is None else _bc.resolve_dimension(dimension).key \
            if str(dimension).lower() != "overview" else "overview"
        dim_tb = widgets.ToggleButtons(options=dims, value=init_dim, description="Characteristic:",
                                       style={"description_width": "initial", "button_width": "auto"})
        sel_leads = tuple(self.selection.policy.leads)
        lead_opts = [(f"Day {l}", (l,)) for l in self.lead_times]
        windows = [("Days 1–3", (1, 2, 3)), ("Days 4–7", (4, 5, 6, 7)), ("Days 1–7", tuple(self.lead_times))]
        lead_opts += [w for w in windows if set(w[1]) <= set(self.lead_times)]
        if sel_leads not in [o[1] for o in lead_opts]:
            lead_opts.append((f"Selection ({self._bc_lead_text(sel_leads)})", sel_leads))
        lead_tb = widgets.ToggleButtons(options=lead_opts, value=leads if leads in [o[1] for o in lead_opts] else sel_leads,
                                        description="Lead:", style={"description_width": "initial", "button_width": "auto"})
        avail = [m for m in _bc.ALL_MODELS if m in self.get_basin_characteristics_frame(sel_leads)]
        boxes = {m: widgets.Checkbox(value=m in models, description=m, indent=False,
                                     layout=widgets.Layout(width="auto", margin="0 14px 0 0")) for m in avail}
        out = widgets.VBox()  # fresh Output per render (see launch_cdf_viewer)

        def _render(*_):
            ms = [m for m, b in boxes.items() if b.value] or avail[:1]
            plt.close("all")
            fresh = widgets.Output()
            out.children = (fresh,)
            with fresh:
                self._render_basin_characteristics(None if dim_tb.value == "overview" else dim_tb.value,
                                                   tuple(lead_tb.value), ms, min_n)

        for w in [dim_tb, lead_tb, *boxes.values()]:
            w.observe(_render, names="value")
        display(widgets.VBox([dim_tb, lead_tb, widgets.HBox([widgets.Label("Models:", layout=widgets.Layout(width="90px")),
                                                             *boxes.values()])]), out)
        _render()

    def plot_basin_characteristics_efficacy_legacy(
        self,
        config: str = "global_best",
        lead_time: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 10),
        interactive: bool = False,
        default_table: str = "global_best",
    ):
        """Plots the unified 2x3 horizontal diverging effect-size bar grid.

        Args:
            config: 'both' (renders grouped bars comparing Global Best DA vs Oracle Per-Basin Best),
                    'global_best' (default, evaluates the single best deployable model),
                    'per_basin_best' (oracle upper bound across all sweep configs),
                    or an explicit configuration ID string.
            lead_time: Forecast horizon in days (defaults to lead 1).
            title: Custom suptitle.
            figsize: Figure dimensions.
            interactive: If True, renders a single-cell ipywidgets viewer to switch between
                         Comparison Plot, Global Best DA, Per-Basin Oracle, and Table modes.
            default_table: Initial table selection when interactive=True ('global_best', 'per_basin_best', 'both', 'none').
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot basin characteristics efficacy.")
            return
        if interactive:
            self.launch_physical_characteristics_viewer(
                default_config=config,
                default_table=default_table,
                default_lead=lead_time,
                figsize=figsize,
            )
            return

        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        if config.lower() in ("both", "compare", "all"):
            df_primary = self.get_physical_strata(config="global_best", lead_time=ref_lead)
            df_compare = self.get_physical_strata(config="per_basin_best", lead_time=ref_lead)
            if df_primary.empty and df_compare.empty:
                print("[WARNING] No physical stratification statistics available.")
                return
            fig = plot_basin_characteristics_efficacy(
                df_primary,
                df_strata_compare=df_compare,
                label_primary="Global Best DA",
                label_compare="Per-Basin Best DA (Oracle)",
                title=title,
                figsize=figsize,
            )
        else:
            df_strata = self.get_physical_strata(config=config, lead_time=ref_lead)
            if df_strata.empty:
                print("[WARNING] No physical stratification statistics available.")
                return

            cfg_label = "Global Best DA" if config == "global_best" else ("Per-Basin Best DA (Oracle)" if config == "per_basin_best" else config)
            custom_title = title or f"Physical Catchment Characteristics Impact on DA Efficacy [{cfg_label}] (Lead {ref_lead} Day)"
            fig = plot_basin_characteristics_efficacy(df_strata, title=custom_title, figsize=figsize)
        plt.show()

    def launch_physical_characteristics_viewer(
        self,
        default_config: str = "both",
        default_table: str = "global_best",
        default_lead: Optional[int] = None,
        figsize: Tuple[int, int] = (18, 10),
    ):
        """Launches a single-cell interactive widget bar for Stage 6 Physical Characteristics."""
        try:
            import ipywidgets as widgets
        except ImportError:
            self.plot_basin_characteristics_efficacy_legacy(config=default_config, lead_time=default_lead, interactive=False)
            self.show_basin_characteristics_table(config=default_table, lead_time=default_lead)
            return

        init_lead = default_lead if (default_lead is not None and default_lead in self.lead_times) else (1 if 1 in self.lead_times else self.lead_times[0])

        plot_toggle = widgets.ToggleButtons(
            options=[
                ("Comparison Plot (Global Best vs Oracle)", "both"),
                ("Global Best DA Plot", "global_best"),
                ("Per-Basin Oracle Plot", "per_basin_best"),
                ("Hide Plot", "none"),
            ],
            value=default_config if default_config in ("both", "global_best", "per_basin_best", "none") else "both",
            description="Plot View:",
            button_style="info",
            style={"description_width": "initial", "button_width": "auto"},
        )

        table_toggle = widgets.ToggleButtons(
            options=[
                ("Global Best DA Table", "global_best"),
                ("Per-Basin Oracle Table", "per_basin_best"),
                ("Both Tables", "both"),
                ("Hide Table", "none"),
            ],
            value=default_table if default_table in ("global_best", "per_basin_best", "both", "none") else "global_best",
            description="Table View:",
            button_style="",
            style={"description_width": "initial", "button_width": "auto"},
        )

        lead_dropdown = widgets.Dropdown(
            options=[(f"Lead +{lt}d (t+{lt})", lt) for lt in self.lead_times],
            value=init_lead,
            description="Horizon:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="190px"),
        )

        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                p_cfg = plot_toggle.value
                t_cfg = table_toggle.value
                lt = int(lead_dropdown.value)

                if p_cfg != "none":
                    self.plot_basin_characteristics_efficacy_legacy(
                        config=p_cfg,
                        lead_time=lt,
                        figsize=figsize,
                        interactive=False,
                    )
                if t_cfg != "none":
                    self.show_basin_characteristics_table(
                        config=t_cfg,
                        lead_time=lt,
                    )

        plot_toggle.observe(_render, names="value")
        table_toggle.observe(_render, names="value")
        lead_dropdown.observe(_render, names="value")

        row1 = widgets.HBox([plot_toggle, lead_dropdown], layout=widgets.Layout(margin="0 0 6px 0", align_items="center"))
        row2 = widgets.HBox([table_toggle], layout=widgets.Layout(margin="0 0 8px 0", align_items="center"))
        display(widgets.VBox([row1, row2]), out)
        _render()

    def show_basin_characteristics_table(
        self,
        config: str = "global_best",
        lead_time: Optional[int] = None,
    ):
        """Displays formatted and styled 6-dimension physical characteristics matrix table.

        Args:
            config: 'both' (displays both Global Best DA and Oracle tables),
                    'global_best' (default), 'per_basin_best' (oracle), or specific config ID.
            lead_time: Forecast horizon in days (defaults to lead 1).
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot display basin characteristics table.")
            return
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        if config.lower() in ("both", "compare", "all"):
            df_gb = self.get_physical_strata(config="global_best", lead_time=ref_lead)
            df_pb = self.get_physical_strata(config="per_basin_best", lead_time=ref_lead)
            display(HTML("<h4>1. Single Deployable Benchmark: Global Best DA</h4>"))
            display(style_physical_characteristics_table(df_gb))
            display(HTML("<h4 style='margin-top: 15px;'>2. Theoretical Upper Bound: Per-Basin Best DA (Oracle)</h4>"))
            display(style_physical_characteristics_table(df_pb))
        else:
            df_strata = self.get_physical_strata(config=config, lead_time=ref_lead)
            if df_strata.empty:
                print("[WARNING] No physical stratification statistics available.")
                return
            styled = style_physical_characteristics_table(df_strata)
            display(styled)

    # -------------------------------------------------------------
    # Stage 5B: Candidate Gauge Identification & Basin Case Studies
    # -------------------------------------------------------------
    def get_category_candidates(
        self,
        lead_time: Optional[int] = None,
        config: str = "global_best",
        top_per_category: int = 2,
    ) -> pd.DataFrame:
        """Discovers contrasting exemplary catchments across all 5 physical conditions."""
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        cfg_id = get_global_best_config(self.df_eval, lead_time=ref_lead) if config == "global_best" else ("Per-Basin Best DA" if config == "per_basin_best" else config)
        return get_category_candidates(
            df_eval=self.df_eval,
            df_meta=self.df_meta,
            df_ts=self._candidate_ts(),
            lead_time=ref_lead,
            config_id=cfg_id,
            top_per_category=top_per_category,
        )

    def show_category_candidates(
        self,
        lead_time: Optional[int] = None,
        config: str = "global_best",
        top_per_category: int = 2,
    ):
        """Displays rich styled ranking table of candidate catchments for each category."""
        df_cand = self.get_category_candidates(lead_time=lead_time, config=config, top_per_category=top_per_category)
        if df_cand.empty:
            print("[WARNING] No candidate catchments found.")
            return
        styler = style_candidate_gauges_table(df_cand)
        display(styler)

    def find_candidate_basins(
        self,
        category: Optional[str] = None,
        sub_category: Optional[str] = None,
        lead_time: Optional[int] = None,
        config: str = "global_best",
        min_delta: Optional[float] = None,
        max_delta: Optional[float] = None,
        min_base_nse: Optional[float] = None,
        max_base_nse: Optional[float] = None,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
        min_epoch_spread: Optional[float] = None,
        sort_by: str = "skill_desc",
        top_n: int = 10,
        min_gap: Optional[float] = None,
        max_gap: Optional[float] = None,
    ) -> pd.DataFrame:
        """Interactively filters and ranks candidate gauges matching specific criteria.

        ``min_gap`` / ``max_gap`` filter on the Per-Basin minus Global Best DA skill-score gap;
        ``sort_by`` accepts 'skill_desc', 'skill_asc', 'gap_desc', 'gap_asc'.
        """
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        gb_cfg = get_global_best_config(self.df_eval, lead_time=ref_lead)
        cfg_id = gb_cfg if config == "global_best" else ("Per-Basin Best DA" if config == "per_basin_best" else config)
        return find_candidate_basins(
            df_eval=self.df_eval,
            df_meta=self.df_meta,
            df_ts=self._candidate_ts(),
            category=category,
            sub_category=sub_category,
            lead_time=ref_lead,
            config_id=cfg_id,
            min_delta=min_delta,
            max_delta=max_delta,
            min_base_nse=min_base_nse,
            max_base_nse=max_base_nse,
            min_area=min_area,
            max_area=max_area,
            min_epoch_spread=min_epoch_spread,
            sort_by=sort_by,
            top_n=top_n,
            min_gap=min_gap,
            max_gap=max_gap,
            gb_cfg=gb_cfg,
        )

    # -------------------------------------------------------------
    # Stage 6b: Configuration-Catchment Affinity (Static PCA + Clustering)
    # -------------------------------------------------------------
    def analyze_config_affinity(
        self,
        lead_times: Tuple[int, ...] = (1,),
        k: Optional[int] = None,
        method: str = "kmeans",
        min_gap: float = 0.02,
        var_target: float = 0.9,
        group_col: str = "spatial_block",
        n_splits: int = 5,
        n_perm: int = 1000,
        use_model_scaler: bool = False,
        interactive: bool = True,
        default_view: str = "policy",
        marker_size: float = 10.0,
        random_state: int = 0,
    ):
        """Stage 6b: does static-attribute structure explain which DA config wins per basin?

        Builds the basin x config skill-score regret matrix, a PCA of the model's 84
        static attributes, clusters that space, and evaluates attribute-driven
        config-assignment policies (Cluster / kNN / Predicted Best) against Global
        Best and the per-basin Oracle under spatially grouped CV. Results are cached
        on ``self.config_affinity`` (latest) and ``self._config_affinity_cache``.

        Args:
          lead_times: Leads whose skill score is averaged, e.g. ``(1,)`` or ``(1, 2, 3)``.
          k: Number of clusters; ``None`` picks by silhouette (kmeans) / BIC (gmm).
          method: ``"kmeans"`` or ``"gmm"``.
          min_gap: Winner-minus-runner-up SS gap defining a *confident* winner.
          var_target: Fraction of variance the retained PCs must explain.
          group_col: CV grouping: ``"spatial_block"``, ``"country"``, ``"wmo_reg"`` or ``"none"``.
          n_splits: CV folds.
          n_perm: Permutations for the Cramér's V association tests.
          use_model_scaler: Standardize with training ``scaler.nc`` instead of log1p + z-score.
          interactive: Show the widget viewer; otherwise render every view once.
          default_view: Initial view: policy / heatmap / biplot / map / axis / importance / clusters.
          marker_size: Map marker size.
          random_state: Seed.
        """
        from da_eval import config_affinity as ca  # pylint: disable=g-import-not-at-top

        if not hasattr(self, "_config_affinity_cache"):
            self._config_affinity_cache = {}

        def _get(leads, kk, meth):
            key = (tuple(leads), kk, meth, min_gap, var_target, group_col, n_splits, use_model_scaler)
            if key not in self._config_affinity_cache:
                print(f"Computing Stage 6b affinity for leads {list(leads)} (k={kk or 'auto'}, {meth})...")
                self._config_affinity_cache[key] = ca.run_config_affinity(
                    self.df_eval, lead_times=leads, k=kk, method=meth, min_gap=min_gap,
                    var_target=var_target, group_col=group_col, n_splits=n_splits, n_perm=n_perm,
                    use_model_scaler=use_model_scaler, attr_zarr_path=self.attr_zarr_path,
                    random_state=random_state,
                )
            self.config_affinity = self._config_affinity_cache[key]
            return self.config_affinity

        def _show(res, view):
            if view == "policy":
                print(ca.interpret_affinity(res))
                display(ca.style_policy_table(res["policy_df"]))
                ca.plot_policy_ladder(res); plt.show()
            elif view == "heatmap":
                ca.plot_cluster_regret_heatmap(res); plt.show()
            elif view == "clusters":
                display(res["cluster_summary"].style.format({"Mean Regret (cluster best)": "{:.4f}", "Mean Regret (global best)": "{:.4f}"}))
                display(res["association_df"].style.format({"cramers_v": "{:.3f}", "null_mean": "{:.3f}", "p_value": "{:.4f}", "n": "{:.0f}"}))
                display(res["pc_names"])
            elif view == "biplot":
                ca.plot_pca_biplot(res, color_by="confident_winner"); plt.show()
                ca.plot_pca_biplot(res, color_by="cluster"); plt.show()
            elif view == "map":
                ca.plot_cluster_map(res, color_by="cluster", marker_size=marker_size); plt.show()
                ca.plot_cluster_map(res, color_by="confident_winner", marker_size=marker_size); plt.show()
            elif view == "axis":
                fig = ca.plot_axis_affinity(res)
                if fig is not None:
                    plt.show()
            elif view == "importance":
                fig = ca.plot_attribute_importance(res)
                if fig is not None:
                    plt.show()

        views = [("Policy Ladder", "policy"), ("Cluster × Config Regret", "heatmap"), ("Cluster Profiles & Tests", "clusters"),
                 ("PCA Biplot", "biplot"), ("Map", "map"), ("Hyperparameter Axes", "axis"), ("Attribute Importance", "importance")]
        try:
            import ipywidgets as widgets  # pylint: disable=g-import-not-at-top
        except ImportError:
            widgets = None
        if not interactive or widgets is None:
            res = _get(lead_times, k, method)
            for _, v in views:
                _show(res, v)
            return None  # Results live in self.config_affinity (avoids dumping the dict in notebooks).

        view_tg = widgets.ToggleButtons(options=views, value=default_view if default_view in dict(views).values() else "policy",
                                        description="View:", button_style="info",
                                        style={"description_width": "initial", "button_width": "auto"})
        lead_opts = [("t+1", (1,)), ("mean t+1..3", (1, 2, 3)), ("mean t+1..7", tuple(self.lead_times))]
        if tuple(lead_times) not in [o[1] for o in lead_opts]:
            lead_opts.insert(0, (f"leads {list(lead_times)}", tuple(lead_times)))
        lead_dd = widgets.Dropdown(options=lead_opts, value=tuple(lead_times), description="Target:",
                                   style={"description_width": "initial"}, layout=widgets.Layout(width="200px"))
        k_dd = widgets.Dropdown(options=[("auto", None)] + [(str(i), i) for i in range(2, 13)], value=k,
                                description="k:", style={"description_width": "initial"}, layout=widgets.Layout(width="120px"))
        meth_dd = widgets.Dropdown(options=[("K-Means", "kmeans"), ("GMM", "gmm")], value=method, description="Method:",
                                   style={"description_width": "initial"}, layout=widgets.Layout(width="170px"))
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                _show(_get(lead_dd.value, k_dd.value, meth_dd.value), view_tg.value)

        for w in (view_tg, lead_dd, k_dd, meth_dd):
            w.observe(_render, names="value")
        display(widgets.VBox([view_tg, widgets.HBox([lead_dd, k_dd, meth_dd])]), out)
        _render()

    def plot_basin_epoch_case_study(
        self,
        basin_id: str,
        year: int = 2017,
        date_range: Optional[Tuple[str, str]] = None,
        lead_times: Tuple[int, int] = (1, 7),
        target_backbone: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 9.5),
        show_rho: bool = True,
        rhos: Optional[List] = None,
        extra_configs: Optional[List[str]] = None,
        models: Optional[List] = None,
        precip: Optional[List[str]] = None,
        paper: bool = False,
        legend: Optional[str] = None,
        show_metrics: Optional[bool] = None,
        precip_legend: bool = True,
        font_scale: Optional[float] = None,
        show_title: bool = True,
    ) -> plt.Figure:
        """Renders publication-ready 2-subplot Basin Case Study hydrograph (Lead 1 & Lead 7).

        ``show_rho`` adds the Global-Best PP / Per-Basin PP PP references (generated on the fly from
        Baseline + observed flow). ``rhos`` (e.g. ``[0.3, 0.9]``) adds PP series for arbitrary rho values;
        their NSE/SS/KGE are computed from the generated series when no metrics CSV exists.

        ``models`` selects exactly which lines are drawn (Observed is always shown), e.g.
        ``["Baseline", "Global-Best DA", "Per-Basin PP", "S49_05_...", 0.7]``. Roles:
        ``"Baseline"``, ``"Global-Best DA"``, ``"Per-Basin DA"``, ``"Global-Best PP"``, ``"Per-Basin PP"``;
        anything else is a Config ID and numbers are PP rho values. ``None`` = default set.

        ``precip`` overlays an inverted precipitation hyetograph per provider, e.g.
        ``["HRES", "GraphCast", "ERA5-Land", "CPC", "IMERG"]``. HRES/GraphCast are the forecast the model
        at lead ``L`` for that day (issued ``L-1`` days earlier); ERA5-Land/CPC/IMERG are observed/reanalysis.

        Paper styling (see ``static_plots.plot_basin_epoch_case_study``): ``paper=True`` (big fonts, short names,
        shared legend above, no metrics), ``legend="inside"|"top"|"bottom"``, ``show_metrics``,
        ``precip_legend``, ``font_scale``, ``show_title``.
        """
        extras = list(extra_configs or []) + [ar1_config_name(float(r)) for r in (rhos or [])]
        if models is not None:
            _, model_cfgs = split_case_study_models(models)
            models = list(models) + extras  # rhos / extra_configs still honoured alongside models
            extras = extras + [c for c in model_cfgs if c not in extras]
        if self.ts_store is not None:
            self.ensure_timeseries([basin_id], extra_configs=extras)  # ~0.05 s per new basin/config; no bulk load
        precip_df = None
        names = normalize_precip_names(precip)
        if names:
            store = self.get_precip_store()
            if store is not None:
                precip_df = store.get_basin(basin_id, lead_times=lead_times, products=names)
                if precip_df.empty:
                    print(f"[INFO] No precipitation forcings for {basin_id} in {store.root}")
        fig = plot_basin_epoch_case_study(
            basin_id=basin_id,
            df_eval=self.df_eval,
            df_ts=self.df_ts,
            df_meta=self.df_meta,
            data_dir=self.data_dir,
            year=year,
            date_range=date_range,
            lead_times=lead_times,
            target_backbone=target_backbone,
            figsize=figsize,
            show_rho=show_rho,
            extra_configs=extras,
            models=models,
            precip_df=precip_df,
            paper=paper,
            legend=legend,
            show_metrics=show_metrics,
            precip_legend=precip_legend,
            font_scale=font_scale,
            show_title=show_title,
        )
        display(fig)
        plt.close(fig)
        return fig

    def get_precip_store(self, forcing_dir: Optional[str] = None, subset_dir: Optional[str] = None,
                         download: bool = True):
        """Opens (and on first use caches from CNS) the per-basin precipitation forcings.

        Looks for ``<data_dir>/_ts_cache/forcings_*``; if missing and ``download``, copies only the precipitation
        arrays of the eval subset (default ``forcings.DEFAULT_SUBSET``, ~0.3 GB per 2 years, a few minutes).
        """
        if getattr(self, "_precip_store", None) is not None and forcing_dir is None:
            return self._precip_store
        root = Path(forcing_dir) if forcing_dir else PrecipForcingStore.discover(self.data_dir)
        if root is None and download:
            subset = subset_dir or self._infer_precip_subset()
            root = Path(self.data_dir) / "_ts_cache" / f"forcings_{subset.rstrip('/').split('/')[-1]}"
            print(f"[INFO] Caching precipitation forcings from {subset} -> {root} (one-off, a few minutes)...")
            try:
                cache_precip_subset(subset, root)
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING] Could not cache forcings: {e}")
                return None
        if root is None:
            print("[WARNING] No local precipitation forcings found (pass forcing_dir= or download=True).")
            return None
        self._precip_store = PrecipForcingStore(root)
        return self._precip_store

    def _infer_precip_subset(self) -> str:
        """Eval subset matching the staged run's date range (2017-2018 default; 2017-2023 for multi-year runs)."""
        import glob as _glob
        try:
            import xarray as xr
            stores = sorted(_glob.glob(os.path.join(str(self.data_dir), "*baseline*", "shard_*", "test", "model_epoch*",
                                                    "test_results*.zarr")))
            if stores:
                last = pd.Timestamp(xr.open_zarr(stores[0], consolidated=True)["date"].values[-1])
                if last > pd.Timestamp("2018-12-31"):
                    return DEFAULT_PRECIP_SUBSET.rstrip("/").rsplit("/", 1)[0] + "/sweep4287_2017_2023"
        except Exception:  # noqa: BLE001
            pass
        return DEFAULT_PRECIP_SUBSET

    def show_case_study_suite(
        self,
        n_per_regime: int = 1,
        year: int = 2017,
        category: str = "aridity",
        min_delta: float = 0.02,
        sort_by: str = "skill_desc",
        top_n: int = 5,
        interactive: bool = True,
        models: Optional[List] = None,
        rhos: Optional[List] = None,
        min_gap: Optional[float] = None,
        precip: Optional[List[str]] = None,
        paper: bool = False,
        legend: Optional[str] = None,
        show_metrics: Optional[bool] = None,
        precip_legend: bool = True,
        font_scale: Optional[float] = None,
    ):
        """Consolidated Stage 7 workflow: loads timeseries on demand, discovers candidate gauges,
        and renders an interactive selector for Hydrograph Case Studies (Lead 1 & Lead 7) and Regime Tables.

        ``models`` chooses the plotted lines (see ``plot_basin_epoch_case_study``), e.g.
        ``models=["Baseline", "Global-Best DA", "Per-Basin PP", 0.7]``. In interactive mode it seeds a
        multi-select "Models" widget (Ctrl/Shift-click) listing the roles, fixed-rho PP and every DA config.
        ``rhos`` adds extra PP rho values to that list.

        ``min_gap`` keeps basins where Per-Basin DA beats Global Best DA by at least this skill score
        (``PB−GB SS Gap``); ``sort_by='gap_desc'`` ranks by that gap. Both are also widgets in interactive mode.

        Figure style (also a widget row): ``paper`` (big fonts, short names), ``legend`` ("inside" / "top" /
        "bottom"; paper default "top"), ``show_metrics`` (NSE/SS/KGE in legend; paper default off),
        ``precip_legend`` (precipitation key), ``font_scale``.
        """
        style = dict(paper=paper, legend=legend, show_metrics=show_metrics, precip_legend=precip_legend,
                     font_scale=font_scale)
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot launch case study suite.")
            return

        if self.ts_store is not None:
            # Lazy mode: hydrographs are read per selected basin; candidate finders use a compact,
            # cached lead-1 frame (Baseline + Global Best DA) instead of the full timeseries.
            self.get_strata_timeseries()
        elif self.df_ts.empty:
            print("[INFO] Forecast timeseries not yet loaded. Loading on-demand timeseries for candidate gauges...")
            self.load_timeseries()

        df_all_cands = self.get_category_candidates(top_per_category=max(2, n_per_regime))
        df_filtered = self.find_candidate_basins(
            category=category,
            min_delta=min_delta,
            sort_by=sort_by,
            top_n=top_n,
            min_gap=min_gap,
        )

        if not interactive:
            self.show_category_candidates(top_per_category=max(2, n_per_regime))
            sel_basin = (
                df_filtered["Basin ID"].iloc[0]
                if not df_filtered.empty
                else (df_all_cands["Basin ID"].iloc[0] if not df_all_cands.empty else self.df_eval["Basin ID"].iloc[0])
            )
            self.plot_basin_epoch_case_study(basin_id=str(sel_basin), year=year, models=models, rhos=rhos, precip=precip,
                                             **style)
            return

        try:
            import ipywidgets as widgets
        except ImportError:
            self.show_case_study_suite(
                n_per_regime=n_per_regime, year=year, category=category,
                min_delta=min_delta, sort_by=sort_by, top_n=top_n, interactive=False,
                models=models, rhos=rhos, min_gap=min_gap, precip=precip, **style,
            )
            return

        # Build candidate dropdown options combining regime candidates + filtered candidates
        candidate_options = []
        seen_basins = set()
        for src_df in [df_all_cands, df_filtered]:
            if src_df is None or src_df.empty:
                continue
            for _, row in src_df.iterrows():
                bid = str(row.get("Basin ID", ""))
                if not bid or bid in seen_basins:
                    continue
                seen_basins.add(bid)
                bname = str(row.get("Basin Name", bid))
                cat_lbl = str(row.get("Physical Category", row.get("Baseline Tier", "")))
                ss_val = row.get("NSE Skill Score", row.get("NSE Delta", np.nan))
                ss_str = ((f"SS: {units.fmt_ss(ss_val)}" if "NSE Skill Score" in row else f"SS: {ss_val:+.3f}")
                          if pd.notna(ss_val) else "")
                label = f"{bid} — {bname} [{cat_lbl}] ({ss_str})".strip()
                candidate_options.append((label, bid))

        if not candidate_options:
            for bid in self.df_eval["Basin ID"].dropna().astype(str).unique()[:15]:
                candidate_options.append((bid, bid))

        view_toggle = widgets.ToggleButtons(
            options=[
                ("Case Study Hydrographs (Lead 1 & 7)", "hydrograph"),
                ("Hydrograph + Regime Candidate Table", "both"),
                ("5-Regime Candidate Leaderboard", "category_table"),
                ("Filtered Regime Query Table", "filter_table"),
            ],
            value="both",
            description="Stage 7 View:",
            button_style="info",
            style={"description_width": "initial", "button_width": "auto"},
        )

        basin_dropdown = widgets.Dropdown(
            options=candidate_options,
            value=candidate_options[0][1],
            description="Case Study Gauge:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="480px"),
        )

        year_dropdown = widgets.Dropdown(
            options=[2017, 2018, 2019, 2020],
            value=year if year in (2017, 2018, 2019, 2020) else 2017,
            description="Year:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        regime_dropdown = widgets.Dropdown(
            options=["all", "aridity", "snow_fraction", "seasonality", "regulation", "area"],
            value="all",
            description="Regime:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="180px"),
        )

        base_nse_slider = widgets.FloatRangeSlider(
            value=(0.40, 0.85),
            min=-0.50,
            max=1.00,
            step=0.05,
            description="Base NSE:",
            readout_format=".2f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px"),
        )

        min_skill_slider = widgets.FloatSlider(
            value=0.30,
            min=-1.00,
            max=0.95,
            step=0.05,
            description="Min Global Best SS:",
            readout_format=".2f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px"),
        )

        # Models multi-select: roles first, then fixed-rho PP configs, then individual DA configs.
        all_cfgs = [str(c) for c in self.df_eval["Config ID"].dropna().unique()]
        skip = {"Baseline", "Per-Basin Best DA"}

        def _rho_key(c):
            r = parse_ar1_config(c)
            return r if isinstance(r, float) else 0.0

        rho_cfgs = sorted({c for c in all_cfgs if is_rho_config(c) and "per_basin" not in c.lower()}
                          | {ar1_config_name(float(r)) for r in (rhos or [])}, key=_rho_key)
        da_cfgs = sorted(c for c in all_cfgs if c not in skip and not is_rho_config(c))
        model_options = ([(f"[Role] {r}", r) for r in CASE_STUDY_ROLES]
                         + [(c.replace("AR1_postprocess_", "PP "), c) for c in rho_cfgs]
                         + [(c, c) for c in da_cfgs])
        opt_values = {v for _, v in model_options}
        if models is None:
            init_models = list(CASE_STUDY_ROLES)
        else:
            roles0, cfgs0 = split_case_study_models(models)
            init_models = [r for r in CASE_STUDY_ROLES if r in roles0] + cfgs0
            for c in cfgs0:
                if c not in opt_values:
                    model_options.append((c, c))
                    opt_values.add(c)
        models_select = widgets.SelectMultiple(
            options=model_options,
            value=tuple(v for v in init_models if v in opt_values),
            rows=8,
            description="Models:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="480px"),
        )

        gap_slider = widgets.FloatSlider(
            value=float(min_gap) if min_gap is not None else 0.0,
            min=0.0,
            max=1.0,
            step=0.02,
            description="Min PB−GB SS Gap:",
            readout_format=".2f",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="300px"),
        )
        gap_lead_dd = widgets.Dropdown(
            options=[(f"t+{lt}", lt) for lt in self.lead_times],
            value=1 if 1 in self.lead_times else self.lead_times[0],
            description="Gap @",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="120px"),
        )
        sort_dd = widgets.Dropdown(
            options=[("Global Best SS ↓", "skill_desc"), ("PB−GB Gap ↓", "gap_desc"),
                     ("Per-Basin SS ↓", "pb_desc"), ("Global Best SS ↑", "skill_asc")],
            value=sort_by if sort_by in ("skill_desc", "gap_desc", "skill_asc") else "skill_desc",
            description="Sort:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="210px"),
        )
        gap_cache: Dict[int, pd.DataFrame] = {}

        def _gap_frame(lt: int) -> pd.DataFrame:
            if lt not in gap_cache:
                gap_cache[lt] = compute_gb_pb_gap(self.df_eval, lead_time=lt).set_index("Basin ID")
            return gap_cache[lt]

        search_box = widgets.Text(
            placeholder="ID or name, e.g. camelscl, 09444500 (Enter)",
            description="Search:",
            continuous_update=False,  # filter on Enter / blur so each keystroke doesn't re-render the plot
            style={"description_width": "initial"},
            layout=widgets.Layout(width="330px"),
        )

        init_precip = set(normalize_precip_names(precip))
        precip_toggles = [
            widgets.ToggleButton(
                value=name in init_precip,
                description=name,
                tooltip=("Lead-L forecast for this day (issued L-1 days earlier)"
                         if PRECIP_PRODUCTS[name][2] else "Observed / reanalysis precipitation on the valid day"),
                layout=widgets.Layout(width="110px"),
            )
            for name in PRECIP_PRODUCTS
        ]

        _leg0 = (legend or ("top" if paper else "inside")).lower()
        _leg0 = {"above": "top", "below": "bottom"}.get(_leg0, _leg0)
        w_paper = widgets.Checkbox(value=bool(paper), description="Paper style (big text, short names)",
                                   indent=False, layout=widgets.Layout(width="260px"))
        w_legend = widgets.ToggleButtons(options=[("Above", "top"), ("Below", "bottom"), ("Inside", "inside")],
                                         value=_leg0 if _leg0 in ("top", "bottom", "inside") else "inside",
                                         description="Model legend:", style={"description_width": "initial",
                                                                             "button_width": "70px"})
        w_metrics = widgets.Checkbox(value=(not paper) if show_metrics is None else bool(show_metrics),
                                     description="NSE / SS / KGE in legend", indent=False,
                                     layout=widgets.Layout(width="200px"))
        w_pkey = widgets.Checkbox(value=bool(precip_legend), description="Precipitation key", indent=False,
                                  layout=widgets.Layout(width="160px"))
        w_font = widgets.FloatSlider(value=float(font_scale) if font_scale is not None else (2.0 if paper else 1.0),
                                     min=1.0, max=3.5, step=0.1, description="Font ×:", readout_format=".1f",
                                     continuous_update=False, style={"description_width": "initial"},
                                     layout=widgets.Layout(width="220px"))

        out = widgets.VBox()

        name_map: Dict[str, str] = {}
        if not self.df_meta.empty and "Basin ID" in self.df_meta.columns and "gauge_name" in self.df_meta.columns:
            name_map = self.df_meta.drop_duplicates("Basin ID").set_index("Basin ID")["gauge_name"].astype(str).to_dict()

        def _update_basin_options(*_):
            ref_lead = 1 if 1 in self.lead_times else self.lead_times[0]
            gb_cfg = get_global_best_config(self.df_eval, lead_time=ref_lead)
            df1 = self.df_eval[self.df_eval["Lead Time (Days)"] == ref_lead].copy()
            gb_rows = df1[df1["Config ID"] == gb_cfg][["Basin ID", "Base NSE", "NSE Skill Score", "DA NSE"]].copy()
            g = _gap_frame(gap_lead_dd.value)
            gb_rows["Gap"] = gb_rows["Basin ID"].map(g[GAP_COL])
            gb_rows["PB SS"] = gb_rows["Basin ID"].map(g["Per-Basin SS"])
            gb_rows["PB Cfg"] = gb_rows["Basin ID"].map(g["Per-Basin Config"]) if "Per-Basin Config" in g else ""

            # Filter by slider ranges
            low_b, high_b = base_nse_slider.value
            m_ss = min_skill_slider.value
            mask = (gb_rows["Base NSE"] >= low_b) & (gb_rows["Base NSE"] <= high_b) & (gb_rows["NSE Skill Score"] >= m_ss)
            if gap_slider.value > 0:
                mask &= gb_rows["Gap"] >= gap_slider.value

            # Text search on Basin ID or gauge name; comma-separated terms are OR-ed.
            terms = [t.strip().lower() for t in search_box.value.split(",") if t.strip()]
            note = ""
            if terms:
                ids = gb_rows["Basin ID"].astype(str)
                hay = (ids + " " + ids.map(name_map).fillna("")).str.lower()
                hit = hay.apply(lambda s: any(t in s for t in terms))
                if (mask & hit).any():
                    mask &= hit
                else:
                    mask = hit  # nothing passes the sliders: search all basins instead
                    note = " (search ignores sliders: no hits within filters)" if hit.any() else ""
            sort_col, asc = {"skill_desc": ("NSE Skill Score", False), "skill_asc": ("NSE Skill Score", True),
                             "gap_desc": ("Gap", False), "pb_desc": ("PB SS", False)}[sort_dd.value]
            matched = gb_rows[mask].sort_values(sort_col, ascending=asc)

            glt = gap_lead_dd.value
            new_options = []
            for _, r in matched.head(400).iterrows():
                bid = str(r["Basin ID"])
                b_name = name_map.get(bid, bid)
                dp = None if units.ss_percent() else 2
                gap_str = f" | PB−GB@t+{glt}: {units.fmt_ss(r['Gap'], decimals=dp)}" if pd.notna(r["Gap"]) else ""
                lbl = (f"{bid} — {b_name} [Base NSE: {r['Base NSE']:.2f} → "
                       f"GB SS: {units.fmt_ss(r['NSE Skill Score'], decimals=dp)}{gap_str}]")
                new_options.append((lbl, bid))

            if not new_options:
                new_options = [("No catchments meet criteria (widen filters)", "")]
            basin_dropdown.options = new_options
            if new_options and new_options[0][1]:
                basin_dropdown.value = new_options[0][1]
            n_lbl.value = f"{len(matched):,} gauges match{note}"

        n_lbl = widgets.Label("")

        def _render(*_):
            plt.close("all")
            fresh = widgets.Output()
            out.children = (fresh,)
            with fresh:
                v_mode = view_toggle.value
                b_id = basin_dropdown.value
                yr_val = int(year_dropdown.value)
                reg_val = regime_dropdown.value

                if v_mode in ("category_table", "both"):
                    self.show_category_candidates(top_per_category=max(2, n_per_regime))
                elif v_mode == "filter_table":
                    sub_cands = self.find_candidate_basins(
                        category=reg_val if reg_val != "all" else None,
                        lead_time=gap_lead_dd.value,
                        min_delta=min_delta,
                        min_base_nse=base_nse_slider.value[0],
                        max_base_nse=base_nse_slider.value[1],
                        sort_by="gap_desc" if sort_dd.value == "gap_desc" else sort_by,
                        top_n=top_n,
                        min_gap=gap_slider.value if gap_slider.value > 0 else None,
                    )
                    cols = [c for c in ["Basin ID", "Basin Name", "Baseline Tier", "Base NSE", "DA NSE", "NSE Skill Score",
                                        "Per-Basin SS", GAP_COL, "Per-Basin Config"] if c in sub_cands.columns]
                    display(sub_cands[cols] if cols else sub_cands)

                if v_mode in ("hydrograph", "both") and b_id:
                    g = _gap_frame(gap_lead_dd.value)
                    if b_id in g.index and "Per-Basin Config" in g:
                        print(f"Per-Basin DA config for {b_id}: {g.at[b_id, 'Per-Basin Config']} "
                              f"(PB−GB SS gap @t+{gap_lead_dd.value}: {units.fmt_ss(g.at[b_id, GAP_COL])})")
                    self.plot_basin_epoch_case_study(basin_id=b_id, year=yr_val, models=list(models_select.value),
                                                     precip=[t.description for t in precip_toggles if t.value],
                                                     paper=w_paper.value, legend=w_legend.value,
                                                     show_metrics=w_metrics.value, precip_legend=w_pkey.value,
                                                     font_scale=w_font.value)

        view_toggle.observe(_render, names="value")
        basin_dropdown.observe(_render, names="value")
        year_dropdown.observe(_render, names="value")
        regime_dropdown.observe(_render, names="value")
        models_select.observe(_render, names="value")
        for t in precip_toggles:
            t.observe(_render, names="value")
        for w in (w_paper, w_legend, w_metrics, w_pkey, w_font):
            w.observe(_render, names="value")
        for w in (base_nse_slider, min_skill_slider, gap_slider, gap_lead_dd, sort_dd, search_box):
            w.observe(_update_basin_options, names="value")

        _update_basin_options()

        row1 = widgets.HBox([view_toggle], layout=widgets.Layout(margin="0 0 6px 0", align_items="center"))
        row2 = widgets.HBox([base_nse_slider, min_skill_slider], layout=widgets.Layout(margin="0 0 6px 0", align_items="center"))
        row2b = widgets.HBox([gap_slider, gap_lead_dd, sort_dd, n_lbl], layout=widgets.Layout(margin="0 0 6px 0", align_items="center"))
        row3 = widgets.HBox([search_box, basin_dropdown, year_dropdown, regime_dropdown], layout=widgets.Layout(margin="0 0 8px 0", align_items="center"))
        row4 = widgets.HBox([models_select, widgets.VBox([widgets.Label("Precipitation overlay:")] + precip_toggles)],
                            layout=widgets.Layout(margin="0 0 8px 0", align_items="flex-start"))
        row5 = widgets.HBox([w_paper, w_legend, w_metrics, w_pkey, w_font],
                            layout=widgets.Layout(margin="0 0 8px 0", align_items="center"))
        display(widgets.VBox([row1, row2, row2b, row3, row4, row5]), out)
        _render()

    # -------------------------------------------------------------
    # Stage 6: Optimal Basin Distribution & Win Share (Globally Best vs. Per-Basin DA)
    # -------------------------------------------------------------
    def plot_optimal_basin_distribution(
        self,
        top_n: int = 10,
        lead_time: Optional[int] = None,
        plot_type: str = "combined",
        figsize: Optional[tuple] = None,
    ):
        """Plots distribution of Per-Basin DA winners (session selection) across catchments; ``lead_time`` sets the
        displayed global medians only:

        - plot_type="combined": 2-panel figure with Horizontal Bar Chart & Donut Chart
        - plot_type="bar": Standalone Horizontal Bar Chart
        - plot_type="donut" or "pie": Standalone Donut Chart
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot optimal basin distribution.")
            return
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        fig = plot_top_n_optimal_basin_distribution(
            self.df_eval,
            top_n=top_n,
            lead_time=ref_lead,
            plot_type=plot_type,
            figsize=figsize,
        )
        plt.show()

    def show_optimal_basin_distribution_table(
        self,
        top_n: int = 10,
        lead_time: Optional[int] = None,
    ):
        """Displays formatted and styled table showing top N configurations by basins won (session selection),

        highlighting the Globally Best DA configuration and market share.
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot display optimal basin distribution table.")
            return
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        table = build_top_n_optimal_basin_distribution_table(
            self.df_eval,
            top_n=top_n,
            lead_time=ref_lead,
        )
        if table.empty:
            print("[WARNING] Optimal basin distribution table is empty.")
            return
        styled = style_top_n_optimal_basin_distribution_table(table)
        display(styled)

    # -------------------------------------------------------------
    # Stage 7: Interactive Dashboard
    # -------------------------------------------------------------
    def launch_dashboard(
        self,
        year: int = 2017,
        show_star: bool = True,
        interactive_map: bool = False,
        vmin: float = -0.2,
        vmax: float = 0.2,
        filter_complete_streamflow: bool = False,
        color_by: str = "skill_nse",
        color_mode: str = "binned",
        bin_thresholds="0.01, 0.05",
        marker_size: float = 10.0,
        region: str = "global",
        koppen: str = "off",
        koppen_filter="all",
        koppen_alpha: float = 0.4,
        font_scale: float = 1.5,
        view_mode: str = "inspector_global",
    ):
        """Launches the linked interactive dashboard (Countries Map <--> Hydrograph with KGE/NSE + Info Card).

        The hydrograph shows exactly four deterministic series at the selected lead time: observed
        streamflow, the open-loop baseline, Global Best DA, and Per-Basin Best DA.

        Args:
            year: Evaluation year to display in the hydrograph (default 2017).
            show_star: Whether to highlight active catchment with a star marker (ipywidgets).
            interactive_map: If True, launches Bokeh TapTool map; otherwise launches ipywidgets.
            vmin: Lower bound for continuous map colorbar scale (default -0.2).
            vmax: Upper bound for continuous map colorbar scale (default 0.2).
            filter_complete_streamflow: If True, filters displayed catchments to those with zero missing q_obs in the plotting period.
            color_by: Initial map metric ('skill_nse', 'delta', 'da', or 'baseline').
            color_mode: 'binned' (default 5-category discrete color map) or 'continuous'.
            bin_thresholds: Positive thresholds pair (default '0.01, 0.05' -> bins <-0.05, -0.05..-0.01, +-0.01, +0.01..+0.05, >+0.05) or 4 explicit edges.
            marker_size: Circle marker size (`s`, default 10.0).
            region: Geographic zoom preset ('global', 'north_america', 'south_america', 'europe', 'australia').
            koppen: Köppen-Geiger underlay (Beck et al. 2023, 1991–2020): 'off', 'groups' (A–E) or 'classes' (30).
            koppen_filter: 'all', a group letter (e.g. 'B' = Arid) or classes ('BSk, Cfb') to restrict catchments.
            koppen_alpha: Underlay opacity (default 0.4).
            font_scale: Text size multiplier for the publication / paper maps (default 1.5).
            view_mode: Initial mode ('inspector_global', 'map_pub_global', 'map_grid' = paper 2x2 of
                Global vs Per-Basin DA at two leads, ...).
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot launch dashboard.")
            return
        lazy = self.ts_store is not None
        if not lazy and self.df_ts.empty:
            print("[INFO] Forecast timeseries not yet loaded. Loading on demand for top models...")
            self.load_timeseries()
        dash = InteractiveDADashboard(
            df_eval=self.df_eval,
            df_ts=self.df_ts,
            df_meta=self.df_meta,
            ts_loader=(lambda b: self.ensure_timeseries([b])) if lazy else None,
            ts_configs=self._ts_configs() if lazy else None,
            ts_basins=self.ts_store.basins if lazy else None,
            obs_df=self.get_obs_timeseries() if (lazy and filter_complete_streamflow) else None,
            obs_loader=self.get_obs_timeseries if lazy else None,
            default_year=year,
            default_vmin=vmin,
            default_vmax=vmax,
            default_filter_complete_streamflow=filter_complete_streamflow,
            default_color_by=color_by,
            default_color_mode=color_mode,
            default_bin_thresholds=bin_thresholds,
            default_marker_size=marker_size,
            default_region=region,
            default_koppen=koppen,
            default_koppen_filter=koppen_filter,
            default_koppen_alpha=koppen_alpha,
            koppen_df=self.get_basin_koppen(),
        )
        if interactive_map:
            dash.launch(
                interactive_map=True,
                year=year,
                vmin=vmin,
                vmax=vmax,
                filter_complete_streamflow=filter_complete_streamflow,
                color_by=color_by,
                color_mode=color_mode,
                bin_thresholds=bin_thresholds,
                marker_size=marker_size,
                region=region,
            )
        else:
            dash.launch_ipywidgets(
                year=year,
                show_star=show_star,
                vmin=vmin,
                vmax=vmax,
                filter_complete_streamflow=filter_complete_streamflow,
                color_by=color_by,
                color_mode=color_mode,
                bin_thresholds=bin_thresholds,
                marker_size=marker_size,
                region=region,
                font_scale=font_scale,
                view_mode=view_mode,
            )

    def get_basin_koppen(self) -> pd.DataFrame:
        """Per-basin Köppen-Geiger class (Beck et al. 2023, 1991–2020, 1-km grid) — majority class over each
        Caravan catchment polygon. Cached on disk (``data/koppen_geiger``) and on the session."""
        if getattr(self, "_df_koppen", None) is None:
            from da_eval import koppen as kg
            ll = None
            src = self.df_meta if (self.df_meta is not None and {"lat", "lon"} <= set(self.df_meta.columns)) else self.df_eval
            if {"lat", "lon"} <= set(src.columns):
                ll = src[["Basin ID", "lat", "lon"]]
            self._df_koppen = kg.compute_basin_koppen(self.df_eval["Basin ID"].dropna().astype(str).unique(), lat_lon=ll)
        return self._df_koppen

    KG_TABLE_STATS = ("median", "% > 0", "% harm", "N")

    @staticmethod
    def style_koppen_table(tbl: pd.DataFrame, metric: str = "NSE Skill Score", color: bool = True):
        """Styled Köppen table (display only): counts / shares as integers, stats with 3 decimals; for the NSE
        skill score the median / mean / q10 columns follow ``set_ss_percent``. ``color`` adds cell gradients."""
        ss = metric == "NSE Skill Score" and units.ss_percent()
        fmt = {c: ("{:.0f}" if c[0] in ("N", "% > 0", "% harm")
                   else (units.ss_formatter() if ss else "{:.3f}")) for c in tbl.columns}
        sty_ = tbl.style.format(fmt, na_rep="–")
        if color:
            for s_ in dict.fromkeys(c[0] for c in tbl.columns):
                cs = [c for c in tbl.columns if c[0] == s_]
                if s_ in ("median", "mean", "q10"):
                    v = np.nanmax(np.abs(tbl[cs].to_numpy(float))) or 1.0
                    sty_ = sty_.background_gradient(cmap="RdBu", vmin=-v, vmax=v, subset=cs)
                elif s_ == "% > 0":
                    sty_ = sty_.background_gradient(cmap="Blues", vmin=0, vmax=100, subset=cs)
                elif s_ == "% harm":
                    sty_ = sty_.background_gradient(cmap="Reds", vmin=0, vmax=50, subset=cs)
        if ss:
            sty_ = sty_.format_index(lambda c: units.ss_label(c) if c in ("median", "mean", "q10") else c,
                                     axis=1, level=0)
        return sty_

    def koppen_skill_table(self, lead_times=(1, 3, 7), metric: str = "NSE Skill Score",
                           by: str = "group", configs: Optional[List[str]] = None,
                           harm_threshold: float = -0.01, models=("global_da", "per_basin_da", "per_basin_rho"),
                           stats=KG_TABLE_STATS, min_n: int = 0,
                           common_basins: bool = False, interactive: bool = False, styled: bool = False):
        """Skill per Köppen-Geiger group (or class) for the headline models, per lead time.

        Columns per model/lead: any of ``stats`` in {median, mean, % > 0, % harm, q10, N}
        (``% harm`` = share of basins with metric < ``harm_threshold``).
        ``models``: reference keys among global_da, per_basin_da, global_rho, per_basin_rho; ``configs`` adds
        explicit Config IDs. Reference models follow the session selection (``session.set_selection``).
        ``min_n`` hides climates with fewer basins; ``common_basins`` restricts every model to the same basins.
        ``interactive=True`` shows toggles for all of the above (with a colour-scaled table).
        ``styled=True`` (static mode) returns the colour-scaled Styler (skill score in the ``set_ss_percent`` unit)
        instead of the raw numeric DataFrame.
        """
        if interactive:
            return self._koppen_skill_table_widget(lead_times, metric, by, configs, harm_threshold, models,
                                                   stats, min_n, common_basins)
        if styled:
            tbl = self.koppen_skill_table(lead_times=lead_times, metric=metric, by=by, configs=configs,
                                          harm_threshold=harm_threshold, models=models, stats=stats, min_n=min_n,
                                          common_basins=common_basins)
            return tbl if tbl.empty else self.style_koppen_table(tbl, metric=metric)
        from da_eval import koppen as kg
        from da_eval.risk_return import resolve_reference_configs
        kg_df = self.get_basin_koppen()
        key = "kg_group" if by == "group" else "kg_symbol"
        refs = resolve_reference_configs(self.df_eval)
        names = {"global_da": "Global-Best DA", "per_basin_da": "Per-Basin DA",
                 "global_rho": "Global-Best PP", "per_basin_rho": "Per-Basin PP"}
        labels = {}
        for m in models or ():
            if m in refs:
                labels[refs[m]] = names[m]
        present = set(self.df_eval["Config ID"])
        for c in configs or ():
            if c in present:
                labels.setdefault(c, str(c))
        if not labels:
            return pd.DataFrame()
        d = self.df_eval[self.df_eval["Config ID"].isin(labels) & self.df_eval["Lead Time (Days)"].isin(lead_times)]
        d = d[["Basin ID", "Config ID", "Lead Time (Days)", metric]].dropna(subset=[metric])
        if common_basins:
            nb = d[["Basin ID", "Config ID", "Lead Time (Days)"]].drop_duplicates().groupby("Basin ID").size()
            d = d[d["Basin ID"].isin(nb[nb == len(labels) * len(set(lead_times))].index)]
        d = d.merge(kg_df[["Basin ID", "kg_group", "kg_symbol"]], on="Basin ID", how="left")
        d[key] = d[key].fillna("?")
        order = list(labels.values())
        rows = []
        for (k, cfg, lt), g in d.groupby([key, "Config ID", "Lead Time (Days)"]):
            v = g[metric].to_numpy(float)
            rows.append({key: k, "Model": labels[cfg], "Lead": int(lt), "N": len(v),
                         "median": np.median(v), "mean": v.mean(), "q10": np.quantile(v, 0.1),
                         "% > 0": (v > 0).mean() * 100, "% harm": (v < harm_threshold).mean() * 100})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        if min_n:
            n_by = out.groupby(key)["N"].max()
            out = out[out[key].isin(n_by[n_by >= min_n].index)]
        if by == "group":
            out.insert(1, "Climate", out[key].map(lambda g: kg.KG_GROUPS.get(g, ("?",))[0]))
        idx = [c for c in [key, "Climate"] if c in out.columns]
        stats = [s_ for s_ in stats if s_ in out.columns] or ["median"]
        tbl = out.pivot_table(index=idx, columns=["Model", "Lead"], values=stats)
        cols = [(s_, m, l) for s_ in stats for m in order for l in sorted(set(lead_times)) if (s_, m, l) in tbl.columns]
        return tbl[cols].round(3)

    def _koppen_skill_table_widget(self, lead_times, metric, by, configs, harm_threshold, models,
                                   stats, min_n, common_basins):
        import ipywidgets as widgets
        lay = lambda w: widgets.Layout(width=w)  # noqa: E731
        sty = {"description_width": "initial"}
        init = {int(l) for l in lead_times}
        w_leads = [widgets.ToggleButton(value=lt in init, description=f"t+{lt}", layout=lay("52px"))
                   for lt in self.lead_times]
        w_by = widgets.ToggleButtons(options=[("Groups (A–E)", "group"), ("Classes", "class")],
                                     value="class" if by != "group" else "group", style={"button_width": "105px"})
        m_opts = [(lbl, col) for lbl, col in (("NSE skill score", "NSE Skill Score"), ("ΔNSE", "NSE Delta"),
                                              ("NSE (DA)", "DA NSE"), ("KGE skill score", "KGE Skill Score"),
                                              ("ΔKGE", "KGE Delta")) if col in self.df_eval.columns]
        w_metric = widgets.Dropdown(options=m_opts, value=metric if metric in [o[1] for o in m_opts] else m_opts[0][1],
                                    description="Metric:", layout=lay("210px"), style=sty)
        names = {"global_da": "Global-Best DA", "per_basin_da": "Per-Basin DA",
                 "global_rho": "Global-Best PP", "per_basin_rho": "Per-Basin PP"}
        w_models = {k: widgets.ToggleButton(value=k in (models or ()), description=v, layout=lay("145px"))
                    for k, v in names.items()}
        sweep = sorted(c for c in self.df_eval["Config ID"].dropna().unique()
                       if not is_rho_config(c) and "per-basin" not in c.lower() and "baseline" not in c.lower())
        w_extra = widgets.SelectMultiple(options=sweep, value=tuple(c for c in (configs or ()) if c in sweep),
                                         description="+ configs:", rows=4, layout=lay("520px"), style=sty)
        all_stats = ["median", "mean", "q10", "% > 0", "% harm", "N"]
        w_stats = {s_: widgets.ToggleButton(value=s_ in stats, description=s_, layout=lay("75px")) for s_ in all_stats}
        w_harm = widgets.Dropdown(options=[("harm < −0.01", -0.01), ("harm < −0.05", -0.05), ("harm < 0", 0.0)],
                                  value=harm_threshold if harm_threshold in (-0.01, -0.05, 0.0) else -0.01, layout=lay("130px"))
        w_min = widgets.BoundedIntText(value=int(min_n), min=0, max=10000, description="Min basins:",
                                       layout=lay("150px"), style=sty)
        w_common = widgets.Checkbox(value=common_basins, description="Same basins for all models", indent=False,
                                    layout=lay("210px"))
        w_color = widgets.Checkbox(value=True, description="Colour cells", indent=False, layout=lay("120px"))
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel = [lt for lt, b in zip(self.lead_times, w_leads) if b.value] or [self.lead_times[0]]
                st = [s_ for s_, b in w_stats.items() if b.value] or ["median"]
                tbl = self.koppen_skill_table(lead_times=sel, metric=w_metric.value, by=w_by.value,
                                              configs=list(w_extra.value), harm_threshold=w_harm.value,
                                              models=[k for k, b in w_models.items() if b.value],
                                              stats=st, min_n=w_min.value,
                                              common_basins=w_common.value)
                if tbl is None or tbl.empty:
                    print("No data for this selection.")
                    return
                self.koppen_table = tbl  # latest table, e.g. for .to_latex()
                sty_ = self.style_koppen_table(tbl, metric=w_metric.value, color=w_color.value)
                m_lbl = {c: l for l, c in m_opts}[w_metric.value]
                display(HTML(f"<b>{m_lbl}"
                             f" by Köppen-Geiger {'group' if w_by.value == 'group' else 'class'}</b> "
                             f"(Beck et al. 2023; reference models {self.selection.label()}). "
                             f"Latest table: <code>session.koppen_table</code>"))
                display(sty_)

        for w in (*w_leads, w_by, w_metric, *w_models.values(), w_extra, *w_stats.values(), w_harm, w_min,
                  w_common, w_color):
            w.observe(_render, names="value")
        box = lambda ws, m="0 0 4px 0": widgets.HBox(ws, layout=widgets.Layout(margin=m, flex_flow="row wrap"))  # noqa: E731
        controls = widgets.VBox([
            box([widgets.HTML("<b style='margin-right:6px'>Leads:</b>"), *w_leads, w_by]),
            box([w_metric, widgets.HTML("<b style='margin:0 6px'>Show:</b>"), *w_stats.values(), w_harm]),
            box([widgets.HTML("<b style='margin-right:6px'>Models:</b>"), *w_models.values()]),
            box([w_extra]),
            box([w_min, w_common, w_color], "0 0 8px 0"),
        ])
        display(controls, out)
        _render()
        return None

    def plot_catchment_map(
        self,
        color_by: str = "skill_nse",
        show_star: bool = False,
        star_basin: Optional[str] = None,
        lead_time: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        vmin: float = -0.2,
        vmax: float = 0.2,
        config_mode: str = "global_best",
        filter_complete_streamflow: bool = False,
        year: Optional[int] = None,
        color_mode: str = "binned",
        bin_thresholds="0.01, 0.05",
        marker_size: float = 12.0,
        region: str = "global",
        koppen: str = "off",
        koppen_filter="all",
        koppen_alpha: float = 0.4,
        font_scale: float = 1.5,
        grid_leads=(1, 7),
        edge_width: Optional[float] = None,
        n_bins: int = 7,
    ) -> plt.Figure:
        """Renders a publication-ready global catchment map without stars (or with optional star).

        Colour scale options: ``color_mode="quantile"`` (``n_bins`` equal-count bins), ``"binned"`` with any
        list of edges in ``bin_thresholds``, or ``"continuous"`` with ``vmin="auto", vmax="auto"``.
        ``edge_width=0`` removes marker outlines.

        ``config_mode='grid'`` gives the paper 2x2 (rows = ``grid_leads``, columns = Global-Best DA | Per-Basin DA)
        with one colourbar per panel and a shared Köppen-Geiger legend at the bottom. ``font_scale`` sizes all text.

        Köppen-Geiger: ``koppen`` in {'off','groups','classes'} adds a Beck et al. (2023) underlay;
        ``koppen_filter`` ('all', 'B', 'BSk, Cfb', ...) restricts plotted catchments.

        Args:
            color_by: 'delta' (colored by Lead 1 ΔNSE), 'skill_nse' (colored by NSE Skill Score),
                      'baseline' (colored by Base NSE), 'da' (colored by DA NSE),
                      or 'uniform' (clean uniform blue dots for publication station maps).
            show_star: If True, stars a specific basin. Default is False (clean, no star).
            star_basin: Basin ID to star if show_star=True.
            lead_time: Forecast horizon (defaults to lead 1).
            figsize: Figure dimensions (default (15, 7.5)).
            title: Custom plot title.
            vmin: Lower bound for continuous NSE / ΔNSE / Skill Score colorbar scale (default -0.2).
            vmax: Upper bound for continuous NSE / ΔNSE / Skill Score colorbar scale (default 0.2).
            config_mode: 'global_best' (default), 'per_basin_best' (Oracle), or 'both' (2-panel side-by-side comparison).
            filter_complete_streamflow: If True, filters basins to those with zero missing q_obs in `year`.
            year: Evaluation year for complete streamflow filtering.
            color_mode: 'binned' (default 5-category discrete color map) or 'continuous'.
            bin_thresholds: Positive thresholds pair (default '0.01, 0.05') or 4 explicit edges.
            marker_size: Circle marker size (`s`, default 12.0).
            region: Geographic zoom preset ('global', 'north_america', 'south_america', 'europe', 'australia').
        """
        if self.df_eval.empty:
            print("[WARNING] Evaluation records are empty. Cannot plot catchment map.")
            return None
        ts_for_filter = self.df_ts
        if filter_complete_streamflow:
            if self.ts_store is not None:
                ts_for_filter = self.get_obs_timeseries()  # observed flow only, cached
            elif self.df_ts.empty:
                print("[INFO] Loading forecast timeseries on demand to filter complete-streamflow basins...")
                self.load_timeseries()
                ts_for_filter = self.df_ts
        ref_lead = lead_time if lead_time is not None else (1 if 1 in self.lead_times else self.lead_times[0])
        fig = plot_catchment_map(
            df_eval=self.df_eval,
            df_meta=self.df_meta,
            color_by=color_by,
            show_star=show_star,
            star_basin=star_basin,
            lead_time=ref_lead,
            figsize=figsize,
            title=title,
            vmin=vmin,
            vmax=vmax,
            config_mode=config_mode,
            filter_complete_streamflow=filter_complete_streamflow,
            year=year,
            df_ts=ts_for_filter,
            color_mode=color_mode,
            bin_thresholds=bin_thresholds,
            marker_size=marker_size,
            region=region,
            koppen=koppen,
            koppen_filter=koppen_filter,
            koppen_alpha=koppen_alpha,
            koppen_df=self.get_basin_koppen() if (koppen != "off" or str(koppen_filter).lower() != "all") else None,
            font_scale=font_scale,
            grid_leads=grid_leads,
            edge_width=edge_width,
            n_bins=n_bins,
        )
        plt.show(block=False)
        return fig

    # -------------------------------------------------------------
    # Stage 9: External Benchmark Comparison (Zenodo 2024 Dual-LSTM & Pretrained Foundation Model)
    # -------------------------------------------------------------
    def compare_with_zenodo_baseline(
        self,
        zenodo_dir: str = "/usr/local/google/home/kruparell/zenodo_2024_paper/metrics/hydrograph_metrics/per_gauge/google/2014/dual_lstm/full_run",
        metric: str = "NSE",
        lead_time_focus: int = 1,
        show_table: bool = True,
        show_plots: bool = True,
        pretrained_metrics_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compares active DA evaluation results against the 2024 Zenodo Dual-LSTM paper baseline
        and the official Pretrained Flood Hub model metrics.

        Automatically resolves both direct GRDC_* basin IDs and CAMELS/HYSETS/Caravans IDs
        via the Caravans V2 nat_id crosswalk.

        Args:
            zenodo_dir: Directory containing per-gauge GRDC_*.csv metric files from Zenodo 2024 paper.
            metric: Metric to compare ('NSE' or 'KGE').
            lead_time_focus: Primary lead horizon for ECDF and scatter plot (default: 1 day).
            show_table: Display formatted multi-leadtime comparison matrix.
            show_plots: Render 4-panel comparative visualization suite.
            pretrained_metrics_path: Path to pretrained model metrics.csv or test_metrics.csv.

        Returns:
            Tuple of (df_comparison_table, df_zenodo_metrics).
        """
        if self.df_eval.empty:
            print("[WARNING] Active session evaluation dataframe is empty.")
            return pd.DataFrame(), pd.DataFrame()

        eval_basins = self.df_eval["Basin ID"].dropna().unique().tolist()
        print(f"Scanning Zenodo 2024 Dual-LSTM metrics across {len(eval_basins):,} evaluated catchments...")

        df_zenodo = load_zenodo_gauge_metrics(
            zenodo_dir=zenodo_dir,
            requested_basins=eval_basins,
            attr_zarr_path=self.attr_zarr_path,
        )

        if df_zenodo.empty:
            print("[WARNING] No overlapping gauges found in Zenodo directory for active experiment.")
            return pd.DataFrame(), pd.DataFrame()

        matched_cnt = df_zenodo["Basin ID"].nunique()
        direct_cnt = df_zenodo[df_zenodo["Match Type"] == "Direct GRDC"]["Basin ID"].nunique()
        cross_cnt = df_zenodo[df_zenodo["Match Type"] == "Crosswalk (nat_id)"]["Basin ID"].nunique()
        print(
            f"[MATCH] Found {matched_cnt:,} overlapping gauges with Zenodo 2024 Dual-LSTM "
            f"({direct_cnt:,} Direct GRDC | {cross_cnt:,} Cross-matched via nat_id)."
        )

        # Ingest precalculated benchmark metrics from pretrained model (metrics.csv / test_metrics.csv)
        df_pretrained = load_pretrained_metrics(
            metrics_path=pretrained_metrics_path,
            requested_basins=eval_basins,
        )
        if not df_pretrained.empty:
            pt_matched = df_pretrained["Basin ID"].nunique()
            print(f"[MATCH] Loaded {pt_matched:,} precalculated benchmark metrics from pretrained model.")
            self.df_pretrained = df_pretrained

        df_table = build_zenodo_comparison_table(
            df_eval=self.df_eval,
            df_zenodo=df_zenodo,
            metric=metric,
            df_pretrained=df_pretrained,
        )

        if show_table and not df_table.empty:
            display(style_zenodo_comparison_table(df_table, metric=metric))

        if show_plots:
            fig = plot_zenodo_comparison_suite(
                df_eval=self.df_eval,
                df_zenodo=df_zenodo,
                metric=metric,
                lead_time_focus=lead_time_focus,
                df_pretrained=df_pretrained,
            )
            plt.show()

        return df_table, df_zenodo

    def show_zenodo_per_gauge_table(
        self,
        zenodo_dir: str = "/usr/local/google/home/kruparell/zenodo_2024_paper/metrics/hydrograph_metrics/per_gauge/google/2014/dual_lstm/full_run",
        lead_time: int = 1,
        metric: str = "NSE",
        top_n: int = 25,
        sort_by: str = "DA vs Zenodo",
        pretrained_metrics_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Displays a side-by-side per-gauge leaderboard comparing DA models against Zenodo Dual-LSTM and Pretrained Model."""
        from da_eval.zenodo_benchmark import build_zenodo_per_gauge_table, style_zenodo_per_gauge_table

        if self.df_eval.empty:
            return pd.DataFrame()

        eval_basins = self.df_eval["Basin ID"].dropna().unique().tolist()
        df_zenodo = load_zenodo_gauge_metrics(
            zenodo_dir=zenodo_dir,
            requested_basins=eval_basins,
            attr_zarr_path=self.attr_zarr_path,
        )
        if df_zenodo.empty:
            return pd.DataFrame()

        df_pretrained = pd.DataFrame()
        if pretrained_metrics_path is not None:
            df_pretrained = load_pretrained_metrics(
                metrics_path=pretrained_metrics_path,
                requested_basins=eval_basins,
            )
        elif hasattr(self, "df_pretrained") and not self.df_pretrained.empty:
            df_pretrained = self.df_pretrained
        else:
            df_pretrained = load_pretrained_metrics(None, requested_basins=eval_basins)

        df_gauges = build_zenodo_per_gauge_table(
            df_eval=self.df_eval,
            df_zenodo=df_zenodo,
            lead_time=lead_time,
            metric=metric,
            top_n=top_n,
            sort_by=sort_by,
            df_pretrained=df_pretrained,
        )
        display(style_zenodo_per_gauge_table(df_gauges, metric=metric))
        return df_gauges

    def show_pretrained_benchmark_summary(
        self,
        metric: str = "NSE",
        lead_time: int = 1,
        pretrained_metrics_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Displays a direct statistical comparison against the Pretrained Flood Hub model (metrics.csv)."""
        import numpy as np

        if self.df_eval.empty:
            return pd.DataFrame()

        eval_basins = self.df_eval["Basin ID"].dropna().unique().tolist()
        if pretrained_metrics_path is not None:
            df_pt = load_pretrained_metrics(metrics_path=pretrained_metrics_path, requested_basins=eval_basins)
        elif hasattr(self, "df_pretrained") and not self.df_pretrained.empty:
            df_pt = self.df_pretrained
        else:
            df_pt = load_pretrained_metrics(None, requested_basins=eval_basins)

        if df_pt.empty:
            print("[WARNING] Pretrained model metrics not available for comparison.")
            return pd.DataFrame()

        self.df_pretrained = df_pt
        m_upper = metric.upper()
        base_col = f"Base {m_upper}" if f"Base {m_upper}" in self.df_eval.columns else "Base NSE"
        da_col = f"DA {m_upper}" if f"DA {m_upper}" in self.df_eval.columns else "DA NSE"
        delta_col = f"{m_upper} Delta" if f"{m_upper} Delta" in self.df_eval.columns else "NSE Delta"
        pt_col = f"Pretrained {m_upper}"

        if pt_col not in df_pt.columns:
            print(f"[WARNING] Column '{pt_col}' not found in pretrained metrics.")
            return pd.DataFrame()

        ref_lead = lead_time if lead_time in self.lead_times else (1 if 1 in self.lead_times else self.lead_times[0])
        best_cfg = get_global_best_config(self.df_eval, lead_time=ref_lead, metric_col=delta_col)

        df_l1 = self.df_eval[self.df_eval["Lead Time (Days)"] == ref_lead].copy()
        base_l1 = df_l1.drop_duplicates("Basin ID").set_index("Basin ID")[base_col]
        global_l1 = df_l1[df_l1["Config ID"] == best_cfg].drop_duplicates("Basin ID").set_index("Basin ID")[da_col]
        oracle_l1 = df_l1[df_l1["Config ID"] == "Per-Basin Best DA"].drop_duplicates("Basin ID").set_index("Basin ID")[da_col]

        pt_series = df_pt.drop_duplicates("Basin ID").set_index("Basin ID")[pt_col].dropna()
        common_basins = pt_series.index.intersection(base_l1.index)
        if len(common_basins) == 0:
            print("[WARNING] No overlapping basins found with Pretrained Flood Hub metrics.")
            return pd.DataFrame()

        pt_vals = pt_series.loc[common_basins]
        base_vals = base_l1.loc[common_basins]
        global_vals = global_l1.loc[common_basins].dropna()
        oracle_vals = oracle_l1.loc[common_basins].dropna()

        summary_rows = [
            {
                "Model / Pipeline": "Pretrained Flood Hub Model (metrics.csv)",
                "Evaluated Catchments": len(common_basins),
                f"Median {m_upper}": pt_vals.median(),
                f"Mean {m_upper}": pt_vals.mean(),
                f"Median Δ{m_upper} vs Pretrained": 0.0,
                "Win Rate vs Pretrained": np.nan,
            },
            {
                "Model / Pipeline": "OpenHydroNets Baseline (Open-Loop)",
                "Evaluated Catchments": len(common_basins),
                f"Median {m_upper}": base_vals.median(),
                f"Mean {m_upper}": base_vals.mean(),
                f"Median Δ{m_upper} vs Pretrained": (base_vals - pt_vals).median(),
                "Win Rate vs Pretrained": (base_vals > pt_vals).mean() * 100.0,
            },
            {
                "Model / Pipeline": f"Global Best DA ({best_cfg})",
                "Evaluated Catchments": len(global_vals),
                f"Median {m_upper}": global_vals.median(),
                f"Mean {m_upper}": global_vals.mean(),
                f"Median Δ{m_upper} vs Pretrained": (global_vals - pt_vals.loc[global_vals.index]).median(),
                "Win Rate vs Pretrained": (global_vals > pt_vals.loc[global_vals.index]).mean() * 100.0,
            },
            {
                "Model / Pipeline": "Per-Basin Best DA (Oracle)",
                "Evaluated Catchments": len(oracle_vals),
                f"Median {m_upper}": oracle_vals.median(),
                f"Mean {m_upper}": oracle_vals.mean(),
                f"Median Δ{m_upper} vs Pretrained": (oracle_vals - pt_vals.loc[oracle_vals.index]).median(),
                "Win Rate vs Pretrained": (oracle_vals > pt_vals.loc[oracle_vals.index]).mean() * 100.0,
            },
        ]

        df_pt_summary = pd.DataFrame(summary_rows)
        print(f"=== Pretrained Model (metrics.csv) Benchmark Synthesis ({len(common_basins):,} Overlapping Catchments | Lead t+{ref_lead}) ===")
        display(
            df_pt_summary.style.format(
                {
                    "Evaluated Catchments": "{:,}",
                    f"Median {m_upper}": "{:.4f}",
                    f"Mean {m_upper}": "{:.4f}",
                    f"Median Δ{m_upper} vs Pretrained": "{:+.4f}",
                    "Win Rate vs Pretrained": "{:.1f}%",
                },
                na_rep="-",
            )
            .background_gradient(subset=[f"Median {m_upper}", f"Mean {m_upper}"], cmap="YlGnBu", vmin=0.3, vmax=0.85)
            .background_gradient(subset=[f"Median Δ{m_upper} vs Pretrained"], cmap="RdYlGn", vmin=-0.1, vmax=0.1)
        )
        return df_pt_summary

    # -------------------------------------------------------------
    # Extended Paper Metrics & Cross-Validation Suites (Per-Leadtime Columns + Interactive Toggles)
    # -------------------------------------------------------------
    def _get_extended_zarr_caches(self, overwrite: bool = False) -> Dict[str, pd.DataFrame]:
        """Loads (or builds on first call) the Zarr-derived parquet caches in ``<data_dir>/_ts_cache/``."""
        from da_eval.extended_metrics import build_or_load_zarr_caches

        if overwrite or not hasattr(self, "_ext_zarr_caches") or not self._ext_zarr_caches:
            if self.data_dir is None:
                return {"yearly_stats": pd.DataFrame(), "persistence": pd.DataFrame(), "event_peaks": pd.DataFrame()}
            self._ext_zarr_caches = build_or_load_zarr_caches(self.data_dir, overwrite=overwrite)
        return self._ext_zarr_caches

    def _filter_basins_for_suite(
        self,
        catchment_filter: str = "all",
        df_pers: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[set], str]:
        """Resolves ``(matching_basin_set, label)`` for a ``catchment_filter`` key in ``CATCHMENT_FILTER_OPTIONS``."""
        from da_eval.extended_metrics import CATCHMENT_FILTER_OPTIONS, filter_basins_by_catchment_type

        cf = str(catchment_filter or "all")
        lbl = dict((v, k) for k, v in CATCHMENT_FILTER_OPTIONS).get(cf, "All Catchments")
        if cf == "all":
            return None, lbl
        kg_df = None
        if cf.startswith("kg_"):
            try:
                kg_df = self.get_basin_koppen()
            except Exception:  # noqa: BLE001
                kg_df = None
        bset, lbl = filter_basins_by_catchment_type(
            self.df_eval,
            catchment_filter=cf,
            df_meta=getattr(self, "df_meta", None),
            df_koppen=kg_df,
            df_pers=df_pers,
        )
        return bset, lbl

    def _make_suite_model_toggles(self, widgets_mod, show: Optional[Sequence[str]] = None):
        """Creates the standard 4-model ToggleButton bar matching ``session.plot_ecdfs``."""
        init_show = set(("global_da", "per_basin_da", "global_rho", "per_basin_rho") if show is None else show)
        has_rho = self.df_eval["Config ID"].apply(is_rho_config).any() if not self.df_eval.empty else True
        _model_keys = [k for k in ECDF_MODELS if has_rho or "rho" not in k]
        w_models = {
            k: widgets_mod.ToggleButton(
                value=k in init_show,
                description=ECDF_MODEL_LABELS[k],
                layout=widgets_mod.Layout(width="155px"),
                style={"text_color": palette.color(k), "font_weight": "bold"},
                tooltip=f"Toggle {ECDF_MODEL_LABELS[k]}",
            )
            for k in _model_keys
        }
        return w_models

    def show_wilcoxon_significance_table(
        self,
        windows: Optional[Dict[str, Sequence[int]] | str] = None,
        metric: str = "skill_nse",
        lead_time: int = 1,
        interactive: bool = True,
        show: Optional[Sequence[str]] = None,
        catchment_filter: str = "all",
        stat_view: str = "compact",
        common_basins: bool = True,
        metric_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Displays Paired Two-Sided Wilcoxon Signed-Rank p-values (with Holm–Bonferroni correction)
        and win rates vs. Open-Loop Baseline and vs. PP post-processing with **Leadtimes (t+1..t+7) as Columns** by default.
        """
        from da_eval.extended_metrics import (
            CATCHMENT_FILTER_OPTIONS,
            build_wilcoxon_significance_table,
            style_wilcoxon_significance_table,
        )

        metric_map = {
            "skill_nse": "NSE Skill Score",
            "NSE Skill Score": "NSE Skill Score",
            "delta_nse": "NSE Delta",
            "NSE Delta": "NSE Delta",
            "skill_kge": "KGE Skill Score",
            "KGE Skill Score": "KGE Skill Score",
            "delta_kge": "KGE Delta",
            "KGE Delta": "KGE Delta",
        }
        m_in = metric_col if metric_col is not None else metric
        m_col = metric_map.get(m_in, m_in)
        caches = self._get_extended_zarr_caches()
        df_pers = caches.get("persistence", pd.DataFrame())

        def _compute(win_val, m_val, show_val, cf_val, sv_val):
            bset, _ = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            df_in = self.df_eval[self.df_eval["Basin ID"].isin(bset)] if bset is not None else self.df_eval
            return build_wilcoxon_significance_table(
                df_in,
                windows=win_val,
                metric_col=m_val,
                common_basins=common_basins,
                show=show_val,
                stat_view=sv_val,
                lead_times=self.lead_times,
            )

        df0 = _compute(windows, m_col, show, catchment_filter, stat_view)
        if not interactive:
            display(style_wilcoxon_significance_table(df0))
            return df0

        try:
            import ipywidgets as widgets
        except ImportError:
            display(style_wilcoxon_significance_table(df0))
            return df0

        w_models = self._make_suite_model_toggles(widgets, show=show)
        w_metric = widgets.ToggleButtons(
            options=[
                ("NSE Skill Score", "NSE Skill Score"),
                ("ΔNSE", "NSE Delta"),
                ("KGE Skill Score", "KGE Skill Score"),
                ("ΔKGE", "KGE Delta"),
            ],
            value=m_col if m_col in ("NSE Skill Score", "NSE Delta", "KGE Skill Score", "KGE Delta") else "NSE Skill Score",
            description="Metric:",
            button_style="info",
            style={"description_width": "initial", "button_width": "135px"},
        )
        w_cols = widgets.ToggleButtons(
            options=[("Per-Leadtime (t+1..t+7)", "per_lead"), ("Grouped Windows (Short/Med/Full)", "grouped")],
            value="grouped" if (isinstance(windows, str) and windows == "grouped") else "per_lead",
            description="Columns:",
            style={"description_width": "initial", "button_width": "210px"},
        )
        w_stat = widgets.Dropdown(
            options=[
                ("Compact Summary (Median, Win%, Harm%, vs PP)", "compact"),
                ("All Rows (incl. Raw Holm-Bonferroni p-values)", "all"),
                ("Median + Significance Stars Only", "median"),
                ("Win Rate (%) vs Baseline Only", "win_base"),
                ("Difference vs PP Only", "vs_ar1"),
            ],
            value=stat_view if stat_view in ("compact", "all", "median", "win_base", "vs_ar1") else "compact",
            description="Rows:",
            layout=widgets.Layout(width="340px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchments:",
            layout=widgets.Layout(width="340px"),
            style={"description_width": "initial"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_models = [k for k, b in w_models.items() if b.value] or list(w_models.keys())
                tbl = _compute(w_cols.value, w_metric.value, sel_models, w_cf.value, w_stat.value)
                display(style_wilcoxon_significance_table(tbl))

        for w in (*w_models.values(), w_metric, w_cols, w_stat, w_cf):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([w_metric, w_cols], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([w_cf, w_stat], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([widgets.HTML("<b style='margin-right:8px'>Models:</b>"), *w_models.values()],
                         layout=widgets.Layout(margin="0 0 6px 0", align_items="center")),
        ])
        display(controls, out)
        _render()
        return df0

    def show_kge_decomposition_suite(
        self,
        windows: Optional[Dict[str, Sequence[int]] | str] = None,
        stat: str = "median",
        lead_time: int = 1,
        interactive: bool = True,
        show: Optional[Sequence[str]] = None,
        catchment_filter: str = "all",
        value_mode: str = "both",
        component_filter: str = "all",
        show_table: bool = True,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """Displays KGE & NSE Component Decomposition (Pearson-r, Alpha-NSE, Beta-KGE, Beta-NSE)
        with **Leadtimes (t+1..t+7) as Columns** by default, plus interactive model, component, catchment, and leadtime toggles.
        """
        from da_eval.extended_metrics import (
            CATCHMENT_FILTER_OPTIONS,
            build_kge_decomposition_table,
            plot_kge_decomposition,
            style_kge_decomposition_table,
        )

        caches = self._get_extended_zarr_caches()
        df_pers = caches.get("persistence", pd.DataFrame())
        ref_lt = int(lead_time) if int(lead_time) in self.lead_times else 1

        def _compute_df(win_val, st_val, show_val, cf_val, vm_val, comp_val):
            bset, _ = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            df_in = self.df_eval[self.df_eval["Basin ID"].isin(bset)] if bset is not None else self.df_eval
            return build_kge_decomposition_table(
                df_in,
                windows=win_val,
                stat=st_val,
                show=show_val,
                value_mode=vm_val,
                component_filter=comp_val,
                lead_times=self.lead_times,
            ), df_in

        df0, df_in0 = _compute_df(windows, stat, show, catchment_filter, value_mode, component_filter)
        if not interactive:
            if show_table and not df0.empty:
                display(style_kge_decomposition_table(df0))
            if show_plot and not df0.empty:
                fig = plot_kge_decomposition(df_in0, lead_times=self.lead_times, lead_time_focus=ref_lt,
                                             component_focus=component_filter, show=show)
                plt.show(block=False)
                plt.close(fig)
            return df0

        try:
            import ipywidgets as widgets
        except ImportError:
            if show_table and not df0.empty:
                display(style_kge_decomposition_table(df0))
            if show_plot and not df0.empty:
                fig = plot_kge_decomposition(df_in0, lead_times=self.lead_times, lead_time_focus=ref_lt,
                                             component_focus=component_filter, show=show)
                plt.show(block=False)
                plt.close(fig)
            return df0

        w_models = self._make_suite_model_toggles(widgets, show=show)
        w_comp = widgets.ToggleButtons(
            options=[
                ("All Components", "all"),
                ("Correlation r", "Pearson-r"),
                ("Variability α (σ_sim/σ_obs)", "Alpha-NSE"),
                ("Volume Bias β_KGE", "Beta-KGE"),
                ("Norm. Bias β_NSE", "Beta-NSE"),
                ("Overall KGE", "KGE"),
            ],
            value=component_filter if component_filter in ("all", "Pearson-r", "Alpha-NSE", "Beta-KGE", "Beta-NSE", "KGE") else "all",
            description="Component:",
            button_style="info",
            style={"description_width": "initial", "button_width": "155px"},
        )
        w_lead = widgets.IntSlider(
            value=ref_lt,
            min=min(self.lead_times),
            max=max(self.lead_times),
            step=1,
            description="Lead Time (t+L):",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
            style={"description_width": "initial"},
        )
        w_cols = widgets.ToggleButtons(
            options=[("Per-Leadtime (t+1..t+7)", "per_lead"), ("Grouped Windows", "grouped")],
            value="grouped" if (isinstance(windows, str) and windows == "grouped") else "per_lead",
            description="Columns:",
            style={"description_width": "initial", "button_width": "175px"},
        )
        w_vmode = widgets.Dropdown(
            options=[("Both (Raw + Skill Score SS_M)", "both"), ("Raw Values Only", "raw"), ("Skill Score SS_M Only", "ss")],
            value=value_mode if value_mode in ("both", "raw", "ss") else "both",
            description="Table Values:",
            layout=widgets.Layout(width="270px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchments:",
            layout=widgets.Layout(width="330px"),
            style={"description_width": "initial"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_models = [k for k, b in w_models.items() if b.value] or list(w_models.keys())
                tbl, df_sub = _compute_df(w_cols.value, stat, sel_models, w_cf.value, w_vmode.value, w_comp.value)
                if show_plot and not df_sub.empty:
                    fig = plot_kge_decomposition(
                        df_sub,
                        lead_times=self.lead_times,
                        lead_time_focus=int(w_lead.value),
                        component_focus=w_comp.value,
                        show=sel_models,
                    )
                    plt.show(block=False)
                    plt.close(fig)
                if show_table and not tbl.empty:
                    display(style_kge_decomposition_table(tbl))

        for w in (*w_models.values(), w_comp, w_lead, w_cols, w_vmode, w_cf):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([w_comp], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([w_lead, w_cf, w_vmode, w_cols], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([widgets.HTML("<b style='margin-right:8px'>Models:</b>"), *w_models.values()],
                         layout=widgets.Layout(margin="0 0 6px 0", align_items="center")),
        ])
        display(controls, out)
        _render()
        return df0

    def show_assimilation_overfitting_suite(
        self,
        lead_time: int = 1,
        interactive: bool = True,
        catchment_filter: str = "all",
        show_table: bool = True,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """Compares in-window optimization fit (t+0) against out-of-sample forecast skill across per-leadtime
        columns ``t+1..t+7``, with an interactive Lead Time slider (``t+L``) and catchment regime filter.
        """
        from da_eval.extended_metrics import (
            CATCHMENT_FILTER_OPTIONS,
            build_assimilation_overfitting_table,
            plot_assimilation_overfitting,
            style_assimilation_overfitting_table,
        )

        caches = self._get_extended_zarr_caches()
        df_pers = caches.get("persistence", pd.DataFrame())
        ref_lt = int(lead_time) if int(lead_time) in self.lead_times else 1

        def _compute(lt_val, cf_val):
            bset, _ = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            df_in = self.df_eval[self.df_eval["Basin ID"].isin(bset)] if bset is not None else self.df_eval
            return build_assimilation_overfitting_table(df_in, lead_time=lt_val), df_in

        df0, df_in0 = _compute(ref_lt, catchment_filter)
        if not interactive:
            if show_table and not df0.empty:
                display(style_assimilation_overfitting_table(df0))
            if show_plot and not df0.empty:
                fig = plot_assimilation_overfitting(df_in0, lead_time=ref_lt)
                plt.show(block=False)
                plt.close(fig)
            return df0

        try:
            import ipywidgets as widgets
        except ImportError:
            if show_table and not df0.empty:
                display(style_assimilation_overfitting_table(df0))
            if show_plot and not df0.empty:
                fig = plot_assimilation_overfitting(df_in0, lead_time=ref_lt)
                plt.show(block=False)
                plt.close(fig)
            return df0

        w_lead = widgets.IntSlider(
            value=ref_lt,
            min=min(self.lead_times),
            max=max(self.lead_times),
            step=1,
            description="Target Lead (t+L):",
            continuous_update=False,
            layout=widgets.Layout(width="300px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchments:",
            layout=widgets.Layout(width="360px"),
            style={"description_width": "initial"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                tbl, df_sub = _compute(int(w_lead.value), w_cf.value)
                if show_plot and not tbl.empty:
                    fig = plot_assimilation_overfitting(df_sub, lead_time=int(w_lead.value))
                    plt.show(block=False)
                    plt.close(fig)
                if show_table and not tbl.empty:
                    display(style_assimilation_overfitting_table(tbl))

        for w in (w_lead, w_cf):
            w.observe(_render, names="value")

        controls = widgets.HBox([w_lead, w_cf], layout=widgets.Layout(margin="0 0 6px 0", flex_flow="row wrap"))
        display(controls, out)
        _render()
        return df0

    def show_persistence_autocorrelation_suite(
        self,
        lead_time: int = 1,
        interactive: bool = True,
        show: Optional[Sequence[str]] = None,
        catchment_filter: str = "all",
        autocorr_type: str = "flow_lagL",
        strata_view: str = "both",
        show_table: bool = True,
        show_plot: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates Naive Persistence NSE, Forecast Skill (SS_NSE), and the **Streamflow & Residual Autocorrelation
        of Observed, Baseline, DA models (Global DA, Per-Basin DA), and Post-Processing models (Global PP, Per-Basin PP)**
        across per-leadtime columns ``t+1..t+7`` and stratified by Lag-1 Open-Loop Residual Autocorrelation Quartiles $\\rho_e(1)$.

        Includes interactive toggles for:
          - **Models** (``global_da``, ``per_basin_da``, ``global_rho``, ``per_basin_rho``)
          - **Catchment Type / Climate Regime** (All, Arid vs. Humid, Köppen A–E, Flashy vs. Baseflow, Snow-Dominant, Base NSE tiers)
          - **Lead Time Slider** (``t+1 .. t+7``) to inspect how model vs. observed autocorrelation and skill evolve with horizon
          - **Autocorrelation Mode** (Lag-$L$ Streamflow $\\rho_Q(L)$, Issue-Date Flow Persistence $\\text{Corr}(\\hat{Q}(t+L|t), Q_{\\text{obs}}(t))$,
            Residual Error Autocorrelation $\\rho_e(L)$ & Assumed PP $\\rho^L$, Nowcast-to-Lead Flow Correlation, or Lag-1 Flow/Error)
        """
        from da_eval.extended_metrics import (
            AUTOCORR_MODE_SPECS,
            CATCHMENT_FILTER_OPTIONS,
            build_persistence_autocorr_table,
            plot_persistence_and_autocorrelation,
            style_persistence_autocorr_tables,
        )

        caches = self._get_extended_zarr_caches()
        df_pers = caches.get("persistence", pd.DataFrame())
        if df_pers.empty:
            print("[WARNING] No persistence/autocorrelation Zarr cache available for this experiment.")
            return pd.DataFrame(), pd.DataFrame()

        ref_lt = int(lead_time) if int(lead_time) in self.lead_times else 1

        def _compute(lt_val, ac_val, show_val, cf_val, sv_val):
            bset, cf_label = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            df_ev_sub = self.df_eval[self.df_eval["Basin ID"].isin(bset)] if bset is not None else self.df_eval
            df_p_sub = df_pers[df_pers["Basin ID"].isin(bset)] if bset is not None else df_pers
            lead_df, strata_df = build_persistence_autocorr_table(
                df_ev_sub,
                df_p_sub,
                lead_times=self.lead_times,
                autocorr_type=ac_val,
                show=show_val,
                strata_view=sv_val,
            )
            return lead_df, strata_df, df_ev_sub, df_p_sub, cf_label

        lead_df0, strata_df0, ev0, p0, cf_lbl0 = _compute(ref_lt, autocorr_type, show, catchment_filter, strata_view)
        if not interactive:
            if show_table:
                sty_lead, sty_strata = style_persistence_autocorr_tables(lead_df0, strata_df0)
                display(sty_lead)
                display(sty_strata)
            if show_plot:
                fig = plot_persistence_and_autocorrelation(
                    ev0, p0, lead_time=ref_lt, autocorr_type=autocorr_type, show=show, catchment_label=cf_lbl0
                )
                plt.show(block=False)
                plt.close(fig)
            return lead_df0, strata_df0

        try:
            import ipywidgets as widgets
        except ImportError:
            if show_table:
                sty_lead, sty_strata = style_persistence_autocorr_tables(lead_df0, strata_df0)
                display(sty_lead)
                display(sty_strata)
            if show_plot:
                fig = plot_persistence_and_autocorrelation(
                    ev0, p0, lead_time=ref_lt, autocorr_type=autocorr_type, show=show, catchment_label=cf_lbl0
                )
                plt.show(block=False)
                plt.close(fig)
            return lead_df0, strata_df0

        w_models = self._make_suite_model_toggles(widgets, show=show)
        w_ac = widgets.Dropdown(
            options=[
                ("1. Lag-L Streamflow Autocorr ρ_Q(L) = Corr(Q̂(t+L,L), Q̂(t,L))", "flow_lagL"),
                ("2. Issue-Date Flow Persistence Corr(Q̂(t+L|t), Q_obs(t))", "issue_to_lead"),
                ("3. Residual Error Autocorr ρ_e(L) & Assumed PP ρ^L", "res_lagL"),
                ("4. Nowcast-to-Lead Flow Corr(Q̂(t+L|t), Q̂(t|t))", "nowcast_to_lead"),
                ("5. Daily Step-to-Step Flow Autocorr ρ_Q(1)", "flow_lag1"),
                ("6. Daily Step-to-Step Error Autocorr ρ_e(1)", "res_lag1"),
            ],
            value=autocorr_type if autocorr_type in AUTOCORR_MODE_SPECS else "flow_lagL",
            description="Autocorr View:",
            layout=widgets.Layout(width="440px"),
            style={"description_width": "initial"},
        )
        w_lead = widgets.IntSlider(
            value=ref_lt,
            min=min(self.lead_times),
            max=max(self.lead_times),
            step=1,
            description="Lead Time (t+L):",
            continuous_update=False,
            layout=widgets.Layout(width="290px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchment Type:",
            layout=widgets.Layout(width="350px"),
            style={"description_width": "initial"},
        )
        w_strata = widgets.ToggleButtons(
            options=[
                ("Both (SS_NSE + Autocorr)", "both"),
                ("Autocorrelation Only", "autocorr"),
                ("Skill Score (SS_NSE) Only", "skill"),
            ],
            value=strata_view if strata_view in ("both", "autocorr", "skill") else "both",
            description="Quartile Table:",
            button_style="info",
            style={"description_width": "initial", "button_width": "165px"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_models = [k for k, b in w_models.items() if b.value] or list(w_models.keys())
                sel_lt = int(w_lead.value)
                l_df, s_df, ev_s, p_s, cf_lbl = _compute(
                    sel_lt, w_ac.value, sel_models, w_cf.value, w_strata.value
                )
                if show_plot and not p_s.empty:
                    fig = plot_persistence_and_autocorrelation(
                        ev_s,
                        p_s,
                        lead_time=sel_lt,
                        autocorr_type=w_ac.value,
                        show=sel_models,
                        catchment_label=cf_lbl,
                    )
                    plt.show(block=False)
                    plt.close(fig)
                if show_table and not l_df.empty:
                    sty_l, sty_s = style_persistence_autocorr_tables(l_df, s_df)
                    display(sty_l)
                    display(sty_s)

        for w in (*w_models.values(), w_ac, w_lead, w_cf, w_strata):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([w_ac, w_lead], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([w_cf, w_strata], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([widgets.HTML("<b style='margin-right:8px'>Models:</b>"), *w_models.values()],
                         layout=widgets.Layout(margin="0 0 6px 0", align_items="center")),
        ])
        display(controls, out)
        _render()
        return lead_df0, strata_df0

    def show_event_peak_and_exceedance_suite(
        self,
        windows: Optional[Dict[str, Sequence[int]] | str] = None,
        lead_time: int = 1,
        interactive: bool = True,
        show: Optional[Sequence[str]] = None,
        catchment_filter: str = "all",
        threshold: str = "both",
        value_mode: str = "both",
        show_table: bool = True,
        show_plot: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluates High-Flow Exceedance (Precision, Recall/POD, CSI, F1, Zero-Alarm % at Q90 and Q95)
        and Flood Peak Timing/Magnitude (Signed & Absolute PTE in ±2d and [-2,+7]d windows, Exact 0-day %, PFE, FHV)
        with **Leadtimes (t+1..t+7) as Columns** by default, plus interactive toggles.
        """
        from da_eval.extended_metrics import (
            CATCHMENT_FILTER_OPTIONS,
            build_event_and_peak_summary_table,
            plot_event_and_peak_metrics,
            style_event_and_peak_tables,
        )

        caches = self._get_extended_zarr_caches()
        df_event = caches.get("event_peaks", pd.DataFrame())
        df_pers = caches.get("persistence", pd.DataFrame())
        if df_event.empty:
            print("[WARNING] No event/peak Zarr cache available for this experiment.")
            return pd.DataFrame(), pd.DataFrame()

        ref_lt = int(lead_time) if int(lead_time) in self.lead_times else 1

        def _compute(win_val, show_val, cf_val, thr_val, vm_val):
            bset, _ = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            ev_sub = df_event[df_event["Basin ID"].isin(bset)] if bset is not None else df_event
            exc_df, peak_df = build_event_and_peak_summary_table(
                ev_sub,
                windows=win_val,
                show=show_val,
                threshold=thr_val,
                value_mode=vm_val,
                lead_times=self.lead_times,
            )
            return exc_df, peak_df, ev_sub

        exc0, peak0, ev0 = _compute(windows, show, catchment_filter, threshold, value_mode)
        if not interactive:
            if show_table:
                sty_exc, sty_peak = style_event_and_peak_tables(exc0, peak0)
                display(sty_exc)
                display(sty_peak)
            if show_plot:
                fig = plot_event_and_peak_metrics(ev0, lead_time_focus=ref_lt, threshold="Q95" if threshold == "both" else threshold, show=show)
                plt.show(block=False)
                plt.close(fig)
            return exc0, peak0

        try:
            import ipywidgets as widgets
        except ImportError:
            if show_table:
                sty_exc, sty_peak = style_event_and_peak_tables(exc0, peak0)
                display(sty_exc)
                display(sty_peak)
            if show_plot:
                fig = plot_event_and_peak_metrics(ev0, lead_time_focus=ref_lt, threshold="Q95" if threshold == "both" else threshold, show=show)
                plt.show(block=False)
                plt.close(fig)
            return exc0, peak0

        w_models = self._make_suite_model_toggles(widgets, show=show)
        w_thr = widgets.ToggleButtons(
            options=[("Both (Q90 & Q95)", "both"), ("Q95 (Extreme High Flow)", "Q95"), ("Q90 (High Flow)", "Q90")],
            value=threshold if threshold in ("both", "Q95", "Q90") else "both",
            description="Threshold:",
            button_style="info",
            style={"description_width": "initial", "button_width": "160px"},
        )
        w_lead = widgets.IntSlider(
            value=ref_lt,
            min=min(self.lead_times),
            max=max(self.lead_times),
            step=1,
            description="Lead Time (t+L):",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
            style={"description_width": "initial"},
        )
        w_cols = widgets.ToggleButtons(
            options=[("Per-Leadtime (t+1..t+7)", "per_lead"), ("Grouped Windows", "grouped")],
            value="grouped" if (isinstance(windows, str) and windows == "grouped") else "per_lead",
            description="Columns:",
            style={"description_width": "initial", "button_width": "175px"},
        )
        w_vmode = widgets.Dropdown(
            options=[("Both (Raw + Skill Score)", "both"), ("Raw Values Only", "raw"), ("Skill Score SS Only", "ss")],
            value=value_mode if value_mode in ("both", "raw", "ss") else "both",
            description="Values:",
            layout=widgets.Layout(width="240px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchments:",
            layout=widgets.Layout(width="330px"),
            style={"description_width": "initial"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_models = [k for k, b in w_models.items() if b.value] or list(w_models.keys())
                e_df, p_df, ev_s = _compute(w_cols.value, sel_models, w_cf.value, w_thr.value, w_vmode.value)
                if show_plot and not ev_s.empty:
                    plot_q = "Q90" if w_thr.value == "Q90" else "Q95"
                    fig = plot_event_and_peak_metrics(
                        ev_s, lead_time_focus=int(w_lead.value), threshold=plot_q, show=sel_models
                    )
                    plt.show(block=False)
                    plt.close(fig)
                if show_table and not e_df.empty:
                    sty_e, sty_p = style_event_and_peak_tables(e_df, p_df)
                    display(sty_e)
                    display(sty_p)

        for w in (*w_models.values(), w_thr, w_lead, w_cols, w_vmode, w_cf):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([w_thr, w_cols], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([w_lead, w_cf, w_vmode], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([widgets.HTML("<b style='margin-right:8px'>Models:</b>"), *w_models.values()],
                         layout=widgets.Layout(margin="0 0 6px 0", align_items="center")),
        ])
        display(controls, out)
        _render()
        return exc0, peak0

    def show_temporal_cv_suite(
        self,
        cv_years: Sequence[int] = (2017, 2018, 2019, 2020, 2021, 2022),
        lead_time: int = 1,
        interactive: bool = True,
        show: Optional[Sequence[str]] = None,
        catchment_filter: str = "all",
        windows: Optional[Dict[str, Sequence[int]] | str] = None,
        stat_view: str = "median",
        show_table: bool = True,
        show_plot: bool = True,
    ) -> Dict[str, Any]:
        """Runs 6-Fold Leave-One-Year-Out Cross-Validation (5-Year Train -> 1-Year Out-of-Sample Test)
        with **Leadtimes (t+1..t+7) as Columns** by default across summary, calibration-ladder, and fold tables,
        plus interactive toggles for Lead Time ``t+L``, Catchment Regime, and Model comparison.
        """
        from da_eval.extended_metrics import (
            CATCHMENT_FILTER_OPTIONS,
            compute_temporal_cv,
            plot_temporal_cv_results,
            style_temporal_cv_tables,
        )

        caches = self._get_extended_zarr_caches()
        df_yearly = caches.get("yearly_stats", pd.DataFrame())
        df_pers = caches.get("persistence", pd.DataFrame())
        if df_yearly.empty:
            print("[WARNING] No yearly sufficient-statistics cache available for temporal CV.")
            return {}

        ref_lt = int(lead_time) if int(lead_time) in self.lead_times else 1

        def _compute(lt_val, win_val, cf_val, sv_val):
            bset, _ = self._filter_basins_for_suite(cf_val, df_pers=df_pers)
            y_sub = df_yearly[df_yearly["Basin ID"].isin(bset)] if bset is not None else df_yearly
            return compute_temporal_cv(
                y_sub,
                cv_years=cv_years,
                lead_time=lt_val,
                windows=win_val,
                stat_view=sv_val,
            )

        cv_res0 = _compute(ref_lt, windows, catchment_filter, stat_view)
        if not cv_res0:
            print("[WARNING] Temporal CV requires multi-year data across cv_years.")
            return {}

        if not interactive:
            if show_table:
                styled = style_temporal_cv_tables(
                    cv_res0["summary_table"],
                    cv_res0["fold_table"],
                    cv_res0.get("calibration_ladder_table"),
                )
                for s_tbl in (styled if isinstance(styled, tuple) else (styled,)):
                    display(s_tbl)
            if show_plot:
                fig = plot_temporal_cv_results(cv_res0, lead_time=ref_lt, show=show)
                plt.show(block=False)
                plt.close(fig)
            return cv_res0

        try:
            import ipywidgets as widgets
        except ImportError:
            if show_table:
                styled = style_temporal_cv_tables(
                    cv_res0["summary_table"],
                    cv_res0["fold_table"],
                    cv_res0.get("calibration_ladder_table"),
                )
                for s_tbl in (styled if isinstance(styled, tuple) else (styled,)):
                    display(s_tbl)
            if show_plot:
                fig = plot_temporal_cv_results(cv_res0, lead_time=ref_lt, show=show)
                plt.show(block=False)
                plt.close(fig)
            return cv_res0

        w_models = self._make_suite_model_toggles(widgets, show=show)
        w_lead = widgets.IntSlider(
            value=ref_lt,
            min=min(self.lead_times),
            max=max(self.lead_times),
            step=1,
            description="Lead Time (t+L):",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
            style={"description_width": "initial"},
        )
        w_cols = widgets.ToggleButtons(
            options=[("Per-Leadtime (t+1..t+7)", "per_lead"), ("Grouped Windows", "grouped")],
            value="grouped" if (isinstance(windows, str) and windows == "grouped") else "per_lead",
            description="Columns:",
            style={"description_width": "initial", "button_width": "175px"},
        )
        w_stat = widgets.Dropdown(
            options=[
                ("Median SS_NSE Only", "median"),
                ("All (Median, Q10 Risk, % Harmed)", "all"),
                ("Q10 Downside Risk Only", "q10"),
                ("% Basins Harmed (<-1%) Only", "harm"),
            ],
            value=stat_view if stat_view in ("median", "all", "q10", "harm") else "median",
            description="Summary Rows:",
            layout=widgets.Layout(width="270px"),
            style={"description_width": "initial"},
        )
        w_cf = widgets.Dropdown(
            options=list(CATCHMENT_FILTER_OPTIONS),
            value=catchment_filter if catchment_filter in [v for _, v in CATCHMENT_FILTER_OPTIONS] else "all",
            description="Catchments:",
            layout=widgets.Layout(width="330px"),
            style={"description_width": "initial"},
        )
        out = widgets.Output()

        def _render(*_):
            with out:
                out.clear_output(wait=True)
                sel_models = [k for k, b in w_models.items() if b.value] or list(w_models.keys())
                sel_lt = int(w_lead.value)
                res = _compute(sel_lt, w_cols.value, w_cf.value, w_stat.value)
                if not res:
                    print("[WARNING] No basins match the selected filter across all CV years.")
                    return
                if show_plot:
                    fig = plot_temporal_cv_results(res, lead_time=sel_lt, show=sel_models)
                    plt.show(block=False)
                    plt.close(fig)
                if show_table:
                    styled = style_temporal_cv_tables(
                        res["summary_table"],
                        res["fold_table"],
                        res.get("calibration_ladder_table"),
                    )
                    for s_tbl in (styled if isinstance(styled, tuple) else (styled,)):
                        display(s_tbl)

        for w in (*w_models.values(), w_lead, w_cols, w_stat, w_cf):
            w.observe(_render, names="value")

        controls = widgets.VBox([
            widgets.HBox([w_lead, w_cols, w_stat], layout=widgets.Layout(margin="0 0 4px 0", flex_flow="row wrap")),
            widgets.HBox([w_cf, widgets.HTML("<b style='margin:0 8px 0 12px'>Models:</b>"), *w_models.values()],
                         layout=widgets.Layout(margin="0 0 6px 0", align_items="center", flex_flow="row wrap")),
        ])
        display(controls, out)
        _render()
        return cv_res0


DAEvaluationSession = DAAnalysisSession
