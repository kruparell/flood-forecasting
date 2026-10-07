"""Lazy, per-basin access to tester.py ``test_results*.zarr`` forecast stores.

Why: the legacy eager loader materialised every (config, basin, lead, day) row
(~438M rows for Stage 49), which never finishes. The stores are chunked one
basin per chunk, so reading a single basin for a handful of configs is ~0.1 s.

Components
----------
* ``LazyZarrTimeseries``: discovers stores, builds (and caches to
  ``<data_dir>/_ts_cache/index.json``) a (config, basin) -> (store, row) index,
  and reads basins on demand with an LRU cache. Store handles open lazily.
* ``obs_frame``: observed streamflow only, all basins, one lead (cached parquet).
* ``lead_frame``: baseline + one DA config at one lead for many basins
  (vectorised; cached parquet) for the seasonal-flow strata.
* ``block_to_frame``: vectorised (basin, date, time_step) -> long DataFrame
  conversion shared with the eager legacy loader.

``samples`` are reduced with the CMAL **mixture mean** (index 0 of the
``cmal_deterministic`` 10-point summary), matching ``tester.py``'s
``mixture_mean`` reduction used for the metrics CSVs. See ``reduce_samples``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASELINE_CFG = "Baseline"
INDEX_VERSION = 1


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def config_id_from_store_path(z_path: str) -> Tuple[str, str]:
    """Returns ``(cfg_id, raw_folder_name)`` for a ``.../<cfg>/[shard_NN/]test/<epoch>/test_results*.zarr`` path.

    Mirrors the Config ID normalisation used by the legacy CSV / Zarr loaders.
    """
    parts = Path(z_path).parts
    cfg_name = None
    for i, p in enumerate(parts):
        if p == "test" and i > 0:
            candidate = parts[i - 1]
            cfg_name = parts[i - 2] if (re.match(r"^shard_\d+$", candidate) and i >= 2) else candidate
            break
    if not cfg_name:
        cfg_name = os.path.basename(os.path.dirname(z_path))
    low = cfg_name.lower()
    if "baseline" in low:
        return BASELINE_CFG, cfg_name
    if "mf2lstm" in low:
        return cfg_name, cfg_name
    if any(k in low for k in ("both_", "dyn_", "stat_", "all_", "triple_", "dualdyn_", "_fc_")) and not low.startswith("embedded_"):
        return f"embedded_{cfg_name}", cfg_name
    return cfg_name, cfg_name


SAMPLE_REDUCTION = "mixture_mean"


def reduce_samples(sim: np.ndarray, reduction: Optional[str] = None) -> np.ndarray:
    """Reduces the trailing ``samples`` axis to a point estimate.

    Mirrors ``tester.py::_reduce_samples``. For the ``cmal_deterministic`` head the
    samples axis is NOT a set of draws but the 10-point summary
    ``[mixture_mean, q0.1, ..., q0.9]``, so:

    * ``mixture_mean`` (default): index 0, the CMAL conditional mean. This is
      what the Stage 49 ``test_metrics*.csv`` NSE/KGE are computed from
      (verified: recomputed NSE matches the CSV to 3 d.p.).
    * ``mixture_median``: index 5 (q0.5).
    * ``mean``: arithmetic mean across the 10 summary points. It blends
      heterogeneous quantiles and is biased (e.g. hysets_02323500 lead-1 NSE
      0.73 vs 0.90 for the mixture mean), so only use it for true sample draws.
    """
    reduction = (reduction or SAMPLE_REDUCTION).lower()
    if sim.ndim < 4:
        return sim
    if sim.shape[-1] == 1 or reduction == "mixture_mean":
        return sim[..., 0]
    if reduction == "mixture_median":
        return sim[..., 5]
    with np.errstate(invalid="ignore"):
        if reduction == "median":
            return np.nanmedian(sim, axis=-1)
        return np.nanmean(sim, axis=-1)


# -----------------------------------------------------------------------------
# PP residual post-processing (synthesised on the fly from the Baseline store)
# -----------------------------------------------------------------------------
PER_BASIN_RHO = "per_basin"
_AR1_RE = re.compile(r"^ar1_(?:postprocess_)?rho[_=]?([0-9]*\.?[0-9]+)$", re.IGNORECASE)


def parse_ar1_config(cfg: Optional[str]):
    """``AR1_postprocess_rho0.65`` / ``AR1_rho0.7`` -> 0.65 / 0.7; ``AR1_Per_Basin_Best_Rho`` -> ``PER_BASIN_RHO``; else None."""
    if not cfg:
        return None
    c = str(cfg).strip()
    m = _AR1_RE.match(c)
    if m:
        return float(m.group(1))
    low = c.lower()
    if low.startswith("ar1_") and ("per_basin" in low or "per-basin" in low):
        return PER_BASIN_RHO
    return None


def ar1_config_name(rho: float) -> str:
    """Canonical Config ID for a fixed-rho PP series (matches the metrics folder naming)."""
    return f"AR1_postprocess_rho{float(rho):g}"


def ar1_from_baseline(sim: np.ndarray, obs: np.ndarray, ts_coords: Sequence[int], rho) -> np.ndarray:
    """PP error correction of the open-loop Baseline forecast.

    For an issue date T with observed flow ``q_obs(T)`` and Baseline nowcast ``q_base(T, t+0)``::

        e_T          = q_obs(T) - q_base(T, t+0)
        q_ar1(T, L)  = max(0, q_base(T, L) + rho**L * e_T)      L = 1..7

    If ``q_obs(T)`` is missing, ``e_T = 0`` (falls back to the Baseline).

    Verified against the Stage 49 ``AR1_postprocess_rho*`` metrics CSVs: identical NSE (float32
    noise, <2e-5) for basins with no observation gaps at issue time. For basins with gaps the
    original pipeline's gap handling is unknown, so NSE differs slightly (median ~5e-3).

    Args:
      sim: Baseline forecasts ``(B, D, T)`` (samples already reduced).
      obs: Observations ``(B, D, T)``; ``obs[..., ts==0]`` is ``q_obs(T)``.
      ts_coords: ``time_step`` coordinate values (must contain 0).
      rho: Scalar or per-basin array of shape ``(B,)``.
    Returns:
      Array like ``sim`` with leads >= 1 corrected (t+0 unchanged).
    """
    ts = list(ts_coords)
    if 0 not in ts:
        raise ValueError("PP reconstruction needs the t+0 nowcast (time_step 0) in the Baseline store.")
    i0 = ts.index(0)
    o = obs if obs.ndim == 3 else obs[:, :, None]
    e = np.nan_to_num(o[:, :, min(i0, o.shape[2] - 1)] - sim[:, :, i0], nan=0.0)  # (B, D)
    rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1, 1) if np.ndim(rho) else float(rho)  # (B,1) or scalar
    out = sim.astype(np.float64, copy=True)
    for k, lead in enumerate(ts[: sim.shape[2]]):
        if int(lead) >= 1:
            corr = sim[:, :, k] + np.power(rho_arr, int(lead)) * e
            out[:, :, k] = np.where(np.isnan(corr), sim[:, :, k], np.maximum(0.0, corr))
    return out


def lead_indices(ts_coords: Sequence[int], n_steps: int, lead_times: Optional[Iterable[int]] = None) -> List[Tuple[int, int]]:
    """``[(ts_idx, lead)]`` following the legacy convention (t+0 aware; leads 1..7 only)."""
    wanted = set(int(l) for l in lead_times) if lead_times is not None else None
    has_zero = 0 in list(ts_coords)
    out = []
    for ts_idx, ts_val in enumerate(ts_coords):
        if ts_idx >= n_steps:
            break
        lead = int(ts_val) if has_zero else ts_idx + 1
        if 1 <= lead <= 7 and (wanted is None or lead in wanted):
            out.append((ts_idx, lead))
    return out


def block_to_frame(
    basins: Sequence[str],
    dates: np.ndarray,
    ts_coords: Sequence[int],
    sim: Optional[np.ndarray],
    obs: Optional[np.ndarray],
    cfg_id: Optional[str],
    lead_times: Optional[Iterable[int]] = None,
    sim_col: str = "q_da",
) -> pd.DataFrame:
    """Vectorised conversion of ``(basin, date, time_step)`` arrays into the long timeseries schema.

    Columns: Valid Date, Basin ID, [Config ID], Lead Time (Days), [q_obs], [sim_col].
    """
    ref = sim if sim is not None else obs
    n_b, n_d = ref.shape[0], ref.shape[1]
    n_t = ref.shape[2] if ref.ndim == 3 else 1
    pairs = lead_indices(ts_coords, n_t, lead_times)
    if not pairs or n_b == 0:
        return pd.DataFrame()
    dates = pd.to_datetime(dates).values.astype("datetime64[ns]")
    frames = []
    basins_arr = np.asarray(basins, dtype=object)
    for ts_idx, lead in pairs:
        d = {
            "Valid Date": np.tile(dates + np.timedelta64(lead, "D"), n_b),
            "Basin ID": np.repeat(basins_arr, n_d),
            "Lead Time (Days)": np.full(n_b * n_d, lead, dtype=np.int64),
        }
        if cfg_id is not None:
            d["Config ID"] = cfg_id
        if obs is not None:
            o = obs if obs.ndim == 3 else obs[:, :, None]
            d["q_obs"] = o[:, :, min(ts_idx, o.shape[2] - 1)].reshape(-1)
        if sim is not None:
            d[sim_col] = sim[:, :, ts_idx].reshape(-1)
        frames.append(pd.DataFrame(d))
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Lazy store
# -----------------------------------------------------------------------------
class LazyZarrTimeseries:
    """On-demand per-basin reader over all ``test_results*.zarr`` stores in ``data_dir``."""

    def __init__(self, data_dir: str | Path, cache_dir: Optional[str | Path] = None,
                 num_workers: int = 16, lru_size: int = 256, verbose: bool = True):
        self.data_dir = str(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(self.data_dir) / "_ts_cache"
        self.num_workers = num_workers
        self.verbose = verbose
        self._handles: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._lru: "OrderedDict[Tuple[str, str], pd.DataFrame]" = OrderedDict()
        self._lru_size = lru_size
        self._raw_lru: "OrderedDict[Tuple[str, int], tuple]" = OrderedDict()
        # Per-basin rho for the synthetic ``AR1_Per_Basin_Best_Rho`` series (see ``set_basin_rho``).
        self.basin_rho: Dict[str, float] = {}
        self._load_or_build_index()

    # ---------------- discovery & index ----------------
    @staticmethod
    def discover(data_dir: str) -> List[str]:
        stores = glob.glob(os.path.join(data_dir, "*", "shard_*", "test", "*", "test_results*.zarr"))
        if not stores:
            stores = glob.glob(os.path.join(data_dir, "**", "test_results*.zarr"), recursive=True)
        return sorted(stores)

    @staticmethod
    def has_stores(data_dir: str) -> bool:
        return bool(LazyZarrTimeseries.discover(str(data_dir)))

    def _index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def _load_or_build_index(self):
        stores = self.discover(self.data_dir)
        rel = [os.path.relpath(s, self.data_dir) for s in stores]
        idx_path = self._index_path()
        if idx_path.exists():
            try:
                cached = json.loads(idx_path.read_text())
                if cached.get("version") == INDEX_VERSION and sorted(cached["stores"].keys()) == sorted(rel):
                    self._set_index(cached)
                    if self.verbose:
                        print(f"[LAZY TS] Loaded cached index: {len(self.store_meta)} stores, "
                              f"{len(self.configs)} configs, {len(self.basins):,} basins.")
                    return
            except Exception:
                pass
        if self.verbose:
            print(f"[LAZY TS] Building one-time basin index over {len(stores)} zarr stores (cached afterwards)...")

        def _meta(store_rel: str):
            import zarr  # pylint: disable=g-import-not-at-top

            g = zarr.open_consolidated(os.path.join(self.data_dir, store_rel), mode="r")
            basins = [str(b) for b in g["basin"][:]]
            ts = [int(x) for x in g["time_step"][:]] if "time_step" in g else list(range(1, 8))
            units = dict(g["date"].attrs).get("units", "days since 1970-01-01")
            raw = g["date"][:]
            origin = pd.Timestamp(units.split("since", 1)[1].strip()) if "since" in units else pd.Timestamp("1970-01-01")
            step = units.split("since", 1)[0].strip() if "since" in units else "days"
            unit = {"days": "D", "hours": "h", "seconds": "s", "minutes": "min"}.get(step, "D")
            dates = (origin + pd.to_timedelta(raw.astype("int64"), unit=unit)).strftime("%Y-%m-%d").tolist()
            obs_var = "streamflow_obs" if "streamflow_obs" in g else "q_obs"
            sim_var = "streamflow_sim" if "streamflow_sim" in g else "q_sim"
            cfg_id, raw_name = config_id_from_store_path(store_rel)
            return store_rel, {"cfg_id": cfg_id, "raw_name": raw_name, "basins": basins, "time_step": ts,
                               "dates": dates, "obs_var": obs_var, "sim_var": sim_var,
                               "sim_ndim": int(g[sim_var].ndim)}

        with ThreadPoolExecutor(self.num_workers) as ex:
            metas = dict(ex.map(_meta, rel))
        index = {"version": INDEX_VERSION, "stores": metas}
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_path().write_text(json.dumps(index))
        except Exception as e:  # Read-only staging dir: keep in memory.
            print(f"[LAZY TS] Could not cache index ({e}); continuing in memory.")
        self._set_index(index)
        if self.verbose:
            print(f"[LAZY TS] Indexed {len(self.store_meta)} stores, {len(self.configs)} configs, {len(self.basins):,} basins.")

    def _set_index(self, index: dict):
        self.store_meta: Dict[str, dict] = index["stores"]
        self.loc: Dict[str, Dict[str, Tuple[str, int]]] = {}
        self.raw_names: Dict[str, str] = {}
        for store_rel, m in self.store_meta.items():
            cfg = m["cfg_id"]
            self.raw_names[cfg] = m["raw_name"]
            d = self.loc.setdefault(cfg, {})
            for i, b in enumerate(m["basins"]):
                d.setdefault(b, (store_rel, i))
        self.configs = sorted(self.loc.keys())
        self.basins = sorted({b for d in self.loc.values() for b in d})
        self._lower_basin = {b.lower(): b for b in self.basins}
        self._basin_set = set(self.basins)

    # ---------------- config / basin resolution ----------------
    @staticmethod
    def _norm(cfg: str) -> str:
        n = str(cfg).strip().lower()
        n = re.sub(r"_stat1e-0?6$", "", n)
        return n[len("embedded_"):] if n.startswith("embedded_") else n

    def resolve_config(self, cfg: Optional[str]) -> Optional[str]:
        """Maps an evaluation Config ID to the store Config ID (handles embedded_/_stat1e-06 variants).

        PP configs (``AR1_postprocess_rho<x>``, ``AR1_rho<x>``, ``AR1_Per_Basin_Best_Rho``) resolve to a
        synthetic series computed on the fly from the Baseline store (see ``ar1_from_baseline``).
        """
        if not cfg:
            return None
        if cfg in self.loc:
            return cfg
        rho = parse_ar1_config(cfg)
        if rho is not None:
            if not self.supports_ar1:
                return None
            return "AR1_Per_Basin_Best_Rho" if rho == PER_BASIN_RHO else ar1_config_name(rho)
        if "baseline" in str(cfg).lower():
            return BASELINE_CFG if BASELINE_CFG in self.loc else None
        n = self._norm(cfg)
        for c in self.configs:
            if self._norm(c) == n or self._norm(self.raw_names.get(c, c)) == n:
                return c
        return None

    def resolve_basin(self, basin: str) -> Optional[str]:
        b = str(basin)
        return b if b in self._basin_set else self._lower_basin.get(b.lower())

    # ---------------- PP synthetic series ----------------
    @property
    def supports_ar1(self) -> bool:
        """True when a Baseline store with a t+0 nowcast exists (needed for the PP residual)."""
        if BASELINE_CFG not in self.loc:
            return False
        return all(0 in self.store_meta[s]["time_step"] for s in {v[0] for v in self.loc[BASELINE_CFG].values()})

    def set_basin_rho(self, mapping: Dict[str, float]):
        """Sets the per-basin rho used for ``AR1_Per_Basin_Best_Rho`` (invalidates its cached frames)."""
        self.basin_rho = {str(k): float(v) for k, v in mapping.items() if pd.notna(v)}
        with self._lock:
            for key in [k for k in self._lru if k[0] == "AR1_Per_Basin_Best_Rho"]:
                self._lru.pop(key, None)

    def _rho_for(self, rc: str, basin: str) -> Optional[float]:
        rho = parse_ar1_config(rc)
        if rho == PER_BASIN_RHO:
            return self.basin_rho.get(basin)
        return rho

    def _source_cfg(self, rc: str) -> str:
        """Store that physically holds ``rc`` (Baseline for synthetic PP series)."""
        return BASELINE_CFG if parse_ar1_config(rc) is not None else rc

    def basins_for(self, cfg: str) -> List[str]:
        rc = self.resolve_config(cfg)
        if rc is None:
            return []
        basins = list(self.loc.get(self._source_cfg(rc), {}).keys())
        if parse_ar1_config(rc) == PER_BASIN_RHO:
            basins = [b for b in basins if b in self.basin_rho]
        return basins

    # ---------------- store access ----------------
    def _group(self, store_rel: str):
        with self._lock:
            g = self._handles.get(store_rel)
            if g is None:
                import zarr  # pylint: disable=g-import-not-at-top

                g = zarr.open_consolidated(os.path.join(self.data_dir, store_rel), mode="r")
                self._handles[store_rel] = g
            return g

    def _read_rows(self, store_rel: str, rows: Sequence[int], want_sim: bool = True, want_obs: bool = True):
        """Reads ``rows`` (basin positions) from one store -> (sim(B,D,T) mean-reduced, obs(B,D,T))."""
        m = self.store_meta[store_rel]
        g = self._group(store_rel)
        rows = list(rows)
        sel = rows if len(rows) != 1 else slice(rows[0], rows[0] + 1)
        sim = obs = None
        if want_sim:
            arr = g[m["sim_var"]]
            raw = arr.oindex[sel] if isinstance(sel, list) else arr[sel]
            raw = raw[:, 0] if raw.ndim >= 4 else raw  # drop freq
            sim = reduce_samples(raw) if raw.ndim == 4 else raw
        if want_obs:
            arr = g[m["obs_var"]]
            raw = arr.oindex[sel] if isinstance(sel, list) else arr[sel]
            if raw.ndim == 5:
                raw = raw[:, 0, :, :, 0]
            elif raw.ndim == 4:
                raw = raw[:, 0]
            obs = raw
        return sim, obs

    def _read_row_cached(self, store_rel: str, row: int):
        """Single-row read with a small LRU (lets many PP rho variants share one Baseline read)."""
        key = (store_rel, int(row))
        with self._lock:
            hit = self._raw_lru.get(key)
            if hit is not None:
                self._raw_lru.move_to_end(key)
                return hit
        val = self._read_rows(store_rel, [row])
        with self._lock:
            self._raw_lru[key] = val
            if len(self._raw_lru) > 64:
                self._raw_lru.popitem(last=False)
        return val

    # ---------------- public API ----------------
    def get_basin(self, basin: str, configs: Sequence[str], lead_times: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """Long-format timeseries for one basin and ``configs`` (Baseline returned as ``q_base``).

        Schema matches the legacy loader: Valid Date, Basin ID, Config ID, Lead Time (Days),
        q_obs, q_da, q_base. Results are LRU-cached per (config, basin).
        """
        b = self.resolve_basin(basin)
        if b is None:
            return pd.DataFrame()
        base = self._cfg_basin_frame(BASELINE_CFG, b) if BASELINE_CFG in self.loc else pd.DataFrame()
        frames = []
        for cfg in dict.fromkeys(configs):
            rc = self.resolve_config(cfg)
            if rc is None or rc == BASELINE_CFG:
                continue
            f = self._cfg_basin_frame(rc, b)
            if not f.empty:
                frames.append(f.assign(**{"Config ID": cfg}))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if not base.empty:
            out = out.merge(base[["Valid Date", "Lead Time (Days)", "q_da"]].rename(columns={"q_da": "q_base"}),
                            on=["Valid Date", "Lead Time (Days)"], how="left")
        else:
            out["q_base"] = np.nan
        if lead_times is not None:
            out = out[out["Lead Time (Days)"].isin(list(lead_times))]
        return out.reset_index(drop=True)

    def _cfg_basin_frame(self, cfg: str, basin: str) -> pd.DataFrame:
        key = (cfg, basin)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return self._lru[key]
        loc = self.loc.get(self._source_cfg(cfg), {}).get(basin)
        if loc is None:
            return pd.DataFrame()
        store_rel, row = loc
        m = self.store_meta[store_rel]
        sim, obs = self._read_row_cached(store_rel, row)
        if parse_ar1_config(cfg) is not None:
            rho = self._rho_for(cfg, basin)
            if rho is None:
                return pd.DataFrame()
            sim = ar1_from_baseline(sim, obs, m["time_step"], rho)
        f = block_to_frame([basin], np.array(m["dates"], dtype="datetime64[ns]"), m["time_step"], sim, obs, cfg)
        with self._lock:
            self._lru[key] = f
            if len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
        return f

    def load_many(self, basins: Iterable[str], cfg: str, lead_times: Optional[Iterable[int]] = None,
                  want_sim: bool = True, want_obs: bool = True, sim_col: str = "q_da") -> pd.DataFrame:
        """Vectorised read of many basins for one config (parallel across shard stores)."""
        rc = self.resolve_config(cfg)
        if rc is None:
            return pd.DataFrame()
        is_ar1 = parse_ar1_config(rc) is not None
        src = self._source_cfg(rc)
        by_store: Dict[str, List[Tuple[int, str]]] = {}
        for b in basins:
            rb = self.resolve_basin(b)
            loc = self.loc[src].get(rb) if rb else None
            if loc and (not is_ar1 or self._rho_for(rc, rb) is not None):
                by_store.setdefault(loc[0], []).append((loc[1], rb))

        def _one(item):
            store_rel, pairs = item
            pairs.sort()
            m = self.store_meta[store_rel]
            need_obs = want_obs or (is_ar1 and want_sim)
            sim, obs = self._read_rows(store_rel, [p[0] for p in pairs], want_sim, need_obs)
            if is_ar1 and want_sim:
                rho = np.array([self._rho_for(rc, p[1]) for p in pairs], dtype=np.float64)
                sim = ar1_from_baseline(sim, obs, m["time_step"], rho)
            return block_to_frame([p[1] for p in pairs], np.array(m["dates"], dtype="datetime64[ns]"),
                                  m["time_step"], sim, obs if want_obs else None, None, lead_times, sim_col=sim_col)

        with ThreadPoolExecutor(self.num_workers) as ex:
            frames = [f for f in ex.map(_one, by_store.items()) if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _cached_parquet(self, name: Optional[str], builder) -> pd.DataFrame:
        if name is None:  # Uncacheable (e.g. depends on the in-memory per-basin rho map).
            return builder()
        path = self.cache_dir / name
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass
        df = builder()
        if not df.empty:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
            except Exception as e:
                print(f"[LAZY TS] Could not cache {name} ({e}).")
        return df

    def obs_frame(self, lead_time: int = 1) -> pd.DataFrame:
        """Observed streamflow only (all basins, one lead): Valid Date, Basin ID, Lead Time (Days), q_obs."""
        src = BASELINE_CFG if BASELINE_CFG in self.loc else self.configs[0]

        def _build():
            if self.verbose:
                print(f"[LAZY TS] Reading observed streamflow for {len(self.loc[src]):,} basins (one-time, cached)...")
            return self.load_many(self.loc[src].keys(), src, lead_times=[lead_time], want_sim=False)

        return self._cached_parquet(f"obs_lead{lead_time}.parquet", _build)

    def lead_frame(self, cfg: str, lead_time: int = 1) -> pd.DataFrame:
        """Baseline + ``cfg`` at one lead for all basins (cached): Valid Date, Basin ID, Config ID, Lead, q_obs, q_da, q_base."""
        rc = self.resolve_config(cfg)
        if rc is None:
            return pd.DataFrame()
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rc)

        def _build():
            if self.verbose:
                print(f"[LAZY TS] Reading lead-{lead_time} series for {rc} + Baseline across all basins (one-time, cached)...")
            da = self.load_many(self.basins_for(rc), rc, lead_times=[lead_time])
            if da.empty:
                return da
            if BASELINE_CFG in self.loc:
                base = self.load_many(da["Basin ID"].unique(), BASELINE_CFG, lead_times=[lead_time],
                                      want_obs=False, sim_col="q_base")
                da = da.merge(base, on=["Valid Date", "Basin ID", "Lead Time (Days)"], how="left")
            else:
                da["q_base"] = np.nan
            return da

        name = None if parse_ar1_config(rc) == PER_BASIN_RHO else f"lead{lead_time}_{SAMPLE_REDUCTION}_{safe}.parquet"
        df = self._cached_parquet(name, _build)
        if not df.empty:
            df["Config ID"] = cfg
        return df
