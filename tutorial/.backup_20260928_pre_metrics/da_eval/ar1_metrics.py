"""Offline AR(1) error-correction benchmark metrics built from a staged Baseline ``test_results.zarr``.

Writes, next to the DA configs of a staged experiment, the same metric folders the notebook already
understands (``AR1_postprocess_rho<x>`` and ``AR1_Per_Basin_Best_Rho``)::

    <data_dir>/AR1_postprocess_rho0.5/shard_NN/test/model_epochNNN/test_metrics.csv
    <data_dir>/AR1_Per_Basin_Best_Rho/shard_NN/test/model_epochNNN/test_metrics.csv

Columns: ``basin, NSE_lead{L}, Base_NSE_lead{L} (L=1..7), NSE_mean_lead1_7, Base_NSE_mean_lead1_7``.
The correction is ``lazy_timeseries.ar1_from_baseline`` (``q = max(0, q_base(T,L) + rho**L * e_T)``) on the
CMAL mixture mean (sample 0), so metrics match the hydrographs drawn on the fly. The per-basin variant picks
the fixed rho with the best lead-1 NSE for each basin.
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from da_eval.lazy_timeseries import ar1_config_name, ar1_from_baseline

DEFAULT_RHOS = (0.1, 0.3, 0.5, 0.65, 0.75, 0.85, 0.92)
PER_BASIN_DIR = "AR1_Per_Basin_Best_Rho"


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    m = ~np.isnan(obs) & ~np.isnan(sim)
    if m.sum() < 2:
        return np.nan
    o, s = obs[m], sim[m]
    den = np.sum((o - o.mean()) ** 2)
    return float(1.0 - np.sum((o - s) ** 2) / den) if den > 0 else np.nan


def _basin_rows(sim: np.ndarray, obs: np.ndarray, ts: Sequence[int], rhos: Sequence[float]):
    """sim/obs: (D, T) for one basin. Returns {rho: [NSE_lead1..7]} and the baseline [NSE_lead1..7]."""
    ts = list(ts)
    leads = [(ts.index(L), L) for L in range(1, 8) if L in ts]
    base = [_nse(obs[:, k], sim[:, k]) for k, _ in leads]
    out = {}
    for rho in rhos:
        corr = ar1_from_baseline(sim[None], obs[None], ts, rho)[0]
        out[rho] = [_nse(obs[:, k], corr[:, k]) for k, _ in leads]
    return out, base, [L for _, L in leads]


def _frame(basins, vals, base, leads):
    rows = []
    for b, v, bb in zip(basins, vals, base):
        r = {"basin": b}
        for L, x, y in zip(leads, v, bb):
            r[f"NSE_lead{L}"] = x
            r[f"Base_NSE_lead{L}"] = y
        r[f"NSE_mean_lead1_{leads[-1]}"] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        r[f"Base_NSE_mean_lead1_{leads[-1]}"] = np.nanmean(bb) if np.isfinite(bb).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def build_ar1_metrics(data_dir: str, rhos: Sequence[float] = DEFAULT_RHOS, baseline_cfg: str = "E_baseline_0da",
                      overwrite: bool = False, workers: int = 16) -> Optional[str]:
    """Writes AR(1) metric CSVs for every Baseline shard in ``data_dir``. Returns a short summary."""
    import zarr

    stores = sorted(glob.glob(os.path.join(data_dir, baseline_cfg, "shard_*", "test", "model_epoch*", "test_results*.zarr")))
    if not stores:
        print(f"[AR1] No Baseline test_results*.zarr under {data_dir}/{baseline_cfg}; stage with sync_timeseries=True.")
        return None
    n_done = 0
    marker = os.path.join(data_dir, PER_BASIN_DIR, ".generated_by_ar1_metrics")
    if not overwrite and os.path.isdir(os.path.join(data_dir, PER_BASIN_DIR)) and not os.path.exists(marker):
        return "AR1 metrics already present (externally generated); skipped"
    for store in stores:
        rel = os.path.relpath(os.path.dirname(store), os.path.join(data_dir, baseline_cfg))  # shard_NN/test/model_epochNNN
        targets = [os.path.join(data_dir, ar1_config_name(r), rel, "test_metrics.csv") for r in rhos]
        targets.append(os.path.join(data_dir, PER_BASIN_DIR, rel, "test_metrics.csv"))
        if not overwrite and all(os.path.exists(t) for t in targets):
            continue
        g = zarr.open_group(store, mode="r")
        basins = [str(b) for b in g["basin"][:]]
        ts = [int(t) for t in g["time_step"][:]]
        sim_a, obs_a = g["streamflow_sim"], g["streamflow_obs"]

        def _one(i):
            sim = np.asarray(sim_a[i, 0, :, :, 0], dtype=np.float64)  # sample 0 = mixture mean
            obs = np.asarray(obs_a[i, 0], dtype=np.float64)
            return _basin_rows(sim, obs, ts, rhos)

        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(_one, range(len(basins))))
        leads = res[0][2]
        base = [r[1] for r in res]
        for rho, tgt in zip(rhos, targets):
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            _frame(basins, [r[0][rho] for r in res], base, leads).to_csv(tgt, index=False)
        # Per-basin best rho (by lead-1 NSE).
        best = []
        for r in res:
            l1 = {rho: v[0] for rho, v in r[0].items() if np.isfinite(v[0])}
            best.append(r[0][max(l1, key=l1.get)] if l1 else [np.nan] * len(leads))
        os.makedirs(os.path.dirname(targets[-1]), exist_ok=True)
        _frame(basins, best, base, leads).to_csv(targets[-1], index=False)
        open(marker, "w").close()
        n_done += 1
        print(f"[AR1] {rel}: {len(basins)} basins -> {len(rhos)} fixed-rho + per-basin CSVs", flush=True)
    return f"{n_done} shard(s) written, {len(stores) - n_done} already present"
