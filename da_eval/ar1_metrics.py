"""Offline PP error-correction benchmark metrics built from a staged Baseline ``test_results.zarr``.

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


AR1_METRIC_NAMES = ("NSE", "KGE", "Pearson-r", "Alpha-NSE", "Beta-KGE", "Beta-NSE")


def _all_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Computes NSE, KGE, Pearson-r, Alpha-NSE, Beta-KGE, Beta-NSE for 1D arrays (matching tester.py)."""
    m = np.isfinite(obs) & np.isfinite(sim)
    n = int(m.sum())
    if n < 2:
        return {k: np.nan for k in AR1_METRIC_NAMES}
    o, s = obs[m], sim[m]
    mu_o, mu_s = float(o.mean()), float(s.mean())
    do, ds = o - mu_o, s - mu_s
    sst = float(np.dot(do, do))
    sse = float(np.dot(o - s, o - s))
    std_o = np.sqrt(sst / n)
    std_s = np.sqrt(float(np.dot(ds, ds)) / n)
    nse = (1.0 - sse / sst) if sst > 0 else np.nan
    r = float(np.dot(do, ds) / (n * std_o * std_s)) if (std_o > 0 and std_s > 0) else np.nan
    alpha = float(std_s / std_o) if std_o > 0 else np.nan
    beta_kge = float(mu_s / mu_o) if abs(mu_o) > 0 else np.nan
    beta_nse = float((mu_s - mu_o) / std_o) if std_o > 0 else np.nan
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta_kge - 1.0) ** 2)) if (
        np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta_kge)
    ) else np.nan
    return {
        "NSE": nse,
        "KGE": kge,
        "Pearson-r": r,
        "Alpha-NSE": alpha,
        "Beta-KGE": beta_kge,
        "Beta-NSE": beta_nse,
    }


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return _all_metrics(obs, sim)["NSE"]


def _basin_rows(sim: np.ndarray, obs: np.ndarray, ts: Sequence[int], rhos: Sequence[float]):
    """sim/obs: (D, T) for one basin. Returns {rho: {metric: [lead1..7]}} and baseline {metric: [lead1..7]}."""
    ts = list(ts)
    leads = [(ts.index(L), L) for L in range(1, 8) if L in ts]
    base_per_lead = [_all_metrics(obs[:, k], sim[:, k]) for k, _ in leads]
    base = {m: [d[m] for d in base_per_lead] for m in AR1_METRIC_NAMES}
    out = {}
    for rho in rhos:
        corr = ar1_from_baseline(sim[None], obs[None], ts, rho)[0]
        m_per_lead = [_all_metrics(obs[:, k], corr[:, k]) for k, _ in leads]
        out[rho] = {m: [d[m] for d in m_per_lead] for m in AR1_METRIC_NAMES}
    return out, base, [L for _, L in leads]


def _frame(basins, vals, base, leads):
    rows = []
    for b, v, bb in zip(basins, vals, base):
        r = {"basin": b}
        for m in AR1_METRIC_NAMES:
            mv = v[m]
            mb = bb[m]
            for L, x, y in zip(leads, mv, mb):
                r[f"{m}_lead{L}"] = x
                r[f"Base_{m}_lead{L}"] = y
            r[f"{m}_mean_lead1_{leads[-1]}"] = float(np.nanmean(mv)) if np.isfinite(mv).any() else np.nan
            r[f"Base_{m}_mean_lead1_{leads[-1]}"] = float(np.nanmean(mb)) if np.isfinite(mb).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _csv_has_full_metrics(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        cols = set(pd.read_csv(path, nrows=1).columns)
        return {"NSE_lead1", "KGE_lead1", "Pearson-r_lead1", "Alpha-NSE_lead1", "Beta-KGE_lead1", "Beta-NSE_lead1"}.issubset(cols)
    except Exception:
        return False


def build_ar1_metrics(data_dir: str, rhos: Sequence[float] = DEFAULT_RHOS, baseline_cfg: str = "E_baseline_0da",
                      overwrite: bool = False, workers: int = 16) -> Optional[str]:
    """Writes PP metric CSVs for every Baseline shard in ``data_dir``. Returns a short summary."""
    import zarr

    stores = sorted(glob.glob(os.path.join(data_dir, baseline_cfg, "shard_*", "test", "model_epoch*", "test_results*.zarr")))
    if not stores:
        stores = sorted(glob.glob(os.path.join(data_dir, baseline_cfg, "test", "model_epoch*", "test_results*.zarr")))
    if not stores:
        print(f"[PP] No Baseline test_results*.zarr under {data_dir}/{baseline_cfg}; stage with sync_timeseries=True.")
        return None
    n_done = 0
    for store in stores:
        rel = os.path.relpath(os.path.dirname(store), os.path.join(data_dir, baseline_cfg))  # shard_NN/test/model_epochNNN
        targets = [os.path.join(data_dir, ar1_config_name(r), rel, "test_metrics.csv") for r in rhos]
        targets.append(os.path.join(data_dir, PER_BASIN_DIR, rel, "test_metrics.csv"))
        if not overwrite and all(_csv_has_full_metrics(t) for t in targets):
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
            l1 = {rho: v["NSE"][0] for rho, v in r[0].items() if np.isfinite(v["NSE"][0])}
            best.append(r[0][max(l1, key=l1.get)] if l1 else {m: [np.nan] * len(leads) for m in AR1_METRIC_NAMES})
        os.makedirs(os.path.dirname(targets[-1]), exist_ok=True)
        _frame(basins, best, base, leads).to_csv(targets[-1], index=False)
        marker = os.path.join(data_dir, PER_BASIN_DIR, ".generated_by_ar1_metrics")
        open(marker, "w").close()
        n_done += 1
        print(f"[PP] {rel}: {len(basins)} basins -> {len(rhos)} fixed-rho + per-basin CSVs (full NSE/KGE/r/alpha/beta)", flush=True)
    return f"{n_done} shard(s) written, {len(stores) - n_done} already present"
