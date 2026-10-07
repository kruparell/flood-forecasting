"""da_interactive_helpers.py

Modular helper routines for OpenHydroNets Interactive Data Assimilation notebooks.
Provides self-contained data loading, CMAL inference, 4D-Var optimization, and
publication-grade multi-panel diagnostic plotting for:
  1. Single-Initialization Event Inspection (2-panel hydrograph comparison)
  2. N-Day Lead Time Continuous Rolling Forecast Evaluation
  3. Precipitation Forcing 4D-Var Diagnostics (Decoupled Wp / Wq)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr

from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig

try:
    from googlehydrology.modelzoo.head import ensure_y_hat
except (ImportError, AttributeError):
    try:
        import importlib
        import googlehydrology.modelzoo.head as _h
        importlib.reload(_h)
        from googlehydrology.modelzoo.head import ensure_y_hat
    except Exception:
        from googlehydrology.utils.cmal_deterministic import generate_predictions
        def ensure_y_hat(pred: Any, use_median: bool = True) -> Dict[str, Any]:
            if not isinstance(pred, dict):
                return {'y_hat': pred}
            res = dict(pred)
            if 'mu' in res and 'pi' in res and 'b' in res and 'tau' in res:
                if use_median:
                    cmal_summary = generate_predictions(res['mu'], res['b'], res['tau'], res['pi'])
                    res['y_hat'] = cmal_summary[..., 5:6]
                else:
                    b_clamp = torch.clamp(res['b'].float(), min=1e-5)
                    tau_clamp = torch.clamp(res['tau'].float(), min=1e-6, max=1.0 - 1e-6)
                    means = res['mu'].float() + b_clamp * (1 - 2 * tau_clamp) / (tau_clamp * (1 - tau_clamp))
                    pi_norm = res['pi'].float() / torch.sum(res['pi'].float(), dim=-1, keepdim=True)
                    res['y_hat'] = torch.sum(pi_norm * means, dim=-1, keepdim=True)
            elif 'y_hat' not in res and 'mu' in res:
                res['y_hat'] = res['mu'] if res['mu'].ndim == 3 else res['mu'].unsqueeze(-1)
            return res

# Ensure the in-memory head module has ensure_y_hat attached for any downstream callers
try:
    import googlehydrology.modelzoo.head as _h
    if not hasattr(_h, 'ensure_y_hat'):
        _h.ensure_y_hat = ensure_y_hat
except Exception:
    pass

try:
    from tutorial.notebooks.helpers.multimet_helpers import prepare_multimet_batch
except ImportError:
    try:
        from notebooks.helpers.multimet_helpers import prepare_multimet_batch
    except ImportError:
        try:
            from notebooks.multimet_helpers import prepare_multimet_batch
        except ImportError:
            from multimet_helpers import prepare_multimet_batch

# Caches for static attributes and streamflow datasets
_DS_ATTRS_CACHE: Optional[xr.Dataset] = None
_DS_STREAMFLOW_CACHE: Optional[xr.Dataset] = None
_BASIN_STREAMFLOW_CACHE: Dict[str, xr.Dataset] = {}
_BASIN_ATTRS_CACHE: Dict[str, pd.DataFrame] = {}


# ==============================================================================
# 1. Catchment Data Resolution & Metric Utilities
# ==============================================================================

def resolve_basin_streamflow_and_attrs(
    basin_id: str,
    caravan_dir: Path,
    date_slice: slice = slice('2014-01-01', '2021-12-31'),
) -> Tuple[pd.DataFrame, xr.Dataset]:
    """Ensures static attributes and streamflow timeseries are loaded for basin_id without KeyError."""
    global _DS_ATTRS_CACHE, _DS_STREAMFLOW_CACHE

    basin_id = str(basin_id).strip()
    if basin_id not in _BASIN_ATTRS_CACHE:
        if _DS_ATTRS_CACHE is None:
            _DS_ATTRS_CACHE = xr.open_zarr(caravan_dir / 'attributes.zarr', consolidated=True)
        _BASIN_ATTRS_CACHE[basin_id] = _DS_ATTRS_CACHE.sel(basin=[basin_id]).to_dataframe()

    if basin_id not in _BASIN_STREAMFLOW_CACHE:
        if _DS_STREAMFLOW_CACHE is None:
            _DS_STREAMFLOW_CACHE = xr.open_zarr(caravan_dir / 'streamflow.zarr', consolidated=True)
        _BASIN_STREAMFLOW_CACHE[basin_id] = _DS_STREAMFLOW_CACHE.sel(
            basin=basin_id, date=date_slice
        ).load()

    return _BASIN_ATTRS_CACHE[basin_id], _BASIN_STREAMFLOW_CACHE[basin_id]


# ---- Basin catalogue & search (any Caravan gauge, not just presets) ----------
_BASIN_CATALOGUE: Optional[pd.DataFrame] = None
_CATALOGUE_COLS = ['gauge_name', 'country', 'area', 'aridity', 'frac_snow', 'gauge_lat', 'gauge_lon']


def load_basin_catalogue(caravan_dir: Path) -> pd.DataFrame:
    """Returns a cached table of every gauge in ``attributes.zarr`` (index = basin ID).

    Columns (where available): ``gauge_name``, ``country``, ``area`` (km²),
    ``aridity``, ``frac_snow``, ``gauge_lat``, ``gauge_lon``. ~1 s on first call.
    """
    global _DS_ATTRS_CACHE, _BASIN_CATALOGUE
    if _BASIN_CATALOGUE is None:
        if _DS_ATTRS_CACHE is None:
            _DS_ATTRS_CACHE = xr.open_zarr(caravan_dir / 'attributes.zarr', consolidated=True)
        cols = [c for c in _CATALOGUE_COLS if c in _DS_ATTRS_CACHE.data_vars]
        df = _DS_ATTRS_CACHE[cols].to_dataframe()
        df.index = df.index.astype(str)
        df['gauge_name'] = df.get('gauge_name', pd.Series('', index=df.index)).fillna('').astype(str)
        _BASIN_CATALOGUE = df.sort_index()
    return _BASIN_CATALOGUE


def search_basin_catalogue(
    query: str,
    caravan_dir: Path,
    limit: int = 300,
    country: Optional[str] = None,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Case-study-style gauge search over the full Caravan catalogue.

    ``query`` matches basin ID or gauge name (case-insensitive substring);
    comma-separated terms are OR-ed, e.g. ``"camelscl, 09444500, danube"``.
    ``country`` is a case-insensitive substring of the attribute (the column
    mixes full names and ISO codes, e.g. ``"Chile"``/``"CL"``, ``"United States"``/``"US"``).
    ``dataset`` restricts to a basin-ID prefix such as ``"camelscl"`` or ``"GRDC"``.
    An empty query returns the first ``limit`` gauges.
    """
    cat = load_basin_catalogue(caravan_dir)
    if dataset:
        cat = cat[cat.index.str.lower().str.startswith(str(dataset).lower())]
    if country:
        cat = cat[cat['country'].astype(str).str.lower().str.contains(str(country).lower(), regex=False)]
    terms = [t.strip().lower() for t in str(query or '').split(',') if t.strip()]
    if terms:
        hay = (cat.index.to_series().str.lower() + ' ' + cat['gauge_name'].str.lower())
        hit = hay.apply(lambda s: any(t in s for t in terms))
        cat = cat[hit.values]
    return cat.head(int(limit))


def format_basin_option(basin_id: str, row: Optional[pd.Series] = None) -> str:
    """``"<id> — <name> [<country>] (<area> km²)"`` label for dropdowns."""
    if row is None:
        return str(basin_id)
    name = str(row.get('gauge_name', '') or '').strip()
    ctry = str(row.get('country', '') or '').strip()
    area = row.get('area', np.nan)
    area_s = f"{float(area):,.0f} km²" if pd.notna(area) else ''
    bits = [str(basin_id)]
    if name:
        bits.append(f"— {name}")
    if ctry:
        bits.append(f"[{ctry}]")
    if area_s:
        bits.append(f"({area_s})")
    return ' '.join(bits)


def observation_coverage(b_stream: xr.Dataset, year: Optional[int] = None) -> Dict[str, Any]:
    """Valid-observation summary for a basin (optionally restricted to one calendar year)."""
    s = b_stream['streamflow'].to_series()
    if year is not None:
        s = s[str(int(year))]
    valid = s.dropna()
    return {
        'n_valid': int(len(valid)),
        'n_total': int(len(s)),
        'first_valid': valid.index.min() if len(valid) else None,
        'last_valid': valid.index.max() if len(valid) else None,
    }


def suggest_init_date(
    b_stream: xr.Dataset,
    year: int = 2017,
    peak_lead: int = 2,
    min_window_valid_frac: float = 0.5,
    window: int = 90,
) -> Optional[datetime.date]:
    """Suggests a forecast issue date ``t0`` for a case study.

    Picks the largest observed daily flow in ``year`` and returns
    ``t0 = peak_date - peak_lead`` so the peak falls inside the 7-day forecast
    horizon. Candidates whose preceding ``window`` days have fewer than
    ``min_window_valid_frac`` valid observations are skipped (next-largest peak
    is tried). Returns ``None`` if the year has no usable observations.
    """
    s = b_stream['streamflow'].to_series()
    s_year = s[str(int(year))].dropna()
    if s_year.empty:
        return None
    for peak_date in s_year.sort_values(ascending=False).index[:30]:
        t0 = peak_date - pd.Timedelta(days=int(peak_lead))
        win = s[t0 - pd.Timedelta(days=int(window) - 1): t0]
        if len(win) and win.notna().mean() >= float(min_window_valid_frac) and (t0 + pd.Timedelta(days=7)) <= s.index.max():
            return t0.date()
    return None


def calc_hydro_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Calculates NSE and KGE with variance-collapse protections against low-flow artifacts."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[valid], sim[valid]
    if len(o) < 3 or np.var(o) < 1e-6:
        return {'NSE': np.nan, 'KGE': np.nan, 'RMSE': np.nan, 'Pearson-r': np.nan}
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1.0 - (np.sum((o - s) ** 2) / denom)) if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    std_o, std_s = float(np.std(o)), float(np.std(s))
    mean_o, mean_s = float(np.mean(o)), float(np.mean(s))
    if std_o > 1e-6 and std_s > 1e-6 and mean_o > 1e-6:
        r = float(np.corrcoef(o, s)[0, 1])
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (std_s / std_o - 1.0) ** 2 + (mean_s / mean_o - 1.0) ** 2))
    else:
        kge = nse
        r = np.nan
    return {'NSE': nse, 'KGE': kge, 'RMSE': rmse, 'Pearson-r': r}


def extract_deterministic_y(out: Dict[str, Any], q_mean: float, q_std: float) -> np.ndarray:
    """Extracts deterministic streamflow (mm/day) from CMAL mixture outputs."""
    out_det = ensure_y_hat(dict(out), use_median=True)
    raw = out_det['y_hat'][0, :, 0].detach().cpu().numpy()
    return np.maximum(0.0, raw * q_std + q_mean)


def fetch_multimet_batch(
    basin_id: str,
    issue_date: str,
    model_cfg: Any,
    scaler: xr.Dataset,
    caravan_dir: Path,
    multimet_dir: Path,
    device: torch.device = torch.device('cpu'),
) -> Tuple[Dict[str, Any], xr.Dataset]:
    """Fetches a single 365-day MultiMet batch aligned with Caravans streamflow."""
    b_attrs, b_stream = resolve_basin_streamflow_and_attrs(basin_id, caravan_dir)
    batch = prepare_multimet_batch(
        mode='multimet_0_and_1_to_7',
        basin_id=basin_id,
        issue_date=issue_date,
        cfg=model_cfg,
        scaler=scaler,
        caravan_attrs=b_attrs,
        ds_caravan=b_stream,
        multimet_dir=multimet_dir,
        forecast_product='HRES',
        hindcast_window_days=358,
        forecast_lead_days=7,
    )
    if device != torch.device('cpu'):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch, b_stream


# ==============================================================================
# 2. Tier 1: Single-Initialization 2-Panel Hydrograph Diagnostic
# ==============================================================================

def normalize_da_target(target: Union[str, Sequence[str]]) -> List[str]:
    """Normalizes target aliases (e.g. embedded_both, embedded_all, c_both) into component lists."""
    if isinstance(target, (list, tuple)):
        return list(target)
    target_str = str(target).strip()
    if target_str == 'embedded_both':
        return ['static_embedding', 'hindcast_embedding']
    elif target_str == 'embedded_both_dynamic':
        return ['hindcast_embedding', 'forecast_embedding']
    elif target_str == 'embedded_all':
        return ['static_embedding', 'hindcast_embedding', 'forecast_embedding']
    elif target_str == 'c_both':
        return ['c_0_hindcast', 'c_0_forecast']
    return [target_str]


def _build_assimilation_config(
    model: nn.Module,
    target: Union[str, Sequence[str]],
    window: int,
    lr: float,
    epochs: int,
    bg: float,
    loss: str = 'MSE',
    state_anchor: str = 'window_start',
    tol: Optional[float] = None,
) -> AssimilationConfig:
    """Assembles an `AssimilationConfig` for the notebook's 365-day batches.

    `loss` selects the data-misfit term minimized by the 4D-Var solver:
      * `'MSE'`  - deterministic squared error on the expectation of the head.
      * `'CMAL'` - the full Countable Mixture of Asymmetric Laplacians negative
        log-likelihood, i.e. the same probabilistic objective the foundation
        model was trained with. This requires a mixture head, so the number of
        mixture components must match the loaded model's `n_distributions`.

    `state_anchor` only affects recurrent-state targets (`c_0_*`, `h_0_*`):
      * `'window_start'` - the control variable is the catchment state entering
        the assimilation window, which is the standard 4D-Var state update.
      * `'sequence_start'` - the control variable is the state at day 0 of the
        365-day input sequence, i.e. up to a year before the window. Kept for
        reproducing earlier sweeps only.
    """
    resolved_targets = normalize_da_target(target)
    cfg_dict: Dict[str, Any] = {
        'seq_length': 365,
        'assimilation_lead_time': 7,
        'assimilation_window': int(window),
        'history': 1,
        'loss': str(loss),
        'optimizer': 'Adam',
        'learning_rate': {0: float(lr)},
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
        'predict_n_hindcast': 1,
        'assimilation_targets': resolved_targets,
        'regularization_weight': float(bg),
        'static_embedding_regularization_weight': float(bg),
        'hindcast_embedding_regularization_weight': float(bg),
        'forecast_embedding_regularization_weight': float(bg),
        'recurrent_state_regularization_weight': float(bg),
        'epochs': int(epochs),
        'learning_rate_epoch_drop': 100,
        'state_anchor': str(state_anchor),
        'dev_mode': True,
    }
    if tol is not None and float(tol) > 0:
        cfg_dict['early_stopping_tolerance'] = float(tol)
    if str(loss).lower().replace('loss', '') == 'cmal':
        # The CMAL log-likelihood slices the head output into `n_distributions`
        # blocks per target, so the DA config has to agree with the model.
        cfg_dict['n_distributions'] = int(getattr(model.cfg, 'n_distributions', 3))
    return AssimilationConfig(cfg_dict)


def run_single_init_da(
    model: nn.Module,
    batch: Dict[str, Any],
    target: Union[str, Sequence[str]],
    window: int = 30,
    lr: float = 0.05,
    epochs: int = 50,
    bg: float = 0.1,
    q_mean: float = 0.0,
    q_std: float = 1.0,
    loss: str = 'MSE',
    state_anchor: str = 'window_start',
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs baseline and 4D-Var assimilation for a single initialization.

    The `Assimilation` engine optimizes the requested component over the
    historical window, splices it back into the full-length sequence, and runs a
    single clean forward pass over the whole sequence, so no manual chunk
    stitching (and no cold-start LSTM state reset) is needed here.
    """
    target_seq_len = batch['y'].shape[1] if 'y' in batch and hasattr(batch['y'], 'shape') else None
    with torch.no_grad():
        base_out = model(batch)
        q_base = extract_deterministic_y(base_out, q_mean, q_std)
        if target_seq_len is not None and len(q_base) > target_seq_len:
            q_base = q_base[-target_seq_len:]

    assim_cfg = _build_assimilation_config(
        model=model, target=target, window=window, lr=lr, epochs=epochs, bg=bg,
        loss=loss, state_anchor=state_anchor, tol=tol,
    )
    da_eng = Assimilation(assim_cfg)
    da_out = da_eng.assimilate(model, batch, verbose=False)
    q_da = extract_deterministic_y(da_out, q_mean, q_std)
    if target_seq_len is not None and len(q_da) > target_seq_len:
        q_da = q_da[-target_seq_len:]

    return q_base, q_da


def plot_single_init_da_hydrographs(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    q_base: np.ndarray,
    q_da: np.ndarray,
    basin_id: str,
    issue_date_str: str,
    target_name: str,
    window: int,
    w_start_idx: int,
    t0_idx: int = 357,
    da_end_idx: int = 358,
    lead_time: int = 7,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Renders the publication-grade 2-panel hydrograph comparison figure."""
    m_base_full = calc_hydro_metrics(obs, q_base)
    m_da_full = calc_hydro_metrics(obs, q_da)

    # Bottom Panel: Zoom focused on the actual event horizon around t0
    # For large assimilation windows (e.g. 90-365d), frame the last 35 days before t0
    # plus the 7-day forecast lead so the event details are clearly visible.
    zoom_lookback_days = min(int(window), 35)
    zoom_start = max(0, t0_idx - zoom_lookback_days)
    zoom_end = min(len(dates), da_end_idx + lead_time)

    z_dates = dates[zoom_start:zoom_end]
    z_obs = obs[zoom_start:zoom_end]
    z_base = q_base[zoom_start:zoom_end]
    z_da = q_da[zoom_start:zoom_end]

    m_base_zoom = calc_hydro_metrics(z_obs, z_base)
    m_da_zoom = calc_hydro_metrics(z_obs, z_da)

    lead_sl = slice(da_end_idx, min(len(dates), da_end_idx + lead_time))
    m_base_lead = calc_hydro_metrics(obs[lead_sl], q_base[lead_sl])
    m_da_lead = calc_hydro_metrics(obs[lead_sl], q_da[lead_sl])

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(15, 9.5), dpi=120, gridspec_kw={'height_ratios': [1.3, 1.2]}
    )

    # -------------------------------------------------------------------------
    # 1. Top Panel: Full Evaluation Period (Continuous 365-Day Hydrograph)
    # -------------------------------------------------------------------------
    lbl_obs = r"Observed ($Q_{\mathrm{obs}}$)"
    lbl_base_full = (
        f"Pretrained Base (No DA) [NSE={m_base_full['NSE']:.3f}, KGE={m_base_full['KGE']:.3f}]"
        if np.isfinite(m_base_full['NSE']) else "Pretrained Base (No DA)"
    )
    lbl_da_full = (
        f"DA Assimilated ({target_name}) [NSE={m_da_full['NSE']:.3f}, KGE={m_da_full['KGE']:.3f}]"
        if np.isfinite(m_da_full['NSE']) else f"DA Assimilated ({target_name})"
    )

    ax_top.plot(dates, obs, color='black', linestyle='-', marker='.', markersize=4,
                linewidth=1.8, alpha=0.85, label=lbl_obs, zorder=5)
    ax_top.plot(dates, q_base, color='#2563eb', linestyle='--', linewidth=2.0,
                label=lbl_base_full, zorder=3)
    ax_top.plot(dates, q_da, color='#dc2626', linestyle='-', linewidth=2.2,
                label=lbl_da_full, zorder=4)

    # Active assimilation window shading & boundary lines
    ax_top.axvspan(dates[w_start_idx], dates[t0_idx], color='#fed7aa', alpha=0.35,
                   label=f'DA Assimilation Window ({window}d)')
    ax_top.axvline(dates[w_start_idx], color='#d97706', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_top.axvline(dates[t0_idx], color='#dc2626', linestyle='--', linewidth=1.5,
                   alpha=0.9, label=f'Forecast Init $t_0$ ({issue_date_str})')

    ax_top.set_title(f"{basin_id} - Full Period Hydrograph Comparison", fontsize=13, fontweight='bold', pad=10)
    ax_top.set_ylabel(r"Streamflow ($\mathrm{mm/day}$)", fontsize=11, fontweight='bold')
    ax_top.set_xlim(dates[0], dates[-1])
    ax_top.set_ylim(bottom=0.0)
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.9)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # -------------------------------------------------------------------------
    # 2. Bottom Panel: Zoomed Event Horizon Around t0
    # -------------------------------------------------------------------------
    lbl_base_zoom = (
        f"Pretrained Base [Window NSE={m_base_zoom['NSE']:.3f} | Lead 1-7 NSE={m_base_lead['NSE']:.3f}, KGE={m_base_lead['KGE']:.3f}]"
        if np.isfinite(m_base_lead['NSE']) else "Pretrained Base (No DA)"
    )
    lbl_da_zoom = (
        f"DA Assimilated [Window NSE={m_da_zoom['NSE']:.3f} | Lead 1-7 NSE={m_da_lead['NSE']:.3f}, KGE={m_da_lead['KGE']:.3f}]"
        if np.isfinite(m_da_lead['NSE']) else "DA Assimilated"
    )

    ax_bottom.plot(z_dates, z_obs, color='black', linestyle='-', marker='o', markersize=4.5,
                   linewidth=2.0, alpha=0.85, label=lbl_obs, zorder=5)
    ax_bottom.plot(z_dates, z_base, color='#2563eb', linestyle='--', linewidth=2.0,
                   label=lbl_base_zoom, zorder=3)
    ax_bottom.plot(z_dates, z_da, color='#dc2626', linestyle='-', linewidth=2.2,
                   label=lbl_da_zoom, zorder=4)

    # Shaded spans for visible assimilation window and 7-day forecast lead
    win_vis_start = max(dates[w_start_idx], z_dates[0])
    ax_bottom.axvspan(win_vis_start, dates[t0_idx], color='#fed7aa', alpha=0.35,
                      label=f'Assimilation Window ({window}d)')
    fc_end_idx = min(len(dates) - 1, da_end_idx + lead_time - 1)
    ax_bottom.axvspan(dates[da_end_idx], dates[fc_end_idx], color='#bfdbfe', alpha=0.35,
                      label=f'Forecast Horizon (Leads 1..{lead_time})')
    ax_bottom.axvline(dates[t0_idx], color='#dc2626', linestyle='--', linewidth=2.0,
                      label=f'Forecast Init $t_0$ ({issue_date_str})')

    ax_bottom.set_title(
        f"{basin_id} - Zoomed Event & Forecast Horizon ($t_0$ = {issue_date_str})",
        fontsize=12, fontweight='bold', pad=8
    )
    ax_bottom.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax_bottom.set_ylabel(r"Streamflow ($\mathrm{mm/day}$)", fontsize=11, fontweight='bold')
    ax_bottom.set_xlim(z_dates[0], z_dates[-1])
    ax_bottom.set_ylim(bottom=0.0)
    ax_bottom.grid(True, linestyle='--', alpha=0.5)
    ax_bottom.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.9)
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.tight_layout()
    return fig, (ax_top, ax_bottom)


def ar1_postprocess(
    obs: np.ndarray,
    q_base: np.ndarray,
    rho: float = 0.65,
    t0_idx: int = 357,
    da_end_idx: int = 358,
    lead_time: int = 7,
) -> np.ndarray:
    """AR(1) error-persistence post-processing of the open-loop baseline.

    Mirrors the ``AR1_postprocess_rho<x>`` reference in the DA evaluation
    sessions: ``q_pp(T+L) = max(0, q_base(T+L) + rho**L * (q_obs(T) - q_base(T)))``
    for leads ``L = 1..lead_time``. Returns an array aligned with ``q_base`` that
    is NaN outside the forecast horizon (and NaN everywhere if ``q_obs(T)`` is
    missing).
    """
    q_pp = np.full_like(np.asarray(q_base, dtype=np.float64), np.nan)
    err_t0 = float(obs[t0_idx]) - float(q_base[t0_idx])
    if not np.isfinite(err_t0):
        return q_pp
    for L in range(1, lead_time + 1):
        i = da_end_idx + L - 1
        if i >= len(q_base):
            break
        q_pp[i] = max(0.0, float(q_base[i]) + (float(rho) ** L) * err_t0)
    return q_pp


def plot_paper_da_hydrograph(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    q_base: np.ndarray,
    q_da: np.ndarray,
    basin_id: str,
    window: int,
    w_start_idx: int,
    t0_idx: int = 357,
    da_end_idx: int = 358,
    lead_time: int = 7,
    q_pp: Optional[np.ndarray] = None,
    display_days: Optional[int] = None,
    show_obs: bool = True,
    show_baseline: bool = True,
    show_da: bool = True,
    show_pp: bool = True,
    title: Optional[str] = None,
    title_prefix: Optional[str] = None,
    metrics: Sequence[str] = ('NSE',),
    metric_decimals: int = 2,
    period_labels: bool = True,
    period_label_names: Sequence[str] = ('Warm-up', 'Assimilation window', 'Forecast'),
    period_label_y: float = 0.96,
    period_label_fontsize: float = 10.0,
    shade_periods: bool = True,
    show_boundaries: bool = True,
    show_legend: bool = True,
    legend_loc: str = 'upper left',
    legend_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    obs_marker: Optional[str] = '.',
    figsize: Tuple[float, float] = (10.0, 4.0),
    dpi: int = 150,
    date_fmt: str = '%d %b %Y',
    xlabel: Optional[str] = None,
    ylabel: str = r"Streamflow ($\mathrm{mm\,d^{-1}}$)",
    log_y: bool = False,
    fontsize: float = 11.0,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Dict[str, float]]]:
    """Renders a single-panel, publication-style DA hydrograph.

    Compared to `plot_single_init_da_hydrographs`, this gives explicit control
    over what ends up in the figure:

    * `display_days` - total number of days shown. The plot always ends
      `lead_time` days after the end of the assimilation window, so this only
      controls how far back the x-axis reaches (i.e. how much warm-up is
      visible). `None` shows the entire sequence.
    * Period labels ("Warm-up", "Assimilation window", "Forecast") are drawn
      inside the axes (centered over each shaded span), not in the legend.
    * Skill scores are computed over the **forecast horizon only** and placed
      in the title. `metrics` picks which ones (`'NSE'`, `'KGE'`, `'RMSE'`,
      `'Pearson-r'`); pass an empty tuple to omit them entirely. `title`
      overrides the whole title; `title_prefix` replaces only the basin label.
    * `legend_labels` / `colors` accept keys `'obs'`, `'base'`, `'da'`.

    Returns the figure, the axis and the forecast-horizon metrics
    (`{'base': {...}, 'da': {...}}`) so they can be quoted in captions.
    """
    n = len(dates)
    fc_end_idx = min(n - 1, da_end_idx + lead_time - 1)

    if display_days is None or int(display_days) <= 0:
        disp_start = 0
    else:
        disp_start = max(0, fc_end_idx + 1 - int(display_days))
    disp_end = fc_end_idx + 1  # exclusive

    d = dates[disp_start:disp_end]
    o = obs[disp_start:disp_end]
    qb = q_base[disp_start:disp_end]
    qd = q_da[disp_start:disp_end]

    has_pp = q_pp is not None and show_pp
    lead_sl = slice(da_end_idx, fc_end_idx + 1)
    fc_metrics = {
        'base': calc_hydro_metrics(obs[lead_sl], q_base[lead_sl]),
        'da': calc_hydro_metrics(obs[lead_sl], q_da[lead_sl]),
    }
    if has_pp:
        fc_metrics['pp'] = calc_hydro_metrics(obs[lead_sl], np.asarray(q_pp)[lead_sl])

    lbl = {'obs': 'Observed', 'base': 'No assimilation', 'da': 'With assimilation',
           'pp': 'AR(1) post-processing'}
    if legend_labels:
        lbl.update({k: v for k, v in legend_labels.items() if v})
    col = {'obs': 'black', 'base': '#2563eb', 'da': '#dc2626', 'pp': '#16a34a',
           'warmup': '#f3f4f6', 'window': '#fed7aa', 'forecast': '#bfdbfe'}
    if colors:
        col.update({k: v for k, v in colors.items() if v})

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # ---- Period shading (drawn first so lines sit on top) -------------------
    win_start_date = dates[w_start_idx]
    t0_date = dates[t0_idx]
    fc_end_date = dates[fc_end_idx]
    disp_start_date = d[0]
    has_warmup = disp_start_date < win_start_date

    if shade_periods:
        if has_warmup:
            ax.axvspan(disp_start_date, win_start_date, color=col['warmup'], alpha=0.9, zorder=0, lw=0)
        ax.axvspan(max(win_start_date, disp_start_date), t0_date, color=col['window'], alpha=0.35, zorder=0, lw=0)
        ax.axvspan(t0_date, fc_end_date, color=col['forecast'], alpha=0.35, zorder=0, lw=0)
    if show_boundaries:
        if has_warmup:
            ax.axvline(win_start_date, color='#6b7280', linestyle=':', linewidth=1.0, zorder=1)
        ax.axvline(t0_date, color='#374151', linestyle='--', linewidth=1.2, zorder=1)

    # ---- Series --------------------------------------------------------------
    if show_obs:
        ax.plot(d, o, color=col['obs'], linestyle='-', marker=obs_marker, markersize=4,
                linewidth=1.6, alpha=0.9, label=lbl['obs'], zorder=5)
    if show_baseline:
        ax.plot(d, qb, color=col['base'], linestyle='--', linewidth=1.8, label=lbl['base'], zorder=3)
    if has_pp:
        qp = np.asarray(q_pp)[disp_start:disp_end]
        ax.plot(d, qp, color=col['pp'], linestyle=':', linewidth=2.0, marker='o', markersize=3,
                label=lbl['pp'], zorder=3.5)
    if show_da:
        ax.plot(d, qd, color=col['da'], linestyle='-', linewidth=2.0, label=lbl['da'], zorder=4)

    # ---- In-plot period labels ----------------------------------------------
    if period_labels:
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        names = list(period_label_names) + [''] * (3 - len(period_label_names))

        def _mid(a, b):
            return a + (b - a) / 2

        spans = []
        if has_warmup and names[0]:
            spans.append((_mid(disp_start_date, win_start_date), names[0]))
        if names[1]:
            spans.append((_mid(max(win_start_date, disp_start_date), t0_date), names[1]))
        if names[2]:
            spans.append((_mid(t0_date, fc_end_date), names[2]))
        for x, txt in spans:
            ax.text(x, period_label_y, txt, transform=trans, ha='center', va='top',
                    fontsize=period_label_fontsize, fontweight='bold', color='#111827',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.75),
                    zorder=10)

    # ---- Title with forecast-only metrics -----------------------------------
    if title is None:
        short = {'base': lbl.get('base_short', 'no DA'), 'pp': lbl.get('pp_short', 'PP'),
                 'da': lbl.get('da_short', 'DA')}
        shown = [('base', show_baseline), ('pp', has_pp), ('da', show_da)]
        parts = []
        for m in metrics:
            vals = []
            for key, on in shown:
                if on and np.isfinite(fc_metrics.get(key, {}).get(m, np.nan)):
                    vals.append(f"{fc_metrics[key][m]:.{metric_decimals}f} ({short[key]})")
            if vals:
                parts.append(f"{m}: " + " → ".join(vals))
        head = title_prefix if title_prefix is not None else basin_id
        if parts:
            title = f"{head}  |  Forecast " + ";  ".join(parts)
        else:
            title = head
    if title:
        ax.set_title(title, fontsize=fontsize + 1, fontweight='bold', pad=8)

    # ---- Axes cosmetics ------------------------------------------------------
    ax.set_xlim(d[0], d[-1])
    if log_y:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(fontsize=fontsize - 1, loc=legend_loc, frameon=True, framealpha=0.9)

    fig.tight_layout()
    return fig, ax, fc_metrics


# ==============================================================================
# 3. Tier 2: 365-Day Continuous Rolling Forecast Evaluation
# ==============================================================================

def run_yearly_rolling_evaluation(
    model: nn.Module,
    basin_id: str,
    year: int,
    target: Union[str, Sequence[str]],
    window: int,
    lr: float,
    epochs: int,
    bg: float,
    model_cfg: Any,
    scaler: xr.Dataset,
    caravan_dir: Path,
    multimet_dir: Path,
    q_mean: float,
    q_std: float,
    device: torch.device = torch.device('cpu'),
    progress_bar: Optional[Any] = None,
    loss: str = 'MSE',
    state_anchor: str = 'window_start',
    tol: Optional[float] = None,
) -> Dict[str, Any]:
    """Executes rolling daily DA across all 365 issue dates of a calendar year."""
    b_attrs, b_stream = resolve_basin_streamflow_and_attrs(basin_id, caravan_dir)
    issue_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D').strftime('%Y-%m-%d')

    assim_cfg = _build_assimilation_config(
        model=model, target=target, window=window, lr=lr, epochs=epochs, bg=bg,
        loss=loss, state_anchor=state_anchor, tol=tol,
    )
    da_eng = Assimilation(assim_cfg)

    records_base = []
    records_da = []
    if progress_bar:
        progress_bar.max = len(issue_dates)
        progress_bar.value = 0

    for i, dt in enumerate(issue_dates):
        batch, _ = fetch_multimet_batch(
            basin_id=basin_id,
            issue_date=dt,
            model_cfg=model_cfg,
            scaler=scaler,
            caravan_dir=caravan_dir,
            multimet_dir=multimet_dir,
            device=device,
        )
        dt_obj = pd.to_datetime(dt)

        # Baseline (Lead 1..7)
        with torch.no_grad():
            q_base_7 = extract_deterministic_y(model(batch), q_mean, q_std)[-7:]

        # DA (Lead 1..7)
        q_da_7 = extract_deterministic_y(da_eng.assimilate(model, batch, verbose=False), q_mean, q_std)[-7:]

        for L in range(1, 8):
            v_date = (dt_obj + pd.Timedelta(days=L)).strftime('%Y-%m-%d')
            records_base.append({'valid_date': v_date, 'lead_time': L, 'q_sim': q_base_7[L - 1]})
            records_da.append({'valid_date': v_date, 'lead_time': L, 'q_sim': q_da_7[L - 1]})

        if progress_bar:
            progress_bar.value = i + 1

    df_base = pd.DataFrame(records_base)
    df_da = pd.DataFrame(records_da)

    summary_rows = []
    for L in range(1, 8):
        sub_b = df_base[df_base['lead_time'] == L].set_index('valid_date')
        sub_da = df_da[df_da['lead_time'] == L].set_index('valid_date')
        v_dates = sub_b.index.intersection(pd.to_datetime(b_stream['date'].values).strftime('%Y-%m-%d'))
        obs_q = b_stream['streamflow'].sel(date=v_dates).values.astype(np.float32)

        m_b = calc_hydro_metrics(obs_q, sub_b.loc[v_dates, 'q_sim'].values)
        m_da = calc_hydro_metrics(obs_q, sub_da.loc[v_dates, 'q_sim'].values)
        summary_rows.append({
            'Lead Time': f'{L} Day(s)',
            'Base NSE': round(m_b['NSE'], 3) if np.isfinite(m_b['NSE']) else np.nan,
            'DA NSE': round(m_da['NSE'], 3) if np.isfinite(m_da['NSE']) else np.nan,
            'Δ NSE': round(m_da['NSE'] - m_b['NSE'], 3) if np.isfinite(m_da['NSE']) and np.isfinite(m_b['NSE']) else np.nan,
            'Base KGE': round(m_b['KGE'], 3) if np.isfinite(m_b['KGE']) else np.nan,
            'DA KGE': round(m_da['KGE'], 3) if np.isfinite(m_da['KGE']) else np.nan,
            'Δ KGE': round(m_da['KGE'] - m_b['KGE'], 3) if np.isfinite(m_da['KGE']) and np.isfinite(m_b['KGE']) else np.nan,
        })

    return {
        'summary_table': pd.DataFrame(summary_rows),
        'df_base': df_base,
        'df_da': df_da,
        'ds_streamflow': b_stream,
    }


def plot_yearly_rolling_hydrographs(
    rolling_results: Dict[str, Any],
    basin_id: str,
    year: int,
    leads_to_plot: Sequence[int] = (1, 3, 7),
) -> Tuple[plt.Figure, Any]:
    """Plots continuous rolling forecast hydrographs comparing Baseline vs DA across chosen lead times."""
    df_base = rolling_results['df_base']
    df_da = rolling_results['df_da']
    b_stream = rolling_results['ds_streamflow']

    n_plots = len(leads_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4.0 * n_plots), sharex=True, dpi=120)
    if n_plots == 1:
        axes = [axes]

    for ax, L in zip(axes, leads_to_plot):
        sub_b = df_base[df_base['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
        sub_da = df_da[df_da['lead_time'] == L].sort_values('valid_date').set_index('valid_date')

        v_dates = sub_b.index.intersection(pd.to_datetime(b_stream['date'].values).strftime('%Y-%m-%d'))
        dates_dt = pd.to_datetime(v_dates)
        obs_q = b_stream['streamflow'].sel(date=v_dates).values.astype(np.float32)
        q_b = sub_b.loc[v_dates, 'q_sim'].values
        q_d = sub_da.loc[v_dates, 'q_sim'].values

        m_b = calc_hydro_metrics(obs_q, q_b)
        m_d = calc_hydro_metrics(obs_q, q_d)

        ax.plot(dates_dt, obs_q, color='black', linestyle='-', linewidth=1.6, alpha=0.85, label=r"Observed ($Q_{\mathrm{obs}}$)", zorder=5)
        ax.plot(dates_dt, q_b, color='#2563eb', linestyle='--', linewidth=1.8,
                label=f"Baseline | NSE: {m_b['NSE']:.3f}, KGE: {m_b['KGE']:.3f}")
        ax.plot(dates_dt, q_d, color='#dc2626', linestyle='-', linewidth=2.0,
                label=f"DA Assimilated | NSE: {m_d['NSE']:.3f}, KGE: {m_d['KGE']:.3f} (ΔNSE: {m_d['NSE'] - m_b['NSE']:+.3f})")

        ax.set_title(f"365-Day Rolling Daily Hydrograph | Basin: {basin_id} | Lead Time = {L} Day(s) Ahead ({year})",
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(r"Streamflow ($\mathrm{mm/day}$)", fontsize=10, fontweight='bold')
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9, loc='upper right', frameon=True, framealpha=0.9)

    axes[-1].set_xlabel("Date", fontsize=11, fontweight='bold')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.tight_layout()
    return fig, axes


# ==============================================================================
# 4. Tier 3: Precipitation Forcing 4D-Var Optimization & Plotting
# ==============================================================================

def run_precip_epoch_sweep(
    model: nn.Module,
    batch_data: Dict[str, Any],
    p_cfg: Dict[str, Any],
    epoch_list: List[int] = [0, 5, 10, 20],
    q_mean: float = 0.0,
    q_std: float = 1.0,
    device: torch.device = torch.device('cpu'),
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Optimizes dynamic precipitation input products via DA gradient backprop with decoupled Wp and Wq."""
    lr = float(p_cfg.get('learning_rate', 0.1))
    precip_window = int(p_cfg.get('precip_window', p_cfg.get('assimilation_window', 90)))
    loss_window = int(p_cfg.get('observation_window', p_cfg.get('loss_window', 14)))
    bg_weight = float(p_cfg.get('bg_regularization_weight', 1e-5))
    max_epochs = max(epoch_list)

    p_keys = [k for k in batch_data['x_d'].keys() if any(t in k.lower() for t in ['precip', 'precipitation'])]
    fc_p_keys = [k for k in batch_data.get('x_d_forecast', {}).keys() if any(t in k.lower() for t in ['precip', 'precipitation'])]

    if not p_keys:
        raise ValueError(f"No precipitation features found in batch: {list(batch_data['x_d'].keys())}")

    t_end = 358
    t_start_p = max(0, t_end - precip_window)
    t_start_q = max(0, t_end - loss_window)
    win_len = t_end - t_start_p

    p_orig = {k: batch_data['x_d'][k][:, t_start_p:t_end, :].detach().clone() for k in p_keys}
    p_opt = {k: p_orig[k].clone().requires_grad_(True) for k in p_keys}

    optimizer = torch.optim.Adam(list(p_opt.values()), lr=lr)
    results = {}
    precip_evol = {}

    y_target_loss = batch_data['y'][:, t_start_q:t_end, :].clone()
    obs_valid = ~torch.isnan(y_target_loss)

    def _clone_batch(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _clone_batch(v) for k, v in d.items()}
        elif isinstance(d, torch.Tensor):
            return d.clone()
        return d

    for ep in range(max_epochs + 1):
        if ep in epoch_list:
            with torch.no_grad():
                b_eval = _clone_batch(batch_data)
                for k in p_keys:
                    b_eval['x_d'][k][:, t_start_p:t_end, :] = p_opt[k]
                    if 'x_d_hindcast' in b_eval and k in b_eval['x_d_hindcast']:
                        b_eval['x_d_hindcast'][k][:, t_start_p:t_end, :] = p_opt[k]
                for k in fc_p_keys:
                    matching = k if k in p_opt else p_keys[0]
                    fc_val = b_eval['x_d_forecast'][k].detach().clone()
                    fc_offset = max(0, fc_val.shape[1] - batch_data['y'].shape[1])
                    fc_start_p = t_start_p + fc_offset
                    fc_end_p = t_end + fc_offset
                    fc_val[:, fc_start_p:fc_end_p, :] = p_opt[matching]
                    b_eval['x_d_forecast'][k] = fc_val

                out_eval = model(b_eval)
                results[ep] = extract_deterministic_y(out_eval, q_mean, q_std)[-batch_data['y'].shape[1]:]
                precip_evol[ep] = {k: p_opt[k].detach().cpu().numpy().copy() for k in p_keys}

        if ep == max_epochs:
            break

        optimizer.zero_grad()
        b_step = _clone_batch(batch_data)
        for k in p_keys:
            b_step['x_d'][k][:, t_start_p:t_end, :] = p_opt[k]
            if 'x_d_hindcast' in b_step and k in b_step['x_d_hindcast']:
                b_step['x_d_hindcast'][k][:, t_start_p:t_end, :] = p_opt[k]
        for k in fc_p_keys:
            matching = k if k in p_opt else p_keys[0]
            fc_val = b_step['x_d_forecast'][k].detach().clone()
            fc_offset = max(0, fc_val.shape[1] - batch_data['y'].shape[1])
            fc_start_p = t_start_p + fc_offset
            fc_end_p = t_end + fc_offset
            fc_val[:, fc_start_p:fc_end_p, :] = p_opt[matching]
            b_step['x_d_forecast'][k] = fc_val

        out = model(b_step)
        out_det = ensure_y_hat(dict(out), use_median=True)
        pred_offset = max(0, out_det['y_hat'].shape[1] - batch_data['y'].shape[1])
        y_hat_loss = out_det['y_hat'][:, t_start_q + pred_offset : t_end + pred_offset, :]

        if obs_valid.any():
            loss_mse = torch.mean((y_hat_loss[obs_valid] - y_target_loss[obs_valid]) ** 2)
        else:
            loss_mse = torch.tensor(0.0, device=device)

        loss_bg = torch.tensor(0.0, device=device)
        for k in p_keys:
            loss_bg = loss_bg + bg_weight * torch.sum((p_opt[k] - p_orig[k]) ** 2)

        total_loss = loss_mse + loss_bg
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for k in p_keys:
                p_opt[k].clamp_(min=-3.0)

    return results, precip_evol


def plot_interactive_precip_da_hydrographs(
    dates_full: np.ndarray,
    obs_full: np.ndarray,
    ep_preds: Dict[int, np.ndarray],
    precip_evol: Dict[int, Dict[str, np.ndarray]],
    da_start_idx: int,
    loss_start_idx: int,
    da_end_idx: int,
    basin_id: str,
    issue_date: str,
    lr: float,
    bg: float,
    precip_window: int,
    loss_window: int,
    scaler: Optional[xr.Dataset] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Renders 3-panel figure: Streamflow hydrograph, Lead 1-7 forecast, and unnormalized rainfall updates."""
    fig, (ax_q, ax_f, ax_p) = plt.subplots(
        3, 1, figsize=(15, 11), dpi=120, gridspec_kw={'height_ratios': [2.0, 1.2, 1.4]}
    )

    lead_start = da_end_idx
    lead_end = min(len(dates_full), da_end_idx + 7)
    plot_start = max(0, da_start_idx - 10)
    plot_end = min(len(dates_full), da_end_idx + 7)
    x_dates = [pd.to_datetime(d) for d in dates_full[plot_start:plot_end]]

    obs_lead = obs_full[lead_start:lead_end]
    base_lead = ep_preds[0][lead_start:lead_end]
    m_base = calc_hydro_metrics(obs_lead, base_lead)
    lbl_base = f"Baseline Model (No DA) | Lead 1-7 NSE: {m_base['NSE']:.2f}, KGE: {m_base['KGE']:.2f}"

    # Panel 1: Hydrograph
    ax_q.plot(x_dates, obs_full[plot_start:plot_end], 'k-o', label=r'Observed Streamflow ($Q_{\mathrm{obs}}$)', linewidth=2.0, alpha=0.85, zorder=5)
    ax_q.plot(x_dates, ep_preds[0][plot_start:plot_end], color='#2563eb', linestyle='--', label=lbl_base, linewidth=2.0)

    cmap = plt.cm.get_cmap('plasma')
    epochs_to_plot = [ep for ep in sorted(ep_preds.keys()) if ep > 0]
    for idx, ep in enumerate(epochs_to_plot):
        color = cmap(idx / max(1, len(epochs_to_plot) - 1))
        sim_lead = ep_preds[ep][lead_start:lead_end]
        m = calc_hydro_metrics(obs_lead, sim_lead)
        ax_q.plot(x_dates, ep_preds[ep][plot_start:plot_end], color=color,
                  label=f"Precip DA (Epoch {ep}) | Lead 1-7 NSE: {m['NSE']:.2f}, KGE: {m['KGE']:.2f}", linewidth=1.8)

    ax_q.axvspan(pd.to_datetime(dates_full[da_start_idx]), pd.to_datetime(dates_full[da_end_idx - 1]), color='orange', alpha=0.12, label=f'Rainfall Correction ($W_P = {precip_window}$d)')
    ax_q.axvspan(pd.to_datetime(dates_full[loss_start_idx]), pd.to_datetime(dates_full[da_end_idx - 1]), color='green', alpha=0.15, label=f'Streamflow Loss ($W_Q = {loss_window}$d)')
    ax_q.axvline(pd.to_datetime(issue_date), color='red', linestyle=':', linewidth=1.8, label=f'Issue Date $t_0$ ({issue_date})')

    ax_q.set_title(f"Precipitation 4D-Var Data Assimilation | Basin: {basin_id} | $W_P$: {precip_window}d | $W_Q$: {loss_window}d | LR: {lr:.1e} | $\\lambda_{{bg}}$: {bg:.1e}", fontsize=11, fontweight='bold')
    ax_q.set_ylabel(r"Streamflow ($\mathrm{mm/day}$)", fontsize=10, fontweight='bold')
    ax_q.grid(True, linestyle='--', alpha=0.5)
    ax_q.legend(fontsize=8.5, loc='upper left')

    # Panel 2: 7-Day Forecast Horizon Profile
    lead_indices = np.arange(1, (lead_end - lead_start) + 1)
    ax_f.plot(lead_indices, obs_lead, 'k-o', label='Observed Streamflow', linewidth=2.2)
    ax_f.plot(lead_indices, base_lead, 'b--s', label=f"Baseline (NSE: {m_base['NSE']:.2f}, KGE: {m_base['KGE']:.2f})", linewidth=1.8)
    for idx, ep in enumerate(epochs_to_plot):
        color = cmap(idx / max(1, len(epochs_to_plot) - 1))
        sim_lead = ep_preds[ep][lead_start:lead_end]
        m = calc_hydro_metrics(obs_lead, sim_lead)
        ax_f.plot(lead_indices, sim_lead, color=color, marker='^', label=f"DA Epoch {ep} (NSE: {m['NSE']:.2f}, KGE: {m['KGE']:.2f})", linewidth=1.6)

    ax_f.set_title(f"7-Day Forecast Profile (Leads $L = 1..7$ Days Ahead)", fontsize=10, fontweight='bold')
    ax_f.set_xlabel("Lead Time (Days)", fontsize=10, fontweight='bold')
    ax_f.set_ylabel(r"Forecast ($\mathrm{mm/day}$)", fontsize=9, fontweight='bold')
    ax_f.set_xticks(lead_indices)
    ax_f.grid(True, linestyle='--', alpha=0.5)
    ax_f.legend(fontsize=8.5, loc='upper right')

    # Panel 3: Unnormalized Rainfall Updates
    win_dates = [pd.to_datetime(d) for d in dates_full[da_start_idx:da_end_idx]]
    p_keys = list(precip_evol[0].keys())
    last_ep = max(epochs_to_plot) if epochs_to_plot else 0
    prod_colors = {
        'hres_total_precipitation': '#2563eb',
        'cpc_precipitation': '#059669',
        'imerg_precipitation': '#d97706',
        'graphcast_total_precipitation': '#7c3aed',
        'era5land_total_precipitation': '#0891b2',
    }

    for p_k in p_keys:
        p_col = prod_colors.get(p_k, '#475569')
        p_short = p_k.replace('_precipitation', '').replace('_total', '').upper()
        if scaler is not None and p_k in scaler.data_vars:
            pm = float(scaler[p_k].sel(parameter='mean').values)
            ps = float(scaler[p_k].sel(parameter='std').values)
            orig_p = np.maximum(0.0, precip_evol[0][p_k][0, :, 0] * ps + pm)
            opt_p = np.maximum(0.0, precip_evol[last_ep][p_k][0, :, 0] * ps + pm) if last_ep in precip_evol else orig_p
        else:
            orig_p = precip_evol[0][p_k][0, :, 0]
            opt_p = precip_evol[last_ep][p_k][0, :, 0] if last_ep in precip_evol else orig_p

        ax_p.plot(win_dates, orig_p, color=p_col, linestyle='--', linewidth=1.5, marker='o', fillstyle='none',
                  markersize=4, alpha=0.6, label=f'Orig {p_short} ({np.sum(orig_p):.1f} mm)')
        ax_p.plot(win_dates, opt_p, color=p_col, linestyle='-', linewidth=2.0, marker='s',
                  markersize=4, alpha=0.95, label=f'Opt {p_short} ({np.sum(opt_p):.1f} mm)')

    ax_p.set_title(f"Rainfall Forcing Over {precip_window}-Day Window (mm/day | Epoch {last_ep})", fontsize=10, fontweight='bold')
    ax_p.set_xlabel("Date", fontsize=10, fontweight='bold')
    ax_p.set_ylabel(r"Precipitation ($\mathrm{mm/day}$)", fontsize=9, fontweight='bold')
    ax_p.set_ylim(bottom=0.0)
    ax_p.grid(True, linestyle='--', alpha=0.5)
    ax_p.legend(fontsize=8, loc='upper left', ncol=2)

    plt.tight_layout()
    return fig, (ax_q, ax_f, ax_p)
