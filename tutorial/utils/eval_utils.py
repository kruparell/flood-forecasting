"""Hydrologic evaluation, assimilation configuration, and metric utilities."""

from typing import Dict, Any, Union, List
import numpy as np
import xarray as xr

from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.evaluation import metrics as gh_metrics


def build_assim_config(p_cfg: Dict[str, Any], forecast_lead_days: int = 7) -> AssimilationConfig:
    """Constructs AssimilationConfig dictionary from parameter grid specification."""
    dist_from_fc = p_cfg.get('distance_from_forecast', 1)
    assim_lead_time = forecast_lead_days + dist_from_fc

    targets = p_cfg.get('assimilation_targets', ['c_n_forecast'])
    if isinstance(targets, tuple):
        targets = list(targets)

    cfg_dict = {
        'seq_length': 365,
        'history': 1,
        'assimilation_window': p_cfg['assimilation_window'],
        'assimilation_lead_time': assim_lead_time,
        'predict_n_hindcast': dist_from_fc,
        'predict_last_n': forecast_lead_days,
        'learning_rate': p_cfg['learning_rate'],
        'epochs': p_cfg['epochs'],
        'bg_regularization_weight': p_cfg['bg_regularization_weight'],
        'optimizer': 'Adam',
        'loss': 'MSE',
        'assimilation_targets': targets,
        'target_variables': ['streamflow'],
    }
    return AssimilationConfig(cfg_dict)


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Computes standard hydrologic metrics using googlehydrology.evaluation.metrics."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[valid], sim[valid]

    metric_keys = ['NSE', 'KGE', 'Alpha-NSE', 'Beta-KGE', 'Pearson-r', 'RMSE', 'FHV', 'FMS', 'FLV']
    default_res = {k: np.nan for k in metric_keys}

    if len(o) < 5 or np.std(o) == 0:
        return default_res

    obs_da = xr.DataArray(o, dims=['time'])
    sim_da = xr.DataArray(s, dims=['time'])

    metrics_to_compute = ['nse', 'kge', 'alpha-nse', 'beta-kge', 'pearson-r', 'rmse']
    try:
        res = gh_metrics.calculate_metrics(obs_da, sim_da, metrics=metrics_to_compute)
    except Exception:
        res = {k: np.nan for k in default_res}

    # Compute FDC metrics safely if data length allows (>= 30 timesteps)
    try:
        if len(o) >= 30:
            res['FHV'] = gh_metrics.fdc_fhv(obs_da, sim_da)
            res['FMS'] = gh_metrics.fdc_fms(obs_da, sim_da)
            res['FLV'] = gh_metrics.fdc_flv(obs_da, sim_da)
        else:
            res['FHV'], res['FMS'], res['FLV'] = np.nan, np.nan, np.nan
    except Exception:
        res['FHV'], res['FMS'], res['FLV'] = np.nan, np.nan, np.nan

    out = {}
    for k in metric_keys:
        val = res.get(k, np.nan)
        out[k] = round(float(val), 4) if not np.isnan(val) else np.nan
    return out


def unnormalize_streamflow(q_norm: np.ndarray, q_mean: float, q_std: float, min_val: float = 0.1) -> np.ndarray:
    """Converts normalized model output back to physical streamflow scale (m3/s or mm/d)."""
    return np.maximum(min_val, q_norm * q_std + q_mean)
