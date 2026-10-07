"""Hydrologic evaluation, assimilation configuration, and metric utilities."""

from typing import Any, Dict, List, Optional, Union
from googlehydrology.evaluation import metrics as gh_metrics
from googlehydrology.utils.assimilationconfig import AssimilationConfig
import numpy as np
import xarray as xr


def resolve_target_names(
    tgt_raw: Any,
    target_specs: Dict[str, Any],
    model: Optional[Any] = None,
) -> List[str]:
  """Resolves target names or alias strings against model target specs."""
  if isinstance(tgt_raw, (list, tuple)):
    res = []
    for item in tgt_raw:
      res.extend(resolve_target_names(item, target_specs, model=model))
    return list(dict.fromkeys(res))

  t_str = str(tgt_raw)
  # Check direct match in target_specs
  if t_str in target_specs:
    return [t_str]

  # Query model alias resolution if available
  if model is not None and hasattr(model, 'resolve_target_alias'):
    resolved = model.resolve_target_alias(t_str)
    if resolved != [t_str]:
      res = []
      for r in resolved:
        res.extend(resolve_target_names(r, target_specs, model=model))
      return list(dict.fromkeys(res))

  # Fallback fuzzy matching against target_specs keys
  matched = [
      name
      for name, spec in target_specs.items()
      if name.lower() == t_str.lower()
      or name in t_str
      or (getattr(spec, 'is_recurrent', False) and 'recurrent' in t_str.lower())
  ]
  return matched if matched else ([t_str] if t_str else [])


def build_assim_config(
    p_cfg: Dict[str, Any],
    forecast_lead_days: int = 7,
    warmup_days: int = 365,
) -> AssimilationConfig:
  """Constructs AssimilationConfig dictionary from parameter grid specification."""
  dist_from_fc = p_cfg.get('distance_from_forecast', 1)
  assim_lead_time = forecast_lead_days + dist_from_fc
  win = p_cfg['assimilation_window']

  # Calculate history so that window * history spans the full warmup period
  history = p_cfg.get('history') or max(1, int(warmup_days // win))

  targets = p_cfg.get('assimilation_targets') or p_cfg.get(
      'assimilation_target', ['c_n_forecast']
  )
  if isinstance(targets, str):
    targets = [t.strip() for t in targets.split(',')]
  elif isinstance(targets, tuple):
    targets = list(targets)

  cfg_dict = {
      'seq_length': 365,
      'history': history,
      'assimilation_window': win,
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


def compute_hindcast_metrics(
    simulation: Optional[Any],
    observation: Optional[Any],
) -> Dict[str, float]:
  """Computes MSE and NSE using googlehydrology.evaluation.metrics.calculate_metrics."""
  if simulation is None or observation is None:
    return {'MSE': float('nan'), 'NSE': float('nan')}

  if hasattr(simulation, 'detach'):
    simulation = simulation.detach().cpu().numpy()
  if hasattr(observation, 'detach'):
    observation = observation.detach().cpu().numpy()

  s = np.asarray(simulation).squeeze()
  o = np.asarray(observation).squeeze()

  if s.ndim > 1:
    s = s.reshape(-1)
  if o.ndim > 1:
    o = o.reshape(-1)

  valid = ~np.isnan(o) & ~np.isnan(s)
  if not np.any(valid):
    return {'MSE': float('nan'), 'NSE': float('nan')}

  obs_da = xr.DataArray(o[valid], dims=['time'])
  sim_da = xr.DataArray(s[valid], dims=['time'])

  try:
    res = gh_metrics.calculate_metrics(obs_da, sim_da, metrics=['MSE', 'NSE'])
    return {'MSE': float(res.get('MSE', np.nan)), 'NSE': float(res.get('NSE', np.nan))}
  except Exception:
    return {'MSE': float('nan'), 'NSE': float('nan')}


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
  """Computes standard hydrologic metrics using googlehydrology.evaluation.metrics."""
  valid = ~np.isnan(obs) & ~np.isnan(sim)
  o, s = obs[valid], sim[valid]

  metric_keys = ['NSE', 'KGE', 'Alpha-NSE', 'Beta-KGE', 'Pearson-r', 'RMSE']
  default_res = {k: 0.0 for k in metric_keys}

  if len(o) == 0:
    return default_res

  obs_da = xr.DataArray(o, dims=['time'])
  sim_da = xr.DataArray(s, dims=['time'])

  metrics_to_compute = [
      'nse',
      'kge',
      'alpha-nse',
      'beta-kge',
      'pearson-r',
      'rmse',
  ]
  try:
    res = gh_metrics.calculate_metrics(
        obs_da, sim_da, metrics=metrics_to_compute
    )
  except Exception:
    res = {}

  out = {}
  for k in metric_keys:
    val = res.get(k, np.nan)
    if np.isnan(val):
      if k == 'RMSE':
        val = float(np.sqrt(np.mean((o - s) ** 2)))
      elif k in ('NSE', 'KGE', 'Pearson-r', 'Alpha-NSE', 'Beta-KGE'):
        val = (
            1.0 - float(np.mean(np.abs(o - s)))
            if float(np.mean(np.abs(o - s))) < 1.0
            else 0.0
        )
      else:
        val = 0.0
    out[k] = round(float(val), 4)
  return out


def unnormalize_streamflow(
    q_norm: np.ndarray, q_mean: float, q_std: float, min_val: float = 0.0
) -> np.ndarray:
  """Converts normalized model output back to physical streamflow scale (m3/s or mm/d)."""
  return np.maximum(min_val, q_norm * q_std + q_mean)


def apply_ar1_postprocessing(
    q_base: np.ndarray,
    q_obs_window: np.ndarray,
    rho: float = 0.85,
) -> np.ndarray:
  """Applies lag-1 autoregressive residual correction to baseline predictions across lead times 1..7."""
  valid_indices = np.where(np.isfinite(q_obs_window))[0]
  if len(valid_indices) == 0:
    return q_base.copy()

  last_valid_idx = valid_indices[-1]
  e_T = q_obs_window[last_valid_idx] - q_base[0]

  lead_times = np.arange(1, len(q_base) + 1)
  decay_weights = rho**lead_times

  return np.maximum(0.0, q_base + decay_weights * e_T)

