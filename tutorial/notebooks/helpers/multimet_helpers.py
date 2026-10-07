"""MultiMet Weather Forcing & Feature Engineering Helpers for Data Assimilation."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import xarray as xr

from googlehydrology import datasetzoo
from googlehydrology import evaluation
from googlehydrology import modelzoo
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology import utils

try:
    from googlehydrology.modelzoo.head import ensure_y_hat
except ImportError:
    def ensure_y_hat(out: Any, use_median: bool = True) -> Any:
        return out


DEFAULT_MULTIMET_DIR = Path(os.environ.get('MULTIMET_DIR', 'data/Caravans_MultiMet'))
_GLOBAL_MULTIMET_DS_CACHE: Dict[str, xr.Dataset] = {}


def norm_var(scaler: xr.Dataset, var_name: str, val: np.ndarray) -> np.ndarray:
    """Normalize feature values using mean and std from scaler.nc."""
    m_key = f"{var_name}_sim" if f"{var_name}_sim" in scaler else var_name
    s_key = f"{var_name}_std" if f"{var_name}_std" in scaler else (f"{var_name}_sim" if f"{var_name}_sim" in scaler else var_name)

    if m_key in scaler:
        mean_val = float(scaler[m_key].sel(parameter='mean').values)
        std_val = float(scaler[s_key].sel(parameter='std').values)
    else:
        mean_val = 0.0
        std_val = 1.0

    if std_val == 0 or np.isnan(std_val):
        std_val = 1.0

    return (val - mean_val) / std_val


def _get_source_var(union_map: dict, f_name: str) -> str:
    """Resolve target variable name to its source product name using union_map."""
    if union_map and f_name in union_map:
        return union_map[f_name]

    var_map = {
        'hres_total_precipitation': 'era5land_total_precipitation',
        'hres_temperature_2m': 'era5land_temperature_2m',
        'hres_surface_net_solar_radiation': 'era5land_surface_net_solar_radiation',
        'hres_surface_net_thermal_radiation': 'era5land_surface_net_thermal_radiation',
        'hres_surface_pressure': 'era5land_surface_pressure',
        'graphcast_total_precipitation': 'era5land_total_precipitation',
        'graphcast_temperature_2m': 'era5land_temperature_2m',
        'cpc_total_precipitation': 'era5land_total_precipitation',
        'cpc_precipitation': 'era5land_total_precipitation',
        'imerg_total_precipitation': 'era5land_total_precipitation',
        'imerg_precipitation': 'era5land_total_precipitation',
    }
    if f_name in var_map:
        return var_map[f_name]

    for prefix in ['hres_', 'graphcast_', 'cpc_', 'imerg_']:
        if f_name.startswith(prefix):
            return 'era5land_' + f_name[len(prefix):]
    return f_name


def _get_product_for_var(var_name: str) -> str:
    """Determines MultiMet product directory from variable prefix."""
    v = var_name.lower()
    if v.startswith('graphcast'):
        return 'GRAPHCAST'
    if v.startswith('hres'):
        return 'HRES'
    if v.startswith('cpc'):
        return 'CPC'
    if v.startswith('imerg'):
        return 'IMERG'
    return 'ERA5_LAND'


def load_multimet_product(
    product_name: str,
    multimet_dir: Path | str = DEFAULT_MULTIMET_DIR,
    basin_id: Optional[str] = None
) -> xr.Dataset:
    """Loads a MultiMet product timeseries.zarr store."""
    prod_path = Path(multimet_dir) / product_name / 'timeseries.zarr'
    try:
        ds = xr.open_zarr(prod_path, consolidated=True, decode_timedelta=False)
    except Exception:
        ds = xr.open_zarr(prod_path, decode_timedelta=False)
    if basin_id is not None:
        ds = ds.sel(basin=basin_id)
    return ds


def prepare_multimet_batch(
    mode: str,
    basin_id: str,
    issue_date: str,
    cfg: Any,
    scaler: xr.Dataset,
    caravan_attrs: pd.DataFrame,
    ds_caravan: xr.Dataset,
    multimet_dir: Path | str = DEFAULT_MULTIMET_DIR,
    hindcast_window_days: int = 365,
    forecast_lead_days: int = 7,
    forecast_product: str = 'HRES',
    fixed_leadtime: int = 0
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Builds a standardized PyTorch batch dictionary for MeanEmbeddingForecastLSTM with Data Assimilation."""
    valid_modes = ['all_reanalysis', 'reanalysis_and_fixed_leadtime', 'multimet_0_and_1_to_7']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

    multimet_path = Path(multimet_dir)
    t0_dt = pd.to_datetime(issue_date)

    start_date_h = (t0_dt - pd.Timedelta(days=hindcast_window_days - 1)).strftime('%Y-%m-%d')
    end_date_h = t0_dt.strftime('%Y-%m-%d')
    end_date_f = (t0_dt + pd.Timedelta(days=forecast_lead_days)).strftime('%Y-%m-%d')

    dates_h = pd.date_range(start_date_h, end_date_h).strftime('%Y-%m-%d').values
    dates_f = pd.date_range(start_date_h, end_date_f).strftime('%Y-%m-%d').values

    # Extract observed streamflow from Caravan dataset
    obs_discharge = ds_caravan['streamflow'].sel(date=dates_f).values.astype(np.float32)

    # Streamflow normalization parameters
    q_mean = float(scaler['streamflow'].sel(parameter='mean').values)
    q_std = float(scaler['streamflow'].sel(parameter='std').values)

    hindcast_dict = {}
    forecast_dict = {}
    union_map = getattr(cfg, 'union_mapping', {})

    def _get_dataset(prod_name: str) -> xr.Dataset:
        cache_key = f"{multimet_path}_{prod_name}_{basin_id}"
        if cache_key not in _GLOBAL_MULTIMET_DS_CACHE:
            p = multimet_path / prod_name / 'timeseries.zarr'
            try:
                _GLOBAL_MULTIMET_DS_CACHE[cache_key] = xr.open_zarr(p, consolidated=True, decode_timedelta=False).sel(basin=basin_id)
            except Exception:
                _GLOBAL_MULTIMET_DS_CACHE[cache_key] = xr.open_zarr(p, decode_timedelta=False).sel(basin=basin_id)
        return _GLOBAL_MULTIMET_DS_CACHE[cache_key]

    def _fetch_var(var_name: str, dates_target: np.ndarray, lead_idx: Optional[int] = 0) -> np.ndarray:
        prod = _get_product_for_var(var_name)
        ds = _get_dataset(prod)
        da = ds[var_name]
        if 'lead_time' in da.dims and lead_idx is not None:
            lead_offset_days = int(lead_idx) + 1
            shifted_dates = (pd.to_datetime(dates_target) - pd.Timedelta(days=lead_offset_days)).strftime('%Y-%m-%d').values
            da = da.sel(date=shifted_dates).isel(lead_time=lead_idx)
        else:
            da = da.sel(date=dates_target)
        return da.values.astype(np.float32)

    # -------------------------------------------------------------------------
    # Option 1: All Reanalysis (Lead 0 / Observation across entire span)
    # -------------------------------------------------------------------------
    if mode == 'all_reanalysis':
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_f, lead_idx=0)
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 2: Reanalysis (Hindcast) & Fixed Leadtime Forecast
    # -------------------------------------------------------------------------
    elif mode == 'reanalysis_and_fixed_leadtime':
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_f, lead_idx=fixed_leadtime)
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 3: MultiMet Ensemble (Lead 0 for history, Lead 1..7 for forecast)
    # -------------------------------------------------------------------------
    elif mode == 'multimet_0_and_1_to_7':
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                prod = _get_product_for_var(src_var)
                ds = _get_dataset(prod)
                da = ds[src_var]
                
                if 'lead_time' in da.dims:
                    shifted_dates_h = (pd.to_datetime(dates_h) - pd.Timedelta(days=1)).strftime('%Y-%m-%d').values
                    overlap_vals = da.sel(date=shifted_dates_h).isel(lead_time=0).values.astype(np.float32)
                    fc_1_7 = da.sel(date=issue_date).isel(lead_time=slice(0, forecast_lead_days)).values.astype(np.float32)
                    f_arr = np.concatenate([overlap_vals, fc_1_7], axis=0)
                else:
                    f_arr = da.sel(date=dates_f).values.astype(np.float32)
                
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, f_arr), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        obs_discharge = obs_discharge.copy()
        obs_discharge[-forecast_lead_days:] = np.nan

    # -------------------------------------------------------------------------
    # Static Attributes & Target Tensor Formatting
    # -------------------------------------------------------------------------
    if basin_id not in caravan_attrs.index:
        raise ValueError(f"Basin '{basin_id}' not found in static attributes dataset.")
    basin_attrs = caravan_attrs.loc[basin_id]
    static_vals = []
    for attr_name in cfg.static_attributes:
        if attr_name not in basin_attrs or pd.isna(basin_attrs[attr_name]):
            raise ValueError(f"Basin '{basin_id}' is missing required static attribute '{attr_name}'.")
        raw_val = float(basin_attrs[attr_name])
        normed_val = norm_var(scaler, attr_name, raw_val)
        static_vals.append(normed_val)

    x_s_tensor = torch.tensor(static_vals, dtype=torch.float32).unsqueeze(0)
    obs_q_norm = (obs_discharge - q_mean) / q_std
    y_tensor = torch.tensor(obs_q_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    batch_data = {
        'date': np.tile(dates_f, (1, 1)),
        'x_s': x_s_tensor,
        'x_d_hindcast': hindcast_dict,
        'x_d_forecast': forecast_dict,
        'x_d': hindcast_dict,
        'y': y_tensor,
        'c_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32),
        'h_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32),
    }

    return batch_data


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Computes NSE, KGE, Pearson-r, and RMSE between observed and simulated discharge."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    if not np.any(valid):
        return {'NSE': np.nan, 'KGE': np.nan, 'Pearson-r': np.nan, 'RMSE': np.nan}

    o, s = obs[valid], sim[valid]
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - (np.sum((o - s) ** 2) / denom)) if denom != 0 else np.nan
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    r = float(np.corrcoef(o, s)[0, 1]) if len(o) > 1 else np.nan
    std_o, std_s = np.std(o), np.std(s)
    kge = float(1 - np.sqrt((r - 1)**2 + (std_s/std_o - 1)**2 + (np.mean(s)/np.mean(o) - 1)**2)) if std_o > 0 and np.mean(o) > 0 else np.nan

    return {'NSE': nse, 'KGE': kge, 'Pearson-r': r, 'RMSE': rmse}


def multimet_batch_generator(
    basin_list: List[str],
    issue_dates: List[str],
    cfg: Any,
    scaler: xr.Dataset,
    caravan_attrs: pd.DataFrame,
    ds_streamflow: Optional[xr.Dataset] = None,
    mode: str = 'multimet_0_and_1_to_7',
    forecast_product: str = 'HRES',
    hindcast_window_days: int = 358,
    forecast_lead_days: int = 7,
    fixed_leadtime: int = 0,
    multimet_dir: Path | str = DEFAULT_MULTIMET_DIR
):
    """Generator yielding (basin_id, issue_date, batch_data) using prepare_multimet_batch."""
    for b_id in basin_list:
        ds_b = ds_streamflow.sel(basin=b_id) if ds_streamflow is not None and 'basin' in ds_streamflow.dims else ds_streamflow
        for dt in issue_dates:
            batch = prepare_multimet_batch(
                mode=mode,
                basin_id=b_id,
                issue_date=dt,
                cfg=cfg,
                scaler=scaler,
                caravan_attrs=caravan_attrs,
                ds_caravan=ds_b,
                multimet_dir=multimet_dir,
                forecast_product=forecast_product,
                hindcast_window_days=hindcast_window_days,
                forecast_lead_days=forecast_lead_days,
                fixed_leadtime=fixed_leadtime
            )
            yield b_id, dt, batch


def run_multibatch_da_pipeline(
    basin_list: List[str],
    issue_dates: List[str],
    da_cfg: Union[dict, Any],
    model: Optional[BaseModel] = None,
    cfg: Any = None,
    scaler: Optional[xr.Dataset] = None,
    caravan_attrs: Optional[pd.DataFrame] = None,
    ds_streamflow: Optional[xr.Dataset] = None,
    lead_times: List[int] = [0, 3, 5, 7],
    multimet_dir: Path | str = DEFAULT_MULTIMET_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Dict[str, Any]]]]:
    """Runs Data Assimilation across multiple issue dates and lead times."""
    from googlehydrology.utils.assimilationconfig import AssimilationConfig
    from googlehydrology.evaluation.assimilation import Assimilation

    if isinstance(da_cfg, dict):
        da_cfg = AssimilationConfig(da_cfg)
    assim = Assimilation(da_cfg)

    # Resolve global variables if not explicitly passed
    if model is None or cfg is None or scaler is None or caravan_attrs is None or ds_streamflow is None:
        try:
            import __main__
            if model is None: model = getattr(__main__, 'model', None)
            if cfg is None: cfg = getattr(__main__, 'cfg', None)
            if scaler is None: scaler = getattr(__main__, 'scaler', None)
            if caravan_attrs is None: caravan_attrs = getattr(__main__, 'caravan_attrs', None)
            if ds_streamflow is None: ds_streamflow = getattr(__main__, 'ds_streamflow', getattr(__main__, 'ds_caravan', None))
        except Exception:
            pass

    if model is None or cfg is None or scaler is None or caravan_attrs is None or ds_streamflow is None:
        raise ValueError("model, cfg, scaler, caravan_attrs, and ds_streamflow must not be None.")

    q_mean = float(scaler['streamflow'].sel(parameter='mean').values)
    q_std = float(scaler['streamflow'].sel(parameter='std').values)

    records = []
    batch_hydrographs = {}

    for b_id in basin_list:
        ds_b = ds_streamflow.sel(basin=b_id) if 'basin' in ds_streamflow.dims else ds_streamflow
        for dt in issue_dates:
            batch_hydrographs[dt] = {}
            for L in lead_times:
                batch_L = prepare_multimet_batch(
                    mode='reanalysis_and_fixed_leadtime',
                    basin_id=b_id,
                    issue_date=dt,
                    cfg=cfg,
                    scaler=scaler,
                    caravan_attrs=caravan_attrs,
                    ds_caravan=ds_b,
                    multimet_dir=multimet_dir,
                    forecast_product='HRES',
                    fixed_leadtime=L,
                    hindcast_window_days=358,
                    forecast_lead_days=7
                )
                batch_dates = pd.to_datetime(batch_L['date'][0]).strftime('%Y-%m-%d').values
                obs_q = ds_b['streamflow'].sel(date=batch_dates).values.astype(np.float32)

                with torch.no_grad():
                    b_out = ensure_y_hat(model(batch_L), use_median=True)
                    y_b = b_out['y_hat'] if not isinstance(b_out['y_hat'], dict) else b_out['y_hat'].get('mean', b_out['y_hat'].get('q50'))
                    q_base_norm = y_b[0, :, 0].detach().cpu().numpy()
                q_base = np.maximum(0.1, q_base_norm * q_std + q_mean)

                da_out = ensure_y_hat(assim.assimilate(model, batch_L, verbose=False), use_median=True)
                y_d = da_out['y_hat'] if not isinstance(da_out['y_hat'], dict) else da_out['y_hat'].get('mean', da_out['y_hat'].get('q50'))
                q_da_norm = y_d[0, :, 0].detach().cpu().numpy()
                q_da = np.maximum(0.1, q_da_norm * q_std + q_mean)

                batch_hydrographs[dt][L] = {
                    'dates': batch_dates,
                    'obs': obs_q,
                    'baseline': q_base,
                    'da_model': q_da
                }

                m_base = compute_metrics(obs_q, q_base)
                m_da = compute_metrics(obs_q, q_da)

                records.append({
                    'Basin ID': b_id,
                    'Issue Date': dt,
                    'Lead Time (Days)': L,
                    'Base NSE': round(m_base['NSE'], 3),
                    'Base KGE': round(m_base['KGE'], 3),
                    'DA NSE': round(m_da['NSE'], 3),
                    'DA KGE': round(m_da['KGE'], 3),
                    'NSE Delta': round(m_da['NSE'] - m_base['NSE'], 3),
                    'KGE Delta': round(m_da['KGE'] - m_base['KGE'], 3)
                })

    df_metrics = pd.DataFrame(records)
    return df_metrics, batch_hydrographs
