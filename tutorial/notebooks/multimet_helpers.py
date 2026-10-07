"""MultiMet Batch Preparation Helpers for Google Hydrology & Caravans

This module provides helper functions to load Caravans / CaravansMultiMet data
and construct input batches (hindcast_dict, forecast_dict, x_s, y) for model
inference and Data Assimilation under 3 configurable modes:

1. 'all_reanalysis':
   Uses ERA5-Land reanalysis from Caravans for all dates across both hindcast
   and forecast dictionaries.

2. 'reanalysis_and_fixed_leadtime':
   Uses ERA5-Land reanalysis for the hindcast period, and a fixed-leadtime
   forecast product (e.g., ECMWF HRES at lead_time=0 or fixed leadtime) for the
   forecast dictionary.

3. 'multimet_0_and_1_to_7':
   Uses 0-day leadtime (nowcast) forecast data for both hindcast and forecast
   inputs up to forecast issue date t0. For the final 7 days (t0+1..t0+7):
   - hindcast_dict receives NaNs.
   - forecast_dict receives leadtime 1..7 forecasts issued at date t0.
"""

from pathlib import Path
import sys
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import xarray as xr

from googlehydrology.utils.gfile_utils import get_gfile

try:
    import googlehydrology
    sys.modules.setdefault('neuralhydrology', googlehydrology)
    sys.modules.setdefault('neuralhydrology.modelzoo', googlehydrology.modelzoo)
    if hasattr(googlehydrology.modelzoo, 'cudalstm'):
        sys.modules.setdefault('neuralhydrology.modelzoo.cudalstm', googlehydrology.modelzoo.cudalstm)
except ImportError:
    pass


def norm_var(scaler: xr.Dataset, var_name: str, val: np.ndarray) -> np.ndarray:
    """Normalize feature values using mean and std from scaler.nc."""
    m_key = f"{var_name}_sim" if f"{var_name}_sim" in scaler else var_name
    if m_key not in scaler:
        return val
    mean = float(scaler[m_key].sel(parameter='mean').values)
    std = float(scaler[m_key].sel(parameter='std').values)
    if std < 1e-6:
        std = 1.0
    return (val - mean) / std


def _get_source_var(union_map: dict, f_name: str) -> str:
    """Map feature name to ERA5-Land reanalysis key in ds_caravan."""
    if f_name in union_map:
        return union_map[f_name]
    
    var_map = {
        'hres_total_precipitation': 'era5land_total_precipitation',
        'hres_temperature_2m': 'era5land_temperature_2m',
        'hres_surface_net_solar_radiation': 'era5land_surface_net_solar_radiation',
        'hres_surface_net_thermal_radiation': 'era5land_surface_net_thermal_radiation',
        'hres_surface_pressure': 'era5land_surface_pressure',
        'graphcast_total_precipitation': 'era5land_total_precipitation',
        'graphcast_temperature_2m': 'era5land_temperature_2m',
    }
    if f_name in var_map:
        return var_map[f_name]

    for prefix in ['hres_', 'graphcast_', 'cpc_', 'imerg_']:
        if f_name.startswith(prefix):
            return 'era5land_' + f_name[len(prefix):]
    return f_name


def prepare_multimet_batch(
    mode: str,
    basin_id: str,
    issue_date: str,
    cfg,
    scaler: xr.Dataset,
    caravan_attrs: pd.DataFrame,
    ds_caravan: xr.Dataset,
    multimet_dir: Union[Path, str] = '/usr/local/google/home/kruparell/Caravans_MultiMet',
    hindcast_window_days: int = 90,
    forecast_lead_days: int = 7,
    forecast_product: str = 'HRES',
    fixed_leadtime: int = 0
) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]]]:
    """Prepare a PyTorch batch dictionary for model forward pass / Data Assimilation.

    Parameters
    ----------
    mode : str
        One of ['all_reanalysis', 'reanalysis_and_fixed_leadtime', 'multimet_0_and_1_to_7']
    basin_id : str
        Basin identifier (e.g. 'camels_12451000')
    issue_date : str
        Forecast issue date t0 (e.g. '2020-04-30')
    cfg : Config
        Google Hydrology run configuration instance
    scaler : xr.Dataset
        Dataset containing feature normalization statistics (scaler.nc)
    caravan_attrs : pd.DataFrame
        DataFrame of catchment static attributes indexed by gauge_id
    ds_caravan : xr.Dataset
        Full Caravan dataset for streamflow observations
    multimet_dir : Path or str
        Path to CaravansMultiMet directory containing Zarr archives
    hindcast_window_days : int
        Number of historical days up to issue_date t0 (default 90)
    forecast_lead_days : int
        Number of forecast lead days after t0 (default 7)
    forecast_product : str
        Forecast product name ('HRES' or 'GRAPHCAST')
    fixed_leadtime : int
        Fixed lead_time index (0-based) for 'reanalysis_and_fixed_leadtime' mode

    Returns
    -------
    dict
        PyTorch batch dictionary ready for model forward pass containing:
        'date', 'x_s', 'x_d', 'x_d_forecast', 'y', 'c_n', 'h_n'
    """
    valid_modes = ['all_reanalysis', 'reanalysis_and_fixed_leadtime', 'multimet_0_and_1_to_7']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

    multimet_path = Path(multimet_dir)
    t0_dt = pd.to_datetime(issue_date)
    
    start_date = (t0_dt - pd.Timedelta(days=hindcast_window_days - 1)).strftime('%Y-%m-%d')
    end_date = (t0_dt + pd.Timedelta(days=forecast_lead_days)).strftime('%Y-%m-%d')
    
    dates_arr = pd.date_range(start_date, end_date)
    dates_str = dates_arr.strftime('%Y-%m-%d').values
    T_total = len(dates_str)  # hindcast_window_days + forecast_lead_days

    # Extract observed streamflow from Caravan dataset
    ds_caravan_slice = ds_caravan['streamflow'].sel(date=slice(start_date, end_date))
    obs_discharge = ds_caravan_slice.values.astype(np.float32)

    # Streamflow normalization parameters
    q_mean = float(scaler['streamflow'].sel(parameter='mean').values)
    q_std = float(scaler['streamflow'].sel(parameter='std').values)

    # Feature mapping dictionaries
    hindcast_dict = {}
    forecast_dict = {}

    # Open MultiMet product Zarr if required
    prod_zarr_name = 'HRES' if forecast_product.upper() == 'HRES' else 'GRAPHCAST'
    multimet_ds = None
    if mode != 'all_reanalysis':
        prod_path = multimet_path / prod_zarr_name / 'timeseries.zarr'
        gf = get_gfile()
        prod_exists = False
        prod_str = str(prod_path)
        if gf and prod_str.startswith('/cns/'):
            prod_exists = gf.Exists(prod_str)
        else:
            prod_exists = prod_path.exists()

        if prod_exists:
            try:
                multimet_ds = xr.open_zarr(prod_path, consolidated=True, decode_timedelta=False).sel(basin=basin_id)
            except Exception:
                multimet_ds = xr.open_zarr(prod_path, decode_timedelta=False).sel(basin=basin_id)

    union_map = getattr(cfg, 'union_mapping', {})

    def _load_reanalysis_map():
        if 'total_precipitation_sum' in ds_caravan:
            return {
                'era5land_total_precipitation': ds_caravan['total_precipitation_sum'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_precipitation': ds_caravan['total_precipitation_sum'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_temperature_2m': ds_caravan['temperature_2m_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_temperature': ds_caravan['temperature_2m_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_surface_net_solar_radiation': ds_caravan['surface_net_solar_radiation_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_solar_radiation': ds_caravan['surface_net_solar_radiation_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_surface_net_thermal_radiation': ds_caravan['surface_net_thermal_radiation_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_thermal_radiation': ds_caravan['surface_net_thermal_radiation_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_surface_pressure': ds_caravan['surface_pressure_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
                'era5land_pressure': ds_caravan['surface_pressure_mean'].sel(date=slice(start_date, end_date)).values.astype(np.float32),
            }
        else:
            era5_path = multimet_path / 'ERA5_LAND' / 'timeseries.zarr'
            try:
                era5_ds = xr.open_zarr(era5_path, consolidated=True, decode_timedelta=False).sel(basin=basin_id, date=dates_str)
            except Exception:
                era5_ds = xr.open_zarr(era5_path, decode_timedelta=False).sel(basin=basin_id, date=dates_str)
            
            p_val = era5_ds['era5land_total_precipitation'].values.astype(np.float32)
            t_val = era5_ds['era5land_temperature_2m'].values.astype(np.float32)
            s_val = era5_ds['era5land_surface_net_solar_radiation'].values.astype(np.float32)
            th_val = era5_ds['era5land_surface_net_thermal_radiation'].values.astype(np.float32)
            pr_val = era5_ds['era5land_surface_pressure'].values.astype(np.float32)
            return {
                'era5land_total_precipitation': p_val,
                'era5land_precipitation': p_val,
                'era5land_temperature_2m': t_val,
                'era5land_temperature': t_val,
                'era5land_surface_net_solar_radiation': s_val,
                'era5land_solar_radiation': s_val,
                'era5land_surface_net_thermal_radiation': th_val,
                'era5land_thermal_radiation': th_val,
                'era5land_surface_pressure': pr_val,
                'era5land_pressure': pr_val,
            }

    # -------------------------------------------------------------------------
    # Option 1: All Reanalysis (Caravans ERA5-Land for all dates & dictionaries)
    # -------------------------------------------------------------------------
    if mode == 'all_reanalysis':
        dyn_map = _load_reanalysis_map()

        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                source_var = _get_source_var(union_map, f_name)
                val = dyn_map[source_var]
                normed = norm_var(scaler, source_var, val)
                hindcast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                source_var = _get_source_var(union_map, f_name)
                val = dyn_map[source_var]
                normed = norm_var(scaler, source_var, val)
                forecast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 2: Caravans Reanalysis (Hindcast) & Fixed Leadtime Forecast
    # -------------------------------------------------------------------------
    elif mode == 'reanalysis_and_fixed_leadtime':
        reanalysis_map = _load_reanalysis_map()

        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                source_var = _get_source_var(union_map, f_name)
                val = reanalysis_map[source_var]
                normed = norm_var(scaler, source_var, val)
                hindcast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                if multimet_ds is not None and f_name in multimet_ds:
                    val = multimet_ds[f_name].sel(date=dates_str).isel(lead_time=fixed_leadtime).values.astype(np.float32)
                else:
                    source_var = _get_source_var(union_map, f_name)
                    val = reanalysis_map[source_var]
                normed = norm_var(scaler, f_name, val)
                forecast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 3: Leadtime 0 Forecasts for Hindcast & Forecast, NaNs/Leadtimes 1..7 for Final 7 Days
    # -------------------------------------------------------------------------
    elif mode == 'multimet_0_and_1_to_7':
        reanalysis_map = _load_reanalysis_map()

        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                if multimet_ds is not None and f_name in multimet_ds:
                    lead0_vals = multimet_ds[f_name].sel(date=dates_str).isel(lead_time=0).values.astype(np.float32)
                else:
                    source_var = _get_source_var(union_map, f_name)
                    lead0_vals = reanalysis_map[source_var]
                
                h_arr = lead0_vals.copy()
                h_arr[-forecast_lead_days:] = np.nan  # Mask final 7 forecast days with NaNs
                normed = norm_var(scaler, f_name, h_arr)
                hindcast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                if multimet_ds is not None and f_name in multimet_ds:
                    lead0_vals = multimet_ds[f_name].sel(date=dates_str).isel(lead_time=0).values.astype(np.float32)
                    fc_1_7_vals = multimet_ds[f_name].sel(date=issue_date).isel(lead_time=slice(0, forecast_lead_days)).values.astype(np.float32)
                    
                    f_arr = lead0_vals.copy()
                    f_arr[-forecast_lead_days:] = fc_1_7_vals
                else:
                    source_var = _get_source_var(union_map, f_name)
                    f_arr = reanalysis_map[source_var]
                
                normed = norm_var(scaler, f_name, f_arr)
                forecast_dict[f_name] = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

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
        'date': np.tile(dates_str, (1, 1)),
        'x_s': x_s_tensor,
        'x_d': hindcast_dict,
        'x_d_forecast': forecast_dict,
        'y': y_tensor,
        'c_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32),
        'h_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32)
    }

    return batch_data
