"""High-throughput Zarr Dataset Reader for Google Hydrology & Caravans.

Extracts static attributes and dynamic meteorological timeseries directly from
consolidated Zarr stores on CNS into contiguous memory tensors with zero NetCDF
or CSV file queries.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

try:
  from google3.pyglib import gfile
except ImportError:
  try:
    from pyglib import gfile
  except ImportError:
    gfile = None


def get_gfile():
  """Returns the gfile module for CNS filesystem operations."""
  return gfile


def safe_sel_basins(ds: xr.Dataset, requested_basins: List[str]) -> xr.Dataset:
  """Safely subsets dataset along 'basin' coordinate with case-insensitive fallback."""
  coord_name = (
      'basin'
      if 'basin' in ds.coords
      else ('gauge_id' if 'gauge_id' in ds.coords else None)
  )
  if coord_name is None:
    raise KeyError("Dataset has neither 'basin' nor 'gauge_id' coordinate.")
  available = set(ds.coords[coord_name].values)
  valid = [b for b in requested_basins if b in available]
  if not valid:
    lower_map = {str(b).lower(): b for b in available}
    valid = [
        lower_map[b.lower()] for b in requested_basins if b.lower() in lower_map
    ]
  if not valid:
    return ds.isel({coord_name: slice(0, 0)})
  return ds.sel({coord_name: valid})


def normalize_to_target_store_id(
    b_id: str, available_ids: set
) -> Optional[str]:
  """Normalizes standard Caravan ID (e.g.

  hysets_04010500) to target store gauge_id.
  """
  if b_id in available_ids:
    return b_id
  if b_id.upper() in available_ids:
    return b_id.upper()
  if b_id.lower() in available_ids:
    return b_id.lower()

  parts = str(b_id).split('_')
  if len(parts) >= 2:
    prefix = parts[0].upper()
    suffix = '_'.join(parts[1:])

    for cand in [
        f'CARAVAN_{prefix}_{suffix}',
        f'CARAVAN_{prefix}_{suffix.upper()}',
        f'{prefix}_{suffix}',
        f'USGS_{suffix}',
    ]:
      if cand in available_ids:
        return cand
  return None


class ZarrDatasetReader:
  """High-throughput reader loading consolidated Zarr archives into RAM."""

  def __init__(
      self,
      caravan_base_dir: str = '/cns/jn-d/home/floods/hydro_model/work/kruparell/Caravans_V2',
      multimet_dir: str = '/cns/jn-d/home/floods/hydro_model/datasets/external/Caravans_MultiMet',
      new_streamflow_zarr_path: Optional[
          str
      ] = '/cns/jn-d/home/floods/hydro_model/datasets/distributed_model/rivretrieve/streamflow_by_internal_ids_20260210_right_labeled.zarr',
      forecast_product: str = 'HRES',
      use_union_mapping: bool = False,
  ):
    self.caravan_base_dir = Path(caravan_base_dir)
    self.multimet_dir = Path(multimet_dir)
    self.new_streamflow_zarr_path = new_streamflow_zarr_path
    self.forecast_product = forecast_product.upper()
    self.use_union_mapping = use_union_mapping

    # Paths to consolidated Zarr archives
    sf_path = self.caravan_base_dir / 'streamflow.zarr'
    gf = get_gfile()
    if (
        str(sf_path).startswith('/cns/') and gf and gf.Exists(str(sf_path))
    ) or os.path.exists(str(sf_path)):
      self.targets_zarr_path = str(sf_path)
    else:
      self.targets_zarr_path = str(
          self.caravan_base_dir / 'streamflow_targets.zarr'
      )

    self.attributes_zarr_path = str(self.caravan_base_dir / 'attributes.zarr')

    self.hres_zarr_path = str(self.multimet_dir / 'HRES' / 'timeseries.zarr')
    self.graphcast_zarr_path = str(
        self.multimet_dir / 'GRAPHCAST' / 'timeseries.zarr'
    )
    self.imerg_zarr_path = str(self.multimet_dir / 'IMERG' / 'timeseries.zarr')
    self.cpc_zarr_path = str(self.multimet_dir / 'CPC' / 'timeseries.zarr')
    self.era5_zarr_path = str(
        self.multimet_dir / 'ERA5_LAND' / 'timeseries.zarr'
    )

    prod_name = 'HRES' if self.forecast_product == 'HRES' else 'GRAPHCAST'
    self.forecast_zarr_path = str(
        self.multimet_dir / prod_name / 'timeseries.zarr'
    )

    # Lazy dataset handles
    self._targets_ds: Optional[xr.Dataset] = None
    self._new_targets_ds: Optional[xr.Dataset] = None
    self._attributes_ds: Optional[xr.Dataset] = None
    self._forecast_ds: Optional[xr.Dataset] = None
    self._hres_ds: Optional[xr.Dataset] = None
    self._graphcast_ds: Optional[xr.Dataset] = None
    self._imerg_ds: Optional[xr.Dataset] = None
    self._cpc_ds: Optional[xr.Dataset] = None
    self._era5_ds: Optional[xr.Dataset] = None

  def _open_zarr_safe(
      self, path_str: str, provider_name: Optional[str] = None
  ) -> Optional[xr.Dataset]:
    """Safely opens a consolidated Zarr store on CNS or local POSIX."""
    if not path_str:
      return None
    gf = get_gfile()
    if path_str.startswith('/cns/'):
      if gf and not gf.Exists(path_str):
        name_tag = f' ({provider_name})' if provider_name else ''
        print(
            f'[DATASET LOAD WARNING] Path does not exist{name_tag}: {path_str}',
            flush=True,
        )
        return None
    elif not os.path.exists(path_str):
      name_tag = f' ({provider_name})' if provider_name else ''
      print(
          f'[DATASET LOAD WARNING] Path does not exist{name_tag}: {path_str}',
          flush=True,
      )
      return None

    import time

    start_t = time.time()
    start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_t))
    name_tag = f' [{provider_name}]' if provider_name else ''
    print(
        f'[{start_str}] [DATASET LOAD START]{name_tag} Opening Zarr store:'
        f' {path_str}',
        flush=True,
    )

    try:
      ds = xr.open_zarr(path_str, consolidated=True, decode_timedelta=False)
      elapsed = time.time() - start_t
      end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
      basins_cnt = len(ds.coords.get('basin', ds.coords.get('gauge_id', [])))
      print(
          f'[{end_str}] [DATASET LOAD COMPLETE]{name_tag} Successfully loaded'
          f' Zarr store: {path_str} in {elapsed:.2f}s (Format: Consolidated'
          f' Zarr, Basins: {basins_cnt})',
          flush=True,
      )
      return ds
    except Exception:
      try:
        ds = xr.open_zarr(path_str, decode_timedelta=False)
        elapsed = time.time() - start_t
        end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        basins_cnt = len(ds.coords.get('basin', ds.coords.get('gauge_id', [])))
        print(
            f'[{end_str}] [DATASET LOAD COMPLETE]{name_tag} Successfully loaded'
            f' Zarr store: {path_str} in {elapsed:.2f}s (Format:'
            f' Unconsolidated Zarr, Basins: {basins_cnt})',
            flush=True,
        )
        return ds
      except Exception as e:
        print(
            f'[DATASET LOAD ERROR]{name_tag} Failed to open Zarr store at'
            f' {path_str}: {e}',
            flush=True,
        )
        return None

  @property
  def targets_ds(self) -> xr.Dataset:
    if self._targets_ds is None:
      self._targets_ds = self._open_zarr_safe(
          self.targets_zarr_path, 'Targets Streamflow'
      )
      if self._targets_ds is None:
        raise FileNotFoundError(
            f'Target streamflow Zarr not found: {self.targets_zarr_path}'
        )
    return self._targets_ds

  @property
  def new_targets_ds(self) -> Optional[xr.Dataset]:
    if self._new_targets_ds is None and self.new_streamflow_zarr_path:
      self._new_targets_ds = self._open_zarr_safe(
          self.new_streamflow_zarr_path, 'New Streamflow 2026'
      )
    return self._new_targets_ds

  @property
  def attributes_ds(self) -> xr.Dataset:
    if self._attributes_ds is None:
      self._attributes_ds = self._open_zarr_safe(
          self.attributes_zarr_path, 'Attributes'
      )
      if self._attributes_ds is None:
        raise FileNotFoundError(
            f'Attributes Zarr not found: {self.attributes_zarr_path}'
        )
    return self._attributes_ds

  @property
  def forecast_ds(self) -> xr.Dataset:
    if self._forecast_ds is None:
      self._forecast_ds = self._open_zarr_safe(
          self.forecast_zarr_path, self.forecast_product
      )
      if self._forecast_ds is None:
        raise FileNotFoundError(
            f'Forecast Zarr not found: {self.forecast_zarr_path}'
        )
    return self._forecast_ds

  @property
  def hres_ds(self) -> Optional[xr.Dataset]:
    if self._hres_ds is None:
      self._hres_ds = self._open_zarr_safe(self.hres_zarr_path, 'HRES')
    return self._hres_ds

  @property
  def graphcast_ds(self) -> Optional[xr.Dataset]:
    if self._graphcast_ds is None:
      self._graphcast_ds = self._open_zarr_safe(
          self.graphcast_zarr_path, 'GRAPHCAST'
      )
    return self._graphcast_ds

  @property
  def imerg_ds(self) -> Optional[xr.Dataset]:
    if self._imerg_ds is None:
      self._imerg_ds = self._open_zarr_safe(self.imerg_zarr_path, 'IMERG')
    return self._imerg_ds

  @property
  def cpc_ds(self) -> Optional[xr.Dataset]:
    if self._cpc_ds is None:
      self._cpc_ds = self._open_zarr_safe(self.cpc_zarr_path, 'CPC')
    return self._cpc_ds

  @property
  def era5_ds(self) -> Optional[xr.Dataset]:
    if self._era5_ds is None:
      self._era5_ds = self._open_zarr_safe(self.era5_zarr_path, 'ERA5_LAND')
    return self._era5_ds

  def load_static_attributes(
      self, basin_ids: List[str], attr_names: List[str]
  ) -> pd.DataFrame:
    """Extracts static attributes DataFrame from attributes.zarr for requested basins."""
    sub_ds = safe_sel_basins(self.attributes_ds, basin_ids)
    available_vars = [v for v in attr_names if v in sub_ds.data_vars]

    df = sub_ds[available_vars].to_dataframe()
    if 'basin' in df.index.names:
      df = df.reset_index().set_index('basin')
    elif 'gauge_id' in df.index.names:
      df = df.reset_index().set_index('gauge_id')

    # Reorder/reindex to match requested basins and attributes
    df = df.reindex(index=basin_ids, columns=attr_names)
    return df

  def load_shard_timeseries_tensors(
      self,
      basin_ids: List[str],
      start_date: str,
      end_date: str,
      hindcast_window_days: int = 365,
      forecast_lead_days: int = 7,
  ) -> Dict[str, Dict[str, Any]]:
    """Batch-extracts continuous timeseries arrays for all basins in the shard into RAM.

    Uses the new 2026 streamflow target store with seamless fallback to Caravans
    V2.
    Tracks full per-basin filtering diagnostics in self.last_filtering_report.
    """
    import logging

    t0_start = pd.to_datetime(start_date)
    t0_end = pd.to_datetime(end_date)

    expanded_start = (
        t0_start - pd.Timedelta(days=hindcast_window_days)
    ).strftime('%Y-%m-%d')
    targets_end = t0_end.strftime('%Y-%m-%d')
    fc_expanded_end = (t0_end + pd.Timedelta(days=forecast_lead_days)).strftime(
        '%Y-%m-%d'
    )

    # -------------------------------------------------------------------------
    # STAGE 1: Check Coordinate Overlap with Zarr Stores
    # -------------------------------------------------------------------------
    fc_coords = self.forecast_ds.coords.get(
        'basin', self.forecast_ds.coords.get('gauge_id', [])
    )
    avail_fc = (
        set(fc_coords.values)
        if hasattr(fc_coords, 'values')
        else set(fc_coords)
    )

    # Basins present in HRES
    valid_basins = [b for b in basin_ids if b in avail_fc]
    if not valid_basins:
      lower_fc = {str(b).lower(): b for b in avail_fc}
      valid_basins = [b for b in basin_ids if b.lower() in lower_fc]

    missing_in_hres = [b for b in basin_ids if b not in valid_basins]

    logging.info(
        f'[COORDINATE CHECK] Requested: {len(basin_ids)} basins. '
        f'Present in HRES: {len(valid_basins)}/{len(basin_ids)}.'
    )
    if missing_in_hres:
      logging.info(
          f'[COORDINATE CHECK] Missing in HRES ({len(missing_in_hres)} basins):'
          f' sample={missing_in_hres[:5]}'
      )

    filtering_records = []
    for b in missing_in_hres:
      filtering_records.append({
          'basin_id': b,
          'status': 'FILTERED_MISSING_HRES',
          'reason': 'Not found in MultiMet HRES timeseries.zarr',
          'nan_count': -1,
          'total_days': -1,
          'pct_nan': -1.0,
          'first_valid_date': 'NONE',
          'last_valid_date': 'NONE',
          'first_nan_date': 'NONE',
          'last_nan_date': 'NONE',
      })

    if not valid_basins:
      logging.warning(
          f'None of the {len(basin_ids)} requested shard basins found in HRES'
          ' Zarr.'
      )
      self.last_filtering_report = pd.DataFrame(filtering_records)
      return {}

    fc_issue_start = (
        pd.to_datetime(expanded_start) - pd.Timedelta(days=1)
    ).strftime('%Y-%m-%d')

    # Subset HRES forecast dataset
    sub_fc = (
        safe_sel_basins(self.forecast_ds, valid_basins)
        .sel(date=slice(fc_issue_start, fc_expanded_end))
        .load()
    )

    # Date index alignment from forecast dataset (expanded_start to fc_expanded_end)
    all_dates_str = np.array([str(d)[:10] for d in sub_fc['date'].values])
    mask_target_dates = (all_dates_str >= expanded_start) & (
        all_dates_str <= fc_expanded_end
    )
    eval_dates_str = all_dates_str[mask_target_dates]
    shifted_eval_dates_str = (
        pd.to_datetime(eval_dates_str) - pd.Timedelta(days=1)
    ).strftime('%Y-%m-%d').values
    date_to_idx = {d: i for i, d in enumerate(eval_dates_str)}
    total_days = len(eval_dates_str)

    logging.info(
        f'[DATE RANGE CHECK] Evaluation window from {eval_dates_str[0]} to'
        f' {eval_dates_str[-1]} ({total_days} days total).'
    )

    # Check availability in new streamflow store vs Caravans V2
    avail_new_gids = (
        set(self.new_targets_ds.coords['gauge_id'].values)
        if self.new_targets_ds is not None
        else set()
    )

    # Load V2 targets slice as fallback
    v2_valid = [
        b for b in valid_basins if b in self.targets_ds.coords.get('basin', [])
    ]
    sub_v2_targets = (
        safe_sel_basins(self.targets_ds, v2_valid)
        .sel(date=slice(expanded_start, fc_expanded_end))
        .load()
        if v2_valid
        else None
    )
    v2_streamflow_var = (
        'streamflow'
        if sub_v2_targets is not None and 'streamflow' in sub_v2_targets
        else (
            list(sub_v2_targets.data_vars.keys())[0]
            if sub_v2_targets is not None
            else None
        )
    )

    # Pre-slice and batch load external multimet datasets for all shard basins at once
    sub_gc = None
    if self.graphcast_ds is not None:
      gc_valid = [
          b for b in valid_basins if b in self.graphcast_ds.coords.get('basin', [])
      ]
      if gc_valid:
        try:
          sub_gc = (
              safe_sel_basins(self.graphcast_ds, gc_valid)
              .sel(date=slice(fc_issue_start, fc_expanded_end))
              .load()
          )
        except Exception as e:
          logging.warning(f'Could not batch load GraphCast: {e}')

    sub_im = None
    if self.imerg_ds is not None:
      im_valid = [
          b for b in valid_basins if b in self.imerg_ds.coords.get('basin', [])
      ]
      if im_valid:
        try:
          sub_im = (
              safe_sel_basins(self.imerg_ds, im_valid)
              .sel(date=slice(expanded_start, fc_expanded_end))
              .load()
          )
        except Exception as e:
          logging.warning(f'Could not batch load IMERG: {e}')

    sub_cpc = None
    if self.cpc_ds is not None:
      cpc_valid = [
          b for b in valid_basins if b in self.cpc_ds.coords.get('basin', [])
      ]
      if cpc_valid:
        try:
          sub_cpc = (
              safe_sel_basins(self.cpc_ds, cpc_valid)
              .sel(date=slice(expanded_start, fc_expanded_end))
              .load()
          )
        except Exception as e:
          logging.warning(f'Could not batch load CPC: {e}')

    sub_era5 = None
    if self.use_union_mapping and self.era5_ds is not None:
      era5_valid = [
          b for b in valid_basins if b in self.era5_ds.coords.get('basin', [])
      ]
      if era5_valid:
        try:
          sub_era5 = (
              safe_sel_basins(self.era5_ds, era5_valid)
              .sel(date=slice(fc_issue_start, fc_expanded_end))
              .load()
          )
        except Exception as e:
          logging.warning(f'Could not batch load ERA5: {e}')

    fc_var_map = {
        'hres_total_precipitation': (
            'hres_total_precipitation'
            if 'hres_total_precipitation' in sub_fc
            else (
                'total_precipitation'
                if 'total_precipitation' in sub_fc
                else None
            )
        ),
        'hres_temperature_2m': (
            'hres_temperature_2m'
            if 'hres_temperature_2m' in sub_fc
            else ('temperature_2m' if 'temperature_2m' in sub_fc else None)
        ),
        'hres_surface_net_solar_radiation': (
            'hres_surface_net_solar_radiation'
            if 'hres_surface_net_solar_radiation' in sub_fc
            else (
                'surface_net_solar_radiation'
                if 'surface_net_solar_radiation' in sub_fc
                else None
            )
        ),
        'hres_surface_net_thermal_radiation': (
            'hres_surface_net_thermal_radiation'
            if 'hres_surface_net_thermal_radiation' in sub_fc
            else (
                'surface_net_thermal_radiation'
                if 'surface_net_thermal_radiation' in sub_fc
                else None
            )
        ),
        'hres_surface_pressure': (
            'hres_surface_pressure'
            if 'hres_surface_pressure' in sub_fc
            else ('surface_pressure' if 'surface_pressure' in sub_fc else None)
        ),
    }

    shard_data = {}
    filtered_out_details = []

    # -------------------------------------------------------------------------
    # STAGE 2: Extract Streamflow Targets from New Store (with V2 Fallback)
    # -------------------------------------------------------------------------
    # Pre-extract basin areas for unit conversion (m3/s -> mm/day)
    basin_areas = {}
    if self.attributes_ds is not None:
      area_var = None
      for cand in ['area', 'area_calc', 'area_hydroatlas', 'drainage_area']:
        if cand in self.attributes_ds.data_vars:
          area_var = cand
          break
      if area_var:
        try:
          sub_attr = safe_sel_basins(self.attributes_ds, valid_basins)
          b_coord = (
              'basin'
              if 'basin' in sub_attr.coords
              else ('gauge_id' if 'gauge_id' in sub_attr.coords else None)
          )
          if b_coord:
            for b, a_val in zip(
                sub_attr.coords[b_coord].values, sub_attr[area_var].values
            ):
              try:
                basin_areas[str(b)] = float(a_val)
              except Exception:
                pass
        except Exception as e:
          logging.warning(f'Could not extract basin areas: {e}')

    # Load CAMELS ground-truth topography areas (area_gages2) to fix coarse polygon inflation (area_geospa_fabric)
    camels_true_areas = {}
    camels_old_areas = {}
    camels_topo_candidates = [
        '/cns/jn-d/home/floods/hydro_model/datasets/external/CAMELS/camels_attributes_v2.0/camels_attributes_v2.0/camels_topo.txt',
        '/google/src/cloud/kruparell/googlehydrology_rebased/google3/third_party/py/googlehydrology/test/test_data/camels_us/camels_attributes_v2.0/camels_topo.txt',
    ]
    topo_path = next((p for p in camels_topo_candidates if (gfile and gfile.Exists(p)) or os.path.exists(p)), None)
    if topo_path:
      try:
        open_fn = gfile.GFile if (gfile and topo_path.startswith('/cns/')) else open
        with open_fn(topo_path, 'r') as f:
          df_topo = pd.read_csv(f, sep=';', dtype={'gauge_id': str})
          for _, r in df_topo.iterrows():
            gid = str(r['gauge_id']).zfill(8)
            a_true = float(r['area_gages2'])
            a_geospa = float(r['area_geospa_fabric'])
            if a_true > 0:
              for k in [f'camels_{gid}', f'CAMELS_{gid}', f'USGS_{gid}', gid]:
                camels_true_areas[k] = a_true
                camels_old_areas[k] = a_geospa
                basin_areas[k] = a_true
      except Exception as e:
        logging.warning(f'Could not load CAMELS topo ground truth: {e}')

    is_right_labeled = (
        self.new_streamflow_zarr_path is not None
        and 'right_labeled' in str(self.new_streamflow_zarr_path).lower()
    )
    targets_slice_end = (
        (pd.to_datetime(fc_expanded_end) + pd.Timedelta(days=1)).strftime(
            '%Y-%m-%d'
        )
        if is_right_labeled
        else fc_expanded_end
    )

    for b_id in valid_basins:
      norm_id = normalize_to_target_store_id(b_id, avail_new_gids)
      obs_series = None

      if norm_id and self.new_targets_ds is not None:
        try:
          s_raw = self.new_targets_ds['streamflow'].sel(
              gauge_id=norm_id, time=slice(expanded_start, targets_slice_end)
          )
          if is_right_labeled:
            raw_times = [
                (pd.to_datetime(t) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                for t in s_raw['time'].values
            ]
          else:
            raw_times = [str(t)[:10] for t in s_raw['time'].values]
          raw_vals = s_raw.values.astype(np.float32)

          # Convert raw volumetric discharge (m3/s / CMS) to specific discharge (mm/day):
          # q [mm/day] = (Q [m3/s] * 86.4) / Area [km2]
          b_area = basin_areas.get(
              b_id, basin_areas.get(norm_id, basin_areas.get(str(b_id).lower()))
          )
          if b_area is not None and b_area > 0:
            raw_vals = (raw_vals * 86.4) / b_area

          time_map = dict(zip(raw_times, raw_vals))
          obs_series = np.array(
              [time_map.get(d, np.nan) for d in eval_dates_str],
              dtype=np.float32,
          )
        except Exception:
          obs_series = None

      # Fallback to Caravans V2 if needed (already in mm/day)
      if obs_series is None or np.isnan(obs_series).all():
        if (
            sub_v2_targets is not None
            and b_id in sub_v2_targets.coords['basin'].values
        ):
          try:
            v2_raw = sub_v2_targets[v2_streamflow_var].sel(basin=b_id)
            v2_dates = [str(d)[:10] for d in v2_raw['date'].values]
            v2_vals = v2_raw.values.astype(np.float32)

            # Correct CAMELS-US Caravan V2 normalization if inflated by area_geospa_fabric
            old_a = camels_old_areas.get(b_id, camels_old_areas.get(str(b_id).lower()))
            true_a = camels_true_areas.get(b_id, camels_true_areas.get(str(b_id).lower()))
            if (
                old_a is not None
                and true_a is not None
                and old_a > 0
                and true_a > 0
                and abs(old_a - true_a) > 1e-3
            ):
              v2_vals = v2_vals * (old_a / true_a)

            time_map = dict(zip(v2_dates, v2_vals))
            obs_series = np.array(
                [time_map.get(d, np.nan) for d in eval_dates_str],
                dtype=np.float32,
            )
          except Exception:
            obs_series = None

      if obs_series is None:
        obs_series = np.full((total_days,), np.nan, dtype=np.float32)


      is_nan_mask = np.isnan(obs_series)

      # Skip only if basin has literally ZERO valid observations
      if is_nan_mask.all():
        filtering_records.append({
            'basin_id': b_id,
            'status': 'FILTERED_NO_OBSERVATIONS',
            'reason': (
                'Contains 0 valid streamflow observations across entire window'
            ),
            'nan_count': total_days,
            'total_days': total_days,
            'pct_nan': 100.0,
            'first_valid_date': 'NONE',
            'last_valid_date': 'NONE',
            'first_nan_date': eval_dates_str[0],
            'last_nan_date': eval_dates_str[-1],
        })
        filtered_out_details.append(b_id)
        continue

      nan_count = int(is_nan_mask.sum())
      valid_dates = eval_dates_str[~is_nan_mask]
      pct_missing = (nan_count / total_days) * 100
      first_valid = valid_dates[0]
      last_valid = valid_dates[-1]

      source_label = (
          f'New 2026 Store ({norm_id})'
          if (norm_id and not is_nan_mask.all())
          else 'Caravans V2'
      )

      filtering_records.append({
          'basin_id': b_id,
          'status': 'RETAINED',
          'reason': (
              f'Source: {source_label}, {total_days - nan_count}/{total_days}'
              f' valid days ({pct_missing:.1f}% NaNs). Valid: [{first_valid} to'
              f' {last_valid}]'
          ),
          'nan_count': nan_count,
          'total_days': total_days,
          'pct_nan': pct_missing,
          'first_valid_date': first_valid,
          'last_valid_date': last_valid,
          'first_nan_date': (
              eval_dates_str[is_nan_mask][0] if nan_count > 0 else 'NONE'
          ),
          'last_nan_date': (
              eval_dates_str[is_nan_mask][-1] if nan_count > 0 else 'NONE'
          ),
      })

      # Extract features for retained basin
      fc_lead0_feats = {}
      fc_1_7_feats = {}
      hindcast_feats = {}

      # 1. HRES Features (3D: basin, date, lead_time)
      for std_name, raw_var in fc_var_map.items():
        if raw_var is not None and raw_var in sub_fc:
          fc_slice = sub_fc[raw_var].sel(basin=b_id)
          lead0_arr = (
              fc_slice.sel(date=shifted_eval_dates_str)
              .isel(lead_time=0)
              .values.astype(np.float32)
          )
          fc_lead0_feats[std_name] = lead0_arr
          fc_1_7_feats[std_name] = (
              fc_slice.sel(date=eval_dates_str)
              .isel(lead_time=slice(0, forecast_lead_days))
              .values.astype(np.float32)
          )

          hindcast_feats[std_name] = lead0_arr

      # 2. GraphCast Features (3D: basin, date, lead_time)
      if sub_gc is not None and b_id in sub_gc.coords.get('basin', []):
        try:
          b_gc = sub_gc.sel(basin=b_id)
          for gc_std, gc_var in [
              ('graphcast_temperature_2m', 'temperature_2m'),
              ('graphcast_total_precipitation', 'total_precipitation'),
          ]:
            if gc_var in b_gc:
              gc_slice = b_gc[gc_var]
              gc_l0 = (
                  gc_slice.sel(date=shifted_eval_dates_str)
                  .isel(lead_time=0)
                  .values.astype(np.float32)
              )
              gc_1_7 = (
                  gc_slice.sel(date=eval_dates_str)
                  .isel(lead_time=slice(0, forecast_lead_days))
                  .values.astype(np.float32)
              )
              fc_lead0_feats[gc_std] = gc_l0
              fc_1_7_feats[gc_std] = gc_1_7
              hindcast_feats[gc_std] = gc_l0
        except Exception as e:
          logging.warning(
              f'Could not extract GraphCast for basin {b_id}: {e}'
          )

      # 3. IMERG Features (2D: basin, date)
      if sub_im is not None and b_id in sub_im.coords.get('basin', []):
        try:
          b_im = sub_im.sel(basin=b_id)
          im_var = (
              'precipitation'
              if 'precipitation' in b_im
              else list(b_im.data_vars.keys())[0]
          )
          im_dates = [str(d)[:10] for d in b_im['date'].values]
          im_vals = b_im[im_var].values.astype(np.float32)
          im_map = dict(zip(im_dates, im_vals))
          im_l0 = np.array(
              [im_map.get(d, np.nan) for d in eval_dates_str], dtype=np.float32
          )
          fc_lead0_feats['imerg_precipitation'] = im_l0
          hindcast_feats['imerg_precipitation'] = im_l0
          fc_1_7_feats['imerg_precipitation'] = np.full(
              (total_days, forecast_lead_days), np.nan, dtype=np.float32
          )
        except Exception as e:
          logging.warning(f'Could not extract IMERG for basin {b_id}: {e}')

      # 4. CPC Features (2D: basin, date)
      if sub_cpc is not None and b_id in sub_cpc.coords.get('basin', []):
        try:
          b_cpc = sub_cpc.sel(basin=b_id)
          cpc_var = (
              'precipitation'
              if 'precipitation' in b_cpc
              else list(b_cpc.data_vars.keys())[0]
          )
          cpc_dates = [str(d)[:10] for d in b_cpc['date'].values]
          cpc_vals = b_cpc[cpc_var].values.astype(np.float32)
          cpc_map = dict(zip(cpc_dates, cpc_vals))
          cpc_l0 = np.array(
              [cpc_map.get(d, np.nan) for d in eval_dates_str], dtype=np.float32
          )
          fc_lead0_feats['cpc_precipitation'] = cpc_l0
          hindcast_feats['cpc_precipitation'] = cpc_l0
          fc_1_7_feats['cpc_precipitation'] = np.full(
              (total_days, forecast_lead_days), np.nan, dtype=np.float32
          )
        except Exception as e:
          logging.warning(f'Could not extract CPC for basin {b_id}: {e}')

      # 5. Union Mapping Backfilling (if requested and ERA5-Land is available)
      if sub_era5 is not None and b_id in sub_era5.coords.get('basin', []):
        try:
          b_era5 = sub_era5.sel(basin=b_id)
          era5_dates = [str(d)[:10] for d in b_era5['date'].values]
          era5_date_map = {d: i for i, d in enumerate(era5_dates)}

          era5_var_map = {
              'total_precipitation': (
                  'era5land_total_precipitation'
                  if 'era5land_total_precipitation' in b_era5
                  else (
                      'total_precipitation'
                      if 'total_precipitation' in b_era5
                      else None
                  )
              ),
              'temperature_2m': (
                  'era5land_temperature_2m'
                  if 'era5land_temperature_2m' in b_era5
                  else ('temperature_2m' if 'temperature_2m' in b_era5 else None)
              ),
              'surface_net_solar_radiation': (
                  'era5land_surface_net_solar_radiation'
                  if 'era5land_surface_net_solar_radiation' in b_era5
                  else (
                      'surface_net_solar_radiation'
                      if 'surface_net_solar_radiation' in b_era5
                      else None
                  )
              ),
              'surface_net_thermal_radiation': (
                  'era5land_surface_net_thermal_radiation'
                  if 'era5land_surface_net_thermal_radiation' in b_era5
                  else (
                      'surface_net_thermal_radiation'
                      if 'surface_net_thermal_radiation' in b_era5
                      else None
                  )
              ),
              'surface_pressure': (
                  'era5land_surface_pressure'
                  if 'era5land_surface_pressure' in b_era5
                  else (
                      'surface_pressure'
                      if 'surface_pressure' in b_era5
                      else None
                  )
              ),
          }

          union_specs = [
              ('hres_total_precipitation', 'total_precipitation'),
              ('hres_temperature_2m', 'temperature_2m'),
              ('hres_surface_net_solar_radiation', 'surface_net_solar_radiation'),
              ('hres_surface_net_thermal_radiation', 'surface_net_thermal_radiation'),
              ('hres_surface_pressure', 'surface_pressure'),
              ('cpc_precipitation', 'total_precipitation'),
              ('imerg_precipitation', 'total_precipitation'),
              ('graphcast_total_precipitation', 'total_precipitation'),
              ('graphcast_temperature_2m', 'temperature_2m'),
          ]

          lead_time_forecast_features = {
              'hres_total_precipitation',
              'hres_temperature_2m',
              'hres_surface_net_solar_radiation',
              'hres_surface_net_thermal_radiation',
              'hres_surface_pressure',
              'graphcast_total_precipitation',
              'graphcast_temperature_2m',
          }

          for f_name, var_type in union_specs:
            ev = era5_var_map.get(var_type)
            if ev and ev in b_era5:
              e_vals = b_era5[ev].values.astype(np.float32)
              target_dates = (
                  shifted_eval_dates_str
                  if f_name in lead_time_forecast_features
                  else eval_dates_str
              )
              e_aligned_l0 = np.array(
                  [
                      e_vals[era5_date_map[d]]
                      if d in era5_date_map
                      else np.nan
                      for d in target_dates
                  ],
                  dtype=np.float32,
              )
              # In union mapping mode, replace feature with ERA5-Land reanalysis
              fc_lead0_feats[f_name] = e_aligned_l0
              hindcast_feats[f_name] = e_aligned_l0

              # Map Lead 1..7 (ERA5 reanalysis shifted by lead time)
              f_1_7 = np.zeros(
                  (total_days, forecast_lead_days), dtype=np.float32
              )
              for L_idx in range(forecast_lead_days):
                L_shift = L_idx + 1
                f_1_7[:, L_idx] = np.array(
                    [
                        e_vals[
                            era5_date_map[
                                (
                                    pd.to_datetime(d)
                                    + pd.Timedelta(days=L_shift)
                                ).strftime('%Y-%m-%d')
                            ]
                        ]
                        if (
                            pd.to_datetime(d) + pd.Timedelta(days=L_shift)
                        ).strftime('%Y-%m-%d')
                        in era5_date_map
                        else np.nan
                        for d in eval_dates_str
                    ],
                    dtype=np.float32,
                )
              fc_1_7_feats[f_name] = f_1_7
        except Exception as e:
          logging.warning(
              f'Could not apply ERA5 union mapping for basin {b_id}: {e}'
          )

      shard_data[b_id] = {
          'dates': eval_dates_str,
          'date_to_idx': date_to_idx,
          'y_obs': obs_series,
          'hindcast_features': hindcast_feats,
          'forecast_lead0_features': fc_lead0_feats,
          'forecast_1_7_features': fc_1_7_feats,
      }

    # -------------------------------------------------------------------------
    # STAGE 3: Final Shard Filtering Scorecard
    # -------------------------------------------------------------------------
    logging.info(
        '======================================================================\n'
        '  SHARD FILTERING SUMMARY\n'
        f'  - Total Assigned:           {len(basin_ids)}\n'
        f'  - Missing in HRES:          {len(missing_in_hres)}\n'
        f'  - Retained for DA:          {len(shard_data)}\n'
        f'  - Union Mapping Mode:       {self.use_union_mapping}\n'
        '======================================================================'
    )

    self.last_filtering_report = pd.DataFrame(filtering_records)
    return shard_data

  def load_single_basin_tensors(
      self, basin_id: str, start_date: str, end_date: str
  ) -> Optional[Dict[str, Any]]:
    """Loads continuous timeseries arrays for a single basin into RAM."""
    res = self.load_shard_timeseries_tensors([basin_id], start_date, end_date)
    return res.get(basin_id, None)

  def close(self) -> None:
    """Closes open Xarray datasets to free resources."""
    for attr in [
        '_targets_ds',
        '_new_targets_ds',
        '_attributes_ds',
        '_forecast_ds',
        '_hres_ds',
        '_graphcast_ds',
        '_imerg_ds',
        '_cpc_ds',
        '_era5_ds',
    ]:
      ds = getattr(self, attr, None)
      if ds is not None:
        try:
          ds.close()
        except Exception:
          pass
        setattr(self, attr, None)
