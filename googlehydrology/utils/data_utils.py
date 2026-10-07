"""Data and File I/O Utilities for Caravans & Flood Forecasting."""

import glob
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def slice_hydrology_batch(
    batch_dict: dict, slice_start_step: int, slice_end_step: int
) -> dict:
  """Fast, zero-copy slicing of 3D sequence tensors in a googlehydrology batch dict."""
  non_sequence_keys = {
      'x_s',
      'x_one_hot',
      'static_features',
      'last_prediction',
      'assimilation_overrides',
      'c_n',
      'h_n',
      'c_0',
      'h_0',
      'c_n_forecast',
      'h_n_forecast',
      'c_0_forecast',
      'h_0_forecast',
      'c_n_hindcast',
      'h_n_hindcast',
      'c_0_hindcast',
      'h_0_hindcast',
      'h_n_opt',
      'c_n_opt',
      'h_0_opt',
      'c_0_opt',
  }
  hindcast_length, forecast_length = None, None
  if 'x_d_hindcast' in batch_dict and isinstance(
      batch_dict['x_d_hindcast'], dict
  ):
    for val in batch_dict['x_d_hindcast'].values():
      if isinstance(val, torch.Tensor) and val.ndim == 3:
        hindcast_length = val.shape[1]
        break
  if 'x_d_forecast' in batch_dict and isinstance(
      batch_dict['x_d_forecast'], dict
  ):
    for val in batch_dict['x_d_forecast'].values():
      if isinstance(val, torch.Tensor) and val.ndim == 3:
        forecast_length = val.shape[1]
        break
  lead_delta = (
      (forecast_length - hindcast_length)
      if (
          hindcast_length is not None
          and forecast_length is not None
          and forecast_length > hindcast_length
      )
      else 0
  )

  result = {}
  for key, val in batch_dict.items():
    if isinstance(val, dict) and key != 'assimilation_overrides':
      if key in ('x_d_forecast', 'forecast_features'):
        result[key] = slice_hydrology_batch(
            val, slice_start_step, slice_end_step + lead_delta
        )
      else:
        result[key] = slice_hydrology_batch(
            val, slice_start_step, slice_end_step
        )
    elif (
        isinstance(val, torch.Tensor)
        and key not in non_sequence_keys
        and val.ndim == 3
    ):
      sequence_length = val.shape[1]
      slice_end_index = (
          min(slice_end_step + lead_delta, sequence_length)
          if key in ('x_d_forecast', 'y')
          else min(slice_end_step, sequence_length)
      )
      result[key] = val[
          :, min(slice_start_step, sequence_length) : slice_end_index, :
      ]
    elif (
        isinstance(val, np.ndarray) and key in ('date', 'y') and val.ndim == 2
    ):
      sequence_length = val.shape[1]
      slice_end_index = min(slice_end_step + lead_delta, sequence_length)
      result[key] = val[
          :, min(slice_start_step, sequence_length) : slice_end_index
      ]
    else:
      result[key] = val
  return result


try:
  from googlehydrology.utils.gfile_utils import get_gfile
except ImportError:

  def get_gfile():
    try:
      from google3.pyglib import gfile

      return gfile
    except ImportError:
      try:
        from pyglib import gfile

        return gfile
      except ImportError:
        return None


def load_basin_list(
    file_path: Union[str, Path],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    num_basins: Optional[int] = None,
    random_seed: int = 42,
    exclude_basins: Optional[Set[str]] = None,
    caravan_base_dir: Optional[Union[str, Path]] = None,
    static_attributes: Optional[List[str]] = None,
) -> List[str]:
  """Reads basin list CSV or TXT file, filters for complete date coverage, excludes specified basins,

  and reproducibly samples num_basins. Supports /cns/ paths inside Borg
  containers via gfile.
  """
  file_path_str = str(file_path)

  if file_path_str.startswith('/cns/'):
    gfile = get_gfile()
    if gfile and not gfile.Exists(file_path_str):
      raise FileNotFoundError(
          f'Basin list file not found on CNS: {file_path_str}'
      )
    elif not gfile and not Path(file_path_str).exists():
      raise FileNotFoundError(f'Basin list file not found: {file_path_str}')
  else:
    if not Path(file_path_str).exists():
      raise FileNotFoundError(f'Basin list file not found: {file_path_str}')

  if file_path_str.lower().endswith('.csv'):
    if file_path_str.startswith('/cns/'):
      gfile = get_gfile()
      if gfile:
        with gfile.GFile(file_path_str, 'r') as f:
          df = pd.read_csv(f)
      else:
        df = pd.read_csv(file_path_str)
    else:
      df = pd.read_csv(file_path_str)

    if (
        start_date
        and end_date
        and {'valid_start_date', 'valid_end_date'}.issubset(df.columns)
    ):
      eval_start = pd.to_datetime(start_date)
      eval_end = pd.to_datetime(end_date)
      df['valid_start_date'] = pd.to_datetime(df['valid_start_date'])
      df['valid_end_date'] = pd.to_datetime(df['valid_end_date'])

      valid_mask = (df['valid_start_date'] <= eval_start) & (
          df['valid_end_date'] >= eval_end
      )
      df_covered = df[valid_mask]
      if not df_covered.empty:
        df = df_covered
      else:
        overlap_mask = (df['valid_start_date'] <= eval_end) & (
            df['valid_end_date'] >= eval_start
        )
        if overlap_mask.any():
          df = df[overlap_mask]

    if 'basin_id' in df.columns:
      basins = df['basin_id'].dropna().astype(str).str.strip().unique().tolist()
    else:
      basins = df.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
  else:
    if file_path_str.startswith('/cns/'):
      gfile = get_gfile()
      if gfile:
        with gfile.GFile(file_path_str, 'r') as f:
          basins = [
              line.strip()
              for line in f
              if line.strip() and not line.startswith('#')
          ]
      else:
        with open(file_path_str, 'r') as f:
          basins = [
              line.strip()
              for line in f
              if line.strip() and not line.startswith('#')
          ]
    else:
      with open(file_path_str, 'r') as f:
        basins = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

  if exclude_basins:
    basins = [b for b in basins if b not in exclude_basins]

  if caravan_base_dir and static_attributes:
    attrs_df = load_caravan_attributes(caravan_base_dir)
    if not attrs_df.empty:
      avail_cols = [c for c in static_attributes if c in attrs_df.columns]
      if avail_cols:
        valid_basin_set = set(attrs_df.dropna(subset=avail_cols).index)
        before_cnt = len(basins)
        basins = [b for b in basins if b in valid_basin_set]
        print(
            f'  -> Static Attribute NaN Filter: {len(basins)} basins remaining'
            f' out of {before_cnt} (excluded {before_cnt - len(basins)} with'
            ' NaN attributes).'
        )

  if num_basins and 0 < num_basins < len(basins):
    rng = np.random.RandomState(random_seed)
    basins = sorted(list(rng.choice(basins, size=num_basins, replace=False)))
  else:
    basins = sorted(basins)

  return basins


# Priority order of candidate source columns for each standardized target attribute feature.
# High-priority columns appear first (e.g. FAO aridity is prioritized over ERA5_LAND).
CARAVAN_FEATURE_PRIORITY: Dict[str, List[str]] = {
    'aridity': ['aridity_FAO_PM', 'aridity_ERA5_LAND', 'ari_ix_sav'],
    'aridity_ERA5_LAND': ['aridity_FAO_PM', 'aridity_ERA5_LAND', 'ari_ix_sav'],
    'pet_mean': ['pet_mean_FAO_PM', 'pet_mean_ERA5_LAND', 'pet_mm_syr'],
    'pet_mean_ERA5_LAND': [
        'pet_mean_FAO_PM',
        'pet_mean_ERA5_LAND',
        'pet_mm_syr',
    ],
    'p_seasonality': [
        'seasonality_FAO_PM',
        'seasonality_ERA5_LAND',
        'cmi_ix_syr',
    ],
    'seasonality_ERA5_LAND': [
        'seasonality_FAO_PM',
        'seasonality_ERA5_LAND',
        'cmi_ix_syr',
    ],
    'moisture_index': ['moisture_index_FAO_PM', 'moisture_index_ERA5_LAND'],
    'moisture_index_ERA5_LAND': [
        'moisture_index_FAO_PM',
        'moisture_index_ERA5_LAND',
    ],
    'fraction_snow': ['fraction_snow', 'frac_snow', 'snw_pc_syr'],
    'frac_snow': ['fraction_snow', 'frac_snow', 'snw_pc_syr'],
}

# Legacy alias dict for backwards-compatibility
CARAVAN_ATTRIBUTE_ALIASES: Dict[str, str] = {
    'aridity_FAO_PM': 'aridity',
    'aridity_ERA5_LAND': 'aridity',
    'ari_ix_sav': 'aridity',
    'pet_mean_FAO_PM': 'pet_mean',
    'pet_mean_ERA5_LAND': 'pet_mean',
    'pet_mm_syr': 'pet_mean',
    'seasonality_FAO_PM': 'p_seasonality',
    'seasonality_ERA5_LAND': 'p_seasonality',
    'cmi_ix_syr': 'p_seasonality',
    'frac_snow': 'fraction_snow',
    'snw_pc_syr': 'fraction_snow',
}


def standardize_caravan_attributes(df: pd.DataFrame) -> pd.DataFrame:
  """Standardizes attribute column names across datasets using priority fallback chains.

  Handles cases where candidate columns exist but contain all or partial NaNs,
  ensuring that higher-priority sources (e.g. FAO aridity over ERA5_LAND) take
  precedence per-row whenever valid data is present.
  """
  for target_col, candidate_cols in CARAVAN_FEATURE_PRIORITY.items():
    # Find candidates that exist in df and contain at least some non-NaN data
    valid_candidates = [
        c for c in candidate_cols if c in df.columns and df[c].notna().any()
    ]
    if not valid_candidates:
      # If all candidates in df are 100% NaN, still initialize target if any candidate exists
      present_candidates = [c for c in candidate_cols if c in df.columns]
      if present_candidates and target_col not in df.columns:
        df[target_col] = df[present_candidates[0]]
      continue

    # Sequentially combine candidates in priority order (first candidate with non-null wins per-row)
    resolved = pd.to_numeric(df[valid_candidates[0]], errors='coerce')
    for next_candidate in valid_candidates[1:]:
      resolved = resolved.combine_first(
          pd.to_numeric(df[next_candidate], errors='coerce')
      )

    df[target_col] = resolved

  return df


def load_caravan_attributes(caravan_base_dir: Union[str, Path]) -> pd.DataFrame:
  """Loads and combines all CSV static attribute files across all source folders under attributes/."""
  caravan_base_dir_str = str(caravan_base_dir)
  attributes_dir = os.path.join(caravan_base_dir_str, 'attributes')
  source_dfs = []

  if caravan_base_dir_str.startswith('/cns/'):
    gfile = get_gfile()
    if gfile:
      if not gfile.Exists(attributes_dir):
        return pd.DataFrame()
      csv_by_dir: Dict[str, List[str]] = {}
      try:
        res = subprocess.run(
            ['fileutil', 'ls', attributes_dir],
            capture_output=True,
            text=True,
            check=True,
        )
        subdirs = [
            line.strip() for line in res.stdout.splitlines() if line.strip()
        ]
      except Exception:
        subdirs = []

      for sub_path in subdirs:
        try:
          res_f = subprocess.run(
              ['fileutil', 'ls', sub_path],
              capture_output=True,
              text=True,
              check=True,
          )
          files = [
              line.strip() for line in res_f.stdout.splitlines() if line.strip()
          ]
          for f in files:
            if f.endswith('.csv'):
              csv_by_dir.setdefault(sub_path, []).append(f)
        except Exception:
          pass

      for src_dir, files in csv_by_dir.items():
        src_dfs = []
        for csv_file in files:
          try:
            with gfile.GFile(csv_file, 'r') as f:
              df = pd.read_csv(f)
            if 'gauge_id' in df.columns:
              src_dfs.append(df.set_index('gauge_id'))
          except Exception:
            pass
        if src_dfs:
          df_src = pd.concat(src_dfs, axis=1)
          df_src = df_src.loc[:, ~df_src.columns.duplicated()]
          source_dfs.append(df_src)

      if not source_dfs:
        return pd.DataFrame()

      combined = pd.concat(source_dfs, axis=0)
      combined = combined.loc[:, ~combined.columns.duplicated()]
      combined = standardize_caravan_attributes(combined)
      return combined

  # Standard POSIX fallback
  caravan_base_dir_path = Path(caravan_base_dir)
  attributes_dir_path = caravan_base_dir_path / 'attributes'
  if not attributes_dir_path.exists():
    return pd.DataFrame()

  for source_folder in attributes_dir_path.iterdir():
    if not source_folder.is_dir():
      continue
    csv_files = list(source_folder.glob('*.csv'))
    if not csv_files:
      continue

    src_dfs = []
    for csv_file in csv_files:
      try:
        df = pd.read_csv(csv_file)
        if 'gauge_id' in df.columns:
          src_dfs.append(df.set_index('gauge_id'))
      except Exception:
        pass
    if src_dfs:
      df_src = pd.concat(src_dfs, axis=1)
      df_src = df_src.loc[:, ~df_src.columns.duplicated()]
      source_dfs.append(df_src)

  if not source_dfs:
    return pd.DataFrame()

  combined = pd.concat(source_dfs, axis=0)
  combined = combined.loc[:, ~combined.columns.duplicated()]
  combined = standardize_caravan_attributes(combined)
  return combined


def build_basin_nc_index(
    caravan_base_dir: Union[str, Path],
) -> Dict[str, Union[str, Path]]:
  """Scans dataset directory once to build a fast O(1) basin_id -> NetCDF file path lookup index."""
  caravan_base_dir_str = str(caravan_base_dir)
  timeseries_dir = os.path.join(caravan_base_dir_str, 'timeseries', 'netcdf')
  if caravan_base_dir_str.startswith('/cns/'):
    gfile = get_gfile()
    if gfile:
      if not gfile.Exists(timeseries_dir):
        return {}
      nc_files = gfile.Glob(os.path.join(timeseries_dir, '*', '*.nc'))
      return {os.path.splitext(os.path.basename(f))[0]: f for f in nc_files}

  timeseries_dir_path = Path(caravan_base_dir) / 'timeseries' / 'netcdf'
  if not timeseries_dir_path.exists():
    return {}

  nc_files = list(timeseries_dir_path.glob('**/*.nc'))
  return {f.stem: f for f in nc_files}


def broadcast_batch_dict(d: dict, batch_size: int) -> dict:
  """Broadcasts single-batch dictionary tensors to batch_size along dim 0."""
  res = {}
  for k, v in d.items():
    if isinstance(v, dict):
      res[k] = broadcast_batch_dict(v, batch_size)
    elif isinstance(v, torch.Tensor):
      if v.shape[0] == 1 and batch_size > 1:
        res[k] = v.expand(batch_size, *v.shape[1:]).clone()
      else:
        res[k] = v
    else:
      res[k] = v
  return res


def align_batch_shape(
    tensor_1_k_1: torch.Tensor, target_tensor: torch.Tensor
) -> torch.Tensor:
  """Reshapes a [1, K, 1] per-config tensor to align with target_tensor's batch dim."""
  K = tensor_1_k_1.shape[1]
  if target_tensor.shape[0] == K:
    return tensor_1_k_1.view(K, *([1] * (target_tensor.ndim - 1)))
  elif target_tensor.ndim >= 2 and target_tensor.shape[1] == K:
    return tensor_1_k_1.view(1, K, *([1] * (target_tensor.ndim - 2)))
  return tensor_1_k_1


def extract_state_tensor(
    state_val: Optional[torch.Tensor], t_idx: int, batch_size: int = 1
) -> Optional[torch.Tensor]:
  """Extracts state tensor at index t_idx with shape [layers, batch, hidden], broadcast if needed."""
  if state_val is None:
    return None
  if not isinstance(state_val, torch.Tensor):
    return state_val
  if state_val.ndim == 4:  # [num_layers, batch, seq_len, hidden_size]
    t_clamp = min(max(0, t_idx), state_val.shape[2] - 1)
    extracted = state_val[:, :, t_clamp, :].detach().clone()
  elif state_val.ndim == 3:  # [num_layers, batch, hidden_size]
    extracted = state_val.detach().clone()
  else:
    return state_val.detach().clone()

  if extracted.ndim >= 2 and extracted.shape[1] == 1 and batch_size > 1:
    extracted = extracted.expand(
        extracted.shape[0], batch_size, *extracted.shape[2:]
    ).clone()
  return extracted

