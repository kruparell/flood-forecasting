# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from pathlib import Path
from typing import Hashable, Iterator, Optional

import dask
import dask.array
from googlehydrology.utils.gfile_utils import get_gfile, is_strict_zarr_only
import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

SCALER_FILE_NAME = 'scaler.zarr'
LEGACY_SCALER_FILE_NAME = 'scaler.nc'


def get_scaler_param(
    scaler_ds: Optional[xr.Dataset], key: str, param_type: str
) -> float:
  """Safely retrieves scaler parameters supporting both ['center', 'scale'] and ['mean', 'std']."""
  if scaler_ds is None or key not in scaler_ds:
    return 0.0 if param_type == 'center' else 1.0
  p_coord = scaler_ds.coords.get('parameter', None)
  if p_coord is None:
    return 0.0 if param_type == 'center' else 1.0
  p_vals = [str(x) for x in p_coord.values]
  cands = ['center', 'mean'] if param_type == 'center' else ['scale', 'std']
  for p in cands:
    if p in p_vals:
      try:
        return float(scaler_ds[key].sel(parameter=p).values)
      except Exception:
        pass
  return 0.0 if param_type == 'center' else 1.0



def _calc_stats(dataset: xr.Dataset, needed: set[str]):
    stats = {
        'mean': dataset.mean(skipna=True),
        'median': (
            dataset.quantile(q=0.5, skipna=True) if 'median' in needed else None
        ),
        'min': dataset.min(skipna=True) if {'min', 'minmax'} & needed else None,
        'max': dataset.max(skipna=True) if {'max', 'minmax'} & needed else None,
    }

    stats['minmax'] = (
        stats['max'] - stats['min'] if 'minmax' in needed else None
    )

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance (Naive):
    # Var(X) = E[X**2] - E**2[X] instead of xr.std that subtracts mean from all
    # elements, to force dask work element wise. Non negative var failsafes if
    # there is propagating cancellation. For benchmark ds abs error was < 0.0001
    var = (dataset**2).mean(skipna=True) - stats['mean'] ** 2
    stats['std'] = var.clip(min=0) ** 0.5

    return stats


def _calc_types(
    dataset: xr.Dataset,
    types: dict[Hashable, str],
    none_value: float,
    stats: dict[str, xr.DataArray],
) -> Iterator[xr.DataArray]:
    """Yields pre-calculated statistics for each feature.

    Helper for building final 'center' and 'scale' params for the scaler.
    Determines needed statistic type ('mean', 'std', 'none', etc) for `types` for each feature.
    Looks up those in `stats` contains already-computed values for the feature.
    """
    for feature in dataset.data_vars:
        a_type = types[feature].lower()  # Get stat type like 'mean' for feature
        if a_type == 'none':
            yield xr.DataArray(none_value, name=feature)
        else:
            try:
                yield stats[a_type][feature]
            except KeyError:
                raise ValueError(f'Unknown method {a_type}')


class Scaler:
  """Scaler for a dataset that contains multiple features.

    Parameters
    ----------
    scaler_dir : pathlib.Path
        Directory for loading a pre-calculated scaler or saving this scaler if it is calculated.
    calculate_scaler : bool
        Flag to indicate if the scaler should be computed (the alternative is to load an existing scaler file).
    custom_normalization : dict[str, dict[str, float]]
        Feature-specific scaling instructions as a mapping from feature name to centering and/or scaling type.
        See docs for a list of accepted types and their meaning.
    dataset : xr.Dataset | None
        Dataset to use for calculating a new scaler. Cannot be supplied if `calculate_scaler` is False.

    Raises
    -------
    ValueError for incompatible loading/calculating instructions.
    """

  def __init__(
        self,
        scaler_dir: Path,
        calculate_scaler,
        custom_normalization: dict[str, dict[str, float]] = {},
        dataset: xr.Dataset | None = None,
    ):
    # Consistency check.
    if not calculate_scaler and dataset is not None:
      raise ValueError(
                'Do not pass a dataset if you are loading a pre-calculated scaler.'
            )

    # Load or calculate scaling parameters.
    self.scaler = None
    self.scaler_dir = scaler_dir
    self.loaded_from_cache = False
    self.source_zarr_path = None
    if not calculate_scaler:
      self.load()
      self.check_zero_scale()
      self.loaded_from_cache = True
    else:
      self._custom_normalization = custom_normalization
      if dataset is not None:
        self.calculate(dataset)

  def load(self):
    import time

    start_t = time.time()
    start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_t))
    scaler_dir_str = str(self.scaler_dir)
    print(
        f'[{start_str}] [SCALER LOAD START] Loading scaler cache from'
        f' {scaler_dir_str}',
        flush=True,
    )

    gf = get_gfile()
    scaler_zarr_str = f"{scaler_dir_str.rstrip('/')}/{SCALER_FILE_NAME}"
    scaler_nc_str = f"{scaler_dir_str.rstrip('/')}/{LEGACY_SCALER_FILE_NAME}"

    if scaler_dir_str.startswith('/cns/') and gf:
      if gf.Exists(scaler_zarr_str) and gf.IsDirectory(scaler_zarr_str):
        self.source_zarr_path = scaler_zarr_str
        try:
          self.scaler = xr.open_zarr(
              scaler_zarr_str, consolidated=True
          ).compute()
          fmt = 'Consolidated Zarr'
        except Exception:
          self.scaler = xr.open_zarr(scaler_zarr_str).compute()
          fmt = 'Unconsolidated Zarr'
      elif gf.Exists(scaler_nc_str):
        if is_strict_zarr_only():
          raise FileNotFoundError(
              "Strict Zarr Policy Violation: Required 'scaler.zarr' store not"
              f" found in {self.scaler_dir}. Legacy 'scaler.nc' fallback is"
              ' prohibited.'
          )
        with gf.GFile(scaler_nc_str, 'rb') as f:
          bytes_data = f.read()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
          tmp.write(bytes_data)
          tmp.flush()
          self.scaler = xr.open_dataset(tmp.name).compute()
          fmt = 'Legacy NetCDF'
      elif gf.Exists(scaler_zarr_str):
        try:
          self.scaler = xr.open_zarr(
              scaler_zarr_str, consolidated=True
          ).compute()
          fmt = 'Consolidated Zarr'
        except Exception:
          self.scaler = xr.open_zarr(scaler_zarr_str).compute()
          fmt = 'Unconsolidated Zarr'
      else:
        raise ValueError(f'Scaler file not found in {self.scaler_dir}')
    else:
      scaler_zarr = self.scaler_dir / SCALER_FILE_NAME
      scaler_nc = self.scaler_dir / LEGACY_SCALER_FILE_NAME
      if scaler_zarr.is_dir():
        self.scaler = xr.open_zarr(scaler_zarr).load()
        fmt = 'Local Zarr'
      elif scaler_nc.exists():
        if is_strict_zarr_only():
          raise FileNotFoundError(
              "Strict Zarr Policy Violation: Required 'scaler.zarr' store not"
              f" found in {self.scaler_dir}. Legacy 'scaler.nc' fallback is"
              ' prohibited.'
          )
        with open(scaler_nc, 'rb') as f:
          self.scaler = xr.load_dataset(f)
        fmt = 'Local NetCDF'
      elif scaler_zarr.exists():
        try:
          self.scaler = xr.open_zarr(scaler_zarr).load()
          fmt = 'Local Zarr'
        except Exception:
          with open(scaler_zarr, 'rb') as f:
            self.scaler = xr.load_dataset(f)
          fmt = 'Local NetCDF'
      else:
        raise ValueError(f'Scaler file not found in {self.scaler_dir}')

    elapsed = time.time() - start_t
    end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(
        f'[{end_str}] [SCALER LOAD COMPLETE] Loaded scaler cache in'
        f' {elapsed:.2f}s (Format: {fmt})',
        flush=True,
    )

    if self.scaler is not None and 'parameter' not in self.scaler.indexes:
      param_names = ['center', 'scale', 'mean', 'std']
      self.scaler = self.scaler.assign_coords(
          parameter=pd.Index(param_names, name='parameter')
      )

    if self.scaler is not None:
      for v in list(self.scaler.data_vars.keys()):
        if (
            np.issubdtype(self.scaler[v].dtype, np.floating)
            and self.scaler[v].dtype != np.float32
        ):
          self.scaler[v] = self.scaler[v].astype(np.float32)

  def calculate(
      self,
      dataset: xr.Dataset,
  ):
    # Option for custom scaling for each feature.
    centering_types = {feature: 'mean' for feature in dataset.data_vars}
    scaling_types = {feature: 'std' for feature in dataset.data_vars}
    for feature, norm in self._custom_normalization.items():
      if 'centering' in norm:
        centering_types[feature] = norm['centering']
      if 'scaling' in norm:
        scaling_types[feature] = norm['scaling']

    needed = set(centering_types.values()) | set(scaling_types.values())
    stats = _calc_stats(dataset, needed)

    # Select the appropriate center and scale statistic for each feature.
    center = xr.merge(_calc_types(dataset, centering_types, 0.0, stats))
    scale = xr.merge(_calc_types(dataset, scaling_types, 1.0, stats))

    # Combine parameters into a single xarray.Dataset with a 'parameter' coordinate.
    param_names = ['center', 'scale', 'mean', 'std']
    param_index = pd.Index(param_names, name='parameter', dtype=object)
    scaler = xr.concat(
            [center, scale, stats['mean'], stats['std']], 
            dim=param_index
        )

    # Expand the scaler dataset to include 'obs' and 'sim' versions of all variables.
    obs_scaler = scaler.rename(
            {var: f'{var}_obs' for var in scaler.data_vars}
        )
    sim_scaler = scaler.rename(
            {var: f'{var}_sim' for var in scaler.data_vars}
        )
    scaler = xr.merge([scaler, obs_scaler, sim_scaler])

    # Handle cases where part of the scaler is already calculated. Simply add new features.
    if self.scaler is not None:
      self.scaler = xr.merge([self.scaler, scaler])
    else:
      self.scaler = scaler

    if not is_any_lazy(
            self.scaler
        ):  # ensure allowing side-effects on compute
      self.scaler = self.scaler.chunk('auto')

  def save(self):
    if self.scaler is None:
      raise ValueError(
                'You are trying to save a scaler that has not been computed.'
            )
    _assert_computed(self.scaler)

    scaler_dir_str = str(self.scaler_dir)
    gf = get_gfile()
    scaler_zarr_str = f"{scaler_dir_str.rstrip('/')}/{SCALER_FILE_NAME}"
    scaler_nc_str = f"{scaler_dir_str.rstrip('/')}/{LEGACY_SCALER_FILE_NAME}"

    if self.loaded_from_cache:
      if gf and gf.Exists(scaler_zarr_str):
        print(
            '  [SCALER SAVE] Scaler was loaded from cache and already exists'
            f' at {scaler_zarr_str}. Skipping re-save.',
            flush=True,
        )
        return
      elif os.path.exists(scaler_zarr_str):
        print(
            '  [SCALER SAVE] Scaler was loaded from cache and already exists'
            f' at {scaler_zarr_str}. Skipping re-save.',
            flush=True,
        )
        return

    if scaler_dir_str.startswith('/cns/') and gf:
      gf.MakeDirs(scaler_dir_str)
      nc_bytes = self.scaler.to_netcdf()
      with gf.GFile(scaler_nc_str, 'wb') as f:
        f.write(nc_bytes)
      try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
          local_zarr = os.path.join(tmp_dir, 'scaler.zarr')
          self.scaler.to_zarr(local_zarr, mode='w', consolidated=True)
          if gf.Exists(scaler_zarr_str):
            gf.DeleteRecursively(scaler_zarr_str)
          gf.MakeDirs(scaler_zarr_str)
          for root, dirs, files in os.walk(local_zarr):
            rel = os.path.relpath(root, local_zarr)
            cns_d = (
                scaler_zarr_str
                if rel == '.'
                else os.path.join(scaler_zarr_str, rel)
            )
            if not gf.Exists(cns_d):
              gf.MakeDirs(cns_d)
            for f_name in files:
              gf.Copy(
                  os.path.join(root, f_name),
                  os.path.join(cns_d, f_name),
                  overwrite=True,
              )
      except Exception as e:
        pass
    else:
      os.makedirs(self.scaler_dir, exist_ok=True)
      scaler_file = self.scaler_dir / SCALER_FILE_NAME
      self.scaler.to_zarr(scaler_file, mode='w')
      scaler_nc = self.scaler_dir / LEGACY_SCALER_FILE_NAME
      self.scaler.to_netcdf(scaler_nc)

  def check_zero_scale(self):
    _assert_computed(self.scaler)

    scales_to_check = self.scaler.sel(parameter=['scale', 'std'])
    is_zero = (scales_to_check == 0).any('parameter').to_dataarray().compute()
    zero_mask = is_zero.values
    if zero_mask.any():
      features = list(is_zero['variable'].values[zero_mask])
      raise ValueError(f'Zero scale values found for features: {features}.')

  def _map_shifted_features(self, dataset: xr.Dataset):
    """Maps shifted features (e.g., 'streamflow_shift1') to base feature scaling parameters ('streamflow')."""
    if self.scaler is None:
      return
    for feature in dataset.data_vars:
      if feature not in self.scaler.data_vars:
        if feature.endswith('_shift1'):
          base_feat = feature[:-7]
          if base_feat in self.scaler.data_vars:
            self.scaler[feature] = self.scaler[base_feat]
        elif '_shift' in feature:
          base_feat = feature.split('_shift')[0]
          if base_feat in self.scaler.data_vars:
            self.scaler[feature] = self.scaler[base_feat]

  def scale(self, dataset: xr.Dataset) -> xr.Dataset:
    """Scale a data set with a precalculated scaler.

    $$ scaled_dataset = (dataset - center) / scale $$

    Applies a linear transformation to the features (data_vars) in an
    xr.Dataset.
    This transformation is the inverse of the one applied by self.unscale().
    Agnostic to the dimensions and coordinates of the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be scaled.

    Returns
    -------
    xr.Dataset
        The new dataset where all scalable features are scaled.

    Raises
    ------
    ValueError if the dataset contains features that are not in the scaler
    parameters.
    """
    self._map_shifted_features(dataset)
    missing_features = [
        feature for feature in dataset if feature not in self.scaler.data_vars
    ]
    if any(missing_features):
      raise ValueError(
          'Requesting to scale variables that are not part of the scaler:'
          f' {missing_features}'
      )
    return (dataset - self.scaler.sel(parameter='center')) / self.scaler.sel(
        parameter='scale'
    )

  def unscale(self, dataset: xr.Dataset) -> xr.Dataset:
    """Un-scale a data set with a precalculated scaler.

    $$ scaled_dataset = dataset * scale + center $$

    Applies a linear transformation to the features (data_vars) in an
    xr.Dataset.
    This transformation is the inverse of the one applied by self.scale().
    Agnostic to the dimensions and coordinates of the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be un-scaled.

    Returns
    -------
    xr.Dataset
        The new dataset where all scalable features are un-scaled.

    Raises
    ------
    ValueError if the dataset contains features that are not in the scaler
    parameters.
    """
    self._map_shifted_features(dataset)
    missing_features = [
        feature for feature in dataset if feature not in self.scaler.data_vars
    ]
    if any(missing_features):
      raise ValueError(
          'Requesting to unscale variables that are not part of the scaler:'
          f' {missing_features}'
      )
    return dataset * self.scaler.sel(parameter='scale') + self.scaler.sel(
            parameter='center'
        )


def is_any_lazy(dataset: xr.Dataset) -> bool:
    return any(
        isinstance(var.data, dask.array.Array)
        for var in dataset.data_vars.values()
    )

def _assert_computed(da: xr.DataArray | xr.Dataset | None):
  assert da is not None
  if getattr(da, 'chunks', None):
    if hasattr(da, 'compute'):
      da = da.compute()
  chunks = getattr(da, 'chunks', None)
  assert not chunks, f'`scaler` needs to be computed yet has {chunks=}'
