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

from collections.abc import Iterable, Iterator
import functools
import itertools
import logging
import math
import os
from pathlib import Path

import dask
import dask.dataframe as dd
import dask.delayed
from googlehydrology.datautils.utils import safe_sel_basins
from googlehydrology.utils.gfile_utils import (
    GFileZarrStore,
    get_gfile,
    gfile_exists,
    gfile_open,
    is_cns_path,
)
from googlehydrology.utils.tqdm import AutoRefreshTqdm as tqdm
import numpy as np
import pandas as pd
import xarray

LOGGER = logging.getLogger(__name__)

# Standard synonym/alias mapping for static catchment attributes (e.g. CAMELS/HydroATLAS/Caravan).
STATIC_ATTRIBUTE_ALIASES: dict[str, str] = {
    # Topography / Elevation
    'elevation_mean': 'ele_mt_sav',
    'elev_mean': 'ele_mt_sav',
    'mean_elevation': 'ele_mt_sav',
    'elevation_min': 'ele_mt_smn',
    'elev_min': 'ele_mt_smn',
    'min_elevation': 'ele_mt_smn',
    'elevation_max': 'ele_mt_smx',
    'elev_max': 'ele_mt_smx',
    'max_elevation': 'ele_mt_smx',
    'elevation': 'ele_mt_sav',
    'slope_mean': 'slp_dg_sav',
    'mean_slope': 'slp_dg_sav',
    'slope': 'slp_dg_sav',
    # Climate & Hydrology
    'aridity': 'ari_ix_sav',
    'aridity_index': 'ari_ix_sav',
    'pet_mean': 'pet_mm_syr',
    'pet_annual': 'pet_mm_syr',
    'precip_annual': 'pre_mm_syr',
    'precipitation_annual': 'pre_mm_syr',
    'aet_annual': 'aet_mm_syr',
    'climate_moisture_index': 'cmi_ix_syr',
    'snow_fraction': 'snw_pc_syr',
    'snow_percent': 'snw_pc_syr',
    'frac_snow': 'frac_snow',
    'p_seasonality': 'p_seasonality',
    # Land Cover & Vegetation
    'forest_fraction': 'for_pc_sse',
    'forest_percent': 'for_pc_sse',
    'forest_frac': 'for_pc_sse',
    'crop_fraction': 'crp_pc_sse',
    'crop_percent': 'crp_pc_sse',
    'crop_frac': 'crp_pc_sse',
    'urban_fraction': 'urb_pc_sse',
    'urban_percent': 'urb_pc_sse',
    'urban_frac': 'urb_pc_sse',
    'grass_frac': 'pst_pc_sse',
    'shrub_frac': 'pnv_pc_s04',
    'pasture_fraction': 'pst_pc_sse',
    'glacier_fraction': 'gla_pc_sse',
    'water_frac': 'water',
    'other_frac': 'other',
    'dom_veg_hydrology': 'land_use_main',
    'dom_veg_nature': 'glc_cl_smj',
    'dom_veg_climatology': 'clz_cl_smj',
    'lai_max': 'laiyrmax',
    'lai_diff': 'laiyrmin',
    'gvf_max': 'glc_pc_s01',
    'gvf_diff': 'glc_pc_s02',
    # Soils & Geology
    'sand_fraction': 'snd_pc_sav',
    'sand_frac': 'snd_pc_sav',
    'silt_fraction': 'slt_pc_sav',
    'silt_frac': 'slt_pc_sav',
    'clay_fraction': 'cly_pc_sav',
    'clay_frac': 'cly_pc_sav',
    'soil_depth': 'soildepth1',
    'soil_depth_pelletier': 'soildepth1',
    'soil_depth_statsgo': 'soildepth2',
    'soil_porosity': 'thetas1',
    'soil_conductivity': 'ksat1',
    'max_water_content': 'swc_pc_syr',
    'organic_frac': 'soc_th_sav',
    'carbonate_rocks_frac': 'kar_pc_sse',
    'geol_permeability': 'sgr_dk_sav',
    'root_depth': 'soildepth3',
    'root_depth_50': 'soildepth1',
    'root_depth_99': 'soildepth2',
    # Human & Socioeconomic
    'gdp_mean': 'gdp_ud_sav',
    'population_density': 'ppd_pk_sav',
    'human_footprint': 'hft_ix_s09',
}


def _resolve_and_select_attributes(
    ds: xarray.Dataset, features: list[str]
) -> xarray.Dataset:
  """Resolves requested feature names using alias mapping, selects them, and renames them to match requested names."""
  if not features:
    return ds

  available_cols = set(ds.data_vars).union(ds.coords)
  selected_cols = []
  rename_map = {}
  missing_features = []

  for f in features:
    if f in available_cols:
      selected_cols.append(f)
    elif (
        f in STATIC_ATTRIBUTE_ALIASES
        and STATIC_ATTRIBUTE_ALIASES[f] in available_cols
    ):
      source_col = STATIC_ATTRIBUTE_ALIASES[f]
      selected_cols.append(source_col)
      rename_map[source_col] = f
      LOGGER.info(
          "Resolved static attribute alias '%s' -> '%s' and renamed to '%s'.",
          f,
          source_col,
          f,
      )
    else:
      # Check reverse alias: e.g. user requested HydroATLAS name but dataset has intuitive name
      reverse_match = None
      for alias, target in STATIC_ATTRIBUTE_ALIASES.items():
        if target == f and alias in available_cols:
          reverse_match = alias
          break
      if reverse_match:
        selected_cols.append(reverse_match)
        rename_map[reverse_match] = f
        LOGGER.info(
            "Resolved static attribute alias '%s' -> '%s' and renamed to '%s'.",
            f,
            reverse_match,
            f,
        )
      else:
        missing_features.append(f)

  if missing_features:
    raise ValueError(
        f'Requested static attributes {missing_features} not found in dataset. '
        f'Available attributes: {sorted(list(available_cols))}'
    )

  # Select only the needed columns (preserve order, remove duplicates)
  selected_cols = list(dict.fromkeys(selected_cols))
  ds = ds[selected_cols]
  if rename_map:
    ds = ds.rename(rename_map)

  return ds


def _find_zarr_store(
    path: Path | str, preferred_names: list[str]
) -> Path | None:
  """Finds a Zarr store given a directory or file path."""
  path_str = str(path)
  if path_str.startswith('gs://') or path_str.startswith('gs:/'):
    # For GCS paths, assume valid store if ends with zarr or subpath
    if path_str.endswith('.zarr'):
      return Path(path_str)
    for name in preferred_names:
      return Path(f"{path_str.rstrip('/')}/{name}")
    return Path(path_str)

  gf = get_gfile()
  if gf and path_str.startswith('/cns/'):
    if (
        path_str.endswith('.zarr')
        or gf.Exists(f'{path_str}/.zgroup')
        or gf.Exists(f'{path_str}/zarr.json')
        or gf.Exists(f'{path_str}/.zmetadata')
    ):
      return Path(path_str)
    for name in preferred_names:
      candidate = f"{path_str.rstrip('/')}/{name}"
      if gf.Exists(candidate) and (
          candidate.endswith('.zarr')
          or gf.Exists(f'{candidate}/.zgroup')
          or gf.Exists(f'{candidate}/zarr.json')
          or gf.Exists(f'{candidate}/.zmetadata')
      ):
        return Path(candidate)
    return None

  p = Path(path)
  if (
      p.suffix == '.zarr'
      or (p / '.zgroup').exists()
      or (p / 'zarr.json').exists()
      or (p / '.zmetadata').exists()
  ):
    return p
  for name in preferred_names:
    candidate = p / name
    if candidate.exists() and (
        candidate.suffix == '.zarr'
        or (candidate / '.zgroup').exists()
        or (candidate / 'zarr.json').exists()
        or (candidate / '.zmetadata').exists()
    ):
      return candidate
  return None


from googlehydrology.datautils.utils import safe_sel_basins
from googlehydrology.utils.gfile_utils import GFileZarrStore, is_cns_path, is_strict_zarr_only


def load_caravan_attributes(
    data_dir: Path | str,
    basins: list[str] | None = None,
    subdataset: str | None = None,
    features: list[str] | None = None,
) -> xarray.Dataset:
  """Load the attributes of the Caravan dataset.

  Supports Zarr stores (preferred) and legacy Caravan CSV directories.

  Parameters
  ----------
  data_dir : Path | str
      Path to attributes Zarr store or root directory of Caravan attributes.
  basins : list[str], optional
      If passed, returns only attributes for the basins specified in this list.
      Otherwise, the attributes of all
      basins are returned.
  subdataset : str, optional
      If passed (legacy CSV mode), returns only the attributes of one
      sub-dataset.
  features: list[str], optional
      If passed, will only return the specified features (columns) in the
      statics datasets.

  Returns
  -------
  xarray.Dataset
      A basin indexed Dataset with all attributes as coordinates.
  """
  import time

  start_t = time.time()
  start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_t))
  print(
      f'[{start_str}] [CARAVAN LOAD START] Loading attributes from'
      f" '{data_dir}'",
      flush=True,
  )

  zarr_store = _find_zarr_store(
      data_dir, ['attributes.zarr', 'attributes', 'statics.zarr', 'statics']
  )
  if zarr_store is not None:
    store_path = zarr_store.as_posix().replace('gs:/', 'gs://')
    try:
      ds = xarray.open_zarr(store_path, consolidated=True, chunks='auto')
      fmt = 'Consolidated Zarr'
    except Exception:
      try:
        ds = xarray.open_zarr(store_path, consolidated=False, chunks='auto')
        fmt = 'Unconsolidated Zarr'
      except Exception:
        ds = xarray.open_zarr(store_path, consolidated=False)
        fmt = 'Unconsolidated Zarr'
    if features:
      ds = _resolve_and_select_attributes(ds, features)
    if basins:
      ds = safe_sel_basins(ds, basins)

    return ds

  if is_strict_zarr_only():
    raise FileNotFoundError(
        "Strict Zarr Policy Violation: Consolidated 'attributes.zarr' store"
        f" not found in '{data_dir}'. Legacy CSV directory fallback is"
        ' prohibited.'
    )

  # Legacy CSV loader fallback
  data_dir_str = str(data_dir)
  attr_dir_str = (
      f"{data_dir_str.rstrip('/')}/attributes"
      if gfile_exists(f"{data_dir_str.rstrip('/')}/attributes")
      else data_dir_str
  )
  gf = get_gfile()

  if subdataset:
    subdataset_dir_str = f'{attr_dir_str}/{subdataset}'
    if not gfile_exists(subdataset_dir_str):
      raise FileNotFoundError(
          f'No subdataset {subdataset} found at {subdataset_dir_str}.'
      )
    subdataset_dirs = [subdataset_dir_str]
  else:
    if is_cns_path(attr_dir_str) and gf:
      entries = [
          e.decode('utf-8') if isinstance(e, bytes) else str(e)
          for e in gf.ListDir(attr_dir_str)
      ]
      subdataset_dirs = [
          f'{attr_dir_str}/{d}'
          for d in entries
          if gf.IsDirectory(f'{attr_dir_str}/{d}')
      ]
    else:
      attr_dir = Path(attr_dir_str)
      subdataset_dirs = [str(d) for d in attr_dir.glob('*') if d.is_dir()]

  if basins:
    subdataset_names = list(set(x.split('_')[0] for x in basins))
    if subdataset:
      if (
          len(subdataset_names) > 1
          or subdataset_names[0].lower() != subdataset.lower()
      ):
        raise ValueError(
            'At least one of the passed basins is not part of the passed'
            ' subdataset.'
        )
    else:
      subdataset_lower_map = {
          os.path.basename(s.rstrip('/')).lower(): s for s in subdataset_dirs
      }
      valid_subdataset_dirs = []
      missing_subdatasets = []
      for name in subdataset_names:
        if name.lower() in subdataset_lower_map:
          valid_subdataset_dirs.append(subdataset_lower_map[name.lower()])
        else:
          missing_subdatasets.append(name)

      if missing_subdatasets:
        raise FileNotFoundError(
            f'Could not find subdataset directories for {missing_subdatasets}'
            f' in {attr_dir_str}.'
        )
      subdataset_dirs = valid_subdataset_dirs

  LOGGER.debug('load legacy attribute files')
  ds = _load_attribute_files_of_subdatasets(subdataset_dirs, features or [])

  if basins:
    ds = safe_sel_basins(ds, basins)

  return ds


def load_caravan_timeseries(
    data_dir: Path | str,
    basins: list[str],
    target_features: list[str],
    *,
    csv: bool = False,
    batch_size: int = 500,
) -> xarray.Dataset:
  """Load the timeseries data of basins from the Caravan dataset.

  Supports Zarr stores (preferred) and legacy multi-file NetCDF/CSV datasets.

  Parameters
  ----------
  data_dir : Path | str
      Path to timeseries Zarr store or root directory of Caravan timeseries.
  basins : list[str]
      List of basin ID strings.
  target_features : list[str]
      The target variables to select.
  csv: bool, optional
      Whether to load CSV files instead of NC files (legacy mode).
  batch_size : int, optional
      Batch size for legacy multi-file loader.

  Returns
  -------
  xarray.Dataset
      A combined Dataset with 'basin' and 'date' coordinates.
  """
  import time

  start_t = time.time()
  start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_t))
  print(
      f'[{start_str}] [CARAVAN TIMESERIES START] Loading streamflow timeseries'
      f" from '{data_dir}'",
      flush=True,
  )

  zarr_store = _find_zarr_store(
      data_dir,
      [
          'streamflow_targets.zarr',
          'streamflow_targets',
          'streamflow.zarr',
          'streamflow',
          'targets.zarr',
          'targets',
          'timeseries.zarr',
          'timeseries',
      ],
  )
  if zarr_store is not None:
    store_path = zarr_store.as_posix().replace('gs:/', 'gs://')
    try:
      ds = xarray.open_zarr(store_path, consolidated=True, chunks='auto')
      fmt = 'Consolidated Zarr'
    except Exception:
      try:
        ds = xarray.open_zarr(store_path, consolidated=False, chunks='auto')
        fmt = 'Unconsolidated Zarr'
      except Exception:
        ds = xarray.open_zarr(store_path, consolidated=False)
        fmt = 'Unconsolidated Zarr'
    if target_features:
      available = [f for f in target_features if f in ds.data_vars]
      ds = ds[available]
    if basins:
      ds = safe_sel_basins(ds, basins)

    elapsed = time.time() - start_t
    end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(
        f'[{end_str}] [CARAVAN TIMESERIES COMPLETE] Loaded streamflow'
        f' timeseries in {elapsed:.2f}s (Format: {fmt}, Path: {store_path})',
        flush=True,
    )
    return ds

  return load_caravan_timeseries_together(
      data_dir=Path(data_dir),
      basins=basins,
      target_features=target_features,
      csv=csv,
      batch_size=batch_size,
  )


def load_csvs_as_ds(basin_to_path: dict[str, Path]) -> xarray.Dataset:
  """Load timeseries data from CSV files into a single xarray Dataset."""
  datas = (
      dd.read_csv(path, parse_dates=['date'], dtype=np.float32)
      for path in basin_to_path.values()
  )
  datas = [df.assign(basin=basin) for basin, df in zip(basin_to_path, datas)]
  return dd.concat(datas).compute().set_index(['basin', 'date']).to_xarray()


def load_caravan_timeseries_together(
    data_dir: Path,
    basins: list[str],
    target_features: list[str],
    *,
    csv: bool = False,
    batch_size: int = 500,
) -> xarray.Dataset:
  """Legacy multi-file Caravan timeseries loader."""
  bar_off = logging.getLogger().level > logging.DEBUG

  def basin_to_path(basin: str) -> Path:
    subdataset = basin.partition('_')[0]
    kind = 'csv' if csv else 'netcdf'
    ext = 'csv' if csv else 'nc'
    base = f"{str(data_dir).rstrip('/')}/timeseries/{kind}"
    candidates = [
        f'{base}/{subdataset}/{basin}.{ext}',
        f'{base}/{subdataset.lower()}/{basin}.{ext}',
        f'{base}/{subdataset.upper()}/{basin}.{ext}',
    ]
    for path_str in candidates:
      if gfile_exists(path_str):
        return Path(path_str)
    raise FileNotFoundError(
        f'No basin file found at any candidate: {candidates}.'
    )

  def select(ds: xarray.Dataset) -> xarray.Dataset:
    return ds[target_features]

  paths = tuple(map(basin_to_path, basins))

  if csv:
    return select(load_csvs_as_ds(dict(zip(basins, paths))))

  combine = functools.partial(
      xarray.combine_nested,
      concat_dim='basin',
      coords='minimal',
      compat='override',
      combine_attrs='override',
  )

  open_dataset_args = {'chunks': {'date': 'auto'}}

  def open_dataset(ds_path: Path) -> tuple[xarray.Dataset, float, float]:
    with gfile_open(ds_path, 'rb') as f:
      ds = select(xarray.open_dataset(f, **open_dataset_args)).load()
    first_date, last_date = ds['date'].isel(date=[0, -1]).data
    return ds, first_date, last_date

  def open_datasets(
      batch_paths: tuple[Path],
  ) -> Iterator[tuple[xarray.Dataset, float, float]]:
    dss, n = map(open_dataset, batch_paths), len(batch_paths)
    yield from tqdm(
            dss, desc='Read', unit='file', leave=False, disable=bar_off, total=n
        )

  def process_batch(batch_paths: Iterable[Path]) -> xarray.Dataset:
    datasets, starts, ends = zip(*open_datasets(batch_paths))
    start, end = min(starts), max(ends)
    date = pd.date_range(start=start, end=end, freq='D', name='date')
    datasets = [ds.reindex(date=date) for ds in datasets]
    return combine(datasets, join='override')

  def batchify() -> Iterator[xarray.Dataset]:
    batches = map(process_batch, itertools.batched(paths, batch_size))
    total = math.ceil(len(paths) / batch_size)
    yield from tqdm(
        batches, desc='Gather', unit='batch', total=total, disable=bar_off
    )

  return combine(tuple(batchify()), join='outer').assign_coords(basin=basins)


def _load_attribute_files_of_subdatasets(
    datasets: list[str | Path], features: list[str]
) -> xarray.Dataset:
  """Loads all attribute CSV files, indexing gauge_id to basin."""
  gf = get_gfile()
  csv_files = []
  for d in datasets:
    d_str = str(d)
    if is_cns_path(d_str) and gf:
      for f in gf.ListDir(d_str):
        if f.endswith('.csv'):
          csv_files.append(f'{d_str}/{f}')
    else:
      for f in Path(d_str).glob('*.csv'):
        csv_files.append(str(f))

  dss = []
  for csv_file in csv_files:
    with gfile_open(csv_file, 'r') as fp:
      df64 = pd.read_csv(fp, index_col='gauge_id')
    df = df64.astype({
        col: np.float32
        for col in df64.select_dtypes(include=[np.number]).columns
    })
    df.rename_axis('basin', inplace=True)
    if features:
      df.drop(
          columns=(e for e in df.columns if e not in features), inplace=True
      )
    dss.append(df.to_xarray().chunk({'basin': -1}))

  if not dss:
    raise FileNotFoundError(f'No attribute CSV files found in {datasets}')

  return xarray.merge(dss, join='outer', compat='no_conflicts')
