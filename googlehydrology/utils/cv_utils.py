import csv
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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


def read_text_content(file_path: Union[str, Path]) -> str:
  """Reads text content from POSIX or CNS path with gfile and fileutil fallback."""
  file_path_str = str(file_path)
  if file_path_str.startswith('/cns/'):
    gf = get_gfile()
    if gf:
      try:
        with gf.GFile(file_path_str, 'r') as f:
          content = f.read()
        if isinstance(content, bytes):
          content = content.decode('utf-8')
        return content
      except Exception:
        pass
    # Fallback to fileutil CLI
    try:
      res = subprocess.run(
          ['fileutil', 'cat', file_path_str],
          capture_output=True,
          text=True,
          check=True,
      )
      return res.stdout
    except Exception as e:
      raise RuntimeError(f'Failed to read CNS file {file_path_str}: {e}')
  with open(file_path_str, 'r', encoding='utf-8') as f:
    return f.read()


def ensure_dir(path_str: Union[str, Path]) -> None:
  """Ensures directory exists on POSIX or CNS path."""
  path_str = str(path_str)
  if path_str.startswith('/cns/'):
    gf = get_gfile()
    if gf:
      try:
        if not gf.Exists(path_str):
          gf.MakeDirs(path_str)
        return
      except Exception:
        pass
    try:
      subprocess.run(
          ['fileutil', 'mkdir', '-p', path_str],
          capture_output=True,
          check=False,
      )
    except Exception:
      pass
  else:
    os.makedirs(path_str, exist_ok=True)


def write_lines(file_path: Union[str, Path], lines: List[str]) -> str:
  """Writes list of strings to POSIX or CNS path."""
  file_path_str = str(file_path)
  parent_dir = os.path.dirname(file_path_str)
  if parent_dir:
    ensure_dir(parent_dir)

  content = '\n'.join(lines) + '\n'
  if file_path_str.startswith('/cns/'):
    gf = get_gfile()
    if gf:
      try:
        with gf.GFile(file_path_str, 'w') as f:
          f.write(content)
        return file_path_str
      except Exception:
        pass
    import tempfile

    with tempfile.NamedTemporaryFile(
        'w', delete=False, encoding='utf-8'
    ) as tmp:
      tmp.write(content)
      tmp_path = tmp.name
    try:
      subprocess.run(
          ['fileutil', 'cp', '-f', tmp_path, file_path_str],
          check=True,
          capture_output=True,
      )
      return file_path_str
    finally:
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  with open(file_path_str, 'w') as f:
    f.write(content)
  return file_path_str


def read_lines(file_path: Union[str, Path]) -> List[str]:
  """Reads lines from POSIX or CNS file path."""
  content = read_text_content(file_path)
  return [
      l.strip()
      for l in content.splitlines()
      if l.strip() and not l.startswith('#')
  ]


def split_basins_into_k_groups(
    basins: List[str],
    k: int = 5,
    seed: int = 42,
) -> Dict[int, List[str]]:
  """Splits a list of unique basin IDs into k equal-sized random groups."""
  rng = random.Random(seed)
  unique_basins = sorted(list(set(basins)))
  shuffled = list(unique_basins)
  rng.shuffle(shuffled)
  groups = {i: [] for i in range(k)}
  for idx, b in enumerate(shuffled):
    groups[idx % k].append(b)
  for i in range(k):
    groups[i].sort()
  return groups


def load_basin_groups(
    source_path: Union[str, Path],
    num_groups: int = 5,
    seed: int = 42,
    subsample_total: Optional[int] = None,
) -> Dict[int, List[str]]:
  """Loads basin groups from master CSV with group_id, or directory with basin_group_*.txt, optionally subsampling."""
  source_str = str(source_path)
  gf = get_gfile()

  def _subsample_if_needed(
      raw_groups: Dict[int, List[str]],
  ) -> Dict[int, List[str]]:
    if not subsample_total or subsample_total <= 0:
      return raw_groups
    total_available = sum(len(b_list) for b_list in raw_groups.values())
    if total_available <= subsample_total:
      return raw_groups

    target_per_group = subsample_total // num_groups
    remainder = subsample_total % num_groups

    subsampled = {}
    for g_idx in range(num_groups):
      b_list = sorted(list(set(raw_groups.get(g_idx, []))))
      n_target = target_per_group + (1 if g_idx < remainder else 0)
      if n_target >= len(b_list):
        subsampled[g_idx] = b_list
      else:
        rng = random.Random(seed + g_idx)
        subsampled[g_idx] = sorted(rng.sample(b_list, n_target))
    return subsampled

  is_dir = False
  if source_str.startswith('/cns/'):
    if gf and gf.Exists(source_str) and gf.IsDirectory(source_str):
      is_dir = True
    elif not (
        source_str.lower().endswith('.csv')
        or source_str.lower().endswith('.txt')
    ):
      is_dir = True
  elif os.path.isdir(source_str) or not (
      source_str.lower().endswith('.csv') or source_str.lower().endswith('.txt')
  ):
    is_dir = True

  if is_dir:
    groups = {}
    for g_idx in range(num_groups):
      for candidate_name in [f'group_{g_idx}.txt', f'basin_group_{g_idx}.txt']:
        candidate = os.path.join(source_str, candidate_name)
        if source_str.startswith('/cns/') and gf and gf.Exists(candidate):
          groups[g_idx] = read_lines(candidate)
          break
        elif source_str.startswith('/cns/') and not gf:
          try:
            groups[g_idx] = read_lines(candidate)
            break
          except Exception:
            pass
        elif os.path.exists(candidate):
          groups[g_idx] = read_lines(candidate)
          break
    if len(groups) == num_groups:
      return _subsample_if_needed(groups)

  if source_str.lower().endswith('.csv'):
    content = read_text_content(source_str)
    lines = content.splitlines()

    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames or []
    basin_candidates = ['gauge_id', 'basin_id', 'basin']
    basin_col = next(
        (c for c in basin_candidates if c in fieldnames),
        (fieldnames[0] if fieldnames else None),
    )

    group_candidates = ['group_id', 'fold', 'group']
    group_col = next((c for c in group_candidates if c in fieldnames), None)

    if group_col:
      groups = {i: [] for i in range(num_groups)}
      for row in reader:
        g_id = int(row[group_col])
        b_id = str(row[basin_col]).strip()
        if b_id:
          groups[g_id].append(b_id)
      for i in range(num_groups):
        groups[i] = sorted(list(set(groups[i])))
      return _subsample_if_needed(groups)
    else:
      basins = [
          str(row[basin_col]).strip()
          for row in reader
          if str(row[basin_col]).strip()
      ]
      return _subsample_if_needed(
          split_basins_into_k_groups(basins, k=num_groups, seed=seed)
      )

  basins = read_lines(source_str)
  return _subsample_if_needed(
      split_basins_into_k_groups(basins, k=num_groups, seed=seed)
  )


def get_basin_fold_partition(
    groups: Dict[int, List[str]],
    val_group: Optional[int],
    test_group: int,
) -> Tuple[List[str], List[str], List[str]]:
  """Partitions groups into Train (all non-test groups) and Test (1 group)."""
  test_basins = sorted(groups[test_group])
  val_basins = (
      sorted(groups[val_group])
      if (val_group is not None and val_group != test_group)
      else test_basins
  )
  train_basins = []
  for g_idx, b_list in groups.items():
    if g_idx != test_group and (
        val_group is None or val_group == test_group or g_idx != val_group
    ):
      train_basins.extend(b_list)
  train_basins = sorted(list(set(train_basins)))
  return train_basins, val_basins, test_basins


def get_date_fold_partition(
    date_groups: Dict[int, Dict[str, Any]],
    val_group: int,
    test_group: int,
) -> Dict[str, Any]:
  """Partitions 4 2-year date blocks into Train (2 blocks), Val (1 block), Test (1 block)."""
  train_indices = [
      g
      for g in sorted(date_groups.keys())
      if g != val_group and g != test_group
  ]

  tr_full_start = min(date_groups[g]['full_start'] for g in train_indices)
  tr_full_end = max(date_groups[g]['full_end'] for g in train_indices)

  val_info = date_groups[val_group]
  test_info = date_groups[test_group]

  val_eval_yr = val_info.get('eval_year') or (
      val_info.get('eval_start', '').split('-')[0]
      if val_info.get('eval_start')
      else ''
  )
  test_eval_yr = test_info.get('eval_year') or (
      test_info.get('eval_start', '').split('-')[0]
      if test_info.get('eval_start')
      else ''
  )

  return {
      'train_full_start': tr_full_start,
      'train_full_end': tr_full_end,
      'train_groups': train_indices,
      'val_full_start': val_info['full_start'],
      'val_full_end': val_info['full_end'],
      'val_warmup_start': val_info.get('warmup_start', val_info['full_start']),
      'val_warmup_end': val_info.get('warmup_end', val_info['full_end']),
      'val_eval_start': val_info.get('eval_start', val_info['full_end']),
      'val_eval_end': val_info.get('eval_end', val_info['full_end']),
      'val_eval_year': val_eval_yr,
      'test_full_start': test_info['full_start'],
      'test_full_end': test_info['full_end'],
      'test_warmup_start': test_info.get(
          'warmup_start', test_info['full_start']
      ),
      'test_warmup_end': test_info.get('warmup_end', test_info['full_end']),
      'test_eval_start': test_info.get('eval_start', test_info['full_end']),
      'test_eval_end': test_info.get('eval_end', test_info['full_end']),
      'test_eval_year': test_eval_yr,
  }


def compute_single_year_test_temporal_split(
    test_year: int,
    data_start_year: int = 2016,
    data_end_year: int = 2023,
) -> Dict[str, Any]:
  """Computes train and test date ranges for a single test year.

  Args:
      test_year: Evaluated test year (between data_start_year + 1 and
        data_end_year).
      data_start_year: Earliest data year available (default 2016).
      data_end_year: Latest data year available (default 2023).

  Returns:
      Dict with test and train start/end dates.
  """
  if test_year < data_start_year + 1 or test_year > data_end_year:
    raise ValueError(
        f'test_year must be between {data_start_year + 1} and {data_end_year},'
        f' got {test_year}'
    )

  test_start_date = f'{test_year - 1}-01-01'
  test_end_date = f'{test_year}-12-31'

  train_start_dates = []
  train_end_dates = []

  # 1. Pre-Test Block (requires data_start_year as spinup year)
  if test_year - 2 >= data_start_year + 1:
    train_start_dates.append(f'{data_start_year}-01-01')
    train_end_dates.append(f'{test_year - 2}-12-31')

  # 2. Post-Test Block (test_year + 1 acts as spinup year)
  if test_year + 2 <= data_end_year:
    train_start_dates.append(f'{test_year + 1}-01-01')
    train_end_dates.append(f'{data_end_year}-12-31')

  return {
      'test_year': test_year,
      'test_warmup_year': test_year - 1,
      'test_full_start': test_start_date,
      'test_full_end': test_end_date,
      'test_start_date': test_start_date,
      'test_end_date': test_end_date,
      'test_eval_start': f'{test_year}-01-01',
      'test_eval_end': f'{test_year}-12-31',
      'val_full_start': None,
      'val_full_end': None,
      'val_eval_start': None,
      'val_eval_end': None,
      'train_full_start': (
          train_start_dates[0] if train_start_dates else test_start_date
      ),
      'train_full_end': (
          train_end_dates[-1] if train_end_dates else test_end_date
      ),
      'train_start_dates': train_start_dates,
      'train_end_dates': train_end_dates,
  }


# Alias for backwards compatibility / concise naming
compute_single_year_date_partitions = compute_single_year_test_temporal_split


def generate_single_year_test_cv_folds(
    test_years: Union[int, List[int]] = 2017,
    num_basin_groups: int = 5,
    data_start_year: int = 2016,
    data_end_year: int = 2023,
    time_splits_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
  """Generates CV fold specifications for single test year temporal splits."""
  if isinstance(test_years, int):
    years = [test_years]
  elif isinstance(test_years, (list, tuple)):
    years = list(test_years)
  else:
    years = [int(test_years)]

  # Load from time_splits_file if available
  precomputed_splits = {}
  if time_splits_file:
    time_splits_str = str(time_splits_file)
    try:
      gf = get_gfile()
      if time_splits_str.startswith('/cns/') and gf:
        with gf.GFile(time_splits_str, 'r') as f:
          precomputed_splits = json.loads(f.read())
      else:
        with open(time_splits_str, 'r', encoding='utf-8') as f:
          precomputed_splits = json.load(f)
    except Exception:
      precomputed_splits = {}

  folds = []
  fold_counter = 0

  for yr in years:
    if str(yr) in precomputed_splits:
      d_partition = precomputed_splits[str(yr)]
    else:
      d_partition = compute_single_year_test_temporal_split(
          test_year=yr,
          data_start_year=data_start_year,
          data_end_year=data_end_year,
      )
    for b_test in range(num_basin_groups):
      b_val = b_test
      b_train = [g for g in range(num_basin_groups) if g != b_test]

      fold_id = f'test_yr{yr}_b{b_test}'

      folds.append({
          'fold_index': fold_counter,
          'fold_id': fold_id,
          'date_set': f'test_year_{yr}',
          'test_year': yr,
          'basin_test_group': b_test,
          'basin_val_group': b_val,
          'basin_train_groups': b_train,
          'time_test_group': 0,
          'time_val_group': 0,
          'time_train_groups': [0],
          'date_partition': d_partition,
      })
      fold_counter += 1

  return folds


def generate_standard_cv_folds(
    num_basin_groups: int = 5,
    num_time_groups: int = 4,
    date_set: str = '2018',
    time_splits_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
  """Generates cross-validation fold specifications for single year(s) through 2023."""
  ds_clean = str(date_set).lower()
  parts = [p.strip() for p in ds_clean.split(',') if p.strip()]

  if any(p in ('all_years', 'all') for p in parts):
    years = list(range(2017, 2024))
  else:
    years = [
        int(p.replace('test_year_', '').replace('year_', ''))
        for p in parts
        if p.replace('test_year_', '').replace('year_', '').isdigit()
    ]
    if not years:
      years = [2018]

  return generate_single_year_test_cv_folds(
      test_years=years,
      num_basin_groups=num_basin_groups,
      data_start_year=2016,
      data_end_year=2023,
      time_splits_file=time_splits_file,
  )


def materialize_fold_basin_files(
    output_dir: Union[str, Path],
    fold_id: str,
    train_basins: List[str],
    val_basins: List[str],
    test_basins: List[str],
) -> Tuple[str, str, str]:
  """Writes train, val, and test basin text files for a specific fold."""
  output_dir_str = str(output_dir)
  ensure_dir(output_dir_str)
  base_dir = (
      output_dir_str
      if output_dir_str.rstrip('/').endswith(f'fold_{fold_id}')
      else os.path.join(output_dir_str, f'fold_{fold_id}')
  )
  ensure_dir(base_dir)

  train_path = os.path.join(base_dir, 'train_basins.txt')
  test_path = os.path.join(base_dir, 'test_basins.txt')
  write_lines(train_path, train_basins)
  write_lines(test_path, test_basins)

  val_path = ''
  if val_basins:
    val_path = os.path.join(base_dir, 'val_basins.txt')
    write_lines(val_path, val_basins)

  return train_path, val_path, test_path


def get_scaler_cache_dir(
    base_scaler_dir: str,
    date_set: str,
    fold_id: Union[int, str],
) -> str:
  """Returns standard scaler cache directory path for a fold and date set."""
  clean_date_set = str(date_set).strip()
  return os.path.join(
      base_scaler_dir, f'scaler_cache_date_{clean_date_set}_fold_{fold_id}'
  )


def check_file_exists(file_path: str) -> bool:
  """Checks if file exists on POSIX or CNS."""
  if not file_path:
    return False
  file_path_str = str(file_path)
  if file_path_str.startswith('/cns/'):
    gf = get_gfile()
    return bool(gf and gf.Exists(file_path_str))
  return os.path.exists(file_path_str)


def is_valid_scaler(scaler_dir: str, required_features: List[str]) -> bool:
  """Checks if scaler cache exists and contains all required model features."""
  if not scaler_dir:
    return False
  zarr_path = os.path.join(scaler_dir, 'scaler.zarr')
  nc_path = os.path.join(scaler_dir, 'scaler.nc')

  scaler_path = zarr_path if check_file_exists(zarr_path) else nc_path
  if not check_file_exists(scaler_path):
    return False

  try:
    from googlehydrology.utils.gfile_utils import is_strict_zarr_only

    if scaler_path == nc_path and is_strict_zarr_only():
      print(
          "  [WARNING] Strict Zarr Policy: 'scaler.zarr' missing in"
          f" {scaler_dir}. Legacy 'scaler.nc' is ignored.",
          flush=True,
      )
      return False
  except ImportError:
    pass

  try:
    import xarray as xr

    if scaler_path.endswith('.zarr'):
      ds = xr.open_zarr(scaler_path, consolidated=True)
    elif scaler_path.startswith('/cns/'):
      gf = get_gfile()
      if not gf:
        return False
      with gf.GFile(scaler_path, 'rb') as f:
        content = f.read()
      import io

      ds = xr.open_dataset(io.BytesIO(content))
    else:
      ds = xr.open_dataset(scaler_path)

    scaler_vars = set()
    for coord_name in [
        'feature',
        'attribute',
        'variable',
        'features',
        'attributes',
    ]:
      if coord_name in ds.coords:
        scaler_vars.update(str(v) for v in ds[coord_name].values.tolist())
    scaler_vars.update(str(k) for k in ds.data_vars.keys())

    missing = [
        f
        for f in required_features
        if f not in scaler_vars
        and not (f == 'streamflow_shift1' and 'streamflow' in scaler_vars)
    ]
    if missing:
      missing_names = [str(m) for m in missing]
      print(
          f'  [INFO] Precomputed scaler at {scaler_path} is missing'
          f' {len(missing)} variable(s): {missing_names[:5]}... Computing'
          ' scaler dynamically across training basins.',
          flush=True,
      )
      return False
    return True
  except Exception as e:
    print(
        f'  [INFO] Could not validate scaler cache ({e}). Will compute scaler'
        ' dynamically.',
        flush=True,
    )
    return False


def format_date_to_config(
    date_input: Union[str, List[str], Tuple[str, ...]],
) -> Union[str, List[str]]:
  """Formats ISO date strings or lists into %d/%m/%Y format expected by Config."""
  if not date_input:
    return ''
  import pandas as pd

  if isinstance(date_input, (list, tuple)):
    return [pd.to_datetime(d).strftime('%d/%m/%Y') for d in date_input]
  elif isinstance(date_input, str) and ',' in date_input:
    return [
        pd.to_datetime(d.strip()).strftime('%d/%m/%Y')
        for d in date_input.split(',')
    ]
  return pd.to_datetime(date_input).strftime('%d/%m/%Y')
