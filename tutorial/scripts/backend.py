# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hydrology model evaluation, metrics calculation, and visualization backend."""

import glob
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr

# Dynamically resolve repository root and tutorial directories
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = next(
    (p for p in [_CURRENT_DIR, _CURRENT_DIR.parent, _CURRENT_DIR.parent.parent, _CURRENT_DIR.parent.parent.parent] if (p / 'googlehydrology').is_dir()),
    _CURRENT_DIR.parent.parent
)
_TUTORIAL_DIR = _REPO_ROOT / 'tutorial' if (_REPO_ROOT / 'tutorial').is_dir() else _CURRENT_DIR.parent
_SCRIPTS_DIR = _TUTORIAL_DIR / 'scripts'
_NOTEBOOKS_DIR = _TUTORIAL_DIR / 'notebooks'

for _p in [str(_REPO_ROOT), str(_TUTORIAL_DIR), str(_SCRIPTS_DIR), str(_NOTEBOOKS_DIR), str(_TUTORIAL_DIR / 'src')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from googlehydrology.evaluation import metrics


def _resolve_path(path_str: Union[str, Path]) -> Path:
    """Resolves a file or directory path relative to cwd, tutorial dir, or repo root."""
    p = Path(path_str)
    if p.exists():
        return p
    for base in [_TUTORIAL_DIR, _REPO_ROOT, _CURRENT_DIR.parent]:
        candidate = base / p
        if candidate.exists():
            return candidate
    return p


# --- Model Run Discovery & Configuration ---

def find_model_run_dirs(base_dir: Union[str, Path]) -> Dict[str, str]:
    """Find directories containing test results within base_dir."""
    resolved_base = _resolve_path(base_dir)
    run_dirs = {}
    if not resolved_base.is_dir():
        return run_dirs

    for subdir in resolved_base.iterdir():
        if subdir.is_dir():
            test_dir = subdir / 'test'
            if test_dir.is_dir():
                for epoch_dir in test_dir.glob('model_epoch*'):
                    if (epoch_dir / 'test_results.zarr').is_dir():
                        run_dirs[subdir.name] = str(subdir)
                        break
    return run_dirs


def read_basin_list(file_path: Union[str, Path]) -> Set[str]:
    """Read a text file containing one basin ID per line."""
    resolved = _resolve_path(file_path)
    if not resolved.is_file():
        return set()
    with open(resolved, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def load_model_config_and_basins(run_dir: Union[str, Path]) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """Load model configuration dict and associated train/test basin ID sets."""
    resolved_dir = _resolve_path(run_dir)
    config_path = resolved_dir / 'config.yml'
    if not config_path.exists():
        for cand in [
            _TUTORIAL_DIR / 'configs' / 'train-config.yml',
            _TUTORIAL_DIR / 'model-runs' / 'generic-meanembedding-50basin_2107_080323' / 'config.yml',
            resolved_dir / 'config.yaml',
        ]:
            if cand.exists():
                config_path = cand
                break
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_basins = read_basin_list(config.get('train_basin_file', ''))
    test_basins = read_basin_list(config.get('test_basin_file', ''))
    return config, train_basins, test_basins


# --- Geospatial Visualization ---

def plot_colored_shapefile(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: Optional[str] = None,
    colors: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    missing_kwds: Optional[Dict[str, Any]] = None
):
    """Plot a GeoDataFrame colored by a specific attribute column."""
    fig, ax = plt.subplots(figsize=figsize)
    if colors:
        for category, color in colors.items():
            subset = gdf[gdf[column] == category]
            if not subset.empty:
                subset.plot(ax=ax, color=color, label=category, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.legend()
    else:
        gdf.plot(ax=ax, column=column, legend=True, cmap=cmap, alpha=0.7, edgecolor='black', linewidth=0.5, missing_kwds=missing_kwds)
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_train_test_shapefile(
    shapefile_path: Union[str, Path],
    train_basin_ids: Union[List[str], Set[str]],
    test_basin_ids: Union[List[str], Set[str]],
    model_name: str
):
    """Visualize geographical distribution of train and test basin sets."""
    resolved_shp = _resolve_path(shapefile_path)
    gdf_all = gpd.read_file(resolved_shp)
    id_col = 'gauge_id'

    is_train = gdf_all[id_col].isin(train_basin_ids)
    only_test = set(test_basin_ids) - set(train_basin_ids)
    is_test = gdf_all[id_col].isin(only_test)

    gdf_all['dataset'] = 'Not Used'
    gdf_all.loc[is_test, 'dataset'] = 'Test'
    gdf_all.loc[is_train, 'dataset'] = 'Train'

    colors = {'Train': 'purple', 'Test': 'orange', 'Not Used': 'lightgrey'}
    plot_colored_shapefile(gdf_all, column='dataset', title=f"Train & Test Basin Sets: {model_name}", colors=colors)


# --- Test Results & Performance Metrics ---

def load_test_results(run_dir: Union[str, Path], file_name: str = 'test_results.zarr') -> Tuple[xr.Dataset, int]:
    """Load test results from the latest available epoch in run_dir."""
    resolved_dir = _resolve_path(run_dir)
    pattern = str(resolved_dir / 'test' / 'model_epoch*' / file_name)
    result_files = glob.glob(pattern)
    if not result_files:
        raise RuntimeError(f"No results matching '{file_name}' found in {resolved_dir / 'test'}")

    def get_epoch(p: str) -> int:
        match = re.search(r'model_epoch(\d+)', p)
        return int(match.group(1)) if match else -1

    latest_path = max(result_files, key=get_epoch)
    return xr.open_zarr(latest_path, consolidated=False), get_epoch(latest_path)


def calculate_metrics_for_run(sim_data: xr.DataArray, obs_data: xr.DataArray) -> pd.DataFrame:
    """Calculate hydrological metrics for each basin and lead time."""
    results = []
    common_gauges = list(set(sim_data['basin'].values) & set(obs_data['basin'].values))
    metrics_list = metrics.get_available_metrics()

    for gauge_id in tqdm(common_gauges, desc="Processing Gauges"):
        sim_gauge = sim_data.sel(basin=gauge_id, freq='1D').load()
        obs_gauge = obs_data.sel(basin=gauge_id, freq='1D').load()

        for lt in sim_gauge['time_step'].values:
            calc_res = metrics.calculate_metrics(
                obs=obs_gauge.sel(time_step=lt),
                sim=sim_gauge.sel(time_step=lt),
                metrics=metrics_list,
                resolution="1D",
                datetime_coord="date"
            )
            df_m = calc_res.to_frame().T if isinstance(calc_res, pd.Series) else (pd.DataFrame([calc_res]) if isinstance(calc_res, dict) else calc_res)
            df_m['basin_id'] = gauge_id
            df_m['lead_time'] = lt
            results.append(df_m)

    df_all = pd.concat(results, ignore_index=True)
    return df_all.set_index(['basin_id', 'lead_time'])


def load_data_and_metrics(
    model_run_dir: Union[str, Path],
    test_basin_ids: Set[str],
    calculate_statistics: bool = False,
    model_name: str = 'Model'
) -> Tuple[xr.Dataset, Optional[pd.DataFrame]]:
    """Load model output dataset and precalculated or newly calculated performance metrics."""
    resolved_dir = _resolve_path(model_run_dir)
    model_data, _ = load_test_results(resolved_dir)
    metrics_path = resolved_dir / 'test' / 'precalculated_metrics.csv'

    if calculate_statistics or not metrics_path.exists():
        model_metrics = calculate_metrics_for_run(model_data['streamflow_sim'], model_data['streamflow_obs'])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        model_metrics.to_csv(metrics_path)
    else:
        model_metrics = pd.read_csv(metrics_path)
        if {'basin_id', 'lead_time'}.issubset(model_metrics.columns):
            model_metrics = model_metrics.set_index(['basin_id', 'lead_time'])

    return model_data, model_metrics


def plot_lead_time_zero_scores(
    metrics_df: pd.DataFrame,
    train_basin_ids: Set[str],
    test_basin_ids: Set[str],
    metric_name: str,
    model_name: str = 'Model'
):
    """Plot horizontal bar chart of metric scores at lead time 0 across basins."""
    scores = metrics_df.xs(0, level='lead_time')[metric_name] if 'lead_time' in metrics_df.index.names else metrics_df[metric_name]
    df = scores.reset_index()
    df['basin_type'] = df['basin_id'].apply(lambda x: 'Train' if x in train_basin_ids else ('Test' if x in test_basin_ids else 'Other'))
    df = df.sort_values(by=metric_name, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'Train': 'purple', 'Test': 'orange', 'Other': 'gray'}
    for basin_type, group in df.groupby('basin_type'):
        ax.barh(group['basin_id'], group[metric_name], color=colors.get(basin_type, 'gray'), label=basin_type)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim([-0.5, 1])
    ax.set_xlabel(f"{metric_name} Score (Lead Time 0)")
    ax.set_title(f"{model_name} Performance: {metric_name}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison_metrics_vs_lead_time(
    base_metrics_df: pd.DataFrame,
    finetune_metrics_df: Optional[pd.DataFrame],
    basin_id: str,
    metric_name: str
):
    """Plot metric scores across lead times comparing base and fine-tuned models."""
    plt.figure(figsize=(10, 6))
    base_vals = base_metrics_df.loc[basin_id, metric_name]
    plt.plot(base_vals.index, base_vals.values, marker='o', label='Base Model')

    if finetune_metrics_df is not None:
        ft_vals = finetune_metrics_df.loc[basin_id, metric_name]
        plt.plot(ft_vals.index, ft_vals.values, marker='o', linestyle='--', label='Fine-Tuned Model')

    plt.title(f"{metric_name} vs. Lead Time for Basin {basin_id}")
    plt.xlabel("Lead Time (days)")
    plt.ylabel(f"{metric_name} Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# --- Fine-Tuning Helpers ---

def replace_placeholders(data: Any, basin_id: Union[str, int]) -> Any:
    """Recursively replace 'FINETUNE_BASIN' placeholder in nested dicts/lists."""
    if isinstance(data, dict):
        return {k: replace_placeholders(v, basin_id) for k, v in data.items()}
    if isinstance(data, list):
        return [replace_placeholders(elem, basin_id) for elem in data]
    if isinstance(data, str):
        return data.replace('FINETUNE_BASIN', str(basin_id))
    return data


def create_basin_list_file(basin_id: str, output_dir: Union[str, Path] = 'basin-lists') -> str:
    """Generate a single-basin list text file for fine-tuning."""
    out_path = Path(output_dir)
    if not out_path.is_absolute() and not out_path.exists():
        for base in [_TUTORIAL_DIR, _REPO_ROOT]:
            if (base / out_path).exists():
                out_path = base / out_path
                break
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{basin_id}.txt"
    with open(file_path, 'w') as f:
        f.write(f"{basin_id}\n")
    return str(file_path)


def generate_basin_finetune_config(
    template_path: Union[str, Path],
    basin_id: str,
    base_model_dir: Union[str, Path],
    output_path: Union[str, Path]
):
    """Generate a YAML fine-tuning configuration file from a template."""
    resolved_tmpl = _resolve_path(template_path)
    with open(resolved_tmpl, 'r') as f:
        config = yaml.safe_load(f)

    config = replace_placeholders(config, basin_id)
    config['base_run_dir'] = str(base_model_dir)
    config['run_dir'] = str(base_model_dir)

    out_file = Path(output_path)
    if not out_file.is_absolute() and not out_file.parent.exists():
        out_file = _TUTORIAL_DIR / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


# --- Hydrological Metrics & Continuous Rolling Hydrographs ---

def compute_hydro_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Compute core hydrological performance metrics: NSE, KGE, Pearson-r, and RMSE."""
    o_arr = np.asarray(obs).flatten()
    s_arr = np.asarray(sim).flatten()
    valid = (~np.isnan(o_arr)) & (~np.isnan(s_arr))
    o, s = o_arr[valid], s_arr[valid]

    if len(o) < 2:
        return {'NSE': np.nan, 'KGE': np.nan, 'Pearson-r': np.nan, 'RMSE': np.nan}

    denom = float(np.sum((o - np.mean(o)) ** 2))
    nse = float(1.0 - np.sum((o - s) ** 2) / (denom + 1e-8)) if denom > 0 else np.nan
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))

    std_s, std_o = float(np.std(s)), float(np.std(o))
    if std_o > 1e-8 and std_s > 1e-8:
        r = float(np.corrcoef(o, s)[0, 1])
        mean_ratio = float(np.mean(s) / (np.mean(o) + 1e-8))
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (std_s / (std_o + 1e-8) - 1.0) ** 2 + (mean_ratio - 1.0) ** 2))
    else:
        r, kge = np.nan, np.nan

    return {'NSE': nse, 'KGE': kge, 'Pearson-r': r, 'RMSE': rmse}


calculate_timeseries_metrics = compute_hydro_metrics


def extract_rolling_hydrograph(
    data: Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]],
    basin_id: Optional[str] = None,
    lead_time: int = 5,
    obs_data: Optional[Union[xr.Dataset, xr.DataArray, np.ndarray]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    window_days: Optional[int] = None,
    align_target_dates: bool = True
) -> pd.DataFrame:
    """Extract rolling lead-time hydrograph time series for fixed lead time L across daily forecast issues."""
    if isinstance(data, dict):
        b_key = basin_id if basin_id else 'camels'
        if basin_id in data and isinstance(data[basin_id], dict):
            binfo = data[basin_id]
            dates = pd.to_datetime(binfo.get('dates', binfo.get('date')))
            obs_arr = np.asarray(binfo.get('obs', binfo.get('streamflow_obs'))).flatten()
            sim_arr = np.asarray(binfo.get('sim', binfo.get('streamflow_sim'))).flatten()
        else:
            sim_key = next((k for k in [f'{b_key}_sim', f'{b_key}_finetune', f'{b_key}_zero', 'streamflow_sim', 'sim'] if k in data), None)
            obs_key = next((k for k in [f'{b_key}_obs', 'streamflow_obs', 'obs'] if k in data), None)
            if sim_key is not None:
                sim_arr = np.asarray(data[sim_key]).flatten()
                obs_arr = np.asarray(data[obs_key]).flatten() if obs_key is not None else np.full_like(sim_arr, np.nan)
                dates = pd.to_datetime(data.get('date', data.get('dates', pd.date_range('2011-01-01', periods=len(sim_arr), freq='D'))))
            else:
                obs_arr = np.asarray(data.get('streamflow_obs', data.get('obs'))).flatten()
                sim_arr = np.asarray(data.get('streamflow_sim', data.get('sim'))).flatten()
                dates = pd.to_datetime(data.get('date', data.get('dates', pd.date_range('2011-01-01', periods=len(sim_arr), freq='D'))))
        chosen_lt = lead_time

    elif isinstance(data, (xr.Dataset, xr.DataArray)):
        ds = data
        if 'basin' in ds.coords and basin_id is not None and basin_id in ds['basin'].values:
            ds = ds.sel(basin=basin_id)
        if 'freq' in ds.coords:
            ds = ds.sel(freq='1D')

        available_lts = ds['time_step'].values if 'time_step' in ds.coords else [0]
        chosen_lt = lead_time if lead_time in available_lts else available_lts[-1 if lead_time > max(available_lts) else 0]

        if 'streamflow_sim' in ds:
            sim_da = ds['streamflow_sim']
            sim_arr = sim_da.sel(time_step=chosen_lt).values.flatten() if ('time_step' in sim_da.dims or 'time_step' in sim_da.coords) else sim_da.values.flatten()
        else:
            sim_arr = ds.values.flatten()

        if 'streamflow_obs' in ds:
            obs_da = ds['streamflow_obs']
            obs_arr = obs_da.sel(time_step=chosen_lt).values.flatten() if ('time_step' in obs_da.dims or 'time_step' in obs_da.coords) else obs_da.values.flatten()
        elif obs_data is not None:
            obs_arr = np.asarray(obs_data.values if hasattr(obs_data, 'values') else obs_data).flatten()
        else:
            obs_arr = np.full_like(sim_arr, np.nan)

        dates = pd.to_datetime(ds['date'].values if 'date' in ds.coords else np.arange(len(sim_arr)))

    elif isinstance(data, pd.DataFrame):
        df_sub = data[data['basin_id'] == basin_id] if ('basin_id' in data.columns and basin_id is not None) else data
        sim_col = next((c for c in ['streamflow_sim', 'sim'] if c in df_sub.columns), df_sub.columns[1] if len(df_sub.columns) > 1 else df_sub.columns[0])
        obs_col = next((c for c in ['streamflow_obs', 'obs'] if c in df_sub.columns), df_sub.columns[0])
        sim_arr = df_sub[sim_col].values
        obs_arr = df_sub[obs_col].values
        dates = pd.to_datetime(df_sub.index if isinstance(df_sub.index, pd.DatetimeIndex) else df_sub.get('date', pd.date_range('2011-01-01', periods=len(sim_arr), freq='D')))
        chosen_lt = lead_time
    else:
        raise TypeError(f"Unsupported data type for extract_rolling_hydrograph: {type(data)}")

    issue_dates = pd.to_datetime(dates)
    target_dates = issue_dates + pd.to_timedelta(int(chosen_lt), unit='D')
    index_dates = target_dates if align_target_dates else issue_dates

    df_res = pd.DataFrame({
        'streamflow_obs': obs_arr,
        'streamflow_sim': sim_arr,
        'obs': obs_arr,
        'sim': sim_arr,
        'issue_date': issue_dates,
        'target_date': target_dates,
        'basin_id': basin_id or 'unknown',
        'lead_time': chosen_lt
    }, index=pd.DatetimeIndex(index_dates, name='date'))

    if start_date is not None:
        df_res = df_res[df_res.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_res = df_res[df_res.index <= pd.to_datetime(end_date)]
    if window_days is not None and len(df_res) > window_days:
        df_res = df_res.iloc[:window_days]

    return df_res


def plot_rolling_hydrograph(
    model_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    basin_id: Optional[str] = None,
    lead_time: int = 5,
    finetune_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    secondary_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    window_days: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    model_name: str = 'Base Model',
    finetune_name: str = 'Fine-Tuned Model',
    model_labels: Optional[Tuple[str, str]] = None,
    align_target_dates: bool = True,
    ax: Optional[plt.Axes] = None,
    show_metrics: bool = True,
    data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot continuous rolling lead-time hydrograph comparing predictions against observed streamflow."""
    primary_data = model_data if model_data is not None else data
    if primary_data is None:
        raise ValueError("Must provide model_data or data to plot_rolling_hydrograph")

    sec_data = finetune_data if finetune_data is not None else secondary_data
    lbl_base = model_labels[0] if model_labels is not None else model_name
    lbl_ft = model_labels[1] if model_labels is not None else finetune_name

    df_base = extract_rolling_hydrograph(
        primary_data, basin_id=basin_id, lead_time=lead_time,
        start_date=start_date, end_date=end_date, window_days=window_days,
        align_target_dates=align_target_dates
    )

    fig = ax.get_figure() if ax is not None else plt.subplots(figsize=figsize)[0]
    ax = ax if ax is not None else fig.axes[0]

    obs_vals = df_base['streamflow_obs'].values
    sim_base = df_base['streamflow_sim'].values
    dates = df_base.index

    ax.plot(dates, obs_vals, label='Observed Streamflow ($Q_{obs}$)', color='#111827', linewidth=2.0, alpha=0.85)

    base_m = compute_hydro_metrics(obs_vals, sim_base)
    base_lbl = f"{lbl_base} (L={lead_time}d) [NSE={base_m['NSE']:+.3f}, KGE={base_m['KGE']:+.3f}, r={base_m['Pearson-r']:+.3f}]" if (show_metrics and not np.isnan(base_m['NSE'])) else f"{lbl_base} (L={lead_time}d)"
    ax.plot(dates, sim_base, label=base_lbl, color='#1d4ed8', linestyle='--', linewidth=1.8, alpha=0.9)

    if sec_data is not None:
        df_ft = extract_rolling_hydrograph(
            sec_data, basin_id=basin_id, lead_time=lead_time,
            start_date=start_date, end_date=end_date, window_days=window_days,
            align_target_dates=align_target_dates
        )
        sim_ft = df_ft['streamflow_sim'].values
        ft_m = compute_hydro_metrics(obs_vals, sim_ft)
        ft_lbl = f"{lbl_ft} (L={lead_time}d) [NSE={ft_m['NSE']:+.3f}, KGE={ft_m['KGE']:+.3f}, r={ft_m['Pearson-r']:+.3f}]" if (show_metrics and not np.isnan(ft_m['NSE'])) else f"{lbl_ft} (L={lead_time}d)"
        ax.plot(dates, sim_ft, label=ft_lbl, color='#dc2626', linestyle='-.', linewidth=1.8, alpha=0.95)

    date_lbl = "Verification Target Date (Issue Date + L days)" if align_target_dates else "Forecast Issue Date"
    ax.set_xlabel(date_lbl, fontsize=11, fontweight='bold')
    ax.set_ylabel("Streamflow Discharge (mm/day)", fontsize=11, fontweight='bold')

    if title is None:
        basin_str = f" for Basin {basin_id}" if basin_id else ""
        title = f"Continuous Rolling Hydrograph{basin_str} (Lead Time L={lead_time} days)"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    return fig, ax


def interactive_rolling_hydrograph(
    base_model_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    finetune_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    secondary_data: Optional[Union[xr.Dataset, xr.DataArray, pd.DataFrame, Dict[str, Any]]] = None,
    default_basin: Optional[str] = None,
    default_lead_time: int = 5,
    available_basins: Optional[List[str]] = None,
    basins: Optional[List[str]] = None,
    model_name: str = 'Base Model',
    finetune_name: str = 'Fine-Tuned Model',
    model_labels: Optional[Tuple[str, str]] = None,
    window_days: Optional[int] = 365,
    **kwargs
):
    """Render interactive or static rolling hydrograph widget across basins and lead times."""
    actual_model_data = base_model_data if base_model_data is not None else kwargs.get('data')

    b_list = available_basins or basins
    if b_list is None:
        ds = actual_model_data if actual_model_data is not None else finetune_data
        if ds is not None and hasattr(ds, 'coords') and 'basin' in ds.coords:
            b_list = list(ds['basin'].values)
        elif isinstance(ds, dict):
            b_list = list(ds.keys())
        else:
            b_list = ['camels_13235000', 'camels_04115265', 'camels_07057500', 'camels_12115000', 'camels_12377150']

    b_default = default_basin or (b_list[0] if b_list else 'camels_13235000')

    try:
        import ipywidgets as widgets
        from IPython.display import display

        basin_dropdown = widgets.Dropdown(options=b_list, value=b_default if b_default in b_list else b_list[0], description='Basin:')
        lead_slider = widgets.IntSlider(value=default_lead_time, min=0, max=10, step=1, description='Lead Time (d):')
        window_slider = widgets.IntSlider(value=window_days or 365, min=30, max=730, step=30, description='Window (days):')

        def _update(basin, lead, win):
            plt.close('all')
            plot_rolling_hydrograph(
                model_data=actual_model_data,
                basin_id=basin,
                lead_time=lead,
                finetune_data=finetune_data if finetune_data is not None else secondary_data,
                window_days=win,
                model_name=model_name,
                finetune_name=finetune_name,
                model_labels=model_labels
            )
            plt.show()

        out = widgets.interactive_output(_update, {'basin': basin_dropdown, 'lead': lead_slider, 'win': window_slider})
        display(widgets.VBox([widgets.HBox([basin_dropdown, lead_slider, window_slider]), out]))
    except Exception:
        plot_rolling_hydrograph(
            model_data=actual_model_data,
            basin_id=b_default,
            lead_time=default_lead_time,
            finetune_data=finetune_data if finetune_data is not None else secondary_data,
            window_days=window_days,
            model_name=model_name,
            finetune_name=finetune_name,
            model_labels=model_labels
        )
        plt.show()