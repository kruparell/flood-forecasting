#!/usr/bin/env python3
"""Evaluation Pipeline for Data Assimilation & A-RLSTM on Ungauged Basins.

This script evaluates model performance (Data Assimilation and Autoregressive LSTM)
on out-of-sample (ungauged) basins that were not included in hyperparameter optimization or training.

Key Features:
1. Automatic hyperparameter & used-basin extraction with CLI override options.
2. Basin exclusion & valid interval filtering with file existence check.
3. Multi-processing parallel worker pool with thread-safety and device handling.
4. Comprehensive 9-metric evaluation using googlehydrology.evaluation.metrics.
5. Export of daily timeseries, detailed gauge-level metrics with deltas, and mean/median lead-time summaries.
"""

import os
import sys
import glob
import re
import yaml
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Preload scientific libraries before modifying sys.path or spawning processes
import torch
import numpy as np
import pandas as pd
import xarray as xr

# Add flood-forecasting repository path to sys.path
sys.path.append('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from googlehydrology.utils.config import Config
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.modelzoo.arlstm import ARLSTM
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.evaluation import metrics as gh_metrics
from multimet_helpers import prepare_multimet_batch

from tutorial.utils import (
    load_basin_list as shared_load_basin_list,
    load_caravan_attributes,
    load_model_checkpoint,
    build_assim_config,
    compute_all_metrics,
    unnormalize_streamflow,
)

# Default Paths & Setup
DEFAULT_CARAVAN_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/Caravan-nc')
DEFAULT_PRETRAINED_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs')
DEFAULT_PARAM_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/large_scale_param_selection_results')
DEFAULT_BASIN_LIST = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/basin-lists/CaravansV2_basin_valid_intervals.csv')
DEFAULT_OUTPUT_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/ungauged_evaluation_results')

DEFAULT_START_DATE = '2017-01-01'
DEFAULT_END_DATE = '2017-01-15'
ALL_LEADTIMES = list(range(1, 8))


def extract_best_config_and_used_basins(param_selection_dir: Path, override_config_id: str = None):
    """Extracts top hyperparameter config and set of basins used in optimization."""
    rankings_path = param_selection_dir / "summary_overall_param_rankings.csv"
    detailed_path = param_selection_dir / "detailed_basin_parameter_eval.csv"

    if not rankings_path.exists() or not detailed_path.exists():
        raise FileNotFoundError(f"Missing parameter selection CSVs in {param_selection_dir}")

    rankings_df = pd.read_csv(rankings_path)
    if rankings_df.empty:
        raise ValueError(f"Rankings file {rankings_path} is empty.")

    config_id = override_config_id or rankings_df.iloc[0]["Config ID"]

    # Parse config_id using regex
    pattern = r"Config_\d+_LR(?P<lr>[\d\.e\-]+)_Ep(?P<ep>\d+)_W(?P<w>\d+)_Bg(?P<bg>[\d\.e\-]+)_(?P<target>c_keys|h_keys)"
    match = re.search(pattern, config_id)

    if match:
        lr = float(match.group("lr"))
        ep = int(match.group("ep"))
        w = int(match.group("w"))
        bg = float(match.group("bg"))
        target_type = match.group("target")
        targets = ['c_n_forecast'] if target_type == 'c_keys' else ['h_n_forecast']
    else:
        # Fallback to standard default parameters if non-standard ID
        lr, ep, w, bg = 0.01, 5, 7, 1e-8
        targets = ['c_n_forecast']

    p_spec = {
        'config_id': config_id,
        'learning_rate': lr,
        'epochs': ep,
        'assimilation_window': w,
        'bg_regularization_weight': bg,
        'assimilation_targets': targets,
        'distance_from_forecast': 1
    }

    # Extract unique used basin IDs safely
    used_basins = set()
    if detailed_path.exists() and detailed_path.stat().st_size > 0:
        try:
            detailed_df = pd.read_csv(detailed_path)
            if 'Basin ID' in detailed_df.columns:
                used_basins = set(detailed_df['Basin ID'].dropna().astype(str).str.strip().unique())
        except Exception:
            pass

    return p_spec, used_basins


def load_and_sample_ungauged_basins(
    basin_list_path: Path,
    caravan_base_dir: Path,
    used_basins: set,
    start_date: str,
    end_date: str,
    num_basins: int = 50,
    random_seed: int = 42
) -> list:
    """Filters out used basins, checks coverage and NetCDF file existence, and samples ungauged basins."""
    if not basin_list_path.exists():
        raise FileNotFoundError(f"Basin list file not found: {basin_list_path}")

    eval_start = pd.to_datetime(start_date)
    eval_end = pd.to_datetime(end_date)

    if basin_list_path.suffix.lower() == ".csv":
        df = pd.read_csv(basin_list_path)
        if {"valid_start_date", "valid_end_date"}.issubset(df.columns):
            df["valid_start_date"] = pd.to_datetime(df["valid_start_date"])
            df["valid_end_date"] = pd.to_datetime(df["valid_end_date"])
            valid_mask = (df["valid_start_date"] <= eval_start) & (df["valid_end_date"] >= eval_end)
            df = df[valid_mask]

        if "basin_id" in df.columns:
            basins = df["basin_id"].dropna().astype(str).str.strip().unique().tolist()
        else:
            basins = df.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
    else:
        with open(basin_list_path, "r") as f:
            basins = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Exclude basins used during hyperparameter optimization / tuning
    ungauged_pool = [b for b in basins if b not in used_basins]

    # Verify disk existence of NetCDF files in Caravans efficiently via single directory scan
    timeseries_dir = caravan_base_dir / 'timeseries' / 'netcdf'
    existing_files = list(timeseries_dir.glob('**/*.nc'))
    existing_basin_set = {f.stem for f in existing_files}
    existing_basins = [b for b in ungauged_pool if b in existing_basin_set]

    if not existing_basins:
        raise ValueError("No ungauged basins with existing NetCDF files were found!")

    # Reproducible sampling
    if 0 < num_basins < len(existing_basins):
        rng = np.random.RandomState(random_seed)
        sampled_basins = sorted(list(rng.choice(existing_basins, size=num_basins, replace=False)))
    else:
        sampled_basins = sorted(existing_basins)

    return sampled_basins


def load_caravan_attributes(caravan_base_dir: Path) -> pd.DataFrame:
    """Loads and combines all CSV static attribute files across all source folders."""
    attributes_dir = caravan_base_dir / 'attributes'
    source_dfs = []

    if not attributes_dir.exists():
        return pd.DataFrame()

    for source_folder in attributes_dir.iterdir():
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
    return combined.loc[:, ~combined.columns.duplicated()]


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
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


def build_assim_config(p_cfg: dict, forecast_lead_days: int = 7) -> AssimilationConfig:
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


def process_single_ungauged_basin_task(kwargs: dict) -> tuple:
    """Worker task evaluating baseline vs model (DA or A-RLSTM) predictions per basin."""
    b_id = kwargs['b_id']
    p_spec = kwargs['p_spec']
    model_cfg_dict = kwargs['model_cfg_dict']
    checkpoint_path = kwargs['checkpoint_path']
    scaler_path = kwargs['scaler_path']
    q_mean = kwargs['q_mean']
    q_std = kwargs['q_std']
    caravan_attrs = kwargs['caravan_attrs']
    issue_dates = kwargs['issue_dates']
    device_str = kwargs['device_str']
    caravan_base_dir = Path(kwargs['caravan_base_dir'])
    all_leadtimes = kwargs['all_leadtimes']
    model_type = kwargs.get('model_type', 'da')

    device = torch.device(device_str)
    if device_str == 'cpu':
        torch.set_num_threads(1)

    model_cfg = Config(model_cfg_dict)

    if model_type == 'arlstm':
        model = ARLSTM(model_cfg)
    else:
        model = MeanEmbeddingForecastLSTM(model_cfg)

    ds_b = None
    scaler = None

    try:
        raw_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
        model.load_state_dict(clean_ckpt, strict=True)
        model.to(device)
        model.eval()

        scaler = xr.open_dataset(scaler_path)
        assim_engine = Assimilation(build_assim_config(p_spec)) if model_type == 'da' else None

        timeseries_dir = caravan_base_dir / 'timeseries' / 'netcdf'
        matches = glob.glob(f"{timeseries_dir}/**/{b_id}.nc", recursive=True)
        if not matches:
            return [], []
        ds_b = xr.open_dataset(matches[0])

        records_base = []
        records_sim = []

        for dt in issue_dates:
            batch_b = prepare_multimet_batch(
                mode='multimet_0_and_1_to_7',
                basin_id=b_id,
                issue_date=dt,
                cfg=model_cfg,
                scaler=scaler,
                caravan_attrs=caravan_attrs,
                ds_caravan=ds_b,
                forecast_product='HRES',
                hindcast_window_days=358,
                forecast_lead_days=7
            )

            dt_obj = pd.to_datetime(dt)

            # Baseline forecast pass
            with torch.no_grad():
                base_out = model(batch_b)
                q_base_7_norm = base_out['y_hat'][0, -7:, 0].cpu().numpy()
            q_base_7 = np.maximum(0.1, q_base_7_norm * q_std + q_mean)

            # Active Model Pass (DA vs A-RLSTM)
            if model_type == 'da':
                sim_out = assim_engine.assimilate(model, batch_b, verbose=False)
                q_sim_7_norm = sim_out['y_hat'][0, -7:, 0].cpu().numpy()
            elif model_type == 'arlstm':
                with torch.no_grad():
                    sim_out = model(batch_b)
                    q_sim_7_norm = sim_out['y_hat'][0, -7:, 0].cpu().numpy()
            else:
                q_sim_7_norm = q_base_7_norm

            q_sim_7 = np.maximum(0.1, q_sim_7_norm * q_std + q_mean)

            for L_idx, L in enumerate(all_leadtimes):
                valid_date = (dt_obj + pd.Timedelta(days=L)).strftime('%Y-%m-%d')
                records_base.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_base_7[L_idx]})
                records_sim.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_sim_7[L_idx]})

        df_base = pd.DataFrame(records_base)
        df_sim = pd.DataFrame(records_sim)

        task_records = []
        records_ts = []

        ds_dates = pd.Index(pd.to_datetime(ds_b['date'].values).strftime('%Y-%m-%d'))

        for L in all_leadtimes:
            df_b_base = df_base[df_base['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
            df_b_sim = df_sim[df_sim['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
            valid_dates = df_b_base.index.intersection(ds_dates)

            if len(valid_dates) == 0:
                continue

            obs_q = ds_b['streamflow'].sel(date=valid_dates).values.astype(np.float32)
            q_base_seq = df_b_base.loc[valid_dates, 'q_sim'].values
            q_sim_seq = df_b_sim.loc[valid_dates, 'q_sim'].values

            m_base = compute_all_metrics(obs_q, q_base_seq)
            m_sim = compute_all_metrics(obs_q, q_sim_seq)

            record = {
                'Config ID': p_spec['config_id'],
                'Basin ID': b_id,
                'Model Type': model_type.upper(),
                'Lead Time (Days)': L,
            }

            for metric_key in ['NSE', 'KGE', 'Alpha-NSE', 'Beta-KGE', 'Pearson-r', 'RMSE', 'FHV', 'FMS', 'FLV']:
                val_base = m_base.get(metric_key, np.nan)
                val_sim = m_sim.get(metric_key, np.nan)
                delta = val_sim - val_base if not (np.isnan(val_sim) or np.isnan(val_base)) else np.nan

                record[f'Base {metric_key}'] = val_base
                record[f'Sim {metric_key}'] = val_sim
                record[f'Delta {metric_key}'] = round(delta, 4) if not np.isnan(delta) else np.nan

            task_records.append(record)

            for d_idx, v_date in enumerate(valid_dates):
                records_ts.append({
                    'Config ID': p_spec['config_id'],
                    'Basin ID': b_id,
                    'Model Type': model_type.upper(),
                    'Lead Time (Days)': L,
                    'Valid Date': v_date,
                    'q_obs': obs_q[d_idx],
                    'q_base': q_base_seq[d_idx],
                    'q_sim': q_sim_seq[d_idx]
                })

        return task_records, records_ts

    except Exception as e:
        logging.error(f"Error processing basin {b_id}: {e}")
        return [], []
    finally:
        if ds_b is not None:
            ds_b.close()
        if scaler is not None:
            scaler.close()


def main():
    parser = argparse.ArgumentParser(description="Ungauged Basin Evaluation Pipeline for DA & A-RLSTM")
    parser.add_argument('--param_selection_dir', type=str, default=str(DEFAULT_PARAM_DIR), help="Path to param selection results")
    parser.add_argument('--config_id', type=str, default=None, help="Optional explicit Config ID override")
    parser.add_argument('--num_basins', type=int, default=50, help="Number of ungauged basins to sample")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for basin sampling")
    parser.add_argument('--num_workers', type=int, default=4, help="Parallel worker count")
    parser.add_argument('--start_date', type=str, default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory path")
    parser.add_argument('--caravan_base_dir', type=str, default=str(DEFAULT_CARAVAN_DIR), help="Path to Caravans base directory")
    parser.add_argument('--basin_list_path', type=str, default=str(DEFAULT_BASIN_LIST), help="Path to basin list file")
    parser.add_argument('--pretrained_dir', type=str, default=str(DEFAULT_PRETRAINED_DIR), help="Path to pretrained model folder")
    parser.add_argument('--model_type', type=str, default='da', choices=['da', 'arlstm'], help="Model evaluation mode ('da' or 'arlstm')")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"evaluate_ungauged_basins_{args.model_type}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    )
    logger = logging.getLogger(__name__)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("======================================================================")
    logger.info(f" Ungauged Basin Evaluation Pipeline ({args.model_type.upper()} Mode) ")
    logger.info("======================================================================")
    logger.info(f"Compute Device: {device_str}")
    logger.info(f"Parallel Worker Processes: {args.num_workers}")
    logger.info(f"Evaluation Date Range: {args.start_date} to {args.end_date}")

    # Extract hyperparameter spec and used basins
    param_dir = Path(args.param_selection_dir)
    p_spec, used_basins = extract_best_config_and_used_basins(param_dir, override_config_id=args.config_id)
    logger.info(f"Selected Config Spec: {p_spec}")
    logger.info(f"Identified {len(used_basins)} used basins to exclude from ungauged sampling.")

    # Sample ungauged basins
    caravan_base_dir = Path(args.caravan_base_dir)
    basin_list_path = Path(args.basin_list_path)
    ungauged_basins = load_and_sample_ungauged_basins(
        basin_list_path=basin_list_path,
        caravan_base_dir=caravan_base_dir,
        used_basins=used_basins,
        start_date=args.start_date,
        end_date=args.end_date,
        num_basins=args.num_basins,
        random_seed=args.random_seed
    )
    logger.info(f"Sampled {len(ungauged_basins)} ungauged basins: {ungauged_basins[:5]} ...")

    # Load Model Metadata & Scaler
    pretrained_dir = Path(args.pretrained_dir)
    config_path = pretrained_dir / 'config.yml'
    checkpoint_path = pretrained_dir / 'model_epoch085.pt'
    scaler_path = pretrained_dir / 'scaler.nc'

    with open(config_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)

    scaler = xr.open_dataset(scaler_path)
    caravan_attrs = load_caravan_attributes(caravan_base_dir)

    if 'streamflow_sim' in scaler:
        q_mean = float(scaler['streamflow_sim'].sel(parameter='center').values)
        q_std = float(scaler['streamflow_sim'].sel(parameter='scale').values)
    else:
        q_mean = float(scaler['streamflow_mean'].values)
        q_std = float(scaler['streamflow_std'].values)
    scaler.close()

    issue_dates = list(pd.date_range(args.start_date, args.end_date, freq='D').strftime('%Y-%m-%d'))
    logger.info(f"Evaluating across {len(issue_dates)} issue dates: {issue_dates[0]} to {issue_dates[-1]}")

    # Build Parallel Tasks
    tasks = []
    for b_id in ungauged_basins:
        tasks.append({
            'b_id': b_id,
            'p_spec': p_spec,
            'model_cfg_dict': model_cfg_dict,
            'checkpoint_path': str(checkpoint_path),
            'scaler_path': str(scaler_path),
            'q_mean': q_mean,
            'q_std': q_std,
            'caravan_attrs': caravan_attrs,
            'issue_dates': issue_dates,
            'device_str': device_str,
            'caravan_base_dir': str(caravan_base_dir),
            'all_leadtimes': ALL_LEADTIMES,
            'model_type': args.model_type
        })

    all_metrics_records = []
    all_ts_records = []

    logger.info(f"Starting parallel evaluation across {len(tasks)} tasks using {args.num_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_ungauged_basin_task, task): task['b_id'] for task in tasks}
        for future in as_completed(futures):
            b_id = futures[future]
            try:
                m_recs, ts_recs = future.result()
                if m_recs:
                    all_metrics_records.extend(m_recs)
                    all_ts_records.extend(ts_recs)
                    logger.info(f"Successfully evaluated ungauged basin: {b_id}")
                else:
                    logger.warning(f"No valid evaluation records produced for basin: {b_id}")
            except Exception as e:
                logger.error(f"Task for basin {b_id} failed with exception: {e}")

    if not all_metrics_records:
        logger.error("No metrics were generated. Exiting.")
        return

    # Write Output Files
    df_metrics = pd.DataFrame(all_metrics_records)
    metrics_csv_path = out_dir / "metrics_ungauged_basins.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved gauge-level metrics to: {metrics_csv_path}")

    df_ts = pd.DataFrame(all_ts_records)
    ts_csv_path = out_dir / "timeseries_predictions_ungauged.csv"
    df_ts.to_csv(ts_csv_path, index=False)
    logger.info(f"Saved timeseries predictions to: {ts_csv_path}")

    # Aggregations: Mean and Median across basins grouped by Lead Time
    numeric_cols = [c for c in df_metrics.columns if c not in ['Config ID', 'Basin ID', 'Model Type', 'Lead Time (Days)']]
    df_mean = df_metrics.groupby('Lead Time (Days)')[numeric_cols].mean().reset_index()
    df_median = df_metrics.groupby('Lead Time (Days)')[numeric_cols].median().reset_index()

    df_summary = pd.DataFrame({'Lead Time (Days)': ALL_LEADTIMES})
    for col in numeric_cols:
        df_summary[f"{col} (Mean)"] = df_mean[col]
        df_summary[f"{col} (Median)"] = df_median[col]

    summary_csv_path = out_dir / "summary_ungauged_metrics_by_leadtime.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved mean and median summary metrics by lead time to: {summary_csv_path}")
    logger.info("Evaluation Pipeline Completed Successfully!")


if __name__ == '__main__':
    main()
