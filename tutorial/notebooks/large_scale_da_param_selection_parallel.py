#!/usr/bin/env python3
"""Large Scale Data Assimilation Parameter Selection & Evaluation Pipeline (Parallel & Shard-aware).

This script executes large-scale Data Assimilation across catchment basins
from a specified basin list over an evaluation date range to evaluate and rank DA parameter sets.

Key Enhancements for Parallelism & Borg Sharding:
1. Multi-Processing Worker Pool: Parallelizes (basin, config) task evaluations across multiple CPU/GPU worker processes.
2. Borg Task Sharding: Supports --shard_idx and --num_shards (or auto-detects BORG_TASK_INDEX and BORG_REPLICA_COUNT)
   to shard basin lists across distributed Borg replicas.
3. Thread/Process Safety: Isolates dataset file handles and PyTorch model instances per worker process.
"""

import os
import sys
from pathlib import Path
import glob
import yaml
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import itertools

# Preload foundational scientific libraries before modifying sys.path
import torch
import numpy as np
import pandas as pd
import xarray as xr

# Add flood-forecasting repository path to sys.path
sys.path.append('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from googlehydrology.utils.config import Config
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.evaluation.assimilation import Assimilation
from multimet_helpers import prepare_multimet_batch

from tutorial.utils import (
    load_basin_list,
    load_caravan_attributes,
    build_assim_config,
    open_file,
    file_exists,
)


# ==============================================================================
# 1. DIRECTORY, DEVICE & DEFAULT SETUP
# ==============================================================================
CARAVAN_BASE_DIR = Path('/usr/local/google/home/kruparell/Caravans_V2')
MULTIMET_DIR = Path('/usr/local/google/home/kruparell/Caravans_MultiMet')
PRETRAINED_DIR = '/usr/local/google/home/kruparell/flood-forecasting/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs'
OUTPUT_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/large_scale_param_selection_results')
BASIN_LIST_PATH = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/basin-lists/CaravansV2_basin_valid_intervals.csv')

# Evaluation Date Range and Lead Times
START_DATE = '2017-01-01'
END_DATE = '2017-01-15'
ALL_LEADTIMES = list(range(1, 8))  # Lead times 1..7

# Sampling & Reproducibility Settings
NUM_BASINS = 2      # Target number of basins to evaluate from basin list
RANDOM_SEED = 42       # Fixed seed for reproducible random subsetting

param_grid = {
    'learning_rate': [1e-2], # 1e-1, 
    'epochs': [5], # , 15
    'bg_regularization_weight': [1e-8], # 1e-6, 
    'assimilation_targets': ['c_n_forecast'], # 'c_n_hindcast', 
    'assimilation_window': [7],
    'distance_from_forecast': [1],

}

def generate_param_configs(grid: dict) -> list:
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    
    for idx, combo in enumerate(itertools.product(*values)):
        cfg = dict(zip(keys, combo))
        
        # Build readable target label for Config ID
        targets = cfg['assimilation_targets']
        target_label = "h_keys" if "h_hindcast" in targets[1] else "c_keys"
        
        cfg['config_id'] = (
            f"Config_{idx+1}_LR{cfg['learning_rate']}_Ep{cfg['epochs']}_"
            f"W{cfg['assimilation_window']}_Bg{cfg['bg_regularization_weight']}_{target_label}"
        )
        configs.append(cfg)
    return configs

PARAM_CONFIGS = generate_param_configs(param_grid)


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
from tutorial.utils import compute_all_metrics

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Computes NSE and KGE accuracy metrics between observations and simulation."""
    res = compute_all_metrics(obs, sim)
    return {'NSE': res.get('NSE', np.nan), 'KGE': res.get('KGE', np.nan)}


# ==============================================================================
# 3. WORKER FUNCTION FOR PARALLEL EXECUTION
# ==============================================================================
def process_single_basin_config_task(kwargs: dict) -> list:
    """Worker task evaluating one (basin_id, parameter_config) pair.
    
    Self-contained: loads model checkpoint, opens dataset handle, performs
    base + Data Assimilation forward passes across issue dates, and computes metrics.
    """
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

    config_id = p_spec['config_id']
    device = torch.device(device_str)

    # Prevent thread over-subscription inside multiprocessing workers on CPU
    if device_str == 'cpu':
        torch.set_num_threads(1)

    # 1. Load Pretrained Model & Scaler strictly within worker
    model_cfg = Config(model_cfg_dict)
    model = MeanEmbeddingForecastLSTM(model_cfg)
    raw_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    model.to(device)
    model.eval()

    scaler = xr.open_dataset(scaler_path)
    assim_engine = Assimilation(build_assim_config(p_spec))

    # 2. Locate and open NetCDF for basin
    try:
        timeseries_dir = caravan_base_dir / 'timeseries' / 'netcdf'
        matches = glob.glob(f"{timeseries_dir}/**/{b_id}.nc", recursive=True)
        if not matches:
            scaler.close()
            return [], []
        ds_b = xr.open_dataset(matches[0])
    except Exception:
        scaler.close()
        return [], []

    records_base = []
    records_da = []

    # 3. Iterate issue dates
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

        # Baseline forecast
        with torch.no_grad():
            base_out = model(batch_b)
            q_base_7_norm = base_out['y_hat'][0, -7:, 0].cpu().numpy()
        q_base_7 = np.maximum(0.1, q_base_7_norm * q_std + q_mean)

        # Data Assimilation forecast
        da_out = assim_engine.assimilate(model, batch_b, verbose=False)
        q_da_7_norm = da_out['y_hat'][0, -7:, 0].cpu().numpy()
        q_da_7 = np.maximum(0.1, q_da_7_norm * q_std + q_mean)

        for L_idx, L in enumerate(all_leadtimes):
            valid_date = (dt_obj + pd.Timedelta(days=L)).strftime('%Y-%m-%d')
            records_base.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_base_7[L_idx]})
            records_da.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_da_7[L_idx]})

    df_base = pd.DataFrame(records_base)
    df_da = pd.DataFrame(records_da)

    task_records = []
    records_ts = []

    ds_dates = pd.Index(pd.to_datetime(ds_b['date'].values).strftime('%Y-%m-%d'))

    for L in all_leadtimes:
        df_b_base = df_base[df_base['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
        df_b_da = df_da[df_da['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
        valid_dates = df_b_base.index.intersection(ds_dates)

        if len(valid_dates) == 0:
            continue

        obs_q = ds_b['streamflow'].sel(date=valid_dates).values.astype(np.float32)
        q_base_seq = df_b_base.loc[valid_dates, 'q_sim'].values
        q_da_seq = df_b_da.loc[valid_dates, 'q_sim'].values

        m_base = compute_metrics(obs_q, q_base_seq)
        m_da = compute_metrics(obs_q, q_da_seq)

        task_records.append({
            'Config ID': config_id,
            'Basin ID': b_id,
            'Lead Time (Days)': L,
            'Base NSE': round(m_base['NSE'], 3) if not np.isnan(m_base['NSE']) else np.nan,
            'Base KGE': round(m_base['KGE'], 3) if not np.isnan(m_base['KGE']) else np.nan,
            'DA NSE': round(m_da['NSE'], 3) if not np.isnan(m_da['NSE']) else np.nan,
            'DA KGE': round(m_da['KGE'], 3) if not np.isnan(m_da['KGE']) else np.nan,
            'NSE Delta': round(m_da['NSE'] - m_base['NSE'], 3) if not (np.isnan(m_da['NSE']) or np.isnan(m_base['NSE'])) else np.nan,
            'KGE Delta': round(m_da['KGE'] - m_base['KGE'], 3) if not (np.isnan(m_da['KGE']) or np.isnan(m_base['KGE'])) else np.nan,
        })

        for d_idx, v_date in enumerate(valid_dates):
            records_ts.append({
                'Config ID': config_id,
                'Basin ID': b_id,
                'Lead Time (Days)': L,
                'Valid Date': v_date,
                'q_obs': obs_q[d_idx],
                'q_base': q_base_seq[d_idx],
                'q_da': q_da_seq[d_idx]
            })


    ds_b.close()
    scaler.close()
    return task_records, records_ts


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Parallel Large Scale DA Parameter Selection Pipeline")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel worker processes (default: 32)")
    parser.add_argument('--shard_idx', type=int, default=0, help="Shard index (0-based) for Borg task sharding")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards for Borg task sharding")
    parser.add_argument('--num_basins', type=int, default=NUM_BASINS, help="Number of basins to sample from list")
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help="Random seed for basin sampling")
    parser.add_argument('--start_date', type=str, default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end_date', type=str, default=END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR), help="Output directory path")
    parser.add_argument('--caravan_base_dir', type=str, default=str(CARAVAN_BASE_DIR), help="Path to Caravans base directory")
    parser.add_argument('--basin_list_path', type=str, default=str(BASIN_LIST_PATH), help="Path to basin list txt or csv file")
    args = parser.parse_args()

    caravan_base_dir = Path(args.caravan_base_dir)
    basin_list_path = Path(args.basin_list_path)

    # Detect Borg environment variables if present
    shard_idx = int(os.environ.get('BORG_TASK_INDEX', args.shard_idx))
    num_shards = int(os.environ.get('BORG_REPLICA_COUNT', os.environ.get('BORG_TASK_NUM', args.num_shards)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logger Setup
    log_filename = f"large_scale_param_selection_shard_{shard_idx}.log" if num_shards > 1 else "large_scale_param_selection.log"
    log_file = out_dir / log_filename

    class UnbufferedStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[UnbufferedStreamHandler(sys.stdout), logging.FileHandler(log_file, mode='a')]
    )
    logger = logging.getLogger(__name__)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("======================================================================")
    logger.info(" Large-Scale DA Parameter Selection Pipeline (Parallel & Sharded)    ")
    logger.info("======================================================================")
    logger.info(f"Hardware Compute Device: {device_str}")
    logger.info(f"Parallel Worker Processes: {args.num_workers}")
    logger.info(f"Borg Task Sharding: Shard {shard_idx + 1} of {num_shards}")
    logger.info(f"Full Caravans Path: {caravan_base_dir}")
    logger.info(f"Basin List Path: {basin_list_path}")
    logger.info(f"Evaluation Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Target Lead Times: {ALL_LEADTIMES}")
    logger.info(f"Log File Location: {log_file}")

    # Load and shard basin list
    all_eval_basins = load_basin_list(basin_list_path,  start_date=args.start_date, end_date= args.end_date,
 num_basins=args.num_basins, random_seed=args.random_seed)

    eval_basins = all_eval_basins[shard_idx::num_shards]
    logger.info(f"Total sampled basins: {len(all_eval_basins)}. Assigned to this shard ({shard_idx}): {len(eval_basins)} basins.")
    logger.info(f"Testing {len(PARAM_CONFIGS)} Parameter Sets: {[c['config_id'] for c in PARAM_CONFIGS]}\n")

    # Metadata & Scaler checks
    config_path = os.path.join(PRETRAINED_DIR, 'config.yml')
    checkpoint_path = os.path.join(PRETRAINED_DIR, 'model_epoch085.pt')
    scaler_path = os.path.join(PRETRAINED_DIR, 'scaler.nc')

    with open_file(config_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)

    with open_file(scaler_path, 'rb') as f:
        scaler = xr.open_dataset(f).load()
    caravan_attrs = load_caravan_attributes(caravan_base_dir)

    if 'streamflow_sim' in scaler:
        q_mean = float(scaler['streamflow_sim'].sel(parameter='center').values)
        q_std = float(scaler['streamflow_sim'].sel(parameter='scale').values)
    else:
        q_mean = float(scaler['streamflow_mean'].values)
        q_std = float(scaler['streamflow_std'].values)
    scaler.close()

    issue_dates = list(pd.date_range(args.start_date, args.end_date, freq='D').strftime('%Y-%m-%d'))
    logger.info(f"Generated {len(issue_dates)} issue dates: {issue_dates[:3]} ... {issue_dates[-1:]}")

    # Prepare worker tasks
    task_args_list = []
    for p_spec in PARAM_CONFIGS:
        for b_id in eval_basins:
            task_args_list.append({
                'b_id': b_id,
                'p_spec': p_spec,
                'model_cfg_dict': model_cfg_dict,
                'checkpoint_path': checkpoint_path,
                'scaler_path': scaler_path,
                'q_mean': q_mean,
                'q_std': q_std,
                'caravan_attrs': caravan_attrs,
                'issue_dates': issue_dates,
                'device_str': device_str,
                'caravan_base_dir': str(caravan_base_dir),
                'all_leadtimes': ALL_LEADTIMES,
            })

    total_tasks = len(task_args_list)
    logger.info(f"Total task units to execute: {total_tasks} across {args.num_workers} worker processes.\n")

    detailed_records = []
    all_timeseries_records = []

    # Execute tasks (Serial vs Parallel)
    if args.num_workers <= 1:
        logger.info("Executing sequentially (num_workers=1)...")
        for i, t_kwargs in enumerate(task_args_list, start=1):
            logger.info(f" Processing Task [{i}/{total_tasks}]: Config {t_kwargs['p_spec']['config_id']} | Basin {t_kwargs['b_id']}")
            recs_task, recs_ts = process_single_basin_config_task(t_kwargs)
            detailed_records.extend(recs_task)
            all_timeseries_records.extend(recs_ts)
    else:
        logger.info(f"Launching ProcessPoolExecutor with {args.num_workers} parallel workers...")
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
            future_to_task = {executor.submit(process_single_basin_config_task, t_kwargs): t_kwargs for t_kwargs in task_args_list}
            completed_count = 0
            for future in as_completed(future_to_task):
                t_info = future_to_task[future]
                completed_count += 1
                try:
                    recs_task, recs_ts = future.result()
                    detailed_records.extend(recs_task)
                    all_timeseries_records.extend(recs_ts)
                    logger.info(f" Completed Task [{completed_count}/{total_tasks}]: Config {t_info['p_spec']['config_id']} | Basin {t_info['b_id']} (Records: {len(recs_task)})")
                except Exception as exc:
                    logger.error(f" Task failed for Config {t_info['p_spec']['config_id']} | Basin {t_info['b_id']}: {exc}")

    # Save detailed per-basin per-config results
    df_detailed = pd.DataFrame(detailed_records)
    detailed_csv_name = f"detailed_basin_parameter_eval_shard_{shard_idx}.csv" if num_shards > 1 else "detailed_basin_parameter_eval.csv"
    detailed_csv = out_dir / detailed_csv_name
    df_detailed.to_csv(detailed_csv, index=False)
    logger.info(f"\nSaved detailed metrics CSV: {detailed_csv}")

    # Save timeseries predictions CSV for hydrograph plotting
    df_timeseries = pd.DataFrame(all_timeseries_records)
    ts_csv_name = f"timeseries_forecasts_and_obs_shard_{shard_idx}.csv" if num_shards > 1 else "timeseries_forecasts_and_obs.csv"
    ts_csv = out_dir / ts_csv_name
    df_timeseries.to_csv(ts_csv, index=False)
    logger.info(f"Saved timeseries predictions CSV for hydrograph plotting: {ts_csv}")

    if df_detailed.empty:
        logger.warning("No valid detailed records were produced.")
        return

    # Aggregate Summary Reports
    logger.info("\n======================================================================")
    logger.info(" AGGREGATE PARAMETER SELECTION & RANKING SUMMARY                      ")
    logger.info("======================================================================")

    summary_leadtime = df_detailed.groupby(['Config ID', 'Lead Time (Days)'])[
        ['Base NSE', 'DA NSE', 'NSE Delta', 'Base KGE', 'DA KGE', 'KGE Delta']
    ].mean().reset_index()

    summary_leadtime_csv = out_dir / (f"summary_param_leadtime_averages_shard_{shard_idx}.csv" if num_shards > 1 else "summary_param_leadtime_averages.csv")
    summary_leadtime.to_csv(summary_leadtime_csv, index=False)
    logger.info("\n--- Mean Metrics by Config ID and Lead Time ---")
    logger.info(f"\n{summary_leadtime.to_string(index=False)}")

    summary_overall = summary_leadtime.groupby('Config ID')[
        ['Base NSE', 'DA NSE', 'NSE Delta', 'Base KGE', 'DA KGE', 'KGE Delta']
    ].mean().sort_values('DA NSE', ascending=False).reset_index()

    overall_csv = out_dir / (f"summary_overall_param_rankings_shard_{shard_idx}.csv" if num_shards > 1 else "summary_overall_param_rankings.csv")
    summary_overall.to_csv(overall_csv, index=False)
    logger.info("\n--- Overall Parameter Config Rankings (Ranked by Average DA NSE) ---")
    logger.info(f"\n{summary_overall.to_string(index=False)}")

    best_per_leadtime = []
    for L in ALL_LEADTIMES:
        sub = summary_leadtime[summary_leadtime['Lead Time (Days)'] == L]
        valid_sub = sub.dropna(subset=['DA NSE'])
        if valid_sub.empty:
            continue
        best_row = valid_sub.loc[valid_sub['DA NSE'].idxmax()]
        best_per_leadtime.append({
            'Lead Time (Days)': L,
            'Best Config ID': best_row['Config ID'],
            'Mean DA NSE': round(best_row['DA NSE'], 3),
            'Mean NSE Delta': round(best_row['NSE Delta'], 3),
            'Mean DA KGE': round(best_row['DA KGE'], 3),
            'Mean KGE Delta': round(best_row['KGE Delta'], 3),
        })

    if best_per_leadtime:
        df_best_per_leadtime = pd.DataFrame(best_per_leadtime)
        best_leadtime_csv = out_dir / (f"best_param_config_per_leadtime_shard_{shard_idx}.csv" if num_shards > 1 else "best_param_config_per_leadtime.csv")
        df_best_per_leadtime.to_csv(best_leadtime_csv, index=False)
        logger.info("\n--- Best Performing Parameter Config for EACH Lead Time (1..7 Days) ---")
        logger.info(f"\n{df_best_per_leadtime.to_string(index=False)}")

    logger.info("\n======================================================================")
    logger.info("Pipeline completed successfully!")
    logger.info(f"All summary reports saved in: {out_dir}")
    logger.info("======================================================================\n")


if __name__ == '__main__':
    main()
