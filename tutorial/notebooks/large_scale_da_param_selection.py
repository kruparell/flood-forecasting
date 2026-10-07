#!/usr/bin/env python3
"""Large Scale Data Assimilation Parameter Selection & Evaluation Pipeline.

Based on the structure of large_scale_da_param_selection_parallel.py, but evaluates
tasks sequentially per process (leaving parallel execution to XManager).

Key Features:
1. Config Dictionary Grid: User-definable param_grid dictionary using generate_param_configs().
2. Detailed Metrics & Timeseries Output: Generates per-basin metric evaluations and
   timeseries forecasts/observations CSV for hydrograph plotting.
3. CNS Data Paths: Configured for CNS storage paths.
"""

import os
import sys
from pathlib import Path
import glob
import yaml
import logging
import argparse
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
    compute_all_metrics,
    open_file,
    file_exists,
)


# ==============================================================================
# 1. DIRECTORY, DEVICE & DEFAULT SETUP (CNS PATHS)
# ==============================================================================
CARAVAN_BASE_DIR = Path('/cns/jn-d/home/floods/hydro_model/work/kruparell/Caravans_V2')
MULTIMET_DIR = Path('/cns/jn-d/home/floods/hydro_model/datasets/external/Caravans_MultiMet')
PRETRAINED_DIR = Path('/cns/jn-d/home/floods/hydro_model/work/kruparell/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs')
OUTPUT_DIR = Path('/cns/jn-d/home/floods/hydro_model/work/kruparell/large_scale_param_selection_results')
BASIN_LIST_PATH = Path('/cns/jn-d/home/floods/hydro_model/work/configs/CaravansV2_basin_valid_intervals.csv')

# Evaluation Date Range and Lead Times
START_DATE = '2017-01-01'
END_DATE = '2017-01-15'
ALL_LEADTIMES = list(range(1, 8))  # Lead times 1..7

# Sampling & Reproducibility Settings
NUM_BASINS = 2      # Target number of basins to evaluate from basin list
RANDOM_SEED = 42       # Fixed seed for reproducible random subsetting

# User-Definable Parameter Grid Dictionary
param_grid = {
    'learning_rate': [1e-2], # 1e-1, 
    'epochs': [5], # , 15
    'bg_regularization_weight': [1e-8], # 1e-6, 
    'assimilation_targets': ['c_n_forecast'], # 'c_n_hindcast', 
    'assimilation_window': [7],
    'distance_from_forecast': [1],
}

def generate_param_configs(grid: dict) -> list:
    """Generates parameter configuration list from grid dictionary combinations."""
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    
    for idx, combo in enumerate(itertools.product(*values)):
        cfg = dict(zip(keys, combo))
        
        # Build readable target label for Config ID
        targets = cfg['assimilation_targets']
        target_str = targets[0] if isinstance(targets, list) else str(targets)
        target_label = "h_keys" if "h_hindcast" in target_str else "c_keys"
        
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
def find_caravan_nc(basin_id: str, caravan_base_dir: Path) -> str:
    """Locates NetCDF timeseries file within Caravans dataset directory."""
    timeseries_dir = caravan_base_dir / 'timeseries' / 'netcdf'
    matches = glob.glob(f"{timeseries_dir}/**/{basin_id}.nc", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"NetCDF file for basin {basin_id} not found in {timeseries_dir}")

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Computes NSE and KGE accuracy metrics between observations and simulation."""
    res = compute_all_metrics(obs, sim)
    return {'NSE': res.get('NSE', np.nan), 'KGE': res.get('KGE', np.nan)}


# ==============================================================================
# 3. CLI ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Large Scale DA Parameter Selection Pipeline")
    parser.add_argument('--caravan_base_dir', type=str, default=str(CARAVAN_BASE_DIR), help='Caravans dataset CNS path')
    parser.add_argument('--multimet_dir', type=str, default=str(MULTIMET_DIR), help='MultiMet dataset CNS path')
    parser.add_argument('--pretrained_dir', type=str, default=str(PRETRAINED_DIR), help='Pretrained model directory path')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR), help='Output directory path in CNS')
    parser.add_argument('--basin_list_path', type=str, default=str(BASIN_LIST_PATH), help='Basin list file path')
    parser.add_argument('--start_date', type=str, default=START_DATE, help='Evaluation start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=END_DATE, help='Evaluation end date (YYYY-MM-DD)')
    parser.add_argument('--basin_id', type=str, default=None, help='Single basin ID to evaluate (overrides NUM_BASINS sampling)')
    parser.add_argument('--num_basins', type=int, default=NUM_BASINS, help='Number of basins to sample')
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help='Random seed for sampling')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate override for single run')
    parser.add_argument('--epochs', type=int, default=None, help='Epochs override for single run')
    parser.add_argument('--assimilation_window', type=int, default=None, help='Assimilation window override for single run')
    parser.add_argument('--bg_regularization_weight', type=float, default=None, help='Bg weight override for single run')
    return parser.parse_args()


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    args = parse_args()

    caravan_dir = Path(args.caravan_base_dir)
    pretrained_dir = Path(args.pretrained_dir)
    custom_output_dir = Path(args.output_dir)
    custom_output_dir.mkdir(parents=True, exist_ok=True)
    basin_list_path = Path(args.basin_list_path)

    log_file = custom_output_dir / "large_scale_param_selection.log"

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("======================================================================")
    logger.info(" Large-Scale DA Parameter Selection Pipeline                          ")
    logger.info("======================================================================")
    logger.info(f"Hardware Compute Device: {device}")
    logger.info(f"Caravans CNS Path: {caravan_dir}")
    logger.info(f"Pretrained Model Path: {pretrained_dir}")
    logger.info(f"Evaluation Date Boundaries: {args.start_date} to {args.end_date}")
    logger.info(f"Target Lead Times: {ALL_LEADTIMES}")
    logger.info(f"Log File Location: {log_file}")

    if args.basin_id:
        eval_basins = [args.basin_id]
        logger.info(f"Evaluating Single Basin: {args.basin_id}")
    else:
        eval_basins = load_basin_list(
            basin_list_path,
            start_date=args.start_date,
            end_date=args.end_date,
            num_basins=args.num_basins,
            random_seed=args.random_seed,
        )
        logger.info(f"Loaded {len(eval_basins)} basins from {basin_list_path.name} (NUM_BASINS={args.num_basins}, Seed={args.random_seed}).")

    # Determine parameter configs
    if args.learning_rate is not None or args.epochs is not None or args.assimilation_window is not None or args.bg_regularization_weight is not None:
        param_configs = [{
            'config_id': f"Config_Custom_LR{args.learning_rate}_Ep{args.epochs}_W{args.assimilation_window}_Bg{args.bg_regularization_weight}",
            'learning_rate': args.learning_rate if args.learning_rate is not None else 1e-1,
            'epochs': args.epochs if args.epochs is not None else 10,
            'assimilation_window': args.assimilation_window if args.assimilation_window is not None else 7,
            'bg_regularization_weight': args.bg_regularization_weight if args.bg_regularization_weight is not None else 1e-6,
            'distance_from_forecast': 1,
        }]
    else:
        param_configs = PARAM_CONFIGS

    logger.info(f"Testing {len(param_configs)} Parameter Sets: {[c['config_id'] for c in param_configs]}\n")

    # 1. Load Pretrained Model and Scaling Statistics
    config_path = pretrained_dir / 'config.yml'
    checkpoint_path = pretrained_dir / 'model_epoch085.pt'
    scaler_path = pretrained_dir / 'scaler.nc'

    with open_file(config_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)

    model_cfg = Config(model_cfg_dict)
    model = MeanEmbeddingForecastLSTM(model_cfg)
    with open_file(checkpoint_path, 'rb') as f:
        raw_ckpt = torch.load(f, map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    model.to(device)
    model.eval()
    logger.info(f"Pretrained Model loaded successfully on device [{device}]: {type(model).__name__}")

    with open_file(scaler_path, 'rb') as f:
        scaler = xr.open_dataset(f).load()
    caravan_attrs = load_caravan_attributes(caravan_dir)

    # Target normalization parameters from scaler
    if 'streamflow_sim' in scaler:
        q_mean = float(scaler['streamflow_sim'].sel(parameter='center').values)
        q_std = float(scaler['streamflow_sim'].sel(parameter='scale').values)
    else:
        q_mean = float(scaler['streamflow_mean'].values)
        q_std = float(scaler['streamflow_std'].values)

    # Sample issue dates across specified date range
    issue_dates = pd.date_range(args.start_date, args.end_date, freq='D').strftime('%Y-%m-%d')
    logger.info(f"Generated {len(issue_dates)} issue dates: {list(issue_dates)}")

    detailed_records = []
    all_timeseries_records = []

    # 2. Iterate across Parameter Sets, Basins, and Issue Dates
    for p_spec in param_configs:
        config_id = p_spec['config_id']
        logger.info(f"\n======================================================================")
        logger.info(f" Running Parameter Set: {config_id}")
        logger.info(f" (LR={p_spec['learning_rate']}, Epochs={p_spec['epochs']}, Window={p_spec['assimilation_window']}d, BgWeight={p_spec['bg_regularization_weight']}, Gap={p_spec.get('distance_from_forecast', 1)}d)")
        logger.info(f"======================================================================")

        assim_engine = Assimilation(build_assim_config(p_spec))

        for b_idx, b_id in enumerate(eval_basins, start=1):
            logger.info(f" Processing Basin [{b_idx}/{len(eval_basins)}]: {b_id}")
            try:
                nc_path = find_caravan_nc(b_id, caravan_dir)
                ds_b = xr.open_dataset(nc_path)
            except Exception as e:
                logger.warning(f"  Skipping {b_id} (error loading dataset: {e})")
                continue

            records_base = []
            records_da = []
            current_month = None

            for dt in issue_dates:
                dt_month = pd.to_datetime(dt).strftime('%Y-%m')
                if dt_month != current_month:
                    logger.info(f"    [Basin {b_id}] Month: {dt_month}")
                    current_month = dt_month

                # Prepare batch in multimet_0_and_1_to_7 mode
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

                # Open-loop baseline forecast
                with torch.no_grad():
                    base_out = model(batch_b)
                    q_base_7_norm = base_out['y_hat'][0, -7:, 0].cpu().numpy()
                q_base_7 = np.maximum(0.1, q_base_7_norm * q_std + q_mean)

                # Data Assimilation forecast
                da_out = assim_engine.assimilate(model, batch_b, verbose=False)
                q_da_7_norm = da_out['y_hat'][0, -7:, 0].cpu().numpy()
                q_da_7 = np.maximum(0.1, q_da_7_norm * q_std + q_mean)

                for L_idx, L in enumerate(ALL_LEADTIMES):
                    valid_date = (dt_obj + pd.Timedelta(days=L)).strftime('%Y-%m-%d')
                    records_base.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_base_7[L_idx]})
                    records_da.append({'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_da_7[L_idx]})

            df_base = pd.DataFrame(records_base)
            df_da = pd.DataFrame(records_da)

            ds_dates = pd.Index(pd.to_datetime(ds_b['date'].values).strftime('%Y-%m-%d'))

            # Compute continuous metrics for each lead time L = 1..7 for this basin
            for L in ALL_LEADTIMES:
                df_b_base = df_base[df_base['lead_time'] == L].sort_values('valid_date').set_index('valid_date')
                df_b_da = df_da[df_da['lead_time'] == L].sort_values('valid_date').set_index('valid_date')

                valid_dates = df_b_base.index.intersection(ds_dates)

                if len(valid_dates) == 0:
                    logger.warning(f"    [WARNING] No valid date overlap for basin {b_id} at Lead Time L={L}!")
                    continue

                obs_q = ds_b['streamflow'].sel(date=valid_dates).values.astype(np.float32)
                q_base_seq = df_b_base.loc[valid_dates, 'q_sim'].values
                q_da_seq = df_b_da.loc[valid_dates, 'q_sim'].values

                m_base = compute_metrics(obs_q, q_base_seq)
                m_da = compute_metrics(obs_q, q_da_seq)

                detailed_records.append({
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
                    all_timeseries_records.append({
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

    # Save detailed per-basin per-config results
    df_detailed = pd.DataFrame(detailed_records)
    detailed_csv = custom_output_dir / "detailed_basin_parameter_eval.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    logger.info(f"\nSaved detailed metrics CSV: {detailed_csv}")

    # Save timeseries predictions CSV for hydrograph plotting
    df_timeseries = pd.DataFrame(all_timeseries_records)
    ts_csv = custom_output_dir / "timeseries_forecasts_and_obs.csv"
    df_timeseries.to_csv(ts_csv, index=False)
    logger.info(f"Saved timeseries predictions CSV for hydrograph plotting: {ts_csv}")

    # ==============================================================================
    # 5. AGGREGATE SUMMARY & SELECTION REPORTS
    # ==============================================================================
    if df_detailed.empty:
        logger.warning("No evaluation records produced.")
        return

    logger.info("\n======================================================================")
    logger.info(" AGGREGATE PARAMETER SELECTION & RANKING SUMMARY                      ")
    logger.info("======================================================================")

    # 5a. Mean metrics grouped by Config ID and Lead Time
    summary_leadtime = df_detailed.groupby(['Config ID', 'Lead Time (Days)'])[
        ['Base NSE', 'DA NSE', 'NSE Delta', 'Base KGE', 'DA KGE', 'KGE Delta']
    ].mean().reset_index()

    leadtime_csv = custom_output_dir / "summary_param_leadtime_averages.csv"
    summary_leadtime.to_csv(leadtime_csv, index=False)
    logger.info("\n--- Mean Metrics by Config ID and Lead Time ---")
    logger.info(f"\n{summary_leadtime.to_string(index=False)}")

    # 5b. Overall parameter config rankings (averaged across all lead times)
    summary_overall = summary_leadtime.groupby('Config ID')[
        ['Base NSE', 'DA NSE', 'NSE Delta', 'Base KGE', 'DA KGE', 'KGE Delta']
    ].mean().sort_values('DA NSE', ascending=False).reset_index()

    overall_csv = custom_output_dir / "summary_overall_param_rankings.csv"
    summary_overall.to_csv(overall_csv, index=False)
    logger.info("\n--- Overall Parameter Config Rankings (Ranked by Average DA NSE) ---")
    logger.info(f"\n{summary_overall.to_string(index=False)}")

    # 5c. Best parameter config selection for EACH Lead Time L = 1..7
    best_per_leadtime = []
    for L in ALL_LEADTIMES:
        sub = summary_leadtime[summary_leadtime['Lead Time (Days)'] == L]
        valid_sub = sub.dropna(subset=['DA NSE'])
        if valid_sub.empty:
            logger.warning(f"    [WARNING] No valid DA NSE scores for Lead Time L={L}, skipping best selection.")
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
        best_leadtime_csv = custom_output_dir / "best_param_config_per_leadtime.csv"
        df_best_per_leadtime.to_csv(best_leadtime_csv, index=False)
        logger.info("\n--- Best Performing Parameter Config for EACH Lead Time (1..7 Days) ---")
        logger.info(f"\n{df_best_per_leadtime.to_string(index=False)}")

    logger.info("\n======================================================================")
    logger.info("Large-scale parameter selection pipeline completed successfully!")
    logger.info(f"All summary reports saved in: {custom_output_dir}")
    logger.info("======================================================================\n")


if __name__ == '__main__':
    main()
