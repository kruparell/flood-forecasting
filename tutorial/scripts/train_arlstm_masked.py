#!/usr/bin/env python3
"""Train Autoregressive LSTM (ARLSTM) with Streamflow Masking.

This script loads basins from the CaravansV2 basin valid intervals list,
filters for complete coverage across user-defined date ranges, reproducibly
samples N basins (default: 200), configures an ARLSTM training setup using
MultiMet dynamic forcing features and Caravan NetCDF targets, applies a 50%
streamflow observation masking during training (random holdout), and executes
the training process using googlehydrology.

Usage Examples:
---------------
1. Default 200 basins (2017 date range):
   python train_arlstm_masked.py --num_basins 200 --start_date 2017-01-01 --end_date 2017-12-31 --epochs 10 --missing_fraction 0.5

2. Custom basin count and training date range:
   python train_arlstm_masked.py --num_basins 50 --start_date 2017-01-01 --end_date 2018-12-31 --epochs 10
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd

# Force stdout to unbuffered / line-buffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure flood-forecasting repository is present on sys.path
FLOOD_FORECASTING_ROOT = Path('/usr/local/google/home/kruparell/flood-forecasting')
if str(FLOOD_FORECASTING_ROOT) not in sys.path:
    sys.path.append(str(FLOOD_FORECASTING_ROOT))

from googlehydrology.utils.config import Config
from googlehydrology.training.train import start_training

# Default dataset and directory paths
DEFAULT_BASIN_LIST_PATH = FLOOD_FORECASTING_ROOT / 'tutorial' / 'basin-lists' / 'CaravansV2_basin_valid_intervals.csv'
DEFAULT_RUN_DIR = FLOOD_FORECASTING_ROOT / 'tutorial' / 'model-runs'
CARAVANS_DYNAMICS_DIR = Path('/usr/local/google/home/kruparell/Caravans_MultiMet')
CARAVANS_STATIC_TARGETS_DIR = Path('/usr/local/google/home/kruparell/Caravans_V2')


class EpochLossPrinter(logging.Handler):
    """Custom logging handler to highlight and print epoch average loss in real time."""
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if "average loss" in msg:
                print("\n" + "=" * 65, flush=True)
                print(f"  >>> {msg.strip()} <<<", flush=True)
                print("=" * 65 + "\n", flush=True)
        except Exception:
            self.handleError(record)


def load_and_sample_basins(
    file_path: Path,
    start_date: str,
    end_date: str,
    num_basins: int = 1,
    random_seed: int = 42,
    statics_dir: Path = CARAVANS_STATIC_TARGETS_DIR,
) -> list[str]:
    """Loads basin list from CSV or TXT, filters for valid coverage between start_date and end_date,
    filters for available static dataset directories, and reproducibly samples num_basins.
    """
    logger = logging.getLogger(__name__)

    print(f"[Step 1/4] Reading basin list from: {file_path}", flush=True)
    if not file_path.exists():
        raise FileNotFoundError(f"Basin list file not found: {file_path}")

    eval_start = pd.to_datetime(start_date)
    eval_end = pd.to_datetime(end_date)

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

        # Filter for full coverage across [start_date, end_date] if valid date columns exist
        if {"valid_start_date", "valid_end_date"}.issubset(df.columns):
            df["valid_start_date"] = pd.to_datetime(df["valid_start_date"])
            df["valid_end_date"] = pd.to_datetime(df["valid_end_date"])

            valid_mask = (df["valid_start_date"] <= eval_start) & (df["valid_end_date"] >= eval_end)
            df_filtered = df[valid_mask]

            print(f"  -> Found {len(df)} total basins in CSV. Filtered for date coverage [{start_date} to {end_date}]: {len(df_filtered)} basins match.", flush=True)

            if not df_filtered.empty:
                df = df_filtered
            else:
                print(f"  [WARNING] No basins matched strict date coverage between {start_date} and {end_date} in CSV. Falling back to all basins.", flush=True)

        if "basin_id" in df.columns:
            basins = df["basin_id"].dropna().astype(str).str.strip().unique().tolist()
        else:
            basins = df.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist()
    else:
        with open(file_path, "r") as f:
            basins = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not basins:
        raise ValueError(f"No valid basins found in {file_path}.")

    # Restrict basins to subdatasets present in statics_dir
    attributes_dir = statics_dir / 'attributes'
    if attributes_dir.exists():
        valid_subdatasets = set(os.listdir(attributes_dir))
        matched_basins = [b for b in basins if b.split('_')[0] in valid_subdatasets]
        print(f"  -> Filtered for subdatasets present in {attributes_dir}: {len(matched_basins)} matched.", flush=True)
        if matched_basins:
            basins = matched_basins
        else:
            print(f"  [WARNING] No basins matched subdatasets in {attributes_dir}. Using full basin list.", flush=True)


    basins = [b for b in basins if b.startswith("camelsch")]
    # Reproducible random sampling
    if 0 < num_basins < len(basins):
        rng = np.random.RandomState(random_seed)
        basins = sorted(list(rng.choice(basins, size=num_basins, replace=False)))
        print(f"  -> Randomly sampled {len(basins)} basins (seed={random_seed}).", flush=True)
    else:
        basins = sorted(basins)
        print(f"  -> Using all {len(basins)} available basins.", flush=True)

    return basins

def format_date_to_config(date_str: str) -> str:
    """Converts YYYY-MM-DD or other date string formats into DD/MM/YYYY for Config."""
    dt = pd.to_datetime(date_str)
    return dt.strftime('%d/%m/%Y')

def create_training_config(
    basin_file_path: Path,
    num_basins: int,
    start_date: str,
    end_date: str,
    epochs: int = 10,
    missing_fraction: float = 0.99,
    run_dir: Path = DEFAULT_RUN_DIR,
    device: str = 'cpu',
    seed: int = 42,
    hidden_size: int = 64,
    learning_rate: float = 0.003,
) -> Config:
    """Builds and returns a googlehydrology Config object for training ARLSTM."""
    experiment_name = f"arlstm-{num_basins}basin-masked_{int(missing_fraction*100)}pct"


    required_min_len = int(np.ceil(missing_fraction / (1.0 - missing_fraction))) if missing_fraction < 1.0 else 10
    mean_missing_length = max(10, required_min_len)

    config_dict = {
        'experiment_name': experiment_name,
        'run_dir': str(run_dir),
        'logging_level': 'INFO',
        'dev_mode': True,
        'dataset': 'multimet',
        'train_basin_file': str(basin_file_path),
        'validation_basin_file': str(basin_file_path),
        'test_basin_file': str(basin_file_path),
        'targets_data_dir': str(CARAVANS_STATIC_TARGETS_DIR),
        'statics_data_dir': str(CARAVANS_STATIC_TARGETS_DIR),
        'dynamics_data_dir': str(CARAVANS_DYNAMICS_DIR),
        'load_as_csv': False,
        # Dates formatted for Config
        'train_start_date': format_date_to_config(start_date),
        'train_end_date': format_date_to_config(end_date),
        'validation_start_date': format_date_to_config(start_date),
        'validation_end_date': format_date_to_config(end_date),
        'test_start_date': format_date_to_config(start_date),
        'test_end_date': format_date_to_config(end_date),
        # Dynamic inputs
        'hindcast_inputs': {
            'cpc': ['cpc_precipitation'],
            'imerg': ['imerg_precipitation']
        },
        'forecast_inputs': {
            'hres': ['hres_temperature_2m', 'hres_total_precipitation']
        },
        'autoregressive_inputs': ['streamflow_shift1'],
        'dynamic_inputs': ['hres_temperature_2m', 'hres_total_precipitation', 'imerg_precipitation', 'cpc_precipitation'],
        'union_mapping': {
            'cpc_precipitation': 'era5land_total_precipitation',
            'imerg_precipitation': 'era5land_total_precipitation',
            'hres_temperature_2m': 'era5land_temperature_2m',
            'hres_total_precipitation': 'era5land_total_precipitation'
        },
        # Static attributes & Target (All 85 attributes matching pretrained model)
        'static_attributes': [
            'p_mean', 'pet_mean_ERA5_LAND', 'aridity_ERA5_LAND', 'frac_snow', 'moisture_index_ERA5_LAND',
            'seasonality_ERA5_LAND', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
            'aet_mm_syr', 'ari_ix_sav', 'crp_pc_sse', 'ele_mt_sav', 'ero_kh_sav', 'for_pc_sse',
            'gdp_ud_ssu', 'gla_pc_sse', 'glc_pc_s01', 'glc_pc_s02', 'glc_pc_s03', 'glc_pc_s04',
            'glc_pc_s06', 'glc_pc_s07', 'glc_pc_s08', 'glc_pc_s09', 'glc_pc_s10', 'glc_pc_s11',
            'glc_pc_s12', 'glc_pc_s13', 'glc_pc_s14', 'glc_pc_s15', 'glc_pc_s16', 'glc_pc_s17',
            'glc_pc_s18', 'glc_pc_s19', 'glc_pc_s20', 'glc_pc_s21', 'glc_pc_s22', 'hft_ix_s09',
            'hft_ix_s93', 'inu_pc_slt', 'inu_pc_smn', 'inu_pc_smx', 'ire_pc_sse', 'kar_pc_sse',
            'lka_pc_sse', 'nli_ix_sav', 'pac_pc_sse', 'pet_mm_syr', 'pnv_pc_s01', 'pnv_pc_s02',
            'pnv_pc_s03', 'pnv_pc_s04', 'pnv_pc_s05', 'pnv_pc_s06', 'pnv_pc_s07', 'pnv_pc_s08',
            'pnv_pc_s09', 'pnv_pc_s10', 'pnv_pc_s11', 'pnv_pc_s12', 'pnv_pc_s13', 'pnv_pc_s14',
            'pnv_pc_s15', 'ppd_pk_sav', 'pre_mm_syr', 'prm_pc_sse', 'rdd_mk_sav', 'snw_pc_syr',
            'swc_pc_syr', 'tmp_dc_syr', 'urb_pc_sse', 'wet_pc_s01', 'wet_pc_s02', 'wet_pc_s03',
            'wet_pc_s04', 'wet_pc_s05', 'wet_pc_s06', 'wet_pc_s07', 'wet_pc_s08', 'wet_pc_s09',
            'wet_pc_sg1', 'wet_pc_sg2'
        ],
        'target_variables': ['streamflow'],
        # Model architecture (matching template)
        'model': 'arlstm',
        'hidden_size': hidden_size,
        'head': 'cmal',
        'n_distributions': 3,
        'n_hidden': 100,
        'n_samples': 100,  # Number of samples for CMAL probabilistic evaluation
        'output_activation': 'linear',
        'seq_length': 180,
        'timestep_counter': True,
        'output_dropout': 0.4,
        # Streamflow observation masking (random holdout)
        'random_holdout_from_dynamic_features': {
            'streamflow_shift1': {
                'missing_fraction': missing_fraction,
                'mean_missing_length': mean_missing_length
            }
        },
        # Model initialization & Embeddings (matching template)
        'initial_forget_bias': 3,
        'weight_init_opts': ['lstm-ih-xavier', 'lstm-hh-orthogonal', 'fc-xavier'],
        'statics_embedding': {'type': 'fc', 'hiddens': [100, 100, 20], 'activation': 'tanh', 'dropout': 0.0},
        'dynamics_embedding': {'type': 'fc', 'hiddens': [20, 20, 20, 20], 'activation': 'tanh', 'dropout': 0.0},
        'hindcast_embedding': {'type': 'fc', 'hiddens': [100, 20], 'activation': 'tanh', 'dropout': 0.0},
        'forecast_embedding': {'type': 'fc', 'hiddens': [20, 20, 20, 20], 'activation': 'tanh', 'dropout': 0.0},
        # Training hyperparameters (CMAL negative log likelihood loss)
        'device': device,
        'seed': seed,
        'loss': 'CMALLoss',
        'optimizer': 'Adam',
        'epochs': epochs,
        'batch_size': 16,
        'save_weights_every': 1,
        'learning_rate': learning_rate,
        'learning_rate_strategy': 'StepLR',
        'initial_learning_rate': learning_rate,
        'learning_rate_drop_factor': 0.8,
        'learning_rate_epochs_drop': 2,
        'clip_gradient_norm': 1,
        'metrics': ['NSE'],
        'predict_last_n': 7,
        'lead_time': 7,
        'num_workers': 0,
        'log_loss_every_nth_update': 1,
        'verbose': True,
    }

    return Config(config_dict, dev_mode=True)

def main():
    parser = argparse.ArgumentParser(description="Train ARLSTM with random streamflow observation masking.")
    parser.add_argument('--num_basins', type=int, default=200, help="Number of basins to sample for training (default: 200)")
    parser.add_argument('--start_date', type=str, default='2017-01-01', help="Training start date (YYYY-MM-DD or DD/MM/YYYY)")
    parser.add_argument('--end_date', type=str, default='2017-12-31', help="Training end date (YYYY-MM-DD or DD/MM/YYYY)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('--missing_fraction', type=float, default=0.5, help="Fraction of streamflow observations to mask during training (default: 0.5)")
    parser.add_argument('--hidden_size', type=int, default=64, help="LSTM hidden layer size (default: 64)")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument('--random_seed', type=int, default=42, help="Random seed for reproducible basin sampling and model initialization (default: 42)")
    parser.add_argument('--basin_list_path', type=str, default=str(DEFAULT_BASIN_LIST_PATH), help="Path to Caravans basin list CSV or TXT file")
    parser.add_argument('--run_dir', type=str, default=str(DEFAULT_RUN_DIR), help="Output directory for trained model runs")
    parser.add_argument('--device', type=str, default='cpu', help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    # Configure explicit unbuffered logging to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    epoch_loss_handler = EpochLossPrinter()
    epoch_loss_handler.setLevel(logging.INFO)
    epoch_loss_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [handler, epoch_loss_handler]

    logger = logging.getLogger(__name__)

    print("======================================================================", flush=True)
    print(" Autoregressive LSTM (ARLSTM) Training with Progress Monitoring       ", flush=True)
    print("======================================================================", flush=True)
    print(f"Target Number of Basins: {args.num_basins}", flush=True)
    print(f"Training Date Range: {args.start_date} to {args.end_date}", flush=True)
    print(f"Number of Epochs: {args.epochs}", flush=True)
    print(f"Streamflow Missing Fraction: {args.missing_fraction * 100:.1f}%", flush=True)
    print(f"Random Seed: {args.random_seed}", flush=True)
    print(f"Basin List Path: {args.basin_list_path}", flush=True)
    print(f"Output Directory: {args.run_dir}", flush=True)
    print(f"Device: {args.device}\n", flush=True)

    # 1. Load and sample valid basins
    sampled_basins = load_and_sample_basins(
        file_path=Path(args.basin_list_path),
        start_date=args.start_date,
        end_date=args.end_date,
        num_basins=args.num_basins,
        random_seed=args.random_seed,
    )
    print(f"[Step 2/4] Successfully sampled {len(sampled_basins)} valid basins.", flush=True)
    print(f"  -> Sampled Basins Preview: {sampled_basins[:5]} ... {sampled_basins[-2:]}", flush=True)

    # 2. Save sampled basin list to file for training tracking
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    basin_file_path = run_dir / f"sampled_{len(sampled_basins)}basins_seed{args.random_seed}.txt"
    basin_file_path.write_text('\n'.join(sampled_basins) + '\n')
    print(f"[Step 3/4] Saved sampled basin list to: {basin_file_path}\n", flush=True)

    # 3. Create Config object
    cfg = create_training_config(
        basin_file_path=basin_file_path,
        num_basins=len(sampled_basins),
        start_date=args.start_date,
        end_date=args.end_date,
        epochs=args.epochs,
        missing_fraction=args.missing_fraction,
        run_dir=run_dir,
        device=args.device,
        seed=args.random_seed,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
    )

    # 4. Start model training with exception tracking
    print(f"[Step 4/4] Initializing datasets, model, and starting training loop for {args.epochs} epochs...", flush=True)
    print("----------------------------------------------------------------------", flush=True)

    try:
        start_training(cfg)
        print("\n======================================================================", flush=True)
        print(" Training completed successfully! ", flush=True)
        print("======================================================================", flush=True)
    except Exception as e:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        print(" TRAINING FAILED WITH AN EXCEPTION: ", flush=True)
        print(f" Exception Type: {type(e).__name__}", flush=True)
        print(f" Exception Error: {e}", flush=True)
        print(" Full Traceback:", flush=True)
        traceback.print_exc(file=sys.stdout)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
