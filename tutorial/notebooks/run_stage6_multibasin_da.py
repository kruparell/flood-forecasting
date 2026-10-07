#!/usr/bin/env python3
"""Stage 6: Multi-Basin Continuous Forecast Matrix Evaluation Script (multimet_0_and_1_to_7 mode).

This script executes Stage 6 of the Data Assimilation tutorial pipeline:
1. Loads pretrained Google Hydrology foundation model (MeanEmbeddingForecastLSTM).
2. Generates daily forecast batches in 'multimet_0_and_1_to_7' mode across an issue date interval.
   - Hindcast period (t <= t_0) receives Leadtime 0 nowcast forcings (with NaN masking in final 7 days).
   - Forecast period (t_0+1 .. t_0+7) receives MultiMet forecast product at Leadtimes 1..7.
3. Constructs a Forecast Matrix (issue_date x lead_time) for both baseline and Data Assimilation predictions.
4. Extracts continuous hydrographs and performance metrics (NSE & KGE) for ALL Lead Times L = 1..7.
5. Computes accuracy metrics across 3 basins and saves:
   - Comprehensive multi-leadtime metrics CSV (stage6_all_leadtimes_multimet_metrics.csv).
   - Hydrograph plots for each lead time L = 1..7 (hydrograph_continuous_leadtime_L.png).
"""

import os
import sys
from pathlib import Path
import glob
import yaml

# Preload foundational scientific libraries before modifying sys.path
import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add flood-forecasting repository path to sys.path
sys.path.append('/usr/local/google/home/kruparell/flood-forecasting')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from googlehydrology.utils.config import Config
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.evaluation.assimilation import Assimilation
from multimet_helpers import prepare_multimet_batch


# ==============================================================================
# 1. DIRECTORY & CONFIGURATION SETUP
# ==============================================================================
MULTIMET_DIR = Path('/usr/local/google/home/kruparell/Caravans_MultiMet')
PRETRAINED_DIR = '/usr/local/google/home/kruparell/flood-forecasting/pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs'
OUTPUT_DIR = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/notebooks/stage6_results')

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define Catchment Basins and Issue Date Interval
EVAL_BASINS = ['camels_12451000', 'camels_12377150', 'camels_12115000']
START_DATE = '2020-01-01'
END_DATE = '2020-01-31'
ALL_LEADTIMES = list(range(1, 8))  # Evaluate ALL leadtimes 1..7


# ==============================================================================
# 2. SINGLE ASSIMILATION PERIOD CONFIGURATION
# ==============================================================================
# - WINDOW_LENGTH: Duration of the single DA optimization window (in days)
# - DISTANCE_FROM_FORECAST: Gap/distance between DA window end and forecast start (in days)
# ==============================================================================
WINDOW_LENGTH = 60             # Single DA optimization window length (days)
DISTANCE_FROM_FORECAST = 14    # Distance from start of forecast period (days)
FORECAST_LEAD_DAYS = 7         # Length of forecast rollout period (days)
SEQUENCE_LENGTH = 365          # Total batch sequence length (days)

# Derived lead time offset from end of sequence
ASSIMILATION_LEAD_TIME = FORECAST_LEAD_DAYS + DISTANCE_FROM_FORECAST

cfg_dict_assim = {
    # Timeline & Window Layout
    'seq_length': SEQUENCE_LENGTH,
    'history': 1,                            # Single block update
    'assimilation_window': WINDOW_LENGTH,    # Length of single DA window
    'assimilation_lead_time': ASSIMILATION_LEAD_TIME,
    'predict_n_hindcast': DISTANCE_FROM_FORECAST,
    'predict_last_n': FORECAST_LEAD_DAYS,
    
    # Data Assimilation Optimization Settings
    'learning_rate': 1e-1,
    'epochs': 10,
    'bg_regularization_weight': 1e-6,
    'optimizer': 'Adam',
    'loss': 'MSE',
    'assimilation_targets': ['c_n'],
    'target_variables': ['streamflow'],
}


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def find_caravan_nc(basin_id: str) -> str:
    """Locates NetCDF timeseries file for a given Caravan basin ID."""
    search_dirs = [
        Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/Caravan-nc/timeseries/netcdf'),
        Path('/usr/local/google/home/kruparell/Caravans/Caravan-nc/timeseries/netcdf')
    ]
    for sdir in search_dirs:
        matches = glob.glob(f"{sdir}/**/{basin_id}.nc", recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find Caravan NetCDF for basin: {basin_id}")

def load_caravan_attributes() -> pd.DataFrame:
    """Loads and merges static attribute tables across Caravan CSV files."""
    attr_paths = (
        glob.glob('/usr/local/google/home/kruparell/flood-forecasting/tutorial/Caravan-nc/attributes/**/*.csv', recursive=True) +
        glob.glob('/usr/local/google/home/kruparell/Caravans/Caravan-nc/attributes/**/*.csv', recursive=True)
    )
    dfs = []
    seen = set()
    for p in attr_paths:
        fname = os.path.basename(p)
        if fname not in seen and os.path.exists(p):
            seen.add(fname)
            try:
                df = pd.read_csv(p)
                if 'gauge_id' in df.columns:
                    dfs.append(df.set_index('gauge_id'))
            except Exception:
                pass
    combined = pd.concat(dfs, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()]

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Computes NSE and KGE accuracy metrics between observations and simulation."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[valid], sim[valid]
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - (np.sum((o - s) ** 2) / denom)) if denom != 0 else np.nan
    std_o, std_s = np.std(o), np.std(s)
    r = float(np.corrcoef(o, s)[0, 1]) if len(o) > 1 else np.nan
    kge = float(1 - np.sqrt((r - 1)**2 + (std_s/std_o - 1)**2 + (np.mean(s)/np.mean(o) - 1)**2)) if std_o > 0 and np.mean(o) > 0 else np.nan
    return {'NSE': nse, 'KGE': kge}


# ==============================================================================
# 4. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("======================================================================")
    print(" Stage 6: Forecast Matrix Evaluation for ALL Lead Times L = 1..7     ")
    print("======================================================================")
    print(f"Basins to evaluate: {EVAL_BASINS}")
    print(f"Issue Date Interval: {START_DATE} to {END_DATE}")
    print(f"Evaluating Lead Times: {ALL_LEADTIMES}")
    print(f"DA Single Window Length: {WINDOW_LENGTH} Days")
    print(f"Distance to Forecast Start: {DISTANCE_FROM_FORECAST} Days")
    print(f"Results Output Directory: {OUTPUT_DIR}\n")

    # 1. Load Pretrained Foundation Model
    config_path = os.path.join(PRETRAINED_DIR, 'config.yml')
    checkpoint_path = os.path.join(PRETRAINED_DIR, 'model_epoch085.pt')
    scaler_path = os.path.join(PRETRAINED_DIR, 'scaler.nc')

    with open(config_path, 'r') as f:
        model_cfg_dict = yaml.safe_load(f)

    model_cfg = Config(model_cfg_dict)
    model = MeanEmbeddingForecastLSTM(model_cfg)
    raw_ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    model.eval()

    scaler = xr.open_dataset(scaler_path)
    caravan_attrs = load_caravan_attributes()
    
    # Target normalization parameters from scaler
    if 'streamflow_sim' in scaler:
        q_mean = float(scaler['streamflow_sim'].sel(parameter='center').values)
        q_std = float(scaler['streamflow_sim'].sel(parameter='scale').values)
    else:
        q_mean = float(scaler['streamflow_mean'].values)
        q_std = float(scaler['streamflow_std'].values)

    # 2. Instantiate Data Assimilation Engine
    assim_cfg = AssimilationConfig(cfg_dict_assim)
    assim = Assimilation(assim_cfg)

    # Daily issue dates across evaluation interval
    issue_dates = pd.date_range(START_DATE, END_DATE, freq='D').strftime('%Y-%m-%d')

    records_base = []
    records_da = []
    records_summary = []

    # 3. Construct Forecast Matrix across Daily Issue Dates
    for b_id in EVAL_BASINS:
        print(f"\n---> Generating Forecast Matrix for Basin: {b_id} <---")
        nc_path = find_caravan_nc(b_id)
        ds_b = xr.open_dataset(nc_path)

        current_month = None
        for dt in issue_dates:
            dt_month = pd.to_datetime(dt).strftime('%Y-%m')
            if dt_month != current_month:
                print(f"  [Basin {b_id}] Switched to month: {dt_month}")
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
                forecast_lead_days=FORECAST_LEAD_DAYS
            )

            dt_obj = pd.to_datetime(dt)

            # Open-loop baseline forecast
            with torch.no_grad():
                base_out = model(batch_b)
                q_base_7_norm = base_out['y_hat'][0, -7:, 0].numpy()
            q_base_7 = np.maximum(0.1, q_base_7_norm * q_std + q_mean)

            # Data Assimilation forecast
            da_out = assim.assimilate(model, batch_b, verbose=False)
            q_da_7_norm = da_out['y_hat'][0, -7:, 0].numpy()
            q_da_7 = np.maximum(0.1, q_da_7_norm * q_std + q_mean)

            # Record predictions for ALL leadtimes L = 1..7
            for L_idx, L in enumerate(ALL_LEADTIMES):
                valid_date = (dt_obj + pd.Timedelta(days=L)).strftime('%Y-%m-%d')
                records_base.append({'basin_id': b_id, 'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_base_7[L_idx]})
                records_da.append({'basin_id': b_id, 'issue_date': dt, 'valid_date': valid_date, 'lead_time': L, 'q_sim': q_da_7[L_idx]})

    df_fc_base = pd.DataFrame(records_base)
    df_fc_da = pd.DataFrame(records_da)

    # 4. Extract Continuous Hydrographs and Compute Metrics for ALL Lead Times L = 1..7
    print("\n---> Computing Continuous Hydrograph Metrics for ALL Lead Times (1..7) <---")

    for L in ALL_LEADTIMES:
        fig, axes = plt.subplots(len(EVAL_BASINS), 1, figsize=(14, 3.8 * len(EVAL_BASINS)), sharex=True, dpi=120)
        if len(EVAL_BASINS) == 1:
            axes = [axes]

        for idx, b_id in enumerate(EVAL_BASINS):
            nc_path = find_caravan_nc(b_id)
            ds_b = xr.open_dataset(nc_path)

            df_b_base = df_fc_base[(df_fc_base['basin_id'] == b_id) & (df_fc_base['lead_time'] == L)].sort_values('valid_date').set_index('valid_date')
            df_b_da = df_fc_da[(df_fc_da['basin_id'] == b_id) & (df_fc_da['lead_time'] == L)].sort_values('valid_date').set_index('valid_date')

            valid_dates = df_b_base.index.intersection(pd.to_datetime(ds_b['date'].values).strftime('%Y-%m-%d'))
            obs_q = ds_b['streamflow'].sel(date=valid_dates).values.astype(np.float32)

            q_base_series = df_b_base.loc[valid_dates, 'q_sim'].values
            q_da_series = df_b_da.loc[valid_dates, 'q_sim'].values

            m_base = compute_metrics(obs_q, q_base_series)
            m_da = compute_metrics(obs_q, q_da_series)

            records_summary.append({
                'Basin ID': b_id,
                'Lead Time (Days)': L,
                'Mode': 'multimet_0_and_1_to_7',
                'Base NSE': round(m_base['NSE'], 3),
                'Base KGE': round(m_base['KGE'], 3),
                'DA NSE': round(m_da['NSE'], 3),
                'DA KGE': round(m_da['KGE'], 3),
                'NSE Delta': round(m_da['NSE'] - m_base['NSE'], 3),
                'KGE Delta': round(m_da['KGE'] - m_base['KGE'], 3)
            })

            plot_dates = pd.to_datetime(valid_dates)
            ax = axes[idx]
            ax.plot(plot_dates, obs_q, label="Caravans Observed ($Q_{obs}$)", color="#111827", linewidth=2.0)
            ax.plot(plot_dates, q_base_series, label=f"Baseline Model (NSE={m_base['NSE']:.2f}, KGE={m_base['KGE']:.2f})", color="#2563eb", linestyle="--", linewidth=1.6)
            ax.plot(plot_dates, q_da_series, label=f"Data Assimilation Assimilated (NSE={m_da['NSE']:.2f}, KGE={m_da['KGE']:.2f})", color="#059669", linewidth=1.8)

            ax.set_title(f"Catchment Basin: {b_id} (Continuous Lead Time L = {L} Hydrograph)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Discharge (mm/day)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        axes[-1].set_xlabel("Valid Forecast Date", fontsize=10)
        fig.suptitle(f"Stage 6 Continuous Lead-Time L = {L} Forecast Hydrographs (multimet_0_and_1_to_7)", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()

        # Save hydrograph plot for Lead Time L
        plot_path = OUTPUT_DIR / f"hydrograph_continuous_leadtime_{L}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved continuous hydrograph plot for Lead Time L = {L}: {plot_path}")

    # 5. Save Summary Metrics DataFrame for ALL Lead Times to CSV
    df_metrics = pd.DataFrame(records_summary)
    csv_path = OUTPUT_DIR / "stage6_all_leadtimes_multimet_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    print("\n======================================================================")
    print(f"Successfully saved all leadtimes metrics CSV to: {csv_path}")
    print("======================================================================")
    print(df_metrics.to_string(index=False))

if __name__ == '__main__':
    main()
