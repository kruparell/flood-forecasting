import os
import sys
import glob
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from googlehydrology.utils.config import Config
from googlehydrology.training.train import start_training
from googlehydrology.evaluation.evaluate import start_evaluation
from googlehydrology.evaluation.metrics import calculate_metrics

def main():
    # 1. Train the model
    config_path = Path(__file__).resolve().parent / 'configs' / 'train-arlstm-local.yml'
    print(f"Loading config from {config_path}")
    config = Config(config_path)
    
    print("Starting training...")
    start_training(config)
    print("Training complete.")

    # 2. Find the run directory
    run_dirs = glob.glob(str(Path(__file__).resolve().parent / 'model-runs' / 'arlstm-local-example_*'))
    if not run_dirs:
        raise RuntimeError("No run directories found after training!")
    
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    run_dir_path = Path(latest_run_dir).resolve()
    print(f"Using run directory: {run_dir_path}")

    # 3. Baseline Evaluation
    config_baseline = Config(run_dir_path / 'config.yml')
    print("Running baseline evaluation...")
    start_evaluation(
        cfg=config_baseline,
        run_dir=run_dir_path,
        period='test',
        data_assimilation=False
    )
    print("Baseline evaluation complete.")

    # 4. DA Evaluation
    config_da = Config(run_dir_path / 'config.yml')
    da_config_dict = {
        'assimilation_config': {
            'assimilation_lead_time': 0,
            'assimilation_window': 1,
            'history': 10,
            'learning_rate': 0.05,
            'assimilation_targets': ['h_n', 'c_n'],
            'epochs': 10,
            'loss': 'MSE',
            'optimizer': 'Adam'
        }
    }
    config_da.update_config(da_config_dict)
    print("Running DA evaluation...")
    start_evaluation(
        cfg=config_da,
        run_dir=run_dir_path,
        period='test',
        data_assimilation=True
    )
    print("DA evaluation complete.")

    # 5. Compare Results
    ds_no_da = xr.open_zarr(run_dir_path / 'test_results.zarr')
    ds_da = xr.open_zarr(run_dir_path / 'test_results_data_assimilation.zarr')

    basins = list(ds_no_da.basin.values)
    print(f"\nAvailable basins: {basins}")

    nse_no_da_list = []
    nse_da_list = []

    for b in basins:
        o = ds_no_da['streamflow_obs'].sel(basin=b).squeeze()
        s_no_da = ds_no_da['streamflow_sim'].sel(basin=b).squeeze()
        s_da = ds_da['streamflow_sim'].sel(basin=b).squeeze()
        
        if o.isnull().all():
            continue
            
        m_no_da = calculate_metrics(o, s_no_da, metrics=['NSE'])
        m_da = calculate_metrics(o, s_da, metrics=['NSE'])
        
        print(f"Basin {b} - NSE | No DA: {m_no_da['NSE']:.4f} | With DA: {m_da['NSE']:.4f}")
        
        if not np.isnan(m_no_da['NSE']):
            nse_no_da_list.append(m_no_da['NSE'])
        if not np.isnan(m_da['NSE']):
            nse_da_list.append(m_da['NSE'])

    print("\nAverage NSE over all valid basins:")
    print(f"  Without DA: {np.mean(nse_no_da_list):.4f}")
    print(f"  With DA:    {np.mean(nse_da_list):.4f}")

if __name__ == '__main__':
    main()
