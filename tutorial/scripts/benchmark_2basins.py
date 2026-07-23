"""2-basin rapid evaluation and comparison for MeanEmbedding, Data Assimilation, and AR-LSTM."""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

repo_root = next(
    (p for p in [Path(__file__).resolve().parents[2], Path.cwd()] if (p / 'googlehydrology').is_dir()),
    Path('/usr/local/google/home/kruparell/flood-forecasting')
)
for p in [str(repo_root), str(repo_root / 'tutorial' / 'scripts'), str(repo_root / 'tutorial')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from googlehydrology.utils.config import Config
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.datasetzoo.multimet import _convert_to_tensor
import backend


def evaluate_basin(tester_me, tester_ar, assimilation, sample, basin_id):
    """Evaluates a single basin across Baseline, Data Assimilation, and AR-LSTM models."""
    batch = tester_me.dataset.collate_fn([{k: _convert_to_tensor(k, v) for k, v in sample.items()}])
    obs_raw = batch['y'][0, :, 0].cpu().numpy()

    # 1. Baseline: MeanEmbeddingForecastLSTM
    tester_me.model.eval()
    with torch.no_grad():
        base_out = tester_me.model(batch)
        y_hat_base = base_out['y_hat'][0]
        s_base_raw = y_hat_base[:, 0].cpu().numpy() if y_hat_base.ndim == 2 else y_hat_base.cpu().numpy()

    # 2. Data Assimilation
    da_out = assimilation.assimilate(tester_me.model, batch, verbose=False, check_timing=False)
    y_hat_da = da_out['y_hat'][0]
    s_da_raw = y_hat_da[:, 0].cpu().numpy() if y_hat_da.ndim == 2 else y_hat_da.cpu().numpy()

    # 3. Autoregressive LSTM
    batch_ar = tester_ar.dataset.collate_fn([{k: _convert_to_tensor(k, v) for k, v in sample.items()}])
    if 'y' in batch_ar and 'x_d' in batch_ar:
        y_ar = batch_ar['y']
        y_shift1 = torch.roll(y_ar, shifts=1, dims=1)
        y_shift1[:, 0, :] = float('nan')
        batch_ar['x_d']['streamflow_shift1'] = y_shift1

    batch_ar = tester_ar.model.pre_model_hook(batch_ar, is_train=False)
    tester_ar.model.eval()
    with torch.no_grad():
        ar_out = tester_ar.model(batch_ar)
        y_hat_ar = ar_out['y_hat'][0]
        s_ar_raw = y_hat_ar[:, 0].cpu().numpy() if y_hat_ar.ndim == 2 else y_hat_ar.cpu().numpy()

    # Alignment and metrics calculation
    obs_hind = obs_raw[-5:]
    s_base_hind, s_da_hind = s_base_raw[:5], s_da_raw[:5]
    s_base_fc = s_base_raw[5:] if len(s_base_raw) >= 12 else s_base_raw[-7:]
    s_da_fc = s_da_raw[5:] if len(s_da_raw) >= 12 else s_da_raw[-7:]
    s_ar_fc = s_ar_raw[5:] if len(s_ar_raw) >= 12 else s_ar_raw[-7:]

    k_len = min(len(obs_raw), len(s_base_fc), len(s_da_fc), len(s_ar_fc))
    obs_fc = obs_raw[-k_len:]

    m_base_hind = backend.compute_hydro_metrics(obs_hind, s_base_hind)
    m_da_hind = backend.compute_hydro_metrics(obs_hind, s_da_hind)
    m_base = backend.compute_hydro_metrics(obs_fc, s_base_fc[-k_len:])
    m_da = backend.compute_hydro_metrics(obs_fc, s_da_fc[-k_len:])
    m_ar = backend.compute_hydro_metrics(obs_fc, s_ar_fc[-k_len:])

    return {
        'basin': basin_id,
        'NSE_Hindcast_Baseline': m_base_hind['NSE'],
        'NSE_Hindcast_DataAssimilation': m_da_hind['NSE'],
        'PearsonR_Hindcast_Baseline': m_base_hind['Pearson-r'],
        'PearsonR_Hindcast_DataAssimilation': m_da_hind['Pearson-r'],
        'RMSE_Hindcast_Baseline': m_base_hind['RMSE'],
        'RMSE_Hindcast_DataAssimilation': m_da_hind['RMSE'],
        'NSE_Baseline': m_base['NSE'],
        'NSE_DataAssimilation': m_da['NSE'],
        'NSE_ARLSTM': m_ar['NSE'],
        'PearsonR_Baseline': m_base['Pearson-r'],
        'PearsonR_DataAssimilation': m_da['Pearson-r'],
        'PearsonR_ARLSTM': m_ar['Pearson-r'],
        'KGE_Baseline': m_base['KGE'],
        'KGE_DataAssimilation': m_da['KGE'],
        'KGE_ARLSTM': m_ar['KGE'],
        'RMSE_Baseline': m_base['RMSE'],
        'RMSE_DataAssimilation': m_da['RMSE'],
        'RMSE_ARLSTM': m_ar['RMSE'],
    }


def main():
    parser = argparse.ArgumentParser(description="2-Basin Rapid Benchmark Evaluation")
    parser.add_argument("--num-basins", type=int, default=2, help="Number of basins to evaluate (default: 2)")
    args = parser.parse_args()

    run_dir_me = repo_root / 'tutorial' / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
    run_dir_ar = repo_root / 'tutorial' / 'model-runs' / 'arlstm-50basin-example_2107_080318'

    cfg_me = Config(run_dir_me / 'config.yml')
    cfg_me.update_config({'assimilation_config': {
        'assimilation_lead_time': 0, 'assimilation_window': 1, 'history': 5,
        'learning_rate': 0.05, 'assimilation_targets': ['h_n', 'c_n'],
        'epochs': 5, 'loss': 'MSE', 'optimizer': 'Adam'
    }})
    assimilation = Assimilation(cfg_me.assimilation_config)
    tester_me = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='test', init_model=True)
    tester_ar = RegressionTester(cfg=Config(run_dir_ar / 'config.yml'), run_dir=run_dir_ar, period='test', init_model=True)

    basins = tester_me.basins[:args.num_basins] if args.num_basins else tester_me.basins
    results = [evaluate_basin(tester_me, tester_ar, assimilation, tester_me.dataset[i], b) for i, b in enumerate(basins)]

    df = pd.DataFrame(results)
    out_csv = repo_root / 'tutorial' / 'model-runs' / '2_basin_final_comparison_results.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Evaluated {len(df)} basins. Results saved to: {out_csv}")


if __name__ == '__main__':
    main()
