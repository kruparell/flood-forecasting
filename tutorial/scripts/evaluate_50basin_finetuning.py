"""50-Basin fine-tuning workflow execution and evaluation."""

import copy
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_TUTORIAL_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TUTORIAL_DIR.parent

for _p in [_REPO_ROOT, _TUTORIAL_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from googlehydrology.datasetzoo.multimet import _convert_to_tensor
from googlehydrology.evaluation.tester import RegressionTester
from googlehydrology.training.loss import MaskedMSELoss
from googlehydrology.utils.config import Config


def main():
    print("Executing 50-basin fine-tuning evaluation...")
    run_dir_me = _TUTORIAL_DIR / 'model-runs' / 'generic-meanembedding-50basin_2107_080323'
    cfg_me = Config(run_dir_me / 'config.yml')

    tester_base = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='test', init_model=True)
    basins = tester_base.basins
    tester_train = RegressionTester(cfg=cfg_me, run_dir=run_dir_me, period='train', init_model=False)
    loss_fn = MaskedMSELoss(cfg_me)

    results = []
    streamflow_records = {}

    for idx, basin_id in enumerate(basins):
        t0 = time.time()
        sample_test = tester_base.dataset[idx]
        batch_test = tester_base.dataset.collate_fn([{k: _convert_to_tensor(k, v) for k, v in sample_test.items()}])
        obs_raw = batch_test['y'][0, :, 0].cpu().numpy()

        tester_base.model.eval()
        with torch.no_grad():
            out_base = tester_base.model(batch_test)
            y_hat_base = out_base['y_hat'][0]
            sim_base_raw = y_hat_base[-1, :].cpu().numpy() if y_hat_base.ndim == 2 else y_hat_base.cpu().numpy()

        model_ft = copy.deepcopy(tester_base.model)
        for param in model_ft.parameters():
            param.requires_grad = False
        if hasattr(model_ft, 'static_embedding_fc'):
            for param in model_ft.static_embedding_fc.parameters():
                param.requires_grad = True
        if hasattr(model_ft, 'head'):
            for param in model_ft.head.parameters():
                param.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.01)

        sample_train = tester_train.dataset[idx]
        batch_train = tester_train.dataset.collate_fn([{k: _convert_to_tensor(k, v) for k, v in sample_train.items()}])

        model_ft.train()
        for _ in range(5):
            optimizer.zero_grad()
            out_train = model_ft(batch_train)
            loss = loss_fn(out_train['y_hat'], batch_train['y'])
            loss.backward()
            optimizer.step()

        model_ft.eval()
        with torch.no_grad():
            out_ft = model_ft(batch_test)
            y_hat_ft = out_ft['y_hat'][0]
            sim_ft_raw = y_hat_ft[-1, :].cpu().numpy() if y_hat_ft.ndim == 2 else y_hat_ft.cpu().numpy()

        k_steps = min(len(obs_raw), len(sim_base_raw), len(sim_ft_raw))
        obs = obs_raw[-k_steps:]
        sim_base = sim_base_raw[-k_steps:]
        sim_ft = sim_ft_raw[-k_steps:]

        def calc(o: np.ndarray, s: np.ndarray):
            var_o = np.var(o)
            nse = float(1.0 - np.mean((s - o) ** 2) / (var_o + 1e-8))
            rmse = float(np.sqrt(np.mean((s - o) ** 2)))
            std_s, std_o = float(np.std(s)), float(np.std(o))
            if std_s > 1e-7 and std_o > 1e-7:
                r = float(np.corrcoef(o, s)[0, 1])
                mean_ratio = float(np.mean(s) / (np.mean(o) + 1e-8))
                kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (std_s / std_o - 1.0) ** 2 + (mean_ratio - 1.0) ** 2))
            else:
                r, kge = float(np.nan), float(np.nan)
            return {'NSE': nse, 'Pearson-r': r, 'KGE': kge, 'RMSE': rmse}

        m_base = calc(obs, sim_base)
        m_ft = calc(obs, sim_ft)

        results.append({
            'basin': basin_id,
            'NSE_Base': m_base['NSE'],
            'NSE_FineTuned': m_ft['NSE'],
            'KGE_Base': m_base['KGE'],
            'KGE_FineTuned': m_ft['KGE'],
            'PearsonR_Base': m_base['Pearson-r'],
            'PearsonR_FineTuned': m_ft['Pearson-r'],
            'RMSE_Base': m_base['RMSE'],
            'RMSE_FineTuned': m_ft['RMSE'],
        })
        streamflow_records[f'{basin_id}_obs'] = obs
        streamflow_records[f'{basin_id}_sim_base'] = sim_base
        streamflow_records[f'{basin_id}_sim_finetune'] = sim_ft

        print(f"[{idx+1:02d}/{len(basins):02d}] {basin_id:16s} | Base NSE: {m_base['NSE']:+.4f} -> FT NSE: {m_ft['NSE']:+.4f} ({time.time()-t0:.2f}s)")

    df = pd.DataFrame(results)
    out_csv = _TUTORIAL_DIR / 'model-runs' / '50basin_finetuning_evaluation_results.csv'
    df.to_csv(out_csv, index=False)
    np.savez_compressed(_TUTORIAL_DIR / 'model-runs' / '50basin_finetuning_streamflows.npz', **streamflow_records)
    print(f"Saved fine-tuning results to {out_csv}")


if __name__ == '__main__':
    main()
