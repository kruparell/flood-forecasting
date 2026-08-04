import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, '/usr/local/google/home/kruparell/flood-forecasting')

from googlehydrology.utils.config import Config
from googlehydrology.datasetzoo.multimet import Multimet
from googlehydrology.modelzoo.arlstm import ARLSTM
from googlehydrology.datautils.utils import load_basin_file
from tutorial.utils import collate_multimet_samples

def debug():
    cfg = Config(Path("/usr/local/google/home/kruparell/flood-forecasting/tutorial/configs/train-arlstm-local.yml"))
    print("Dataset:", cfg.dataset)
    print("AR inputs:", cfg.autoregressive_inputs)
    print("Holdout:", cfg.random_holdout_from_dynamic_features)
    print("Target noise:", cfg.target_noise_std)
    
    basins = load_basin_file(cfg.train_basin_file)
    ds = Multimet(cfg, is_train=True, period="train", basins=basins, compute_scaler=True)
    
    sample = ds[0]
    x_d = sample.get('x_d_hindcast', sample.get('x_d', None))
    sf_shift1 = x_d['streamflow_shift1']
    y = sample['y']
    print("sf_shift1 shape:", sf_shift1.shape, "NaN count:", np.isnan(sf_shift1).sum().item())
    print("y shape:", y.shape, "NaN count:", np.isnan(y).sum().item())
    print("sf_shift1 sample (first 10):", sf_shift1[:10].squeeze())
    print("y sample (first 10):", y[:10].squeeze())
    
    batch = collate_multimet_samples([ds[0], ds[1]])
    model = ARLSTM(cfg)
    model.eval()
    
    batch = model.pre_model_hook(batch, is_train=False)
    out = model(batch)
    y_hat = out['y_hat']
    print("y_hat shape:", y_hat.shape)
    print("y_hat sample (first 10):", y_hat[0, :10].squeeze().detach())

    # Check MSE between ground-truth sf_shift1 and y
    sf_tensor = batch['x_d']['streamflow_shift1']
    y_tensor = batch['y']
    print("Direct MSE between sf_shift1 and y:", torch.mean((sf_tensor[:, 1:, :] - y_tensor[:, 1:, :]) ** 2).item())

if __name__ == "__main__":
    debug()
