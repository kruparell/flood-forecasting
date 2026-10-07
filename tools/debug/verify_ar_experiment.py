import torch
import numpy as np
import pandas as pd
from pathlib import Path
from googlehydrology.utils.config import Config
from googlehydrology.datasetzoo.multimet import Multimet
from googlehydrology.modelzoo.arlstm import ARLSTM
from googlehydrology.datautils.utils import load_basin_file

def manual_collate(samples):
    batch = {}
    for k in samples[0].keys():
        if k == 'date':
            batch[k] = [s[k] for s in samples]
        elif k == 'x_d':
            batch[k] = {
                var: torch.stack([torch.tensor(s[k][var], dtype=torch.float32) for s in samples])
                for var in samples[0][k].keys()
            }
        elif isinstance(samples[0][k], np.ndarray):
            batch[k] = torch.stack([torch.tensor(s[k], dtype=torch.float32) for s in samples])
        else:
            batch[k] = [s[k] for s in samples]
    return batch

def run_experiment(use_ar_residual=False, name="Experiment"):
    print(f"\n=======================================================")
    print(f"   RUNNING: {name} (use_ar_residual={use_ar_residual})")
    print(f"=======================================================")
    
    cfg_path = Path("/usr/local/google/home/kruparell/flood-forecasting/tutorial/configs/train-arlstm-local.yml")
    cfg = Config(cfg_path)
    cfg.use_ar_residual = use_ar_residual
    
    # 1. Dataset Setup
    train_basins = load_basin_file(cfg.train_basin_file)
    test_basins = load_basin_file(cfg.test_basin_file)
    
    train_ds = Multimet(cfg, is_train=True, period="train", basins=train_basins, compute_scaler=True)
    test_ds = Multimet(cfg, is_train=False, period="test", basins=test_basins, compute_scaler=False)
    
    # 2. Model Setup
    model = ARLSTM(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    
    # 3. Quick Training Loop (5 Epochs)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=manual_collate
    )
    
    model.train()
    for epoch in range(1, 6):
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = model.pre_model_hook(batch, is_train=True)
            optimizer.zero_grad()
            out = model(batch)
            y_hat = out['y_hat']
            y = batch['y']
            
            # Mask out NaNs
            valid = ~torch.isnan(y)
            loss = criterion(y_hat[valid], y[valid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            if steps >= 50: # Limit updates per epoch for quick test
                break
        print(f"Epoch {epoch}/5 - Train Loss: {total_loss / steps:.6f}")
        
    # 4. Evaluation on Test Basins
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=manual_collate
    )
    
    all_y, all_y_hat, all_sf = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = model.pre_model_hook(batch, is_train=False)
            out = model(batch)
            y_hat = out['y_hat']
            y = batch['y']
            sf = batch['x_d']['streamflow_shift1']
            
            all_y.append(y.numpy())
            all_y_hat.append(y_hat.numpy())
            all_sf.append(sf.numpy())
            
    y_arr = np.concatenate(all_y, axis=0)
    y_hat_arr = np.concatenate(all_y_hat, axis=0)
    sf_arr = np.concatenate(all_sf, axis=0)
    
    # Calculate NSE across basins
    nses = []
    for b in range(y_arr.shape[0]):
        y_b = y_arr[b].flatten()
        y_hat_b = y_hat_arr[b].flatten()
        valid = ~np.isnan(y_b) & ~np.isnan(y_hat_b)
        if np.sum(valid) > 0:
            var_y = np.var(y_b[valid])
            mse = np.mean((y_b[valid] - y_hat_b[valid]) ** 2)
            nse = 1.0 - mse / var_y if var_y > 0 else 0.0
            nses.append(nse)
            
    median_nse = np.median(nses)
    print(f"--> Summary Test Median NSE: {median_nse:.6f}")
    return median_nse

if __name__ == "__main__":
    nse_std = run_experiment(use_ar_residual=False, name="Standard ARLSTM (No Residual)")
    nse_res = run_experiment(use_ar_residual=True, name="Enhanced ARLSTM (With AR Residual Connection)")
    
    print("\n=======================================================")
    print("                COMPARISON SUMMARY                     ")
    print("=======================================================")
    print(f"Standard ARLSTM Test Median NSE : {nse_std:.6f}")
    print(f"Enhanced ARLSTM Test Median NSE : {nse_res:.6f}")
    print("=======================================================")
