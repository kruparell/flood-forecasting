import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add openhydronets_next root to sys.path
REPO_ROOT = Path('/usr/local/google/home/kruparell/openhydronets_next')
sys.path.insert(0, str(REPO_ROOT))

from googlehydrology.utils.config import Config
from googlehydrology.modelzoo.arlstm import ARLSTM
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.training.loss import MaskedMSELoss, MaskedCMALLoss

PRETRAINED_DIR = REPO_ROOT / "pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs"

def get_base_config_dict():
    with open(PRETRAINED_DIR / 'config.yml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['run_dir'] = PRETRAINED_DIR
    cfg_dict['dev_mode'] = True
    cfg_dict['verbose'] = False
    return cfg_dict

def create_arlstm_config(overrides=None):
    cfg_dict = get_base_config_dict()
    cfg_dict['model'] = 'arlstm'
    cfg_dict['autoregressive_inputs'] = ['streamflow_shift1']
    if overrides:
        cfg_dict.update(overrides)
    return Config(cfg_dict, dev_mode=True)

def generate_sample_batch(cfg, batch_size=2, seq_len=30, include_nan_ar=False):
    num_static = len(cfg.static_attributes)
    x_s = torch.randn(batch_size, num_static)
    
    # Hindcast features
    hindcast_dict = {}
    for group, feats in cfg.hindcast_inputs.items():
        for f in feats:
            hindcast_dict[f] = torch.randn(batch_size, seq_len, 1)
            
    # Forecast features
    forecast_dict = {}
    for group, feats in cfg.forecast_inputs.items():
        for f in feats:
            forecast_dict[f] = torch.randn(batch_size, seq_len, 1)
            
    # Autoregressive streamflow feature
    ar_streamflow = torch.randn(batch_size, seq_len, 1)
    if include_nan_ar:
        # Simulate missing streamflow (NaNs) in the second half of sequence
        ar_streamflow[:, seq_len // 2:, :] = float('nan')
        
    hindcast_dict['streamflow_shift1'] = ar_streamflow
    
    # Target discharge
    y = torch.randn(batch_size, seq_len, 1)
    
    batch = {
        'x_s': x_s,
        'x_d': hindcast_dict,
        'x_d_forecast': forecast_dict,
        'streamflow_shift1': ar_streamflow,
        'y': y,
        'c_n': torch.zeros((1, batch_size, cfg.hidden_size)),
        'h_n': torch.zeros((1, batch_size, cfg.hidden_size))
    }
    return batch

def test_regression_head():
    print("--- Test 1: Standard Regression Head ---")
    cfg = create_arlstm_config({'head': 'regression', 'loss': 'MSE'})
    model = ARLSTM(cfg)
    model.train()
    print(f"  Initialized ARLSTM: output_size={model.output_size}, num_ar_inputs={model._num_ar_inputs}")
    assert model.output_size == 1
    assert model._num_ar_inputs == 1

    batch = generate_sample_batch(cfg, batch_size=2, seq_len=20)
    res = model(batch)
    print(f"  Forward pass output keys: {list(res.keys())}")
    assert 'y_hat' in res
    assert res['y_hat'].shape == (2, 20, 1)
    assert not torch.isnan(res['y_hat']).any(), "Predictions should not contain NaNs"
    
    # Test Loss & Gradient Flow
    loss_fn = MaskedMSELoss(cfg)
    loss, _ = loss_fn(res, batch)
    print(f"  MSE Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    print("  ✓ Regression Head Forward + Backward Test Passed!\n")

def test_cmal_head():
    print("--- Test 2: CMAL Probabilistic Head ---")
    cfg = create_arlstm_config({'head': 'cmal', 'n_distributions': 3, 'loss': 'cmal'})
    model = ARLSTM(cfg)
    model.train()
    print(f"  Initialized ARLSTM with CMAL: output_size={model.output_size}, num_ar_inputs={model._num_ar_inputs}")
    assert model.output_size == 12  # 1 target * 4 params * 3 distributions
    assert model._num_ar_inputs == 1

    batch = generate_sample_batch(cfg, batch_size=2, seq_len=20)
    res = model(batch)
    print(f"  Forward pass output keys: {list(res.keys())}")
    for k in ['mu', 'b', 'tau', 'pi', 'y_hat']:
        assert k in res, f"Expected {k} in CMAL output"
    assert res['mu'].shape == (2, 20, 3)
    assert res['y_hat'].shape == (2, 20, 1)
    assert not torch.isnan(res['y_hat']).any(), "Expected mean y_hat should not contain NaNs"
    
    # Test CMAL Loss & Gradient Flow
    loss_fn = MaskedCMALLoss(cfg)
    loss, _ = loss_fn(res, batch)
    print(f"  CMAL NLL Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    print("  ✓ CMAL Head Forward + Backward Test Passed!\n")

def test_missing_observation_autoregressive_feedback():
    print("--- Test 3: Missing Streamflow NaN Autoregressive Feedback (Closed-Loop) ---")
    cfg = create_arlstm_config({'head': 'regression'})
    model = ARLSTM(cfg)
    model.eval()

    # Batch with 50% NaNs in the streamflow autoregressive input
    batch = generate_sample_batch(cfg, batch_size=2, seq_len=30, include_nan_ar=True)
    assert torch.isnan(batch['streamflow_shift1'][:, 15:, :]).all()

    res = model(batch)
    y_hat = res['y_hat']
    print(f"  Output y_hat shape: {y_hat.shape}")
    assert y_hat.shape == (2, 30, 1)
    assert not torch.isnan(y_hat).any(), "Model must autonomously substitute its own predictions for NaNs without propagating NaN outputs!"
    print("  ✓ Autoregressive Self-Feedback Substitution Test Passed!\n")

def test_weight_transfer_from_foundation():
    print("--- Test 4: Weight Transfer from Pretrained Foundation Checkpoint ---")
    cfg_ar = create_arlstm_config({'head': 'regression'})
    
    # 1. Load Foundation Checkpoint
    ckpt_path = PRETRAINED_DIR / 'model_epoch085.pt'
    raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    
    # 2. Instantiate ARLSTM Model
    ar_model = ARLSTM(cfg_ar)
    ar_sd = ar_model.state_dict()
    
    # Transfer compatible weights
    transferred = []
    for k, v in clean_ckpt.items():
        if k in ar_sd and ar_sd[k].shape == v.shape:
            ar_sd[k] = v
            transferred.append(k)
            
    ar_model.load_state_dict(ar_sd)
    print(f"  Successfully transferred {len(transferred)} weight tensors ({len(transferred)}/{len(clean_ckpt)}) from {ckpt_path.name} to ARLSTM.")
    
    # Verify forward pass with transferred weights
    batch = generate_sample_batch(cfg_ar, batch_size=2, seq_len=25)
    out = ar_model(batch)
    assert 'y_hat' in out
    print(f"  ARLSTM forward pass with transferred foundation weights successful! Output shape: {out['y_hat'].shape}")
    print("  ✓ Weight Transfer from Foundation Checkpoint Test Passed!\n")

if __name__ == '__main__':
    print("=================================================================")
    print("       Testing Autoregressive LSTM (ARLSTM) in openhydronets     ")
    print("=================================================================\n")
    test_regression_head()
    test_cmal_head()
    test_missing_observation_autoregressive_feedback()
    test_weight_transfer_from_foundation()
    print("=================================================================")
    print("       ALL ARLSTM UNIT & SYSTEM TESTS PASSED SUCCESSFULLY!       ")
    print("=================================================================")
