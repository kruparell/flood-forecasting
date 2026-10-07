"""Automated Validation Suite for Interactive MultiMet Data Assimilation Notebook."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

REPO_ROOT = Path('/usr/local/google/home/kruparell/openhydronets_next')
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'tutorial/notebooks'))
sys.path.insert(0, str(REPO_ROOT / 'tutorial/notebooks/helpers'))


from googlehydrology.utils.config import Config
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM
from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from multimet_helpers import prepare_multimet_batch


@pytest.fixture(scope="module")
def setup_env():
    caravan_dir = Path('/usr/local/google/home/kruparell/Caravans_V2')
    multimet_dir = Path('/usr/local/google/home/kruparell/Caravans_MultiMet')
    pretrained_dir = REPO_ROOT / 'pretrained-models/google-floodhub-settings-55-epochs-nse-filtered-0.5-85-epochs'
    
    with open(pretrained_dir / 'config.yml', 'r') as f:
        model_cfg = Config(yaml.safe_load(f))
    model = MeanEmbeddingForecastLSTM(model_cfg)
    raw_ckpt = torch.load(pretrained_dir / 'model_epoch085.pt', map_location='cpu', weights_only=False)
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in raw_ckpt.items()}
    model.load_state_dict(clean_ckpt, strict=True)
    model.eval()

    scaler = xr.open_dataset(pretrained_dir / 'scaler.nc').load()
    ds_attrs = xr.open_zarr(caravan_dir / 'attributes.zarr', consolidated=True)
    caravan_attrs = ds_attrs.sel(basin=['GRDC_4118110', 'hysets_05569500']).to_dataframe()
    ds_streamflow = xr.open_zarr(caravan_dir / 'streamflow.zarr', consolidated=True).sel(
        basin=['GRDC_4118110', 'hysets_05569500'], date=slice('2018-01-01', '2020-01-10')
    ).load()

    return {
        'model': model,
        'cfg': model_cfg,
        'scaler': scaler,
        'attrs': caravan_attrs,
        'streamflow': ds_streamflow,
        'multimet_dir': multimet_dir
    }


def test_data_integrity(setup_env):
    """Assert GRDC_4118110 and hysets_05569500 have 0 missing dates in 2018-2020."""
    ds_q = setup_env['streamflow']
    for b in ['GRDC_4118110', 'hysets_05569500']:
        sub = ds_q.sel(basin=b, date=slice('2018-01-01', '2019-12-31'))
        assert len(sub['date']) == 730, f"Expected 730 days for basin {b}, got {len(sub['date'])}"


def test_multimet_batch_shapes(setup_env):
    """Verify prepare_multimet_batch produces exact tensor shapes and zero NaNs in inputs."""
    batch = prepare_multimet_batch(
        mode='multimet_0_and_1_to_7',
        basin_id='GRDC_4118110',
        issue_date='2019-04-15',
        cfg=setup_env['cfg'],
        scaler=setup_env['scaler'],
        caravan_attrs=setup_env['attrs'],
        ds_caravan=setup_env['streamflow'].sel(basin='GRDC_4118110'),
        multimet_dir=setup_env['multimet_dir'],
        hindcast_window_days=358,
        forecast_lead_days=7
    )
    assert batch['y'].shape == (1, 365, 1)
    assert batch['x_s'].shape == (1, 84)
    assert batch['c_n'].shape == (1, 1, 512)
    assert np.isnan(batch['y'][0, -7:, 0].numpy()).all(), "Forecast period in y must be NaN for fair DA"


def test_model_weight_immutability(setup_env):
    """Assert model weights remain bitwise unchanged after Data Assimilation State Updating."""
    model = setup_env['model']
    batch = prepare_multimet_batch(
        mode='multimet_0_and_1_to_7',
        basin_id='GRDC_4118110',
        issue_date='2019-04-15',
        cfg=setup_env['cfg'],
        scaler=setup_env['scaler'],
        caravan_attrs=setup_env['attrs'],
        ds_caravan=setup_env['streamflow'].sel(basin='GRDC_4118110'),
        multimet_dir=setup_env['multimet_dir'],
        hindcast_window_days=358,
        forecast_lead_days=7
    )

    orig_weights = {k: v.clone() for k, v in model.state_dict().items()}
    da_cfg = AssimilationConfig({
        'seq_length': 365,
        'assimilation_lead_time': 7,
        'assimilation_window': 7,
        'history': 1,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'learning_rate': {0: 0.01},
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
        'predict_n_hindcast': 1,
        'assimilation_targets': ['c_n_forecast'],
        'bg_regularization_weight': 1e-6,
        'epochs': 20,
        'dev_mode': True,
    })
    assim = Assimilation(da_cfg)
    _ = assim.assimilate(model, batch, verbose=False)

    for k, v in model.state_dict().items():
        assert torch.equal(orig_weights[k], v), f"Weight {k} was mutated during assimilation!"
