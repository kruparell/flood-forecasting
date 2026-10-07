"""Unit tests for googlehydrology.evaluation.assimilation."""

import unittest
from unittest.mock import patch
from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig
import numpy as np
import pandas as pd
import torch


class MockEmbeddingModel(torch.nn.Module):
  """Mock model with static and dynamic embeddings for unit testing."""

  def __init__(self, hidden_dim: int = 4):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.fc = torch.nn.Linear(hidden_dim, 1)

  def forward(self, data):
    e_stat = data.get('static_embedding', None)
    if e_stat is None:
      e_stat = torch.ones(1, self.hidden_dim)
    e_dyn = data.get('hindcast_embedding', None)
    if e_dyn is None:
      x_d = data.get('x_d', data.get('x_d_hindcast', None))
      if isinstance(x_d, dict):
        x_d = list(x_d.values())[0]
      if x_d is not None:
        e_dyn = x_d
      else:
        e_dyn = torch.zeros(1, 10, self.hidden_dim)
    y_hat = (e_stat.unsqueeze(1) + e_dyn).sum(dim=-1, keepdim=True)
    return {
        'y_hat': y_hat,
        'static_embedding': e_stat,
        'hindcast_embedding': e_dyn,
    }


class AssimilationTest(unittest.TestCase):

  def test_ar1_postprocessing(self):
    """Verifies AR(1) post-processing exponential decay and non-negativity constraint."""
    try:
      from googlehydrology.evaluation.eval_utils import apply_ar1_postprocessing
    except ImportError:
      self.skipTest('googlehydrology.evaluation.eval_utils is not available in this repository.')
    q_base = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    q_obs_window = np.array([5.0, 8.0, 12.0], dtype=np.float32)
    q_ar1 = apply_ar1_postprocessing(q_base, q_obs_window, rho=0.5)
    self.assertAlmostEqual(q_ar1[0], 11.0, places=3)
    self.assertAlmostEqual(q_ar1[1], 10.5, places=3)

    q_obs_low = np.array([0.0, 0.0, -100.0], dtype=np.float32)
    q_ar1_neg = apply_ar1_postprocessing(q_base, q_obs_low, rho=0.85)
    self.assertTrue((q_ar1_neg >= 0.0).all())

  def _create_mef_model_and_data(
      self, hidden_size=16, seq_length=14, lead_time=2
  ):
    from googlehydrology.utils.config import Config
    from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM

    cfg_dict = {
        'model': 'mean_embedding_forecast_lstm',
        'head': 'regression',
        'hidden_size': hidden_size,
        'seq_length': seq_length,
        'lead_time': lead_time,
        'predict_last_n': lead_time,
        'target_variables': ['streamflow'],
        'static_attributes': ['area'],
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'dynamics_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'hindcast_inputs': ['era5_precip'],
        'forecast_inputs': ['hres_precip'],
        'compile': False,
        'dev_mode': True,
    }
    cfg = Config(cfg_dict, dev_mode=True)
    model = MeanEmbeddingForecastLSTM(cfg=cfg)
    data = {
        'x_s': torch.ones(1, 1),
        'x_d_hindcast': {
            'era5_precip': torch.randn(1, seq_length - lead_time, 1)
        },
        'x_d_forecast': {'hres_precip': torch.randn(1, seq_length, 1)},
        'y': torch.ones(1, seq_length, 1) * 2.0,
    }
    return model, data

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_itemised_type2_embedded_all_da_with_mean_embedding_forecast_lstm(
      self, mock_scaler
  ):
    """Itemised Test: Type 2 Embedded DA (embedded_all: static, hindcast, forecast) with MeanEmbeddingForecastLSTM."""
    model, data = self._create_mef_model_and_data()
    da_cfg_dict = {
        'seq_length': 14,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 2,
        'learning_rate': 0.05,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_all'],
        'static_embedding_regularization_weight': 1e-4,
        'hindcast_embedding_regularization_weight': 0.01,
        'target_variables': ['streamflow'],
        'predict_last_n': 2,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    assim = Assimilation(da_cfg)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertEqual(res['y_hat'].shape[1], 14)
    self.assertIn('static_embedding', res)
    self.assertIn('hindcast_embedding', res)
    self.assertIn('forecast_embedding', res)
    self.assertIsNotNone(res['static_embedding'])
    self.assertIsNotNone(res['hindcast_embedding'])
    self.assertIsNotNone(res['forecast_embedding'])
    self.assertTrue(all(p.requires_grad for p in model.parameters()))
    # `assimilation.py` must return the model's native output dict untouched.
    # Computing metrics is the tester's job, never the assimilator's.
    self.assertNotIn('hindcast_metrics_pre', res)
    self.assertNotIn('hindcast_metrics_post', res)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_embedded_both_da_with_mean_embedding_forecast_lstm(
      self, mock_scaler
  ):
    """Itemised Test: Embedded DA (embedded_both: static and hindcast) with MeanEmbeddingForecastLSTM."""
    model, data = self._create_mef_model_and_data()
    da_cfg_dict = {
        'seq_length': 14,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 2,
        'learning_rate': 0.05,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'static_embedding_regularization_weight': 1e-4,
        'hindcast_embedding_regularization_weight': 0.01,
        'target_variables': ['streamflow'],
        'predict_last_n': 2,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    assim = Assimilation(da_cfg)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertEqual(res['y_hat'].shape[1], 14)
    self.assertIn('static_embedding', res)
    self.assertIn('hindcast_embedding', res)
    self.assertNotIn('forecast_embedding', res)
    self.assertTrue(all(p.requires_grad for p in model.parameters()))

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_component_based_da_with_explicit_regularization_weights(
      self, mock_scaler
  ):
    """Verifies model-agnostic component-based DA using assimilation_components dictionary."""
    model, data = self._create_mef_model_and_data()
    model.eval()

    da_cfg_dict = {
        'seq_length': 14,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 2,
        'learning_rate': 0.05,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_components': {
            'static_embedding': 1e-4,
            'hindcast_embedding': 0.02,
        },
        'target_variables': ['streamflow'],
        'predict_last_n': 2,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    self.assertIn('static_embedding', da_cfg.assimilation_components)
    self.assertEqual(
        da_cfg.assimilation_components['static_embedding']['weight'], 1e-4
    )
    self.assertIn('hindcast_embedding', da_cfg.assimilation_components)
    self.assertEqual(
        da_cfg.assimilation_components['hindcast_embedding']['weight'], 0.02
    )

    assim = Assimilation(da_cfg)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertIn('static_embedding', res)
    self.assertIn('hindcast_embedding', res)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_component_based_da_with_per_component_lr(self, mock_scaler):
    """Verifies component-based DA with per-component learning rates and weights."""
    model, data = self._create_mef_model_and_data()
    model.eval()

    da_cfg_dict = {
        'seq_length': 14,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 2,
        'learning_rate': 0.01,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_components': {
            'static_embedding': {'weight': 1e-4, 'learning_rate': 0.02},
            'hindcast_embedding': {'weight': 0.05, 'lr': 0.08},
            'forecast_embedding': {'weight': 0.05, 'lr': 0.08},
        },
        'target_variables': ['streamflow'],
        'predict_last_n': 2,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    self.assertEqual(
        da_cfg.assimilation_components['static_embedding']['lr'], 0.02
    )
    self.assertEqual(
        da_cfg.assimilation_components['hindcast_embedding']['lr'], 0.08
    )

    assim = Assimilation(da_cfg)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertIn('static_embedding', res)
    self.assertIn('hindcast_embedding', res)
    self.assertIn('forecast_embedding', res)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_regularization_uses_mean_normalization(self, mock_scaler):
    """Verifies that the regularizer uses torch.mean (intensive) rather than torch.sum (extensive)."""
    model, data = self._create_mef_model_and_data()

    da_cfg_dict = {
        'seq_length': 14,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 2,
        'learning_rate': 0.01,
        'epochs': 3,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_components': {
            'hindcast_embedding': {'weight': 1.0},
        },
        'target_variables': ['streamflow'],
        'predict_last_n': 2,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    assim = Assimilation(da_cfg)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertFalse(torch.isnan(res['y_hat']).any())

  def test_warm_start_identical_when_zero_lr_or_epochs(self):
    """Verifies that baseline and assimilation outputs match identically (|y_assim - y_base| == 0) at lr=0 or epochs=0."""
    model = MockEmbeddingModel(hidden_dim=4)
    data = {
        'x_d': torch.randn(1, 10, 4),
        'y': torch.ones(1, 10, 1) * 3.0,
    }
    base_out = model(data)['y_hat']

    for epochs_val, lr_val in [(0, 0.05), (5, 0.0)]:
      da_cfg_dict = {
          'seq_length': 10,
          'history': 2,
          'assimilation_window_length': 1,
          'assimilation_lead_time': 0,
          'learning_rate': lr_val,
          'epochs': epochs_val,
          'loss': 'MSE',
          'optimizer': 'Adam',
          'assimilation_targets': ['embedded_both'],
          'target_variables': ['streamflow'],
          'predict_last_n': 1,
      }
      da_cfg = AssimilationConfig(da_cfg_dict)
      assim = Assimilation(da_cfg)
      res = assim.assimilate(model, data, verbose=False)
      diff = (res['y_hat'] - base_out).abs().max().item()
      self.assertEqual(diff, 0.0)

  def test_embedding_da_continuity_as_lr_approaches_zero(self):
    """Verifies that as learning_rate -> 0, embedding DA smoothly and continuously tends to the baseline."""
    model = MockEmbeddingModel(hidden_dim=4)
    data = {
        'x_d': torch.randn(1, 10, 4),
        'y': torch.ones(1, 10, 1) * 10.0,
    }
    base_out = model(data)['y_hat']

    diffs = []
    lrs = [0.1, 0.01, 0.001, 1e-4, 0.0]

    for lr_val in lrs:
      da_cfg_dict = {
          'seq_length': 10,
          'history': 2,
          'assimilation_window_length': 1,
          'assimilation_lead_time': 0,
          'learning_rate': lr_val,
          'epochs': 3,
          'loss': 'MSE',
          'optimizer': 'Adam',
          'assimilation_targets': ['embedded_both'],
          'target_variables': ['streamflow'],
          'predict_last_n': 1,
      }
      da_cfg = AssimilationConfig(da_cfg_dict)
      assim = Assimilation(da_cfg)
      res = assim.assimilate(model, data, verbose=False)
      diff = (res['y_hat'] - base_out).abs().max().item()
      diffs.append(diff)

    self.assertLessEqual(diffs[-1], 1e-6)
    for i in range(len(diffs) - 1):
      self.assertGreaterEqual(diffs[i], diffs[i + 1] - 1e-7)

  def test_cmal_probabilistic_mixture_preservation(self):
    """Tests that CMAL mixture parameters (mu, b, tau, pi) are preserved across unrolled chunks."""

    class MockCMALEmbeddingModel(torch.nn.Module):

      def forward(self, data):
        y_len = data['y'].shape[1] if 'y' in data else 10
        mu = torch.ones(1, y_len, 4) * 2.0
        b = torch.ones(1, y_len, 4) * 0.5
        tau = torch.ones(1, y_len, 4) * 0.5
        pi = torch.ones(1, y_len, 4) * 0.25
        y_hat = torch.ones(1, y_len, 1) * 2.0
        return {
            'y_hat': y_hat,
            'mu': mu,
            'b': b,
            'tau': tau,
            'pi': pi,
            'static_embedding': torch.ones(1, 8),
            'hindcast_embedding': torch.ones(1, y_len, 8),
        }

    cfg_dict = {
        'seq_length': 10,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 0,
        'learning_rate': 0.0,
        'epochs': 1,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 1,
    }
    cfg = AssimilationConfig(cfg_dict)
    assim = Assimilation(cfg)
    model = MockCMALEmbeddingModel()
    data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
    res = assim.assimilate(model, data, verbose=False)
    for k in ['mu', 'b', 'tau', 'pi']:
      self.assertIn(k, res)
      self.assertEqual(res[k].shape[1], 10)

  def test_model_requires_grad_preserved_after_assimilation(self):
    """Verifies that model parameters retain their original requires_grad status after assimilate."""

    class LinearEmbeddingModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

      def forward(self, data):
        x = data.get('x_d_hindcast', data.get('x_d'))
        return {
            'y_hat': self.fc(x),
            'static_embedding': torch.ones(1, 4),
            'hindcast_embedding': torch.ones(1, x.shape[1], 4),
        }

    model = LinearEmbeddingModel()
    self.assertTrue(all(p.requires_grad for p in model.parameters()))
    cfg_dict = {
        'seq_length': 10,
        'history': 2,
        'assimilation_window_length': 1,
        'assimilation_lead_time': 0,
        'learning_rate': 0.01,
        'epochs': 1,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 1,
    }
    cfg = AssimilationConfig(cfg_dict)
    assim = Assimilation(cfg)
    data = {'x_d': torch.zeros(1, 10, 2), 'y': torch.ones(1, 10, 1)}
    assim.assimilate(model, data, verbose=False)
    self.assertTrue(all(p.requires_grad for p in model.parameters()))

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_config_with_assimilation_section_and_model_loading(
      self, mock_scaler
  ):
    """Tests Config parsing of assimilation_config and checks get_model validation."""
    from googlehydrology.utils.config import Config
    from googlehydrology.modelzoo import get_model

    cfg_dict = {
        'model': 'mean_embedding_forecast_lstm',
        'head': 'regression',
        'hidden_size': 16,
        'seq_length': 14,
        'lead_time': 2,
        'predict_last_n': 2,
        'target_variables': ['streamflow'],
        'static_attributes': ['area'],
        'statics_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'dynamics_embedding': {
            'type': 'fc',
            'hiddens': [8],
            'activation': 'tanh',
            'dropout': 0.0,
        },
        'hindcast_inputs': ['era5_precip'],
        'forecast_inputs': ['hres_precip'],
        'compile': False,
        'dev_mode': True,
        'assimilation_config': {
            'assimilation_window_length': 1,
            'history': 2,
            'assimilation_lead_time': 2,
            'learning_rate': 0.01,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['embedded_both'],
        },
    }
    cfg = Config(cfg_dict, dev_mode=True)
    self.assertIsNotNone(cfg.assimilation_config)
    self.assertEqual(cfg.assimilation_config.history, 2)
    model = get_model(cfg)
    self.assertIsNotNone(model)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_forecast_gap_check_raises_when_no_gap(self, mock_scaler):
    """Verifies the forecast gap guard raises when assimilation_lead_time != model lead_time."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    da_cfg_dict = {
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 0,
        'learning_rate': 0.01,
        'epochs': 2,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    assim = Assimilation(da_cfg)
    with self.assertRaises(ValueError) as cm:
      assim.assimilate(model, data, verbose=False)
    self.assertIn('[DA Forecast Gap Error]', str(cm.exception))
    self.assertIn('Expected a 7-day unassimilated gap', str(cm.exception))
    # The diagnostic must name the actual gap so the misconfiguration is obvious.
    self.assertIn('but got 0 days', str(cm.exception))

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_forecast_gap_check_passes_with_7day_gap(self, mock_scaler):
    """Verifies that assimilation succeeds when a 7-day gap is present for forecast dates."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    da_cfg_dict = {
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'epochs': 2,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    }
    da_cfg = AssimilationConfig(da_cfg_dict)
    assim = Assimilation(da_cfg)
    self.assertEqual(assim.assimilation_end_step, 358)
    res = assim.assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertEqual(res['y_hat'].shape[1], 365)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_embedding_da_zero_epochs_parity_and_forecast_finite(
      self, mock_scaler
  ):
    """Verifies bit-for-bit open-loop parity at epochs=0 and finite post-DA forecast rollout."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    with torch.no_grad():
      base_out = model(data)
      base_y = base_out['y_hat']

    da_cfg_zero = AssimilationConfig({
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.05,
        'epochs': 0,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    })
    assim_zero = Assimilation(da_cfg_zero)
    res_zero = assim_zero.assimilate(model, data, verbose=False)
    self.assertEqual((res_zero['y_hat'] - base_y).abs().max().item(), 0.0)

    da_cfg_opt = AssimilationConfig({
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.05,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    })
    assim_opt = Assimilation(da_cfg_opt)
    res_opt = assim_opt.assimilate(model, data, verbose=False)
    self.assertTrue(torch.isfinite(res_opt['y_hat'][:, 358:365, :]).all())
    # `MaskedMSELoss._subset_target` slices `pred[:, :, 0:1]`, so DA only ever
    # optimizes channel 0. Averaging over all output channels here would dilute
    # the signal with 11 channels that assimilation never touches.
    obs = data['y'][:, 328:358, 0:1]
    mse_base = torch.mean(
        (base_out['y_hat'][:, 328:358, 0:1] - obs) ** 2
    ).item()
    mse_opt = torch.mean(
        (res_opt['y_hat'][:, 328:358, 0:1] - obs) ** 2
    ).item()
    self.assertLess(mse_opt, mse_base)

  def test_non_finite_gradients_do_not_poison_output(self):
    """Verifies a non-finite gradient is skipped instead of being written into the state.

    This is the Stage 17b failure mode: `clip_grad_norm_` propagates rather than
    blocks a non-finite gradient, so a single bad step writes NaN into the
    optimized tensor. Every later epoch then has a NaN loss and is skipped by the
    pre-backward isfinite guard, which permanently locks the NaN in and makes the
    final forward pass emit all-NaN predictions.
    """

    class InfiniteGradModel(torch.nn.Module):
      """Finite forward, non-finite backward: d/dx sqrt(x) is infinite at x=0."""

      def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

      def forward(self, data):
        e_dyn = data.get('hindcast_embedding', None)
        if e_dyn is None:
          e_dyn = torch.zeros(1, 10, 4)
        y_hat = torch.sqrt(torch.relu(e_dyn)).sum(dim=-1, keepdim=True)
        return {
            'y_hat': y_hat * self.scale,
            'hindcast_embedding': e_dyn,
            'static_embedding': torch.ones(1, 4),
        }

    model = InfiniteGradModel()
    data = {'y': torch.ones(1, 10, 1) * 3.0, 'x_d': torch.zeros(1, 10, 4)}
    da_cfg = AssimilationConfig({
        'seq_length': 10,
        'history': 2,
        'assimilation_window_length': 2,
        'assimilation_lead_time': 0,
        'learning_rate': 0.05,
        'epochs': 5,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['embedded_dyn'],
        'target_variables': ['streamflow'],
        'predict_last_n': 1,
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    self.assertTrue(torch.isfinite(res['y_hat']).all())
    self.assertTrue(torch.isfinite(res['hindcast_embedding']).all())

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_unsupported_component_raises_instead_of_silent_noop(
      self, mock_scaler
  ):
    """Verifies unsupported DA targets fail loudly rather than assimilating nothing.

    Precipitation forcing DA has no baseline tensor this engine can resolve:
    `total_precipitation` is neither a model output nor a top-level batch key
    nor an LSTM initial state. Previously this silently optimized an empty
    component set and reported the result as a successful DA run.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    da_cfg = AssimilationConfig({
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'epochs': 2,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['precip'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    })
    with self.assertRaises(NotImplementedError) as cm:
      Assimilation(da_cfg).assimilate(model, data, verbose=False)
    self.assertIn('[DA Unsupported Component]', str(cm.exception))
    self.assertIn('total_precipitation', str(cm.exception))

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cell_state_da_runs_and_optimizes_both_cell_states(
      self, mock_scaler
  ):
    """Verifies `c_both` assimilates the hindcast and forecast cell states.

    `c_0_hindcast` / `c_0_forecast` name the LSTM state entering the first
    timestep of the assimilation window. The model publishes that state in its
    output dict, so the engine resolves the background from there. Hidden states
    are deliberately excluded: `c_both` optimizes cell states only.
    """
    hidden_size = 8
    model, data = self._create_mef_model_and_data(
        hidden_size=hidden_size, seq_length=365, lead_time=7
    )
    base_cfg = {
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['c_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    }
    background = Assimilation(
        AssimilationConfig({**base_cfg, 'epochs': 0})
    ).assimilate(model, data, verbose=False)
    res = Assimilation(
        AssimilationConfig({**base_cfg, 'epochs': 2})
    ).assimilate(model, data, verbose=False)

    self.assertIn('y_hat', res)
    self.assertTrue(torch.isfinite(res['y_hat']).all())
    for name in ('c_0_hindcast', 'c_0_forecast'):
      self.assertIn(name, res, f'{name} was not assimilated')
      self.assertEqual(res[name].shape[-1], hidden_size)
      self.assertTrue(torch.isfinite(res[name]).all())
      # A component identical to the zero-epoch background means the optimizer
      # never moved it -- the failure mode when the control variable sits too
      # far upstream for a usable gradient to reach it.
      self.assertGreater(
          float((res[name] - background[name]).abs().max()),
          0.0,
          f'{name} did not move away from its background',
      )
    # Hidden states are out of scope for `c_both` and must not be optimized.
    for name in ('h_0_hindcast', 'h_0_forecast'):
      self.assertNotIn(name, res)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cell_state_zero_perturbation_reproduces_baseline(self, mock_scaler):
    """Verifies the `sequence_start` cell-state baseline is all zeros.

    With `state_anchor='sequence_start'` the control variable is the LSTM state
    at t=0, where PyTorch's implicit default is all zeros. Seeding it with the
    model's *terminal* state (`c_n_*`) instead would make a zero-perturbation
    assimilated pass differ from the unassimilated one, confounding the measured
    effect of DA with the effect of the seeding. This pins that pass to a no-op.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    with torch.no_grad():
      baseline = model(data)['y_hat'].clone()

      zeros = torch.zeros(1, 1, 8)
      perturbed_data = dict(data)
      perturbed_data['assimilation_overrides'] = {
          'c_0_hindcast': zeros.clone(),
          'c_0_forecast': zeros.clone(),
      }
      zero_override = model(perturbed_data)['y_hat']

    torch.testing.assert_close(zero_override, baseline)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_state_anchor_does_not_relocate_batch_warm_start(self, mock_scaler):
    """A warm start carried in the batch must stay at the sequence start.

    The notebook batches ship `c_n` / `h_n` as a warm start for the forecast
    LSTM. Those are not assimilation control variables, so moving the anchor
    must not move them -- otherwise turning the anchor on silently perturbs
    every run, including ones that assimilate an embedding and never touch a
    recurrent state at all.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    warm = {
        'h_n': torch.randn(1, 1, 8),
        'c_n': torch.randn(1, 1, 8),
    }
    model.eval()
    with torch.no_grad():
      plain = model({**data, **warm})['y_hat'].clone()
      anchored = model({**data, **warm, 'state_anchor_step': 328})['y_hat']

    torch.testing.assert_close(anchored, plain)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_state_anchor_without_override_is_bit_exact(self, mock_scaler):
    """Splitting the LSTM run at the anchor must not change the prediction.

    The anchor is implemented by running each LSTM in two segments. With no
    override the second segment is seeded with the first segment's terminal
    state, so the split has to be mathematically invisible. If it is not, every
    anchored measurement is contaminated by the split itself.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    with torch.no_grad():
      unsplit = model(data)['y_hat'].clone()
      split = model({**data, 'state_anchor_step': 328})['y_hat']

    torch.testing.assert_close(split, unsplit)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_state_anchor_publishes_reinjectable_background(self, mock_scaler):
    """The published anchor state must round-trip as a zero-increment override.

    The engine takes the model's `c_0_*` / `h_0_*` output as the background for
    the recurrent-state control variables. Feeding that background straight back
    in must therefore reproduce the unassimilated run exactly, otherwise a
    zero-magnitude update would already move the prediction.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    anchored = {**data, 'state_anchor_step': 328}
    with torch.no_grad():
      out = model(anchored)
      # The anchor state is mid-sequence, so it must not be all zeros -- that
      # was precisely the defect of anchoring at the start of the sequence.
      self.assertGreater(float(out['c_0_hindcast'].abs().max()), 0.0)

      round_trip = model({
          **anchored,
          'assimilation_overrides': {
              name: out[name].clone()
              for name in (
                  'c_0_hindcast',
                  'h_0_hindcast',
                  'c_0_forecast',
                  'h_0_forecast',
              )
          },
      })['y_hat']

    torch.testing.assert_close(round_trip, out['y_hat'])

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_partial_state_anchor_inherits_missing_partner(self, mock_scaler):
    """Assimilating only `c_0_*` must leave `h_0_*` on its own trajectory.

    At the start of the sequence an absent partner is correctly zero-filled,
    because zero is what the LSTM would have started from anyway. At a
    mid-sequence anchor zeroing it would discard the hidden state the model had
    actually accumulated, so a cell-state-only update would silently also wipe
    the hidden state.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    anchored = {**data, 'state_anchor_step': 328}
    with torch.no_grad():
      out = model(anchored)
      cell_only = model({
          **anchored,
          'assimilation_overrides': {
              'c_0_hindcast': out['c_0_hindcast'].clone(),
              'c_0_forecast': out['c_0_forecast'].clone(),
          },
      })['y_hat']

    torch.testing.assert_close(cell_only, out['y_hat'])

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_anchor_amplifies_state_gradient(self, mock_scaler):
    """The window-start anchor must deliver a stronger gradient than t=0.

    This is the whole point of the anchor. At t=0 the control variable sits
    `seq_length - lead_time - window` steps upstream of the loss window and the
    adjoint decays across all of them, to the point where the gradient can reach
    the float32 subnormal floor and the optimizer stops moving entirely.
    """
    torch.manual_seed(0)
    anchor_step = 328
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.train(True)
    for p in model.parameters():
      p.requires_grad = False

    def _grad_norm(step: int) -> float:
      anchored = {**data, 'state_anchor_step': step}
      with torch.no_grad():
        # Perturb around each anchor's own background: zeros at t=0 (the LSTM's
        # implicit default) and the model's own state at the window start.
        background = (
            model(anchored)['c_0_hindcast'].clone()
            if step > 0
            else torch.zeros(1, 1, 8)
        )
      control = background.clone().requires_grad_(True)
      pred = model({**anchored, 'assimilation_overrides': {
          'c_0_hindcast': control,
      }})['y_hat']
      window = pred[:, anchor_step:358, :]
      loss = ((window - data['y'][:, anchor_step:358, :]) ** 2).mean()
      loss.backward()
      return float(control.grad.norm())

    self.assertGreater(_grad_norm(anchor_step), 10.0 * _grad_norm(0))

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_state_anchor_outside_sequence_raises(self, mock_scaler):
    """An unusable anchor must fail loudly rather than drop the override.

    A dropped override detaches the control variable from the graph, so the
    optimizer spins on a zero gradient and the run looks like a silent no-op.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    model.eval()
    with self.assertRaises(ValueError) as cm:
      model({
          **data,
          'state_anchor_step': 10_000,
          'assimilation_overrides': {'c_0_hindcast': torch.zeros(1, 1, 8)},
      })
    self.assertIn('state_anchor_step', str(cm.exception))

  def test_state_anchor_config_default_and_validation(self):
    """`window_start` is the default; unknown anchors are rejected."""
    self.assertEqual(AssimilationConfig({}).state_anchor, 'window_start')
    self.assertEqual(
        AssimilationConfig({'state_anchor': 'sequence_start'}).state_anchor,
        'sequence_start',
    )
    with self.assertRaises(ValueError):
      _ = AssimilationConfig({'state_anchor': 'somewhere_else'}).state_anchor

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_sequence_start_anchor_reproduces_legacy_behaviour(
      self, mock_scaler
  ):
    """`state_anchor='sequence_start'` must restore the pre-anchor engine.

    Old sweeps were run with the control variable at t=0, so that formulation
    has to stay reachable for their results to be reproducible.
    """
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    da_cfg = AssimilationConfig({
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'epochs': 0,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['c_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
        'state_anchor': 'sequence_start',
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)

    # At t=0 the background is the LSTM's implicit all-zero initial state.
    for name in ('c_0_hindcast', 'c_0_forecast'):
      torch.testing.assert_close(res[name], torch.zeros_like(res[name]))


  def test_regularization_weight_reaches_recurrent_state_targets(self):
    """Verifies state targets get the state weight, not an embedding weight.

    `c_0_hindcast` contains the substring "hindcast" and `c_0_forecast` contains
    "forecast", so an embedding-first dispatch routes both to the embedding
    weights and leaves the state regularization axis silently inert.
    """
    cfg = AssimilationConfig({
        'assimilation_targets': ['c_both'],
        'bg_regularization_weight': 0.25,
        'hindcast_embedding_regularization_weight': 0.01,
        'forecast_embedding_regularization_weight': 0.02,
        'static_embedding_regularization_weight': 1e-6,
    })
    components = cfg.assimilation_components
    self.assertEqual(components['c_0_hindcast']['weight'], 0.25)
    self.assertEqual(components['c_0_forecast']['weight'], 0.25)

    # Embedding targets must be unaffected by the reordering.
    emb_cfg = AssimilationConfig({
        'assimilation_targets': ['embedded_all'],
        'bg_regularization_weight': 0.25,
        'hindcast_embedding_regularization_weight': 0.01,
        'forecast_embedding_regularization_weight': 0.02,
        'static_embedding_regularization_weight': 1e-6,
    })
    emb_components = emb_cfg.assimilation_components
    self.assertEqual(emb_components['static_embedding']['weight'], 1e-6)
    self.assertEqual(emb_components['hindcast_embedding']['weight'], 0.01)
    self.assertEqual(emb_components['forecast_embedding']['weight'], 0.02)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_masked_nse_loss_assimilation(self, mock_scaler):
    """Verifies DA runs with `loss: 'NSE'` and correctly infers target stds if missing."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=365, lead_time=7
    )
    da_cfg = AssimilationConfig({
        'seq_length': 365,
        'history': 1,
        'assimilation_window_length': 30,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'epochs': 2,
        'loss': 'NSE',
        'optimizer': 'Adam',
        'assimilation_targets': ['static_embedding'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    self.assertIn('y_hat', res)
    self.assertTrue(torch.isfinite(res['y_hat']).all())
    self.assertIn('static_embedding', res)
    self.assertTrue(torch.isfinite(res['static_embedding']).all())

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_info_logging_does_not_abort_at_epoch_zero_and_bg_sensitive(
      self, mock_scaler
  ):
    """Regression test: under INFO logging, epochs > 1 must not break at epoch 0 and bg must differentiate."""
    import logging
    from googlehydrology.evaluation import assimilation as assim_mod

    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=60, lead_time=7
    )
    prev_level = assim_mod.LOGGER.level
    assim_mod.LOGGER.setLevel(logging.INFO)
    try:
      cfg_ep1 = AssimilationConfig({
          'seq_length': 60,
          'history': 1,
          'assimilation_window_length': 20,
          'assimilation_lead_time': 7,
          'learning_rate': 0.05,
          'epochs': 1,
          'loss': 'MSE',
          'optimizer': 'Adam',
          'assimilation_targets': ['hindcast_embedding'],
          'hindcast_embedding_regularization_weight': 0.0,
          'regularization_weight': 0.0,
          'target_variables': ['streamflow'],
          'predict_last_n': 7,
      })
      cfg_ep15_bg0 = AssimilationConfig({
          'seq_length': 60,
          'history': 1,
          'assimilation_window_length': 20,
          'assimilation_lead_time': 7,
          'learning_rate': 0.05,
          'epochs': 15,
          'loss': 'MSE',
          'optimizer': 'Adam',
          'assimilation_targets': ['hindcast_embedding'],
          'hindcast_embedding_regularization_weight': 0.0,
          'regularization_weight': 0.0,
          'target_variables': ['streamflow'],
          'predict_last_n': 7,
      })
      cfg_ep15_bg100 = AssimilationConfig({
          'seq_length': 60,
          'history': 1,
          'assimilation_window_length': 20,
          'assimilation_lead_time': 7,
          'learning_rate': 0.05,
          'epochs': 15,
          'loss': 'MSE',
          'optimizer': 'Adam',
          'assimilation_targets': ['hindcast_embedding'],
          'hindcast_embedding_regularization_weight': 100.0,
          'regularization_weight': 100.0,
          'target_variables': ['streamflow'],
          'predict_last_n': 7,
      })
      res_ep1 = Assimilation(cfg_ep1).assimilate(model, data, verbose=False)
      res_ep15_bg0 = Assimilation(cfg_ep15_bg0).assimilate(
          model, data, verbose=False
      )
      res_ep15_bg100 = Assimilation(cfg_ep15_bg100).assimilate(
          model, data, verbose=False
      )

      # 15 epochs must differ from 1 epoch under INFO logging
      diff_ep = torch.max(
          torch.abs(res_ep15_bg0['y_hat'] - res_ep1['y_hat'])
      ).item()
      self.assertGreater(diff_ep, 1e-5)

      # bg=0 and bg=100 must differ at 15 epochs under INFO logging
      diff_bg = torch.max(
          torch.abs(res_ep15_bg0['y_hat'] - res_ep15_bg100['y_hat'])
      ).item()
      self.assertGreater(diff_bg, 1e-5)
    finally:
      assim_mod.LOGGER.setLevel(prev_level)


if __name__ == '__main__':
  unittest.main()

