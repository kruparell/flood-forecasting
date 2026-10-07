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
    from googlehydrology.evaluation.eval_utils import apply_ar1_postprocessing
    q_base = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    q_obs_window = np.array([5.0, 8.0, 12.0], dtype=np.float32)
    q_ar1 = apply_ar1_postprocessing(q_base, q_obs_window, rho=0.5)
    self.assertAlmostEqual(q_ar1[0], 11.0, places=3)
    self.assertAlmostEqual(q_ar1[1], 10.5, places=3)

    q_obs_low = np.array([0.0, 0.0, -100.0], dtype=np.float32)
    q_ar1_neg = apply_ar1_postprocessing(q_base, q_obs_low, rho=0.85)
    self.assertTrue((q_ar1_neg >= 0.0).all())

  def _create_mef_model_and_data(
      self,
      hidden_size=16,
      seq_length=14,
      lead_time=2,
      head='regression',
      n_distributions=None,
      production_layout=False,
  ):
    """Builds a MeanEmbeddingForecastLSTM and a matching input batch.

    Args:
      hidden_size: LSTM hidden width.
      seq_length: Length of the target series `y`.
      lead_time: Forecast lead.
      head: Model head ('regression' or 'cmal').
      n_distributions: Mixture count, for the CMAL head.
      production_layout: Selects the tensor geometry. The default (False) is
        the historical test layout, hindcast = seq_length - lead_time and
        forecast = seq_length = len(y), so predictions come back exactly as
        long as the targets. Real batches from `multimet` are NOT shaped that
        way: hindcast = seq_length and forecast = seq_length + lead_time, so
        `y_hat` is LONGER than `y` (365 vs 372 in the Stage 43 sweep). Any
        logic that equates the prediction timeline with `len(y)` passes under
        the default layout and fails in production, so set this to True for
        tests that touch timeline arithmetic.

    Returns:
      A (model, data) pair.
    """
    from googlehydrology.utils.config import Config
    from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM

    cfg_dict = {
        'model': 'mean_embedding_forecast_lstm',
        'head': head,
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
    if n_distributions is not None:
      cfg_dict['n_distributions'] = n_distributions
    cfg = Config(cfg_dict, dev_mode=True)
    model = MeanEmbeddingForecastLSTM(cfg=cfg)
    if production_layout:
      hindcast_length = seq_length
      forecast_length = seq_length + lead_time
    else:
      hindcast_length = seq_length - lead_time
      forecast_length = seq_length
    data = {
        'x_s': torch.ones(1, 1),
        'x_d_hindcast': {
            'era5_precip': torch.randn(1, hindcast_length, 1)
        },
        'x_d_forecast': {'hres_precip': torch.randn(1, forecast_length, 1)},
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

    `c_0_hindcast` / `c_0_forecast` are LSTM *initial* states, so they appear in
    neither the model output dict nor the input batch. The engine resolves their
    baseline to an explicit zero tensor, matching the model's implicit default.
    Hidden states are deliberately excluded: `c_both` optimizes cell states only.
    """
    torch.manual_seed(2)
    hidden_size = 8
    model, data = self._create_mef_model_and_data(
        hidden_size=hidden_size, seq_length=365, lead_time=7
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
        'assimilation_targets': ['c_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)

    self.assertIn('y_hat', res)
    self.assertTrue(torch.isfinite(res['y_hat']).all())
    for name in ('c_0_hindcast', 'c_0_forecast'):
      self.assertIn(name, res, f'{name} was not assimilated')
      self.assertEqual(res[name].shape[-1], hidden_size)
      self.assertTrue(torch.isfinite(res[name]).all())
      # A zero baseline that stayed exactly zero would mean the optimizer never
      # touched this component.
      self.assertGreater(float(res[name].abs().sum()), 0.0)
    # Hidden states are out of scope for `c_both` and must not be optimized.
    for name in ('h_0_hindcast', 'h_0_forecast'):
      self.assertNotIn(name, res)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cell_state_zero_perturbation_reproduces_baseline(self, mock_scaler):
    """Verifies the cell-state baseline is the model's own implicit default.

    The baseline for an initial state must be all zeros, which is what PyTorch
    uses when the key is absent. Seeding it with the model's *terminal* state
    (`c_n_*`) instead would make a zero-perturbation assimilated pass differ
    from the unassimilated one, confounding the measured effect of DA with the
    effect of the seeding. This pins the zero-perturbation pass to be a no-op.
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

  def _window_start_cfg(self, window, lr, epochs, anchor, seq_length, lead_time):
    """Builds a `c_both` config pinned to a given `state_anchor`."""
    return AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': lr,
        'epochs': epochs,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'state_anchor': anchor,
        'assimilation_targets': ['c_both'],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
    })

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_zero_lr_is_exact_identity(self, mock_scaler):
    """A zero-LR `window_start` pass must reproduce the open-loop model exactly.

    Under `state_anchor: window_start` the engine runs an unassimilated pass,
    extracts the warmed-up recurrent state entering the window, optimizes a
    perturbation on top of it, and splices the unassimilated prefix back onto
    the assimilated tail. Every one of those steps is a chance to corrupt the
    output. With `learning_rate=0` the perturbation stays at zero, so any
    deviation from the open-loop prediction is a splice bug rather than an
    effect of assimilation -- which would silently bias every measured delta in
    a sweep.
    """
    torch.manual_seed(7)
    seq_length, lead_time = 60, 7
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'].clone()

    da_cfg = self._window_start_cfg(
        window=14, lr=0.0, epochs=5, anchor='window_start',
        seq_length=seq_length, lead_time=lead_time,
    )
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)

    # The in-window tail is recomputed by a *differently shaped* forward pass
    # (the sliced window rather than the full sequence), so reductions happen
    # in a different order and bitwise equality is not attainable. One float32
    # ULP is the right bar: it is still ~6 orders of magnitude tighter than the
    # prefix-corruption bug this test was written to catch.
    torch.testing.assert_close(res['y_hat'], open_loop, rtol=0.0, atol=1e-6)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_leaves_prefix_untouched_and_moves_tail(
      self, mock_scaler
  ):
    """`window_start` must only alter timesteps at/after the window start.

    assimilation_end_step = seq_length - lead_time, and
    assimilation_start_step = end - history*window. Everything before that
    index is spliced verbatim from the unassimilated pass; everything from it
    onward is recomputed. Pinning both halves catches an off-by-one in the
    splice boundary, which would otherwise show up only as a small unexplained
    bias in the forecast metrics.
    """
    torch.manual_seed(11)
    seq_length, lead_time, window = 60, 7, 14
    start = (seq_length - lead_time) - window  # 39

    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'].clone()

    da_cfg = self._window_start_cfg(
        window=window, lr=0.1, epochs=5, anchor='window_start',
        seq_length=seq_length, lead_time=lead_time,
    )
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    y_hat = res['y_hat']

    self.assertEqual(y_hat.shape, open_loop.shape)
    # Prefix is spliced verbatim from the unassimilated pass.
    torch.testing.assert_close(
        y_hat[:, :start], open_loop[:, :start], rtol=0.0, atol=0.0
    )
    # The tail must actually have moved, or the optimizer never reached c_0.
    tail_shift = float((y_hat[:, start:] - open_loop[:, start:]).abs().max())
    self.assertGreater(
        tail_shift, 0.0,
        'window_start assimilation left the in-window tail unchanged; the '
        'gradient is not reaching the initial cell states.',
    )

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_and_sequence_start_are_not_equivalent(
      self, mock_scaler
  ):
    """The two anchors must produce materially different assimilated output.

    `sequence_start` injects `c_0` at t=0, so for a long sequence the LSTM
    forget gates attenuate the perturbation before the assimilation window is
    reached. If the two anchors agree, `state_anchor` is inert and every
    cell-state result is really a `sequence_start` result.
    """
    torch.manual_seed(13)
    seq_length, lead_time, window = 60, 7, 14
    kwargs = dict(
        window=window, lr=0.1, epochs=5,
        seq_length=seq_length, lead_time=lead_time,
    )

    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time
    )
    model.eval()

    res_win = Assimilation(
        self._window_start_cfg(anchor='window_start', **kwargs)
    ).assimilate(model, data, verbose=False)
    res_seq = Assimilation(
        self._window_start_cfg(anchor='sequence_start', **kwargs)
    ).assimilate(model, data, verbose=False)

    divergence = float((res_win['y_hat'] - res_seq['y_hat']).abs().max())
    self.assertGreater(
        divergence, 0.0,
        'window_start and sequence_start produced identical output; the '
        'state_anchor setting is inert.',
    )

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

  def test_assimilation_config_exposes_n_distributions(self):
    """`loss: CMAL` needs n_distributions, which must be a recognized key."""
    self.assertIn('n_distributions', AssimilationConfig.get_allowed_keys())

    default_cfg = AssimilationConfig({
        'seq_length': 10,
        'learning_rate': 0.0,
        'epochs': 1,
        'loss': 'CMAL',
        'optimizer': 'Adam',
        'assimilation_targets': ['static_embedding'],
        'target_variables': ['streamflow'],
    })
    self.assertEqual(default_cfg.n_distributions, 3)

    override_cfg = AssimilationConfig({
        'seq_length': 10,
        'learning_rate': 0.0,
        'epochs': 1,
        'loss': 'CMAL',
        'optimizer': 'Adam',
        'assimilation_targets': ['static_embedding'],
        'target_variables': ['streamflow'],
        'n_distributions': 5,
    })
    self.assertEqual(override_cfg.n_distributions, 5)

  def _cmal_da_cfg(self, **overrides):
    cfg = {
        'seq_length': 60,
        'history': 1,
        'assimilation_window_length': 20,
        'assimilation_lead_time': 7,
        'learning_rate': 0.01,
        'epochs': 2,
        'loss': 'CMAL',
        'optimizer': 'Adam',
        'assimilation_targets': ['static_embedding'],
        'target_variables': ['streamflow'],
        'predict_last_n': 7,
    }
    cfg.update(overrides)
    return AssimilationConfig(cfg)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cmal_loss_assimilation_runs_end_to_end(self, mock_scaler):
    """DA with the CMAL NLL objective must run and emit finite mixture params."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=60, lead_time=7, head='cmal',
        n_distributions=3,
    )
    res = Assimilation(self._cmal_da_cfg()).assimilate(
        model, data, verbose=False
    )

    self.assertIn('y_hat', res)
    self.assertTrue(torch.isfinite(res['y_hat']).all())
    for key in ('mu', 'b', 'tau', 'pi'):
      self.assertIn(key, res)
      self.assertTrue(
          torch.isfinite(res[key]).all(), f'{key} contains non-finite values'
      )
    # The scale parameter must not collapse onto its softplus floor, which is
    # the MLE degeneracy that makes NLL-based adaptation unsafe without
    # background regularization.
    self.assertGreater(float(res['b'].min()), 1e-4)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cmal_loss_rejects_non_mixture_head(self, mock_scaler):
    """A regression head emits no mu/b/tau/pi, so CMAL must fail fast."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=60, lead_time=7, head='regression'
    )
    with self.assertRaisesRegex(ValueError, 'mixture head'):
      Assimilation(self._cmal_da_cfg()).assimilate(model, data, verbose=False)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_cmal_loss_rejects_n_distributions_mismatch(self, mock_scaler):
    """A silently wrong slice width would produce a meaningless loss."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=60, lead_time=7, head='cmal',
        n_distributions=3,
    )
    cfg = self._cmal_da_cfg(n_distributions=5)
    with self.assertRaisesRegex(ValueError, 'n_distributions mismatch'):
      Assimilation(cfg).assimilate(model, data, verbose=False)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_non_cmal_loss_skips_compatibility_check(self, mock_scaler):
    """The guard must not fire for MSE, regardless of n_distributions."""
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=60, lead_time=7, head='regression'
    )
    cfg = self._cmal_da_cfg(loss='MSE', n_distributions=99)
    res = Assimilation(cfg).assimilate(model, data, verbose=False)
    self.assertTrue(torch.isfinite(res['y_hat']).all())

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



  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_slice_still_slices_a_parallel_forecast(self, mock_scaler):
    """The parallel (forecast-longer-than-hindcast) layout must be unchanged.

    Guards the appended-forecast fix against over-reach: when the forecast is a
    full-length parallel array extended by `lead_delta`, global coordinates DO
    apply to it and it must still be sliced.
    """
    from googlehydrology.modelzoo.mean_embedding_forecast_lstm import ForwardData

    seq_length, lead_time, window = 60, 7, 14
    hc_len = seq_length - lead_time  # 53
    start = (seq_length - lead_time) - window  # 39

    model, _ = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time
    )
    data = {
        'x_s': torch.ones(1, 1),
        'x_d_hindcast': {'era5_precip': torch.randn(1, hc_len, 1)},
        'x_d_forecast': {'hres_precip': torch.randn(1, seq_length, 1)},
    }

    fd = ForwardData.from_forward_data(
        data, model.config_data, window_slice=(start, seq_length)
    )
    hc = next(iter(fd.hindcast_features.values()))
    fc = next(iter(fd.forecast_features.values()))

    # Hindcast clamps at its own end: [39, 53) -> 14.
    self.assertEqual(hc.shape[1], hc_len - start)
    # Forecast is extended by lead_delta = 60 - 53 = 7, so [39, 60) -> 21.
    self.assertEqual(fc.shape[1], seq_length - start)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_slice_rejects_an_empty_slice(self, mock_scaler):
    """An empty slice must fail loudly at its origin, not silently propagate.

    The original failure mode gave only a broadcast error from deep inside the
    model, with no mention of `window_slice` or of which array was empty.
    """
    from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
        _concat_tensors_from_dict,
    )

    data = {'era5_precip': torch.randn(1, 10, 1)}
    with self.assertRaisesRegex(ValueError, 'EMPTY slice'):
      _concat_tensors_from_dict(
          data, keys=['era5_precip'], slice_bounds=(50, 60)
      )


  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_runs_under_production_geometry(self, mock_scaler):
    """`window_start` DA must work when `y_hat` is longer than `y`.

    This is the shape mismatch that killed the Stage 43 sweep on Borg. Real
    batches give hindcast = len(y) and forecast = len(y) + lead_time, so the
    model emits predictions on the FORECAST timeline (372 steps) while the
    targets live on a shorter one (365). The prefix splice originally keyed off
    `total_sequence_length`, i.e. `len(y)`, so under this geometry it silently
    declined to splice: `y_hat` stayed at window length, the loss window
    `predicted[:, loss_start:end]` came out EMPTY, and the run died with
    `size of tensor a (7) must match the size of tensor b (0)`.

    Every pre-existing fixture used the parallel layout where len(y_hat) ==
    len(y), which is exactly why this passed locally and failed remotely.
    """
    torch.manual_seed(17)
    seq_length, lead_time, window = 60, 7, 7
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time,
        production_layout=True,
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'].clone()

    # Precondition: the fixture really does reproduce the mismatch. Without
    # this the test could silently degrade into a duplicate of the parallel
    # layout coverage above.
    self.assertGreater(
        open_loop.shape[1], data['y'].shape[1],
        'fixture failed to reproduce the production geometry; the whole '
        'point of this test is len(y_hat) != len(y)',
    )

    da_cfg = self._window_start_cfg(
        window=window, lr=0.1, epochs=3, anchor='window_start',
        seq_length=seq_length, lead_time=lead_time,
    )
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)

    # The result must come back on the PRIOR (forecast) timeline, not the
    # window length and not the target length.
    self.assertEqual(res['y_hat'].shape, open_loop.shape)
    self.assertTrue(torch.isfinite(res['y_hat']).all())

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_window_start_zero_lr_identity_under_production_geometry(
      self, mock_scaler
  ):
    """Zero-LR identity must also hold on the production geometry.

    `test_window_start_zero_lr_is_exact_identity` pins the same invariant on
    the parallel layout. Running it again here is not redundant: the splice
    boundary is computed from different lengths in the two layouts, so an
    off-by-`lead_delta` error would leave one of them intact.
    """
    torch.manual_seed(19)
    seq_length, lead_time, window = 60, 7, 7
    model, data = self._create_mef_model_and_data(
        hidden_size=8, seq_length=seq_length, lead_time=lead_time,
        production_layout=True,
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'].clone()

    da_cfg = self._window_start_cfg(
        window=window, lr=0.0, epochs=3, anchor='window_start',
        seq_length=seq_length, lead_time=lead_time,
    )
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)

    torch.testing.assert_close(res['y_hat'], open_loop, rtol=0.0, atol=1e-6)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_pred_and_target_loss_window_exact_date_alignment(
      self, mock_scaler
  ):
    """Under production layout (len(y)=60, len(y_hat)=67), loss must align y_hat[:, 53:60] with y[:, 46:53].

    In Multimet production batches, `y` has length `seq_length` (60, covering
    [t0 - 52 .. t0 + 7]) so `lead0` (t0) is at index 52 (-8), while `y_hat` has
    length `seq_length + lead_time` (67, covering [t0 - 59 .. t0 + 7]) so
    `lead0` (t0) is at index 59 (-8). Optimizing over a 7-day window must fit
    `y_hat[:, 53:60]` (dates [t0 - 6 .. t0]) to `y[:, 46:53]` (dates
    [t0 - 6 .. t0]), NOT `y_hat[:, 46:53]` (dates [t0 - 13 .. t0 - 7]).
    """
    torch.manual_seed(23)
    seq_length, lead_time, window = 60, 7, 7
    model, data = self._create_mef_model_and_data(
        hidden_size=8,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    # Set contrasting values in `y` at the true window [46:53] (dates t0-6..t0)
    # versus the 7-day-earlier slice [39:46] (dates t0-13..t0-7).
    data['y'] = torch.zeros(1, seq_length, 1)
    data['y'][:, 39:46, :] = -3.0
    data['y'][:, 46:53, :] = 2.5

    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'][..., :1].clone()

    da_cfg = AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': 0.15,
        'epochs': 45,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'regularization': [],
        'regularization_weight': 0.0,
        'assimilation_targets': [
            'static_embedding',
            'hindcast_embedding',
            'forecast_embedding',
        ],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
        'assimilation_components': {
            'static_embedding': {'enabled': True, 'lr': 0.15, 'weight': 0.0},
            'hindcast_embedding': {'enabled': True, 'lr': 0.15, 'weight': 0.0},
            'forecast_embedding': {'enabled': True, 'lr': 0.15, 'weight': 0.0},
        },
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    # True in-window target is y[:, 46:53] (+2.5) and true in-window prediction
    # is y_hat[:, 53:60] (which equals y_hat[:, -14:-7], ending at lead0 = -8).
    target_window = data['y'][:, 46:53, :]
    open_err = torch.mean((open_loop[:, 53:60, :] - target_window) ** 2).item()
    da_err = torch.mean((res_y_hat[:, 53:60, :] - target_window) ** 2).item()
    self.assertLess(
        da_err,
        open_err * 0.65,
        f'Assimilated in-window MSE ({da_err:.4f}) failed to improve over '
        f'open-loop ({open_err:.4f}) on the aligned [t0-6..t0] window.',
    )
    # Predictions in [53:60] must move UP toward y[:, 46:53] (+2.5), NOT down
    # toward the 7-day-shifted slice y[:, 39:46] (-3.0).
    self.assertGreater(
        res_y_hat[:, 53:60, :].mean().item(),
        open_loop[:, 53:60, :].mean().item(),
    )
    # Lead-0 (index -8 in both tensors: 59 in y_hat, 52 in y) must move toward +2.5.
    open_lead0_err = torch.abs(open_loop[:, -8, :] - data['y'][:, -8, :]).item()
    da_lead0_err = torch.abs(res_y_hat[:, -8, :] - data['y'][:, -8, :]).item()
    self.assertLess(da_lead0_err, open_lead0_err)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_dynamic_embedding_optimizes_through_issue_date_t0(
      self, mock_scaler
  ):
    """Dynamic embeddings must be optimized over [pred_start_step:pred_end_step] (50:60), not frozen at 53:60."""
    torch.manual_seed(29)
    seq_length, lead_time, window = 60, 7, 10
    model, data = self._create_mef_model_and_data(
        hidden_size=8,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    data['y'] = torch.full((1, seq_length, 1), 1.75)

    model.eval()
    with torch.no_grad():
      prior_out = model(data)
      open_loop = prior_out['y_hat'][..., :1].clone()
      prior_hc_emb = prior_out['hindcast_embedding'].clone()

    da_cfg = AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': 0.05,
        'epochs': 15,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'regularization': ['background_embedding'],
        'regularization_weight': 1.0,
        'assimilation_targets': ['hindcast_embedding'],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
        'assimilation_components': {
            'hindcast_embedding': {'enabled': True, 'lr': 0.05, 'weight': 0.001},
        },
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    # Pre-window predictions [0:50] must remain completely identical to open_loop
    torch.testing.assert_close(
        res_y_hat[:, :50, :], open_loop[:, :50, :], rtol=0.0, atol=1e-6
    )
    # Predictions inside [53:60] (the 7 days [t0-6..t0] that were previously
    # frozen outside [:53]) must be actively updated by the optimized embedding.
    diff_last_7_in_window = torch.abs(
        res_y_hat[:, 53:60, :] - open_loop[:, 53:60, :]
    ).max().item()
    self.assertGreater(
        diff_last_7_in_window,
        1e-3,
        'Days [t0-6..t0] (indices 53..59 of y_hat) were not updated!',
    )
    # And the optimized window returned in res['hindcast_embedding'] must differ
    # from the baseline window `prior_hc_emb[:, 50:60, :]` at the last step (t0 = index 59).
    self.assertEqual(res['hindcast_embedding'].shape[1], window)
    delta_at_t0 = torch.norm(
        res['hindcast_embedding'][:, -1, :] - prior_hc_emb[:, 59, :]
    ).item()
    self.assertGreater(delta_at_t0, 1e-4)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_window_start_cell_state_anchor_and_slice_alignment(
      self, mock_scaler
  ):
    """Under production layout, window_start must anchor at pred_start_step (50) and reduce lead0 error."""
    torch.manual_seed(31)
    seq_length, lead_time, window = 60, 7, 10
    model, data = self._create_mef_model_and_data(
        hidden_size=8,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    data['y'] = torch.full((1, seq_length, 1), 1.5)

    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'][..., :1].clone()

    da_cfg = self._window_start_cfg(
        window=window,
        lr=0.08,
        epochs=20,
        anchor='window_start',
        seq_length=seq_length,
        lead_time=lead_time,
    )
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    # Prefix [0:50] (pred_start_step = 67 - 7 - 10 = 50) must match open_loop exactly
    torch.testing.assert_close(
        res_y_hat[:, :50, :], open_loop[:, :50, :], rtol=0.0, atol=1e-6
    )
    # In-window MSE on [50:60] against y[:, 43:53] must strictly decrease
    open_mse = torch.mean(
        (open_loop[:, 50:60, :] - data['y'][:, 43:53, :]) ** 2
    ).item()
    da_mse = torch.mean(
        (res_y_hat[:, 50:60, :] - data['y'][:, 43:53, :]) ** 2
    ).item()
    self.assertLess(da_mse, open_mse)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_in_window_nse_and_kge_gain_exceeds_0_1(
      self, mock_scaler
  ):
    """Assimilating under production layout (len(y_hat)=len(y)+lead_time) must yield >=0.1 NSE and KGE gain in-window."""
    torch.manual_seed(42)
    seq_length, lead_time, window = 60, 7, 14
    model, data = self._create_mef_model_and_data(
        hidden_size=12,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'][..., :1].clone()

    # Construct a realistic hydrograph in the assimilation window `y[:, 39:53]`
    # (which aligns with `y_hat[:, 46:60]`, ending at `lead0` = index -8) where
    # the unassimilated open_loop has a systematic bias and amplitude error.
    t_axis = torch.linspace(0.0, 3.14159, window).view(1, window, 1)
    true_hydrograph = open_loop[:, 46:60, :].detach() + 0.35 * torch.sin(t_axis) + 0.2
    data['y'][:, 39:53, :] = true_hydrograph

    da_cfg = AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': 0.12,
        'epochs': 50,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'regularization': ['background_embedding'],
        'regularization_weight': 1.0,
        'assimilation_targets': [
            'static_embedding',
            'hindcast_embedding',
            'forecast_embedding',
        ],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
        'assimilation_components': {
            'static_embedding': {'enabled': True, 'lr': 0.1, 'weight': 1e-5},
            'hindcast_embedding': {'enabled': True, 'lr': 0.12, 'weight': 1e-5},
            'forecast_embedding': {'enabled': True, 'lr': 0.12, 'weight': 1e-5},
        },
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    def _compute_nse_kge(sim_1d: torch.Tensor, obs_1d: torch.Tensor):
      obs_mean = obs_1d.mean()
      sim_mean = sim_1d.mean()
      nse = 1.0 - torch.sum((sim_1d - obs_1d) ** 2) / torch.sum(
          (obs_1d - obs_mean) ** 2
      )
      r_num = torch.sum((sim_1d - sim_mean) * (obs_1d - obs_mean))
      r_den = torch.sqrt(
          torch.sum((sim_1d - sim_mean) ** 2)
          * torch.sum((obs_1d - obs_mean) ** 2)
      )
      r = r_num / (r_den + 1e-8)
      alpha = sim_1d.std() / (obs_1d.std() + 1e-8)
      beta = sim_mean / (obs_mean + 1e-8)
      kge = 1.0 - torch.sqrt(
          (r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
      )
      return float(nse.item()), float(kge.item())

    obs_win = data['y'][0, 39:53, 0]
    open_win = open_loop[0, 46:60, 0]
    da_win = res_y_hat[0, 46:60, 0]

    open_nse, open_kge = _compute_nse_kge(open_win, obs_win)
    da_nse, da_kge = _compute_nse_kge(da_win, obs_win)

    self.assertGreaterEqual(
        da_nse - open_nse,
        0.1,
        f'Expected in-window NSE gain >= 0.1, got {da_nse - open_nse:.4f} '
        f'(open={open_nse:.4f}, da={da_nse:.4f})',
    )
    self.assertGreaterEqual(
        da_kge - open_kge,
        0.1,
        f'Expected in-window KGE gain >= 0.1, got {da_kge - open_kge:.4f} '
        f'(open={open_kge:.4f}, da={da_kge:.4f})',
    )

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_early_stopping_compares_aligned_t0(
      self, mock_scaler
  ):
    """early_stopping_tolerance must compare y_hat[:, 59] (t0) against y[:, 52] (t0), not y_hat[:, 52] (t0-7)."""
    torch.manual_seed(53)
    seq_length, lead_time, window = 60, 7, 7
    model, data = self._create_mef_model_and_data(
        hidden_size=8,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'][..., :1].clone()

    # Set y[:, 52] (t0 in target coordinates) FAR from open_loop[:, 59] (t0 in
    # model coordinates), while setting y[:, 52] EQUAL to open_loop[:, 52]
    # (which is t0-7 in model coordinates!).
    # Under the old bug, `pred_t0 = predicted_streamflow[:, 52]` equaled
    # `obs_t0 = y[:, 52]`, causing an immediate false-positive early stop at
    # epoch 0 without performing any gradient step!
    data['y'] = torch.zeros(1, seq_length, 1)
    data['y'][:, 46:53, :] = open_loop[:, 52:53, :].detach() + 3.0
    data['y'][:, 52:53, :] = open_loop[:, 59:60, :].detach() + 3.0

    da_cfg = AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': 0.08,
        'epochs': 10,
        'early_stopping_tolerance': 0.01,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'regularization': [],
        'regularization_weight': 0.0,
        'assimilation_targets': ['hindcast_embedding'],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
        'assimilation_components': {
            'hindcast_embedding': {'enabled': True, 'lr': 0.08, 'weight': 0.0},
        },
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    # Because open_loop[:, 59] != data['y'][:, 52], it must NOT stop at epoch 0
    # and must update y_hat[:, 59] toward data['y'][:, 52].
    open_t0_err = torch.abs(open_loop[:, 59, :] - data['y'][:, 52, :]).item()
    da_t0_err = torch.abs(res_y_hat[:, 59, :] - data['y'][:, 52, :]).item()
    self.assertLess(da_t0_err, open_t0_err)

  @patch('googlehydrology.modelzoo.basemodel.Scaler')
  def test_production_layout_joint_window_start_and_dynamic_embeddings(
      self, mock_scaler
  ):
    """Joint window_start (c_0_*) + dynamic embeddings under production layout must slice embeddings to W+lead_time and reduce MSE."""
    torch.manual_seed(61)
    seq_length, lead_time, window = 60, 7, 10
    model, data = self._create_mef_model_and_data(
        hidden_size=8,
        seq_length=seq_length,
        lead_time=lead_time,
        production_layout=True,
    )
    data['y'] = torch.full((1, seq_length, 1), 1.6)

    model.eval()
    with torch.no_grad():
      open_loop = model(data)['y_hat'][..., :1].clone()

    da_cfg = AssimilationConfig({
        'seq_length': seq_length,
        'history': 1,
        'assimilation_window_length': window,
        'assimilation_lead_time': lead_time,
        'learning_rate': 0.06,
        'epochs': 15,
        'loss': 'MSE',
        'optimizer': 'Adam',
        'regularization': ['background_embedding'],
        'regularization_weight': 1.0,
        'assimilation_targets': [
            'c_0_hindcast',
            'c_0_forecast',
            'hindcast_embedding',
            'forecast_embedding',
        ],
        'target_variables': ['streamflow'],
        'predict_last_n': lead_time,
        'assimilation_components': {
            'c_0_hindcast': {
                'enabled': True,
                'lr': 0.06,
                'weight': 1e-3,
                'state_anchor': 'window_start',
            },
            'c_0_forecast': {
                'enabled': True,
                'lr': 0.06,
                'weight': 1e-3,
                'state_anchor': 'window_start',
            },
            'hindcast_embedding': {'enabled': True, 'lr': 0.06, 'weight': 1e-3},
            'forecast_embedding': {'enabled': True, 'lr': 0.06, 'weight': 1e-3},
        },
    })
    res = Assimilation(da_cfg).assimilate(model, data, verbose=False)
    res_y_hat = res['y_hat'][..., :1]

    self.assertEqual(res_y_hat.shape[1], seq_length + lead_time)
    torch.testing.assert_close(
        res_y_hat[:, :50, :], open_loop[:, :50, :], rtol=0.0, atol=1e-6
    )
    open_mse = torch.mean(
        (open_loop[:, 50:60, :] - data['y'][:, 43:53, :]) ** 2
    ).item()
    da_mse = torch.mean(
        (res_y_hat[:, 50:60, :] - data['y'][:, 43:53, :]) ** 2
    ).item()
    self.assertLess(da_mse, open_mse)


if __name__ == '__main__':
  unittest.main()


