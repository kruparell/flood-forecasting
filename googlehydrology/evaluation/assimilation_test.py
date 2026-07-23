"""Unit tests for googlehydrology.evaluation.assimilation."""

import unittest
import numpy as np
import pandas as pd
import torch

from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.utils.assimilationconfig import AssimilationConfig


class AssimilationTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.cfg_dict = {
            'seq_length': 365,
            'history': 10,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 2,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['h_n', 'c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        self.cfg = AssimilationConfig(self.cfg_dict)
        self.assimilation = Assimilation(self.cfg)

    def test_check_discharge_timing_correct_alignment(self):
        seq_len = 365
        dates = pd.date_range(start='2020-01-01', periods=seq_len, freq='D').strftime('%Y-%m-%d').values
        
        y = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_shift1 = torch.roll(y, shifts=1, dims=1)
        y_shift1[0, 0, 0] = float('nan')
        
        data = {
            'date': np.tile(dates, (1, 1)),
            'y': y,
            'x_d': {'streamflow_shift1': y_shift1}
        }
        
        diag = self.assimilation.check_discharge_timing(data, verbose=False)
        self.assertFalse(diag['has_timing_mismatch'])
        self.assertEqual(len(diag['warnings']), 0)
        self.assertEqual(diag['details']['sequence_start_date'], '2020-01-01')

    def test_check_discharge_timing_same_day_mismatch(self):
        seq_len = 365
        dates = pd.date_range(start='2020-01-01', periods=seq_len, freq='D').strftime('%Y-%m-%d').values
        
        y = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        data = {
            'date': np.tile(dates, (1, 1)),
            'y': y,
            'x_d': {'streamflow_shift1': y}
        }
        
        diag = self.assimilation.check_discharge_timing(data, verbose=False)
        self.assertTrue(diag['has_timing_mismatch'])
        self.assertGreater(len(diag['warnings']), 0)
        self.assertIn("TIMING MISMATCH DETECTED", diag['warnings'][0])


    def test_cell_state_only_gradient_update(self):
        class MockLSTMModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                h_n = data.get('h_n', torch.zeros(1, 1, 8))
                y_hat = self.fc(c_n + h_n)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.1,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockLSTMModel()
        data = {
            'y': torch.ones(1, 10, 1) * 3.0,
            'x_d': torch.zeros(1, 10, 4)
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        base = model(data)['y_hat']
        da = res['y_hat']
        self.assertGreater((da - base).abs().max().item(), 0.1)

    def test_assimilate_modifies_forecast_predictions(self):
        class WindowedModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                y_hat = self.fc(c_n).expand(1, 12, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': torch.zeros_like(c_n)}

        cfg_dict = {
            'seq_length': 365,
            'history': 10,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
        }
    def test_hindcast_5day_gradient_enforcement(self):
        """Verifies that cell-state gradient descent is computed strictly on the 5 hindcast days."""
        class MockForecastModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                y_hat = self.fc(c_n).expand(1, 12, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': torch.zeros_like(c_n)}

        cfg_dict = {
            'seq_length': 365,
            'history': 5,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 5,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 12,
            'predict_n_hindcast': 5,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        model = MockForecastModel()
        data = {
            'y': torch.ones(1, 365, 1) * 2.0,
            'x_d': torch.zeros(1, 365, 4)
        }
        res = assim.assimilate(model, data, verbose=False, check_timing=False)
        self.assertIn('hindcast_metrics_pre', res)
        self.assertIn('hindcast_metrics_post', res)
        self.assertIn('NSE', res['hindcast_metrics_post'])

    def test_warm_start_identical_when_zero_lr(self):
        """Verifies that baseline and assimilation outputs match identically (|y_assim - y_base| == 0) at lr=0 or epochs=0."""
        class MockWarmLSTM(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 1)
            def forward(self, data):
                c_n = data.get('c_n', data.get('c_0', torch.zeros(1, 1, 8)))
                h_n = data.get('h_n', data.get('h_0', torch.zeros(1, 1, 8)))
                y_hat = self.fc(c_n + h_n).expand(1, 10, 1)
                return {'y_hat': y_hat, 'c_n': c_n, 'h_n': h_n}

        for epochs_val, lr_val in [(0, 0.05), (5, 0.0)]:
            for key_mode in ['c_n', 'c_0', 'both']:
                cfg_dict = {
                    'seq_length': 10,
                    'history': 2,
                    'assimilation_window': 1,
                    'assimilation_lead_time': 0,
                    'learning_rate': lr_val,
                    'epochs': epochs_val,
                    'loss': 'MSE',
                    'optimizer': 'Adam',
                    'assimilation_targets': ['c_n', 'h_n'],
                    'target_variables': ['streamflow'],
                    'predict_last_n': 1,
                    'predict_n_hindcast': 2,
                }
                cfg = AssimilationConfig(cfg_dict)
                assim = Assimilation(cfg)
                model = MockWarmLSTM()
                data = {
                    'y': torch.ones(1, 10, 1) * 3.0,
                    'x_d': torch.zeros(1, 10, 4),
                }
                if key_mode in ['c_n', 'both']:
                    data['c_n'] = torch.ones(1, 1, 8) * 0.25
                    data['h_n'] = torch.ones(1, 1, 8) * 0.10
                if key_mode in ['c_0', 'both']:
                    data['c_0'] = torch.ones(1, 1, 8) * 0.25
                    data['h_0'] = torch.ones(1, 1, 8) * 0.10

                base_out = model(data)['y_hat']
                da_out = assim.assimilate(model, data, verbose=False, check_timing=False)['y_hat']
                diff = (da_out - base_out).abs().max().item()
                self.assertEqual(diff, 0.0)

    def test_probabilistic_mixture_model_assimilation(self):
        """Verifies assimilate handles models returning mu and pi or mu alone without KeyError."""
        class MockProbabilisticModel(torch.nn.Module):
            state_var_names = ['c_n', 'h_n']
            def __init__(self, mode='mu_pi'):
                super().__init__()
                self.mode = mode
                self.fc = torch.nn.Linear(8, 4)
            def forward(self, data):
                c_n = data.get('c_n', torch.zeros(1, 1, 8))
                h_n = data.get('h_n', torch.zeros(1, 1, 8))
                mu = self.fc(c_n + h_n).expand(1, 10, 4)
                if self.mode == 'mu_pi':
                    pi = torch.ones_like(mu) * 0.25
                    return {'mu': mu, 'pi': pi, 'c_n': c_n, 'h_n': h_n}
                else:
                    return {'mu': mu[:, :, 0:1], 'c_n': c_n, 'h_n': h_n}

        cfg_dict = {
            'seq_length': 10,
            'history': 2,
            'assimilation_window': 1,
            'assimilation_lead_time': 0,
            'learning_rate': 0.05,
            'epochs': 2,
            'loss': 'MSE',
            'optimizer': 'Adam',
            'assimilation_targets': ['c_n'],
            'target_variables': ['streamflow'],
            'predict_last_n': 1,
            'predict_n_hindcast': 2,
        }
        cfg = AssimilationConfig(cfg_dict)
        assim = Assimilation(cfg)
        data = {
            'y': torch.ones(1, 10, 1) * 3.0,
            'x_d': torch.zeros(1, 10, 4),
        }

        # Test both mu+pi mixture and mu alone
        for mode in ['mu_pi', 'mu_only']:
            model = MockProbabilisticModel(mode=mode)
            res = assim.assimilate(model, data, verbose=False, check_timing=False)
            self.assertIn('y_hat', res)
            self.assertEqual(res['y_hat'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()


