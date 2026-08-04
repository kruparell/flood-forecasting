import math
import logging
import re
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.modelzoo.mean_embedding_forecast_lstm import MeanEmbeddingForecastLSTM, ForwardData
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head, calc_cmal_mean
from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def _get_cfg_val(cfg: Config, key: str, default: Any = None) -> Any:
    if hasattr(cfg, '_cfg') and isinstance(cfg._cfg, dict) and key in cfg._cfg:
        return cfg._cfg[key]
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


class ARLSTM(MeanEmbeddingForecastLSTM):
    """An Autoregressive LSTM using MeanEmbeddingForecastLSTM as its base architecture.

    This model handles streamflow observations and autoregressive feedback.
    If observations are missing (NaN), the model's previous predictions are substituted.
    The model feeds a binary flag indicating whether the autoregressive feature at each
    timestep is from an observation (0.0) vs. simulated substitution (1.0).

    When predicting parameters of a CMAL (or mixture) distribution, y_hat is computed as the
    expected mean (sum(pi * mu)) to use for autoregressive feedback when discharge is missing.
    """

    def __init__(self, cfg: Config):
        super(ARLSTM, self).__init__(cfg=cfg)

        self._ar_shift = self._get_ar_shift()
        self._num_ar_inputs = len(self.cfg.autoregressive_inputs)
        self._ar_target_indices = [
            self.cfg.target_variables.index(re.compile(r'^(.*)_shift(\d+)$').search(inp)[1])
            for inp in self.cfg.autoregressive_inputs
        ]

        self.verbose = bool(_get_cfg_val(cfg, 'verbose', True))

        # Calculate cell input dimension: static_embedding + dynamic_embedding + x_ar + ar_flags
        base_input_size = self.static_embedding_fc.output_size + self.config_data.hindcast_embedding.hiddens[-1]
        cell_input_dim = base_input_size + 2 * self._num_ar_inputs

        self.hindcast_cell = nn.LSTM(input_size=cell_input_dim, hidden_size=self.config_data.hidden_size)
        self.forecast_cell = nn.LSTM(input_size=cell_input_dim, hidden_size=self.config_data.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.config_data.hidden_size, n_out=self.output_size)
        self._reset_parameters()

        if self.verbose:
            print(f"[ARLSTM Init] Base=MeanEmbeddingForecastLSTM, Cell Input Dim={cell_input_dim}, Output Size={self.output_size}", flush=True)

    def _get_ar_shift(self) -> int:
        shifts = set()
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Autoregressive inputs must be a shifted variable with form <variable>_shift<lag> ',
                                f'where <lag> is an integer. Instead got: {input}.')
            if not capture[1] in self.cfg.target_variables:
                raise ValueError('Autoregressive inputs must be a shifted target variable. ',
                                f'Instead got a shifted version of: {capture[1]}.')
            shifts.add(int(capture[2]))
        if len(shifts) > 1:
            raise ValueError('Only one AR shift is allowed currently. All autoregressive inputs must use the same shift.')
        shift = shifts.pop()
        if shift <= 0:
            raise ValueError('Autoregressive inputs must be shifted by at least one timestep.')
        return shift

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_cell.bias_hh_l0.data[self.config_data.hidden_size:2 * self.config_data.hidden_size] = self.cfg.initial_forget_bias
            self.forecast_cell.bias_hh_l0.data[self.config_data.hidden_size:2 * self.config_data.hidden_size] = self.cfg.initial_forget_bias

    def forward(self,
                data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Autoregressive LSTM model."""
        forward_data = ForwardData.from_forward_data(data, self.config_data)
        static_embedding = self._calc_static_embedding(forward_data)

        # Retrieve dynamic features and sequence length
        sample_dynamic = list(forward_data.hindcast_features.values())[0] if forward_data.hindcast_features else list(forward_data.forecast_features.values())[0]
        batch_size, seq_len, _ = sample_dynamic.shape

        # Compute dynamic embeddings across feature groups
        hindcast_embeddings = [
            self._calc_dynamic_embedding(fc, forward_data.hindcast_features[name], static_embedding, seq_len)
            for name, fc in self.hindcast_embeddings_fc.items()
        ]
        shared_embeddings = [
            self._calc_dynamic_embedding(fc, forward_data.hindcast_features[name], static_embedding, seq_len)
            for name, fc in self.shared_embeddings_fc.items()
        ]

        all_emb = hindcast_embeddings + shared_embeddings
        if len(all_emb) > 1:
            stacked_emb = torch.stack(all_emb, dim=-1)
            dynamic_emb = torch.nanmean(stacked_emb, dim=-1)
        elif len(all_emb) == 1:
            dynamic_emb = all_emb[0]
        else:
            dynamic_emb = static_embedding.new_zeros((batch_size, seq_len, self.config_data.hindcast_embedding.hiddens[-1]))

        # Extract raw AR discharge input tensor from data (handling tensor or dict types)
        ar_tensors = []
        for ar_name in self.cfg.autoregressive_inputs:
            if 'x_d' in data and isinstance(data['x_d'], dict) and ar_name in data['x_d']:
                tensor = data['x_d'][ar_name]
            elif 'x_d_hindcast' in data and isinstance(data['x_d_hindcast'], dict) and ar_name in data['x_d_hindcast']:
                tensor = data['x_d_hindcast'][ar_name]
            elif 'x_d' in data and isinstance(data['x_d'], torch.Tensor):
                tensor = data['x_d'][:, :, -self._num_ar_inputs:]
            elif ar_name in data and isinstance(data[ar_name], torch.Tensor):
                tensor = data[ar_name]
            else:
                found = None
                for k, v in data.items():
                    if isinstance(v, dict) and ar_name in v:
                        found = v[ar_name]
                        break
                if found is not None:
                    tensor = found
                else:
                    raise KeyError(f"Could not find autoregressive input feature '{ar_name}' in data dictionary keys: {list(data.keys())}")

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            ar_tensors.append(tensor)

        x_ar_raw = torch.cat(ar_tensors, dim=-1) # [batch_size, seq_len, num_ar_inputs]

        target_hidden_size = self.config_data.hidden_size
        if h_0 is None:
            h_0 = dynamic_emb.new_zeros((1, batch_size, target_hidden_size))
        if c_0 is None:
            c_0 = dynamic_emb.new_zeros((1, batch_size, target_hidden_size))

        ar_flags = dynamic_emb.new_zeros((batch_size, self._num_ar_inputs))
        last_prediction = dynamic_emb.new_zeros((self._ar_shift, batch_size, self._num_ar_inputs))

        lstm_output = []
        head_outputs: dict[str, list[torch.Tensor]] = {}

        T_hindcast = seq_len - (self.cfg.lead_time or 0) if (self.cfg.lead_time and self.cfg.lead_time < seq_len) else seq_len

        for t in range(seq_len):
            is_hindcast = (t < T_hindcast)
            current_cell = self.hindcast_cell if is_hindcast else self.forecast_cell

            dyn_t = dynamic_emb[:, t, :]
            x_ar_t = x_ar_raw[:, t, :]
            nan_mask = torch.isnan(x_ar_t)
            nan_count = nan_mask.sum().item()

            x_ar = x_ar_t.clone()
            x_ar[nan_mask] = last_prediction[-1, nan_mask]
            ar_flags = dynamic_emb.new_zeros((batch_size, self._num_ar_inputs))
            ar_flags[nan_mask] = 1.0

            static_t = static_embedding.expand(batch_size, -1)
            step_input = torch.cat([static_t, dyn_t, x_ar, ar_flags], dim=-1)

            cell_inputs = torch.unsqueeze(step_input, 0)
            cell_output, (h_0, c_0) = current_cell(cell_inputs, (h_0, c_0))
            lstm_output.append(cell_output.transpose(0, 1))

            head_out = self.head(self.dropout(h_0.transpose(0, 1)))

            # Ensure y_hat is explicitly computed in head_out dict for TorchDynamo compatibility
            if not isinstance(head_out, dict):
                head_out = {'y_hat': head_out}
            elif 'y_hat' not in head_out:
                if 'mu' in head_out and 'pi' in head_out and 'b' in head_out and 'tau' in head_out:
                    head_out['y_hat'] = calc_cmal_mean(head_out['mu'], head_out['b'], head_out['tau'], head_out['pi'])
                elif 'mu' in head_out and 'pi' in head_out:
                    head_out['y_hat'] = torch.sum(head_out['pi'] * head_out['mu'], dim=-1, keepdim=True)
                elif 'mu' in head_out:
                    mu = head_out['mu']
                    if mu.ndim == 3 and mu.shape[-1] > 1:
                        head_out['y_hat'] = torch.mean(mu, dim=-1, keepdim=True)
                    else:
                        head_out['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)

            for key, val in head_out.items():
                if key not in head_outputs:
                    head_outputs[key] = []
                head_outputs[key].append(val)

            pred_raw = head_out['y_hat']
            pred_squeezed = torch.squeeze(pred_raw, dim=1)
            n_dist = pred_squeezed.size(-1) // len(self.cfg.target_variables)
            ar_indices = [idx * n_dist for idx in self._ar_target_indices]
            prediction = pred_squeezed[:, ar_indices]

            new_last_pred = torch.zeros_like(last_prediction)
            new_last_pred[1:] = last_prediction[:-1].clone()
            new_last_pred[0] = prediction
            last_prediction = new_last_pred

            if self.verbose and (t == 0 or t == seq_len // 2 or t == seq_len - 1):
                print(f"  [ARLSTM Forward Step t={t:03d}/{seq_len}] NaNs={nan_count}/{batch_size * self._num_ar_inputs} | "
                      f"Step Input Mean={step_input.mean().item():.4f} | Pred Mean={prediction.mean().item():.4f}", flush=True)
                
                # Hypothesis 4: Variance Shift Check between observed streamflow and substituted predictions
                if nan_mask.any() and (~nan_mask).any():
                    var_obs = x_ar_t[~nan_mask].var().item()
                    var_sub = last_prediction[-1, nan_mask].var().item()
                    print(f"  [H4 Variance Shift t={t:03d}] Observed Streamflow Var={var_obs:.4f} | Substituted Pred Var={var_sub:.4f}", flush=True)

                # Sanity Check: Observed Streamflow Mean & Std Range
                if (~nan_mask).any():
                    mean_obs = x_ar_t[~nan_mask].mean().item()
                    std_obs = x_ar_t[~nan_mask].std().item() if (~nan_mask).sum() > 1 else 0.0
                    print(f"  [Sanity Check t={t:03d}] Masked Ratio={nan_count}/{batch_size} ({nan_count/batch_size*100:.1f}%) | "
                          f"x_ar Observed Mean={mean_obs:.4f}, Std={std_obs:.4f}", flush=True)

                # Hypothesis 1: Exposure Bias & Error Drift Check (if ground-truth target 'y' is in data)
                if 'y' in data and isinstance(data['y'], torch.Tensor):
                    y_target = data['y'][:, t, :] if data['y'].ndim == 3 else data['y']
                    if y_target.shape == prediction.shape:
                        step_err = (prediction - y_target).abs().mean().item()
                        print(f"  [H1 Error Drift t={t:03d}] Mean Abs Error |y_hat - y| = {step_err:.4f}", flush=True)

        pred = {
            'lstm_output': torch.concat(lstm_output, 1),
            'h_n': h_0.transpose(0, 1),
            'c_n': c_0.transpose(0, 1),
        }
        for key, val in head_outputs.items():
            pred[key] = torch.concat(val, 1)

        if 'y_hat' not in pred:
            if 'mu' in pred and 'pi' in pred and 'b' in pred and 'tau' in pred:
                pred['y_hat'] = calc_cmal_mean(pred['mu'], pred['b'], pred['tau'], pred['pi'])
            elif 'mu' in pred and 'pi' in pred:
                pred['y_hat'] = torch.sum(pred['pi'] * pred['mu'], dim=-1, keepdim=True)
            elif 'mu' in pred:
                mu = pred['mu']
                if mu.ndim == 3 and mu.shape[-1] > 1:
                    pred['y_hat'] = torch.mean(mu, dim=-1, keepdim=True)
                else:
                    pred['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)

        return pred
