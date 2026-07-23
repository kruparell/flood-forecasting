"""Autoregressive LSTM (AR-LSTM) with Mean Embedding for dynamic hydrological inputs."""

import re
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from googlehydrology.modelzoo.head import get_head
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    ConfigData,
    ForwardData,
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config


class ARLSTM(MeanEmbeddingForecastLSTM):
    """Autoregressive LSTM model with Mean Embedding for hydrological forecasting."""

    state_var_names = ['h_n', 'c_n']

    def __init__(self, cfg: Config):
        super(ARLSTM, self).__init__(cfg=cfg)

        self._ar_shift = self._get_ar_shift()
        self._num_ar_inputs = len(self.cfg.autoregressive_inputs)
        if self.output_size != self._num_ar_inputs:
            raise ValueError('The AR-LSTM requires output_size to match the number of autoregressive inputs.')

        input_size = (
            self.static_embedding_fc.output_size
            + self.config_data.hindcast_embedding.hiddens[-1]
            + 2 * self._num_ar_inputs
        )
        self.hindcast_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )
        self.cell = self.hindcast_lstm
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
        self._reset_parameters()

    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        state_dict_copy = state_dict.copy()
        for k in [k for k in state_dict_copy if 'forecast_lstm' in k]:
            del state_dict_copy[k]
        if hasattr(self, 'forecast_lstm'):
            del self.forecast_lstm
        return super().load_state_dict(state_dict_copy, strict=False, assign=assign)

    def _get_ar_shift(self) -> int:
        shifts = set()
        for inp in self.cfg.autoregressive_inputs:
            capture = re.search(r'^(.*)_shift(\d+)$', inp)
            if not capture:
                raise ValueError(f"Autoregressive inputs must follow <var>_shift<lag>, got: {inp}")
            if capture[1] not in self.cfg.target_variables:
                raise ValueError(f"Autoregressive inputs must be shifted target variables, got: {capture[1]}")
            shifts.add(int(capture[2]))
        if len(shifts) > 1:
            raise ValueError('All autoregressive inputs must use the same lag shift.')
        shift = shifts.pop() if shifts else 1
        if shift <= 0:
            raise ValueError('Autoregressive inputs must be shifted by at least one timestep.')
        return shift

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[
                self.cfg.hidden_size : 2 * self.cfg.hidden_size
            ] = self.cfg.initial_forget_bias

    def pre_model_hook(self, data: dict[str, torch.Tensor], is_train: bool) -> dict[str, torch.Tensor]:
        """Pre-processes input batches, creating shifted autoregressive observations when needed."""
        data = super().pre_model_hook(data, is_train=is_train)
        if 'y' in data and self.cfg.autoregressive_inputs:
            y = data['y']
            x_d = data.get('x_d', data.get('x_d_hindcast', None))
            if isinstance(x_d, dict):
                for ar_name in self.cfg.autoregressive_inputs:
                    capture = re.search(r'^(.*)_shift(\d+)$', ar_name)
                    if capture:
                        shift, var_base = int(capture[2]), capture[1]
                        if ar_name not in x_d or (isinstance(x_d[ar_name], torch.Tensor) and torch.equal(x_d[ar_name], y)):
                            y_shifted = torch.roll(y, shifts=shift, dims=1)
                            y_shifted[:, :shift, :] = float('nan')

                            mean_key, std_key = f"{var_base}_mean", f"{var_base}_std"
                            if mean_key in data and std_key in data:
                                y_shifted = (y_shifted - data[mean_key]) / data[std_key]
                            elif 'scaler_mean' in data and 'scaler_std' in data:
                                y_shifted = (y_shifted - data['scaler_mean']) / data['scaler_std']
                            x_d[ar_name] = y_shifted
        return data

    def _get_forward_data(self, data: Dict[str, Any]) -> Tuple[ForwardData, torch.Tensor]:
        static_features = data['x_s']
        if 'x_one_hot' in data:
            static_features = torch.cat([static_features, data['x_one_hot']], dim=-1)

        hindcast_dict = data['x_d_hindcast'] if 'x_d_hindcast' in data else data['x_d']
        if isinstance(hindcast_dict, dict):
            x_ar = torch.cat([hindcast_dict[ar] for ar in self.cfg.autoregressive_inputs], dim=-1)
            cleaned = {'x_s': static_features, 'x_d_hindcast': hindcast_dict}
            if 'x_d_forecast' in data:
                cleaned['x_d_forecast'] = data['x_d_forecast']
            forward_data = ForwardData.from_forward_data(cleaned, self.config_data)
        else:
            x_d = hindcast_dict
            if x_d.ndim == 3 and x_d.shape[0] != static_features.shape[0]:
                x_d = x_d.transpose(0, 1)
            x_ar = x_d[:, :, -self._num_ar_inputs:]
            forward_data = ForwardData(
                static_features=static_features,
                hindcast_features={'default': x_d[:, :, :-self._num_ar_inputs]},
                forecast_features={},
            )
        return forward_data, x_ar

    def forward(self, data: Dict[str, Any], h_0: torch.Tensor = None, c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Perform forward unroll with autoregressive feedback."""
        forward_data, x_ar = self._get_forward_data(data)
        static_embedding = self._calc_static_embedding(forward_data)
        target_len = list(forward_data.hindcast_features.values())[0].shape[1] if forward_data.hindcast_features else (
            list(forward_data.forecast_features.values())[0].shape[1] if forward_data.forecast_features else forward_data.static_features.shape[0]
        )

        hindcast_embeddings = [
            self._calc_dynamic_embedding(fc, forward_data.hindcast_features[name], static_embedding, target_len)
            for name, fc in self.hindcast_embeddings_fc.items()
        ]
        shared_embeddings = [
            self._calc_dynamic_embedding(fc, forward_data.forecast_features[name], static_embedding, target_len)
            for name, fc in self.shared_embeddings_fc.items()
        ]
        all_embeddings = hindcast_embeddings + shared_embeddings
        masked_mean = self._masked_mean(all_embeddings) if all_embeddings else x_ar.new_zeros((x_ar.shape[0], x_ar.shape[1], 0))

        embedded_inputs = self._append_static_embedding(masked_mean, static_embedding)
        batch_size, seq_len = x_ar.shape[0], x_ar.shape[1]

        if h_0 is None:
            h_0 = data['h_n'].transpose(0, 1) if 'h_n' in data else embedded_inputs.new_zeros((1, batch_size, self.cfg.hidden_size))
        if c_0 is None:
            c_0 = data['c_n'].transpose(0, 1) if 'c_n' in data else embedded_inputs.new_zeros((1, batch_size, self.cfg.hidden_size))

        ar_flags = embedded_inputs.new_zeros((batch_size, self._num_ar_inputs))
        last_prediction = embedded_inputs.new_zeros((self._ar_shift, batch_size, self.output_size))

        lstm_output, y_hat = [], []
        for t in range(seq_len):
            emb_t = embedded_inputs[:, t, :]
            x_ar_t = x_ar[:, t, :].clone()

            replace_idx = torch.isnan(x_ar_t)
            x_ar_t[replace_idx] = last_prediction[-1, replace_idx]
            ar_flags[:, :] = 0
            ar_flags[replace_idx] = 1

            cell_in = torch.cat([emb_t, x_ar_t, ar_flags], dim=-1).unsqueeze(1)
            cell_out, (h_0, c_0) = self.hindcast_lstm(cell_in, (h_0, c_0))
            lstm_output.append(cell_out)

            pred = self.head(self.dropout(cell_out))['y_hat'].squeeze(1)
            last_prediction[1:] = last_prediction[:-1].clone()
            last_prediction[0] = pred
            y_hat.append(pred)

        return {
            'lstm_output': torch.cat(lstm_output, dim=1),
            'h_n': h_0.transpose(0, 1),
            'c_n': c_0.transpose(0, 1),
            'y_hat': torch.stack(y_hat, dim=1),
        }
