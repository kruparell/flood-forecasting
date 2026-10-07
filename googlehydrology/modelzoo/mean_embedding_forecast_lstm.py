# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Iterable

from googlehydrology.modelzoo.basemodel import AssimilationTargetSpec, BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.utils.config import Config, EmbeddingSpec, WeightInitOpt
from googlehydrology.utils.configutils import group_features_list
from googlehydrology.utils.lstm_utils import lstm_init
import numpy as np
import torch
import torch.nn as nn

FC_XAVIER = WeightInitOpt.FC_XAVIER


# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Iterable, Any, Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.utils.config import Config, EmbeddingSpec, WeightInitOpt
from googlehydrology.utils.configutils import group_features_list
from googlehydrology.utils.lstm_utils import lstm_init

FC_XAVIER = WeightInitOpt.FC_XAVIER


class MeanEmbeddingForecastLSTM(BaseModel):
  r"""A forecasting model using mean embedding and LSTMs for hindcast and forecast.

  This model implements a specific architecture designed to handle missing input
  data in hydrological
  forecasting. It employs separate embedding networks for hindcast and forecast
  inputs, aggregating
  them via a masked mean operation. This allows the model to robustly handle
  situations where some
  input features might be missing (NaN) by effectively ignoring them in the
  aggregation step.

  The model consists of two main LSTM components:

    1.  **Hindcast LSTM:** Unrolls over the full sequence length (seq_length + lead_time). It processes 
        historical data up to the forecast issue time, padded with NaNs for nowcast data over the forecast horizon 
        (which are masked out during aggregation).
    2.  **Forecast LSTM:** Unrolls over the same full sequence length, taking the output sequence 
        of the Hindcast LSTM concatenated along dim=-1 alongside forecast features (e.g., weather forecasts) 
        to produce predictions over the forecast horizon.

  Key features include:

  -   **Static Embeddings:** Static catchment attributes are embedded and
  provided to all dynamic
      embedding networks.
  -   **Dynamic Embeddings:** Hindcast and forecast features are grouped (e.g.,
  by source or type)
      and processed by separate, specific fully-connected embedding networks.
  -   **Masked Mean Aggregation:** The outputs of the dynamic embedding networks
  are aggregated
      using a masked mean, which ensures that missing data (represented as NaNs)
      do not propagate
      errors or bias the embedding.
  -   **Shared Embeddings:** Features present in both hindcast and forecast
  periods can share
      embedding networks to enforce consistent representation.

  This model is based on the approach described in [#]_.

  Parameters
  ----------
  cfg : Config
      The run configuration, containing all hyperparameters and settings for the
      model structure,
      embedding specifications, and input features.

  References
  ----------
  .. [#] Gauch, M., et al. "How to deal w_ missing input data." Hydrology and
  Earth System Sciences 29.21 (2025): 6221-6235.
      https://hess.copernicus.org/articles/29/6221/2025/
  """

  # Names of state variables in the returned dictionary of the forward function
  state_var_names = [
      'h_n',
      'c_n',
      'h_n_hindcast',
      'c_n_hindcast',
      'h_n_forecast',
      'c_n_forecast',
  ]

  # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
  module_parts = [
      'static_embedding_fc',
      'hindcast_embeddings_fc',
      'forecast_embeddings_fc',
      'shared_embeddings_fc',
      'hindcast_lstm',
      'forecast_lstm',
      'head',
  ]

  supported_assimilation_components = [
      'static_embedding',
      'hindcast_embedding',
      'forecast_embedding',
      'c_0_hindcast',
      'h_0_hindcast',
      'c_0_forecast',
      'h_0_forecast',
  ]

  target_aliases = {
      'embedded_all': [
          'static_embedding',
          'hindcast_embedding',
          'forecast_embedding',
      ],
      'all_embeddings': [
          'static_embedding',
          'hindcast_embedding',
          'forecast_embedding',
      ],
      'embedded_both': ['static_embedding', 'hindcast_embedding'],
      'both_embeddings': ['static_embedding', 'hindcast_embedding'],
      'embedded_statics': ['static_embedding'],
      'embedded_stat': ['static_embedding'],
      'embedded_dynamics': ['hindcast_embedding'],
      'embedded_dyn': ['hindcast_embedding'],
      'c_both': ['c_0_hindcast', 'c_0_forecast'],
      'h_both': ['h_0_hindcast', 'h_0_forecast'],
      'c_n_forecast': ['c_0_forecast'],
      'c_fc': ['c_0_forecast'],
      'h_n_forecast': ['h_0_forecast'],
      'h_fc': ['h_0_forecast'],
      'c_n_hindcast': ['c_0_hindcast'],
      'c_hc': ['c_0_hindcast'],
      'h_n_hindcast': ['h_0_hindcast'],
      'h_hc': ['h_0_hindcast'],
      'both_states': [
          'c_0_hindcast',
          'h_0_hindcast',
          'c_0_forecast',
          'h_0_forecast',
      ],
      'all_states': [
          'c_0_hindcast',
          'h_0_hindcast',
          'c_0_forecast',
          'h_0_forecast',
      ],
      'precip': ['total_precipitation'],
      'precipitation': ['total_precipitation'],
  }

  def get_assimilation_target_specs(self) -> dict[str, AssimilationTargetSpec]:
    """Returns target specs mapping for supported DA components."""
    return {
        'static_embedding': AssimilationTargetSpec(
            name='static_embedding',
            is_time_invariant=True,
            default_reg_weight_key='bg_stat_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: (
                outputs.get('static_embedding')
                if 'static_embedding' in outputs
                else self._calc_static_embedding(
                    ForwardData.from_forward_data(data, self.config_data)
                )
            ),
            inject_override_fn=lambda data, t: data.setdefault(
                'assimilation_overrides', {}
            ).__setitem__('static_embedding', t),
        ),
        'hindcast_embedding': AssimilationTargetSpec(
            name='hindcast_embedding',
            is_time_invariant=False,
            default_reg_weight_key='bg_dyn_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: outputs.get(
                'hindcast_embedding'
            ),
            inject_override_fn=lambda data, t: data.setdefault(
                'assimilation_overrides', {}
            ).__setitem__('hindcast_embedding', t),
        ),
        'forecast_embedding': AssimilationTargetSpec(
            name='forecast_embedding',
            is_time_invariant=False,
            default_reg_weight_key='bg_dyn_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: outputs.get(
                'forecast_embedding'
            ),
            inject_override_fn=lambda data, t: data.setdefault(
                'assimilation_overrides', {}
            ).__setitem__('forecast_embedding', t),
        ),
        'c_0_hindcast': AssimilationTargetSpec(
            name='c_0_hindcast',
            is_time_invariant=False,
            is_recurrent=True,
            default_reg_weight_key='regularization_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: (
                data['c_0_hindcast']
                if data.get('c_0_hindcast') is not None
                else torch.zeros_like(outputs.get('c_n_hindcast'))
            ),
            inject_override_fn=lambda data, t: data.__setitem__(
                'c_0_hindcast', t
            ),
        ),
        'h_0_hindcast': AssimilationTargetSpec(
            name='h_0_hindcast',
            is_time_invariant=False,
            is_recurrent=True,
            default_reg_weight_key='regularization_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: (
                data['h_0_hindcast']
                if data.get('h_0_hindcast') is not None
                else torch.zeros_like(outputs.get('h_n_hindcast'))
            ),
            inject_override_fn=lambda data, t: data.__setitem__(
                'h_0_hindcast', t
            ),
        ),
        'c_0_forecast': AssimilationTargetSpec(
            name='c_0_forecast',
            is_time_invariant=False,
            is_recurrent=True,
            default_reg_weight_key='regularization_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: (
                data['c_0_forecast']
                if data.get('c_0_forecast') is not None
                else torch.zeros_like(outputs.get('c_n_forecast'))
            ),
            inject_override_fn=lambda data, t: data.__setitem__(
                'c_0_forecast', t
            ),
        ),
        'h_0_forecast': AssimilationTargetSpec(
            name='h_0_forecast',
            is_time_invariant=False,
            is_recurrent=True,
            default_reg_weight_key='regularization_weight',
            default_lr_key='learning_rate',
            extract_baseline_fn=lambda data, outputs: (
                data['h_0_forecast']
                if data.get('h_0_forecast') is not None
                else torch.zeros_like(outputs.get('h_n_forecast'))
            ),
            inject_override_fn=lambda data, t: data.__setitem__(
                'h_0_forecast', t
            ),
        ),
    }

    precip_keys = [
        'total_precipitation',
        'hres_total_precipitation',
        'graphcast_total_precipitation',
        'cpc_precipitation',
        'imerg_precipitation',
    ]
    if hasattr(self, 'config_data') and hasattr(
        self.config_data, 'hindcast_inputs_grouped'
    ):
      for feats in self.config_data.hindcast_inputs_grouped.values():
        for f in feats:
          if (
              any(
                  p in f.lower()
                  for p in ('precip', 'precipitation', 'tp', 'prcp')
              )
              and f not in precip_keys
          ):
            precip_keys.append(f)

    for p_key in precip_keys:
      specs[p_key] = AssimilationTargetSpec(
          name=p_key,
          is_time_invariant=False,
          is_recurrent=False,
          clamp_min=0.0,
          default_reg_weight_key='bg_regularization_weight',
          default_lr_key='learning_rate',
          extract_baseline_fn=(
              lambda data, outputs, k=p_key: (
                  data.get('x_d', {}).get(k)
                  if isinstance(data.get('x_d'), dict)
                  else None
              )
          ),
      )
    return specs

  def __init__(self, cfg: Config):
    super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

    self.seq_length = cfg.seq_length
    self.lead_time = cfg.lead_time

    self.config_data = ConfigData.from_config(cfg)

    # Static embedding
    self.static_embedding_fc = self._create_fc(
            embedding_spec=self.config_data.statics_embedding,
            input_size=len(self.config_data.static_attributes),
        )

    # Hindcast embedding networks
    self.hindcast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.hindcast_embedding,
                    input_size=(
                        len(self.config_data.hindcast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in set(
                    self.config_data.hindcast_inputs_grouped.keys()
                ).difference(self.config_data.shared_groups)
            }
        )
    # Forecast embedding networks
    self.forecast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.forecast_embedding,
                    input_size=(
                        len(self.config_data.forecast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in set(
                    self.config_data.forecast_inputs_grouped.keys()
                ).difference(self.config_data.shared_groups)
            }
        )
    # Shared embedding networks (between hindcast and forecast LSTMs)
    self.shared_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.forecast_embedding,
                    input_size=(
                        len(self.config_data.forecast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in self.config_data.shared_groups
            }
        )

    # Hindcast LSTM
    self.hindcast_lstm = nn.LSTM(
            input_size=self.static_embedding_fc.output_size
            + self.config_data.hindcast_embedding.hiddens[-1],
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )

    # Forecast LSTM
    forecast_emb_size = (
        self.config_data.forecast_embedding.hiddens[-1]
        if self.config_data.forecast_inputs_grouped
        else 0
    )
    self.forecast_lstm = nn.LSTM(
        input_size=self.static_embedding_fc.output_size
        + forecast_emb_size
        + self.config_data.hidden_size,
        hidden_size=self.config_data.hidden_size,
        batch_first=True,
    )

    # Head
    self.dropout = nn.Dropout(p=cfg.output_dropout)
    self.head = get_head(
        self.cfg,
        n_in=self.config_data.hidden_size,
        n_out=3 * 4,
        n_hidden=100,
    )

    lstm_init(
            lstms=[self.hindcast_lstm, self.forecast_lstm],
            forget_bias=cfg.initial_forget_bias,
            weight_opts=cfg.weight_init_opts,
        )

  def _create_fc(self, embedding_spec: EmbeddingSpec, input_size: int) -> FC:
    assert input_size > 0, 'Cannot create embedding layer with input size 0'

    emb_type = embedding_spec.type.lower()
    assert emb_type == 'fc', f'{emb_type=} not supported'

    hiddens = embedding_spec.hiddens
    assert len(hiddens) > 0, 'hiddens must have at least one entry'

    activation = embedding_spec.activation
    assert len(activation) == len(hiddens), (
            'hiddens and activation layers must match'
        )

    dropout = float(embedding_spec.dropout)

    return FC(
            input_size=input_size,
            hidden_sizes=hiddens,
            activation=activation,
            dropout=dropout,
            xavier_init=FC_XAVIER in self.cfg.weight_init_opts,
        )

  def _prepare_initial_state(
      self,
      data: dict[str, Any],
      h_keys: tuple[str, ...],
      c_keys: tuple[str, ...],
  ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Extracts and formats initial states (h, c) from the data dictionary."""
    overrides = data.get('assimilation_overrides', {})
    h_init = None
    for k in h_keys:
      if k in overrides and isinstance(overrides[k], torch.Tensor):
        h_init = overrides[k]
        break
      if k in data and isinstance(data[k], torch.Tensor):
        h_init = data[k]
        break

    c_init = None
    for k in c_keys:
      if k in overrides and isinstance(overrides[k], torch.Tensor):
        c_init = overrides[k]
        break
      if k in data and isinstance(data[k], torch.Tensor):
        c_init = data[k]
        break

    if h_init is None and c_init is None:
      return None

    if h_init is not None and c_init is None:
      c_init = torch.zeros_like(h_init)
    elif c_init is not None and h_init is None:
      h_init = torch.zeros_like(c_init)

    if h_init.ndim == 3 and h_init.shape[1] == 1:
      h_init = h_init.transpose(0, 1)
    if c_init.ndim == 3 and c_init.shape[1] == 1:
      c_init = c_init.transpose(0, 1)

    return (h_init, c_init)

  def forward(
      self,
      data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
      window_slice: Optional[Tuple[int, int]] = None,
  ) -> dict[str, torch.Tensor]:
    """Perform a forward pass on the MeanEmbeddingForecastLSTM model."""
    return_state_history = data.get('return_state_history', False)
    window_slice = window_slice or data.get('window_slice', None)
    forward_data = ForwardData.from_forward_data(
        data, self.config_data, window_slice=window_slice
    )

    overrides = data.get('assimilation_overrides', {})

    # Allow explicit static embedding override (for Embedded Static DA)
    static_embedding = overrides.get(
        'static_embedding', data.get('static_embedding', None)
    )
    if static_embedding is None:
      static_embedding = self._calc_static_embedding(forward_data)

    target_len = (
        list(forward_data.forecast_features.values())[0].shape[1]
        if forward_data.forecast_features
        else (
            list(forward_data.hindcast_features.values())[0].shape[1]
            if forward_data.hindcast_features
            else forward_data.static_features.shape[0]
        )
    )

    # Allow explicit dynamic embedding override (for Embedded Temporal DA)
    hindcast_dyn_emb = overrides.get(
        'hindcast_embedding', data.get('hindcast_embedding', None)
    )
    if hindcast_dyn_emb is None:
      hindcast_embeddings = [
          self._calc_dynamic_embedding(
              embedding_network=fc,
              dynamic_data=forward_data.hindcast_features[name],
              static_embedding=static_embedding,
              target_len=target_len,
          )
          for name, fc in self.hindcast_embeddings_fc.items()
      ]
    else:
      hindcast_embeddings = hindcast_dyn_emb

    forecast_dyn_emb = overrides.get(
        'forecast_embedding', data.get('forecast_embedding', None)
    )
    if forecast_dyn_emb is None:
      forecast_embeddings = [
          self._calc_dynamic_embedding(
              embedding_network=fc,
              dynamic_data=forward_data.forecast_features[name],
              static_embedding=static_embedding,
              target_len=target_len,
          )
          for name, fc in self.forecast_embeddings_fc.items()
      ]
    else:
      forecast_embeddings = forecast_dyn_emb

    if isinstance(hindcast_dyn_emb, torch.Tensor) and isinstance(
        forecast_dyn_emb, torch.Tensor
    ):
      shared_embeddings = []
    else:
      shared_embeddings = [
          self._calc_dynamic_embedding(
              embedding_network=fc,
              dynamic_data=forward_data.forecast_features[name],
              static_embedding=static_embedding,
              target_len=target_len,
          )
          for name, fc in self.shared_embeddings_fc.items()
      ]

    # 1. Hindcast State Initialization
    hx_hindcast = self._prepare_initial_state(
        data,
        h_keys=('h_0_hindcast', 'h_hindcast'),
        c_keys=('c_0_hindcast', 'c_hindcast'),
    )

    # 2. Hindcast LSTM Pass
    hindcast_state, (h_hc, c_hc) = self._calc_lstm(
        lstm=self.hindcast_lstm,
        embeddings=(
            hindcast_embeddings
            if isinstance(hindcast_dyn_emb, torch.Tensor)
            else (hindcast_embeddings + shared_embeddings)
        ),
        static_embedding=static_embedding,
        hx=hx_hindcast,
        return_state_history=return_state_history,
    )

    # 3. Forecast State Initialization
    hx_forecast = self._prepare_initial_state(
        data,
        h_keys=('h_0_forecast', 'h_forecast', 'h_0', 'h_n'),
        c_keys=('c_0_forecast', 'c_forecast', 'c_0', 'c_n'),
    )

    # 4. Forecast LSTM Pass
    forecast_state, (h_fc, c_fc) = self._calc_lstm(
        lstm=self.forecast_lstm,
        embeddings=(
            forecast_embeddings
            if isinstance(forecast_dyn_emb, torch.Tensor)
            else (forecast_embeddings + shared_embeddings)
        ),
        static_embedding=static_embedding,
        other_inputs=hindcast_state,
        hx=hx_forecast,
        return_state_history=return_state_history,
    )

    if getattr(self.cfg, 'predict_n_hindcast', 0) > 0:
      hind_n = self.cfg.predict_n_hindcast
      combined_state = torch.cat(
          [hindcast_state[:, -hind_n:, :], forecast_state], dim=1
      )
      head = self._calc_head(combined_state)
    else:
      head = self._calc_head(forecast_state)

    if 'y_hat' not in head:
      if 'mu' in head and 'pi' in head:
        head['y_hat'] = torch.sum(head['pi'] * head['mu'], dim=-1, keepdim=True)
      elif 'mu' in head:
        head['y_hat'] = (
            head['mu'] if head['mu'].ndim == 3 else head['mu'].unsqueeze(-1)
        )

    # Attach Hindcast and Forecast states (lists/sequences if return_state_history=True, else terminal tensors)
    head['h_n_hindcast'] = h_hc
    head['c_n_hindcast'] = c_hc
    head['h_n_forecast'] = h_fc
    head['c_n_forecast'] = c_fc

    # Primary state aliases default to forecast states
    head['h_n'] = h_fc
    head['c_n'] = c_fc
    head['static_embedding'] = static_embedding
    head['hindcast_embedding'] = (
        hindcast_embeddings
        if isinstance(hindcast_dyn_emb, torch.Tensor)
        else self._masked_mean(hindcast_embeddings + shared_embeddings)
    )
    head['forecast_embedding'] = (
        forecast_embeddings
        if isinstance(forecast_dyn_emb, torch.Tensor)
        else self._masked_mean(forecast_embeddings + shared_embeddings)
    )
    return head

  @property
  def state_var_names(self) -> list[str]:
    return [
        'h_n',
        'c_n',
        'h_n_hindcast',
        'c_n_hindcast',
        'h_n_forecast',
        'c_n_forecast',
    ]

  def _make_static_embedding_repeated(
      self, time_length: int, static_embedding: torch.Tensor
  ) -> torch.Tensor:
    """Returns the attributes repeated w.r.t the time length."""
    return static_embedding.unsqueeze(1).repeat(1, time_length, 1)

  def _make_nan_padding(
        self,
        batch_size: int,
        nan_padding_length: int,
        embedding_size: int,
        device: str,
    ) -> torch.Tensor:
    """Returns a nan-padding tensor."""
    return torch.full(
            (batch_size, nan_padding_length, embedding_size),
            np.nan,
            device=device,
        )

  def _append_static_embedding(
        self, embedding: torch.Tensor, static_embedding: torch.Tensor
    ) -> torch.Tensor:
    """Append static embedding to another embedding tensor."""
    time_length = embedding.shape[1]
    static_embedding_repeated = self._make_static_embedding_repeated(
            time_length, static_embedding
        )
    return torch.cat([embedding, static_embedding_repeated], dim=-1)

  def _add_nan_padding(
      self, embedding: torch.Tensor, target_len: int
  ) -> torch.Tensor:
    """Pad the embedding tensor with nan value to target timespan length."""
    batch_size = embedding.shape[0]
    nan_padding_length = target_len - embedding.shape[1]
    if nan_padding_length <= 0:
      return embedding
    embedding_size = embedding.shape[2]
    nan_padding = self._make_nan_padding(
        batch_size, nan_padding_length, embedding_size, embedding.device
    )
    return torch.cat([embedding, nan_padding], dim=1)

  def _masked_mean(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Calculate mean between list of tensors, skipping nan values."""
    merged = torch.cat([e.unsqueeze(-1) for e in tensors], dim=-1)
    res = torch.nanmean(merged, dim=-1)
    return torch.nan_to_num(res, nan=0.0)

  def _calc_static_embedding(self, forward_data: 'ForwardData') -> torch.Tensor:
    static_data = forward_data.static_features.to(
        dtype=self.static_embedding_fc.net[0].weight.dtype
    )
    return self.static_embedding_fc(static_data)

  def _calc_dynamic_embedding(
      self,
      embedding_network: nn.Module,
      dynamic_data: torch.Tensor,
      static_embedding: torch.Tensor,
      target_len: int,
  ) -> torch.Tensor:
    dynamic_data = dynamic_data.to(dtype=static_embedding.dtype)
    nan_mask = torch.isnan(dynamic_data).any(dim=-1, keepdim=True)
    safe_dynamic = torch.where(
        nan_mask, torch.zeros_like(dynamic_data), dynamic_data
    )
    dynamic_data_concat = self._append_static_embedding(
        safe_dynamic, static_embedding
    )
    output = embedding_network(dynamic_data_concat)
    output = torch.where(
        nan_mask, torch.full_like(output, float('nan')), output
    )
    return self._add_nan_padding(output, target_len)

  def _calc_lstm(
      self,
      lstm: nn.LSTM,
      embeddings: Iterable[torch.Tensor] | torch.Tensor,
      static_embedding: torch.Tensor,
      other_inputs: torch.Tensor | None = None,
      hx: tuple[torch.Tensor, torch.Tensor] | None = None,
      return_state_history: bool = False,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(embeddings, torch.Tensor):
      masked_mean_embeddings = embeddings
      if (
          other_inputs is not None
          and other_inputs.shape[1] != masked_mean_embeddings.shape[1]
      ):
        target_len = masked_mean_embeddings.shape[1]
        if target_len == 0:
          other_inputs = other_inputs[:, :0, :]
        else:
          other_inputs = other_inputs[:, -target_len:, :]
    elif embeddings:
      masked_mean_embeddings = self._masked_mean(embeddings)
      if (
          other_inputs is not None
          and other_inputs.shape[1] != masked_mean_embeddings.shape[1]
      ):
        target_len = masked_mean_embeddings.shape[1]
        if target_len == 0:
          other_inputs = other_inputs[:, :0, :]
        else:
          other_inputs = other_inputs[:, -target_len:, :]
    else:
      batch_size = static_embedding.shape[0]
      seq_len = other_inputs.shape[1] if other_inputs is not None else 1
      masked_mean_embeddings = torch.zeros(
          (batch_size, seq_len, 0), device=static_embedding.device
      )

    if other_inputs is not None:
      masked_mean_embeddings = torch.cat(
          [masked_mean_embeddings, other_inputs], dim=-1
      )
    lstm_inputs = self._append_static_embedding(
            masked_mean_embeddings, static_embedding
        )

    seq_len = lstm_inputs.shape[1]
    batch_size = static_embedding.shape[0]

    if seq_len == 0:
      output = torch.zeros(
          (batch_size, 0, lstm.hidden_size), device=static_embedding.device
      )
      num_layers = lstm.num_layers
      if return_state_history:
        h_out = torch.zeros(
            (num_layers, batch_size, 0, lstm.hidden_size),
            device=static_embedding.device,
        )
        c_out = torch.zeros(
            (num_layers, batch_size, 0, lstm.hidden_size),
            device=static_embedding.device,
        )
      else:
        if hx is not None:
          h_out, c_out = hx
        else:
          h_out = torch.zeros(
              (num_layers, batch_size, lstm.hidden_size),
              device=static_embedding.device,
          )
          c_out = torch.zeros(
              (num_layers, batch_size, lstm.hidden_size),
              device=static_embedding.device,
          )
      return output, (h_out, c_out)

    # Unroll step-by-step if full sequence history is requested
    if return_state_history:
      outputs = []
      h_list = []
      c_list = []
      curr_hx = hx
      for t in range(seq_len):
        x_t = lstm_inputs[:, t : t + 1, :]
        out_t, curr_hx = lstm(input=x_t, hx=curr_hx)
        outputs.append(out_t)
        h_list.append(curr_hx[0])
        c_list.append(curr_hx[1])

      output = torch.cat(outputs, dim=1)
      h_seq = torch.stack(
          h_list, dim=2
      )  # [num_layers, batch, seq_len, hidden_size]
      c_seq = torch.stack(
          c_list, dim=2
      )  # [num_layers, batch, seq_len, hidden_size]
      return output, (h_seq, c_seq)
    else:
      output, (h_n, c_n) = lstm(input=lstm_inputs, hx=hx)
      return output, (h_n, c_n)

  def _calc_head(self, forecast_state: torch.Tensor) -> dict[str, torch.Tensor]:
    return self.head(self.dropout(forecast_state))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, cfg: Config) -> 'ConfigData':
        statics_embedding = cfg.statics_embedding
        hindcast_embedding = cfg.hindcast_embedding or cfg.dynamics_embedding
        forecast_embedding = cfg.forecast_embedding or cfg.dynamics_embedding
        assert statics_embedding is not None
        assert hindcast_embedding is not None
        assert forecast_embedding is not None

        hindcast_inputs_grouped = group_features_list(cfg.hindcast_inputs)
        forecast_inputs_grouped = group_features_list(cfg.forecast_inputs)
        shared_groups = [
            e for e in hindcast_inputs_grouped if e in forecast_inputs_grouped
        ]
        for group in shared_groups:
            assert (
                hindcast_inputs_grouped[group] == forecast_inputs_grouped[group]
            ), (
                f'Same features must be defined in forecast and hindcast for {group=}'
            )

        return ConfigData(
            hidden_size=cfg.hidden_size,
            statics_embedding=statics_embedding,
            hindcast_embedding=hindcast_embedding,
            forecast_embedding=forecast_embedding,
            static_attributes=tuple(cfg.static_attributes),
            hindcast_inputs_grouped=hindcast_inputs_grouped,
            forecast_inputs_grouped=forecast_inputs_grouped,
            shared_groups=shared_groups,
        )

    hidden_size: int
    statics_embedding: EmbeddingSpec
    hindcast_embedding: EmbeddingSpec
    forecast_embedding: EmbeddingSpec
    static_attributes: tuple[str, ...]
    hindcast_inputs_grouped: dict[str, list[str]]
    forecast_inputs_grouped: dict[str, list[str]]
    shared_groups: list[str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
  @classmethod
  def from_forward_data(
      cls,
      data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
      config_data: ConfigData,
      window_slice: Optional[Tuple[int, int]] = None,
  ) -> 'ForwardData':
    window_slice = window_slice or data.get('window_slice', None)
    hindcast_dict = (
        data['x_d_hindcast'] if 'x_d_hindcast' in data else data['x_d']
    )
    forecast_dict = data.get('x_d_forecast', {})

    start_step, end_step = (0, None) if window_slice is None else window_slice

    # The forecast features extend `lead_delta` steps past the hindcast, so the
    # forecast slice must be widened by that much to stay aligned. Verified
    # against production: hindcast (B, 365, 1), forecast (B, 372, 1).
    lead_delta = 0
    if window_slice is not None and forecast_dict and hindcast_dict:
      first_hc = next(iter(hindcast_dict.values()), None)
      first_fc = next(iter(forecast_dict.values()), None)
      if (
          isinstance(first_hc, torch.Tensor)
          and isinstance(first_fc, torch.Tensor)
          and first_fc.ndim == 3
          and first_hc.ndim == 3
          and first_fc.shape[1] > first_hc.shape[1]
      ):
        lead_delta = first_fc.shape[1] - first_hc.shape[1]

    end_step_hc = end_step
    end_step_fc = (end_step + lead_delta) if end_step is not None else None

    return ForwardData(
        static_features=data['x_s'],
        hindcast_features={
            name: _concat_tensors_from_dict(
                hindcast_dict,
                keys=features,
                slice_bounds=(start_step, end_step_hc),
            )
            for name, features in config_data.hindcast_inputs_grouped.items()
        },
        forecast_features={
            name: _concat_tensors_from_dict(
                forecast_dict,
                keys=features,
                slice_bounds=(start_step, end_step_fc),
            )
            for name, features in config_data.forecast_inputs_grouped.items()
        },
    )

  static_features: torch.Tensor
  hindcast_features: dict[str, torch.Tensor]
  forecast_features: dict[str, torch.Tensor]


def _concat_tensors_from_dict(
    data: dict[str, torch.Tensor],
    *,
    keys: Iterable[str],
    slice_bounds: Optional[Tuple[int, Optional[int]]] = None,
) -> torch.Tensor:
  tensors = []
  for e in keys:
    t = data[e]
    if slice_bounds is not None and t.ndim == 3:
      start, end = slice_bounds
      t_len = t.shape[1]
      s_idx = min(start, t_len)
      e_idx = min(end, t_len) if end is not None else t_len
      t = t[:, s_idx:e_idx, :]
      # An empty slice is never legitimate, and if allowed through it surfaces
      # far downstream as an opaque broadcast error ("size of tensor a (7) must
      # match the size of tensor b (0)") with no indication of the cause. Fail
      # here instead, where the offending bounds are still in scope.
      if t.shape[1] == 0:
        raise ValueError(
            f'window_slice {slice_bounds} produced an EMPTY slice for feature '
            f'{e!r}, whose time dimension is {t_len}. The requested start '
            f'({start}) is at or beyond the end of this array, which usually '
            'means global sequence coordinates were applied to an array that '
            'is indexed locally (e.g. an appended forecast block).'
        )
    tensors.append(t)
  return torch.cat(tensors, dim=-1)
