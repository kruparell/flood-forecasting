# Copyright 2026 Google LLC
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

import re
import torch
from torch import nn

from googlehydrology.modelzoo.basemodel import AssimilationTargetSpec
from googlehydrology.modelzoo.mean_embedding_forecast_lstm import (
    ForwardData,
    MeanEmbeddingForecastLSTM,
)
from googlehydrology.utils.config import Config
from googlehydrology.utils.lstm_utils import lstm_init


class MaskedForecastLSTM(MeanEmbeddingForecastLSTM):
  """Single-Cell Masked-Forecasting LSTM (MF2LSTM, Acuna Espinoza et al., 2026).

  Combines the canonical GoogleHydrology `MeanEmbeddingForecastLSTM` backbone
  (Gauch et al., 2025) with the `MF2LSTM` training and discharge-assimilation
  architecture from `Hy2DL` (Acuna Espinoza et al., 2026):

  1. **Discharge as a Provider Group in Masked-Mean (`ar`)**:
     Lagged observed discharge (`streamflow_shift1`) is embedded into the 20-D
     dynamic latent space by dedicated MLP networks (`ar_hc_embedding_fc` and
     `ar_fc_embedding_fc`, conditioned on static catchment embeddings) and
     aggregated with meteorological provider embeddings via `_masked_mean`
     (`torch.nanmean`). When discharge is `NaN` (missing or masked), its
     embedding is masked to `NaN` and automatically excluded from the mean,
     reducing identically to the open-loop `MeanEmbeddingForecastLSTM`.
  2. **Two-Level Probabilistic Masking (`nan_seq` + `nan_step`)**:
     During training, applies sequence-level group dropout (`nan_seq_prob`,
     dropping the entire `"ar"` provider across both hindcast and forecast for a
     subset of batch samples) and step-level dropout (`nan_step_prob`, dropping
     individual timesteps).
  3. **Teacher-Forcing with Sigmoid-Scheduled Multiplicative Noise (`forward_tf`)**:
     During training (`self.training` and `ar_teacher_forcing=True`), runs a
     single fused `self.lstm` pass across `[hindcast, forecast]` where forecast
     lagged discharge is perturbed by the `Hy2DL` sigmoid noise schedule:
     `sigma(k) = 0.2 / (1 + exp(-(k - 6)))`, `q_tilde = q * (1 + sigma(k) * N(0,1))`.
  4. **Autoregressive Forecast Rollout (`forward_ar`)**:
     During evaluation (`not self.training`), runs the hindcast window in parallel
     and unrolls `self.lstm` step-by-step across the forecast horizon, feeding
     observed `Q_{t0}` at the first forecast step (`lead_time=1`) and recursively
     feeding predicted discharge `Q_hat` into subsequent forecast steps.
  """

  module_parts = [
      'static_embedding_fc',
      'hindcast_embeddings_fc',
      'forecast_embeddings_fc',
      'shared_embeddings_fc',
      'ar_hc_embedding_fc',
      'ar_fc_embedding_fc',
      'lstm',
      'head',
  ]

  def __init__(self, cfg: Config):
    super().__init__(cfg=cfg)

    # 1. Identify assimilated / autoregressive discharge input features
    self.assimilated_inputs = list(
        getattr(cfg, 'assimilated_inputs', None)
        or getattr(cfg, 'autoregressive_inputs', None)
        or cfg._cfg.get('assimilated_inputs', [])
        or cfg._cfg.get('autoregressive_inputs', [])
    )
    if not self.assimilated_inputs:
      self.assimilated_inputs = ['streamflow_shift1']

    self._num_q_inputs = len(self.assimilated_inputs)

    # 2. Dedicated "ar" provider embedding networks (mapping into the same 20-D
    # dynamic embedding space as the meteorological providers, conditioned on
    # static_embedding).
    self.ar_hc_embedding_fc = self._create_fc(
        embedding_spec=self.config_data.hindcast_embedding,
        input_size=self._num_q_inputs + self.static_embedding_fc.output_size,
    )
    self.ar_fc_embedding_fc = self._create_fc(
        embedding_spec=self.config_data.forecast_embedding,
        input_size=self._num_q_inputs + self.static_embedding_fc.output_size,
    )

    # 3. Single shared LSTM cell matching canonical MeanEmbeddingForecastLSTM
    # input dimension: 20 (static) + 20 (masked-mean dynamic+ar) + (1 if counter)
    static_dim = self.config_data.statics_embedding.hiddens[-1]
    dynamic_dim = self.config_data.hindcast_embedding.hiddens[-1]
    assert (
        dynamic_dim == self.config_data.forecast_embedding.hiddens[-1]
    ), 'Hindcast and forecast dynamic embeddings must share output dimension.'

    self._has_counter = bool(getattr(cfg, 'timestep_counter', False))
    self._cell_input_dim = static_dim + dynamic_dim + (
        1 if self._has_counter else 0
    )

    self.lstm = nn.LSTM(
        input_size=self._cell_input_dim,
        hidden_size=self.config_data.hidden_size,
        batch_first=True,
    )

    # 4. Two-level probabilistic masking & teacher-forcing parameters
    # Default to Eduardo's Hy2DL MF2LSTM probabilities: nan_seq=0.3, nan_step=0.5
    self.nan_step_prob = float(
        cfg._cfg.get('streamflow_step_mask_prob', None)
        or cfg._cfg.get('streamflow_mask_prob', 0.5)
    )
    self.nan_seq_prob = float(
        cfg._cfg.get('streamflow_seq_mask_prob', 0.3)
    )
    self.ar_teacher_forcing = bool(
        cfg._cfg.get('ar_teacher_forcing', True)
    )
    self.ar_forecast_mode = str(
        cfg._cfg.get('ar_forecast_mode', 'autoregressive')
    ).lower()

    # Ensure head output dimension matches self.output_size (e.g. 1 for regression, 16 for 4-component CMAL)
    from googlehydrology.modelzoo.head import get_head
    self.head = get_head(
        self.cfg,
        n_in=self.config_data.hidden_size,
        n_out=self.output_size,
        n_hidden=getattr(self.cfg, 'n_hidden', 100) or 100,
    )

    # Cold-start weight initialization identical to canonical filtered model
    self._reset_single_lstm_parameters()
    print(
        f'[MaskedForecastLSTM Cold-Start Init] cell_input_dim={self._cell_input_dim},'
        f' hidden_size={self.config_data.hidden_size},'
        f' ar_provider={self.assimilated_inputs},'
        f' nan_seq_prob={self.nan_seq_prob}, nan_step_prob={self.nan_step_prob},'
        f' ar_teacher_forcing={self.ar_teacher_forcing}',
        flush=True,
    )

  def _reset_single_lstm_parameters(self):
    forget_bias = self.cfg.initial_forget_bias
    if forget_bias is not None:
      self.lstm.bias_hh_l0.data[
          self.config_data.hidden_size : 2 * self.config_data.hidden_size
      ] = forget_bias
    lstm_init(
        lstms=[self.lstm],
        forget_bias=forget_bias,
        weight_opts=self.cfg.weight_init_opts,
    )

  @staticmethod
  def _add_noise_ar(x: torch.Tensor) -> torch.Tensor:
    """Hy2DL sigmoid-scaled multiplicative Gaussian noise for teacher-forced AR."""
    steps = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)
    incremental_factor = (0.2 / (1.0 + torch.exp(-(steps - 6.0)))).view(
        1, -1, 1
    )
    noise = torch.randn_like(x) * incremental_factor
    return x + (x * noise)

  def _extract_q_obs(
      self,
      data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
      batch_size: int,
      total_seq_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor:
    """Extracts lagged streamflow tensor [B, total_seq_len, num_q] across hindcast+forecast."""
    q_tensors = []
    for q_name in self.assimilated_inputs:
      tensor = None
      if (
          'x_d' in data
          and isinstance(data['x_d'], dict)
          and q_name in data['x_d']
      ):
        tensor = data['x_d'][q_name]
      elif (
          'x_d_hindcast' in data
          and isinstance(data['x_d_hindcast'], dict)
          and q_name in data['x_d_hindcast']
      ):
        tensor = data['x_d_hindcast'][q_name]
      elif 'x_d' in data and isinstance(data['x_d'], torch.Tensor):
        tensor = data['x_d'][:, :, -self._num_q_inputs :]
      elif q_name in data and isinstance(data[q_name], torch.Tensor):
        tensor = data[q_name]
      else:
        for _, v in data.items():
          if isinstance(v, dict) and q_name in v:
            tensor = v[q_name]
            break

      if tensor is None and 'y' in data and isinstance(data['y'], torch.Tensor):
        y_tensor = data['y']
        m = re.search(r'_shift(\d+)$', q_name)
        shift_steps = int(m.group(1)) if m else 0
        if shift_steps > 0:
          pad = y_tensor.new_full(
              (y_tensor.shape[0], shift_steps, y_tensor.shape[-1]),
              float('nan'),
          )
          tensor = torch.cat([pad, y_tensor[:, :-shift_steps, :]], dim=1)
        else:
          tensor = y_tensor.clone()

      if tensor is None:
        raise KeyError(
            f"Could not find assimilated input feature '{q_name}' in data"
            f' dictionary keys: {list(data.keys())}'
        )

      if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)

      # If tensor only covers hindcast (T_hc < total_seq_len) and data['y'] is
      # available, fill the forecast horizon from shifted data['y'] for teacher-forcing.
      if tensor.shape[1] < total_seq_len:
        if (
            'y' in data
            and isinstance(data['y'], torch.Tensor)
            and data['y'].shape[1] >= total_seq_len
        ):
          y_tensor = data['y']
          m = re.search(r'_shift(\d+)$', q_name)
          shift_steps = int(m.group(1)) if m else 1
          if shift_steps > 0:
            pad = y_tensor.new_full(
                (y_tensor.shape[0], shift_steps, y_tensor.shape[-1]),
                float('nan'),
            )
            y_shifted = torch.cat([pad, y_tensor[:, :-shift_steps, :]], dim=1)
          else:
            y_shifted = y_tensor
          fc_part = y_shifted[:, tensor.shape[1] : total_seq_len, :]
          tensor = torch.cat([tensor, fc_part], dim=1)
        else:
          pad_len = total_seq_len - tensor.shape[1]
          tensor = torch.cat(
              [
                  tensor,
                  tensor.new_full(
                      (batch_size, pad_len, tensor.shape[-1]), float('nan')
                  ),
              ],
              dim=1,
          )
      elif tensor.shape[1] > total_seq_len:
        tensor = tensor[:, :total_seq_len, :]

      q_tensors.append(tensor)

    return torch.cat(q_tensors, dim=-1).to(dtype=dtype, device=device)

  def _embed_ar_group(
      self,
      q_seq: torch.Tensor,
      static_embedding: torch.Tensor,
      fc_net: nn.Module,
  ) -> torch.Tensor:
    """Embeds `q_seq` [B, L, num_q] via `fc_net` and masks NaN inputs back to NaN."""
    step_nan_mask = torch.isnan(q_seq).any(dim=-1, keepdim=True)
    q_clean = torch.where(step_nan_mask, torch.zeros_like(q_seq), q_seq)
    expanded_static = static_embedding.unsqueeze(1).expand(
        -1, q_seq.shape[1], -1
    )
    ar_in = torch.cat([expanded_static, q_clean], dim=-1)
    ar_emb = fc_net(ar_in)
    return torch.where(step_nan_mask, float('nan'), ar_emb)

  def _build_counter(
      self,
      data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
      batch_size: int,
      hc_len: int,
      fc_len: int,
      dtype: torch.dtype,
      device: torch.device,
  ) -> torch.Tensor | None:
    if not self._has_counter:
      return None
    hc_dict = data.get('x_d_hindcast') or data.get('x_d')
    fc_dict = data.get('x_d_forecast')
    if (
        isinstance(hc_dict, dict)
        and 'timestep_counter' in hc_dict
        and isinstance(fc_dict, dict)
        and 'timestep_counter' in fc_dict
    ):
      c_hc = hc_dict['timestep_counter'][:, :hc_len]
      c_fc = fc_dict['timestep_counter'][:, -fc_len:]
      if c_hc.ndim == 2:
        c_hc = c_hc.unsqueeze(-1)
      if c_fc.ndim == 2:
        c_fc = c_fc.unsqueeze(-1)
      return torch.cat([c_hc, c_fc], dim=1).to(dtype=dtype, device=device)

    c_hc = torch.zeros((batch_size, hc_len, 1), dtype=dtype, device=device)
    c_fc = (
        torch.arange(1, fc_len + 1, dtype=dtype, device=device)
        .view(1, fc_len, 1)
        .expand(batch_size, fc_len, 1)
    )
    return torch.cat([c_hc, c_fc], dim=1)

  def forward(
      self,
      data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
      h_0: torch.Tensor | None = None,
      c_0: torch.Tensor | None = None,
      return_state_history: bool = False,
  ) -> dict[str, torch.Tensor]:
    forward_data = ForwardData.from_forward_data(data, self.config_data)
    overrides = data.get('assimilation_overrides', {})

    # 1. Static catchment embedding [B, 20]
    static_embedding = overrides.get(
        'static_embedding',
        self._calc_static_embedding(forward_data),
    )
    batch_size = static_embedding.shape[0]

    # 2. Determine hindcast length (hc_len) and forecast length (fc_len)
    lead_time = getattr(self.cfg, 'lead_time', 7) or 7
    sample_hc_feat = (
        next(iter(forward_data.hindcast_features.values()))
        if forward_data.hindcast_features
        else None
    )
    sample_fc_feat = (
        next(iter(forward_data.forecast_features.values()))
        if forward_data.forecast_features
        else None
    )
    if sample_hc_feat is not None and sample_fc_feat is not None:
      raw_hc_len = sample_hc_feat.shape[1]
      raw_fc_len = sample_fc_feat.shape[1]
      if lead_time > 0 and raw_hc_len > lead_time:
        # In Multimet, x_d_hindcast has length (seq_length + lead_time = 372) padded with NaNs
        # for the last lead_time (7) steps, while x_d_forecast has length (forecast_overlap = 365)
        # or (seq_length + lead_time = 372) or (lead_time = 7).
        # The hindcast segment is strictly the first (raw_hc_len - lead_time = 365) steps,
        # and the forecast horizon is strictly the final (lead_time = 7) steps.
        hc_len = raw_hc_len - lead_time
        fc_len = min(lead_time, raw_fc_len)
      else:
        hc_len = raw_hc_len
        fc_len = raw_fc_len
    elif sample_hc_feat is not None:
      total_len = sample_hc_feat.shape[1]
      hc_len = max(1, total_len - lead_time)
      fc_len = total_len - hc_len
    else:
      hc_len = getattr(self.cfg, 'seq_length', 365)
      fc_len = lead_time

    # 3. Meteorological provider embeddings (each strictly [B, hc_len, 20] or [B, fc_len, 20])
    hc_met_list = [
        self._calc_dynamic_embedding(
            embedding_network=fc,
            dynamic_data=forward_data.hindcast_features[name][:, :hc_len, :],
            static_embedding=static_embedding,
            target_len=hc_len,
        )
        for name, fc in self.hindcast_embeddings_fc.items()
    ] + [
        self._calc_dynamic_embedding(
            embedding_network=fc,
            dynamic_data=forward_data.hindcast_features[name][:, :hc_len, :],
            static_embedding=static_embedding,
            target_len=hc_len,
        )
        for name, fc in self.shared_embeddings_fc.items()
    ]

    fc_source_dict = (
        forward_data.forecast_features
        if forward_data.forecast_features
        else forward_data.hindcast_features
    )
    fc_met_list = [
        self._calc_dynamic_embedding(
            embedding_network=fc,
            dynamic_data=fc_source_dict[name][:, -fc_len:, :],
            static_embedding=static_embedding,
            target_len=fc_len,
        )
        for name, fc in self.forecast_embeddings_fc.items()
    ] + [
        self._calc_dynamic_embedding(
            embedding_network=fc,
            dynamic_data=fc_source_dict[name][:, -fc_len:, :],
            static_embedding=static_embedding,
            target_len=fc_len,
        )
        for name, fc in self.shared_embeddings_fc.items()
    ]

    total_seq_len = hc_len + fc_len
    q_obs = self._extract_q_obs(
        data=data,
        batch_size=batch_size,
        total_seq_len=total_seq_len,
        dtype=static_embedding.dtype,
        device=static_embedding.device,
    )

    counter_seq = self._build_counter(
        data=data,
        batch_size=batch_size,
        hc_len=hc_len,
        fc_len=fc_len,
        dtype=static_embedding.dtype,
        device=static_embedding.device,
    )

    if h_0 is not None and c_0 is not None:
      hx_init = (
          h_0 if h_0.ndim == 3 else h_0.unsqueeze(0),
          c_0 if c_0.ndim == 3 else c_0.unsqueeze(0),
      )
    else:
      hx_init = self._prepare_initial_state(
          data,
          h_keys=('h_0', 'h_0_hindcast', 'h_0_forecast', 'h_n'),
          c_keys=('c_0', 'c_0_hindcast', 'c_0_forecast', 'c_n'),
      )

    # Allow explicit DA override of hindcast/forecast embeddings
    has_dyn_override = (
        'hindcast_embedding' in overrides or 'forecast_embedding' in overrides
    )

    if self.training and self.ar_teacher_forcing and not has_dyn_override:
      return self._forward_tf(
          q_obs=q_obs,
          static_embedding=static_embedding,
          hc_met_list=hc_met_list,
          fc_met_list=fc_met_list,
          counter_seq=counter_seq,
          hx_init=hx_init,
          hc_len=hc_len,
          fc_len=fc_len,
          return_state_history=return_state_history,
      )
    else:
      return self._forward_ar(
          q_obs=q_obs,
          static_embedding=static_embedding,
          hc_met_list=hc_met_list,
          fc_met_list=fc_met_list,
          counter_seq=counter_seq,
          hx_init=hx_init,
          hc_len=hc_len,
          fc_len=fc_len,
          overrides=overrides,
          return_state_history=return_state_history,
      )

  def _extract_y_hat(self, head_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Extracts the exact expected value y_hat (using calc_cmal_mean for CMAL heads)."""
    from googlehydrology.modelzoo.head import calc_cmal_mean
    if all(k in head_dict for k in ('mu', 'b', 'tau', 'pi')):
      return calc_cmal_mean(
          head_dict['mu'], head_dict['b'], head_dict['tau'], head_dict['pi']
      )
    if 'y_hat' in head_dict:
      return head_dict['y_hat']
    if 'mu' in head_dict and 'pi' in head_dict:
      return torch.sum(head_dict['pi'] * head_dict['mu'], dim=-1, keepdim=True)
    if 'mu' in head_dict:
      return (
          head_dict['mu']
          if head_dict['mu'].ndim == 3
          else head_dict['mu'].unsqueeze(-1)
      )
    raise KeyError(f'Unable to extract y_hat from head keys: {list(head_dict.keys())}')

  def _forward_tf(
      self,
      q_obs: torch.Tensor,
      static_embedding: torch.Tensor,
      hc_met_list: list[torch.Tensor],
      fc_met_list: list[torch.Tensor],
      counter_seq: torch.Tensor | None,
      hx_init: tuple[torch.Tensor, torch.Tensor] | None,
      hc_len: int,
      fc_len: int,
      return_state_history: bool = False,
  ) -> dict[str, torch.Tensor]:
    """Hy2DL Teacher-Forcing training forward pass with independent hc/fc masking & sigmoid noise."""
    batch_size = static_embedding.shape[0]
    total_seq_len = hc_len + fc_len

    # Split observed lagged discharge into hindcast and forecast segments
    q_hc = q_obs[:, :hc_len, :].clone()
    q_fc = q_obs[:, hc_len:total_seq_len, :].clone()

    # Apply Hy2DL sigmoid-scaled multiplicative noise to forecast AR inputs
    if fc_len > 0:
      q_fc = self._add_noise_ar(q_fc)

    # Apply two-level probabilistic masking (nan_seq_prob + nan_step_prob)
    # independently for emb_hc and emb_fc, matching Eduardo's two separate
    # InputLayer instances (self.emb_hc(sample) and self.emb_fc(sample)).
    hc_seq_drop_mask = (
        torch.rand((batch_size, 1, 1), device=q_obs.device) < self.nan_seq_prob
    )
    hc_step_drop = (
        torch.rand((batch_size, hc_len, 1), device=q_obs.device)
        < self.nan_step_prob
    ) | hc_seq_drop_mask

    fc_seq_drop_mask = (
        torch.rand((batch_size, 1, 1), device=q_obs.device) < self.nan_seq_prob
    )
    fc_step_drop = (
        torch.rand((batch_size, fc_len, 1), device=q_obs.device)
        < self.nan_step_prob
    ) | fc_seq_drop_mask

    q_hc = torch.where(hc_step_drop, float('nan'), q_hc)
    if self.ar_forecast_mode == 'masked_all_steps':
      q_fc = torch.full_like(q_fc, float('nan'))
    else:
      q_fc = torch.where(fc_step_drop, float('nan'), q_fc)

    # Embed "ar" provider and aggregate via _masked_mean (nanmean) with met providers
    ar_hc_emb = self._embed_ar_group(
        q_hc, static_embedding, self.ar_hc_embedding_fc
    )
    ar_fc_emb = self._embed_ar_group(
        q_fc, static_embedding, self.ar_fc_embedding_fc
    )

    dyn_hc = self._masked_mean(hc_met_list + [ar_hc_emb])
    dyn_fc = self._masked_mean(fc_met_list + [ar_fc_emb])
    dynamic_emb = torch.cat([dyn_hc, dyn_fc], dim=1)

    expanded_static = static_embedding.unsqueeze(1).expand(
        -1, total_seq_len, -1
    )
    if counter_seq is not None:
      lstm_inputs = torch.cat([expanded_static, dynamic_emb, counter_seq], dim=-1)
    else:
      lstm_inputs = torch.cat([expanded_static, dynamic_emb], dim=-1)

    lstm_output, (h_n, c_n) = self.lstm(lstm_inputs, hx_init)

    if return_state_history:
      h_n_out = [lstm_output[:, t : t + 1, :] for t in range(total_seq_len)]
      c_n_out = [c_n.transpose(0, 1) for _ in range(total_seq_len)]
    else:
      h_n_out = h_n.transpose(0, 1)
      c_n_out = c_n.transpose(0, 1)

    head = self._calc_head(lstm_output)
    head['y_hat'] = self._extract_y_hat(head)

    head['h_n'] = h_n_out
    head['c_n'] = c_n_out
    head['h_n_hindcast'] = h_n_out
    head['c_n_hindcast'] = c_n_out
    head['h_n_forecast'] = h_n_out
    head['c_n_forecast'] = c_n_out
    head['static_embedding'] = static_embedding
    head['hindcast_embedding'] = dyn_hc
    head['forecast_embedding'] = dyn_fc
    return head

  def _forward_ar(
      self,
      q_obs: torch.Tensor,
      static_embedding: torch.Tensor,
      hc_met_list: list[torch.Tensor],
      fc_met_list: list[torch.Tensor],
      counter_seq: torch.Tensor | None,
      hx_init: tuple[torch.Tensor, torch.Tensor] | None,
      hc_len: int,
      fc_len: int,
      overrides: dict[str, torch.Tensor],
      return_state_history: bool = False,
  ) -> dict[str, torch.Tensor]:
    """Hy2DL Autoregressive inference/evaluation rollout across the forecast horizon."""
    total_seq_len = hc_len + fc_len

    # 1. Hindcast phase: parallel pass over [0 : hc_len]
    q_hc = q_obs[:, :hc_len, :].clone()
    hindcast_window_days = self.cfg._cfg.get('ar_hindcast_window_days', None)
    ar_t0_missing = bool(self.cfg._cfg.get('ar_t0_missing', False)) or (
        self.ar_forecast_mode in ('masked', 'masked_t0', 'masked_all_steps')
    )
    if hindcast_window_days is not None:
      w = int(hindcast_window_days)
      if w <= 0:
        # Case 1: No observed streamflow data as input (baseline)
        q_hc = torch.full_like(q_hc, float('nan'))
        ar_t0_missing = True
      elif w < hc_len:
        # Case 3: Streamflow data only for the final `w` (e.g. 14) non-leadtime days.
        # Note: q_obs[:, hc_len] is Q_{t0} (the final non-leadtime day). If ar_t0_missing
        # is False, Q_{t0} is 1 of the `w` days, so keep `w - 1` days in q_hc; otherwise keep `w`.
        hc_keep = w if ar_t0_missing else max(0, w - 1)
        if hc_keep <= 0:
          q_hc = torch.full_like(q_hc, float('nan'))
        else:
          q_hc[:, : hc_len - hc_keep, :] = float('nan')

    ar_hc_emb = self._embed_ar_group(
        q_hc, static_embedding, self.ar_hc_embedding_fc
    )
    if 'hindcast_embedding' in overrides:
      dyn_hc = overrides['hindcast_embedding'][:, :hc_len, :]
    else:
      dyn_hc = self._masked_mean(hc_met_list + [ar_hc_emb])

    expanded_static_hc = static_embedding.unsqueeze(1).expand(-1, hc_len, -1)
    if counter_seq is not None:
      x_hc = torch.cat(
          [expanded_static_hc, dyn_hc, counter_seq[:, :hc_len, :]], dim=-1
      )
    else:
      x_hc = torch.cat([expanded_static_hc, dyn_hc], dim=-1)

    out_hc, (h_t, c_t) = self.lstm(x_hc, hx_init)

    # 2. Forecast phase: step-by-step unrolling over [hc_len : hc_len + fc_len]
    # Exact Hy2DL `MF2LSTM.forward_ar` rollout:
    # - Initialize `ar_t` with `q_obs[:, hc_len : hc_len + 1, :]` (observed discharge Q_{t0}
    #   from the final non-leadtime day).
    # - If `ar_t0_missing` is True (e.g. Case 1 where w<=0, or when Q_{t0} is withheld),
    #   set initial `ar_t = NaN` at step t=0.
    # - At the end of each forecast step t, update `ar_t = y_hat` (`calc_cmal_mean`) so that
    #   steps t=1..6 roll out autoregressively using the model's own predicted discharge!
    if fc_len > 0 and q_obs.shape[1] > hc_len:
      ar_t = q_obs[:, hc_len : hc_len + 1, :].clone()
    else:
      ar_t = q_obs[:, -1:, :].clone()

    if ar_t0_missing:
      ar_t = torch.full_like(ar_t, float('nan'))

    fc_lstm_outs = []
    dyn_fc_steps = []
    static_step = static_embedding.unsqueeze(1)  # [B, 1, 20]

    for t in range(fc_len):
      ar_t_emb = self._embed_ar_group(
          ar_t, static_embedding, self.ar_fc_embedding_fc
      )
      if 'forecast_embedding' in overrides:
        dyn_fc_t = overrides['forecast_embedding'][:, t : t + 1, :]
      else:
        met_step_list = [m[:, t : t + 1, :] for m in fc_met_list]
        dyn_fc_t = self._masked_mean(met_step_list + [ar_t_emb])

      dyn_fc_steps.append(dyn_fc_t)

      if counter_seq is not None:
        c_step = counter_seq[:, hc_len + t : hc_len + t + 1, :]
        x_fc_t = torch.cat([static_step, dyn_fc_t, c_step], dim=-1)
      else:
        x_fc_t = torch.cat([static_step, dyn_fc_t], dim=-1)

      out_fc_t, (h_t, c_t) = self.lstm(x_fc_t, (h_t, c_t))
      fc_lstm_outs.append(out_fc_t)

      if self.ar_forecast_mode in ('masked_all_steps', 't0_only'):
        ar_t = torch.full_like(ar_t, float('nan'))
      else:
        step_head = self._calc_head(out_fc_t)
        q_pred_t = self._extract_y_hat(step_head)
        # Shift predicted discharge y_hat into ar_t for next forecast step
        # (matching Eduardo's Hy2DL forward_ar: ar_t = torch.cat([Q_t, ar_t[:, :, :-1]], dim=2))
        if ar_t.shape[-1] == 1:
          ar_t = q_pred_t
        else:
          ar_t = torch.cat([q_pred_t, ar_t[:, :, :-1]], dim=-1)

    if fc_lstm_outs:
      out_fc = torch.cat(fc_lstm_outs, dim=1)
      lstm_output = torch.cat([out_hc, out_fc], dim=1)
      dyn_fc = torch.cat(dyn_fc_steps, dim=1)
    else:
      lstm_output = out_hc
      dyn_fc = dyn_hc[:, :0, :]

    if return_state_history:
      h_n_out = [lstm_output[:, t : t + 1, :] for t in range(total_seq_len)]
      c_n_out = [c_t.transpose(0, 1) for _ in range(total_seq_len)]
    else:
      h_n_out = h_t.transpose(0, 1)
      c_n_out = c_t.transpose(0, 1)

    head = self._calc_head(lstm_output)
    head['y_hat'] = self._extract_y_hat(head)

    head['h_n'] = h_n_out
    head['c_n'] = c_n_out
    head['h_n_hindcast'] = h_n_out
    head['c_n_hindcast'] = c_n_out
    head['h_n_forecast'] = h_n_out
    head['c_n_forecast'] = c_n_out
    head['static_embedding'] = static_embedding
    head['hindcast_embedding'] = dyn_hc
    head['forecast_embedding'] = dyn_fc
    return head

  def get_assimilation_target_specs(self) -> dict[str, AssimilationTargetSpec]:
    specs = super().get_assimilation_target_specs()
    specs['c_0'] = AssimilationTargetSpec(
        name='c_0',
        is_time_invariant=False,
        is_recurrent=True,
        default_reg_weight_key='regularization_weight',
        default_lr_key='learning_rate',
        extract_baseline_fn=lambda data, outputs: (
            data['c_0']
            if data.get('c_0') is not None
            else torch.zeros_like(outputs.get('c_n'))
        ),
        inject_override_fn=lambda data, t: data.__setitem__('c_0', t),
    )
    specs['h_0'] = AssimilationTargetSpec(
        name='h_0',
        is_time_invariant=False,
        is_recurrent=True,
        default_reg_weight_key='regularization_weight',
        default_lr_key='learning_rate',
        extract_baseline_fn=lambda data, outputs: (
            data['h_0']
            if data.get('h_0') is not None
            else torch.zeros_like(outputs.get('h_n'))
        ),
        inject_override_fn=lambda data, t: data.__setitem__('h_0', t),
    )
    return specs
