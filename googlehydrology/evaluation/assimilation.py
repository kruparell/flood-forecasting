"""Model-agnostic gradient-based Data Assimilation (DA) for hydrological forecasting models.

Responsibility (single, focused):
  1. Run one continuous unassimilated baseline forward pass over the full sequence.
  2. Optimize the configured assimilation components over the historical assimilation
     window, delegating all loss and regularization math to the canonical
     `training/loss.py` and `training/regularization.py` objects.
  3. Run one final clean forward pass and return the NATIVE model output dictionary.

This module must never compute evaluation metrics (NSE/MSE/KGE), never define custom
loss or regularization formulas, and never hardcode model-specific component names.
Metric calculation and distribution sampling are exclusively the job of `tester.py`.
"""

import logging
import re
from typing import Any, Dict, Optional

from googlehydrology.modelzoo.basemodel import BaseModel

try:
    from googlehydrology.modelzoo.head import ensure_y_hat
except (ImportError, AttributeError):
    try:
        import importlib
        import googlehydrology.modelzoo.head as _h
        importlib.reload(_h)
        from googlehydrology.modelzoo.head import ensure_y_hat
    except Exception:
        from googlehydrology.utils.cmal_deterministic import generate_predictions
        def ensure_y_hat(pred: Any, use_median: bool = True) -> Dict[str, Any]:
            if not isinstance(pred, dict):
                return {'y_hat': pred}
            res = dict(pred)
            if 'mu' in res and 'pi' in res and 'b' in res and 'tau' in res:
                if use_median:
                    cmal_summary = generate_predictions(res['mu'], res['b'], res['tau'], res['pi'])
                    res['y_hat'] = cmal_summary[..., 5:6]
                else:
                    b_clamp = torch.clamp(res['b'].float(), min=1e-5)
                    tau_clamp = torch.clamp(res['tau'].float(), min=1e-6, max=1.0 - 1e-6)
                    means = res['mu'].float() + b_clamp * (1 - 2 * tau_clamp) / (tau_clamp * (1 - tau_clamp))
                    pi_norm = res['pi'].float() / torch.sum(res['pi'].float(), dim=-1, keepdim=True)
                    res['y_hat'] = torch.sum(pi_norm * means, dim=-1, keepdim=True)
            elif 'y_hat' not in res and 'mu' in res:
                res['y_hat'] = res['mu'] if res['mu'].ndim == 3 else res['mu'].unsqueeze(-1)
            return res
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.training.regularization import BackgroundEmbeddingRegularization
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.cmal_deterministic import generate_predictions
import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)
logger = LOGGER


def _safe_int(value: Any, default: int = 0) -> int:
  """Best-effort int conversion that tolerates None and mock/sentinel objects."""
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


# Prefixes naming an LSTM *initial* state, as understood by
# `MeanEmbeddingForecastLSTM._prepare_initial_state`.
_INITIAL_STATE_PREFIXES = ('c_0', 'h_0')


def _initial_to_terminal_state_name(comp_name: str) -> Optional[str]:
  """Maps an LSTM initial-state name to the matching terminal-state name.

  `c_0_hindcast` -> `c_n_hindcast`, `h_0_forecast` -> `h_n_forecast`,
  `c_0` -> `c_n`.

  Args:
    comp_name: Assimilation component name.

  Returns:
    The corresponding terminal-state name, or None if `comp_name` does not name
    an initial state.
  """
  for prefix in _INITIAL_STATE_PREFIXES:
    if comp_name == prefix or comp_name.startswith(prefix + '_'):
      return f'{prefix[0]}_n{comp_name[len(prefix):]}'
  return None


def _get_var_lr(lr_cfg: Any, var_name: str) -> float:
  """Retrieves the component-specific learning rate from `lr_cfg`.

  Args:
    lr_cfg: Either a scalar, a sequence, or a mapping of component name to rate.
    var_name: Name of the component whose learning rate is requested.

  Returns:
    The learning rate for `var_name`, falling back to the first configured value.
  """
  if isinstance(lr_cfg, dict):
    if var_name in lr_cfg:
      return float(lr_cfg[var_name])
    # No substring/heuristic name matching: fall back to the first configured rate.
    return float(next(iter(lr_cfg.values())))
  elif isinstance(lr_cfg, (list, tuple)):
    return float(lr_cfg[0])
  else:
    return float(lr_cfg)


class Assimilation(object):
  """Gradient-based data assimilation over a historical window.

  Optimizes the components declared by `AssimilationConfig.assimilation_components`
  and returns the native assimilated model output dictionary.
  """

  def __init__(self, cfg: AssimilationConfig):
    self.cfg = cfg
    self.window = getattr(cfg, 'assimilation_window', 1)
    self.history = getattr(cfg, 'history', 1)
    self.lead_time = getattr(cfg, 'assimilation_lead_time', 0) or getattr(
        cfg, 'lead_time', 0
    )
    self.epochs = getattr(cfg, 'epochs', 10)
    self.targets = getattr(cfg, 'assimilation_targets', [])

    # Boundary timesteps
    self._end_timestep = cfg.seq_length - self.lead_time
    self._start_timestep = max(
        0, self._end_timestep - (self.history * self.window)
    )

    # Public aliases used by callers and tests.
    self.assimilation_window_length = self.window
    self.assimilation_lead_time = self.lead_time
    self.assimilation_start_step = self._start_timestep
    self.assimilation_end_step = self._end_timestep

    if self._end_timestep > cfg.seq_length:
      raise ValueError(
          'Warmup + assimilation period cannot exceed total sequence length.'
      )

    # Component specifications are resolved generically by AssimilationConfig.
    components = getattr(cfg, 'assimilation_components', {}) or {}
    self.components = dict(components) if isinstance(components, dict) else {}

    self._loss_obj = get_loss_obj(cfg)
    reg_terms = get_regularization_obj(cfg=cfg)
    if self.components and not any(
        isinstance(r, BackgroundEmbeddingRegularization) for r in reg_terms
    ):
      reg_terms.append(BackgroundEmbeddingRegularization(cfg=cfg))
    self._loss_obj.set_regularization_terms(reg_terms)

  def _wrap_model_output(self, out: Any) -> Dict[str, Any]:
    """Canonical wrapper: preserves the native model output dict if 'y_hat' is present."""
    if isinstance(out, dict) and 'y_hat' in out:
      return out
    return ensure_y_hat(out, use_median=False)

  def _check_cmal_loss_compatibility(self, model: BaseModel) -> None:
    """Fails fast when `loss: CMAL` is paired with an incompatible model.

    `MaskedCMALLoss` reads `mu`/`b`/`tau`/`pi` and slices them into per-target
    blocks of width `n_distributions`. Because the DA loss is built from the
    `AssimilationConfig` rather than the model config, a stale or defaulted
    `n_distributions` would silently slice the wrong columns and yield a
    plausible-looking but meaningless loss. Likewise, a head that emits no
    mixture parameters would fail deep inside the optimisation loop with an
    opaque KeyError.

    Args:
      model: The model being assimilated.

    Raises:
      ValueError: If the head cannot supply mixture parameters, or if the
        configured `n_distributions` disagrees with the model's.
    """
    if self.cfg.loss.lower() not in ('cmal', 'cmalloss'):
      return

    model_cfg = getattr(model, 'cfg', None)
    head = str(getattr(model_cfg, 'head', '')).lower()
    if head not in ('cmal', 'cmal_deterministic'):
      raise ValueError(
          f'loss=CMAL requires a mixture head that emits mu/b/tau/pi, but the'
          f' model head is {head!r}. Use loss=MSE/NSE/RMSE with this head.'
      )

    model_n = getattr(model_cfg, 'n_distributions', None)
    if model_n is not None and int(model_n) != int(self.cfg.n_distributions):
      raise ValueError(
          'n_distributions mismatch between the assimilation config'
          f' ({self.cfg.n_distributions}) and the model'
          f' ({int(model_n)}). MaskedCMALLoss would slice the wrong columns of'
          ' the head output and report a meaningless loss. Set'
          f' n_distributions: {int(model_n)} in the assimilation config.'
      )

  def _check_forecast_gap(
      self,
      model: BaseModel,
      total_sequence_length: int,
      assimilation_end_step: int,
  ):
    """Fails fast if the assimilation window would overlap the scored forecast horizon.

    This is the sole guarantee that data assimilation never sees observations at or
    after the forecast issue date. A mismatch here means lead-time metrics would be
    computed on days that DA was fitted to, which silently inflates scores.

    Args:
      model: The model being assimilated; its `cfg.lead_time` defines the horizon.
      total_sequence_length: Total number of timesteps in the batch.
      assimilation_end_step: Exclusive end index of the assimilation window.

    Raises:
      ValueError: If the trailing unassimilated gap does not equal the lead time.
    """
    model_lead_time = _safe_int(
        getattr(getattr(model, 'cfg', None), 'lead_time', 0)
    )
    expected_gap = max(
        _safe_int(self.lead_time),
        _safe_int(getattr(self.cfg, 'lead_time', 0)),
        model_lead_time,
    )
    if expected_gap <= 0:
      return

    forecast_gap = total_sequence_length - assimilation_end_step
    if forecast_gap != expected_gap:
      raise ValueError(
          '[DA Forecast Gap Error] Expected a '
          f'{expected_gap}-day unassimilated gap at the end of the sequence for '
          f'forecast dates, but got {forecast_gap} days '
          f'(total_sequence_length={total_sequence_length}, '
          f'assimilation_end_step={assimilation_end_step}, '
          f'assimilation_lead_time={self.lead_time}, '
          f'model_lead_time={model_lead_time}). Set `assimilation_lead_time` to '
          'match the model lead time, otherwise data assimilation would be '
          'fitted on the same days that are scored as forecast skill.'
      )

  def _assimilate_components(
      self,
      model: BaseModel,
      data: Dict[str, torch.Tensor],
      verbose: bool = False,
      **kwargs,
  ) -> Dict[str, Any]:
    """Optimizes the configured components and returns the native model output."""
    active_components = self.components

    # Save original requires_grad, training state, and dropout rates.
    prev_training = model.training
    prev_grad_states = {p: p.requires_grad for p in model.parameters()}
    prev_dropout_p = {
        m: m.p for m in model.modules() if isinstance(m, nn.Dropout)
    }
    prev_rnn_dropout = {
        m: getattr(m, 'dropout', 0.0)
        for m in model.modules()
        if isinstance(m, nn.RNNBase)
    }
    for p in model.parameters():
      p.requires_grad = False

    try:
      observed_discharge = (
          data['y'] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
      )
      total_sequence_length = observed_discharge.shape[1]
      num_target_features = observed_discharge.shape[-1]

      assimilation_end_step = (
          total_sequence_length - self.lead_time
          if self.lead_time > 0
          else min(self._end_timestep, total_sequence_length)
      )
      assimilation_start_step = max(
          0, assimilation_end_step - (self.history * self.window)
      )
      window_length = max(0, assimilation_end_step - assimilation_start_step)
      loss_win_cfg = getattr(self.cfg, 'loss_window', None)
      loss_window_length = min(
          int(loss_win_cfg) if loss_win_cfg is not None else window_length,
          window_length,
      )
      loss_start_step = assimilation_end_step - loss_window_length

      self._check_forecast_gap(
          model, total_sequence_length, assimilation_end_step
      )
      self._check_cmal_loss_compatibility(model)

      default_lr = _get_var_lr(
          self.cfg.learning_rate,
          next(iter(active_components), 'learning_rate'),
      )

      state_anchor = getattr(self.cfg, 'state_anchor', 'sequence_start')
      has_recurrent = any(_initial_to_terminal_state_name(c) is not None for c in active_components)
      use_window_start = state_anchor == 'window_start' and has_recurrent

      # Single full-sequence unassimilated baseline forward pass.
      # Must be generated in eval() mode prior to zeroing dropout/training so CMAL/MLP head stats match native.
      with torch.no_grad():
        if use_window_start:
          data_for_prior = dict(data)
          data_for_prior['return_state_history'] = True
          prior_out = self._wrap_model_output(model(data_for_prior))
        else:
          prior_out = self._wrap_model_output(model(data))

      # Map between Target Timeline (`data['y']`, length `total_sequence_length`,
      # e.g. 365 covering [t0 - 357 .. t0 + 7]) and Model/Prediction Timeline
      # (`prior_out['y_hat']`, `hindcast_embedding`, `forecast_embedding`,
      # `c_n_*`, length `pred_seq_len`, e.g. 372 covering [t0 - 364 .. t0 + 7]).
      # Both timelines end on the exact same forecast date `t0 + lead_time`, so
      # any model-timeline index is shifted by `+lead_offset` relative to `data['y']`.
      pred_y_hat = prior_out.get('y_hat') if isinstance(prior_out, dict) else None
      pred_seq_len = (
          int(pred_y_hat.shape[1])
          if isinstance(pred_y_hat, torch.Tensor) and pred_y_hat.ndim >= 2
          else total_sequence_length
      )
      lead_offset = max(0, pred_seq_len - total_sequence_length)
      pred_end_step = assimilation_end_step + lead_offset
      pred_start_step = assimilation_start_step + lead_offset
      pred_loss_start_step = loss_start_step + lead_offset

      def _get_comp_bounds(comp_name: str, seq_len: int) -> tuple[int, int]:
        """Returns (start_step, end_step) in the component's own timeline."""
        if comp_name != 'y' and seq_len >= pred_end_step:
          return pred_start_step, pred_end_step
        return assimilation_start_step, assimilation_end_step

      # cuDNN requires training mode for RNN backward; dropout is zeroed so that the
      # forward pass stays deterministic and identical to eval().
      model.train(True)
      for m in prev_dropout_p:
        m.p = 0.0
      for m in prev_rnn_dropout:
        m.dropout = 0.0

      baseline_components = {}
      baseline_windows = {}
      for comp_name in active_components:
        base_full = None
        if comp_name in prior_out and prior_out[comp_name] is not None:
          base_full = prior_out[comp_name].detach()
        elif comp_name in data and data[comp_name] is not None:
          base_full = data[comp_name].detach()

        if base_full is None:
          # Third fallback: zero-initialised recurrent states.
          #
          # `c_0_*` / `h_0_*` name the LSTM *initial* states. Those are inputs,
          # not outputs, so they appear in neither `prior_out` nor the input
          # batch. When the key is absent the model falls through to PyTorch's
          # default initial state, which is all zeros -- so an explicit zero
          # tensor is the correct baseline, and a zero perturbation reproduces
          # the unassimilated pass exactly.
          #
          # When state_anchor is 'window_start', the correct baseline is the
          # unassimilated recurrent state entering the assimilation window.
          # A zero perturbation then perfectly reproduces the unassimilated sequence.
          terminal_name = _initial_to_terminal_state_name(comp_name)
          if terminal_name is not None and prior_out.get(terminal_name) is not None:
            t_out = prior_out[terminal_name].detach()
            state_start_idx = (
                pred_start_step
                if (t_out.ndim == 4 and t_out.shape[2] >= pred_end_step)
                else assimilation_start_step
            )
            if use_window_start and state_start_idx > 0 and t_out.ndim == 4:
              # State history dimension: [num_layers, batch, seq_len, hidden_size]
              # t_out[:, :, t, :] is the state AFTER timestep t.
              # State ENTERING state_start_idx is the state AFTER state_start_idx - 1.
              base_full = t_out[:, :, state_start_idx - 1, :].clone()
            else:
              if t_out.ndim == 4:
                # Discard the seq_len dimension to get the initial state shape
                base_full = torch.zeros_like(t_out[:, :, 0, :])
              else:
                base_full = torch.zeros_like(t_out)

        if base_full is not None:
          baseline_components[comp_name] = base_full
          if base_full.ndim == 3:
            c_start, c_end = _get_comp_bounds(comp_name, base_full.shape[1])
            if base_full.shape[1] >= c_end:
              baseline_windows[comp_name] = base_full[
                  :, c_start:c_end, :
              ].clone()
            else:
              baseline_windows[comp_name] = base_full.clone()
          else:
            baseline_windows[comp_name] = base_full.clone()

      # Fail loudly rather than silently optimizing nothing.
      unresolved = [c for c in active_components if c not in baseline_windows]
      if unresolved:
        raise NotImplementedError(
            '[DA Unsupported Component] This assimilation engine could not '
            f'resolve a baseline tensor for {sorted(unresolved)}. Components '
            'must be exposed by the model output dictionary, be present in the '
            'input batch, or name an LSTM initial state (c_0_*/h_0_*) whose '
            'terminal counterpart (c_n_*/h_n_*) the model reports. '
            'Precipitation forcing DA is not supported by this engine yet. '
            f'Resolved components: {sorted(baseline_windows)}.'
        )

      def _inject_components(
          base_dict: Dict[str, Any], comp_map: Dict[str, torch.Tensor]
      ) -> Dict[str, Any]:
        """Splices optimized window tensors back into the full-length sequence."""
        d = dict(base_dict)
        overrides = dict(d.get('assimilation_overrides', {}))
        # When static_embedding is not being modified, cache unassimilated
        # embeddings from prior_out so MeanEmbeddingForecastLSTM.forward does
        # not re-run static/dynamic MLP networks on every optimization epoch.
        if isinstance(prior_out, dict):
          if 'static_embedding' not in comp_map:
            # `hindcast_embedding` / `forecast_embedding` are indexed over the
            # FULL sequence. Injecting them makes
            # MeanEmbeddingForecastLSTM.forward take its override branch and
            # skip the embedding recomputation, which is the only consumer of
            # the window-sliced features. The LSTM would then run the whole
            # sequence and `window_slice` would silently become a no-op: the
            # warmed-up state would be applied at step 0 instead of at the
            # window start, and the prefix splice guard would never match.
            # Under `window_start` we therefore cache only the time-invariant
            # static embedding and let the dynamic embeddings be recomputed on
            # the (much shorter) sliced window.
            if use_window_start and pred_start_step > 0:
              cacheable_keys = ('static_embedding',)
            else:
              cacheable_keys = (
                  'static_embedding',
                  'hindcast_embedding',
                  'forecast_embedding',
              )
            for cached_key in cacheable_keys:
              if (
                  cached_key not in comp_map
                  and cached_key in prior_out
                  and isinstance(prior_out[cached_key], torch.Tensor)
              ):
                cached_t = prior_out[cached_key].detach()
                d[cached_key] = cached_t
                overrides[cached_key] = cached_t

          # When using window_start, we must also cache all non-optimized recurrent
          # states entering the assimilation window so the sliced execution faithfully
          # resumes the unassimilated baseline sequence.
          if use_window_start and pred_start_step > 0:
            for state_prefix in ('c_0_hindcast', 'h_0_hindcast', 'c_0_forecast', 'h_0_forecast', 'c_0', 'h_0'):
              if state_prefix not in comp_map:
                term_key = _initial_to_terminal_state_name(state_prefix)
                if term_key is not None and term_key in prior_out and isinstance(prior_out[term_key], torch.Tensor):
                  t_out = prior_out[term_key].detach()
                  if t_out.ndim == 4:
                    state_start_idx = (
                        pred_start_step
                        if t_out.shape[2] >= pred_end_step
                        else assimilation_start_step
                    )
                    if state_start_idx > 0:
                      cached_state = t_out[:, :, state_start_idx - 1, :].clone()
                      d[state_prefix] = cached_state
                      overrides[state_prefix] = cached_state
        for comp_name, opt_tensor in comp_map.items():
          if (
              comp_name in baseline_components
              and baseline_components[comp_name].ndim == 3
              and opt_tensor.ndim == 3
              and opt_tensor.shape[1] == window_length
          ):
            base_full = baseline_components[comp_name]
            c_start, c_end = _get_comp_bounds(comp_name, base_full.shape[1])
            if base_full.shape[1] >= c_end:
              if use_window_start and pred_start_step > 0 and c_start == pred_start_step:
                # When window_slice=(pred_start_step, pred_end_step) is active,
                # the model executes only over [pred_start_step, end), so a 3D
                # dynamic embedding override must be sliced to [c_start, end).
                spliced = torch.cat(
                    [
                        opt_tensor,
                        base_full[:, c_end:, :],
                    ],
                    dim=1,
                )
              else:
                spliced = torch.cat(
                    [
                        base_full[:, :c_start, :],
                        opt_tensor,
                        base_full[:, c_end:, :],
                    ],
                    dim=1,
                )
            else:
              spliced = opt_tensor
          else:
            spliced = opt_tensor
          d[comp_name] = spliced
          overrides[comp_name] = spliced
        d['assimilation_overrides'] = overrides
        return d

      def _splice_window_prefix(
          out_dict: Dict[str, Any],
      ) -> Dict[str, Any]:
        """Restores the pre-window prefix onto a window-sliced forward output.

        Under `state_anchor: window_start` the model is only run over
        `[pred_start_step, end)` in model-timeline coordinates, so every
        time-indexed output comes back `pred_start_step` steps shorter than
        `prior_out`. Downstream code indexes these tensors on the full prior
        timeline, so the untouched prefix `prior_out[:, :pred_start_step]` is
        prepended.
        """
        spliced_dict = {}
        for k, v in out_dict.items():
          pv = prior_out.get(k) if isinstance(prior_out, dict) else None
          if (
              isinstance(v, torch.Tensor)
              and v.ndim >= 2
              and isinstance(pv, torch.Tensor)
              and pv.ndim >= 2
              and pv.shape[1] > pred_start_step
              and v.shape[1] == pv.shape[1] - pred_start_step
          ):
            spliced_dict[k] = torch.cat(
                [pv[:, :pred_start_step, ...], v], dim=1
            )
          else:
            spliced_dict[k] = v
        return spliced_dict

      parameters_to_optimize = []
      param_groups = []
      optimized_components = {}

      for comp_name, comp_cfg in active_components.items():
        base_win = baseline_windows[comp_name]
        is_finite_base = torch.isfinite(base_win).all().item()
        LOGGER.info(
            '[DA Init] Component %s dtype=%s, device=%s, shape=%s, finite=%s, min=%e, max=%e',
            comp_name,
            base_win.dtype,
            base_win.device,
            list(base_win.shape),
            is_finite_base,
            base_win[torch.isfinite(base_win)].min().item() if is_finite_base else float('nan'),
            base_win[torch.isfinite(base_win)].max().item() if is_finite_base else float('nan'),
        )
        if not is_finite_base:
          LOGGER.warning(
              '[DA Init ERROR] Component %s initialized with NON-FINITE values: NaNs=%d, Infs=%d',
              comp_name,
              torch.isnan(base_win).sum().item(),
              torch.isinf(base_win).sum().item(),
          )
        # Convert to float32 master weights for numerical stability in Adam
        opt_tensor = base_win.clone().detach().float().requires_grad_(True)
        optimized_components[comp_name] = opt_tensor
        parameters_to_optimize.append(opt_tensor)
        # Keep baseline matching float32 for consistent regularization & splicing
        baseline_windows[comp_name] = base_win.float()
        if comp_name in baseline_components:
          baseline_components[comp_name] = baseline_components[comp_name].float()

        comp_lr = comp_cfg.get('lr', comp_cfg.get('learning_rate', default_lr))
        if comp_lr is None:
          comp_lr = default_lr
        param_groups.append({
            'params': [opt_tensor],
            'lr': float(comp_lr),
            'base_lr': float(comp_lr),
        })

      # Last known finite state, used to recover from a poisoned optimizer step.
      last_finite = {
          k: v.detach().clone() for k, v in optimized_components.items()
      }

      if parameters_to_optimize and self.epochs > 0 and window_length > 0:
        optimizer = get_optimizer(parameters_to_optimize, self.cfg)
        optimizer.param_groups.clear()
        for pg in param_groups:
          optimizer.add_param_group(pg)

        drop_factor = getattr(self.cfg, 'learning_rate_drop_factor', 0.75)
        epoch_drop = getattr(self.cfg, 'learning_rate_epoch_drop', 50)

        for epoch in range(self.epochs):
          if epoch_drop > 0 and epoch > 0:
            current_factor = drop_factor ** (epoch // epoch_drop)
            for pg in optimizer.param_groups:
              pg['lr'] = pg['base_lr'] * current_factor

          optimizer.zero_grad()
          step_data = _inject_components(data, optimized_components)
          if use_window_start and pred_start_step > 0:
            step_data['window_slice'] = (pred_start_step, total_sequence_length)
          pred_dict = self._wrap_model_output(model(step_data))
          if use_window_start and pred_start_step > 0:
            pred_dict = _splice_window_prefix(pred_dict)

          # For tolerance and early stopping, use the CMAL median if available, otherwise mean
          with torch.no_grad():
            if (
                'mu' in pred_dict
                and 'b' in pred_dict
                and 'tau' in pred_dict
                and 'pi' in pred_dict
            ):
              cmal_summary = generate_predictions(
                  pred_dict['mu'][:, pred_loss_start_step:pred_end_step, :],
                  pred_dict['b'][:, pred_loss_start_step:pred_end_step, :],
                  pred_dict['tau'][:, pred_loss_start_step:pred_end_step, :],
                  pred_dict['pi'][:, pred_loss_start_step:pred_end_step, :],
              )
              # Index 5 is the 50th percentile (median)
              predicted_loss_window = cmal_summary[..., 5:6]
            else:
              predicted_streamflow = pred_dict['y_hat']
              if predicted_streamflow.ndim == 2:
                predicted_streamflow = predicted_streamflow.unsqueeze(-1)
              predicted_loss_window = predicted_streamflow[
                  :, pred_loss_start_step:pred_end_step, :
              ]

            if predicted_loss_window.ndim == 2:
              predicted_loss_window = predicted_loss_window.unsqueeze(-1)
            if predicted_loss_window.shape[-1] > num_target_features:
              predicted_loss_window = predicted_loss_window[
                  ..., :num_target_features
              ]

          observed_loss_window = observed_discharge[
              :, loss_start_step:assimilation_end_step, :
          ]

          valid_mask = ~torch.isnan(observed_loss_window) & ~torch.isnan(
              predicted_loss_window
          )
          if not valid_mask.any():
            LOGGER.debug(
                'Loss window [%d:%d] contains 0 valid target observations.'
                ' Bypassing gradient update.',
                loss_start_step,
                assimilation_end_step,
            )
            continue

          # Tolerance-based early stopping (e.g. streamflow at t0 within 5% of
          # observed). Because `observed_discharge` is z-score normalized
          # ((Q - mu) / sigma), subtracting the full-sequence minimum `y_zero`
          # recovers Q_obs / sigma >= 0, so `|pred_t0 - obs_t0| / (obs_t0 - y_zero)`
          # equals the physical relative error `|Q_hat - Q_obs| / Q_obs` (with a
          # floor of 0.05*sigma for near-zero baseflow).
          tol = getattr(self.cfg, 'early_stopping_tolerance', None)
          seq_converged = None
          if tol is not None and float(tol) > 0:
            tol_val = float(tol)
            with torch.no_grad():
              y_zero = torch.nan_to_num(
                  observed_discharge, nan=float('inf')
              ).amin(dim=1, keepdim=True)
              y_zero = torch.where(
                  torch.isinf(y_zero), torch.zeros_like(y_zero), y_zero
              )
              pred_t0 = predicted_loss_window[:, -1:, :]
              obs_t0 = observed_loss_window[:, -1:, :]
              valid_t0 = ~torch.isnan(obs_t0) & ~torch.isnan(pred_t0)
              denom_t0 = (obs_t0 - y_zero).clamp(min=0.05)
              rel_err_t0 = torch.abs(pred_t0 - obs_t0) / denom_t0
              # A sequence is converged if its terminal lead-0 step is within tol
              # (or if obs_t0 is NaN, fall back to mean window relative error).
              win_denom = (observed_loss_window - y_zero).clamp(min=0.05)
              win_rel_err = torch.where(
                  valid_mask,
                  torch.abs(predicted_loss_window - observed_loss_window) / win_denom,
                  torch.zeros_like(predicted_loss_window),
              ).amax(dim=(1, 2))
              t0_err_per_seq = torch.where(
                  valid_t0.squeeze(-1).squeeze(-1),
                  rel_err_t0.squeeze(-1).squeeze(-1),
                  win_rel_err,
              )
              seq_converged = t0_err_per_seq <= tol_val
              if bool(seq_converged.all()):
                LOGGER.info(
                    '[DA Early Stop] Epoch %d: all %d sequences within'
                    ' tolerance %.2f%% (max rel err %.2f%%). Stopping early.',
                    epoch,
                    int(seq_converged.numel()),
                    tol_val * 100.0,
                    float(t0_err_per_seq.max().item()) * 100.0,
                )
                break

          # Truncate prediction tensors at `pred_end_step` (issue date t0 on the
          # model timeline) and target tensors at `assimilation_end_step` (issue
          # date t0 on the target timeline) so that `BaseLoss._subset_in_time`
          # (`[:, -loss_window_length:, :]`) aligns the exact same calendar dates
          # `[t0 - loss_window_length + 1 .. t0]` without any forecast leakage.
          pred_for_loss = {
              k: (
                  v[:, :pred_end_step, ...]
                  if (
                      isinstance(v, torch.Tensor)
                      and v.ndim >= 2
                      and v.shape[1] >= pred_end_step
                  )
                  else v
              )
              for k, v in pred_dict.items()
          }

          # Optimize the SAME point estimate that the evaluation scores.
          #
          # The point-estimate losses (MSE/NSE/RMSE) minimize error in
          # `prediction['y_hat']`. For a CMAL head, the model sets that to
          # `sum(pi * mu)` (mean_embedding_forecast_lstm.py), the pi-weighted
          # mean of the mixture LOCATION parameters. It ignores the scale `b`
          # and asymmetry `tau` entirely, so it is neither the conditional mean
          # E[X] nor the median q0.5 -- and `tester.py` reports one of those.
          #
          # Left unaligned the optimizer minimizes one statistic while pushing
          # the scored one the other way. Measured on camels_12451000 at w=1:
          # the t0 error in `sum(pi * mu)` went 0.2233 -> 0.0000 while the
          # median error went 0.2281 -> 0.4229, i.e. 85% WORSE.
          #
          # WHICH statistic is config-driven (`assimilation_point_statistic`)
          # and MUST match `--tester_sample_reduction`. Hardcoding it here
          # would just recreate the same mismatch pointing somewhere else.
          #
          # `generate_predictions` returns [E[X], q0.1 .. q0.9] on the last
          # axis, so E[X] is index 0 and the median q0.5 is index 5.
          #
          # The statistic is recomputed here (rather than reusing the tolerance
          # block above) because that block runs under `torch.no_grad()`.
          if (
              not str(getattr(self.cfg, 'loss', '')).lower().startswith('cmal')
              and all(k in pred_for_loss for k in ('mu', 'b', 'tau', 'pi'))
          ):
            stat = getattr(self.cfg, 'assimilation_point_statistic', 'median')
            stat_index = 0 if str(stat).lower() == 'mixture_mean' else 5
            aligned = generate_predictions(
                pred_for_loss['mu'],
                pred_for_loss['b'],
                pred_for_loss['tau'],
                pred_for_loss['pi'],
            )[..., stat_index : stat_index + 1]
            if aligned.ndim == 2:
              aligned = aligned.unsqueeze(-1)
            if aligned.shape[-1] > num_target_features:
              aligned = aligned[..., :num_target_features]
            # Defence in depth. The quantile search used to emit non-finite
            # values through a `torch.where` overflow; that is fixed at source
            # in `cmal_deterministic._cdf_and_pdf`, but note this guard only
            # inspects FORWARD values. The original failure was NaN in the
            # BACKWARD pass, which this cannot see.
            if torch.isfinite(aligned).all():
              pred_for_loss['y_hat'] = aligned
            else:
              LOGGER.warning(
                  'Epoch %d: CMAL %s non-finite; falling back to the'
                  " model's y_hat for this step (loss/metric alignment"
                  ' skipped).',
                  epoch,
                  stat,
              )

          data_for_loss = {
              k: (
                  v[:, :assimilation_end_step, ...]
                  if (
                      isinstance(v, torch.Tensor)
                      and v.ndim >= 2
                      and v.shape[1] >= assimilation_end_step
                  )
                  else v
              )
              for k, v in step_data.items()
          }
          bs = observed_loss_window.shape[0]
          if (
              'per_basin_target_stds' not in data_for_loss
              and getattr(self.cfg, 'loss', '').lower() == 'nse'
          ):
            # Compute sample std per sequence in the batch so batch_size > 1
            # matches batch_size = 1 identically.
            per_seq_stds = []
            for b_idx in range(bs):
              obs_b = observed_loss_window[b_idx][valid_mask[b_idx]]
              if obs_b.numel() > 1:
                std_b = torch.std(obs_b).clamp(min=1e-3)
              else:
                std_b = torch.tensor(1.0, device=observed_loss_window.device)
              per_seq_stds.append(std_b)
            data_for_loss['per_basin_target_stds'] = torch.stack(
                per_seq_stds
            ).view(bs, 1, 1)
          other_model_data = {
              'optimized_components': optimized_components,
              'baseline_components': baseline_windows,
              'component_weights': {
                  k: v.get('weight', 0.01)
                  for k, v in active_components.items()
              },
          }
          if (
              isinstance(pred_y_hat, torch.Tensor)
              and pred_y_hat.ndim >= 2
              and pred_y_hat.shape[1] >= pred_end_step
          ):
            other_model_data['baseline_y_hat'] = (
                pred_y_hat[:, pred_loss_start_step:pred_end_step, :]
                .detach()
                .float()
            )
          total_loss, loss_terms = self._loss_obj(
              pred_for_loss,
              data_for_loss,
              predict_last_n=loss_window_length,
              other_model_data=other_model_data,
          )

          if not (torch.isfinite(total_loss) and total_loss.requires_grad):
            LOGGER.warning(
                'Non-finite or non-differentiable loss at epoch %d; skipping'
                ' update.',
                epoch,
            )
            continue

          # Remove the 1/bs mean divider across independent batch sequences so
          # each sequence b receives its exact single-sample (batch_size=1) gradient.
          (total_loss * float(bs)).backward()

          # Log gradient statistics and check finiteness before clipping.
          for name, p in zip(active_components.keys(), parameters_to_optimize):
            if p.grad is None:
              LOGGER.warning('[DA Grad] Epoch %d component %s grad is None', epoch, name)
            else:
              g_finite = torch.isfinite(p.grad).all().item()
              g_norm = torch.norm(p.grad).item()
              if not g_finite:
                LOGGER.warning(
                    '[DA Grad ERROR] Epoch %d component %s grad NON-FINITE: NaNs=%d, Infs=%d, norm=%e',
                    epoch,
                    name,
                    torch.isnan(p.grad).sum().item(),
                    torch.isinf(p.grad).sum().item(),
                    g_norm,
                )
              elif epoch == 0:
                LOGGER.info(
                    '[DA Grad Pre-Clip] Epoch 0 component %s grad norm=%e, min=%e, max=%e',
                    name,
                    g_norm,
                    p.grad.min().item(),
                    p.grad.max().item(),
                )

          grads_finite = all(
              p.grad is None or torch.isfinite(p.grad).all()
              for p in parameters_to_optimize
          )
          if not grads_finite:
            LOGGER.warning(
                'Non-finite gradients at epoch %d; skipping optimizer step.',
                epoch,
            )
            optimizer.zero_grad()
            continue

          clip_norm = getattr(self.cfg, 'clip_gradient_norm', 0.0)
          if clip_norm and clip_norm > 0:
            if bs <= 1:
              total_norm = torch.nn.utils.clip_grad_norm_(
                  parameters_to_optimize, clip_norm
              )
            else:
              # Clip each sequence in the batch independently so sequence b is
              # never throttled by sqrt(B) or by outlier sequences in the batch.
              sq_norms = torch.zeros(bs, device=observed_loss_window.device)
              for p in parameters_to_optimize:
                if p.grad is None:
                  continue
                g = p.grad.detach()
                if g.ndim >= 2 and g.shape[0] == bs:
                  sq_norms = sq_norms + g.flatten(1).pow(2).sum(dim=1)
                elif g.ndim == 3 and g.shape[1] == bs:
                  sq_norms = sq_norms + g.permute(1, 0, 2).flatten(1).pow(2).sum(dim=1)
              per_seq_norms = torch.sqrt(sq_norms)
              clip_coef = torch.clamp(
                  clip_norm / (per_seq_norms + 1e-6), max=1.0
              )
              for p in parameters_to_optimize:
                if p.grad is None:
                  continue
                if p.grad.ndim >= 2 and p.grad.shape[0] == bs:
                  view_shape = [bs] + [1] * (p.grad.ndim - 1)
                  p.grad.mul_(clip_coef.view(*view_shape))
                elif p.grad.ndim == 3 and p.grad.shape[1] == bs:
                  p.grad.mul_(clip_coef.view(1, bs, 1))
              total_norm = per_seq_norms.max()
            if epoch == 0:
              LOGGER.info(
                  '[DA Clip] Epoch 0 clip_norm=%f, calculated total_norm=%e',
                  clip_norm,
                  total_norm.item() if isinstance(total_norm, torch.Tensor) else float(total_norm),
              )

          optimizer.step()

          with torch.no_grad():
            if seq_converged is not None and bool(seq_converged.any()):
              for k, t in optimized_components.items():
                if t.ndim >= 2 and t.shape[0] == bs:
                  t[seq_converged] = last_finite[k][seq_converged]
                elif t.ndim == 3 and t.shape[1] == bs:
                  t[:, seq_converged, :] = last_finite[k][:, seq_converged, :]
            non_finite_found = False
            for k, t in optimized_components.items():
              if not torch.isfinite(t).all():
                non_finite_found = True
                LOGGER.warning(
                    '[DA Step ERROR] Epoch %d component %s (dtype=%s) NON-FINITE: NaNs=%d, Infs=%d, min=%e, max=%e (prior baseline finite: %s)',
                    epoch,
                    k,
                    t.dtype,
                    torch.isnan(t).sum().item(),
                    torch.isinf(t).sum().item(),
                    t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float('nan'),
                    t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float('nan'),
                    torch.isfinite(last_finite[k]).all().item(),
                )
              elif epoch == 0:
                LOGGER.info(
                    '[DA Step Success] Epoch 0 component %s (dtype=%s) finite: min=%e, max=%e',
                    k,
                    t.dtype,
                    t.min().item(),
                    t.max().item(),
                )

            if not non_finite_found:
              for k, t in optimized_components.items():
                last_finite[k].copy_(t.detach())

              # Per-epoch convergence trace. DEBUG-gated so it costs nothing in
              # production sweeps; enable with `logging_level: DEBUG`.
              # `displacement` is the L2 distance the component has travelled
              # from its background value, and `rms` normalizes that by the
              # parameter count so components of very different dimension
              # (static_embedding is 20; hindcast_embedding is 20*window) can be
              # compared directly against the scale reported by [DA Init].
              if LOGGER.isEnabledFor(logging.DEBUG):
                if isinstance(loss_terms, dict):
                  terms = ' '.join(
                      f'{k}={float(v):.6e}' for k, v in loss_terms.items()
                  )
                else:
                  terms = f'total={float(total_loss):.6e}'
                LOGGER.debug('[DA Epoch %03d] %s', epoch, terms)
                for k, t in optimized_components.items():
                  delta = (t.detach() - baseline_windows[k]).float()
                  n_el = max(1, delta.numel())
                  LOGGER.debug(
                      '[DA Epoch %03d] %s displacement_l2=%.6e rms=%.6e'
                      ' min=%.6e max=%.6e n_params=%d',
                      epoch,
                      k,
                      torch.norm(delta).item(),
                      (torch.norm(delta).item() / (n_el ** 0.5)),
                      t.min().item(),
                      t.max().item(),
                      n_el,
                  )
            else:
              LOGGER.warning(
                  'Non-finite component values after optimizer step at epoch'
                  ' %d; restoring last finite state and stopping early.',
                  epoch,
              )
              for k, t in optimized_components.items():
                t.copy_(last_finite[k])
              break

      with torch.no_grad():
        for m, p_val in prev_dropout_p.items():
          m.p = p_val
        for m, drop_val in prev_rnn_dropout.items():
          m.dropout = drop_val
        model.eval()

        final_comp_map = {}
        for k, v in optimized_components.items():
          t = v.detach()
          if not torch.isfinite(t).all():
            LOGGER.warning(
                'Component %s is non-finite before the final forward pass;'
                ' using last finite state.',
                k,
            )
            t = last_finite[k]
          final_comp_map[k] = t

        if 'static_embedding' in final_comp_map and (
            'hindcast_embedding' in final_comp_map
            or 'forecast_embedding' in final_comp_map
        ):
          static_only_data = dict(data)
          static_only_data['static_embedding'] = final_comp_map[
              'static_embedding'
          ]
          static_only_data['assimilation_overrides'] = {
              'static_embedding': final_comp_map['static_embedding']
          }
          fresh_out = self._wrap_model_output(model(static_only_data))
          for k in ('hindcast_embedding', 'forecast_embedding'):
            if (
                k in fresh_out
                and isinstance(fresh_out[k], torch.Tensor)
                and k in baseline_components
                and baseline_components[k].ndim == 3
                and fresh_out[k].shape == baseline_components[k].shape
            ):
              _, c_end = _get_comp_bounds(k, baseline_components[k].shape[1])
              baseline_components[k][:, c_end:, :] = (
                  fresh_out[k][:, c_end:, :].detach().float()
              )

        final_data = _inject_components(data, final_comp_map)
        if use_window_start and pred_start_step > 0:
          final_data['window_slice'] = (pred_start_step, total_sequence_length)
        final_out = self._wrap_model_output(model(final_data))
        if use_window_start and pred_start_step > 0:
          final_out = _splice_window_prefix(final_out)

        results = {
            'y_hat': (
                final_out['y_hat'].detach()
                if final_out['y_hat'].ndim == 3
                else final_out['y_hat'].unsqueeze(-1).detach()
            ),
        }
        for key in ['mu', 'b', 'tau', 'pi']:
          if key in final_out and isinstance(final_out[key], torch.Tensor):
            results[key] = final_out[key].detach()
        for comp_name in active_components:
          if comp_name in final_comp_map:
            results[comp_name] = final_comp_map[comp_name].detach()
          elif comp_name in final_out and isinstance(
              final_out[comp_name], torch.Tensor
          ):
            results[comp_name] = final_out[comp_name].detach()

        if not torch.isfinite(results['y_hat']).all():
          LOGGER.error(
              'Assimilated y_hat contains non-finite values after the final'
              ' forward pass. Components: %s.',
              sorted(active_components),
          )

      return results
    finally:
      for p, state in prev_grad_states.items():
        p.requires_grad = state
      for m, p_val in prev_dropout_p.items():
        m.p = p_val
      for m, drop_val in prev_rnn_dropout.items():
        m.dropout = drop_val
      model.train(prev_training)

  def validate_data_structure(self, data: Dict[str, Any]):
    """Validates that required keys exist in the input data dictionary."""
    if 'y' not in data:
      raise KeyError("[DA Validation Error] Missing required key 'y'.")
    if 'x_d_hindcast' not in data and 'x_d' not in data:
      raise KeyError(
          "[DA Validation Error] Batch must contain 'x_d_hindcast' or 'x_d'."
      )

  def check_discharge_timing(
      self, data: Dict[str, Any], verbose: bool = True
  ) -> Dict[str, Any]:
    """Checks for timing mismatches between feature series and targets."""
    diagnostics = {'has_timing_mismatch': False, 'warnings': [], 'details': {}}
    if 'date' in data:
      d = data['date']
      start_date = (
          str(d[0, 0])
          if (isinstance(d, np.ndarray) and d.ndim >= 2)
          else (str(d[0]) if hasattr(d, '__getitem__') else str(d))
      )
      diagnostics['details']['sequence_start_date'] = start_date

    x_d = data.get('x_d', data.get('x_d_hindcast', None))
    x_fc = data.get('x_d_forecast', None)
    y = data.get('y', None)

    if x_d is not None and y is not None:
      x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
      fc_len = 0
      if isinstance(x_fc, dict) and x_fc:
        first_fc = next(iter(x_fc.values()), None)
        if hasattr(first_fc, 'shape') and len(first_fc.shape) >= 2:
          fc_len = int(first_fc.shape[1])
      for feat_name, feat_val in x_d_dict.items():
        match = re.search(r'^(.*)_shift(\d+)$', feat_name)
        if match or 'streamflow' in feat_name or 'discharge' in feat_name:
          shift = int(match.group(2)) if match else 1
          f_tensor = (
              torch.as_tensor(feat_val)
              if not isinstance(feat_val, torch.Tensor)
              else feat_val
          )
          y_tensor = (
              torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y
          )

          # When `x_d_forecast` has length `seq_length + lead_time` (e.g. 372)
          # and `y` has length `seq_length` (e.g. 365), `x_d_hindcast` starts
          # `feat_offset = fc_len - y.shape[1]` (7) steps earlier than `y`.
          feat_offset = (
              max(0, fc_len - int(y_tensor.shape[1]))
              if fc_len > int(y_tensor.shape[1])
              else max(0, int(f_tensor.shape[1]) - int(y_tensor.shape[1]))
          )
          check_t = min(5, y_tensor.shape[1] - 1)
          f_check_t = check_t + feat_offset
          if check_t >= shift and f_check_t < f_tensor.shape[1]:
            val_at_t = (
                f_tensor[0, f_check_t, 0]
                if f_tensor.ndim == 3
                else f_tensor[0, f_check_t]
            )
            y_same = (
                y_tensor[0, check_t, 0]
                if y_tensor.ndim == 3
                else y_tensor[0, check_t]
            )
            y_prev = (
                y_tensor[0, check_t - shift, 0]
                if y_tensor.ndim == 3
                else y_tensor[0, check_t - shift]
            )

            if not (
                torch.isnan(val_at_t)
                or torch.isnan(y_same)
                or torch.isnan(y_prev)
            ):
              if (
                  torch.abs(val_at_t - y_same).item() < 1e-5
                  and torch.abs(val_at_t - y_prev).item() > 1e-4
              ):
                diagnostics['has_timing_mismatch'] = True
                diagnostics['warnings'].append(
                    f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}."
                )
    return diagnostics

  def assimilate(
      self,
      model: BaseModel,
      data: Dict[str, torch.Tensor],
      verbose: bool = False,
      **kwargs,
  ) -> Dict[str, Any]:
    """Runs data assimilation and returns the native assimilated model output.

    Args:
      model: The model to assimilate. Its weights are never modified.
      data: The input batch.
      verbose: If True, also runs the discharge timing diagnostic.
      **kwargs: `check_timing` forces the timing diagnostic.

    Returns:
      The native model output dictionary ('y_hat' plus any distribution
      parameters such as 'mu', 'b', 'tau', 'pi'), left unaltered so that
      `tester.py` owns all sampling and metric computation.

    Raises:
      ValueError: If no assimilation components are configured.
    """
    self.validate_data_structure(data)
    if kwargs.get('check_timing', False) or verbose:
      self.check_discharge_timing(data, verbose=verbose)

    if not self.components:
      raise ValueError(
          '[DA Configuration Error] No assimilation components resolved from '
          f'assimilation_targets={self.targets!r}. Data assimilation would be '
          'a silent no-op.'
      )

    return self._assimilate_components(model, data, verbose=verbose, **kwargs)
