"""Simple Data Assimilation (DA) state updating for hydrological forecasting models."""

import logging
import re
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.head import calc_cmal_mean
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig

logger = logging.getLogger(__name__)


def _copy_data_dict(d: dict) -> dict:
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _copy_data_dict(v)
        elif isinstance(v, torch.Tensor):
            res[k] = v.clone()
        elif isinstance(v, np.ndarray):
            res[k] = v.copy()
        else:
            res[k] = v
    return res


def _infer_seq_len(d: dict) -> int | None:
    """Helper to infer sequence length from target y or 3D feature tensors."""
    if 'y' in d and isinstance(d['y'], (torch.Tensor, np.ndarray)) and d['y'].ndim >= 2:
        return d['y'].shape[1]
    for v in d.values():
        if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 3:
            return v.shape[1]
        elif isinstance(v, dict):
            sub_len = _infer_seq_len(v)
            if sub_len is not None:
                return sub_len
    return None


def _ensure_y_hat(pred: Any) -> Dict[str, Any]:
    """Ensures prediction output dictionary contains 'y_hat'."""
    if not isinstance(pred, dict):
        return {'y_hat': pred}
    res = dict(pred)
    if 'y_hat' not in res:
        if 'mu' in res and 'pi' in res and 'b' in res and 'tau' in res and isinstance(res['mu'], torch.Tensor):
            res['y_hat'] = calc_cmal_mean(res['mu'], res['b'], res['tau'], res['pi'])
        elif 'mu' in res and 'pi' in res:
            res['y_hat'] = torch.sum(res['pi'] * res['mu'], dim=-1, keepdim=True)
        elif 'mu' in res:
            mu = res['mu']
            if isinstance(mu, torch.Tensor):
                if mu.ndim == 3 and mu.shape[-1] > 1:
                    res['y_hat'] = torch.mean(mu, dim=-1, keepdim=True)
                else:
                    res['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)
            elif isinstance(mu, np.ndarray):
                if mu.ndim == 3 and mu.shape[-1] > 1:
                    res['y_hat'] = np.mean(mu, axis=-1, keepdims=True)
                else:
                    res['y_hat'] = mu if mu.ndim == 3 else np.expand_dims(mu, -1)
    return res


def _get_var_lr(lr_cfg: Any, var_name: str) -> float:
    """Retrieves target-specific learning rate from lr_cfg."""
    if isinstance(lr_cfg, dict):
        if var_name in lr_cfg:
            return float(lr_cfg[var_name])
        norm_name = 'c_n' if 'c' in var_name else ('h_n' if 'h' in var_name else var_name)
        if norm_name in lr_cfg:
            return float(lr_cfg[norm_name])
        return float(list(lr_cfg.values())[0])
    elif isinstance(lr_cfg, (list, tuple)):
        return float(lr_cfg[0])
    else:
        return float(lr_cfg)


def _slice_hydrology_batch(d: dict, slice_start: int, slice_end: int) -> dict:
    """Fast, zero-copy slicing of 3D sequence tensors in a googlehydrology batch dict."""
    non_seq_keys = {
        'x_s', 'x_one_hot', 'static_features',
        'c_n', 'h_n', 'c_0', 'h_0',
        'c_0_hindcast', 'h_0_hindcast', 'c_0_forecast', 'h_0_forecast',
        'last_prediction'
    }
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _slice_hydrology_batch(v, slice_start, slice_end)
        elif isinstance(v, torch.Tensor) and k not in non_seq_keys and v.ndim == 3:
            t_len = v.shape[1]
            res[k] = v[:, min(slice_start, t_len):min(slice_end, t_len), :]
        else:
            res[k] = v
    return res


class Assimilation(object):
    """Fast 4D-Var state updating with inline diagnostic logging for MeanEmbeddingForecastLSTM."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self.window = getattr(cfg, 'assimilation_window', 1)
        self.history = getattr(cfg, 'history', 1)
        self.lead_time = getattr(cfg, 'assimilation_lead_time', 0)
        self.epochs = getattr(cfg, 'epochs', 10)
        self.targets = getattr(cfg, 'assimilation_targets', ['c_n'])

        # Boundary timesteps
        self._end_timestep = cfg.seq_length - self.lead_time
        self._start_timestep = max(0, self._end_timestep - (self.history * self.window))

        if self._end_timestep > cfg.seq_length:
            raise ValueError("Warmup + assimilation period cannot exceed total sequence length.")

        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validates that required keys exist in the input data dictionary."""
        if 'y' not in data:
            raise KeyError("[DA Validation Error] Missing required key 'y'.")
        if 'x_d_hindcast' not in data and 'x_d' not in data:
            raise KeyError("[DA Validation Error] Batch must contain 'x_d_hindcast' or 'x_d'.")

    def check_discharge_timing(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Checks for timing mismatches between feature series and targets."""
        diagnostics = {'has_timing_mismatch': False, 'warnings': [], 'details': {}}
        x_d = data.get('x_d', data.get('x_d_hindcast', None))
        y = data.get('y', None)

        if x_d is not None and y is not None:
            x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
            for feat_name, feat_val in x_d_dict.items():
                match = re.search(r'^(.*)_shift(\d+)$', feat_name)
                if match or 'streamflow' in feat_name or 'discharge' in feat_name:
                    shift = int(match.group(2)) if match else 1
                    f_tensor = torch.as_tensor(feat_val) if not isinstance(feat_val, torch.Tensor) else feat_val
                    y_tensor = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y

                    check_t = min(5, y_tensor.shape[1] - 1)
                    if check_t >= shift:
                        val_at_t = f_tensor[0, check_t, 0] if f_tensor.ndim == 3 else f_tensor[0, check_t]
                        y_same = y_tensor[0, check_t, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t]
                        y_prev = y_tensor[0, check_t - shift, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t - shift]

                        if not (torch.isnan(val_at_t) or torch.isnan(y_same) or torch.isnan(y_prev)):
                            if torch.abs(val_at_t - y_same).item() < 1e-5 and torch.abs(val_at_t - y_prev).item() > 1e-4:
                                diagnostics['has_timing_mismatch'] = True
                                diagnostics['warnings'].append(f"TIMING MISMATCH in '{feat_name}' at t={check_t}.")
        return diagnostics

    def _set_model_gradient_mode(self, model: nn.Module):
        """Sets model weights to frozen and keeps eval mode for deterministic evaluation."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # ADD to Assimilation class
    def _parse_target_flags(self) -> Tuple[bool, bool, bool, bool]:
        """Parses self.targets into boolean flags for (c_hc, h_hc, c_fc, h_fc)."""
        opt_c_hc = any(k in self.targets for k in ['c_n_hindcast', 'c_0_hindcast', 'c_hc'])
        opt_h_hc = any(k in self.targets for k in ['h_n_hindcast', 'h_0_hindcast', 'h_hc'])
        opt_c_fc = any(k in self.targets for k in ['c_n_forecast', 'c_0_forecast', 'c_fc', 'c_n', 'c_0'])
        opt_h_fc = any(k in self.targets for k in ['h_n_forecast', 'h_0_forecast', 'h_fc', 'h_n', 'h_0'])
        return opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc

    # REPLACE _extract_states with this
    def _extract_state_tensor(self, state_val: Optional[torch.Tensor], t_idx: int) -> Optional[torch.Tensor]:
        if state_val is None:
            return None
        if state_val.ndim == 4:  # [num_layers, batch, seq_len, hidden_size]
            t_clamp = min(max(0, t_idx), state_val.shape[2] - 1)
            return state_val[:, :, t_clamp, :].detach().clone()
        elif state_val.ndim == 3:  # [num_layers, batch, hidden_size]
            return state_val.detach().clone()
        return None


    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, **kwargs) -> Dict[str, Any]:
        self.validate_data_structure(data)
        if kwargs.get('check_timing', False) or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        self._set_model_gradient_mode(model)

        y_tensor = data['y'] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
        total_len = y_tensor.shape[1]

        a_start = self._start_timestep
        a_end = self._end_timestep

        # In assimilate():
        data_init = dict(data)
        data_init['return_state_history'] = True

        with torch.no_grad():
            full_init = _ensure_y_hat(model(data_init))
            base_y = full_init['y_hat'] if full_init['y_hat'].ndim == 3 else full_init['y_hat'].unsqueeze(-1)

        y_chunks = []
        target_key = self.targets[0] if self.targets else 'c_n'
        lr = _get_var_lr(self.cfg.learning_rate, target_key)


        # =========================================================================
        # PHASE 1: Warmup Phase (0 -> a_start)
        # =========================================================================
        # Extract c and h at t = a_start directly from full_init!
        
        opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc = self._parse_target_flags()

        if a_start > 0:
            y_chunks.append(base_y[:, :a_start, :])
            t_init = a_start - 1

            c_hc_curr = self._extract_state_tensor(full_init.get('c_n_hindcast'), t_init)
            h_hc_curr = self._extract_state_tensor(full_init.get('h_n_hindcast'), t_init)
            c_fc_curr = self._extract_state_tensor(full_init.get('c_n_forecast'), t_init)
            h_fc_curr = self._extract_state_tensor(full_init.get('h_n_forecast'), t_init)
        else:
            # Use 3-D slice at t = 0 if available
            c_hc_curr = self._extract_state_tensor(full_init.get('c_n_hindcast'), 0)
            h_hc_curr = self._extract_state_tensor(full_init.get('h_n_hindcast'), 0)
            c_fc_curr = self._extract_state_tensor(full_init.get('c_n_forecast'), 0)
            h_fc_curr = self._extract_state_tensor(full_init.get('h_n_forecast'), 0)

        if c_hc_curr is None: c_hc_curr = torch.zeros_like(full_init['c_n'])
        if h_hc_curr is None: h_hc_curr = torch.zeros_like(full_init['h_n'])
        if c_fc_curr is None: c_fc_curr = c_hc_curr.clone()
        if h_fc_curr is None: h_fc_curr = h_hc_curr.clone()


        # =========================================================================
        # PHASE 2: Sequential Window 4D-Var Optimization (a_start -> a_end)
        # =========================================================================
        curr_idx = a_start
        for w_idx in range(self.history):
            if curr_idx >= a_end: break
            w_end = min(curr_idx + self.window, a_end)
            win_len = w_end - curr_idx

            # Slice window data
            chunk_data = _slice_hydrology_batch(data, curr_idx, w_end)

            # Determine target optimization flags based on self.targets
            optimize_c = any(k in self.targets for k in ['c_n', 'c_0'])
            optimize_h = any(k in self.targets for k in ['h_n', 'h_0'])

            # Reference states before optimization
            c_hc_opt = c_hc_curr.clone().detach().requires_grad_(True) if opt_c_hc else c_hc_curr.clone().detach()
            h_hc_opt = h_hc_curr.clone().detach().requires_grad_(True) if opt_h_hc else h_hc_curr.clone().detach()
            c_fc_opt = c_fc_curr.clone().detach().requires_grad_(True) if opt_c_fc else c_fc_curr.clone().detach()
            h_fc_opt = h_fc_curr.clone().detach().requires_grad_(True) if opt_h_fc else h_fc_curr.clone().detach()

            opt_vars = [p for p in (c_hc_opt, h_hc_opt, c_fc_opt, h_fc_opt) if p.requires_grad]

            loss_start, loss_final = None, None

            if opt_vars:
                optimizer = get_optimizer(opt_vars, self.cfg)
                for pg in optimizer.param_groups: pg["lr"] = lr

                for epoch in range(self.epochs):
                    optimizer.zero_grad()

                    # Set all 4 states in chunk_data
                    chunk_data['c_0_hindcast'] = c_hc_opt
                    chunk_data['h_0_hindcast'] = h_hc_opt
                    chunk_data['c_0_forecast'] = c_fc_opt
                    chunk_data['h_0_forecast'] = h_fc_opt

                    pred_dict = _ensure_y_hat(model(chunk_data))

                    p_sub = pred_dict['y_hat'][:, :win_len, :]
                    if p_sub.ndim == 2: p_sub = p_sub.unsqueeze(-1)
                    t_sub = chunk_data['y'][:, :win_len, :]

                    mask = ~torch.isnan(t_sub) & ~torch.isnan(p_sub)
                    if mask.any():
                        loss = torch.mean((p_sub[mask] - t_sub[mask]) ** 2)
                        if epoch == 0: loss_start = loss.item()

                        if torch.isfinite(loss):
                            loss.backward()
                            if getattr(self.cfg, 'clip_gradient_norm', 0) > 0:
                                torch.nn.utils.clip_grad_norm_(opt_vars, self.cfg.clip_gradient_norm)
                            optimizer.step()
                            loss_final = loss.item()

            state_checks = [
                ("Hindcast Cell State (c_hc)", c_hc_opt, c_hc_curr, opt_c_hc),
                ("Hindcast Hidden State (h_hc)", h_hc_opt, h_hc_curr, opt_h_hc),
                ("Forecast Cell State (c_fc)", c_fc_opt, c_fc_curr, opt_c_fc),
                ("Forecast Hidden State (h_fc)", h_fc_opt, h_fc_curr, opt_h_fc),
            ]


            # Rollout & State Handoff
            with torch.no_grad():
                chunk_data['c_0_hindcast'] = c_hc_opt.detach()
                chunk_data['h_0_hindcast'] = h_hc_opt.detach()
                chunk_data['c_0_forecast'] = c_fc_opt.detach()
                chunk_data['h_0_forecast'] = h_fc_opt.detach()

                rollout = _ensure_y_hat(model(chunk_data))

                p_roll = rollout['y_hat'][:, :win_len, :]
                if p_roll.ndim == 2: p_roll = p_roll.unsqueeze(-1)
                y_chunks.append(p_roll)

                # Extract terminal states from rollout for next window
                c_hc_curr = self._extract_state_tensor(rollout.get('c_n_hindcast'), win_len - 1)
                h_hc_curr = self._extract_state_tensor(rollout.get('h_n_hindcast'), win_len - 1)
                c_fc_curr = self._extract_state_tensor(rollout.get('c_n_forecast'), win_len - 1)
                h_fc_curr = self._extract_state_tensor(rollout.get('h_n_forecast'), win_len - 1)

            curr_idx = w_end

        # =========================================================================
        # PHASE 3: Post-DA Forecast Horizon (a_end -> total_len)
        # =========================================================================
        if curr_idx < total_len:
            with torch.no_grad():
                fc_data = _slice_hydrology_batch(data, curr_idx, total_len)
                fc_data['c_0_hindcast'] = c_hc_curr
                fc_data['h_0_hindcast'] = h_hc_curr
                fc_data['c_0_forecast'] = c_fc_curr
                fc_data['h_0_forecast'] = h_fc_curr

                fc_pred = _ensure_y_hat(model(fc_data))
                fc_y = fc_pred['y_hat'][:, :(total_len - curr_idx), :]
                if fc_y.ndim == 2: fc_y = fc_y.unsqueeze(-1)

                y_chunks.append(fc_y)

        out_y = torch.cat(y_chunks, dim=1)[:, :total_len, :]

        return {
                    'y_hat': out_y.detach(),
                    'c_n_hindcast': c_hc_curr,
                    'h_n_hindcast': h_hc_curr,
                    'c_n_forecast': c_fc_curr,
                    'h_n_forecast': h_fc_curr,
                    'c_n': c_fc_curr,
                    'h_n': h_fc_curr,
                }