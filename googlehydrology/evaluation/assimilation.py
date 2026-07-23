"""Data Assimilation (DA) state updating for hydrological forecasting models."""

import re
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig


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


def _slice_data_dict(d: dict, slice_end: int = None, slice_start: int = 0, sequence_keys: set = None) -> dict:
    """Copies a data dictionary and slices sequence tensors along dimension 1."""
    res = {}
    non_seq_keys = {'x_s', 'x_one_hot', 'static_features', 'c_n', 'h_n', 'c_0', 'h_0'}
    for k, v in d.items():
        if sequence_keys is not None and k not in sequence_keys:
            if isinstance(v, dict):
                res[k] = _copy_data_dict(v)
            elif isinstance(v, torch.Tensor):
                res[k] = v.clone()
            elif isinstance(v, np.ndarray):
                res[k] = v.copy()
            else:
                res[k] = v
        elif k in non_seq_keys:
            if isinstance(v, dict):
                res[k] = _copy_data_dict(v)
            elif isinstance(v, torch.Tensor):
                res[k] = v.clone()
            elif isinstance(v, np.ndarray):
                res[k] = v.copy()
            else:
                res[k] = v
        elif isinstance(v, dict):
            sub_res = {}
            for key, val in v.items():
                if isinstance(val, torch.Tensor) and val.ndim >= 3 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end, :].clone() if slice_end is not None else val[:, slice_start:, :].clone()
                elif isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end].clone() if slice_end is not None else val[:, slice_start:].clone()
                elif isinstance(val, torch.Tensor) and val.ndim == 1 and len(val) > slice_start:
                    sub_res[key] = val[slice_start:slice_end].clone() if slice_end is not None else val[slice_start:].clone()
                elif isinstance(val, np.ndarray) and val.ndim >= 3 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end, :].copy() if slice_end is not None else val[:, slice_start:, :].copy()
                elif isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end].copy() if slice_end is not None else val[:, slice_start:].copy()
                elif isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > slice_start:
                    sub_res[key] = val[slice_start:slice_end].copy() if slice_end is not None else val[slice_start:].copy()
                elif isinstance(val, torch.Tensor):
                    sub_res[key] = val.clone()
                elif isinstance(val, np.ndarray):
                    sub_res[key] = val.copy()
                else:
                    sub_res[key] = val
            res[k] = sub_res
        elif isinstance(v, torch.Tensor):
            if v.ndim >= 3 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end, :].clone() if slice_end is not None else v[:, slice_start:, :].clone()
            elif v.ndim == 2 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end].clone() if slice_end is not None else v[:, slice_start:].clone()
            elif v.ndim == 1 and len(v) > slice_start:
                res[k] = v[slice_start:slice_end].clone() if slice_end is not None else v[slice_start:].clone()
            else:
                res[k] = v.clone()
        elif isinstance(v, np.ndarray):
            if v.ndim >= 3 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end, :].copy() if slice_end is not None else v[:, slice_start:, :].copy()
            elif v.ndim == 2 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end].copy() if slice_end is not None else v[:, slice_start:].copy()
            elif v.ndim == 1 and len(v) > slice_start:
                res[k] = v[slice_start:slice_end].copy() if slice_end is not None else v[slice_start:].copy()
            else:
                res[k] = v.copy()
        else:
            res[k] = v
    return res


def _detach_and_copy_data_dict(d: dict) -> dict:
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _detach_and_copy_data_dict(v)
        elif isinstance(v, torch.Tensor):
            res[k] = v.detach().clone()
        elif isinstance(v, np.ndarray):
            res[k] = v.copy()
        else:
            res[k] = v
    return res


def _mask_data(v: Any, mask: Any) -> Any:
    if isinstance(v, dict):
        return {key: _mask_data(val, mask) for key, val in v.items()}
    if isinstance(v, torch.Tensor):
        return v[mask]
    if isinstance(v, np.ndarray):
        m = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
        return v[m]
    return v


def _ensure_y_hat(pred: Any) -> Dict[str, Any]:
    """Ensures prediction output dictionary contains 'y_hat', deriving it from 'mu' / 'pi' if necessary."""
    if not isinstance(pred, dict):
        return {'y_hat': pred}
    if 'y_hat' not in pred:
        if 'mu' in pred and 'pi' in pred:
            pred['y_hat'] = torch.sum(pred['pi'] * pred['mu'], dim=-1, keepdim=True)
        elif 'mu' in pred:
            mu = pred['mu']
            if isinstance(mu, torch.Tensor):
                if mu.ndim == 3 and mu.shape[-1] > 1:
                    pred['y_hat'] = torch.mean(mu, dim=-1, keepdim=True)
                else:
                    pred['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)
            elif isinstance(mu, np.ndarray):
                if mu.ndim == 3 and mu.shape[-1] > 1:
                    pred['y_hat'] = np.mean(mu, axis=-1, keepdims=True)
                else:
                    pred['y_hat'] = mu if mu.ndim == 3 else np.expand_dims(mu, -1)
    return pred
"""Data Assimilation (DA) state updating for hydrological forecasting models."""

import re
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig


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


def _slice_data_dict(d: dict, slice_end: int = None, slice_start: int = 0, sequence_keys: set = None) -> dict:
    """Copies a data dictionary and slices sequence tensors along dimension 1."""
    res = {}
    non_seq_keys = {'x_s', 'x_one_hot', 'static_features', 'c_n', 'h_n', 'c_0', 'h_0'}
    for k, v in d.items():
        if sequence_keys is not None and k not in sequence_keys:
            if isinstance(v, dict):
                res[k] = _copy_data_dict(v)
            elif isinstance(v, torch.Tensor):
                res[k] = v.clone()
            elif isinstance(v, np.ndarray):
                res[k] = v.copy()
            else:
                res[k] = v
        elif k in non_seq_keys:
            if isinstance(v, dict):
                res[k] = _copy_data_dict(v)
            elif isinstance(v, torch.Tensor):
                res[k] = v.clone()
            elif isinstance(v, np.ndarray):
                res[k] = v.copy()
            else:
                res[k] = v
        elif isinstance(v, dict):
            sub_res = {}
            for key, val in v.items():
                if isinstance(val, torch.Tensor) and val.ndim >= 3 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end, :].clone() if slice_end is not None else val[:, slice_start:, :].clone()
                elif isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end].clone() if slice_end is not None else val[:, slice_start:].clone()
                elif isinstance(val, torch.Tensor) and val.ndim == 1 and len(val) > slice_start:
                    sub_res[key] = val[slice_start:slice_end].clone() if slice_end is not None else val[slice_start:].clone()
                elif isinstance(val, np.ndarray) and val.ndim >= 3 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end, :].copy() if slice_end is not None else val[:, slice_start:, :].copy()
                elif isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] > slice_start:
                    sub_res[key] = val[:, slice_start:slice_end].copy() if slice_end is not None else val[:, slice_start:].copy()
                elif isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > slice_start:
                    sub_res[key] = val[slice_start:slice_end].copy() if slice_end is not None else val[slice_start:].copy()
                elif isinstance(val, torch.Tensor):
                    sub_res[key] = val.clone()
                elif isinstance(val, np.ndarray):
                    sub_res[key] = val.copy()
                else:
                    sub_res[key] = val
            res[k] = sub_res
        elif isinstance(v, torch.Tensor):
            if v.ndim >= 3 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end, :].clone() if slice_end is not None else v[:, slice_start:, :].clone()
            elif v.ndim == 2 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end].clone() if slice_end is not None else v[:, slice_start:].clone()
            elif v.ndim == 1 and len(v) > slice_start:
                res[k] = v[slice_start:slice_end].clone() if slice_end is not None else v[slice_start:].clone()
            else:
                res[k] = v.clone()
        elif isinstance(v, np.ndarray):
            if v.ndim >= 3 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end, :].copy() if slice_end is not None else v[:, slice_start:, :].copy()
            elif v.ndim == 2 and v.shape[1] > slice_start:
                res[k] = v[:, slice_start:slice_end].copy() if slice_end is not None else v[:, slice_start:].copy()
            elif v.ndim == 1 and len(v) > slice_start:
                res[k] = v[slice_start:slice_end].copy() if slice_end is not None else v[slice_start:].copy()
            else:
                res[k] = v.copy()
        else:
            res[k] = v
    return res


def _detach_and_copy_data_dict(d: dict) -> dict:
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _detach_and_copy_data_dict(v)
        elif isinstance(v, torch.Tensor):
            res[k] = v.detach().clone()
        elif isinstance(v, np.ndarray):
            res[k] = v.copy()
        else:
            res[k] = v
    return res


def _mask_data(v: Any, mask: Any) -> Any:
    if isinstance(v, dict):
        return {key: _mask_data(val, mask) for key, val in v.items()}
    if isinstance(v, torch.Tensor):
        return v[mask]
    if isinstance(v, np.ndarray):
        m = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
        return v[m]
    return v


def _ensure_y_hat(pred: Any) -> Dict[str, Any]:
    """Ensures prediction output dictionary contains 'y_hat'."""
    if not isinstance(pred, dict):
        return {'y_hat': pred}
    if 'y_hat' not in pred:
        if 'mu' in pred and 'pi' in pred:
            pred['y_hat'] = torch.sum(pred['pi'] * pred['mu'], dim=-1, keepdim=True)
        elif 'mu' in pred:
            mu = pred['mu']
            if isinstance(mu, torch.Tensor):
                pred['y_hat'] = mu if mu.ndim == 3 else mu.unsqueeze(-1)
            elif isinstance(mu, np.ndarray):
                pred['y_hat'] = mu if mu.ndim == 3 else np.expand_dims(mu, -1)
    return pred


class Assimilation:
    """Gradient-based state updating for hydrological LSTM / AR-LSTM models."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self._end_timestep = cfg.seq_length - cfg.assimilation_lead_time
        self._start_timestep = max(0, self._end_timestep - self.cfg.assimilation_window)

        if self._end_timestep > cfg.seq_length:
            raise ValueError("The sum of warmup and assimilation periods must not exceed sequence length.")

        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def check_discharge_timing(self, data: Dict[str, torch.Tensor], verbose: bool = True) -> Dict[str, Any]:
        """Checks river discharge timing, date alignment, and autoregressive lag alignment."""
        diagnostics = {
            'has_timing_mismatch': False,
            'warnings': [],
            'details': {}
        }

        if 'date' in data:
            dates = data['date']
            if isinstance(dates, torch.Tensor):
                dates = dates.cpu().numpy()
            sample_dates = dates[0] if dates.ndim > 1 else dates
            seq_len = len(sample_dates)

            diagnostics['details']['sequence_start_date'] = str(sample_dates[0])
            diagnostics['details']['sequence_end_date'] = str(sample_dates[-1])
            diagnostics['details']['start_timestep_date'] = str(sample_dates[self._start_timestep]) if self._start_timestep < seq_len else None
            diagnostics['details']['end_timestep_date'] = str(sample_dates[min(self._end_timestep - 1, seq_len - 1)]) if self._end_timestep <= seq_len else None

        x_d = data.get('x_d', data.get('x_d_hindcast', None))
        y = data.get('y', None)

        if x_d is not None and y is not None:
            x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
            for feat_name, feat_val in x_d_dict.items():
                match = re.search(r'^(.*)_shift(\d+)$', feat_name)
                if match or 'streamflow' in feat_name or 'discharge' in feat_name:
                    shift = int(match.group(2)) if match else 1
                    f_tensor = feat_val if isinstance(feat_val, torch.Tensor) else (torch.from_numpy(feat_val) if isinstance(feat_val, np.ndarray) else None)
                    y_tensor = y if isinstance(y, torch.Tensor) else (torch.from_numpy(y) if isinstance(y, np.ndarray) else None)

                    if f_tensor is None or y_tensor is None:
                        continue

                    check_t = min(self._start_timestep + 5, y_tensor.shape[1] - 1)
                    if check_t >= shift:
                        val_at_t = f_tensor[0, check_t, 0] if f_tensor.ndim == 3 else f_tensor[0, check_t]
                        y_same = y_tensor[0, check_t, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t]
                        y_prev = y_tensor[0, check_t - shift, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t - shift]

                        if not (torch.isnan(val_at_t) or torch.isnan(y_same) or torch.isnan(y_prev)):
                            diff_prev = torch.abs(val_at_t - y_prev).item()
                            diff_same = torch.abs(val_at_t - y_same).item()
                            if diff_same < 1e-5 and diff_prev > 1e-4:
                                msg = (f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}: "
                                       f"matches same-day target y[t] (diff: {diff_same:.6f}) instead of y[t-{shift}].")
                                diagnostics['has_timing_mismatch'] = True
                                diagnostics['warnings'].append(msg)

        return diagnostics

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validates that batch data matches the expected dictionary structure for assimilation."""
        if 'y' not in data:
            raise KeyError("[DA Validation Error] Required key 'y' missing from input data dictionary.")
        if 'x_d_hindcast' not in data and 'x_d' not in data:
            raise KeyError("[DA Validation Error] Input data batch must contain either 'x_d_hindcast' or 'x_d'.")

        for key in ['x_d', 'x_d_hindcast', 'x_d_forecast']:
            if key in data:
                if not isinstance(data[key], (dict, torch.Tensor, np.ndarray)):
                    raise TypeError(f"[DA Validation Error] '{key}' must be dict, torch.Tensor, or np.ndarray.")

        y = data['y']
        if not isinstance(y, (torch.Tensor, np.ndarray)):
            raise TypeError(f"[DA Validation Error] Target 'y' must be Tensor or ndarray, got {type(y)}.")
        if y.ndim != 3:
            raise ValueError(f"[DA Validation Error] Target 'y' must be a 3D tensor, got shape {y.shape}.")

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, check_timing: bool = True):
        """Performs state updating across the assimilation window."""
        self.validate_data_structure(data)
        if check_timing or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # --- 1. Warmup Pass ---
        with torch.no_grad():
            if self._start_timestep > 0:
                warmup_data = _slice_data_dict(data, slice_start=0, slice_end=self._start_timestep)
                warmup_pred = _ensure_y_hat(model(warmup_data))
                c_start = warmup_pred.get('c_n', warmup_pred.get('c_0', warmup_data.get('c_n', warmup_data.get('c_0'))))
                h_start = warmup_pred.get('h_n', warmup_pred.get('h_0', warmup_data.get('h_n', warmup_data.get('h_0'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

                assim_data_base = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
                if c_start is not None:
                    assim_data_base['c_n'] = c_start
                    assim_data_base['c_0'] = c_start
                if h_start is not None:
                    assim_data_base['h_n'] = h_start
                    assim_data_base['h_0'] = h_start

                base_win_pred = _ensure_y_hat(model(assim_data_base))

                seq_len = data['y'].shape[1] if ('y' in data and hasattr(data['y'], 'shape') and data['y'].ndim >= 2) else getattr(self.cfg, 'seq_length', None)
                if seq_len is not None and seq_len > self._start_timestep and warmup_pred['y_hat'].ndim >= 2 and warmup_pred['y_hat'].shape[1] >= self._start_timestep:
                    t_act = seq_len - self._start_timestep
                    warm_y_base = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                    base_win_y = base_win_pred['y_hat'][:, -t_act:, :] if (base_win_pred['y_hat'].ndim >= 2 and base_win_pred['y_hat'].shape[1] >= t_act) else base_win_pred['y_hat']
                    full_base_pred = {'y_hat': torch.cat([warm_y_base, base_win_y], dim=1)}
                else:
                    full_base_pred = base_win_pred
            else:
                full_base_pred = _ensure_y_hat(model(data))
                c_start = data.get('c_n', data.get('c_0', full_base_pred.get('c_0', full_base_pred.get('c_n'))))
                h_start = data.get('h_n', data.get('h_0', full_base_pred.get('h_0', full_base_pred.get('h_n'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

        lr_val = self.cfg.learning_rate
        lr = float(lr_val.get('c_n', list(lr_val.values())[0]) if isinstance(lr_val, dict) else (lr_val[0] if isinstance(lr_val, list) else lr_val))
        bg_weight = getattr(self.cfg, 'bg_regularization_weight', 0.005)
        n_hindcast = getattr(self.cfg, 'predict_n_hindcast', getattr(getattr(model, 'cfg', None), 'predict_n_hindcast', 5))
        use_per_step = getattr(self.cfg, 'use_per_step_updates', True)

        # --- 2. State Optimization ---
        seq_da_y_hats = []
        c_n, h_n = c_start, h_start

        # === CHECKPOINT 1: Start Timestep & Warmup ===
        print(f"[DA Diagnostic 1] _start_timestep: {self._start_timestep} | _end_timestep: {self._end_timestep}", flush=True)
        if 'warmup_pred' in locals():
            print(f"[DA Diagnostic 1] warmup_pred shape: {warmup_pred['y_hat'].shape}", flush=True)


        if use_per_step and c_start is not None:
            # === BLOCK-BASED ASSIMILATION LOOP ===
            W = getattr(self.cfg, 'assimilation_window', 1)
            curr_c = c_start.clone().detach()
            curr_h = h_start.clone().detach() if h_start is not None else None

            for t in range(self._start_timestep, self._end_timestep, W):
                block_end = min(t + W, self._end_timestep)
                step_data = _slice_data_dict(data, slice_start=t, slice_end=block_end)
                target_y = step_data['y'][:, :, 0:1] if step_data['y'].ndim == 3 else step_data['y'].unsqueeze(-1)

                c_opt = curr_c.clone().detach().requires_grad_(True)
                h_opt = curr_h.clone().detach().requires_grad_(True) if curr_h is not None else None

                opt_params = [c_opt] if h_opt is None else [c_opt, h_opt]
                optimizer = get_optimizer(opt_params, self.cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    step_data['c_0'] = c_opt
                    step_data['c_n'] = c_opt
                    if h_opt is not None:
                        step_data['h_0'] = h_opt
                        step_data['h_n'] = h_opt

                    pred = _ensure_y_hat(model(step_data))
                    y_hat = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)

                    valid_mask = ~torch.isnan(target_y) & ~torch.isnan(y_hat)
                    if valid_mask.any():
                        loss_obs = torch.mean((y_hat[valid_mask] - target_y[valid_mask]) ** 2)
                        loss_bg = bg_weight * torch.mean((c_opt - curr_c) ** 2)
                        if h_opt is not None:
                            loss_bg += bg_weight * torch.mean((h_opt - curr_h) ** 2)

                        loss = loss_obs + loss_bg
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                            if hasattr(self.cfg, 'clip_gradient_norm') and self.cfg.clip_gradient_norm and self.cfg.clip_gradient_norm > 0:
                                torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.clip_gradient_norm)
                            optimizer.step()

                with torch.no_grad():
                    step_data['c_0'] = c_opt
                    step_data['c_n'] = c_opt
                    if h_opt is not None:
                        step_data['h_0'] = h_opt
                        step_data['h_n'] = h_opt

                    final_step_pred = _ensure_y_hat(model(step_data))
                    curr_c = final_step_pred.get('c_n', final_step_pred.get('c_0', c_opt)).detach().clone()
                    if curr_h is not None:
                        curr_h = final_step_pred.get('h_n', final_step_pred.get('h_0', h_opt)).detach().clone()

                    seq_da_y_hats.append(final_step_pred['y_hat'])

            da_pred_window_y = torch.cat(seq_da_y_hats, dim=1) if len(seq_da_y_hats) > 0 else full_base_pred['y_hat']
            c_n, h_n = curr_c, curr_h

            # === CHECKPOINT 2: Assimilation Loop Output ===
            print(f"[DA Diagnostic 2] Number of blocks in seq_da_y_hats: {len(seq_da_y_hats)}", flush=True)
            print(f"[DA Diagnostic 2] da_pred_window_y shape (before Section 3): {da_pred_window_y.shape}", flush=True)
        else:
            # === FALLBACK: WINDOWED 4D-VAR ===
            if self._start_timestep > 0:
                assim_data = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
            else:
                assim_data = dict(data)

            c_n_opt = c_start.clone().detach().requires_grad_(True) if c_start is not None else None
            h_n_opt = h_start.clone().detach().requires_grad_(True) if h_start is not None else None
            opt_params = [p for p in [c_n_opt, h_n_opt] if p is not None]

            if len(opt_params) > 0 and self.cfg.epochs > 0 and lr > 0:
                optimizer = get_optimizer(opt_params, self.cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    if c_n_opt is not None:
                        assim_data['c_n'] = c_n_opt
                        assim_data['c_0'] = c_n_opt
                    if h_n_opt is not None:
                        assim_data['h_n'] = h_n_opt
                        assim_data['h_0'] = h_n_opt

                    pred = _ensure_y_hat(model(assim_data))
                    pred_y = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)
                    y_target_window = assim_data['y'][:, :, 0:1] if assim_data['y'].ndim == 3 else assim_data['y'].unsqueeze(-1)

                    valid_mask = ~torch.isnan(y_target_window) & ~torch.isnan(pred_y)
                    loss_obs = torch.mean((pred_y[valid_mask] - y_target_window[valid_mask]) ** 2) if valid_mask.any() else torch.tensor(0.0)
                    loss_bg = bg_weight * torch.mean((c_n_opt - c_start) ** 2) if c_n_opt is not None else torch.tensor(0.0)

                    loss = loss_obs + loss_bg
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        optimizer.step()

            with torch.no_grad():
                if c_n_opt is not None:
                    assim_data['c_n'] = c_n_opt
                    assim_data['c_0'] = c_n_opt
                if h_n_opt is not None:
                    assim_data['h_n'] = h_n_opt
                    assim_data['h_0'] = h_n_opt
                da_pred_window = _ensure_y_hat(model(assim_data))
                da_pred_window_y = da_pred_window['y_hat']
                c_n, h_n = c_n_opt, h_n_opt

        # --- 3. Reconstruct Full Prediction Trajectory & Evaluate Metrics ---
        with torch.no_grad():
            if self._start_timestep > 0 and 'warmup_pred' in locals():
                warm_y_hat = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                if warm_y_hat.ndim == 2:
                    warm_y_hat = warm_y_hat.unsqueeze(-1)
                if da_pred_window_y.ndim == 2:
                    da_pred_window_y = da_pred_window_y.unsqueeze(-1)
                da_y_hat = torch.cat([warm_y_hat, da_pred_window_y], dim=1)
            else:
                da_y_hat = da_pred_window_y

            expected_seq_len = data['y'].shape[1]
            if da_y_hat.shape[1] != expected_seq_len:
                da_y_hat = da_y_hat[:, :expected_seq_len, :]

            # --- Metrics Calculation ---
            y_target_full = data['y'][:, :, 0:1] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            pred_base_y = full_base_pred['y_hat'][:, :, 0:1] if full_base_pred['y_hat'].ndim == 3 else full_base_pred['y_hat'].unsqueeze(-1)

            obs_full = y_target_full[0, :, 0].cpu().numpy()
            base_full = pred_base_y[0, :, 0].detach().cpu().numpy()
            da_full = da_y_hat[0, :, 0].detach().cpu().numpy()

            def _metrics(o, s):
                if len(o) == 0 or len(s) == 0 or len(o) != len(s):
                    return {'NSE': float('nan'), 'RMSE': float('nan'), 'Pearson-r': float('nan')}
                var_o = float(np.var(o))
                if var_o < 1e-8:
                    var_o = 1.0
                nse = float(1.0 - np.mean((s - o) ** 2) / var_o)
                rmse = float(np.sqrt(np.mean((s - o) ** 2)))
                std_s, std_o = float(np.std(s)), float(np.std(o))
                r = float(np.corrcoef(o, s)[0, 1]) if (std_s > 1e-8 and std_o > 1e-8) else 0.0
                return {'NSE': nse, 'RMSE': rmse, 'Pearson-r': r}

            n_h_eval = min(n_hindcast, len(obs_full))
            m_base_hind = _metrics(obs_full[:n_h_eval], base_full[:n_h_eval])
            m_da_hind = _metrics(obs_full[:n_h_eval], da_full[:n_h_eval])

            min_len = min(len(obs_full), len(base_full), len(da_full))
            m_base_fc = _metrics(obs_full[n_h_eval:min_len], base_full[n_h_eval:min_len]) if min_len > n_h_eval else {}
            m_da_fc = _metrics(obs_full[n_h_eval:min_len], da_full[n_h_eval:min_len]) if min_len > n_h_eval else {}

            return {
                'y_hat': da_y_hat.detach(),
                'c_n': c_n.detach() if isinstance(c_n, torch.Tensor) else None,
                'h_n': h_n.detach() if isinstance(h_n, torch.Tensor) else None,
                'hindcast_metrics_pre': m_base_hind,
                'hindcast_metrics_post': m_da_hind,
                'forecast_metrics_pre': m_base_fc,
                'forecast_metrics_post': m_da_fc,
            }


class Assimilation:
    """Gradient-based state updating for hydrological LSTM / AR-LSTM models."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self._end_timestep = cfg.seq_length - cfg.assimilation_lead_time
        self._start_timestep = max(0, self._end_timestep - (cfg.history * cfg.assimilation_window))

        if self._end_timestep > cfg.seq_length:
            raise ValueError("The sum of warmup and assimilation periods must not exceed sequence length.")

        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def check_discharge_timing(self, data: Dict[str, torch.Tensor], verbose: bool = True) -> Dict[str, Any]:
        """Checks river discharge timing, date alignment, and autoregressive lag alignment."""
        diagnostics = {
            'has_timing_mismatch': False,
            'warnings': [],
            'details': {}
        }

        if 'date' in data:
            dates = data['date']
            if isinstance(dates, torch.Tensor):
                dates = dates.cpu().numpy()
            sample_dates = dates[0] if dates.ndim > 1 else dates
            seq_len = len(sample_dates)

            diagnostics['details']['sequence_start_date'] = str(sample_dates[0])
            diagnostics['details']['sequence_end_date'] = str(sample_dates[-1])
            diagnostics['details']['start_timestep_date'] = str(sample_dates[self._start_timestep]) if self._start_timestep < seq_len else None
            diagnostics['details']['end_timestep_date'] = str(sample_dates[min(self._end_timestep - 1, seq_len - 1)]) if self._end_timestep <= seq_len else None

        x_d = data.get('x_d', data.get('x_d_hindcast', None))
        y = data.get('y', None)

        if x_d is not None and y is not None:
            x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
            for feat_name, feat_val in x_d_dict.items():
                match = re.search(r'^(.*)_shift(\d+)$', feat_name)
                if match or 'streamflow' in feat_name or 'discharge' in feat_name:
                    shift = int(match.group(2)) if match else 1
                    f_tensor = feat_val if isinstance(feat_val, torch.Tensor) else (torch.from_numpy(feat_val) if isinstance(feat_val, np.ndarray) else None)
                    y_tensor = y if isinstance(y, torch.Tensor) else (torch.from_numpy(y) if isinstance(y, np.ndarray) else None)

                    if f_tensor is None or y_tensor is None:
                        continue

                    check_t = min(self._start_timestep + 5, y_tensor.shape[1] - 1)
                    if check_t >= shift:
                        val_at_t = f_tensor[0, check_t, 0] if f_tensor.ndim == 3 else f_tensor[0, check_t]
                        y_same = y_tensor[0, check_t, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t]
                        y_prev = y_tensor[0, check_t - shift, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t - shift]

                        if not (torch.isnan(val_at_t) or torch.isnan(y_same) or torch.isnan(y_prev)):
                            diff_prev = torch.abs(val_at_t - y_prev).item()
                            diff_same = torch.abs(val_at_t - y_same).item()
                            if diff_same < 1e-5 and diff_prev > 1e-4:
                                msg = (f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}: "
                                       f"matches same-day target y[t] (diff: {diff_same:.6f}) instead of y[t-{shift}].")
                                diagnostics['has_timing_mismatch'] = True
                                diagnostics['warnings'].append(msg)

        return diagnostics

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validates that batch data matches the expected dictionary structure for assimilation."""
        if 'y' not in data:
            raise KeyError("[DA Validation Error] Required key 'y' missing from input data dictionary.")
        if 'x_d_hindcast' not in data and 'x_d' not in data:
            raise KeyError("[DA Validation Error] Input data batch must contain either 'x_d_hindcast' or 'x_d'.")

        for key in ['x_d', 'x_d_hindcast', 'x_d_forecast']:
            if key in data:
                if not isinstance(data[key], (dict, torch.Tensor, np.ndarray)):
                    raise TypeError(f"[DA Validation Error] '{key}' must be dict, torch.Tensor, or np.ndarray.")

        y = data['y']
        if not isinstance(y, (torch.Tensor, np.ndarray)):
            raise TypeError(f"[DA Validation Error] Target 'y' must be Tensor or ndarray, got {type(y)}.")
        if y.ndim != 3:
            raise ValueError(f"[DA Validation Error] Target 'y' must be a 3D tensor, got shape {y.shape}.")

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, check_timing: bool = True):
        """Performs state updating across the assimilation window."""
        self.validate_data_structure(data)
        if check_timing or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # --- 1. Warmup Pass ---
        with torch.no_grad():
            if self._start_timestep > 0:
                warmup_data = _slice_data_dict(data, slice_start=0, slice_end=self._start_timestep)
                warmup_pred = _ensure_y_hat(model(warmup_data))
                c_start = warmup_pred.get('c_n', warmup_pred.get('c_0', warmup_data.get('c_n', warmup_data.get('c_0'))))
                h_start = warmup_pred.get('h_n', warmup_pred.get('h_0', warmup_data.get('h_n', warmup_data.get('h_0'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

                assim_data_base = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
                if c_start is not None:
                    assim_data_base['c_n'] = c_start
                    assim_data_base['c_0'] = c_start
                if h_start is not None:
                    assim_data_base['h_n'] = h_start
                    assim_data_base['h_0'] = h_start

                base_win_pred = _ensure_y_hat(model(assim_data_base))

                seq_len = data['y'].shape[1] if ('y' in data and hasattr(data['y'], 'shape') and data['y'].ndim >= 2) else getattr(self.cfg, 'seq_length', None)
                if seq_len is not None and seq_len > self._start_timestep and warmup_pred['y_hat'].ndim >= 2 and warmup_pred['y_hat'].shape[1] >= self._start_timestep:
                    t_act = seq_len - self._start_timestep
                    warm_y_base = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                    base_win_y = base_win_pred['y_hat'][:, -t_act:, :] if (base_win_pred['y_hat'].ndim >= 2 and base_win_pred['y_hat'].shape[1] >= t_act) else base_win_pred['y_hat']
                    full_base_pred = {'y_hat': torch.cat([warm_y_base, base_win_y], dim=1)}
                else:
                    full_base_pred = base_win_pred
            else:
                full_base_pred = _ensure_y_hat(model(data))
                c_start = data.get('c_n', data.get('c_0', full_base_pred.get('c_0', full_base_pred.get('c_n'))))
                h_start = data.get('h_n', data.get('h_0', full_base_pred.get('h_0', full_base_pred.get('h_n'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

        lr_val = self.cfg.learning_rate
        lr = float(lr_val.get('c_n', list(lr_val.values())[0]) if isinstance(lr_val, dict) else (lr_val[0] if isinstance(lr_val, list) else lr_val))
        bg_weight = getattr(self.cfg, 'bg_regularization_weight', 0.005)
        n_hindcast = getattr(self.cfg, 'predict_n_hindcast', getattr(getattr(model, 'cfg', None), 'predict_n_hindcast', 5))
        use_per_step = getattr(self.cfg, 'use_per_step_updates', True)

        # --- 2. State Optimization ---
        seq_da_y_hats = []
        c_n, h_n = c_start, h_start

        if use_per_step and c_start is not None:
            # === BLOCK-BASED ASSIMILATION LOOP ===
            W = getattr(self.cfg, 'assimilation_window', 1)
            curr_c = c_start.clone().detach()
            curr_h = h_start.clone().detach() if h_start is not None else None

            for t in range(self._start_timestep, self._end_timestep, W):
                block_end = min(t + W, self._end_timestep)
                step_data = _slice_data_dict(data, slice_start=t, slice_end=block_end)
                target_y = step_data['y'][:, :, 0:1] if step_data['y'].ndim == 3 else step_data['y'].unsqueeze(-1)

                c_opt = curr_c.clone().detach().requires_grad_(True)
                h_opt = curr_h.clone().detach().requires_grad_(True) if curr_h is not None else None

                opt_params = [c_opt] if h_opt is None else [c_opt, h_opt]
                optimizer = get_optimizer(opt_params, self.cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    step_data['c_0'] = c_opt
                    step_data['c_n'] = c_opt
                    if h_opt is not None:
                        step_data['h_0'] = h_opt
                        step_data['h_n'] = h_opt

                    pred = _ensure_y_hat(model(step_data))
                    y_hat = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)

                    valid_mask = ~torch.isnan(target_y) & ~torch.isnan(y_hat)
                    if valid_mask.any():
                        loss_obs = torch.mean((y_hat[valid_mask] - target_y[valid_mask]) ** 2)
                        loss_bg = bg_weight * torch.mean((c_opt - curr_c) ** 2)
                        if h_opt is not None:
                            loss_bg += bg_weight * torch.mean((h_opt - curr_h) ** 2)

                        loss = loss_obs + loss_bg
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                            if hasattr(self.cfg, 'clip_gradient_norm') and self.cfg.clip_gradient_norm and self.cfg.clip_gradient_norm > 0:
                                torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.clip_gradient_norm)
                            optimizer.step()

                with torch.no_grad():
                    step_data['c_0'] = c_opt
                    step_data['c_n'] = c_opt
                    if h_opt is not None:
                        step_data['h_0'] = h_opt
                        step_data['h_n'] = h_opt

                    final_step_pred = _ensure_y_hat(model(step_data))
                    curr_c = final_step_pred.get('c_n', final_step_pred.get('c_0', c_opt)).detach().clone()
                    if curr_h is not None:
                        curr_h = final_step_pred.get('h_n', final_step_pred.get('h_0', h_opt)).detach().clone()

                    seq_da_y_hats.append(final_step_pred['y_hat'])


            # Free-run forecast over the lead-time window [356 -> 366]
            with torch.no_grad():
                seq_len = data['y'].shape[1]
                if seq_len > self._end_timestep:
                    forecast_data = _slice_data_dict(data, slice_start=self._end_timestep, slice_end=None)
                    forecast_data['c_0'] = curr_c
                    forecast_data['c_n'] = curr_c
                    if curr_h is not None:
                        forecast_data['h_0'] = curr_h
                        forecast_data['h_n'] = curr_h
                    fc_pred = _ensure_y_hat(model(forecast_data))
                    seq_da_y_hats.append(fc_pred['y_hat'])

            da_pred_window_y = torch.cat(seq_da_y_hats, dim=1)
            # da_pred_window_y = torch.cat(seq_da_y_hats, dim=1) if len(seq_da_y_hats) > 0 else full_base_pred['y_hat']

            c_n, h_n = curr_c, curr_h

        else:
            # === FALLBACK: WINDOWED 4D-VAR ===
            if self._start_timestep > 0:
                assim_data = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
            else:
                assim_data = dict(data)

            c_n_opt = c_start.clone().detach().requires_grad_(True) if c_start is not None else None
            h_n_opt = h_start.clone().detach().requires_grad_(True) if h_start is not None else None
            opt_params = [p for p in [c_n_opt, h_n_opt] if p is not None]

            if len(opt_params) > 0 and self.cfg.epochs > 0 and lr > 0:
                optimizer = get_optimizer(opt_params, self.cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    if c_n_opt is not None:
                        assim_data['c_n'] = c_n_opt
                        assim_data['c_0'] = c_n_opt
                    if h_n_opt is not None:
                        assim_data['h_n'] = h_n_opt
                        assim_data['h_0'] = h_n_opt

                    pred = _ensure_y_hat(model(assim_data))
                    pred_y = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)
                    y_target_window = assim_data['y'][:, :, 0:1] if assim_data['y'].ndim == 3 else assim_data['y'].unsqueeze(-1)

                    valid_mask = ~torch.isnan(y_target_window) & ~torch.isnan(pred_y)
                    loss_obs = torch.mean((pred_y[valid_mask] - y_target_window[valid_mask]) ** 2) if valid_mask.any() else torch.tensor(0.0)
                    loss_bg = bg_weight * torch.mean((c_n_opt - c_start) ** 2) if c_n_opt is not None else torch.tensor(0.0)

                    loss = loss_obs + loss_bg
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        optimizer.step()

            with torch.no_grad():
                if c_n_opt is not None:
                    assim_data['c_n'] = c_n_opt
                    assim_data['c_0'] = c_n_opt
                if h_n_opt is not None:
                    assim_data['h_n'] = h_n_opt
                    assim_data['h_0'] = h_n_opt
                da_pred_window = _ensure_y_hat(model(assim_data))
                da_pred_window_y = da_pred_window['y_hat']
                c_n, h_n = c_n_opt, h_n_opt


        # Place 1: End of Section 2 (Just after da_pred_window_y is created)
        print(f"\n[DA DIAGNOSTIC] End of Section 2:")
        print(f"  -> _start_timestep: {self._start_timestep}")
        print(f"  -> _end_timestep:   {self._end_timestep}")
        print(f"  -> da_pred_window_y shape: {da_pred_window_y.shape}", flush=True)

        # --- 3. Reconstruct Full Prediction Trajectory (Guaranteed 366) ---
        with torch.no_grad():
            has_warmup = 'warmup_pred' in locals()
            print(f"\n[DA DIAGNOSTIC] Section 3 Start:")
            print(f"  -> Condition (_start_timestep > 0): {self._start_timestep > 0}")
            print(f"  -> Condition ('warmup_pred' in locals()): {has_warmup}", flush=True)

            if self._start_timestep > 0 and has_warmup:
                warm_y_hat = warmup_pred['y_hat'][:, :self._start_timestep, :]
                if warm_y_hat.ndim == 2: warm_y_hat = warm_y_hat.unsqueeze(-1)
                if da_pred_window_y.ndim == 2: da_pred_window_y = da_pred_window_y.unsqueeze(-1)
                
                print(f"  -> Branch taken: WARMUP + DA")
                print(f"  -> warm_y_hat shape: {warm_y_hat.shape}")
                print(f"  -> da_pred_window_y shape: {da_pred_window_y.shape}")
                da_y_hat = torch.cat([warm_y_hat, da_pred_window_y], dim=1)
            else:
                print(f"  -> Branch taken: DIRECT DA WINDOW ONLY")
                da_y_hat = da_pred_window_y

            print(f"  -> da_y_hat shape before expected_seq_len check: {da_y_hat.shape}")

            expected_seq_len = data['y'].shape[1]
            print(f"  -> data['y'] expected_seq_len: {expected_seq_len}")

            if da_y_hat.shape[1] != expected_seq_len:
                print(f"  -> Truncating da_y_hat from {da_y_hat.shape[1]} to {expected_seq_len}")
                da_y_hat = da_y_hat[:, :expected_seq_len, :]

            print(f"  -> Final da_y_hat shape: {da_y_hat.shape}\n", flush=True)


        # # --- 3. Reconstruct Full Prediction Trajectory (Guaranteed 366) ---
        # with torch.no_grad():
        #     if self._start_timestep > 0 and 'warmup_pred' in locals():
        #         warm_y_hat = warmup_pred['y_hat'][:, :self._start_timestep, :]
                
        #         # Ensure 3D shapes before concatenating
        #         if warm_y_hat.ndim == 2: warm_y_hat = warm_y_hat.unsqueeze(-1)
        #         if da_pred_window_y.ndim == 2: da_pred_window_y = da_pred_window_y.unsqueeze(-1)
                    
        #         # Prepend warmup (0 -> start) directly to active DA + forecast window (start -> end + lead)
        #         da_y_hat = torch.cat([warm_y_hat, da_pred_window_y], dim=1)
        #     else:
        #         da_y_hat = da_pred_window_y

            # Truncate strictly if it somehow exceeds expected_seq_len (366)
            expected_seq_len = data['y'].shape[1]
            if da_y_hat.shape[1] > expected_seq_len:
                da_y_hat = da_y_hat[:, :expected_seq_len, :]

            '''        # --- 3. Reconstruct Full Prediction Trajectory & Evaluate Metrics ---
                    with torch.no_grad():
                        if self._start_timestep > 0 and 'warmup_pred' in locals():
                            warm_y_hat = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                            if warm_y_hat.ndim == 2:
                                warm_y_hat = warm_y_hat.unsqueeze(-1)
                            if da_pred_window_y.ndim == 2:
                                da_pred_window_y = da_pred_window_y.unsqueeze(-1)
                            da_y_hat = torch.cat([warm_y_hat, da_pred_window_y], dim=1)
                        else:
                            da_y_hat = da_pred_window_y
            '''

            expected_seq_len = data['y'].shape[1]
            if da_y_hat.shape[1] != expected_seq_len:
                da_y_hat = da_y_hat[:, :expected_seq_len, :]

            # --- Metrics Calculation ---
            y_target_full = data['y'][:, :, 0:1] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            pred_base_y = full_base_pred['y_hat'][:, :, 0:1] if full_base_pred['y_hat'].ndim == 3 else full_base_pred['y_hat'].unsqueeze(-1)

            obs_full = y_target_full[0, :, 0].cpu().numpy()
            base_full = pred_base_y[0, :, 0].detach().cpu().numpy()
            da_full = da_y_hat[0, :, 0].detach().cpu().numpy()

            def _metrics(o, s):
                if len(o) == 0 or len(s) == 0 or len(o) != len(s):
                    return {'NSE': float('nan'), 'RMSE': float('nan'), 'Pearson-r': float('nan')}
                var_o = float(np.var(o))
                if var_o < 1e-8:
                    var_o = 1.0
                nse = float(1.0 - np.mean((s - o) ** 2) / var_o)
                rmse = float(np.sqrt(np.mean((s - o) ** 2)))
                std_s, std_o = float(np.std(s)), float(np.std(o))
                r = float(np.corrcoef(o, s)[0, 1]) if (std_s > 1e-8 and std_o > 1e-8) else 0.0
                return {'NSE': nse, 'RMSE': rmse, 'Pearson-r': r}

            n_h_eval = min(n_hindcast, len(obs_full))
            m_base_hind = _metrics(obs_full[:n_h_eval], base_full[:n_h_eval])
            m_da_hind = _metrics(obs_full[:n_h_eval], da_full[:n_h_eval])

            min_len = min(len(obs_full), len(base_full), len(da_full))
            m_base_fc = _metrics(obs_full[n_h_eval:min_len], base_full[n_h_eval:min_len]) if min_len > n_h_eval else {}
            m_da_fc = _metrics(obs_full[n_h_eval:min_len], da_full[n_h_eval:min_len]) if min_len > n_h_eval else {}

            # === CHECKPOINT 4: Final Output Tensor ===
            print(f"[DA Diagnostic 4] Final da_y_hat shape returned: {da_y_hat.shape}", flush=True)

            return {
                'y_hat': da_y_hat.detach(),
                'c_n': c_n.detach() if isinstance(c_n, torch.Tensor) else None,
                'h_n': h_n.detach() if isinstance(h_n, torch.Tensor) else None,
                'hindcast_metrics_pre': m_base_hind,
                'hindcast_metrics_post': m_da_hind,
                'forecast_metrics_pre': m_base_fc,
                'forecast_metrics_post': m_da_fc,
            }
            
class Old_Assimilation:
    """Gradient-based state updating for hydrological LSTM / AR-LSTM models."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self._end_timestep = cfg.seq_length - cfg.assimilation_lead_time
        self._start_timestep = max(0, self._end_timestep - (cfg.history * cfg.assimilation_window))

        if self._end_timestep > cfg.seq_length:
            raise ValueError("The sum of warmup and assimilation periods must not exceed sequence length.")

        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def check_discharge_timing(self, data: Dict[str, torch.Tensor], verbose: bool = True) -> Dict[str, Any]:
        """Checks river discharge timing, date alignment, and autoregressive lag alignment."""
        diagnostics = {
            'has_timing_mismatch': False,
            'warnings': [],
            'details': {}
        }

        if 'date' in data:
            dates = data['date']
            if isinstance(dates, torch.Tensor):
                dates = dates.cpu().numpy()
            sample_dates = dates[0] if dates.ndim > 1 else dates
            seq_len = len(sample_dates)

            diagnostics['details']['sequence_start_date'] = str(sample_dates[0])
            diagnostics['details']['sequence_end_date'] = str(sample_dates[-1])
            diagnostics['details']['start_timestep_date'] = str(sample_dates[self._start_timestep]) if self._start_timestep < seq_len else None
            diagnostics['details']['end_timestep_date'] = str(sample_dates[min(self._end_timestep - 1, seq_len - 1)]) if self._end_timestep <= seq_len else None

        x_d = data.get('x_d', data.get('x_d_hindcast', None))
        y = data.get('y', None)

        if x_d is not None and y is not None:
            x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
            for feat_name, feat_val in x_d_dict.items():
                match = re.search(r'^(.*)_shift(\d+)$', feat_name)
                if match or 'streamflow' in feat_name or 'discharge' in feat_name:
                    shift = int(match.group(2)) if match else 1
                    f_tensor = feat_val if isinstance(feat_val, torch.Tensor) else (torch.from_numpy(feat_val) if isinstance(feat_val, np.ndarray) else None)
                    y_tensor = y if isinstance(y, torch.Tensor) else (torch.from_numpy(y) if isinstance(y, np.ndarray) else None)

                    if f_tensor is None or y_tensor is None:
                        continue

                    check_t = min(self._start_timestep + 5, y_tensor.shape[1] - 1)
                    if check_t >= shift:
                        val_at_t = f_tensor[0, check_t, 0] if f_tensor.ndim == 3 else f_tensor[0, check_t]
                        y_same = y_tensor[0, check_t, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t]
                        y_prev = y_tensor[0, check_t - shift, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t - shift]

                        if not (torch.isnan(val_at_t) or torch.isnan(y_same) or torch.isnan(y_prev)):
                            diff_prev = torch.abs(val_at_t - y_prev).item()
                            diff_same = torch.abs(val_at_t - y_same).item()
                            if diff_same < 1e-5 and diff_prev > 1e-4:
                                msg = (f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}: "
                                       f"matches same-day target y[t] (diff: {diff_same:.6f}) instead of y[t-{shift}].")
                                diagnostics['has_timing_mismatch'] = True
                                diagnostics['warnings'].append(msg)

        return diagnostics

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validates that batch data matches the expected dictionary structure for assimilation."""
        if 'y' not in data:
            raise KeyError("[DA Validation Error] Required key 'y' missing from input data dictionary.")
        if 'x_d_hindcast' not in data and 'x_d' not in data:
            raise KeyError("[DA Validation Error] Input data batch must contain either 'x_d_hindcast' or 'x_d'.")

        for key in ['x_d', 'x_d_hindcast', 'x_d_forecast']:
            if key in data:
                if not isinstance(data[key], (dict, torch.Tensor, np.ndarray)):
                    raise TypeError(f"[DA Validation Error] '{key}' must be dict, torch.Tensor, or np.ndarray.")

        y = data['y']
        if not isinstance(y, (torch.Tensor, np.ndarray)):
            raise TypeError(f"[DA Validation Error] Target 'y' must be Tensor or ndarray, got {type(y)}.")
        if y.ndim != 3:
            raise ValueError(f"[DA Validation Error] Target 'y' must be a 3D tensor, got shape {y.shape}.")

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, check_timing: bool = True):
        """Performs state updating across the assimilation window."""
        self.validate_data_structure(data)
        if check_timing or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # --- 1. Warmup Pass ---
        with torch.no_grad():
            if self._start_timestep > 0:
                warmup_data = _slice_data_dict(data, slice_start=0, slice_end=self._start_timestep)
                warmup_pred = _ensure_y_hat(model(warmup_data))
                c_start = warmup_pred.get('c_n', warmup_pred.get('c_0', warmup_data.get('c_n', warmup_data.get('c_0'))))
                h_start = warmup_pred.get('h_n', warmup_pred.get('h_0', warmup_data.get('h_n', warmup_data.get('h_0'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

                # Baseline active window pass
                assim_data_base = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
                if c_start is not None:
                    assim_data_base['c_n'] = c_start
                    assim_data_base['c_0'] = c_start
                if h_start is not None:
                    assim_data_base['h_n'] = h_start
                    assim_data_base['h_0'] = h_start

                base_win_pred = _ensure_y_hat(model(assim_data_base))

                seq_len = data['y'].shape[1] if ('y' in data and hasattr(data['y'], 'shape') and data['y'].ndim >= 2) else getattr(self.cfg, 'seq_length', None)
                if seq_len is not None and seq_len > self._start_timestep and warmup_pred['y_hat'].ndim >= 2 and warmup_pred['y_hat'].shape[1] >= self._start_timestep:
                    t_act = seq_len - self._start_timestep
                    warm_y_base = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                    base_win_y = base_win_pred['y_hat'][:, -t_act:, :] if (base_win_pred['y_hat'].ndim >= 2 and base_win_pred['y_hat'].shape[1] >= t_act) else base_win_pred['y_hat']
                    full_base_pred = {'y_hat': torch.cat([warm_y_base, base_win_y], dim=1)}
                else:
                    full_base_pred = base_win_pred
            else:
                full_base_pred = _ensure_y_hat(model(data))
                c_start = data.get('c_n', data.get('c_0', full_base_pred.get('c_0', full_base_pred.get('c_n'))))
                h_start = data.get('h_n', data.get('h_0', full_base_pred.get('h_0', full_base_pred.get('h_n'))))
                c_start = c_start.detach().clone() if c_start is not None else None
                h_start = h_start.detach().clone() if h_start is not None else None

        lr_val = self.cfg.learning_rate
        lr = float(lr_val.get('c_n', list(lr_val.values())[0]) if isinstance(lr_val, dict) else (lr_val[0] if isinstance(lr_val, list) else lr_val))
        bg_weight = getattr(self.cfg, 'bg_regularization_weight', 0.005)
        n_hindcast = getattr(self.cfg, 'predict_n_hindcast', getattr(getattr(model, 'cfg', None), 'predict_n_hindcast', 5))
        use_per_step = getattr(self.cfg, 'use_per_step_updates', True)

        # --- 2. State Optimization ---
        if use_per_step and c_start is not None:
            # === SEQUENTIAL DAY-BY-DAY ASSIMILATION LOOP ===
            curr_c = c_start.clone().detach()
            curr_h = h_start.clone().detach() if h_start is not None else None

            seq_da_y_hats = []
            c_history = [curr_c.clone()]
            h_history = [curr_h.clone()] if curr_h is not None else []

            # Step day-by-day across the historical assimilation window
            for t in range(self._start_timestep, self._end_timestep):
                step_data = _slice_data_dict(data, slice_start=t, slice_end=t + 1)
                target_y = step_data['y'][:, :, 0:1] if step_data['y'].ndim == 3 else step_data['y'].unsqueeze(-1)

                # If missing target observation, step forward without optimization
                if torch.isnan(target_y).all():
                    with torch.no_grad():
                        step_data['c_0'] = curr_c
                        step_data['c_n'] = curr_c
                        if curr_h is not None:
                            step_data['h_0'] = curr_h
                            step_data['h_n'] = curr_h
                        pred = _ensure_y_hat(model(step_data))
                        curr_c = pred.get('c_n', pred.get('c_0', curr_c)).detach().clone()
                        if curr_h is not None:
                            curr_h = pred.get('h_n', pred.get('h_0', curr_h)).detach().clone()
                        seq_da_y_hats.append(pred['y_hat'])
                    continue

                # Optimize c_{t-1} (and h_{t-1}) against observation y_t
                c_opt = curr_c.clone().detach().requires_grad_(True)
                h_opt = curr_h.clone().detach().requires_grad_(True) if curr_h is not None else None

                opt_params = [c_opt] if h_opt is None else [c_opt, h_opt]
                optimizer = get_optimizer(opt_params, self.cfg)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    step_data['c_0'] = c_opt
                    step_data['c_n'] = c_opt
                    if h_opt is not None:
                        step_data['h_0'] = h_opt
                        step_data['h_n'] = h_opt

                    pred = _ensure_y_hat(model(step_data))
                    y_hat = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)

                    valid_mask = ~torch.isnan(target_y) & ~torch.isnan(y_hat)
                    loss_obs = torch.mean((y_hat[valid_mask] - target_y[valid_mask]) ** 2) if valid_mask.any() else torch.tensor(0.0, device=c_opt.device)
                    
                    loss_bg = bg_weight * torch.mean((c_opt - curr_c) ** 2)
                    if h_opt is not None:
                        loss_bg = loss_bg + bg_weight * torch.mean((h_opt - curr_h) ** 2)

                    loss = loss_obs + loss_bg
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        if hasattr(self.cfg, 'clip_gradient_norm') and self.cfg.clip_gradient_norm and self.cfg.clip_gradient_norm > 0:
                            torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.clip_gradient_norm)
                        optimizer.step()

                # Carry forward the updated state to step t+1
                with torch.no_grad():
                    step_data['c_0'] = c_opt.detach()
                    step_data['c_n'] = c_opt.detach()
                    if h_opt is not None:
                        step_data['h_0'] = h_opt.detach()
                        step_data['h_n'] = h_opt.detach()

                    final_step_pred = _ensure_y_hat(model(step_data))
                    curr_c = final_step_pred.get('c_n', final_step_pred.get('c_0', c_opt)).detach().clone()
                    if curr_h is not None:
                        curr_h = final_step_pred.get('h_n', final_step_pred.get('h_0', h_opt)).detach().clone()

                    seq_da_y_hats.append(final_step_pred['y_hat'])
                    c_history.append(curr_c.clone())
                    if curr_h is not None:
                        h_history.append(curr_h.clone())

            # Free run forecast window [T_end, T_seq] using final updated state
            with torch.no_grad():
                seq_len = data['y'].shape[1]
                if seq_len > self._end_timestep:
                    forecast_data = _slice_data_dict(data, slice_start=self._end_timestep, slice_end=None)
                    forecast_data['c_0'] = curr_c
                    forecast_data['c_n'] = curr_c
                    if curr_h is not None:
                        forecast_data['h_0'] = curr_h
                        forecast_data['h_n'] = curr_h
                    fc_pred = _ensure_y_hat(model(forecast_data))
                    seq_da_y_hats.append(fc_pred['y_hat'])

            # Free-run forecast over the lead-time window [356 -> 366]
            with torch.no_grad():
                seq_len = data['y'].shape[1]
                if seq_len > self._end_timestep:
                    forecast_data = _slice_data_dict(data, slice_start=self._end_timestep, slice_end=None)
                    forecast_data['c_0'] = curr_c
                    forecast_data['c_n'] = curr_c
                    if curr_h is not None:
                        forecast_data['h_0'] = curr_h
                        forecast_data['h_n'] = curr_h
                    fc_pred = _ensure_y_hat(model(forecast_data))
                    seq_da_y_hats.append(fc_pred['y_hat'])

            da_pred_window_y = torch.cat(seq_da_y_hats, dim=1)
    
            c_n = curr_c
            h_n = curr_h

        else:
            # === FALLBACK: WINDOWED 4D-VAR (c_start optimization) ===
            if self._start_timestep > 0:
                assim_data = _slice_data_dict(data, slice_start=self._start_timestep, slice_end=None)
            else:
                assim_data = dict(data)

            c_n = c_start.clone().detach().requires_grad_(True) if c_start is not None else None
            h_n = h_start.clone().detach().requires_grad_(True) if h_start is not None else None
            opt_params = [p for p in [c_n, h_n] if p is not None]

            optimizer = get_optimizer(opt_params, self.cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            if self.cfg.epochs > 0 and lr > 0 and len(opt_params) > 0:
                for _ in range(self.cfg.epochs):
                    optimizer.zero_grad()
                    if c_n is not None:
                        assim_data['c_n'] = c_n
                        assim_data['c_0'] = c_n
                    if h_n is not None:
                        assim_data['h_n'] = h_n
                        assim_data['h_0'] = h_n

                    pred = _ensure_y_hat(model(assim_data))
                    pred_y = pred['y_hat'][:, :, 0:1] if pred['y_hat'].ndim == 3 else pred['y_hat'].unsqueeze(-1)
                    y_target_window = assim_data['y'][:, :, 0:1] if assim_data['y'].ndim == 3 else assim_data['y'].unsqueeze(-1)
                    target_obs = y_target_window[:, -pred_y.shape[1]:, :] if y_target_window.shape[1] >= pred_y.shape[1] else y_target_window

                    n_h = min(n_hindcast, pred_y.shape[1], target_obs.shape[1])
                    pred_h = pred_y[:, :n_h, :]
                    target_h = target_obs[:, :n_h, :]

                    valid_mask = ~torch.isnan(target_h) & ~torch.isnan(pred_h)
                    loss_obs = torch.mean((pred_h[valid_mask] - target_h[valid_mask]) ** 2) if valid_mask.any() else torch.tensor(0.0, device=c_n.device)

                    loss_bg = bg_weight * torch.mean((c_n - c_start) ** 2) if c_n is not None else torch.tensor(0.0)
                    if h_n is not None:
                        loss_bg = loss_bg + bg_weight * torch.mean((h_n - h_start) ** 2)

                    loss = loss_obs + loss_bg
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        if hasattr(self.cfg, 'clip_gradient_norm') and self.cfg.clip_gradient_norm and self.cfg.clip_gradient_norm > 0:
                            torch.nn.utils.clip_grad_norm_(opt_params, self.cfg.clip_gradient_norm)
                        optimizer.step()

            with torch.no_grad():
                if c_n is not None:
                    assim_data['c_n'] = c_n
                    assim_data['c_0'] = c_n
                if h_n is not None:
                    assim_data['h_n'] = h_n
                    assim_data['h_0'] = h_n
                da_pred_window = _ensure_y_hat(model(assim_data))
                da_pred_window_y = da_pred_window['y_hat']

        # --- 3. Reconstruct Full Prediction Trajectory & Evaluate Metrics ---
        with torch.no_grad():
            seq_len = data['y'].shape[1] if ('y' in data and hasattr(data['y'], 'shape') and data['y'].ndim >= 2) else getattr(self.cfg, 'seq_length', None)
            if self._start_timestep > 0 and seq_len is not None and seq_len > self._start_timestep and 'warmup_pred' in locals() and warmup_pred['y_hat'].ndim >= 2 and warmup_pred['y_hat'].shape[1] >= self._start_timestep:
                t_act = seq_len - self._start_timestep
                warm_y_hat = warmup_pred['y_hat'][:, :self._start_timestep, :] if warmup_pred['y_hat'].ndim == 3 else warmup_pred['y_hat'][:, :self._start_timestep]
                da_win_y = da_pred_window_y[:, -t_act:, :] if (da_pred_window_y.ndim >= 2 and da_pred_window_y.shape[1] >= t_act) else da_pred_window_y
                da_y_hat = torch.cat([warm_y_hat, da_win_y], dim=1)
            else:
                da_y_hat = da_pred_window_y

            y_target_full = data['y'][:, :, 0:1] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            pred_base_y = full_base_pred['y_hat'][:, :, 0:1] if full_base_pred['y_hat'].ndim == 3 else full_base_pred['y_hat'].unsqueeze(-1)
            target_obs_full = y_target_full[:, -pred_base_y.shape[1]:, :] if y_target_full.shape[1] >= pred_base_y.shape[1] else y_target_full

            obs_full = target_obs_full[0, :, 0].cpu().numpy()
            base_y = full_base_pred['y_hat'][0]
            base_full = base_y[:, 0].detach().cpu().numpy() if base_y.ndim == 2 else base_y.detach().cpu().numpy()
            da_y = da_y_hat[0]
            da_full = da_y[:, 0].detach().cpu().numpy() if da_y.ndim == 2 else da_y.detach().cpu().numpy()

            def _metrics(o, s):
                if len(o) == 0 or len(s) == 0 or len(o) != len(s):
                    return {'NSE': float('nan'), 'RMSE': float('nan'), 'Pearson-r': float('nan')}
                var_o = float(np.var(o))
                if var_o < 1e-8:
                    var_o = 1.0
                nse = float(1.0 - np.mean((s - o) ** 2) / var_o)
                rmse = float(np.sqrt(np.mean((s - o) ** 2)))
                std_s, std_o = float(np.std(s)), float(np.std(o))
                r = float(np.corrcoef(o, s)[0, 1]) if (std_s > 1e-8 and std_o > 1e-8) else 0.0
                return {'NSE': nse, 'RMSE': rmse, 'Pearson-r': r}

            n_h_eval = min(n_hindcast, len(obs_full), len(base_full), len(da_full))
            m_base_hind = _metrics(obs_full[:n_h_eval], base_full[:n_h_eval])
            m_da_hind = _metrics(obs_full[:n_h_eval], da_full[:n_h_eval])

            min_len = min(len(obs_full), len(base_full), len(da_full))
            m_base_fc = _metrics(obs_full[n_h_eval:min_len], base_full[n_h_eval:min_len]) if min_len > n_h_eval else {}
            m_da_fc = _metrics(obs_full[n_h_eval:min_len], da_full[n_h_eval:min_len]) if min_len > n_h_eval else {}

            res = {
                'y_hat': da_y_hat.detach(),
                'c_n': c_n.detach() if isinstance(c_n, torch.Tensor) else None,
                'h_n': h_n.detach() if isinstance(h_n, torch.Tensor) else None,
                'hindcast_metrics_pre': m_base_hind,
                'hindcast_metrics_post': m_da_hind,
                'forecast_metrics_pre': m_base_fc,
                'forecast_metrics_post': m_da_fc,
            }
            return res