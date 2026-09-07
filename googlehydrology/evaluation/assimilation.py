"""Unified Data Assimilation (DA) engine for hydrological forecasting models."""

import logging
import re
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.cmal_deterministic import calc_cmal_mean, ensure_y_hat

logger = logging.getLogger(__name__)

# Backward-compatibility alias
_ensure_y_hat = ensure_y_hat


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
        'last_prediction', 'static_embedding',
    }
    h_len, f_len = None, None
    if 'x_d_hindcast' in d and isinstance(d['x_d_hindcast'], dict):
        for v in d['x_d_hindcast'].values():
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                h_len = v.shape[1]
                break
    if 'x_d_forecast' in d and isinstance(d['x_d_forecast'], dict):
        for v in d['x_d_forecast'].values():
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                f_len = v.shape[1]
                break
    lead_delta = (f_len - h_len) if (h_len is not None and f_len is not None and f_len > h_len) else 0

    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if k in ('x_d_forecast', 'forecast_features'):
                res[k] = _slice_hydrology_batch(v, slice_start, slice_end + lead_delta)
            else:
                res[k] = _slice_hydrology_batch(v, slice_start, slice_end)
        elif isinstance(v, torch.Tensor) and k not in non_seq_keys and v.ndim == 3:
            t_len = v.shape[1]
            end_idx = min(slice_end + lead_delta, t_len) if k in ('x_d_forecast', 'y') else min(slice_end, t_len)
            res[k] = v[:, min(slice_start, t_len):end_idx, :]
        elif isinstance(v, np.ndarray) and k in ('date', 'y') and v.ndim == 2:
            t_len = v.shape[1]
            end_idx = min(slice_end + lead_delta, t_len)
            res[k] = v[:, min(slice_start, t_len):end_idx]
        else:
            res[k] = v
    return res


class _FrozenModelContext:
    def __init__(self, model: nn.Module):
        self.model = model
        self.prev_training = model.training
        self.prev_grad_states = {p: p.requires_grad for p in model.parameters()}

    def __enter__(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, state in self.prev_grad_states.items():
            p.requires_grad = state
        self.model.train(self.prev_training)


class Assimilation(object):
    """Unified Data Assimilation (DA) state and parameter updating for hydrological forecasting models."""

    def __init__(self, cfg: AssimilationConfig):
        if cfg is None:
            raise ValueError("cfg cannot be None.")
        self.cfg = cfg
        self.window = getattr(cfg, 'assimilation_window_length', getattr(cfg, 'assimilation_window', 1))
        self.history = getattr(cfg, 'history', 1)
        self.lead_time = getattr(cfg, 'assimilation_lead_time', 0)
        self.epochs = getattr(cfg, 'epochs', 10)
        self.components = getattr(cfg, 'assimilation_components', {})
        self.targets = getattr(cfg, 'assimilation_targets', list(self.components.keys()) if self.components else ['c_n'])

        # Boundary timesteps
        self.assimilation_end_step = cfg.seq_length - self.lead_time
        self.assimilation_start_step = max(0, self.assimilation_end_step - (self.history * self.window))
        self._end_timestep = self.assimilation_end_step
        self._start_timestep = self.assimilation_start_step

        if self.assimilation_end_step > cfg.seq_length:
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
        if 'date' in data:
            d = data['date']
            start_date = str(d[0, 0]) if (isinstance(d, np.ndarray) and d.ndim >= 2) else (str(d[0]) if hasattr(d, '__getitem__') else str(d))
            diagnostics['details']['sequence_start_date'] = start_date

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
                                diagnostics['warnings'].append(f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}.")
        return diagnostics

    def _parse_target_flags(self) -> Tuple[bool, bool, bool, bool]:
        """Parses self.targets into boolean flags for (c_hc, h_hc, c_fc, h_fc)."""
        opt_c_hc = any(k in self.targets for k in ['c_n_hindcast', 'c_0_hindcast', 'c_hc', 'c_n', 'c_0'])
        opt_h_hc = any(k in self.targets for k in ['h_n_hindcast', 'h_0_hindcast', 'h_hc', 'h_n', 'h_0'])
        opt_c_fc = any(k in self.targets for k in ['c_n_forecast', 'c_0_forecast', 'c_fc', 'c_n', 'c_0'])
        opt_h_fc = any(k in self.targets for k in ['h_n_forecast', 'h_0_forecast', 'h_fc', 'h_n', 'h_0'])
        return opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc

    def _detect_da_type(self) -> str:
        """Determines the DA mode from self.components or self.targets: 'embedding', 'precip', or 'state'."""
        if hasattr(self, 'components') and self.components:
            comp_names = [str(k).lower() for k in self.components.keys()]
            comp_str = ' '.join(comp_names)
            if any(k in comp_str for k in ['precip', 'precipitation', 'forcing', 'rain', 'tp']):
                return 'precip'
            if any(k in comp_str for k in ['embedded', 'embedding', 'static_embedding', 'hindcast_embedding', 'forecast_embedding']):
                return 'embedding'
            # If any component is not a recognized recurrent state, treat as generic component/embedding
            state_keys = {'c_n', 'h_n', 'c_0', 'h_0', 'c_hc', 'h_hc', 'c_fc', 'h_fc', 'c_n_hindcast', 'h_n_hindcast', 'c_n_forecast', 'h_n_forecast'}
            if not all(k in state_keys for k in comp_names):
                return 'embedding'

        targets = [str(t).lower() for t in self.targets] if isinstance(self.targets, (list, tuple)) else [str(self.targets).lower()]
        target_str = ' '.join(targets)
        if any(k in target_str for k in ['embedded', 'embedding', 'static_embedding', 'hindcast_embedding', 'forecast_embedding']):
            return 'embedding'
        elif any(k in target_str for k in ['precip', 'precipitation', 'forcing', 'rain', 'tp']):
            return 'precip'
        return 'state'

    def _parse_embedding_masks(self) -> Tuple[bool, bool, bool]:
        """Returns (mask_e_stat, mask_e_dyn, mask_e_fc) based on configured components or targets."""
        if hasattr(self, 'components') and self.components:
            mask_e_stat = 'static_embedding' in self.components or any('stat' in k.lower() for k in self.components)
            mask_e_dyn = 'hindcast_embedding' in self.components or any(k.lower() in ['hindcast_embedding', 'dyn', 'dynamic'] for k in self.components)
            mask_e_fc = 'forecast_embedding' in self.components or any('forecast' in k.lower() for k in self.components)
            if mask_e_stat or mask_e_dyn or mask_e_fc:
                return mask_e_stat, mask_e_dyn, mask_e_fc

        targets = [str(t).lower() for t in self.targets] if isinstance(self.targets, (list, tuple)) else [str(self.targets).lower()]
        target_str = ' '.join(targets)
        mask_e_stat = any(k in target_str for k in ['stat', 'both', 'all', 'embedding'])
        mask_e_dyn = any(k in target_str for k in ['dyn', 'both', 'all', 'temporal', 'hc', 'hindcast'])
        mask_e_fc = any(k in target_str for k in ['all', 'three', 'fc', 'forecast'])
        return mask_e_stat, mask_e_dyn, mask_e_fc

    def _extract_state_tensor(self, state_val: Optional[torch.Tensor], t_idx: int) -> Optional[torch.Tensor]:
        if state_val is None:
            return None
        if state_val.ndim == 4:  # [num_layers, batch, seq_len, hidden_size]
            t_clamp = min(max(0, t_idx), state_val.shape[2] - 1)
            return state_val[:, :, t_clamp, :].detach().clone()
        elif state_val.ndim == 3:  # [num_layers, batch, hidden_size] or [batch, 1, hidden_size]
            return state_val.detach().clone()
        return None

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, **kwargs) -> Dict[str, Any]:
        self.validate_data_structure(data)
        if kwargs.get('check_timing', False) or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        da_type = self._detect_da_type()
        mask_e_stat, mask_e_dyn, mask_e_fc = self._parse_embedding_masks() if da_type == 'embedding' else (False, False, False)
        opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc = self._parse_target_flags()

        with _FrozenModelContext(model):
            y_tensor = data['y'] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            batch_size = y_tensor.shape[0]
            total_len = y_tensor.shape[1]

            hidden_size = (
                getattr(model, 'hidden_size', None)
                or getattr(getattr(model, 'config_data', None), 'hidden_size', None)
                or getattr(getattr(model, 'cfg', None), 'hidden_size', None)
                or getattr(getattr(model, 'hindcast_lstm', None), 'hidden_size', None)
                or getattr(self.cfg, 'hidden_size', 64)
            )
            if isinstance(hidden_size, dict):
                hidden_size = list(hidden_size.values())[0]
            hidden_size = int(hidden_size)
            device = y_tensor.device

            assimilation_start_step = self.assimilation_start_step
            assimilation_end_step = self.assimilation_end_step

            assimilated_discharge_predictions = []
            dist_chunks = {k: [] for k in ['mu', 'b', 'tau', 'pi']}
            prior_discharge_predictions = []

            target_key = self.targets[0] if self.targets else 'c_n'
            lr = _get_var_lr(self.cfg.learning_rate, target_key)

            c_user = data.get('c_0', data.get('c_n', None))
            h_user = data.get('h_0', data.get('h_n', None))

            # =========================================================================
            # PHASE 1: Warmup Phase (0 -> assimilation_start_step)
            # =========================================================================
            static_embedding_opt = None
            if assimilation_start_step > 0:
                warmup_data = _slice_hydrology_batch(data, 0, assimilation_start_step)
                if c_user is not None: warmup_data['c_0'] = c_user
                if h_user is not None: warmup_data['h_0'] = h_user

                with torch.no_grad():
                    warmup_out = ensure_y_hat(model(warmup_data), use_median=True)
                    base_y_warmup = warmup_out['y_hat']
                    if base_y_warmup.ndim == 2:
                        base_y_warmup = base_y_warmup.unsqueeze(-1)
                    assimilated_discharge_predictions.append(base_y_warmup[:, :assimilation_start_step, :])

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in warmup_out and isinstance(warmup_out[key], torch.Tensor):
                            dist_chunks[key].append(warmup_out[key][:, :assimilation_start_step, ...])

                    c_hc_curr = warmup_out.get('c_n_hindcast', warmup_out.get('c_n'))
                    h_hc_curr = warmup_out.get('h_n_hindcast', warmup_out.get('h_n'))
                    c_fc_curr = warmup_out.get('c_n_forecast', warmup_out.get('c_n', c_hc_curr))
                    h_fc_curr = warmup_out.get('h_n_forecast', warmup_out.get('h_n', h_hc_curr))
            else:
                c_hc_curr = c_user.detach().clone() if c_user is not None else torch.zeros(1, batch_size, hidden_size, device=device)
                h_hc_curr = h_user.detach().clone() if h_user is not None else torch.zeros(1, batch_size, hidden_size, device=device)
                c_fc_curr = c_hc_curr.clone()
                h_fc_curr = h_hc_curr.clone()

            if c_hc_curr is None: c_hc_curr = torch.zeros(1, batch_size, hidden_size, device=device)
            if h_hc_curr is None: h_hc_curr = torch.zeros(1, batch_size, hidden_size, device=device)
            if c_fc_curr is None: c_fc_curr = c_hc_curr.clone()
            if h_fc_curr is None: h_fc_curr = h_hc_curr.clone()

            last_prediction_curr = data.get('last_prediction', None)

            last_e_stat = None
            last_e_dyn = None
            last_e_fc = None
            last_p_opt = None
            last_p_opts = None

            # =========================================================================
            # PHASE 2: Sequential Window Optimization (assimilation_start_step -> assimilation_end_step)
            # =========================================================================
            current_step_index = assimilation_start_step
            for w_idx in range(self.history):
                if current_step_index >= assimilation_end_step: break
                window_end_step = min(current_step_index + self.window, assimilation_end_step)
                window_length = window_end_step - current_step_index

                chunk_data = _slice_hydrology_batch(data, current_step_index, window_end_step)
                chunk_data['c_0_hindcast'] = c_hc_curr
                chunk_data['h_0_hindcast'] = h_hc_curr
                chunk_data['c_0_forecast'] = c_fc_curr
                chunk_data['h_0_forecast'] = h_fc_curr
                chunk_data['c_0'] = c_hc_curr
                chunk_data['h_0'] = h_hc_curr
                chunk_data['c_n'] = c_hc_curr
                chunk_data['h_n'] = h_hc_curr
                if static_embedding_opt is not None:
                    chunk_data['static_embedding'] = static_embedding_opt
                if last_prediction_curr is not None:
                    chunk_data['last_prediction'] = last_prediction_curr

                # -------------------------------------------------------------
                # 2A. EMBEDDED DATA ASSIMILATION
                # -------------------------------------------------------------
                if da_type == 'embedding':
                    with torch.no_grad():
                        pre_pred = ensure_y_hat(model(chunk_data), use_median=True)
                        prior_predicted_discharge_window = pre_pred['y_hat'][:, :window_length, :]
                        if prior_predicted_discharge_window.ndim == 2: prior_predicted_discharge_window = prior_predicted_discharge_window.unsqueeze(-1)
                        prior_discharge_predictions.append(prior_predicted_discharge_window)

                        e_stat_base = pre_pred['static_embedding']
                        e_dyn_base = pre_pred['hindcast_embedding']
                        e_fc_base = pre_pred['forecast_embedding']

                    e_stat_opt = e_stat_base.clone().detach().requires_grad_(True) if mask_e_stat else e_stat_base
                    e_dyn_opt = e_dyn_base.clone().detach().requires_grad_(True) if mask_e_dyn else e_dyn_base
                    e_fc_opt = e_fc_base.clone().detach().requires_grad_(True) if mask_e_fc else e_fc_base
                    parameters_to_optimize = [p for p in (e_stat_opt, e_dyn_opt, e_fc_opt) if p is not None and p.requires_grad]

                    if parameters_to_optimize:
                        comp_dict = getattr(self, 'components', {})
                        lr_stat = comp_dict.get('static_embedding', {}).get('lr', None) or _get_var_lr(self.cfg.learning_rate, 'static_embedding')
                        lr_dyn = comp_dict.get('hindcast_embedding', {}).get('lr', None) or _get_var_lr(self.cfg.learning_rate, 'hindcast_embedding')
                        lr_fc = comp_dict.get('forecast_embedding', {}).get('lr', None) or _get_var_lr(self.cfg.learning_rate, 'forecast_embedding')

                        param_groups = []
                        if mask_e_stat and e_stat_opt.requires_grad:
                            param_groups.append({'params': [e_stat_opt], 'lr': lr_stat})
                        if mask_e_dyn and e_dyn_opt.requires_grad:
                            param_groups.append({'params': [e_dyn_opt], 'lr': lr_dyn})
                        if mask_e_fc and e_fc_opt.requires_grad:
                            param_groups.append({'params': [e_fc_opt], 'lr': lr_fc})

                        optimizer = get_optimizer(param_groups, self.cfg)
                        static_reg_weight = comp_dict.get('static_embedding', {}).get('weight', getattr(self.cfg, 'static_embedding_regularization_weight', getattr(self.cfg, 'bg_stat_weight', 1e-6)))
                        hindcast_reg_weight = comp_dict.get('hindcast_embedding', {}).get('weight', getattr(self.cfg, 'hindcast_embedding_regularization_weight', getattr(self.cfg, 'bg_dyn_weight', getattr(self.cfg, 'regularization_weight', 0.01))))
                        forecast_reg_weight = comp_dict.get('forecast_embedding', {}).get('weight', hindcast_reg_weight)

                        for epoch in range(self.epochs):
                            optimizer.zero_grad()
                            chunk_data['static_embedding'] = e_stat_opt
                            chunk_data['hindcast_embedding'] = e_dyn_opt
                            chunk_data['forecast_embedding'] = e_fc_opt
                            chunk_data['assimilation_overrides'] = {
                                'static_embedding': e_stat_opt,
                                'hindcast_embedding': e_dyn_opt,
                                'forecast_embedding': e_fc_opt,
                            }
                            chunk_data['c_0_hindcast'] = c_hc_curr
                            chunk_data['h_0_hindcast'] = h_hc_curr
                            chunk_data['c_0_forecast'] = c_fc_curr
                            chunk_data['h_0_forecast'] = h_fc_curr
                            if last_prediction_curr is not None:
                                chunk_data['last_prediction'] = last_prediction_curr

                            pred_dict = ensure_y_hat(model(chunk_data), use_median=False)
                            predicted_discharge_window = pred_dict['y_hat'][:, :window_length, :]
                            if predicted_discharge_window.ndim == 2: predicted_discharge_window = predicted_discharge_window.unsqueeze(-1)
                            observed_discharge_window = chunk_data['y'][:, :window_length, :]

                            mask = ~torch.isnan(observed_discharge_window) & ~torch.isnan(predicted_discharge_window)
                            if mask.any():
                                loss = torch.mean((predicted_discharge_window[mask] - observed_discharge_window[mask]) ** 2)
                                reg_loss = 0.0
                                if mask_e_stat and e_stat_opt.requires_grad:
                                    reg_loss = reg_loss + static_reg_weight * torch.mean((e_stat_opt - e_stat_base) ** 2)
                                if mask_e_dyn and e_dyn_opt.requires_grad:
                                    reg_loss = reg_loss + hindcast_reg_weight * torch.mean((e_dyn_opt - e_dyn_base) ** 2)
                                if mask_e_fc and e_fc_opt.requires_grad:
                                    reg_loss = reg_loss + forecast_reg_weight * torch.mean((e_fc_opt - e_fc_base) ** 2)
                                loss = loss + reg_loss

                                if torch.isfinite(loss) and loss.requires_grad:
                                    loss.backward()
                                    if getattr(self.cfg, 'clip_gradient_norm', 0) > 0:
                                        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, self.cfg.clip_gradient_norm)
                                    optimizer.step()

                    with torch.no_grad():
                        chunk_data['static_embedding'] = e_stat_opt.detach()
                        chunk_data['hindcast_embedding'] = e_dyn_opt.detach()
                        chunk_data['forecast_embedding'] = e_fc_opt.detach()
                        chunk_data['assimilation_overrides'] = {
                            'static_embedding': e_stat_opt.detach(),
                            'hindcast_embedding': e_dyn_opt.detach(),
                            'forecast_embedding': e_fc_opt.detach(),
                        }
                        rollout = ensure_y_hat(model(chunk_data), use_median=True)
                        if 'last_prediction' in rollout:
                            last_prediction_curr = rollout['last_prediction']

                        p_roll = rollout['y_hat'][:, :window_length, :]
                        if p_roll.ndim == 2: p_roll = p_roll.unsqueeze(-1)
                        assimilated_discharge_predictions.append(p_roll)

                        for key in ['mu', 'b', 'tau', 'pi']:
                            if key in rollout and isinstance(rollout[key], torch.Tensor):
                                dist_chunks[key].append(rollout[key][:, :window_length, ...])

                        c_hc_curr = rollout.get('c_n_hindcast', rollout.get('c_n'))
                        h_hc_curr = rollout.get('h_n_hindcast', rollout.get('h_n'))
                        c_fc_curr = rollout.get('c_n_forecast', rollout.get('c_n', c_hc_curr))
                        h_fc_curr = rollout.get('h_n_forecast', rollout.get('h_n', h_hc_curr))
                        static_embedding_opt = e_stat_opt.detach()
                        last_e_stat = e_stat_opt.detach()
                        last_e_dyn = e_dyn_opt.detach()
                        last_e_fc = e_fc_opt.detach()

                # -------------------------------------------------------------
                # 2B. PRECIPITATION FORCING DATA ASSIMILATION
                # -------------------------------------------------------------
                elif da_type == 'precip':
                    with torch.no_grad():
                        pre_pred = ensure_y_hat(model(chunk_data), use_median=True)
                        prior_predicted_discharge_window = pre_pred['y_hat'][:, :window_length, :]
                        if prior_predicted_discharge_window.ndim == 2: prior_predicted_discharge_window = prior_predicted_discharge_window.unsqueeze(-1)
                        prior_discharge_predictions.append(prior_predicted_discharge_window)

                    hind_dict = chunk_data.get('x_d_hindcast', chunk_data.get('x_d', None))
                    precip_keys = []
                    if isinstance(hind_dict, dict):
                        cfg_keys = getattr(self.cfg, 'precip_forcing_keys', None)
                        if cfg_keys is None:
                            single_key = getattr(self.cfg, 'precip_forcing_key', None)
                            if single_key:
                                cfg_keys = [single_key] if isinstance(single_key, str) else list(single_key)
                        if cfg_keys:
                            precip_keys = [k for k in cfg_keys if k in hind_dict]
                        else:
                            precip_keys = [
                                k for k in hind_dict.keys()
                                if any(p in k.lower() for p in ['precip', 'tp', 'prcp', 'rain'])
                            ]
                        if not precip_keys:
                            raise ValueError(
                                "Precipitation DA ('precip') failed: no precipitation forcing key "
                                f"found in hindcast inputs. Available keys: {list(hind_dict.keys())}."
                            )

                    precip_opts = {
                        k: hind_dict[k].clone().detach().requires_grad_(True)
                        for k in precip_keys
                        if isinstance(hind_dict.get(k), torch.Tensor)
                    }
                    if precip_opts:
                        parameters_to_optimize = list(precip_opts.values())
                        optimizer = get_optimizer(parameters_to_optimize, self.cfg)
                        for pg in optimizer.param_groups: pg["lr"] = lr
                        minimum_precipitation_bound = getattr(self.cfg, 'precip_min_clip', -3.0)

                        for epoch in range(self.epochs):
                            optimizer.zero_grad()
                            for k, p_opt in precip_opts.items():
                                hind_dict[k] = p_opt
                            chunk_data['c_0_hindcast'] = c_hc_curr
                            chunk_data['h_0_hindcast'] = h_hc_curr
                            chunk_data['c_0_forecast'] = c_fc_curr
                            chunk_data['h_0_forecast'] = h_fc_curr
                            if last_prediction_curr is not None:
                                chunk_data['last_prediction'] = last_prediction_curr

                            pred_dict = ensure_y_hat(model(chunk_data), use_median=False)
                            predicted_discharge_window = pred_dict['y_hat'][:, :window_length, :]
                            if predicted_discharge_window.ndim == 2: predicted_discharge_window = predicted_discharge_window.unsqueeze(-1)
                            observed_discharge_window = chunk_data['y'][:, :window_length, :]

                            mask = ~torch.isnan(observed_discharge_window) & ~torch.isnan(predicted_discharge_window)
                            if mask.any():
                                loss = torch.mean((predicted_discharge_window[mask] - observed_discharge_window[mask]) ** 2)
                                if torch.isfinite(loss) and loss.requires_grad:
                                    loss.backward()
                                    if getattr(self.cfg, 'clip_gradient_norm', 0) > 0:
                                        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, self.cfg.clip_gradient_norm)
                                    optimizer.step()
                                    with torch.no_grad():
                                        for p_opt in precip_opts.values():
                                            p_opt.clamp_(min=minimum_precipitation_bound)

                        with torch.no_grad():
                            for k, p_opt in precip_opts.items():
                                hind_dict[k] = p_opt.detach()
                            last_p_opt = precip_opts[precip_keys[0]].detach()
                            last_p_opts = {k: p_opt.detach() for k, p_opt in precip_opts.items()}

                    with torch.no_grad():
                        rollout = ensure_y_hat(model(chunk_data), use_median=True)
                        if 'last_prediction' in rollout:
                            last_prediction_curr = rollout['last_prediction']

                        p_roll = rollout['y_hat'][:, :window_length, :]
                        if p_roll.ndim == 2: p_roll = p_roll.unsqueeze(-1)
                        assimilated_discharge_predictions.append(p_roll)

                        for key in ['mu', 'b', 'tau', 'pi']:
                            if key in rollout and isinstance(rollout[key], torch.Tensor):
                                dist_chunks[key].append(rollout[key][:, :window_length, ...])

                        c_hc_curr = rollout.get('c_n_hindcast', rollout.get('c_n'))
                        h_hc_curr = rollout.get('h_n_hindcast', rollout.get('h_n'))
                        c_fc_curr = rollout.get('c_n_forecast', rollout.get('c_n', c_hc_curr))
                        h_fc_curr = rollout.get('h_n_forecast', rollout.get('h_n', h_hc_curr))

                # -------------------------------------------------------------
                # 2C. RECURRENT STATE DATA ASSIMILATION (DEFAULT)
                # -------------------------------------------------------------
                else:
                    c_hc_opt = c_hc_curr.clone().detach().requires_grad_(True) if (opt_c_hc and c_hc_curr is not None) else (c_hc_curr.clone().detach() if c_hc_curr is not None else None)
                    h_hc_opt = h_hc_curr.clone().detach().requires_grad_(True) if (opt_h_hc and h_hc_curr is not None) else (h_hc_curr.clone().detach() if h_hc_curr is not None else None)
                    c_fc_opt = c_fc_curr.clone().detach().requires_grad_(True) if (opt_c_fc and c_fc_curr is not None) else (c_fc_curr.clone().detach() if c_fc_curr is not None else None)
                    h_fc_opt = h_fc_curr.clone().detach().requires_grad_(True) if (opt_h_fc and h_fc_curr is not None) else (h_fc_curr.clone().detach() if h_fc_curr is not None else None)

                    parameters_to_optimize = [p for p in (c_hc_opt, h_hc_opt, c_fc_opt, h_fc_opt) if p is not None and p.requires_grad]

                    with torch.no_grad():
                        pre_pred = ensure_y_hat(model(chunk_data), use_median=True)
                        prior_predicted_discharge_window = pre_pred['y_hat'][:, :window_length, :]
                        if prior_predicted_discharge_window.ndim == 2: prior_predicted_discharge_window = prior_predicted_discharge_window.unsqueeze(-1)
                        prior_discharge_predictions.append(prior_predicted_discharge_window)

                    if parameters_to_optimize:
                        optimizer = get_optimizer(parameters_to_optimize, self.cfg)
                        for pg in optimizer.param_groups: pg["lr"] = lr

                        for epoch in range(self.epochs):
                            optimizer.zero_grad()

                            chunk_data['c_0_hindcast'] = c_hc_opt
                            chunk_data['h_0_hindcast'] = h_hc_opt
                            chunk_data['c_0_forecast'] = c_fc_opt
                            chunk_data['h_0_forecast'] = h_fc_opt
                            chunk_data['c_0'] = c_hc_opt if opt_c_hc else (c_fc_opt if opt_c_fc else c_hc_opt)
                            chunk_data['h_0'] = h_hc_opt if opt_h_hc else (h_fc_opt if opt_h_fc else h_hc_opt)
                            chunk_data['c_n'] = chunk_data['c_0']
                            chunk_data['h_n'] = chunk_data['h_0']
                            if last_prediction_curr is not None:
                                chunk_data['last_prediction'] = last_prediction_curr

                            pred_dict = ensure_y_hat(model(chunk_data), use_median=False)
                            predicted_discharge_window = pred_dict['y_hat'][:, :window_length, :]
                            if predicted_discharge_window.ndim == 2: predicted_discharge_window = predicted_discharge_window.unsqueeze(-1)
                            observed_discharge_window = chunk_data['y'][:, :window_length, :]

                            mask = ~torch.isnan(observed_discharge_window) & ~torch.isnan(predicted_discharge_window)
                            if mask.any():
                                loss = torch.mean((predicted_discharge_window[mask] - observed_discharge_window[mask]) ** 2)
                                reg_weight = getattr(self.cfg, 'recurrent_state_regularization_weight', getattr(self.cfg, 'bg_regularization_weight', getattr(self.cfg, 'regularization_weight', 0.0))) or 0.0
                                if reg_weight > 0:
                                    reg_loss = 0.0
                                    if opt_c_hc and c_hc_curr is not None: reg_loss = reg_loss + torch.mean((c_hc_opt - c_hc_curr) ** 2)
                                    if opt_h_hc and h_hc_curr is not None: reg_loss = reg_loss + torch.mean((h_hc_opt - h_hc_curr) ** 2)
                                    if opt_c_fc and c_fc_curr is not None: reg_loss = reg_loss + torch.mean((c_fc_opt - c_fc_curr) ** 2)
                                    if opt_h_fc and h_fc_curr is not None: reg_loss = reg_loss + torch.mean((h_fc_opt - h_fc_curr) ** 2)
                                    loss = loss + reg_weight * reg_loss

                                if torch.isfinite(loss) and loss.requires_grad:
                                    loss.backward()
                                    if getattr(self.cfg, 'clip_gradient_norm', 0) > 0:
                                        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, self.cfg.clip_gradient_norm)
                                    optimizer.step()
                            else:
                                logger.debug('Window [%d:%d] contains 0 valid target observations. Bypassing gradient update.', current_step_index, window_end_step)

                    with torch.no_grad():
                        chunk_data['c_0_hindcast'] = c_hc_opt.detach() if c_hc_opt is not None else None
                        chunk_data['h_0_hindcast'] = h_hc_opt.detach() if h_hc_opt is not None else None
                        chunk_data['c_0_forecast'] = c_fc_opt.detach() if c_fc_opt is not None else None
                        chunk_data['h_0_forecast'] = h_fc_opt.detach() if h_fc_opt is not None else None
                        chunk_data['c_0'] = chunk_data['c_0_hindcast']
                        chunk_data['h_0'] = chunk_data['h_0_hindcast']
                        chunk_data['c_n'] = chunk_data['c_0']
                        chunk_data['h_n'] = chunk_data['h_0']
                        if last_prediction_curr is not None:
                            chunk_data['last_prediction'] = last_prediction_curr

                        rollout = ensure_y_hat(model(chunk_data), use_median=True)
                        if 'last_prediction' in rollout:
                            last_prediction_curr = rollout['last_prediction']

                        p_roll = rollout['y_hat'][:, :window_length, :]
                        if p_roll.ndim == 2: p_roll = p_roll.unsqueeze(-1)
                        assimilated_discharge_predictions.append(p_roll)

                        for key in ['mu', 'b', 'tau', 'pi']:
                            if key in rollout and isinstance(rollout[key], torch.Tensor):
                                dist_chunks[key].append(rollout[key][:, :window_length, ...])

                        c_hc_curr = rollout.get('c_n_hindcast', rollout.get('c_n'))
                        h_hc_curr = rollout.get('h_n_hindcast', rollout.get('h_n'))
                        c_fc_curr = rollout.get('c_n_forecast', rollout.get('c_n', c_hc_curr))
                        h_fc_curr = rollout.get('h_n_forecast', rollout.get('h_n', h_hc_curr))

                current_step_index = window_end_step

            # =========================================================================
            # PHASE 3: Post-DA Forecast Horizon (assimilation_end_step -> total_len)
            # =========================================================================
            if current_step_index < total_len:
                with torch.no_grad():
                    fc_data = _slice_hydrology_batch(data, current_step_index, total_len)
                    fc_data['c_0_hindcast'] = c_hc_curr
                    fc_data['h_0_hindcast'] = h_hc_curr
                    fc_data['c_0_forecast'] = c_fc_curr
                    fc_data['h_0_forecast'] = h_fc_curr
                    fc_data['c_0'] = c_fc_curr
                    fc_data['h_0'] = h_fc_curr
                    fc_data['c_n'] = c_fc_curr
                    fc_data['h_n'] = h_fc_curr
                    if static_embedding_opt is not None:
                        fc_data['static_embedding'] = static_embedding_opt
                    if last_prediction_curr is not None:
                        fc_data['last_prediction'] = last_prediction_curr

                    fc_pred = ensure_y_hat(model(fc_data), use_median=True)
                    fc_y = fc_pred['y_hat'][:, :(total_len - current_step_index), :]
                    if fc_y.ndim == 2: fc_y = fc_y.unsqueeze(-1)
                    assimilated_discharge_predictions.append(fc_y)

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in fc_pred and isinstance(fc_pred[key], torch.Tensor):
                            dist_chunks[key].append(fc_pred[key][:, :(total_len - current_step_index), ...])

            out_y = torch.cat(assimilated_discharge_predictions, dim=1)[:, :total_len, :]

            # Pre/Post Hindcast Validation Metrics
            with torch.no_grad():
                min_len = min(assimilation_end_step - assimilation_start_step, y_tensor.shape[1])
                if min_len > 0 and prior_discharge_predictions:
                    t_hc = y_tensor[:, assimilation_start_step:assimilation_start_step + min_len, :]
                    p_pre = torch.cat(prior_discharge_predictions, dim=1)[:, :min_len, :]
                    p_post = out_y[:, assimilation_start_step:assimilation_start_step + min_len, :]

                    valid_len = min(t_hc.shape[1], p_pre.shape[1], p_post.shape[1])
                    if valid_len > 0:
                        t_hc = t_hc[:, :valid_len, :]
                        p_pre = p_pre[:, :valid_len, :]
                        p_post = p_post[:, :valid_len, :]
                        mask_pre = ~torch.isnan(t_hc) & ~torch.isnan(p_pre)
                        mask_post = ~torch.isnan(t_hc) & ~torch.isnan(p_post)
                    else:
                        mask_pre = torch.zeros(1, dtype=torch.bool)
                        mask_post = torch.zeros(1, dtype=torch.bool)
                else:
                    valid_len = 0
                    p_pre, p_post, t_hc = None, None, None
                    mask_pre = torch.zeros(1, dtype=torch.bool)
                    mask_post = torch.zeros(1, dtype=torch.bool)

                def calc_metrics(sim, obs, mask):
                    if not mask.any() or valid_len == 0 or sim is None or obs is None:
                        return {'MSE': float('nan'), 'NSE': float('nan')}
                    s, o = sim[mask], obs[mask]
                    mse = torch.mean((s - o) ** 2).item()
                    var_o = torch.var(o, unbiased=False).item()
                    nse = 1.0 - (mse / (var_o + 1e-6)) if var_o > 1e-8 else float('nan')
                    return {'MSE': mse, 'NSE': nse}

                metrics_pre = calc_metrics(p_pre, t_hc, mask_pre)
                metrics_post = calc_metrics(p_post, t_hc, mask_post)

            res = {
                'y_hat': out_y.detach(),
                'c_n_hindcast': c_hc_curr,
                'h_n_hindcast': h_hc_curr,
                'c_n_forecast': c_fc_curr,
                'h_n_forecast': h_fc_curr,
                'c_n': c_fc_curr if c_fc_curr is not None else c_hc_curr,
                'h_n': h_fc_curr if h_fc_curr is not None else h_hc_curr,
                'hindcast_metrics_pre': metrics_pre,
                'hindcast_metrics_post': metrics_post,
            }
            if da_type == 'embedding':
                if last_e_stat is not None: res['static_embedding'] = last_e_stat
                if last_e_dyn is not None: res['hindcast_embedding'] = last_e_dyn
                if last_e_fc is not None: res['forecast_embedding'] = last_e_fc
            elif da_type == 'precip' and last_p_opt is not None:
                res['precip'] = last_p_opt
                if last_p_opts is not None:
                    res['precip_dict'] = last_p_opts

            for key, chunks in dist_chunks.items():
                if chunks:
                    res[key] = torch.cat(chunks, dim=1)[:, :total_len, ...]
            return res
