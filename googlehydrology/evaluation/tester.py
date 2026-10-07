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

from contextlib import ExitStack
import itertools
import logging
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Iterator

from googlehydrology.datasetzoo import get_dataset
from googlehydrology.datasetzoo.multimet import MultimetDataLoader
from googlehydrology.datautils.utils import (
    get_frequency_factor,
    load_basin_file,
    sort_frequencies,
)
from googlehydrology.evaluation import plots
from googlehydrology.evaluation.assimilation import Assimilation
from googlehydrology.evaluation.metrics import (
    calculate_metrics,
    get_available_metrics,
)
from googlehydrology.evaluation.utils import (
    BasinBatchSampler,
    get_samples_indexes,
    metrics_to_dataframe,
)
from googlehydrology.modelzoo import get_model
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_regularization_obj
from googlehydrology.training.logger import Logger, do_log_figures
from googlehydrology.utils.config import Config, TesterSamplesReduction
from googlehydrology.utils.errors import AllNaNError
from googlehydrology.utils.gfile_utils import (
    get_gfile,
    gfile_exists,
    gfile_glob,
    gfile_makedirs,
    gfile_open,
    gfile_rmtree,
    torch_safe_load,
    torch_safe_save,
)
from googlehydrology.utils.tqdm import AutoRefreshTqdm as tqdm
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
import torch.cuda
from torch.utils.data import Dataset
import xarray
import zarr

LOGGER = logging.getLogger(__name__)


class BaseTester(object):
  """Base class to run inference on a model.

    Use subclasses of this class to evaluate a trained model on its train, test, or validation period.
    For regression settings, `RegressionTester` is used; for uncertainty prediction, `UncertaintyTester`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

  def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
    self.cfg = cfg
    self.run_dir = run_dir
    self.init_model = init_model
    if period in ['train', 'validation', 'test']:
      self.period = period
    else:
      raise ValueError(
                f'Invalid period {period}. Must be one of ["train", "validation", "test"]'
            )

    # determine device
    self._set_device()

    if self.init_model:
      self.model = get_model(cfg).to(self.device)

    self._disable_pbar = cfg.verbose == 0

    # pre-initialize variables, defined in class methods
    self.basins = None

    # initialize loss object to compute the loss of the evaluation data
    self.loss_obj = get_loss_obj(cfg)
    self.loss_obj.set_regularization_terms(
            get_regularization_obj(cfg=self.cfg)
        )

    self._load_run_data()  # Sets self.basins

    self.dataset = self._get_dataset_all()

    exclude_basins = set(self._calc_exclude_basins())  # Needs self.dataset
    self.basins = [e for e in self.basins if e not in exclude_basins]

    # initialize assimilation object
    self.assimilation = None
    if getattr(self.cfg, 'assimilation_config', None) is not None:
      assim_cfg = self.cfg.assimilation_config
      # Config.lead_time always exists as a property, so a getattr default would
      # be dead code; fall back to the dataset only when the config value is 0.
      lead_time = getattr(self.cfg, 'lead_time', 0) or getattr(
          self.dataset, 'lead_time', 0
      )
      if lead_time and getattr(assim_cfg, 'assimilation_lead_time', 0) == 0:
        LOGGER.warning(
            'assimilation_lead_time was 0; aligning it to the model lead time '
            '(%d) so that data assimilation cannot be fitted on the days that '
            'are scored as forecast skill.',
            lead_time,
        )
        if isinstance(assim_cfg, dict):
          assim_cfg['assimilation_lead_time'] = lead_time
        elif hasattr(assim_cfg, '_cfg') and isinstance(assim_cfg._cfg, dict):
          assim_cfg._cfg['assimilation_lead_time'] = lead_time
      self.assimilation = Assimilation(assim_cfg)

  def _set_device(self):
    if self.cfg.device is not None:
      if self.cfg.device.startswith('cuda'):
        gpu_id = int(self.cfg.device.split(':')[-1])
        if gpu_id > torch.cuda.device_count():
          raise RuntimeError(
                        f'This machine does not have GPU #{gpu_id} '
                    )
        else:
          self.device = torch.device(self.cfg.device)
      elif self.cfg.device == 'mps':
        if torch.backends.mps.is_available():
          self.device = torch.device('mps')
        else:
          raise RuntimeError('MPS device is not available.')
      else:
        self.device = torch.device('cpu')
    else:
      if torch.cuda.is_available():
        self.device = torch.device('cuda:0')
      elif torch.backends.mps.is_available():
        self.device = torch.device('mps')
      else:
        self.device = torch.device('cpu')

  def _load_run_data(self):
    """Load run specific data from run directory"""

    # get list of basins
    self.basins = load_basin_file(
            getattr(self.cfg, f'{self.period}_basin_file')
        )

  def _get_weight_file(self, epoch: int | None):
    """Get file path to weight file"""
    if epoch is None:
      weight_files = sorted(gfile_glob(f'{self.run_dir}/model_epoch*.pt'))
      if not weight_files:
        raise RuntimeError(
            f'No weights found matching {self.run_dir}/model_epoch*.pt'
        )
      weight_file = Path(weight_files[-1])
    else:
      weight_file = self.run_dir / f'model_epoch{str(epoch).zfill(3)}.pt'

    return weight_file

  def _load_weights(self, epoch: int = None):
    """Load weights of a certain (or the last) epoch into the model."""
    weight_file = self._get_weight_file(epoch)

    LOGGER.info('Using the model weights from %s', weight_file)
    state_dict = torch_safe_load(
        weight_file, map_location=self.device, weights_only=True
    )
    # Drop `_orig_mod.` prefix introduced by torch.compile when the model
    # was trained with compile=True.
    if any(k.startswith('_orig_mod.') for k in state_dict):
      state_dict = {
          (k[len('_orig_mod.') :] if k.startswith('_orig_mod.') else k): v
          for k, v in state_dict.items()
      }
    target_model = getattr(self.model, '_orig_mod', self.model)
    target_model.load_state_dict(state_dict)

  def _get_dataset_all(self) -> Dataset:
    """Get dataset for all basins."""
    return get_dataset(
        cfg=self.cfg,
        is_train=False,
        period=self.period,
        basins=None,
        compute_scaler=False,
    )

  def evaluate(
      self,
      epoch: int = None,
      save_results: bool = True,
      metrics: list | dict = [],
      model: torch.nn.Module = None,
      experiment_logger: Logger = None,
      data_assimilation: bool = False,
  ) -> dict:
    """Evaluate the model.

    Parameters
    ----------
    epoch : int, optional
        Define a specific epoch to evaluate. By default, the weights of the last
        epoch are used.
    save_results : bool, optional
        If True, stores the evaluation results in the run directory. By default,
        True.
    metrics : list | dict, optional
        List of metrics to compute during evaluation. Can also be a dict that
        specifies per-target metrics
    model : torch.nn.Module, optional
        If a model is passed, this is used for validation.
    experiment_logger : Logger, optional
        Logger can be passed during training to log metrics
    data_assimilation : bool, optional
        If True, runs evaluation with data assimilation. Default is False.
    """
    if not data_assimilation and getattr(self.cfg, 'assimilate', False):
      data_assimilation = True

    if self.cfg.seed is not None:
      random.seed(self.cfg.seed)
      np.random.seed(self.cfg.seed)
      torch.manual_seed(self.cfg.seed)

    if model is None:
      if self.init_model:
        self._load_weights(epoch=epoch)
        model = self.model
      else:
        raise RuntimeError('No model was initialized for the evaluation')

    # during validation, depending on settings, only evaluate on a random subset of basins
    basins = self.basins
    if (
            self.period == 'validation'
            and len(basins) > self.cfg.validate_n_random_basins
        ):
      basins = random.sample(basins, k=self.cfg.validate_n_random_basins)

    # force model to train-mode when doing mc-dropout evaluation
    if self.cfg.mc_dropout:
      model.train()
    else:
      model.eval()

    batch_sampler = BasinBatchSampler(
            sample_index=self.dataset._sample_index,
            batch_size=self.cfg.batch_size,
            basins_indexes=get_samples_indexes(
                self.basins, samples=list(basins)
            ),
        )
    loader = MultimetDataLoader(
            self.dataset,
            lazy_load=self.cfg.lazy_load,
            logging_level=self.cfg.logging_level,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
            pin_memory=True,  # avoid 1 of 2 mem copies to gpu
        )

    max_figures = min(
            self.cfg.validate_n_random_basins,
            self.cfg.log_n_figures,
            len(basins),
        )
    basins_for_figures = random.sample(list(basins), k=max_figures)

    eval_data_it = self._evaluate(
        model,
        loader,
        self.dataset.frequencies,
        basins,
        data_assimilation=data_assimilation,
    )
    pbar = tqdm(
        eval_data_it,
        file=sys.stdout,
        disable=self._disable_pbar,
        total=len(basins),
    )
    if self.period == 'validation':
      pbar.set_description('# Validation')
    else:
      pbar.set_description(
                '# Inference' if self.cfg.inference_mode else '# Evaluation'
            )

    self._ensure_no_previous_results_saved(
        epoch, data_assimilation=data_assimilation
    )

    metrics_results = {}

    for basin_data in pbar:
      results = {}

      basin = basin_data['basin']
      y_hat = basin_data['preds']
      y = basin_data['obs']
      dates = basin_data['dates']
      all_losses = basin_data['mean_losses']

      # log loss of this basin plus number of samples in the logger to compute epoch aggregates later
      if experiment_logger is not None:
        experiment_logger.log_step(
            **{k: (v, len(loader)) for k, v in all_losses.items()}
        )

      predict_last_n = self.cfg.predict_last_n
      seq_length = self.cfg.seq_length
      # if predict_last_n/seq_length are int, there's only one frequency
      if isinstance(predict_last_n, int):
        predict_last_n = {self.dataset.frequencies[0]: predict_last_n}
      if isinstance(seq_length, int):
        seq_length = {self.dataset.frequencies[0]: seq_length}
      lowest_freq = sort_frequencies(self.dataset.frequencies)[0]

      for freq in self.dataset.frequencies:
        if predict_last_n[freq] == 0:
          continue  # this frequency is not being predicted
        results.setdefault(freq, {})

        # Create data_vars dictionary for the xarray.Dataset
        data_vars = self._create_xarray_data_vars(y_hat[freq], y[freq])

        # freq_range are the steps of the current frequency at each lowest-frequency step
        frequency_factor = int(get_frequency_factor(lowest_freq, freq))

        # Create coords dictionary for the xarray.Dataset. 'date' can be directly infered from the dates
        # dictionary. We index the sample by the date of the last timestep of the sequence. The 'time_step'
        # index that specifies the position in the output sequence (relative to the end) can be inferred by
        # computing the timedelta of the dates. To account for predict_last_n > 1 and multi-freq stuff, we
        # need to add the frequency factor and remove 1 (to start at zero). If this is a forecast model,
        # `date` should refer to the issue dates and the `time_step` coordinates should be positive for
        # positive lead times (negative for any lookback into the hindcast).
        time_step_coords = (
                    (
                        (dates[freq][0, :] - dates[freq][0, -1])
                        / pd.Timedelta(freq)
                    ).astype(np.int64)
                    + frequency_factor
                    - 1
                )
        date_coords = dates[lowest_freq][:, -1]
        # TODO (future) : As in all of the forecast models (but not `Multimet`), this assumes
        # that all lead times are present from 1 to `self.dataset.lead_time`.
        if (
                    hasattr(self.dataset, 'lead_time')
                    and self.dataset.lead_time
                ):
          time_step_coords += self.dataset.lead_time
          date_coords = dates[lowest_freq][
                        :, -self.dataset.lead_time - 1
                    ]
        coords = {'date': date_coords, 'time_step': time_step_coords}
        xr = xarray.Dataset(data_vars=data_vars, coords=coords)

        period_start = getattr(self.cfg, f'{self.period}_start_date', None)
        if isinstance(period_start, (list, tuple)) and len(period_start) > 0:
          period_start = period_start[0]
        period_end = getattr(self.cfg, f'{self.period}_end_date', None)
        if isinstance(period_end, (list, tuple)) and len(period_end) > 0:
          period_end = period_end[0]

        start_dt = (
            period_start if period_start is not None else xr['date'].values[0]
        )
        end_dt = period_end if period_end is not None else xr['date'].values[-1]
        xr = xr.reindex({
            'date': pd.DatetimeIndex(
                pd.date_range(
                    start_dt,
                    end_dt,
                    freq=lowest_freq,
                ),
                name='date',
            )
        })
        xr = self.dataset.scaler.unscale(xr)
        results[freq]['xr'] = xr

        # create datetime range at the current frequency
        freq_date_range = pd.date_range(
            start=dates[lowest_freq][0, -1],
            end=dates[freq][-1, -1],
            freq=freq,
        )
        # remove datetime steps that are not being predicted from the datetime range
        mask = np.ones(frequency_factor).astype(bool)
        mask[: -predict_last_n[freq]] = False
        freq_date_range = freq_date_range[
                    np.tile(mask, len(xr['date']))
                ]

        # only warn once per freq
        if frequency_factor < predict_last_n[freq] and basin == next(
                    iter(basins)
                ):
          tqdm.write(
                        f'Metrics for {freq} are calculated over last {frequency_factor} elements only. '
                        f'Ignoring {predict_last_n[freq] - frequency_factor} predictions per sequence.'
                    )

        if metrics:
          for target_variable in self.cfg.target_variables:
            var_metrics = (
                metrics
                if isinstance(metrics, list)
                else metrics[target_variable]
            )
            if 'all' in var_metrics:
              var_metrics = get_available_metrics()

            is_forecast_lead = (
                hasattr(self.dataset, 'lead_time')
                and self.dataset.lead_time > 1
            ) or (frequency_factor == 1 and predict_last_n[freq] > 1)

            if is_forecast_lead:
              num_lead_steps = (
                  self.dataset.lead_time
                  if (hasattr(self.dataset, 'lead_time') and self.dataset.lead_time)
                  else predict_last_n[freq]
              )
              per_lead_metrics = {}
              lead_days_to_eval = list(range(1, num_lead_steps + 1))
              if predict_last_n[freq] > num_lead_steps:
                lead_days_to_eval = [0] + lead_days_to_eval

              for lead_day in lead_days_to_eval:
                if lead_day == 0:
                  step_idx = -predict_last_n[freq]
                else:
                  step_idx = -num_lead_steps + (lead_day - 1)
                time_slice = slice(
                    step_idx, step_idx + 1 if (step_idx + 1) != 0 else None
                )
                obs_k = (
                    xr.isel(time_step=time_slice)
                    .stack(datetime=['date', 'time_step'])
                    .drop_vars({'datetime', 'date', 'time_step'})[
                        f'{target_variable}_obs'
                    ]
                )
                obs_k['datetime'] = freq_date_range

                if not obs_k.notnull().any():
                  per_lead_metrics[lead_day] = {
                      m: np.nan for m in var_metrics
                  }
                  continue

                sim_k = (
                    xr.isel(time_step=time_slice)
                    .stack(datetime=['date', 'time_step'])
                    .drop_vars({'datetime', 'date', 'time_step'})[
                        f'{target_variable}_sim'
                    ]
                )
                sim_k['datetime'] = freq_date_range

                if target_variable in self.cfg.clip_targets_to_zero:
                  sim_k = xarray.where(sim_k < 0, 0, sim_k)

                sim_k = self._reduce_samples(sim_k)

                try:
                  m_vals = calculate_metrics(
                      obs_k,
                      sim_k,
                      metrics=var_metrics,
                      resolution=freq,
                  )
                except AllNaNError as err:
                  LOGGER.warning(f'Basin {basin} L{lead_day} metric error: {err}')
                  m_vals = {m: np.nan for m in var_metrics}

                per_lead_metrics[lead_day] = m_vals

              # Assemble composite metrics dictionary:
              # 1. Base / Lead 1 metric (unprefixed, for backward compatibility)
              # 2. Per-lead metrics (e.g. NSE_lead1 ... NSE_lead7)
              # 3. Forecast horizon mean across all lead times (e.g. NSE_mean_lead1_7)
              # `lead0` is the last HINDCAST step. Under data assimilation it lies
              # inside the assimilation window, so it is an in-sample score and must
              # never be read as forecast skill. It is suffixed accordingly and is
              # excluded from the forecast-horizon mean.
              values = {}
              for m in var_metrics:
                if 1 in per_lead_metrics and m in per_lead_metrics[1]:
                  values[m] = per_lead_metrics[1][m]
                if 0 in per_lead_metrics and m in per_lead_metrics[0]:
                  values[f'{m}_lead0_in_window'] = per_lead_metrics[0][m]
                for l_day in range(1, num_lead_steps + 1):
                  values[f'{m}_lead{l_day}'] = per_lead_metrics.get(
                      l_day, {}
                  ).get(m, np.nan)
                lead_vals = [
                    values[f'{m}_lead{l_day}']
                    for l_day in range(1, num_lead_steps + 1)
                    if not np.isnan(values.get(f'{m}_lead{l_day}', np.nan))
                ]
                values[f'{m}_mean_lead1_{num_lead_steps}'] = (
                    float(np.mean(lead_vals)) if lead_vals else np.nan
                )
            else:
              # Standard single-step / multi-frequency evaluation
              obs = (
                  xr.isel(
                      time_step=slice(
                          -predict_last_n[freq],
                          -predict_last_n[freq] + 1,
                      )
                  )
                  .stack(datetime=['date', 'time_step'])
                  .drop_vars({'datetime', 'date', 'time_step'})[
                      f'{target_variable}_obs'
                  ]
              )
              obs['datetime'] = freq_date_range
              if obs.notnull().any():
                sim = (
                    xr.isel(
                        time_step=slice(
                            -predict_last_n[freq],
                            -predict_last_n[freq] + 1,
                        )
                    )
                    .stack(datetime=['date', 'time_step'])
                    .drop_vars({'datetime', 'date', 'time_step'})[
                        f'{target_variable}_sim'
                    ]
                )
                sim['datetime'] = freq_date_range

                if target_variable in self.cfg.clip_targets_to_zero:
                  sim = xarray.where(sim < 0, 0, sim)

                sim = self._reduce_samples(sim)

                try:
                  values = calculate_metrics(
                      obs,
                      sim,
                      metrics=var_metrics,
                      resolution=freq,
                  )
                except AllNaNError as err:
                  msg = (
                      f'Basin {basin} '
                      + (
                          f'{target_variable} '
                          if len(self.cfg.target_variables) > 1
                          else ''
                      )
                      + (
                          f'{freq} '
                          if len(self.dataset.frequencies) > 1
                          else ''
                      )
                      + str(err)
                  )
                  LOGGER.warning(msg)
                  values = {
                      metric: np.nan for metric in var_metrics
                  }
              else:
                values = {metric: np.nan for metric in var_metrics}

            # add variable identifier to metrics if needed
            if len(self.cfg.target_variables) > 1:
              values = {
                  f'{target_variable}_{key}': val
                  for key, val in values.items()
              }
            # add frequency identifier to metrics if needed
            if len(self.dataset.frequencies) > 1:
              values = {
                  f'{key}_{freq}': val for key, val in values.items()
              }
            if experiment_logger is not None:
              experiment_logger.log_step(**values)
            results[freq].update(values)


      if basin in basins_for_figures:
        self._create_and_log_figures(
                    basin, results, experiment_logger, epoch or -1
                )

      self._save_incremental_results(
          basin,
          results=results,
          states={},
          save_results=save_results,
          epoch=epoch,
          data_assimilation=data_assimilation,
      )

      if metrics and not experiment_logger:
        for freq, freq_metrics in results.items():
          for name, metric in freq_metrics.items():
            if name == 'xr':
              continue
            metrics_results.setdefault(freq, {}).setdefault(name, []).append(
                metric
            )

      del results
      del basin_data
      import gc

      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if metrics and not experiment_logger:
      for freq, freq_metrics in metrics_results.items():
        for name, metric in freq_metrics.items():
          median = np.nanmedian(metric)
          LOGGER.info('%s %s median=%f', freq, name, median)

    # Consolidate metadata for the output Zarr store if one was created
    if self.cfg.inference_mode and self.period == 'test' and save_results:
      parent_directory = self._parent_directory_for_results(epoch)
      for zarr_name in [f'{self.period}_results.zarr', f'{self.period}_results_data_assimilation.zarr']:
        result_file = parent_directory / zarr_name
        if gfile_exists(result_file):
          try:
            zarr.consolidate_metadata(str(result_file))
            LOGGER.debug('Consolidated metadata for %s', result_file)
          except Exception as e:
            LOGGER.warning(
                'Could not consolidate metadata for %s: %s', result_file, e
            )

  def _calc_exclude_basins(self) -> Iterator[str]:
    if not self.cfg.tester_skip_obs_all_nan:
      return

    period_start, period_end = (
        self.cfg.test_start_date,
        self.cfg.test_end_date,
    )
    if self.period == 'validation':
      period_start, period_end = (
                self.cfg.validation_start_date,
                self.cfg.validation_end_date,
            )

    if self.cfg.lazy_load:
      LOGGER.warning(
                'tester_skip_obs_all_nan combined with lazy_load may be slow, '
                'it goes over all the data.'
            )
    # TODO(future): this may be optimized to work vectorically via xarray on all
    # basins at once.
    for basin in self.basins:
      basin_ds = self.dataset._dataset.sel(basin=basin)
      # Calculate all-nan ranges
      diffs = np.diff(
                basin_ds.streamflow.isnull(), prepend=[0], append=[0]
            )
      (starts,), (ends,) = np.where(diffs == 1), np.where(diffs == -1)

      nan_date_starts = basin_ds.date.data[starts]
      nan_date_ends = basin_ds.date.data[ends - 1]
      for start, end in zip(period_start, period_end):
        if np.any((nan_date_starts <= start) & (nan_date_ends >= end)):
          yield basin

  def _create_and_log_figures(
        self,
        basin: str,
        results: dict,
        experiment_logger: Logger | None,
        epoch: int,
    ):
    for target_var in self.cfg.target_variables:
      for freq in results:
        xr = results[freq]['xr']
        obs = xr[f'{target_var}_obs'].values
        sim = xr[f'{target_var}_sim'].values
        # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
        if target_var in self.cfg.clip_targets_to_zero:
          sim = xarray.where(sim < 0, 0, sim)
        figures = [
                    self._get_plots(
                        obs,
                        sim,
                        title=f'{target_var} - Basin {basin} - Epoch {epoch} - Frequency {freq}',
                    )[0],
                ]
        # make sure the preamble is a valid file name
        preamble = re.sub(r'[^A-Za-z0-9\._\-]+', '', target_var)
        if experiment_logger:
          experiment_logger.log_figures(
                        figures, freq, preamble, self.period, basin
                    )
        else:
          do_log_figures(
                        None,
                        self.cfg.img_log_dir,
                        epoch,
                        figures,
                        freq,
                        preamble,
                        self.period,
                        basin,
                    )

  def _ensure_no_previous_results_saved(
      self, epoch: int | None = None, data_assimilation: bool = False
  ):
    parent_directory = self._parent_directory_for_results(epoch)

    suffix = '_data_assimilation' if data_assimilation else ''
    zarr_stores_to_remove = [
        parent_directory / f'{self.period}_results{suffix}.zarr',
    ]
    for zarr_store in zarr_stores_to_remove:
      gfile_rmtree(zarr_store)

    metrics_csv_path = parent_directory / f'{self.period}_metrics{suffix}.csv'
    if gfile_exists(metrics_csv_path):
      try:
        gf = get_gfile()
        if gf:
          gf.Remove(str(metrics_csv_path))
        else:
          metrics_csv_path.unlink()
      except Exception:
        pass

  def _save_incremental_results(
      self,
      basin: str,
      *,
      results: dict,
      states: dict,
      save_results: bool,
      epoch: int | None,
      data_assimilation: bool = False,
  ):
    """Store results in various formats to disk.

    Developer note: We cannot store the time series data (the xarray objects) as
    netCDF file but have to use
    pickle as a wrapper. The reason is that netCDF files have special
    constraints on the characters/symbols that can
    be used as variable names. However, for convenience we will store metrics,
    if calculated, in a separate csv-file.
    """
    parent_directory = self._parent_directory_for_results(epoch)

    # save metrics any time this function is called, as long as they exist
    if self.cfg.metrics and results:
      metrics_list = self.cfg.metrics
      if isinstance(metrics_list, dict):
        metrics_list = list(set(metrics_list.values()))
      if 'all' in metrics_list:
        metrics_list = get_available_metrics()
      df = metrics_to_dataframe(
                {basin: results}, metrics_list, self.cfg.target_variables
            )
      suffix = '_data_assimilation' if data_assimilation else ''
      metrics_file = parent_directory / f'{self.period}_metrics{suffix}.csv'
      try:
        if not gfile_exists(metrics_file):
          with gfile_open(metrics_file, 'w') as f:
            df.to_csv(f, header=True)
        else:
          with gfile_open(metrics_file, 'a') as f:
            df.to_csv(f, header=False)
      except Exception:
        df.to_csv(metrics_file, mode='a', header=not gfile_exists(metrics_file))

    # store all results in a zarr store
    if (
        results
        and save_results
        and self.cfg.inference_mode
        and self.period == 'test'
    ):
      suffix = '_data_assimilation' if data_assimilation else ''
      result_file = parent_directory / f'{self.period}_results{suffix}.zarr'

      dss = (
          freq_results['xr'].assign_coords(freq=freq)
          for freq, freq_results in results.items()
      )
      ds = xarray.concat(dss, dim='freq').expand_dims(basin=[basin])
      ds = _ensure_unicode_or_bytes_are_strings(ds)

      if gfile_exists(result_file):
        ds.to_zarr(result_file, append_dim='basin', consolidated=False)
      else:
        ds.to_zarr(result_file, mode='w', consolidated=False)

  def _parent_directory_for_results(self, epoch: int | None = None):
    # determine parent directory name and create if needed
    weight_file = self._get_weight_file(epoch=epoch)
    parent_directory = self.run_dir / self.period / weight_file.stem
    gfile_makedirs(parent_directory, exist_ok=True)
    return parent_directory

  def _evaluate(
      self,
      model: BaseModel,
      loader: MultimetDataLoader,
      frequencies: list[str],
      basins: set[str] = set(),
      data_assimilation: bool = False,
  ):
    predict_last_n = self.cfg.predict_last_n
    if isinstance(predict_last_n, int):
      predict_last_n = {
          frequencies[0]: predict_last_n
      }  # if predict_last_n is int, there's only one frequency

    if data_assimilation and self.assimilation is None:
      raise ValueError(
          'Data assimilation was requested but no assimilation object was '
          'constructed. Provide `assimilation_config` in the run config.'
      )

    with ExitStack() as stack:
      if not data_assimilation:
        stack.enter_context(torch.inference_mode())

      basin_samples = itertools.groupby(
          loader, lambda data: data['basin_index'][0].item()
      )
      for basin_index, samples in basin_samples:
        basin = loader.dataset._basins[basin_index]
        if basin not in basins:
          continue

        preds = {}
        obs = {}
        dates = {}
        losses = []
        mean_losses = {}

        for data in samples:
          for key in data:
            if key.startswith('x_d'):
              data[key] = {
                  k: (
                      v.to(self.device).float()
                      if torch.is_floating_point(v)
                      else v.to(self.device)
                  )
                  for k, v in data[key].items()
              }
            elif not key.startswith('date'):
              data[key] = (
                  data[key].to(self.device).float()
                  if torch.is_floating_point(data[key])
                  else data[key].to(self.device)
              )

          with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
            data = model.pre_model_hook(data, is_train=False)
            predictions, loss = self._get_predictions_and_loss(
                model, data, data_assimilation=data_assimilation
            )

          for freq in frequencies:
            if predict_last_n[freq] == 0:
              continue  # no predictions for this frequency
            freq_key = '' if len(frequencies) == 1 else f'_{freq}'
            y_hat_sub, y_sub = self._subset_targets(
                model,
                data,
                predictions,
                predict_last_n[freq],
                freq_key,
            )
            # Date subsetting is universal across all models and thus happens here.
            date_sub = data[f'date{freq_key}'][
                            :, -predict_last_n[freq] :
                        ]

            y_hat_cpu = (
                y_hat_sub.detach().cpu()
                if isinstance(y_hat_sub, torch.Tensor)
                else y_hat_sub
            )
            y_cpu = (
                y_sub.detach().cpu()
                if isinstance(y_sub, torch.Tensor)
                else y_sub
            )

            if freq not in preds:
              preds[freq] = [y_hat_cpu]
              obs[freq] = [y_cpu]
              dates[freq] = [date_sub]
            else:
              preds[freq].append(y_hat_cpu)
              obs[freq].append(y_cpu)
              dates[freq].append(date_sub)

          del data
          del predictions
          del y_hat_sub
          del y_sub
          losses.append(loss)

        # Concatenate accumulated prediction tensors on CPU
        preds = {
            k: (
                torch.cat(v, dim=0)
                if isinstance(v[0], torch.Tensor)
                else np.concatenate(v, axis=0)
            )
            for k, v in preds.items()
            if v
        }
        obs = {
            k: (
                torch.cat(v, dim=0)
                if isinstance(v[0], torch.Tensor)
                else np.concatenate(v, axis=0)
            )
            for k, v in obs.items()
            if v
        }
        dates = {k: np.concatenate(v, axis=0) for k, v in dates.items() if v}

        # set to NaN explicitly if all losses are NaN to avoid RuntimeWarning
        if len(losses) == 0 or not isinstance(losses[0], dict):
          mean_losses['loss'] = np.nan
        else:
          for loss_name in losses[0].keys():
            loss_values = [
                loss[loss_name]
                for loss in losses
                if isinstance(loss, dict) and loss_name in loss
            ]
            mean_losses[loss_name] = (
                np.nanmean(loss_values)
                if loss_values and not np.all(np.isnan(loss_values))
                else np.nan
            )

        res = {
            'basin': basin,
            'preds': _values_to_cpu(preds),
            'obs': _values_to_cpu(obs),
            'dates': dates,
            'losses': losses,
            'mean_losses': mean_losses,
        }
        if torch.cuda.is_available():  # Await gpu to cpu copies
          torch.cuda.synchronize()
        yield res

  def _reduce_samples(self, sim: xarray.DataArray) -> xarray.DataArray:
    """Reduces the probabilistic 'samples' dimension to a point estimate.

    This is the single source of truth for the sample reduction. It is a
    no-op for deterministic heads, which do not emit a 'samples' dimension.

    Note that for the `cmal_deterministic` head the 'samples' axis is NOT a
    set of draws. It is the fixed 10-point summary
    `[mixture_mean, q0.1, ..., q0.9]` emitted by
    `cmal_deterministic.generate_predictions`, so MEAN and MEDIAN blend
    heterogeneous statistics together. Use MIXTURE_MEAN to read the exact
    conditional mean E[X] at index 0, which is the minimizer of the
    MSE-based metrics (NSE, RMSE) this pipeline reports.

    Args:
      sim: Simulated values, optionally carrying a 'samples' dimension.

    Returns:
      `sim` with the 'samples' dimension reduced away, if it was present.

    Raises:
      KeyError: If `cfg.tester_sample_reduction` is not a supported value.
      ValueError: If MIXTURE_MEAN is requested for a head whose 'samples'
        axis does not carry the mixture mean at index 0.
    """
    if 'samples' not in sim.dims:
      return sim
    match self.cfg.tester_sample_reduction:
      case TesterSamplesReduction.MEAN:
        return sim.mean(dim='samples')
      case TesterSamplesReduction.MEDIAN:
        if getattr(self.cfg, 'head', '').lower() == 'cmal_deterministic':
          return sim.isel(samples=5, drop=True)
        return sim.median(dim='samples')
      case TesterSamplesReduction.MIXTURE_MEDIAN:
        head = self.cfg.head.lower()
        if head != 'cmal_deterministic':
          msg = (
              'tester_sample_reduction=mixture_median selects index 5 (q_0.5)'
              ' of the samples axis, which only holds q_0.5 for the'
              f' cmal_deterministic head; got head={head!r}.'
          )
          raise ValueError(msg)
        return sim.isel(samples=5, drop=True)
      case TesterSamplesReduction.MIXTURE_MEAN:
        head = self.cfg.head.lower()
        if head != 'cmal_deterministic':
          msg = (
              'tester_sample_reduction=mixture_mean selects index 0 of the'
              ' samples axis, which only holds the CMAL mixture mean for the'
              f' cmal_deterministic head; got head={head!r}. For that head the'
              ' samples axis is a 10-point summary; for other heads it holds'
              ' genuine random draws and index 0 is meaningless.'
          )
          raise ValueError(msg)
        return sim.isel(samples=0, drop=True)
      case _:
        msg = f'Supported {self.cfg.tester_sample_reduction=}'
        raise KeyError(msg)

  def _get_predictions_and_loss(
      self,
      model: BaseModel,
      data: dict[str, torch.Tensor],
      data_assimilation: bool = False,
  ) -> tuple[torch.Tensor, float]:
    if data_assimilation:
      predictions = self.assimilation.assimilate(model, data)
      _, all_losses = self.loss_obj(predictions, data)
      return predictions, {k: v.item() for k, v in all_losses.items()}
    predictions = model(data)
    _, all_losses = self.loss_obj(predictions, data)
    return predictions, {k: v.item() for k, v in all_losses.items()}

  def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str,
    ):
    raise NotImplementedError

  def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
    raise NotImplementedError

  def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
    raise NotImplementedError


class RegressionTester(BaseTester):
    """Tester class to run inference on a regression model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
        super(RegressionTester, self).__init__(cfg, run_dir, period, init_model)

    def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: np.ndarray,
        freq: str,
    ):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f'{var}_obs'] = (('date', 'time_step'), y[:, :, i])
            data[f'{var}_sim'] = (('date', 'time_step'), y_hat[:, :, i])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.regression_plot(qobs, qsim, title)


class UncertaintyTester(BaseTester):
    """Tester class to run inference on an uncertainty model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        period: str = 'test',
        init_model: bool = True,
    ):
        super(UncertaintyTester, self).__init__(
            cfg, run_dir, period, init_model
        )

    def _get_predictions_and_loss(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        data_assimilation: bool = False,
    ) -> tuple[torch.Tensor, float]:
        if data_assimilation:
            predictions = self.assimilation.assimilate(model, data)
            _, all_losses = self.loss_obj(predictions, data)
            # Sample unconditionally, exactly as the non-DA branch below does.
            # Gating this on an allowlist of heads made the DA path return the
            # raw (3-D) forward output for any other head, which then broke the
            # 4-D indexing in `_create_xarray_data_vars`.
            predictions = model.sample(
                data, self.cfg.n_samples, outputs=predictions
            )
            model.eval()
            return predictions, {k: v.item() for k, v in all_losses.items()}
        outputs = model(data)
        _, all_losses = self.loss_obj(outputs, data)
        predictions = model.sample(data, self.cfg.n_samples, outputs=outputs)
        model.eval()
        return predictions, {k: v.item() for k, v in all_losses.items()}

    def _subset_targets(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        predictions: np.ndarray,
        predict_last_n: int,
        freq: str = None,
    ):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray_data_vars(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f'{var}_obs'] = (('date', 'time_step'), y[:, :, i])
            data[f'{var}_sim'] = (
                ('date', 'time_step', 'samples'),
                y_hat[:, :, i, :],
            )
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.uncertainty_plot(qobs, qsim, title)


def _ensure_unicode_or_bytes_are_strings(ds: xarray.Dataset):
    updates = {
        name: coord.astype('O')
        for name, coord in ds.coords.items()
        if coord.dtype.kind in ('U', 'S')
    }
    return ds.assign_coords(updates)


def _values_to_cpu(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to('cpu', non_blocking=True) for k, v in x.items()}
