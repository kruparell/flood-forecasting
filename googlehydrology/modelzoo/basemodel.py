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
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

from googlehydrology.datautils.scaler import Scaler
from googlehydrology.utils.config import Config
from googlehydrology.utils.samplingutils import sample_pointpredictions


@dataclasses.dataclass(frozen=True)
class AssimilationTargetSpec:
  """Formal specification for an assimilable component/parameter of a model."""

  name: str
  is_time_invariant: bool
  is_recurrent: bool = False
  clamp_min: Optional[float] = None
  default_reg_weight_key: str = "regularization_weight"
  default_lr_key: str = "learning_rate"
  extract_baseline_fn: Optional[
      Callable[
          [Dict[str, Any], Dict[str, torch.Tensor]], Optional[torch.Tensor]
      ]
  ] = None
  inject_override_fn: Optional[
      Callable[[Dict[str, Any], torch.Tensor], None]
  ] = None


class BaseModel(nn.Module):
  """Abstract base model class, don't use this class for model training.

    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
    LSTM model or `EA-LSTM` for training an Entity-Aware-LSTM. Refer to  :doc:`Documentation/Modelzoo </usage/models>`
    for a full list of available models and how to integrate a new model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

  # specify submodules of the model that can later be used for finetuning. Names must match class attributes
  module_parts = []

  # Specify components that this model exposes for Data Assimilation
  supported_assimilation_components: list[str] = []

  def get_supported_assimilation_components(self) -> list[str]:
    """Returns list of component names this model supports for Data Assimilation."""
    return getattr(self, "supported_assimilation_components", [])

  target_aliases: Dict[str, List[str]] = {}

  def get_assimilation_target_specs(self) -> Dict[str, AssimilationTargetSpec]:
    """Returns mapping of supported component names to their TargetSpec definitions."""
    return {}

  def resolve_target_alias(self, alias: str) -> List[str]:
    """Resolves target alias to concrete component names supported by this model."""
    aliases = getattr(self, "target_aliases", {})
    return aliases.get(alias, [alias])

  def get_assimilation_parameters(
      self,
      data: Dict[str, Any],
      targets: Optional[List[str]] = None,
      model_outputs: Optional[Dict[str, torch.Tensor]] = None,
  ) -> Dict[str, torch.Tensor]:
    """Extracts initial baseline tensors for requested targets."""
    specs = self.get_assimilation_target_specs()
    req_targets = targets if targets is not None else list(specs.keys())
    model_outputs = model_outputs or {}

    baseline_tensors = {}
    for tgt in req_targets:
      if tgt in specs and specs[tgt].extract_baseline_fn is not None:
        tensor = specs[tgt].extract_baseline_fn(data, model_outputs)
        if tensor is not None:
          baseline_tensors[tgt] = tensor
      elif "x_d" in data and isinstance(data["x_d"], dict) and tgt in data["x_d"]:
        baseline_tensors[tgt] = data["x_d"][tgt]
    return baseline_tensors

  def inject_assimilation_overrides(
      self,
      data: Dict[str, Any],
      overrides: Dict[str, torch.Tensor],
  ) -> Dict[str, Any]:
    """Injects optimized parameter tensors into input data dict."""
    data_copy = dict(data)
    specs = self.get_assimilation_target_specs()
    for name, tensor in overrides.items():
      if name in specs and specs[name].inject_override_fn is not None:
        specs[name].inject_override_fn(data_copy, tensor)
      elif "x_d" in data_copy and isinstance(data_copy["x_d"], dict) and name in data_copy["x_d"]:
        data_copy["x_d"] = dict(data_copy["x_d"])
        data_copy["x_d"][name] = tensor
        if "x_d_hindcast" in data_copy and isinstance(data_copy["x_d_hindcast"], dict) and name in data_copy["x_d_hindcast"]:
          data_copy["x_d_hindcast"] = dict(data_copy["x_d_hindcast"])
          data_copy["x_d_hindcast"][name] = tensor
        if "x_d_forecast" in data_copy and isinstance(data_copy["x_d_forecast"], dict) and name in data_copy["x_d_forecast"]:
          data_copy["x_d_forecast"] = dict(data_copy["x_d_forecast"])
          fc = data_copy["x_d_forecast"][name].clone()
          fc[:, :tensor.shape[1], :] = tensor
          data_copy["x_d_forecast"][name] = fc
      else:
        data_copy.setdefault("assimilation_overrides", {})[name] = tensor
    return data_copy

  def __init__(self, cfg: Config):
    super(BaseModel, self).__init__()
    self.cfg = cfg
    self.output_size = len(cfg.target_variables)
    if cfg.head.lower() in ['cmal', 'cmal_deterministic']:
      self.output_size *= 4 * cfg.n_distributions
    if cfg.is_finetuning:
      scaler_dir = cfg.base_run_dir
    elif (
        getattr(cfg, 'compute_scaler', True) is False
        and hasattr(cfg, '_cfg')
        and 'base_run_dir' in cfg._cfg
    ):
      scaler_dir = cfg.base_run_dir
    else:
      scaler_dir = cfg.run_dir
    self._scaler = Scaler(
        scaler_dir=scaler_dir,
        calculate_scaler=False,
    )

  def sample(
        self,
        data: dict[str, torch.Tensor],
        n_samples: int,
        *,
        outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
    """Provides point prediction samples from a probabilistic model.

        This function wraps the `sample_pointpredictions` function, which provides different point sampling functions
        for the different uncertainty estimation approaches. There are also options to handle negative point prediction
        samples that arise while sampling from the uncertainty estimates. They can be controlled via the configuration.


        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of point predictions that ought ot be sampled form the model.
        outputs, optional
            Model forward result

        Returns
        -------
        dict[str, torch.Tensor]
            Sampled point predictions
        """
    return sample_pointpredictions(
            self, data, n_samples, self._scaler, outputs=outputs
        )

  def forward(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
    """Perform a forward pass.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        dict[str, torch.Tensor]
            Model output and potentially any intermediate states and activations as a dictionary.
        """
    raise NotImplementedError

  def pre_model_hook(
        self, data: dict[str, torch.Tensor], is_train: bool
    ) -> dict[str, torch.Tensor]:
    """A function to execute before the model in training, validation and test.
        The beahvior can be adapted depending on the run configuration and the provided arguments.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs and labels y.
        is_train : bool
            Defines if the hook is executed in train mode or in validation/test mode.

        Returns
        -------
        data : dict[str, torch.Tensor]
            The modified (or unmodified) data that are used for the training or evaluation.
        """
    # here one can implement additional pre model hooks e.g. based on head
    return data
