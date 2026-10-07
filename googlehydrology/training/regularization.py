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


import torch

from googlehydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Regularization terms subclass this class by implementing the `forward` method.

    Parameters
    ----------
    cfg: Config
        The run configuration.
    name: str
        The name of the regularization term.
    weight: float, optional.
        The weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        other_model_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the regularization term.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Dictionary of predicted variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        ground_truth : dict[str, torch.Tensor]
            Dictionary of ground truth variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        other_model_data : dict[str, torch.Tensor]
            Dictionary of all remaining keys-value pairs in the prediction dictionary that are not directly linked to
            the model predictions but can be useful for regularization purposes, e.g. network internals, weights etc.

        Returns
        -------
        torch.Tensor
            The regularization value.
        """
        raise NotImplementedError


class ForecastOverlapMSERegularization(BaseRegularization):
    """Squared error regularization for penalizing differences between hindcast and forecast models.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(ForecastOverlapMSERegularization, self).__init__(
            cfg, name='forecast_overlap', weight=weight
        )

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        other_model_output: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate the squared difference between hindcast and forecast model during overlap.

        Does not work with multi-frequency models.

        Parameters
        ----------
        prediction : dict[str, torch.Tensor]
            Not used.
        ground_truth : dict[str, torch.Tensor]
            Not used.
        other_model_output : dict[str, dict[str, torch.Tensor]]
            Dictionary containing ``y_forecast_overlap`` and ``y_hindcast_overlap``, which are
            both dictionaries containing keys to relevant model outputs.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations between overlapping portions of hindcast and forecast models.

        Raises
        ------
        ValueError if y_hindcast_overlap or y_forecast_overlap is not present in model output.
        """
        loss = 0
        if 'y_hindcast_overlap' not in other_model_output:
            raise ValueError(
                'y_hindcast_overlap is not present in the model output.'
            )
        if 'y_forecast_overlap' not in other_model_output:
            raise ValueError(
                'y_forecast_overlap is not present in the model output.'
            )
        hindcast = other_model_output['y_hindcast_overlap']
        forecast = other_model_output['y_forecast_overlap']
        loss += torch.mean((hindcast - forecast) ** 2)
        return loss


class BackgroundEmbeddingRegularization(BaseRegularization):
  """L2 regularization penalizing deviation of optimized embeddings from unassimilated baseline embeddings.

  Parameters
  ----------
  cfg : Config
      The run configuration.
  weight : float, optional
      Global weight multiplier, default 1.0.
  name : str, optional
      Name of the regularization term, default 'bg_embedding'.
  """

  def __init__(
      self, cfg: Config, weight: float = 1.0, name: str = 'bg_embedding'
  ):
    super(BackgroundEmbeddingRegularization, self).__init__(
        cfg, name=name, weight=weight
    )
    self.bg_dyn_weight = float(
        getattr(
            cfg,
            'bg_dyn_weight',
            getattr(cfg, 'regularization_weight', 0.01),
        )
    )
    self.bg_stat_weight = float(getattr(cfg, 'bg_stat_weight', 1e-6))

  def forward(
      self,
      prediction: dict[str, torch.Tensor],
      ground_truth: dict[str, torch.Tensor],
      other_model_output: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    loss = 0.0
    opt_comps = other_model_output.get('optimized_components', {})
    base_comps = other_model_output.get('baseline_components', {})
    comp_weights = other_model_output.get('component_weights', {})
    for comp_name, opt_t in opt_comps.items():
      base_t = base_comps.get(comp_name)
      if opt_t is not None and base_t is not None and opt_t.requires_grad:
        w = float(comp_weights.get(comp_name, self.bg_dyn_weight))
        loss = loss + w * torch.mean((opt_t - base_t) ** 2)
    return (
        loss
        if isinstance(loss, torch.Tensor)
        else torch.tensor(loss, dtype=torch.float32)
    )



class PriorPredictionRegularization(BaseRegularization):
  """Regularization penalizing deviation of assimilated model predictions from prior unassimilated predictions.

  Parameters
  ----------
  cfg : Config
      The run configuration.
  weight : float, optional
      Penalty weight multiplier, default 1.0.
  name : str, optional
      Name of the regularization term, default 'prior_prediction'.
  """

  def __init__(
      self, cfg: Config, weight: float = 1.0, name: str = 'prior_prediction'
  ):
    super(PriorPredictionRegularization, self).__init__(
        cfg, name=name, weight=weight
    )

  def forward(
      self,
      prediction: dict[str, torch.Tensor],
      ground_truth: dict[str, torch.Tensor],
      other_model_output: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    prior_y_hat = other_model_output.get('baseline_y_hat')
    y_hat = prediction.get('y_hat')
    if prior_y_hat is None or y_hat is None:
      return torch.tensor(0.0, dtype=torch.float32)
    min_len = min(y_hat.shape[1], prior_y_hat.shape[1])
    diff = y_hat[:, :min_len, :] - prior_y_hat[:, :min_len, :]
    mask = ~torch.isnan(diff)
    if mask.any():
      return torch.mean(diff[mask] ** 2)
    return torch.tensor(0.0, dtype=torch.float32)

