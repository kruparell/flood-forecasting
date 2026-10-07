from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union
from ruamel.yaml import YAML

T = TypeVar("T")


class AssimilationConfig:
  """Configuration class for data assimilation arguments.

  Allowed Configuration Keys:
    - Optimization Parameters:
        - `learning_rate` (float or Dict[int, float]): Base LR or step schedule.
        - `learning_rate_drop_factor` (float): LR decay factor (default: 0.9).
        - `learning_rate_epoch_drop` (int): Frequency of LR drop (default: 5).
        - `epochs` (int): Number of optimization steps/epochs (default: 200).
        - `optimizer` (str): Optimizer name, e.g. 'Adam', 'SGD'.
        - `loss` (str): Loss function name, e.g. 'MSE'.
        - `clip_gradient_norm` (float): Max gradient norm clip (default: 1.0).
        - `early_stopping_tolerance` (float): Relative streamflow error tolerance at t0 for stopping optimization (e.g. 0.05).

    - Horizon & Evaluation Window:
        - `assimilation_window` / `assimilation_window_length` (int): Window length in days.
        - `assimilation_lead_time` (int): Target forecast lead time (0 = same-day/lead 1).
        - `history` (int): Historical context lag (typically 1).
        - `seq_length` (int): Sequence length (typically 365).
        - `predict_last_n` (int): Number of forecast steps to evaluate.
        - `predict_n_hindcast` (int): Number of hindcast steps (default: 5).
        - `use_per_step_updates` (bool): Whether to update state iteratively (default: True).

    - Assimilation Targets & Components:
        - `assimilation_targets` (List[str]): Components to optimize.
            Allowed targets: 'static_embedding', 'hindcast_embedding', 'forecast_embedding',
            'c_0_hindcast', 'h_0_hindcast', 'c_0_forecast', 'h_0_forecast', 'total_precipitation'.
            Aliases: 'embedded_both', 'embedded_all', 'embedded_dynamics', 'embedded_statics',
                     'c_both', 'h_both', 'both_states', 'all_states', 'precip'.
        - `assimilation_components` (Dict): Per-component configuration mapping.

    - Regularization Weights:
        - `regularization_weight` (float): Global/fallback regularization penalty.
        - `static_embedding_regularization_weight` / `bg_stat_weight` (float): Static embedding penalty.
        - `hindcast_embedding_regularization_weight` / `bg_dyn_weight` (float): Hindcast dynamic embedding penalty.
        - `forecast_embedding_regularization_weight` (float): Forecast dynamic embedding penalty.
        - `recurrent_state_regularization_weight` / `bg_regularization_weight` (float): LSTM hidden/cell state penalty.
        - `regularization` (List[str]): List of regularization modes.

    - Feature & Model Specific:
        - `target_variables` (List[str]): Streamflow variable names.
        - `target_loss_weights` (List[float]): Weights per target variable.
        - `model_dropout` (bool): Enable dropout during DA (default: False).
        - `timestep_dropout` (float): Timestep dropout probability (default: 0.0).
        - `no_loss_frequencies` (List[str]): Frequencies without loss.
        - `precip_forcing_keys` (List[str]): Input keys for precipitation DA.
        - `precip_min_clip` (float): Min clip value for precip updates.
  """

  _deprecated_keys = [
      "early_stopping_patience",
      "early_stopping_min_loss",
      "early_stopping_min_lr",
  ]
  _metadata_keys = []

  @classmethod
  def get_allowed_keys(cls) -> List[str]:
    """Returns the complete list of recognized top-level configuration keys."""
    return sorted([
        p for p in dir(cls) if isinstance(getattr(cls, p), property)
    ] + cls._deprecated_keys + cls._metadata_keys)

  def __init__(
      self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False
  ):
    if isinstance(yml_path_or_dict, Path):
      yaml = YAML()
      with yml_path_or_dict.open("r") as fp:
        self._cfg = yaml.load(fp)
    elif isinstance(yml_path_or_dict, dict):
      self._cfg = yml_path_or_dict.copy()
    else:
      raise ValueError(
          f"Cannot create a config from input of type {type(yml_path_or_dict)}."
      )

    if not (self._cfg.get("dev_mode", False) or dev_mode):
      self._check_cfg_keys(self._cfg)

  def _get_value_verbose(self, key: str) -> Any:
    if key not in self._cfg:
      raise ValueError(f"Key '{key}' is required in assimilation_config.")
    return self._cfg[key]

  @staticmethod
  def _as_default_list(value: Union[T, List[T], None]) -> List[T]:
    if value is None:
      return []
    if isinstance(value, list):
      return value
    return [value]

  def as_dict(self) -> dict:
    return self._cfg

  @staticmethod
  def _check_cfg_keys(cfg: dict):
    properties = [
        p
        for p in dir(AssimilationConfig)
        if isinstance(getattr(AssimilationConfig, p), property)
    ]
    unknown = [
        k
        for k in cfg
        if k not in properties
        and k not in AssimilationConfig._deprecated_keys
        and k not in AssimilationConfig._metadata_keys
    ]
    if unknown:
      raise ValueError(
          f"{unknown} are not recognized config keys.\n"
          f"Allowed config keys are: {sorted(properties)}"
      )

  @property
  def assimilation_lead_time(self) -> int:
    if "assimilation_lead_time" in self._cfg:
      return self._cfg["assimilation_lead_time"]
    if "lead_time" in self._cfg:
      return self._cfg["lead_time"]
    return 0

  @property
  def lead_time(self) -> int:
    return self.assimilation_lead_time

  @property
  def assimilation_components(self) -> Dict[str, Dict[str, Any]]:
    """Returns normalized dictionary of components: {name: {'weight': float, 'lr': Optional[float]}}."""
    raw = self._cfg.get("assimilation_components", None)
    if raw is not None:
      result = {}
      if isinstance(raw, dict):
        for k, v in raw.items():
          if isinstance(v, (int, float)):
            result[k] = {"weight": float(v)}
          elif isinstance(v, dict):
            w = float(v.get("weight", v.get("regularization_weight", 0.01)))
            lr = (
                float(v["learning_rate"])
                if "learning_rate" in v
                else (float(v["lr"]) if "lr" in v else None)
            )
            result[k] = {"weight": w, "lr": lr}
          else:
            result[k] = {"weight": 0.01}
        return result
      elif isinstance(raw, (list, tuple)):
        return {str(k): {"weight": 0.01} for k in raw}

    targets = self._as_default_list(self._cfg.get("assimilation_targets", []))
    if not targets:
      return {}

    result = {}
    bg_stat_w = getattr(self, "static_embedding_regularization_weight", getattr(self, "bg_stat_weight", 1e-6))
    bg_dyn_w = getattr(
        self, "hindcast_embedding_regularization_weight", getattr(self, "bg_dyn_weight", getattr(self, "regularization_weight", 0.01))
    )
    bg_fc_w = getattr(
        self, "forecast_embedding_regularization_weight", bg_dyn_w
    )
    bg_state_w = getattr(
        self, "recurrent_state_regularization_weight", getattr(self, "bg_regularization_weight", getattr(self, "regularization_weight", 0.01))
    )

    alias_map = {
        "embedded_all": ["static_embedding", "hindcast_embedding", "forecast_embedding"],
        "all_embeddings": ["static_embedding", "hindcast_embedding", "forecast_embedding"],
        "all": ["static_embedding", "hindcast_embedding", "forecast_embedding"],
        "embedded_both": ["static_embedding", "hindcast_embedding"],
        "both_embeddings": ["static_embedding", "hindcast_embedding"],
        "both": ["static_embedding", "hindcast_embedding"],
        "embedded_dynamics": ["hindcast_embedding"],
        "embedded_dyn": ["hindcast_embedding"],
        "dyn": ["hindcast_embedding"],
        "dynamic": ["hindcast_embedding"],
        "embedded_statics": ["static_embedding"],
        "embedded_stat": ["static_embedding"],
        "stat": ["static_embedding"],
        "static": ["static_embedding"],
        "c_both": ["c_0_hindcast", "c_0_forecast"],
        "h_both": ["h_0_hindcast", "h_0_forecast"],
        "both_states": ["c_0_hindcast", "h_0_hindcast", "c_0_forecast", "h_0_forecast"],
        "all_states": ["c_0_hindcast", "h_0_hindcast", "c_0_forecast", "h_0_forecast"],
        "precip": ["total_precipitation"],
        "precipitation": ["total_precipitation"],
    }

    expanded_targets = []
    for t in targets:
      t_key = str(t).strip().lower()
      if t_key in alias_map:
        expanded_targets.extend(alias_map[t_key])
      else:
        expanded_targets.append(str(t).strip())

    # Order matters. Recurrent-state names are matched FIRST because they carry
    # a branch qualifier that also appears in the embedding heuristics below:
    # "c_0_hindcast" contains "hindcast" and "c_0_forecast" contains "forecast".
    # With the embedding checks first, the state branch -- and therefore
    # `recurrent_state_regularization_weight` / `bg_regularization_weight` --
    # was unreachable for every canonical state name, silently routing state
    # targets to the embedding weights instead.
    state_markers = ("c_n", "h_n", "c_0", "h_0", "state")
    for t in expanded_targets:
      t_lower = t.lower()
      if any(k in t_lower for k in state_markers):
        w = bg_state_w
      elif "stat" in t_lower:
        w = bg_stat_w
      elif any(k in t_lower for k in ["fc", "forecast"]):
        w = bg_fc_w
      elif any(k in t_lower for k in ["dyn", "hindcast"]):
        w = bg_dyn_w
      else:
        w = getattr(self, "regularization_weight", 0.01)
      result[t] = {"weight": w}
    return result


  @property
  def assimilation_targets(self) -> List[str]:
    if "assimilation_targets" in self._cfg:
      targets = self._as_default_list(self._cfg["assimilation_targets"])
      if targets:
        return targets
    if "assimilation_components" in self._cfg:
      return list(self.assimilation_components.keys())
    raise ValueError(
        "At least one assimilation component or target must be specified."
    )

  @property
  def assimilation_window_length(self) -> int:
    if "assimilation_window_length" in self._cfg:
      return int(self._cfg["assimilation_window_length"])
    return self._get_value_verbose("assimilation_window")

  @property
  def assimilation_window(self) -> int:
    return self.assimilation_window_length

  @property
  def loss_window(self) -> int:
    """Number of terminal observation days evaluated in the loss function."""
    if "loss_window" in self._cfg and self._cfg["loss_window"] is not None:
      return int(self._cfg["loss_window"])
    return self.assimilation_window

  @property
  def epochs(self) -> int:
    return self._cfg.get("epochs", 200)

  @property
  def history(self) -> int:
    return self._get_value_verbose("history")

  @property
  def learning_rate(self) -> Dict[int, float]:
    if "learning_rate" in self._cfg and self._cfg["learning_rate"] is not None:
      lr = self._cfg["learning_rate"]
      return {0: lr} if isinstance(lr, (int, float)) else lr
    raise ValueError("No learning rate specified in configuration.")

  @property
  def learning_rate_drop_factor(self) -> float:
    return float(self._cfg.get("learning_rate_drop_factor", 0.9))

  @property
  def learning_rate_epoch_drop(self) -> int:
    return int(self._cfg.get("learning_rate_epoch_drop", 5))

  @property
  def loss(self) -> str:
    return self._get_value_verbose("loss")

  @property
  def model_dropout(self) -> bool:
    return bool(self._cfg.get("model_dropout", False))

  @property
  def n_distributions(self) -> int:
    """Number of mixture components, required when `loss` is CMAL.

    Only read by `MaskedCMALLoss`, which uses it as `output_size_per_target`
    to slice the head output into per-target parameter blocks. It MUST equal
    the value the model was trained with, otherwise the loss silently slices
    the wrong columns of `mu`/`b`/`tau`/`pi`. `Assimilation` validates this
    against `model.cfg.n_distributions` rather than trusting the default.

    Defaults to 3, matching the foundation model.
    """
    return int(self._cfg.get("n_distributions", 3))

  @property
  def no_loss_frequencies(self) -> List[str]:
    return self._as_default_list(self._cfg.get("no_loss_frequencies", []))

  @property
  def optimizer(self) -> str:
    return self._get_value_verbose("optimizer")

  @property
  def predict_last_n(self) -> int:
    if "predict_last_n" in self._cfg:
      val = self._cfg["predict_last_n"]
      return int(list(val.values())[0]) if isinstance(val, dict) else int(val)
    if "lead_time" in self._cfg:
      return int(self._cfg["lead_time"])
    if "assimilation_lead_time" in self._cfg and self._cfg["assimilation_lead_time"] > 0:
      return int(self._cfg["assimilation_lead_time"])
    return 1

  @property
  def regularization(self) -> List[str]:
    return self._as_default_list(self._cfg.get("regularization", []))

  @property
  def seq_length(self) -> int:
    val = self._get_value_verbose("seq_length")
    return int(list(val.values())[0]) if isinstance(val, dict) else int(val)

  @property
  def target_loss_weights(self) -> List[float]:
    return self._cfg.get("target_loss_weights", None)

  @property
  def timestep_dropout(self) -> float:
    drop = float(self._cfg.get("timestep_dropout", 0.0))
    if drop >= 1.0 or drop < 0.0:
      raise ValueError("'timestep_dropout' must be in range [0.0, 1.0).")
    return drop

  @property
  def target_variables(self) -> List[str]:
    return self._get_value_verbose("target_variables")

  @property
  def early_stopping_tolerance(self) -> Union[float, None]:
    val = self._cfg.get("early_stopping_tolerance", None)
    return float(val) if val is not None else None

  @property
  def clip_gradient_norm(self) -> float:
    return float(self._cfg.get("clip_gradient_norm", 1.0))

  @property
  def static_embedding_regularization_weight(self) -> float:
    if "static_embedding_regularization_weight" in self._cfg:
      return float(self._cfg["static_embedding_regularization_weight"])
    return float(self._cfg.get("bg_stat_weight", 1e-6))

  @property
  def bg_stat_weight(self) -> float:
    return self.static_embedding_regularization_weight

  @property
  def hindcast_embedding_regularization_weight(self) -> float:
    if "hindcast_embedding_regularization_weight" in self._cfg:
      return float(self._cfg["hindcast_embedding_regularization_weight"])
    return float(
        self._cfg.get(
            "bg_dyn_weight", self._cfg.get("regularization_weight", 0.01)
        )
    )

  @property
  def bg_dyn_weight(self) -> float:
    return self.hindcast_embedding_regularization_weight

  @property
  def forecast_embedding_regularization_weight(self) -> float:
    if "forecast_embedding_regularization_weight" in self._cfg:
      return float(self._cfg["forecast_embedding_regularization_weight"])
    return self.hindcast_embedding_regularization_weight

  @property
  def regularization_weight(self) -> float:
    if "regularization_weight" in self._cfg:
      return float(self._cfg["regularization_weight"])
    if "hindcast_embedding_regularization_weight" in self._cfg:
      return float(self._cfg["hindcast_embedding_regularization_weight"])
    if "bg_dyn_weight" in self._cfg:
      return float(self._cfg["bg_dyn_weight"])
    return 0.0

  @property
  def recurrent_state_regularization_weight(self) -> float:
    if "recurrent_state_regularization_weight" in self._cfg:
      return float(self._cfg["recurrent_state_regularization_weight"])
    return float(self._cfg.get("bg_regularization_weight", self._cfg.get("regularization_weight", 0.01)))

  @property
  def bg_regularization_weight(self) -> float:
    return self.recurrent_state_regularization_weight

  @property
  def precip_forcing_keys(self) -> List[str]:
    return self._as_default_list(self._cfg.get("precip_forcing_keys", []))

  @property
  def precip_min_clip(self) -> float:
    return float(self._cfg.get("precip_min_clip", -3.0))

  @property
  def predict_n_hindcast(self) -> int:
    return int(self._cfg.get("predict_n_hindcast", 5))

  @property
  def use_per_step_updates(self) -> bool:
    return bool(self._cfg.get("use_per_step_updates", True))

  @property
  def state_anchor(self) -> str:
    return str(self._cfg.get("state_anchor", "window_start"))
