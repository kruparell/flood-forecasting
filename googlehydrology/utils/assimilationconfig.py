"""Configuration for Data Assimilation parameters and state updating."""

from pathlib import Path
from typing import Dict, List, Union

from googlehydrology.utils.baseconfig import BaseConfig


class AssimilationConfig(BaseConfig):
    """Configuration class for data assimilation arguments."""

    _deprecated_keys = []
    _metadata_keys = []

    def __init__(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        super(AssimilationConfig, self).__init__(yml_path_or_dict=yml_path_or_dict, dev_mode=dev_mode)
        if not (self._cfg.get('dev_mode', False) or dev_mode):
            self._check_cfg_keys(self._cfg)

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        properties = [p for p in dir(AssimilationConfig) if isinstance(getattr(AssimilationConfig, p), property)]
        unknown = [k for k in cfg if k not in properties and k not in AssimilationConfig._deprecated_keys and k not in AssimilationConfig._metadata_keys]
        if unknown:
            raise ValueError(f"{unknown} are not recognized config keys.")

    @property
    def assimilation_lead_time(self) -> int:
        return self._get_value_verbose("assimilation_lead_time")

    @property
    def assimilation_targets(self) -> List[str]:
        targets = self._as_default_list(self._cfg.get("assimilation_targets", []))
        if not targets:
            raise ValueError("At least one assimilation target must be specified.")
        return targets

    @property
    def assimilation_window(self) -> int:
        return self._get_value_verbose("assimilation_window")

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
    def no_loss_frequencies(self) -> List[str]:
        return self._as_default_list(self._cfg.get("no_loss_frequencies", []))

    @property
    def optimizer(self) -> str:
        return self._get_value_verbose("optimizer")

    @property
    def predict_last_n(self) -> int:
        return self._get_value_verbose("predict_last_n")

    @property
    def regularization(self) -> List[str]:
        return self._as_default_list(self._cfg.get("regularization", []))

    @property
    def seq_length(self) -> int:
        return self._get_value_verbose("seq_length")

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
    def early_stopping_min_lr(self) -> float:
        return float(self._cfg.get("early_stopping_min_lr", 1e-5))

    @property
    def early_stopping_min_loss(self) -> float:
        return float(self._cfg.get("early_stopping_min_loss", 1e-5))

    @property
    def early_stopping_patience(self) -> int:
        return int(self._cfg.get("early_stopping_patience", 10))

    @property
    def bg_regularization_weight(self) -> float:
        return float(self._cfg.get("bg_regularization_weight", 0.01))

    @property
    def predict_n_hindcast(self) -> int:
        return int(self._cfg.get("predict_n_hindcast", 5))

    @property
    def use_per_step_updates(self) -> bool:
        return bool(self._cfg.get("use_per_step_updates", True))