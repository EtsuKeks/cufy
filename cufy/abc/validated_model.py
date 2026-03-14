from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cufy.abc.data import OptionBatch
from cufy.abc.telemetry_model import TelemetryModel


class ValidatedModel(TelemetryModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def validation_metric(
        self,
        pred_prices: np.ndarray,
        actual_close: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute the error metric on the validation set.
        Lower is better.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_hyperparams(self, hyperparams: dict[str, Any]) -> None:
        """
        Apply the given hyperparameter dictionary to the model state.
        """
        raise NotImplementedError

    @abstractmethod
    def calibrate_inner(self, batch: OptionBatch) -> None:
        """
        Perform the actual calibration (e.g., gradient descent) on the training batch
        using the currently applied hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, batch: OptionBatch) -> None:
        """
        Orchestrate the validation split, hyperparameter search, and final calibration.
        """
        raise NotImplementedError
