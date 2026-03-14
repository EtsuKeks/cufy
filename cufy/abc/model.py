from abc import ABC, abstractmethod

import numpy as np

from cufy.abc.data import OptionBatch


class Model(ABC):
    @abstractmethod
    def find_initial_params(self, batch: OptionBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, batch: OptionBatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def price(self, batch: OptionBatch) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the model has been initialized with starting parameters.
        """
        raise NotImplementedError
