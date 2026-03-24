from abc import ABC, abstractmethod

import numpy as np

from cufy.abc.data import PreparedBatch


class Model(ABC):
    @abstractmethod
    def price(self, batch: PreparedBatch) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_initialized(self) -> bool:
        raise NotImplementedError
