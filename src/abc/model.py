from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def find_initial_params(
        self,
        *,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close: np.ndarray,
        close_IV: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def calibrate(
        self,
        *,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close: np.ndarray,
        close_IV: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def price(self, *, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError
