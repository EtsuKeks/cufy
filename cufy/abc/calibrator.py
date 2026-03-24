from abc import ABC, abstractmethod

from cufy.abc.data import PreparedBatch
from cufy.abc.model import Model


class Calibrator[M: Model](ABC):
    def __init__(self, model: M):
        self.model = model

    @abstractmethod
    def calibrate(self, batch: PreparedBatch) -> None:
        raise NotImplementedError
