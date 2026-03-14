from abc import ABC, abstractmethod
import torch

from cufy.abc.data import OptionBatch
from cufy.abc.parameterized_model import ParameterizedModel
from cufy.backends.torch.utils.torch_utils import Batch1D


class TorchParameterizedModel(ParameterizedModel, ABC):
    @abstractmethod
    def prices_for_param_matrix(self, *, data: Batch1D, param_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _batch_to_1d(batch: OptionBatch) -> Batch1D:
        return Batch1D.from_option_batch(batch)
