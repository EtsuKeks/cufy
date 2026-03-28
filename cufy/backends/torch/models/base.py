from abc import abstractmethod

import numpy as np
import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.data import PreparedBatch
from cufy.core.parameterized_model import ParameterizedModel


class TorchParameterizedModel(ParameterizedModel):
    @abstractmethod
    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def price(self, batch: PreparedBatch) -> np.ndarray:
        if not self.is_initialized():
            raise RuntimeError("Model is not initialized, calibrate it first")
        data = TorchPreparedBatch.from_batch(batch)
        param_matrix = torch.tensor([[p.value for p in self.params]], dtype=config.dtype, device=config.device)
        with torch.no_grad():
            prices_norm = self.prices_for_param_matrix(data=data, param_matrix=param_matrix)
        return (prices_norm * data.F_scale_t).squeeze(0).cpu().numpy()
