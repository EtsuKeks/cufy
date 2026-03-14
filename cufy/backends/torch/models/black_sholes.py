from abc import ABC

import torch

from cufy.backends.torch.utils.implied_vol import bs_price
from cufy.backends.torch.utils.torch_utils import Batch1D
from cufy.abc.parameterized_model import ModelParam
from cufy.backends.torch.models.base import TorchParameterizedModel


class BlackScholes(TorchParameterizedModel, ABC):
    sigma = ModelParam(position=0, min_value=0.001, max_value=10.0)

    def prices_for_param_matrix(self, *, data: Batch1D, param_matrix: torch.Tensor) -> torch.Tensor:
        return bs_price(data=data, sigma=param_matrix[:, 0:1])
