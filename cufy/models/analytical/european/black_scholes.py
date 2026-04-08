import torch

from cufy.core.parameterized_model import ModelParam
from cufy.models.base import TorchParameterizedModel
from cufy.utils.implied_vol import bs_price
from cufy.utils.torch_utils import TorchPreparedBatch


class BlackScholes(TorchParameterizedModel):
    sigma = ModelParam(position=0, min_value=0.001, max_value=100.0)

    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        return bs_price(data=data, sigma=param_matrix[:, 0:1])
