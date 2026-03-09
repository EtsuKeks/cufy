import torch

from src.utils.implied_vol import bs_price
from src.utils.torch_utils import Batch1D
from src.abc.parameterized_model import ParameterizedModel, ModelParam


class BlackScholes(ParameterizedModel):
    sigma = ModelParam(position=0, min_value=0.001, max_value=10.0)

    def prices_for_param_matrix(self, *, data: Batch1D, param_matrix: torch.Tensor) -> torch.Tensor:
        return bs_price(data=data, sigma=param_matrix[:, 0:1])
