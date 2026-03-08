import torch

from src.utils.implied_vol import bs_price, check_settings_tensors
from src.abc.parameterized_model import ParameterizedModel, ModelParam


class BlackScholes(ParameterizedModel):
    sigma = ModelParam(position=0, min_value=0.001, max_value=10.0)

    def prices_for_param_matrix(
        self,
        *,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        is_call: torch.Tensor,
        param_matrix: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        check_settings_tensors(S=S, K=K, T=T, is_call=is_call, r=r, param_matrix=param_matrix)
        return bs_price(S=S, K=K, T=T, is_call=is_call, r=r, sigma=param_matrix[:, 0:1])
