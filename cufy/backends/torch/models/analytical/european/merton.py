import math

import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.implied_vol import bs_price_from_tensors
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.parameterized_model import ModelParam


class Merton(TorchParameterizedModel):
    sigma = ModelParam(position=0, min_value=0.001, max_value=100.0)
    lam = ModelParam(position=1, min_value=0.0, max_value=100.0)
    mu_j = ModelParam(position=2, min_value=-5.0, max_value=5.0)
    sigma_j = ModelParam(position=3, min_value=0.001, max_value=5.0)

    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        sigma = param_matrix[:, 0:1]
        lam = param_matrix[:, 1:2]
        mu_j = param_matrix[:, 2:3]
        sigma_j = param_matrix[:, 3:4]

        T = data.T_t
        F = data.F_t
        K = data.K_t
        df = data.df_t
        is_call = data.is_call_t

        lam_T_raw = lam * T[None, :]
        max_lam_T = float(torch.max(lam_T_raw).item())
        max_jumps = int(max_lam_T + 6.0 * math.sqrt(max_lam_T) + 10)

        var_j = sigma_j.square()
        log_mean_jump = mu_j + 0.5 * var_j
        kappa = torch.exp(log_mean_jump) - 1.0

        i = torch.arange(max_jumps, device=config.device, dtype=config.dtype).reshape(1, -1, 1)
        lam_T = torch.clamp(lam_T_raw, min=config.eps).unsqueeze(1)
        log_i_fact = torch.lgamma(i + 1.0)
        exponent = torch.addcmul(-log_i_fact, i, torch.log(lam_T))
        W = torch.exp(exponent - lam_T)

        F_drifted = F[None, :] * torch.exp((-lam * kappa) * T[None, :])
        exp_jump = torch.exp(i * log_mean_jump.unsqueeze(1))
        F_i = F_drifted.unsqueeze(1) * exp_jump

        inv_T = torch.clamp(T[None, :], min=config.eps).reciprocal()
        var_j_over_T = var_j * inv_T
        sigma_sq = sigma.square().unsqueeze(1)
        sigma_i = torch.sqrt(torch.addcmul(sigma_sq, i, var_j_over_T.unsqueeze(1)))

        return torch.sum(W * bs_price_from_tensors(F=F_i, K=K, T=T, is_call=is_call, df=df, sigma=sigma_i)[0], dim=1)
