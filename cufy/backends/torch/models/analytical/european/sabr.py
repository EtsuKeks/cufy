import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.implied_vol import bs_price
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.parameterized_model import ModelParam


def _sabr_implied_vol_hagan(
    *,
    F: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    eps = config.eps
    F_b = F[None, :]
    K_b = K[None, :]
    T_b = T[None, :]

    one_minus_b = 1.0 - beta
    one_minus_b2 = one_minus_b * one_minus_b
    FK = torch.clamp(F_b * K_b, min=eps)
    FKpow = torch.clamp(FK.pow(one_minus_b), min=eps)
    FKpow_half = torch.sqrt(FKpow)
    logFK = torch.log(torch.clamp(F_b / K_b, min=eps))
    logFK2 = logFK * logFK
    a_fk = torch.clamp(alpha / FKpow_half, min=eps)

    z = nu / a_fk * logFK
    sqrt_term = torch.sqrt(1.0 - 2.0 * rho * z + z * z)
    numer = sqrt_term + z - rho
    xz = torch.log(torch.clamp(numer / (1.0 - rho), min=eps))
    xz = torch.where(xz >= 0, xz.clamp(min=eps), xz.clamp(max=-eps))

    const = one_minus_b2 * logFK2
    A = a_fk / (1.0 + (const / 24.0) + (const * const / 1920.0))
    corr_fk = (
        one_minus_b2 / 24.0 * a_fk * a_fk + rho * beta * nu * a_fk / 4.0 + (2.0 - 3.0 * rho * rho) / 24.0 * (nu * nu)
    )

    last_mult = 1.0 + corr_fk * T_b
    sigma = A * torch.where(torch.abs(z) < eps, 1.0, z / xz) * last_mult
    sigma_atm = a_fk * last_mult
    return torch.where(torch.abs(logFK) < eps, sigma_atm, sigma).clamp_min(eps)


class SABR(TorchParameterizedModel):
    alpha = ModelParam(position=0, min_value=1e-4, max_value=1e6)
    beta = ModelParam(position=1, min_value=0.0, max_value=1.0)
    rho = ModelParam(position=2, min_value=-0.999, max_value=0.999)
    nu = ModelParam(position=3, min_value=1e-4, max_value=100.0)

    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        F_true = data.F_t * data.F_scale_t
        K_true = data.K_t * data.F_scale_t

        alpha = param_matrix[:, 0:1]
        beta = param_matrix[:, 1:2]
        rho = param_matrix[:, 2:3]
        nu = param_matrix[:, 3:4]

        sigma_B = _sabr_implied_vol_hagan(F=F_true, K=K_true, T=data.T_t, alpha=alpha, beta=beta, rho=rho, nu=nu)
        return bs_price(data=data, sigma=sigma_B)
