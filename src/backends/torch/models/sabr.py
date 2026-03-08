import torch

from src.config.config import settings
from src.utils.implied_vol import bs_price, check_settings_tensors
from src.abc.parameterized_model import ParameterizedModel, ModelParam


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
    eps = torch.as_tensor(settings.ppl.epsilon, device=F.device, dtype=settings.dtype)
    F_b = F[None, :]
    K_b = K[None, :]
    T_b = T[None, :]

    one_minus_b = 1.0 - beta
    one_minus_b2 = one_minus_b * one_minus_b
    FK = F_b * K_b
    FKpow_half = torch.clamp(FK.pow(0.5 * one_minus_b), min=eps)
    logFK = torch.log(torch.clamp(F_b / K_b, min=eps))
    logFK2 = logFK * logFK

    z = nu / alpha * FKpow_half * logFK
    sqrt_term = torch.sqrt(torch.clamp(1.0 - 2.0 * rho * z + z * z, min=eps))
    numer = sqrt_term + z - rho
    xz = torch.log(torch.clamp(numer / (1.0 - rho), min=eps))
    xz_safe = torch.where(xz >= 0, xz.clamp(min=eps), xz.clamp(max=-eps))

    denom_A = 1.0 + (one_minus_b2 / 24.0 * logFK2) + ((one_minus_b2 * one_minus_b2) / 1920.0 * logFK2 * logFK2)
    denom_A_safe = torch.where(denom_A >= 0, denom_A.clamp(min=eps), denom_A.clamp(max=-eps))
    A = alpha / FKpow_half / denom_A_safe
    corr_fk = (
        one_minus_b2 / 24.0 * (alpha * alpha) / torch.clamp(FKpow_half * FKpow_half, min=eps)
        + (rho * beta * nu * alpha) / 4.0 / FKpow_half
        + (2.0 - 3.0 * (rho * rho)) / 24.0 * (nu * nu)
    )

    last_mult = (1.0 + corr_fk * T_b)
    sigma = A * torch.where(torch.abs(z) < eps, torch.ones_like(z), z / xz_safe) * last_mult
    sigma_atm = alpha / FKpow_half * last_mult
    return torch.where(torch.abs(K_b - F_b) < eps, sigma_atm, sigma).clamp_min(eps)


class SABR(ParameterizedModel):
    alpha = ModelParam(position=0, min_value=1e-4, max_value=5.0)
    beta = ModelParam(position=1, min_value=0.0, max_value=1.0)
    rho = ModelParam(position=2, min_value=-0.999, max_value=0.999)
    nu = ModelParam(position=3, min_value=1e-4, max_value=10.0)

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
        F = S * torch.exp(r * T)
        alpha = param_matrix[:, 0:1]
        beta = param_matrix[:, 1:2]
        rho = param_matrix[:, 2:3]
        nu = param_matrix[:, 3:4]

        sigma_B = _sabr_implied_vol_hagan(F=F, K=K, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
        return bs_price(S=S, K=K, T=T, is_call=is_call, r=r, sigma=sigma_B)
