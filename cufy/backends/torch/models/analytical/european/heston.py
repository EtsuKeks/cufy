import math
from collections.abc import Mapping

import torch

from cufy.backends.torch.math.integration.quadrature import IntegrationMethod, definite_integral
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.parameterized_model import ModelParam

_PI = math.pi
_PI_INV = 1.0 / math.pi


def heston_characteristic_log_attari(
    w_complex: torch.Tensor,
    iw: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    rho_eta: torch.Tensor,
    eta_sq: torch.Tensor,
    eta_sq_inv: torch.Tensor,
    C_coeff: torch.Tensor,
    V_0: torch.Tensor,
) -> torch.Tensor:
    beta = a - rho_eta * iw

    h = torch.sqrt(torch.addcmul(beta.square(), eta_sq, w_complex))

    num_g = beta - h
    g = num_g / (beta + h)

    exp_h_t = torch.exp(h * (-t))
    one_minus_g_exp = 1.0 - g * exp_h_t
    m_neg = -2.0 * torch.log(one_minus_g_exp / (1.0 - g))
    C_V_bar = torch.addcmul(m_neg, num_g, t) * C_coeff

    num_g_eta = num_g * eta_sq_inv
    D = torch.addcmul(num_g_eta, num_g_eta, exp_h_t, value=-1.0) / one_minus_g_exp

    return torch.addcmul(C_V_bar, D, V_0)


class Heston(TorchParameterizedModel):
    mean_reversion = ModelParam(position=0, min_value=1e-6, max_value=200.0)
    theta = ModelParam(position=1, min_value=1e-8, max_value=100.0)
    volvol = ModelParam(position=2, min_value=1e-6, max_value=100.0)
    rho = ModelParam(position=3, min_value=-0.999, max_value=0.999)

    def __init__(
        self,
        *,
        num_points: int,
        method: IntegrationMethod,
        lower: float,
        upper: float,
        param_overrides: Mapping[str, ModelParam] | None = None,
    ) -> None:
        super().__init__(param_overrides)
        self.num_points = num_points
        self.method: IntegrationMethod = method
        self.lower = lower
        self.upper = upper

    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        if "sigma_atm" not in data.extras_t:
            raise ValueError("Field 'sigma_atm' is required in data.extras_t for Heston model")

        sigma_atm = data.extras_t["sigma_atm"]
        if sigma_atm.shape != data.K_t.shape:
            raise ValueError(f"'sigma_atm' shape ({sigma_atm.shape}) does not match expected shape ({data.K_t.shape})")

        K_t = data.K_t[None, :]
        F_t = data.F_t[None, :]
        is_call_t = data.is_call_t[None, :]
        df_t = data.df_t[None, :]

        a = param_matrix[:, 0:1, None]
        V_bar = param_matrix[:, 1:2, None]
        eta = param_matrix[:, 2:3, None]
        rho_p = param_matrix[:, 3:4, None]

        t_t = data.T_t[None, :, None]
        V_0 = sigma_atm.square()[None, :, None]
        k_log = torch.log(K_t / F_t).unsqueeze(-1)

        rho_eta = rho_p * eta
        eta_sq = eta.square()
        eta_sq_inv = eta_sq.reciprocal()
        C_coeff = a * V_bar * eta_sq_inv

        def integrand(w: torch.Tensor) -> torch.Tensor:
            w_sq = w.square()
            iw = 1j * w
            w_complex = torch.complex(w_sq, w)
            log_cf = heston_characteristic_log_attari(
                w_complex, iw, t_t, a, rho_eta, eta_sq, eta_sq_inv, C_coeff, V_0
            )
            cf = torch.exp(log_cf)

            w_k_log = w * k_log
            cos_w_k_log = torch.cos(w_k_log)
            sin_w_k_log = torch.sin(w_k_log)

            w_inv = w.reciprocal()
            weight = (1.0 + w_sq).reciprocal()
            C1 = torch.addcmul(cos_w_k_log, sin_w_k_log, w_inv, value=-1.0) * weight
            C2 = torch.addcmul(sin_w_k_log, cos_w_k_log, w_inv) * weight

            return torch.addcmul(cf.real * C1, cf.imag, C2)

        integral = definite_integral(
            integrand, lower=self.lower, upper=self.upper, num_points=self.num_points, method=self.method
        )

        undiscounted_call_prices = F_t - K_t * (0.5 + integral * _PI_INV)
        undiscounted_put_prices = undiscounted_call_prices - F_t + K_t
        return torch.where(is_call_t, undiscounted_call_prices, undiscounted_put_prices) * df_t
