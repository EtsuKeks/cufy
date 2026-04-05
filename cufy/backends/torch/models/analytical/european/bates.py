import math
from collections.abc import Mapping

import torch

from cufy.backends.torch.math.integration.quadrature import IntegrationMethod, definite_integral
from cufy.backends.torch.models.analytical.european.heston import heston_characteristic_log_attari
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.parameterized_model import ModelParam

_PI = math.pi
_PI_INV = 1.0 / math.pi


class Bates(TorchParameterizedModel):
    mean_reversion = ModelParam(position=0, min_value=1e-6, max_value=200.0)
    theta = ModelParam(position=1, min_value=1e-8, max_value=100.0)
    volvol = ModelParam(position=2, min_value=1e-6, max_value=100.0)
    rho = ModelParam(position=3, min_value=-0.999, max_value=0.999)
    jump_intensity = ModelParam(position=4, min_value=0.0, max_value=200.0)
    jump_mean_log = ModelParam(position=5, min_value=-10.0, max_value=10.0)
    jump_vol = ModelParam(position=6, min_value=1e-4, max_value=10.0)

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
        self.num_points: int = num_points
        self.method: IntegrationMethod = method
        self.lower: float = lower
        self.upper: float = upper

    def prices_for_param_matrix(self, *, data: TorchPreparedBatch, param_matrix: torch.Tensor) -> torch.Tensor:
        if "sigma_atm" not in data.extras_t:
            raise ValueError("Field 'sigma_atm' is required in data.extras_t for Bates model")

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
        lam = param_matrix[:, 4:5, None]
        mu_J = param_matrix[:, 5:6, None]
        v_J = param_matrix[:, 6:7, None]

        t_t = data.T_t[None, :, None]
        V_0 = sigma_atm.square()[None, :, None]
        k_log = torch.log(K_t / F_t).unsqueeze(-1)

        rho_eta = rho_p * eta
        eta_sq = eta.square()
        eta_sq_inv = eta_sq.reciprocal()
        C_coeff = a * V_bar * eta_sq_inv

        lam_t = lam * t_t
        v_J_sq = v_J.square()
        v_J_sq_half = v_J_sq * 0.5
        v_J_sq_half_neg = -v_J_sq_half
        kappa_J_neg_ij = (torch.exp(mu_J + v_J_sq_half) - 1.0) * -1j

        def integrand(w: torch.Tensor) -> torch.Tensor:
            w_sq = w.square()
            iw = 1j * w
            w_complex = torch.complex(w_sq, w)
            h_log_cf = heston_characteristic_log_attari(
                w_complex, iw, t_t, a, rho_eta, eta_sq, eta_sq_inv, C_coeff, V_0
            )

            ej = torch.complex(w_sq * v_J_sq_half_neg, w * mu_J)
            inner = torch.addcmul(torch.exp(ej) - 1.0, w, kappa_J_neg_ij)
            cf = torch.exp(torch.addcmul(h_log_cf, lam_t, inner))

            w_k_log = w * k_log
            cos_w_k_log = torch.cos(w_k_log)
            sin_w_k_log = torch.sin(w_k_log)

            w_inv = w.reciprocal()
            weight = (1.0 + w_sq).reciprocal()

            C1 = torch.addcmul(cos_w_k_log, sin_w_k_log, w_inv, value=-1.0)
            C2 = torch.addcmul(sin_w_k_log, cos_w_k_log, w_inv)

            return torch.addcmul(cf.real * C1, cf.imag, C2) * weight

        integral = definite_integral(
            integrand, lower=self.lower, upper=self.upper, num_points=self.num_points, method=self.method
        )

        undiscounted_call_prices = F_t - K_t * (0.5 + integral * _PI_INV)
        undiscounted_put_prices = undiscounted_call_prices - F_t + K_t
        return torch.where(is_call_t, undiscounted_call_prices, undiscounted_put_prices) * df_t
