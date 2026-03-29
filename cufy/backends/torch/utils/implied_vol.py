import logging
from typing import cast

import numpy as np
import torch
from torch.autograd.function import once_differentiable

import cufy.backends.torch.config as config
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch

logger = logging.getLogger(__name__)


def bs_price_from_tensors(
    *, F: torch.Tensor, K: torch.Tensor, T: torch.Tensor, is_call: torch.Tensor, df: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    eps_t = config.eps
    F_b, K_b, T_b, is_call_b, df_b = F[None, :], K[None, :], T[None, :], is_call[None, :], df[None, :]

    T_safe = torch.clamp(T_b, min=eps_t)
    sqrtT = torch.sqrt(T_safe)
    sig = torch.clamp(sigma, min=eps_t)
    sig_sqrtT = torch.clamp(sig * sqrtT, min=eps_t)

    logFK = torch.log(torch.clamp(F_b / K_b, min=eps_t))
    half_varT = 0.5 * sig * sig * T_safe
    d1 = (logFK + half_varT) / sig_sqrtT
    d2 = d1 - sig_sqrtT

    Nd1 = torch.special.ndtr(d1)
    Nd2 = torch.special.ndtr(d2)
    call = F_b * Nd1 - K_b * Nd2
    put = K_b * (1.0 - Nd2) - F_b * (1.0 - Nd1)

    res = torch.where(is_call_b, call, put)
    return df_b * res


def bs_price(*, data: TorchPreparedBatch, sigma: torch.Tensor) -> torch.Tensor:
    return bs_price_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, is_call=data.is_call_t, df=data.df_t, sigma=sigma)


def _bs_vega_from_tensors(
    *, F: torch.Tensor, K: torch.Tensor, T: torch.Tensor, df: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    eps_t = config.eps
    F_b, K_b, T_b, df_b = F[None, :], K[None, :], T[None, :], df[None, :]

    T_safe = torch.clamp(T_b, min=eps_t)
    sqrtT = torch.sqrt(T_safe)
    sig = torch.clamp(sigma, min=eps_t)
    sig_sqrtT = torch.clamp(sig * sqrtT, min=eps_t)

    logFK = torch.log(torch.clamp(F_b / K_b, min=eps_t))
    half_varT = 0.5 * sig * sig * T_safe
    d1 = (logFK + half_varT) / sig_sqrtT

    pdf_d1 = torch.exp(-0.5 * d1 * d1) / float(np.sqrt(2.0 * np.pi))
    return df_b * F_b * pdf_d1 * sqrtT


def bs_vega(*, data: TorchPreparedBatch, sigma: torch.Tensor) -> torch.Tensor:
    return _bs_vega_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, df=data.df_t, sigma=sigma)


def proxy_dsigma_dprice(*, data: TorchPreparedBatch, sigma: torch.Tensor) -> torch.Tensor:
    eps_t = config.eps
    vega = _bs_vega_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, df=data.df_t, sigma=sigma).clamp_min(eps_t)
    return 1.0 / vega


def implied_vol_newton(
    *,
    price: torch.Tensor,
    data: TorchPreparedBatch,
    sigma_init: torch.Tensor | None = None,
    max_iter: int = 15,
    max_sigma: float = 100.0,
) -> torch.Tensor:
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if max_sigma <= 0.0:
        raise ValueError(f"max_sigma must be positive, got {max_sigma}")
    return cast(
        torch.Tensor,
        _ImpliedVolNewtonBS.apply(
            price, data.F_t, data.K_t, data.T_t, data.is_call_t, data.df_t, sigma_init, max_iter, max_sigma
        ),
    )


def weighted_iv(*, data: TorchPreparedBatch, pred_prices: torch.Tensor) -> torch.Tensor:
    pred_iv = implied_vol_newton(price=pred_prices, data=data, sigma_init=data.close_IV_t)
    d = pred_iv - data.close_IV_t[None, :]
    s = torch.sum(data.w_t[None, :] * (d * d), 1)
    return torch.sqrt(torch.clamp(s, min=0.0))


class _ImpliedVolNewtonBS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        price: torch.Tensor,
        F: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        is_call: torch.Tensor,
        df: torch.Tensor,
        sigma_init: torch.Tensor | None,
        max_iter: int,
        max_sigma: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            price_p = price.detach()
            F_b = F.detach()[None, :]
            K_b = K.detach()[None, :]
            T_b = T.detach()[None, :]
            is_call_b = is_call.detach()[None, :]
            df_b = df.detach()[None, :]
            eps_t = config.eps
            max_sigma_t = torch.as_tensor(max_sigma, device=config.device, dtype=config.dtype)

            omega = torch.where(is_call_b, 1.0, -1.0)

            lower = torch.clamp((F_b - K_b) * omega, min=0.0) * df_b
            upper = torch.where(is_call_b, F_b, K_b) * df_b

            span = upper - lower
            mid = 0.5 * (lower + upper)
            price_clamped = torch.where(span > 0.0, torch.clamp(price_p, min=lower, max=upper), mid.expand_as(price_p))

            if sigma_init is None:
                sigma = torch.full_like(price_clamped, 0.5)
            else:
                sigma = torch.broadcast_to(sigma_init.detach(), price_clamped.shape).clone()
            sigma = torch.clamp(sigma, min=eps_t, max=max_sigma_t)

            T_safe = torch.clamp(T_b, min=eps_t)
            sqrtT = torch.sqrt(T_safe)
            logFK = torch.log(torch.clamp(F_b / K_b, min=eps_t))
            vega_coeff = df_b * F_b * sqrtT / float(np.sqrt(2.0 * np.pi))

            omega_F = omega * F_b
            omega_K = omega * K_b

            converged_mask = torch.zeros_like(sigma, dtype=torch.bool)

            for _ in range(max_iter):
                sig_sqrtT = torch.clamp(sigma * sqrtT, min=eps_t)

                d1 = (logFK + 0.5 * sigma * sigma * T_safe) / sig_sqrtT
                d2 = d1 - sig_sqrtT

                Nd1_omega = torch.special.ndtr(omega * d1)
                Nd2_omega = torch.special.ndtr(omega * d2)

                model_price = (omega_F * Nd1_omega - omega_K * Nd2_omega) * df_b

                vega = torch.clamp(vega_coeff * torch.exp(-0.5 * d1 * d1), min=eps_t)

                step = (model_price - price_clamped) / vega

                converged_mask = torch.abs(step) <= eps_t
                step = torch.where(converged_mask, 0.0, step)

                sigma = torch.clamp(sigma - step, min=eps_t, max=max_sigma_t)

            not_converged = ~converged_mask
            if not_converged.any():
                logger.warning(
                    f"Newton method did not converge for {not_converged.sum().item()}"
                    f" out of {not_converged.numel()} options"
                )

        ctx.save_for_backward(logFK, sqrtT, T_safe, vega_coeff, sigma.detach())
        return sigma

    @staticmethod
    @once_differentiable
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_sigma: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        if grad_sigma is None:  # type: ignore[comparison-overlap]
            return (None,) * 9

        if any(ctx.needs_input_grad[i] for i in range(1, 8)):  # type: ignore[attr-defined]
            raise RuntimeError(
                "implied_vol_newton: gradients are only supported w.r.t. `price`. "
                "Found a request for gradients w.r.t. other inputs"
            )

        logFK, sqrtT, T_safe, vega_coeff, sigma = ctx.saved_tensors  # type: ignore[attr-defined]
        eps_t = config.eps

        sig = torch.clamp(sigma, min=eps_t)
        sig_sqrtT = torch.clamp(sig * sqrtT, min=eps_t)

        d1 = (logFK + 0.5 * sig * sig * T_safe) / sig_sqrtT
        vega = torch.clamp(vega_coeff * torch.exp(-0.5 * d1 * d1), min=eps_t)

        grad_price = grad_sigma / vega

        return grad_price, None, None, None, None, None, None, None, None

    @staticmethod
    def jvp(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        price_t: torch.Tensor,
        F_t: torch.Tensor,
        K_t: torch.Tensor,
        T_t: torch.Tensor,
        is_call_t: torch.Tensor,
        df_t: torch.Tensor,
        sigma_init_t: torch.Tensor | None,
        max_iter_t: int | None,
        max_sigma_t: float | None,
    ) -> torch.Tensor:
        raise RuntimeError("implied_vol_newton does not support forward-mode autodiff")
