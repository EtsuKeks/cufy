import numpy as np
import torch
from torch.autograd.function import once_differentiable

from src.config.config import settings
from src.utils.torch_utils import Batch1D

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def check_settings_tensors(**tensors: torch.Tensor) -> None:
    dt = settings.dtype
    d = settings.device
    if "is_call" in tensors and tensors["is_call"].dtype != torch.bool:
        raise ValueError("is_call must be torch.bool")
    if "is_call" in tensors and tensors["is_call"].device != d:
        raise ValueError(f"is_call must use settings.device={d}. Got: {tensors['is_call'].device}")

    bad = {k: (v.dtype, v.device) for k, v in tensors.items() if k != "is_call" and (v.dtype != dt or v.device != d)}
    if bad:
        raise ValueError(f"All float tensors must use settings.dtype={dt} and settings.device={d}. Got: {bad}")


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT2PI


def _bs_price_raw(
    *, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, is_call: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    check_settings_tensors(S=S, K=K, T=T, is_call=is_call, r=r, sigma=sigma)
    eps_t = torch.as_tensor(settings.ppl.epsilon, device=S.device, dtype=settings.dtype)
    S_b, K_b, T_b, is_call_b, r_b = S[None, :], K[None, :], T[None, :], is_call[None, :], r[None, :]
    sqrtT = torch.sqrt(torch.clamp(T_b, min=eps_t))
    sig = torch.clamp(sigma, min=eps_t)
    sig_sqrtT = sig * sqrtT

    logSK = torch.log(torch.clamp(S_b, min=eps_t)) - torch.log(torch.clamp(K_b, min=eps_t))
    base = logSK + r_b * T_b
    half_varT = 0.5 * sig * sig * T_b
    d1 = (base + half_varT) / sig_sqrtT
    d2 = (base - half_varT) / sig_sqrtT

    disc = torch.exp(-r_b * T_b)
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    call = S_b * Nd1 - K_b * disc * Nd2
    put = K_b * disc * (1.0 - Nd2) - S_b * (1.0 - Nd1)
    return torch.where(is_call_b, call, put)


def bs_price(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    return _bs_price_raw(S=data.S_t, K=data.K_t, T=data.T_t, is_call=data.is_call_t, r=data.r_t, sigma=sigma)


def _bs_vega_raw(
    *, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    eps_t = torch.as_tensor(settings.ppl.epsilon, device=S.device, dtype=settings.dtype)
    S_b, K_b, T_b, r_b = S[None, :], K[None, :], T[None, :], r[None, :]
    sqrtT = torch.sqrt(torch.clamp(T_b, min=eps_t))
    sig = torch.clamp(sigma, min=eps_t)
    logSK = torch.log(torch.clamp(S_b, min=eps_t)) - torch.log(torch.clamp(K_b, min=eps_t))
    d2 = (logSK + r_b * T_b - 0.5 * sig * sig * T_b) / (sig * sqrtT)
    disc = torch.exp(-r_b * T_b)
    return K_b * disc * norm_pdf(d2) * sqrtT


def bs_vega(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    return _bs_vega_raw(S=data.S_t, K=data.K_t, T=data.T_t, r=data.r_t, sigma=sigma)


def _bs_implied_vol_proxy_dsigma_dprice_raw(
    *, S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    eps_t = torch.as_tensor(settings.ppl.epsilon, device=sigma.device, dtype=sigma.dtype)
    vega = _bs_vega_raw(S=S, K=K, T=T, r=r, sigma=sigma).clamp_min(eps_t)
    return 1.0 / vega


def bs_implied_vol_proxy_dsigma_dprice(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    eps_t = torch.as_tensor(settings.ppl.epsilon, device=sigma.device, dtype=sigma.dtype)
    vega = bs_vega(data=data, sigma=sigma).clamp_min(eps_t)
    return 1.0 / vega


def implied_vol_newton_bs(
    *,
    price: torch.Tensor,
    data: Batch1D,
    sigma_init: torch.Tensor | None = None,
    max_iter: int = 25,
    max_sigma: float = 10.0,
) -> torch.Tensor:
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if max_sigma <= 0.0:
        raise ValueError(f"max_sigma must be positive, got {max_sigma}")
    return _ImpliedVolNewtonBS.apply(
        price, data.S_t, data.K_t, data.T_t, data.is_call_t, data.r_t, sigma_init, max_iter, max_sigma
    )


def weighted_iv_l2_from_prices(*, data: Batch1D, pred_prices: torch.Tensor) -> torch.Tensor:
    pred_iv = implied_vol_newton_bs(price=pred_prices, data=data, sigma_init=data.close_IV_t)
    d = pred_iv - data.close_IV_t[None, :]
    s = torch.sum(data.w_t[None, :] * (d * d), 1)
    return torch.sqrt(torch.clamp(s, min=0.0))


class _ImpliedVolNewtonBS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        price: torch.Tensor,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        is_call: torch.Tensor,
        r: torch.Tensor,
        sigma_init: torch.Tensor | None,
        max_iter: int,
        max_sigma: float,
    ) -> torch.Tensor:
        if sigma_init is not None:
            check_settings_tensors(price=price, S=S, K=K, T=T, is_call=is_call, r=r, sigma_init=sigma_init)
        else:
            check_settings_tensors(price=price, S=S, K=K, T=T, is_call=is_call, r=r)

        with torch.no_grad():
            price_p = price.detach()
            S_p = S.detach()
            K_p = K.detach()
            T_p = T.detach()
            is_call_p = is_call.detach()
            r_p = r.detach()
            eps_t = torch.as_tensor(settings.ppl.epsilon, device=settings.device, dtype=settings.dtype)
            max_sigma_t = torch.as_tensor(max_sigma, device=settings.device, dtype=settings.dtype)

            S_b = S_p[None, :]
            K_b = K_p[None, :]
            T_b = T_p[None, :]
            r_b = r_p[None, :]
            is_call_b = is_call_p[None, :]
            disc = torch.exp(-r_b * T_b)
            lower_call = torch.clamp(S_b - K_b * disc, min=0.0)
            lower_put = torch.clamp(K_b * disc - S_b, min=0.0)
            lower = torch.where(is_call_b, lower_call, lower_put)
            upper = torch.where(is_call_b, S_b, K_b * disc)
            span = upper - lower
            mid = 0.5 * (lower + upper)
            price_clamped = torch.where(span > 0.0, torch.clamp(price_p, min=lower, max=upper), mid.expand_as(price_p))

            if sigma_init is None:
                sigma = torch.full_like(price_clamped, 0.5)
            else:
                sigma = torch.broadcast_to(sigma_init.detach(), price_clamped.shape).clone()
            sigma = torch.clamp(sigma, min=eps_t, max=max_sigma_t)

            for _ in range(max_iter):
                model_price = _bs_price_raw(S=S_p, K=K_p, T=T_p, is_call=is_call_p, r=r_p, sigma=sigma)
                vega = _bs_vega_raw(S=S_p, K=K_p, T=T_p, r=r_p, sigma=sigma).clamp_min(eps_t)
                step = (model_price - price_clamped) / vega
                sigma = torch.clamp(sigma - step, min=eps_t, max=max_sigma_t)
                if bool(torch.all(torch.abs(step) <= eps_t).item()):
                    break

        ctx.save_for_backward(S_p, K_p, T_p, r_p, sigma.detach())
        return sigma

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_sigma):
        if grad_sigma is None:
            return (None,) * 9

        if any(ctx.needs_input_grad[i] for i in range(1, 7)):
            raise RuntimeError(
                "implied_vol_newton_bs: gradients are only supported w.r.t. `price`. "
                "Found a request for gradients w.r.t. other inputs."
            )

        S, K, T, r, sigma = ctx.saved_tensors
        dsigma_dprice = _bs_implied_vol_proxy_dsigma_dprice_raw(S=S, K=K, T=T, r=r, sigma=sigma)
        grad_price = grad_sigma * dsigma_dprice

        return grad_price, None, None, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, price_t, S_t, K_t, T_t, is_call_t, r_t, sigma_init_t, max_iter_t, max_sigma_t):
        raise RuntimeError("implied_vol_newton_bs does not support forward-mode autodiff.")
