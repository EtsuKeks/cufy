import numpy as np
import torch
from torch.autograd import once_differentiable  # type: ignore[attr-defined]
from typing import cast

import cufy.config as config
from cufy.backends.torch.utils.torch_utils import Batch1D

_SQRT2 = float(np.sqrt(2.0))
_SQRT2PI = float(np.sqrt(2.0 * np.pi))


def check_settings_tensors(**tensors: torch.Tensor) -> None:
    dt = config.dtype
    d = config.device
    if "is_call" in tensors and tensors["is_call"].dtype != torch.bool:
        raise ValueError("is_call must be torch.bool")
    if "is_call" in tensors and tensors["is_call"].device != d:
        raise ValueError(f"is_call must use config.device={d}. Got: {tensors['is_call'].device}")

    bad = {k: (v.dtype, v.device) for k, v in tensors.items() if k != "is_call" and (v.dtype != dt or v.device != d)}
    if bad:
        raise ValueError(f"All float tensors must use config.dtype={dt} and config.device={d}. Got: {bad}")


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / _SQRT2PI


def bs_price_from_tensors(
    *, F: torch.Tensor, K: torch.Tensor, T: torch.Tensor, is_call: torch.Tensor, df: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    check_settings_tensors(F=F, K=K, T=T, is_call=is_call, df=df, sigma=sigma)
    eps_t = torch.as_tensor(config.eps, device=F.device, dtype=config.dtype)
    F_b, K_b, T_b, is_call_b, df_b = F[None, :], K[None, :], T[None, :], is_call[None, :], df[None, :]
    
    T_safe = torch.clamp(T_b, min=eps_t)
    sqrtT = torch.sqrt(T_safe)
    sig = torch.clamp(sigma, min=eps_t)
    sig_sqrtT = sig * sqrtT

    logFK = torch.log(torch.clamp(F_b, min=eps_t)) - torch.log(torch.clamp(K_b, min=eps_t))
    half_varT = 0.5 * sig * sig * T_safe
    d1 = (logFK + half_varT) / sig_sqrtT
    d2 = d1 - sig_sqrtT

    # Using torch.special.ndtr is faster and more numerically stable than manual erf
    Nd1 = torch.special.ndtr(d1)
    Nd2 = torch.special.ndtr(d2)
    Nd1_neg = torch.special.ndtr(-d1)
    Nd2_neg = torch.special.ndtr(-d2)
    
    call = df_b * (F_b * Nd1 - K_b * Nd2)
    put = df_b * (K_b * Nd2_neg - F_b * Nd1_neg)
    return torch.where(is_call_b, call, put)


def bs_price(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    return bs_price_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, is_call=data.is_call_t, df=data.df_t, sigma=sigma)


def bs_vega_from_tensors(
    *, F: torch.Tensor, K: torch.Tensor, T: torch.Tensor, df: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    check_settings_tensors(F=F, K=K, T=T, df=df, sigma=sigma)
    eps_t = torch.as_tensor(config.eps, device=F.device, dtype=config.dtype)
    F_b, K_b, T_b, df_b = F[None, :], K[None, :], T[None, :], df[None, :]
    
    T_safe = torch.clamp(T_b, min=eps_t)
    sqrtT = torch.sqrt(T_safe)
    sig = torch.clamp(sigma, min=eps_t)
    
    logFK = torch.log(torch.clamp(F_b, min=eps_t)) - torch.log(torch.clamp(K_b, min=eps_t))
    d1 = (logFK + 0.5 * sig * sig * T_safe) / (sig * sqrtT)
    
    # Vega in Black-76: df * F * phi(d1) * sqrt(T)
    pdf_d1 = torch.exp(-0.5 * d1 * d1) / _SQRT2PI
    return df_b * F_b * pdf_d1 * sqrtT


def bs_vega(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    return bs_vega_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, df=data.df_t, sigma=sigma)


def proxy_dsigma_dprice_from_tensors(
    *, F: torch.Tensor, K: torch.Tensor, T: torch.Tensor, df: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    check_settings_tensors(F=F, K=K, T=T, df=df, sigma=sigma)
    eps_t = torch.as_tensor(config.eps, device=sigma.device, dtype=sigma.dtype)
    vega = bs_vega_from_tensors(F=F, K=K, T=T, df=df, sigma=sigma).clamp_min(eps_t)
    return 1.0 / vega


def proxy_dsigma_dprice(*, data: Batch1D, sigma: torch.Tensor) -> torch.Tensor:
    return proxy_dsigma_dprice_from_tensors(F=data.F_t, K=data.K_t, T=data.T_t, df=data.df_t, sigma=sigma)

def implied_vol_newton(
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
    return cast(torch.Tensor, _ImpliedVolNewtonBS.apply(
        price, data.F_t, data.K_t, data.T_t, data.is_call_t, data.df_t, sigma_init, max_iter, max_sigma
    ))


def weighted_iv(*, data: Batch1D, pred_prices: torch.Tensor) -> torch.Tensor:
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
        if sigma_init is not None:
            check_settings_tensors(price=price, F=F, K=K, T=T, is_call=is_call, df=df, sigma_init=sigma_init)
        else:
            check_settings_tensors(price=price, F=F, K=K, T=T, is_call=is_call, df=df)

        with torch.no_grad():
            price_p = price.detach()
            F_p = F.detach()
            K_p = K.detach()
            T_p = T.detach()
            is_call_p = is_call.detach()
            df_p = df.detach()
            eps_t = torch.as_tensor(config.eps, device=config.device, dtype=config.dtype)
            max_sigma_t = torch.as_tensor(max_sigma, device=config.device, dtype=config.dtype)

            F_b = F_p[None, :]
            K_b = K_p[None, :]
            df_b = df_p[None, :]
            is_call_b = is_call_p[None, :]
            
            # Intrinsic value bounds
            lower_call = torch.clamp(df_b * (F_b - K_b), min=0.0)
            lower_put = torch.clamp(df_b * (K_b - F_b), min=0.0)
            lower = torch.where(is_call_b, lower_call, lower_put)
            upper = torch.where(is_call_b, df_b * F_b, df_b * K_b)
            
            span = upper - lower
            mid = 0.5 * (lower + upper)
            price_clamped = torch.where(span > 0.0, torch.clamp(price_p, min=lower, max=upper), mid.expand_as(price_p))

            if sigma_init is None:
                sigma = torch.full_like(price_clamped, 0.5)
            else:
                sigma = torch.broadcast_to(sigma_init.detach(), price_clamped.shape).clone()
            sigma = torch.clamp(sigma, min=eps_t, max=max_sigma_t)

            for _ in range(max_iter):
                model_price = bs_price_from_tensors(F=F_p, K=K_p, T=T_p, is_call=is_call_p, df=df_p, sigma=sigma)
                vega = bs_vega_from_tensors(F=F_p, K=K_p, T=T_p, df=df_p, sigma=sigma).clamp_min(eps_t)
                step = (model_price - price_clamped) / vega
                sigma = torch.clamp(sigma - step, min=eps_t, max=max_sigma_t)
                if bool(torch.all(torch.abs(step) <= eps_t).item()):
                    break

        ctx.save_for_backward(F_p, K_p, T_p, df_p, sigma.detach())
        return sigma

    @staticmethod
    @once_differentiable
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_sigma: torch.Tensor) -> tuple[torch.Tensor | None, ...]:  # type: ignore[override]
        if grad_sigma is None:  # type: ignore[comparison-overlap]
            return (None,) * 9

        if any(ctx.needs_input_grad[i] for i in range(1, 7)):  # type: ignore[attr-defined]
            raise RuntimeError(
                "implied_vol_newton: gradients are only supported w.r.t. `price`. "
                "Found a request for gradients w.r.t. other inputs"
            )

        F, K, T, df, sigma = ctx.saved_tensors  # type: ignore[attr-defined]
        dsigma_dprice = proxy_dsigma_dprice_from_tensors(F=F, K=K, T=T, df=df, sigma=sigma)
        grad_price = grad_sigma * dsigma_dprice

        return grad_price, None, None, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx: torch.autograd.function.FunctionCtx, price_t: torch.Tensor, F_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, is_call_t: torch.Tensor, df_t: torch.Tensor, sigma_init_t: torch.Tensor, max_iter_t: torch.Tensor, max_sigma_t: torch.Tensor) -> None:  # type: ignore[override]
        raise RuntimeError("implied_vol_newton does not support forward-mode autodiff")
