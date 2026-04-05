from collections.abc import Callable
from typing import Literal

import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.math.integration import gauss_constants

_SIMPSON_WEIGHTS_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}
_SIMPSON_ARANGE_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}
_GAUSS_LEGENDRE_CACHE: dict[tuple[int, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

IntegrationMethod = Literal["simpson", "gauss_legendre"]


def _composite_simpson(
    f: Callable[[torch.Tensor], torch.Tensor], lower: torch.Tensor, upper: torch.Tensor, num_points: int
) -> torch.Tensor:
    n = num_points
    dx = (upper - lower) / (n - 1)

    cache_key = (n, config.dtype, config.device)

    if cache_key not in _SIMPSON_ARANGE_CACHE:
        _SIMPSON_ARANGE_CACHE[cache_key] = torch.arange(n, device=config.device, dtype=config.dtype)
    arange_n = _SIMPSON_ARANGE_CACHE[cache_key]

    if dx.ndim > 0:
        grid = torch.addcmul(lower.unsqueeze(-1), dx.unsqueeze(-1), arange_n)
    else:
        grid = torch.addcmul(lower, dx, arange_n)

    if cache_key not in _SIMPSON_WEIGHTS_CACHE:
        w_first = torch.tensor([1.0 / 3.0], dtype=config.dtype, device=config.device)
        w_mid = torch.tensor([4.0 / 3.0, 2.0 / 3.0], dtype=config.dtype, device=config.device).repeat((n - 3) // 2)
        w_last = torch.tensor([4.0 / 3.0, 1.0 / 3.0], dtype=config.dtype, device=config.device)
        _SIMPSON_WEIGHTS_CACHE[cache_key] = torch.cat([w_first, w_mid, w_last], dim=0)

    weights = _SIMPSON_WEIGHTS_CACHE[cache_key]

    vals = f(grid)
    return torch.sum(vals * weights, dim=-1) * dx


def _gauss_legendre(
    f: Callable[[torch.Tensor], torch.Tensor], lower: torch.Tensor, upper: torch.Tensor, num_points: int
) -> torch.Tensor:
    cache_key = (num_points, config.dtype, config.device)
    if cache_key not in _GAUSS_LEGENDRE_CACHE:
        roots_list = gauss_constants.legendre_roots[num_points]
        coefficients = gauss_constants.legendre_weights[num_points]
        _GAUSS_LEGENDRE_CACHE[cache_key] = (
            torch.tensor(roots_list, dtype=config.dtype, device=config.device),
            torch.tensor(coefficients, dtype=config.dtype, device=config.device),
        )

    roots, weights = _GAUSS_LEGENDRE_CACHE[cache_key]
    half_diff = (upper - lower) * 0.5
    half_sum = (upper + lower) * 0.5
    if half_diff.ndim > 0:
        grid = torch.addcmul(half_sum.unsqueeze(-1), half_diff.unsqueeze(-1), roots)
    else:
        grid = torch.addcmul(half_sum, half_diff, roots)

    vals = f(grid)
    return torch.sum(vals * weights, dim=-1) * half_diff


def definite_integral(
    func: Callable[[torch.Tensor], torch.Tensor],
    *,
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    method: IntegrationMethod,
    num_points: int,
) -> torch.Tensor:
    if isinstance(lower, torch.Tensor):
        if lower.dtype != config.dtype:
            raise ValueError(f"lower must have dtype {config.dtype}, got {lower.dtype}")
        if lower.device != config.device:
            raise ValueError(f"lower must be on device {config.device}, got {lower.device}")
    if isinstance(upper, torch.Tensor):
        if upper.dtype != config.dtype:
            raise ValueError(f"upper must have dtype {config.dtype}, got {upper.dtype}")
        if upper.device != config.device:
            raise ValueError(f"upper must be on device {config.device}, got {upper.device}")

    lower_t = torch.as_tensor(lower, dtype=config.dtype, device=config.device)
    upper_t = torch.as_tensor(upper, dtype=config.dtype, device=config.device)

    if method == "simpson":
        if num_points < 3 or num_points % 2 == 0:
            raise ValueError(f"num_points must be odd and >= 3, got {num_points}")
        return _composite_simpson(func, lower_t, upper_t, num_points)
    if method == "gauss_legendre":
        if num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {num_points}")
        return _gauss_legendre(func, lower_t, upper_t, num_points)
