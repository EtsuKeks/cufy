from dataclasses import dataclass

import numpy as np
import torch

from src.config.config import settings


def to_torch_1d(x: np.ndarray) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D numpy array, got shape={x.shape}")
    return torch.from_numpy(x).to(device=settings.device, dtype=settings.dtype)


def inputs_1d(
    *, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    np_dt = np.float64 if settings.dtype == torch.float64 else np.float32

    return (
        to_torch_1d(S.astype(np_dt, copy=False)),
        to_torch_1d(K.astype(np_dt, copy=False)),
        to_torch_1d(T.astype(np_dt, copy=False)),
        torch.from_numpy(is_call.astype(np.bool_, copy=False)).to(device=settings.device, dtype=torch.bool),
        to_torch_1d(r.astype(np_dt, copy=False)),
    )


@dataclass(frozen=True, slots=True)
class Batch1D:
    S: np.ndarray
    K: np.ndarray
    T: np.ndarray
    is_call: np.ndarray
    close_IV: np.ndarray
    r: np.ndarray
    w: np.ndarray
    S_t: torch.Tensor
    K_t: torch.Tensor
    T_t: torch.Tensor
    is_call_t: torch.Tensor
    close_IV_t: torch.Tensor
    r_t: torch.Tensor
    w_t: torch.Tensor

    @staticmethod
    def from_numpy(
        *,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close_IV: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
    ) -> "Batch1D":
        np_dt = np.float64 if settings.dtype == torch.float64 else np.float32
        S = S.astype(np_dt, copy=False)
        K = K.astype(np_dt, copy=False)
        T = T.astype(np_dt, copy=False)
        is_call = is_call.astype(np.bool_, copy=False)
        close_IV = close_IV.astype(np_dt, copy=False)
        r = r.astype(np_dt, copy=False)
        w = w.astype(np_dt, copy=False)
        S_t, K_t, T_t, is_call_t, r_t = inputs_1d(S=S, K=K, T=T, is_call=is_call, r=r)
        w_t = to_torch_1d(w)
        close_IV_t = to_torch_1d(close_IV)
        return Batch1D(
            S=S,
            K=K,
            T=T,
            is_call=is_call,
            close_IV=close_IV,
            r=r,
            w=w,
            S_t=S_t,
            K_t=K_t,
            T_t=T_t,
            is_call_t=is_call_t,
            close_IV_t=close_IV_t,
            r_t=r_t,
            w_t=w_t,
        )
