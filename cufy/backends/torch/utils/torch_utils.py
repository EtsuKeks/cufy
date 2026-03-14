from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import cufy.config as config
from cufy.abc.data import OptionBatch


def to_torch_1d(x: np.ndarray) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D numpy array, got shape={x.shape}")
    return torch.from_numpy(x).to(device=config.device, dtype=config.dtype)


def inputs_1d(
    *, F: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, df: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    np_dt = np.float64 if config.dtype == torch.float64 else np.float32

    return (
        to_torch_1d(F.astype(np_dt, copy=False)),
        to_torch_1d(K.astype(np_dt, copy=False)),
        to_torch_1d(T.astype(np_dt, copy=False)),
        torch.from_numpy(is_call.astype(np.bool_, copy=False)).to(device=config.device, dtype=torch.bool),
        to_torch_1d(df.astype(np_dt, copy=False)),
    )


@dataclass(frozen=True, slots=True)
class Batch1D:
    F: np.ndarray
    K: np.ndarray
    T: np.ndarray
    is_call: np.ndarray
    close_IV: np.ndarray
    df: np.ndarray
    w: np.ndarray
    F_t: torch.Tensor
    K_t: torch.Tensor
    T_t: torch.Tensor
    is_call_t: torch.Tensor
    close_IV_t: torch.Tensor
    df_t: torch.Tensor
    w_t: torch.Tensor

    def slice(self, n: int) -> "Batch1D":
        return Batch1D(
            F=self.F[:n],
            K=self.K[:n],
            T=self.T[:n],
            is_call=self.is_call[:n],
            close_IV=self.close_IV[:n],
            df=self.df[:n],
            w=self.w[:n],
            F_t=self.F_t[:n],
            K_t=self.K_t[:n],
            T_t=self.T_t[:n],
            is_call_t=self.is_call_t[:n],
            close_IV_t=self.close_IV_t[:n],
            df_t=self.df_t[:n],
            w_t=self.w_t[:n],
        )

    @staticmethod
    def from_option_batch(batch: OptionBatch) -> "Batch1D":
        return Batch1D.from_numpy(
            F=batch.F,
            K=batch.K,
            T=batch.T,
            is_call=batch.is_call,
            close_IV=batch.close_IV,
            df=batch.df,
            w=batch.w,
        )

    @staticmethod
    def from_numpy(
        *,
        F: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close_IV: np.ndarray,
        df: np.ndarray,
        w: np.ndarray,
    ) -> "Batch1D":
        np_dt = np.float64 if config.dtype == torch.float64 else np.float32
        f = F.astype(np_dt, copy=False)
        k = K.astype(np_dt, copy=False)
        t = T.astype(np_dt, copy=False)
        is_call_a = is_call.astype(np.bool_, copy=False)
        close_IV_a = close_IV.astype(np_dt, copy=False)
        df_a = df.astype(np_dt, copy=False)
        w_a = w.astype(np_dt, copy=False)
        F_t, K_t, T_t, is_call_t, df_t = inputs_1d(F=f, K=k, T=t, is_call=is_call_a, df=df_a)
        w_t = to_torch_1d(w_a)
        close_IV_t = to_torch_1d(close_IV_a)
        return Batch1D(
            F=f,
            K=k,
            T=t,
            is_call=is_call_a,
            close_IV=close_IV_a,
            df=df_a,
            w=w_a,
            F_t=F_t,
            K_t=K_t,
            T_t=T_t,
            is_call_t=is_call_t,
            close_IV_t=close_IV_t,
            df_t=df_t,
            w_t=w_t,
        )
