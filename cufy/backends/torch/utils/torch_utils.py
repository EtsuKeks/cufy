from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

import cufy.backends.torch.config as config
from cufy.core.data import PreparedBatch


def to_torch_1d(x: np.ndarray) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D numpy array, got shape={x.shape}")
    return torch.from_numpy(x).to(device=config.device, dtype=config.dtype)


_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class TorchPreparedBatch:
    timestamp: datetime
    F_scale_t: torch.Tensor
    F_t: torch.Tensor
    K_t: torch.Tensor
    T_t: torch.Tensor
    is_call_t: torch.Tensor
    close_t: torch.Tensor
    close_IV_t: torch.Tensor
    df_t: torch.Tensor
    extras_t: dict[str, torch.Tensor]
    w_t: torch.Tensor
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _SENTINEL:
            raise TypeError("TorchPreparedBatch cannot be instantiated directly — use TorchPreparedBatch.from_batch()")

    def slice(self, n: int, clone: bool = False) -> TorchPreparedBatch:
        length = int(self.F_t.shape[0])
        if n <= 0 or n > length:
            raise ValueError(f"n must be in [1, {length}], got {n}")

        def _s(t: torch.Tensor) -> torch.Tensor:
            return t[:n].clone() if clone else t[:n]

        return TorchPreparedBatch(
            timestamp=self.timestamp,
            F_scale_t=_s(self.F_scale_t),
            F_t=_s(self.F_t),
            K_t=_s(self.K_t),
            T_t=_s(self.T_t),
            is_call_t=_s(self.is_call_t),
            close_t=_s(self.close_t),
            close_IV_t=_s(self.close_IV_t),
            df_t=_s(self.df_t),
            extras_t={
                k: _s(v) if v.shape[0] == length else (v.clone() if clone else v) for k, v in self.extras_t.items()
            },
            w_t=_s(self.w_t),
            _token=_SENTINEL,
        )

    @staticmethod
    def from_batch(batch: PreparedBatch) -> TorchPreparedBatch:
        F_scale = to_torch_1d(batch.F)
        return TorchPreparedBatch(
            timestamp=batch.timestamp,
            F_scale_t=F_scale,
            F_t=torch.ones_like(F_scale),
            K_t=to_torch_1d(batch.K) / F_scale,
            T_t=to_torch_1d(batch.T),
            is_call_t=torch.from_numpy(batch.is_call).to(device=config.device, dtype=torch.bool),
            close_t=to_torch_1d(batch.close) / F_scale,
            close_IV_t=to_torch_1d(batch.close_IV),
            df_t=to_torch_1d(batch.df),
            extras_t={
                k: torch.from_numpy(v).to(device=config.device, dtype=config.dtype) for k, v in batch.extras.items()
            },
            w_t=to_torch_1d(batch.w),
            _token=_SENTINEL,
        )
