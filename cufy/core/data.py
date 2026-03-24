import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import numpy as np

import cufy.config as config

_CONFIG_NP_DTYPE = np.dtype(config.dtype)


@dataclass(frozen=True, slots=True)
class OptionBatch:
    timestamp: datetime
    F: np.ndarray
    K: np.ndarray
    T: np.ndarray
    is_call: np.ndarray
    close: np.ndarray
    close_IV: np.ndarray
    df: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.F)
        if n == 0:
            raise ValueError("OptionBatch must have at least one option")

        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if f.name == "timestamp":
                continue
            elif f.name == "extras":
                for key, arr in val.items():
                    if arr.dtype != _CONFIG_NP_DTYPE:
                        raise ValueError(f"OptionBatch.extras[{key!r}] dtype {arr.dtype} does not match config dtype")
            else:
                if len(val) != n:
                    raise ValueError(f"OptionBatch.{f.name} must have length {n}, got {len(val)}")
                if f.name == "is_call":
                    if val.dtype != np.bool_:
                        raise ValueError(f"OptionBatch.{f.name} must have dtype bool, got {val.dtype}")
                else:
                    if val.dtype != _CONFIG_NP_DTYPE:
                        raise ValueError(f"OptionBatch.{f.name} dtype {val.dtype} does not match config dtype")

    def __len__(self) -> int:
        return len(self.F)

    def filter(self, mask: np.ndarray) -> Self:
        if mask.dtype != np.bool_:
            raise ValueError(f"filter mask must be a bool array, got dtype={mask.dtype}")
        if mask.shape != (len(self),):
            raise ValueError(f"filter mask must have shape ({len(self)},), got shape={mask.shape}")

        changes: dict[str, Any] = {
            f.name: getattr(self, f.name)[mask]
            for f in dataclasses.fields(self)
            if f.name not in ("timestamp", "extras")
        }
        changes["extras"] = (
            {k: v[mask] for k, v in self.extras.items()}
            if all(len(v) == len(self) for v in self.extras.values())
            else self.extras
        )
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PreparedBatch(OptionBatch):
    w: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isclose(np.sum(self.w), 1.0):
            raise ValueError(f"PreparedBatch.w must sum to 1.0, got {np.sum(self.w):.6f}")

    @classmethod
    def from_batch(cls, batch: OptionBatch, w: np.ndarray) -> PreparedBatch:
        return cls(**{f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}, w=w)


@dataclass(frozen=True, slots=True)
class PricedBatch(PreparedBatch):
    price: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_prepared(cls, batch: PreparedBatch, price: np.ndarray) -> PricedBatch:
        return cls(**{f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}, price=price)
