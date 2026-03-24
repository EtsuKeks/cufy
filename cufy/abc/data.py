import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

import numpy as np


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
        for f in dataclasses.fields(self):
            if f.name in {"timestamp", "F", "extras"}:
                continue
            arr = getattr(self, f.name)
            if not isinstance(arr, np.ndarray) or len(arr) != n:
                raise ValueError(f"OptionBatch.{f.name} must be a numpy array of length {n}, got {arr}")

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
