from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class OptionBatch:
    timestamp: str
    F: np.ndarray
    K: np.ndarray
    T: np.ndarray
    is_call: np.ndarray
    close: np.ndarray
    close_IV: np.ndarray
    df: np.ndarray
    w: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.F)
        for field_name in ("K", "T", "is_call", "close", "close_IV", "df", "w"):
            arr = getattr(self, field_name)
            if len(arr) != n:
                raise ValueError(
                    f"OptionBatch.{field_name} has length {len(arr)}, expected {n}"
                )
        for key, arr in self.extras.items():
            if len(arr) != n:
                raise ValueError(
                    f"OptionBatch.extras['{key}'] has length {len(arr)}, expected {n}"
                )

    def __len__(self) -> int:
        return len(self.F)

    def filter(self, mask: np.ndarray) -> "OptionBatch":
        return OptionBatch(
            timestamp=self.timestamp,
            F=self.F[mask],
            K=self.K[mask],
            T=self.T[mask],
            is_call=self.is_call[mask],
            close=self.close[mask],
            close_IV=self.close_IV[mask],
            df=self.df[mask],
            w=self.w[mask],
            extras={k: v[mask] for k, v in self.extras.items()},
        )

    def slice(self, indices: np.ndarray) -> "OptionBatch":
        return OptionBatch(
            timestamp=self.timestamp,
            F=self.F[indices],
            K=self.K[indices],
            T=self.T[indices],
            is_call=self.is_call[indices],
            close=self.close[indices],
            close_IV=self.close_IV[indices],
            df=self.df[indices],
            w=self.w[indices],
            extras={k: v[indices] for k, v in self.extras.items()},
        )


@dataclass(frozen=True, slots=True)
class PricedBatch:
    source: OptionBatch
    columns: dict[str, np.ndarray]
    numeric_state: dict[str, float]
    string_state: dict[str, str]

    def __len__(self) -> int:
        return len(self.source)

    @property
    def timestamp(self) -> str:
        return self.source.timestamp

    def col(self, name: str) -> np.ndarray:
        if name not in self.columns:
            raise KeyError(f"PricedBatch has no column '{name}'. Available: {sorted(self.columns)}")
        return self.columns[name]


class DataSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[OptionBatch]:
        raise NotImplementedError
