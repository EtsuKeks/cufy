import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from cufy.abc.data import OptionBatch, PricedBatch
from cufy.abc.model import Model
from cufy.abc.telemetry_model import TelemetryModel


class Runner(ABC):
    @property
    @abstractmethod
    def model(self) -> Model:
        raise NotImplementedError

    @abstractmethod
    def filter(self, batch: OptionBatch) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def weights(self, batch: OptionBatch) -> np.ndarray:
        raise NotImplementedError

    def prepare_batch(self, batch: OptionBatch) -> OptionBatch:
        mask = self.filter(batch)
        if mask.dtype != bool or mask.shape != (len(batch),):
            raise ValueError(
                f"filter() must return a bool array of shape ({len(batch)},), "
                f"got shape={mask.shape} dtype={mask.dtype}"
            )
        if not mask.any():
            raise ValueError("filter() returned an all-False mask — no options left in batch")
        filtered = batch.filter(mask)
        w = self.weights(filtered)
        if w.ndim != 1 or len(w) != len(filtered):
            raise ValueError(
                f"weights() must return a 1-D array of length {len(filtered)}, "
                f"got shape={w.shape}"
            )
        w = w.astype(float)
        total = float(np.sum(w))
        if total <= 0.0 or not np.isfinite(total):
            w = np.ones(len(filtered), dtype=float)
        else:
            w = w / total
        return dataclasses.replace(filtered, w=w)

    def is_initialized(self) -> bool:
        return self.model.is_initialized()

    def find_initial_params(self, batch: OptionBatch) -> None:
        self.model.find_initial_params(batch)

    def calibrate(self, batch: OptionBatch) -> None:
        self.model.calibrate(batch)

    def _enrich_columns(
        self, batch: OptionBatch, prices: np.ndarray, columns: dict[str, np.ndarray]
    ) -> None:
        pass

    def price(self, batch: OptionBatch) -> PricedBatch:
        prices = self.model.price(batch)

        numeric: dict[str, float] = {}
        string: dict[str, str] = {}
        if isinstance(self.model, TelemetryModel):
            numeric = self.model.get_numeric()
            string = self.model.get_string()

        columns: dict[str, np.ndarray] = {"price": prices}
        self._enrich_columns(batch, prices, columns)

        return PricedBatch(
            source=batch,
            columns=columns,
            numeric_state=numeric,
            string_state=string,
        )
