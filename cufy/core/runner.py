from collections.abc import Callable
from typing import Any

import numpy as np

import cufy.config as config
from cufy.core.calibrator import Calibrator
from cufy.core.data import OptionBatch, PreparedBatch, PricedBatch


class Runner:
    def __init__(
        self,
        calibrator: Calibrator[Any],
        filter_fn: Callable[[OptionBatch], np.ndarray],
        weights_fn: Callable[[OptionBatch], np.ndarray],
    ) -> None:
        self._calibrator = calibrator
        self._filter_fn = filter_fn
        self._weights_fn = weights_fn

    def _prepare_batch(self, batch: OptionBatch) -> PreparedBatch:
        mask = self._filter_fn(batch)
        if mask.dtype != np.bool_ or mask.shape != (len(batch),):
            raise ValueError(
                f"filter_fn must return a bool array of shape ({len(batch)},), "
                f"got shape={mask.shape} dtype={mask.dtype}"
            )
        if not mask.any():
            raise ValueError("filter_fn returned an all-False mask — no options left in batch")

        filtered = batch.filter(mask)
        np_dt = np.dtype(config.dtype_str)
        w = self._weights_fn(filtered)
        if w.dtype != np_dt or w.shape != (len(filtered),):
            raise ValueError(
                f"weights_fn must return a numpy array of shape ({len(filtered)},) and dtype={np_dt}, "
                f"got shape={w.shape} dtype={w.dtype}"
            )

        total = np.sum(w)
        if not np.isclose(total, 1.0):
            raise ValueError(f"weights_fn returned weights that do not sum to 1.0, got {total:.6f}")
        return PreparedBatch.from_batch(filtered, w=w / total)

    def calibrate(self, batch: OptionBatch) -> None:
        self._calibrator.calibrate(self._prepare_batch(batch))

    def price(self, batch: OptionBatch) -> PricedBatch:
        prepared = self._prepare_batch(batch)
        return PricedBatch.from_prepared(prepared, price=self._calibrator.model.price(prepared))
