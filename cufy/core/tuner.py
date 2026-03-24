from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import optuna
import optuna.trial

from cufy.core.calibrator import Calibrator
from cufy.core.data import PreparedBatch
from cufy.core.model import Model


class TunableModel(Model, ABC):
    @contextmanager
    @abstractmethod
    def temporary_state(self) -> Iterator[None]:
        raise NotImplementedError

    @abstractmethod
    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        raise NotImplementedError


class TunableCalibrator[TM: TunableModel](Calibrator[TM], ABC):
    @contextmanager
    @abstractmethod
    def temporary_state(self) -> Iterator[None]:
        raise NotImplementedError

    @abstractmethod
    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, batch: PreparedBatch) -> float:
        raise NotImplementedError


def random_split(train_frac: float = 0.8) -> Callable[[PreparedBatch], tuple[PreparedBatch, PreparedBatch]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    rng = np.random.default_rng(None)

    def _split(batch: PreparedBatch) -> tuple[PreparedBatch, PreparedBatch]:
        n = len(batch)
        if n < 2:
            raise ValueError(f"random_split requires at least 2 options, got {n}")
        n_train = min(n - 1, int(n * train_frac))
        chosen = rng.choice(n, size=n_train, replace=False)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[chosen] = True
        return batch.filter(train_mask), batch.filter(~train_mask)

    return _split


class Tuner[TM: TunableModel](Calibrator[TM]):
    def __init__(
        self,
        base_calibrator: TunableCalibrator[TM],
        n_trials: int,
        split_fn: Callable[[PreparedBatch], tuple[PreparedBatch, PreparedBatch]] | None = None,
        study_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base_calibrator.model)
        self.base_calibrator = base_calibrator
        self.n_trials = n_trials
        self.split_fn = split_fn or random_split()
        self.study_kwargs: dict[str, Any] = study_kwargs or {}
        self.best_trial: optuna.trial.FrozenTrial | None = None

    def calibrate(self, batch: PreparedBatch) -> None:
        train_batch, val_batch = self.split_fn(batch)

        def objective(trial: optuna.trial.Trial) -> float:
            with self.model.temporary_state(), self.base_calibrator.temporary_state():
                self.model.suggest(trial)
                self.base_calibrator.suggest(trial)
                self.base_calibrator.calibrate(train_batch)
                return self.base_calibrator.score(val_batch)

        kwargs: dict[str, Any] = {"direction": "minimize", **self.study_kwargs}
        study = optuna.create_study(**kwargs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)

        self.best_trial = study.best_trial
        self.model.suggest(optuna.trial.FixedTrial(study.best_params))
        self.base_calibrator.suggest(optuna.trial.FixedTrial(study.best_params))
        self.base_calibrator.calibrate(batch)
