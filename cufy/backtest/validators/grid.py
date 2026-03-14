from abc import ABC, abstractmethod
from typing import Any

from cufy.abc.data import OptionBatch
from cufy.abc.validated_model import ValidatedModel
from cufy.backtest.splits import make_within_batch_split


class GridSearchValidatedModel(ValidatedModel, ABC):
    """
    A mixin/base class that implements the `calibrate` method using
    a deterministic grid search over hyperparameters.
    """

    @abstractmethod
    def hyperparam_grid(self) -> dict[str, list[Any]]:
        """
        Return a dictionary mapping hyperparameter names to a list of possible values.
        """
        raise NotImplementedError

    def calibrate(self, batch: OptionBatch) -> None:
        grid = self.hyperparam_grid()
        if not grid:
            self.calibrate_inner(batch)
            return

        split = make_within_batch_split(batch)
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]

        best_score = float("inf")
        best_hyperparams: dict[str, Any] = {}

        def _iter_grid(idx: int, current: dict[str, Any]) -> None:
            nonlocal best_score, best_hyperparams
            if idx == len(keys):
                self.apply_hyperparams(current)
                self.calibrate_inner(split.train)
                pred = self.price(split.validation)
                score = self.validation_metric(pred, split.validation.close, split.validation.w)
                if score < best_score:
                    best_score = score
                    best_hyperparams = dict(current)
                return
            for v in value_lists[idx]:
                current[keys[idx]] = v
                _iter_grid(idx + 1, current)

        _iter_grid(0, {})

        # Apply best params and train on the FULL batch
        self.apply_hyperparams(best_hyperparams)
        self.calibrate_inner(batch)
        
        # Save telemetry
        self.save_numeric("val_best_score", best_score)
        for k, v in best_hyperparams.items():
            if isinstance(v, (int, float)):
                self.save_numeric(f"hp_{k}", float(v))
            else:
                self.save_string(f"hp_{k}", str(v))
