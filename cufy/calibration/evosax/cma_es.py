from typing import Any

import jax
import optuna

from .base import BaseEvoSaxCalibrator


class CmaEsCalibrator(BaseEvoSaxCalibrator):
    strategy_name = "CMA_ES"

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        current = self.cfg.strategy_kwargs.get("c_m", 1.0)
        self.cfg.strategy_kwargs["c_m"] = trial.suggest_float("c_m", max(0.1, current * 0.5), min(1.0, current * 1.5))

    def _initialize_state(self, init_rngs: jax.Array, num_populations: int) -> tuple[Any, Any]:
        evo_params = self._evo_params(num_populations)
        init_means = self._sobol_init_means(num_populations)
        state = self._batched_init(init_rngs, evo_params, init_means)
        return state, evo_params
