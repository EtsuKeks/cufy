import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import optuna

from .base import BaseEvoSaxCalibrator


class DeCalibrator(BaseEvoSaxCalibrator):
    strategy_name = "DE"

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        # F (scale factor): effective range [0.4, 1.0] per Das & Suganthan (2011);
        # F=0.8 per Vollrath & Wendland (2009) and Gong & Zhuang (2016) for financial models.
        # If user supplied a value, tune around it; otherwise search the full empirical range.
        cur_f = self.cfg.strategy_kwargs.get("diff_w", 0.8)
        self.cfg.strategy_kwargs["diff_w"] = trial.suggest_float("diff_w", max(0.4, cur_f * 0.5), min(1.0, cur_f * 1.5))

        # CR: if user supplied a value, tune around it (they know their model's separability).
        # CR=0.8 for strongly-coupled models (Heston, Bates); CR=0.3–0.5 for nearly-separable
        # ones (Merton) — pass strategy_kwargs={"cross_over_rate": 0.4} to shift the center.
        cur_cr = self.cfg.strategy_kwargs.get("cross_over_rate", 0.8)
        self.cfg.strategy_kwargs["cross_over_rate"] = trial.suggest_float(
            "cross_over_rate", max(0.05, cur_cr * 0.5), min(1.0, cur_cr * 1.5)
        )

        # num_diff_vectors: 1 vs 2 difference vectors in the mutation step.
        # 2 vectors (a + F*(b-c) + F*(d-e)) help escape local minima on the multimodal
        # Heston/Bates IV landscape. Only tune if the user has not pinned it explicitly.
        if "num_diff_vectors" not in self.cfg.strategy_kwargs:
            self.cfg.strategy_kwargs["num_diff_vectors"] = trial.suggest_categorical("num_diff_vectors", [1, 2])

    def _evo_params(self, num_populations: int) -> Any:
        # Use a random base vector (not the current best) for mutation. Gilli & Schumann (2010)
        # and Vollrath & Wendland (2009) both do this for Heston/Bates. Using the best vector
        # collapses population diversity prematurely on the flat regions of the IV landscape.
        return dataclasses.replace(
            self._strategy.default_params,
            **self.cfg.strategy_kwargs,
            mutate_best_vector=self.cfg.strategy_kwargs.get("mutate_best_vector", False),
            init_min=0.0,
            init_max=1.0,
        )

    def _initialize_state(self, init_rngs: jax.Array, num_populations: int) -> tuple[Any, Any]:
        evo_params = self._evo_params(num_populations)
        p_dim = len(self.model.params)
        state = self._batched_init(init_rngs, evo_params, jnp.zeros((num_populations, p_dim)))
        return state, evo_params
