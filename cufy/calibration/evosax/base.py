import copy
import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import evosax
import jax
import jax.numpy as jnp
import optuna
import torch
import torch.utils.dlpack

import cufy.config as config
from cufy.core.data import PreparedBatch
from cufy.core.tuner import TunableCalibrator
from cufy.models.base import TorchParameterizedModel
from cufy.utils.implied_vol import weighted_iv
from cufy.utils.memory_probation import probe_param_batch_sizes
from cufy.utils.torch_utils import TorchPreparedBatch

jax.config.update("jax_enable_x64", config.dtype_str == "float64")


@dataclass
class BaseEvoSaxConfig:
    num_populations: int
    popsize: int
    num_generations: int
    calibrate_radii: dict[str, float]
    initial_populations_scale: float
    initial_generations_scale: float
    strategy_kwargs: dict[str, Any]
    available_memory_fraction: float
    probe_max_candidates_fraction: float

    def __post_init__(self) -> None:
        if self.num_populations < 1:
            raise ValueError(f"num_populations must be >= 1, got {self.num_populations}")
        if self.popsize < 2:
            raise ValueError(f"popsize must be >= 2, got {self.popsize}")
        if self.num_generations < 1:
            raise ValueError(f"num_generations must be >= 1, got {self.num_generations}")
        bad = [name for name, r in self.calibrate_radii.items() if r < 0.0]
        if bad:
            raise ValueError(f"calibrate_radii has negative radius for parameters: {bad}")
        if self.initial_populations_scale <= 0.0:
            raise ValueError("initial_populations_scale must be > 0.0")
        if self.initial_generations_scale <= 0.0:
            raise ValueError("initial_generations_scale must be > 0.0")
        if not (0.0 < self.available_memory_fraction <= 1.0):
            raise ValueError("available_memory_fraction must be in (0.0, 1.0]")
        if self.probe_max_candidates_fraction <= 0.0:
            raise ValueError("probe_max_candidates_fraction must be > 0.0")


class BaseEvoSaxCalibrator(TunableCalibrator[TorchParameterizedModel]):
    strategy_name: str

    def __init__(self, model: TorchParameterizedModel, cfg: BaseEvoSaxConfig):
        super().__init__(model)
        self.cfg = cfg
        self._param_batch_sizer: Callable[[int], int] | None = None
        self._param_batch_size: int | None = None

        if "sigma_init" in cfg.strategy_kwargs:
            raise ValueError(
                "sigma_init must not be set in strategy_kwargs: it is derived automatically from Sobol coverage."
            )

        param_names = {p.name for p in model.params}
        missing = sorted(param_names - set(cfg.calibrate_radii.keys()))
        if missing:
            raise ValueError(
                f"cfg.calibrate_radii is missing entries for parameters: {missing}. "
                f"All model parameters must have a calibration radius"
            )
        extra = sorted(set(cfg.calibrate_radii.keys()) - param_names)
        if extra:
            raise ValueError(f"cfg.calibrate_radii has entries for unknown parameters: {extra}")

        StrategyClass = evosax.Strategies.get(self.strategy_name)
        if StrategyClass is None:
            raise ValueError(
                f"Unknown EvoSAX strategy: '{self.strategy_name}'. Available strategies: {evosax.Strategies.keys()}"
            )
        self._strategy = StrategyClass(popsize=cfg.popsize, num_dims=len(model.params))
        s = self._strategy
        self._batched_init = jax.jit(jax.vmap(s.initialize, in_axes=(0, None, 0)))
        self._batched_ask = jax.jit(jax.vmap(s.ask, in_axes=(0, 0, None)))
        self._batched_tell = jax.jit(jax.vmap(s.tell, in_axes=(0, 0, 0, None)))

    @property
    def _checked_param_batch_size(self) -> int:
        if self._param_batch_size is None:
            raise RuntimeError("_update_peak_bytes() has not been called yet — call calibrate() first")
        return self._param_batch_size

    @contextmanager
    def temporary_state(self) -> Iterator[None]:
        saved_cfg = copy.deepcopy(self.cfg)
        try:
            yield
        finally:
            self.cfg = saved_cfg

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        raise NotImplementedError

    def score(self, batch: PreparedBatch) -> float:
        if not self.model.is_initialized():
            raise RuntimeError("score() called on uninitialized model")
        data = TorchPreparedBatch.from_batch(batch)
        d = config.device
        p_cur = torch.tensor([p.value for p in self.model.params], device=d, dtype=config.dtype).unsqueeze(0)
        with torch.no_grad():
            preds = self.model.prices_for_param_matrix(data=data, param_matrix=p_cur)
            out = weighted_iv(data=data, pred_prices=preds).detach()
        return float(torch.nan_to_num(out, nan=1e9, posinf=1e9)[0].item())

    def probe(
        self, *, data: TorchPreparedBatch, pmin: torch.Tensor, pmax: torch.Tensor
    ) -> Callable[[int, int], Callable[[], None]]:
        d = config.device
        dt = config.dtype

        def make_f(m: int, n: int):
            def _run():
                with torch.no_grad():
                    U = torch.rand((m, pmin.numel()), device=d, dtype=dt)
                    P = torch.lerp(pmin, pmax, U)
                    _ = self.model.prices_for_param_matrix(data=data.slice(n), param_matrix=P)

            return _run

        return make_f

    def _update_peak_bytes(self, data: TorchPreparedBatch, num_populations: int) -> None:
        n_bucket = int(data.F_t.shape[0])
        if self._param_batch_sizer is not None:
            self._param_batch_size = self._param_batch_sizer(n_bucket)
            return

        d = config.device
        dt = config.dtype
        pmin = torch.tensor([p.min_value for p in self.model.params], device=d, dtype=dt)
        pmax = torch.tensor([p.max_value for p in self.model.params], device=d, dtype=dt)

        self._param_batch_sizer = probe_param_batch_sizes(
            make_f=self.probe(data=data, pmin=pmin, pmax=pmax),
            max_n=n_bucket,
            max_m=max(1, int(num_populations * self.cfg.popsize * self.cfg.probe_max_candidates_fraction)),
            available_memory_fraction=self.cfg.available_memory_fraction,
        )
        self._param_batch_size = self._param_batch_sizer(n_bucket)

    def _sobol_init_means(self, num_populations: int) -> jax.Array:
        engine = torch.quasirandom.SobolEngine(dimension=len(self.model.params), scramble=True)
        sobol_unit = engine.draw(num_populations).to(device=config.device, dtype=config.dtype)
        return jnp.from_dlpack(sobol_unit)

    def _evo_params(self, num_populations: int) -> Any:
        cell_size = num_populations ** (-1.0 / len(self.model.params))
        sigma_init = max(cell_size / 6.0, config.eps_float)
        return dataclasses.replace(self._strategy.default_params, **self.cfg.strategy_kwargs, sigma_init=sigma_init)

    def _initialize_state(self, init_rngs: jax.Array, num_populations: int) -> tuple[Any, Any]:
        raise NotImplementedError

    def calibrate(self, batch: PreparedBatch) -> None:
        # All EvoSax strategies operate in normalised [0, 1]^d space: candidates from ask() are
        # denormalised to [p_min, p_max] here, and only denormalised values are scored.
        # This means the quality of calibrate_radii (p_range = p_max - p_min) directly determines
        # the physical region explored — unlike PuzirCalibrator which normalises distances internally
        # and is therefore less sensitive to p_range choice.
        # With a reasonable p_range the algorithm remains correct: CMA-ES adapts its covariance C
        # to the landscape shape within whatever region is given, so a suboptimal p_range shrinks
        # or expands the search region but does not break algorithmic correctness.
        d = config.device
        dt = config.dtype
        data = TorchPreparedBatch.from_batch(batch)

        if not self.model.is_initialized():
            p_min = torch.tensor([p.min_value for p in self.model.params], device=d, dtype=dt)
            p_max = torch.tensor([p.max_value for p in self.model.params], device=d, dtype=dt)
            num_populations = max(1, int(self.cfg.num_populations * self.cfg.initial_populations_scale))
            num_generations = max(1, int(self.cfg.num_generations * self.cfg.initial_generations_scale))
        else:
            radii = self.cfg.calibrate_radii
            p_min = torch.tensor(
                [max(p.min_value, p.value - radii[p.name]) for p in self.model.params],  # type: ignore[operator]
                device=d,
                dtype=dt,
            )
            p_max = torch.tensor(
                [min(p.max_value, p.value + radii[p.name]) for p in self.model.params],  # type: ignore[operator]
                device=d,
                dtype=dt,
            )
            num_populations = self.cfg.num_populations
            num_generations = self.cfg.num_generations

        self._update_peak_bytes(data=data, num_populations=num_populations)
        eval_batch_size = self._checked_param_batch_size

        rng = jax.random.PRNGKey(42)
        rng, step_rng = jax.random.split(rng)
        init_rngs = jax.random.split(step_rng, num_populations)

        state, evo_params = self._initialize_state(init_rngs, num_populations)

        global_best_score = float("inf")
        global_best_params = None
        _inf = torch.tensor(float("inf"), device=d, dtype=dt)
        for _ in range(num_generations):
            rng, step_rng = jax.random.split(rng)
            ask_rngs = jax.random.split(step_rng, num_populations)

            x_jax, state = self._batched_ask(ask_rngs, state, evo_params)

            x_norm = torch.utils.dlpack.from_dlpack(x_jax).to(device=d, dtype=dt)
            x_norm_clamped = x_norm.view(-1, len(self.model.params)).clamp(0.0, 1.0)
            x_flat = x_norm_clamped * (p_max - p_min) + p_min
            scores_flat = torch.empty(x_flat.shape[0], device=d, dtype=dt)

            with torch.no_grad():
                for s in range(0, x_flat.shape[0], eval_batch_size):
                    chunk = x_flat[s : s + eval_batch_size]
                    preds = self.model.prices_for_param_matrix(data=data, param_matrix=chunk)
                    scores_flat[s : s + eval_batch_size] = weighted_iv(data=data, pred_prices=preds).detach()

            valid_mask = torch.isfinite(scores_flat)
            if valid_mask.any():
                valid_scores = torch.where(valid_mask, scores_flat, _inf)
                min_idx = torch.argmin(valid_scores)
                min_score = float(valid_scores[min_idx].item())
                if min_score < global_best_score:
                    global_best_score = min_score
                    global_best_params = x_flat[min_idx].clone()

            scores_tell = torch.nan_to_num(scores_flat.view(num_populations, self.cfg.popsize), nan=1e9, posinf=1e9)
            x_tell_jax = jax.dlpack.from_dlpack(
                x_norm_clamped.view(num_populations, self.cfg.popsize, len(self.model.params)).contiguous()
            )
            state = self._batched_tell(x_tell_jax, jax.dlpack.from_dlpack(scores_tell), state, evo_params)

        if global_best_params is None:
            raise RuntimeError(
                "Calibration failed: all scores were non-finite across all generations and populations. "
                "Check model parameters bounds, input data validity, or reduce calibrate_radii."
            )
        for i, p in enumerate(self.model.params):
            p.value = float(global_best_params[i].item())
