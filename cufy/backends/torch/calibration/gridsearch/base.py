import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import optuna
import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.implied_vol import weighted_iv
from cufy.backends.torch.utils.memory_probation import probe_param_batch_sizes
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch
from cufy.core.data import PreparedBatch
from cufy.core.tuner import TunableCalibrator


@dataclass
class GridSearchConfig:
    initial_sampler: Literal["grid", "sobol"]
    initial_points: int | None
    calibrate_points_fraction: float | None
    history_points_fraction: float
    available_memory_fraction: float
    probe_max_candidates_fraction: float
    calibrate_radii: dict[str, float]
    grid_points_initial: dict[str, int]
    grid_points_calibrate: dict[str, int]

    def __post_init__(self) -> None:
        if self.initial_points is not None and self.initial_points < 1:
            raise ValueError("initial_points must be >= 1")
        if self.calibrate_points_fraction is not None and self.calibrate_points_fraction <= 0.0:
            raise ValueError("calibrate_points_fraction must be > 0.0")
        if self.history_points_fraction <= 0.0:
            raise ValueError("history_points_fraction must be > 0.0")
        if not (0.0 <= self.available_memory_fraction <= 1.0):
            raise ValueError("available_memory_fraction must be in [0.0, 1.0]")
        if self.probe_max_candidates_fraction <= 0.0:
            raise ValueError("probe_max_candidates_fraction must be > 0.0")


class GridSearchCalibrator(TunableCalibrator[TorchParameterizedModel]):
    def __init__(self, model: TorchParameterizedModel, cfg: GridSearchConfig):
        super().__init__(model)
        self.cfg = cfg

        param_names = {p.name for p in model.params}
        radii = self.cfg.calibrate_radii
        missing = sorted(param_names - set(radii.keys()))
        if missing:
            raise ValueError(
                f"cfg.calibrate_radii is missing entries for parameters: {missing}. "
                f"All model parameters must have a calibration radius"
            )
        extra = sorted(set(radii.keys()) - param_names)
        if extra:
            raise ValueError(f"cfg.calibrate_radii has entries for unknown parameters: {extra}")
        bad = [name for name, r in radii.items() if r < 0.0]
        if bad:
            raise ValueError(f"cfg.calibrate_radii has negative radius for parameters: {bad}")

        if self.cfg.initial_sampler == "sobol":
            if self.cfg.initial_points is None:
                raise ValueError("cfg.initial_points must be set when initial_sampler='sobol'")
            if self.cfg.calibrate_points_fraction is None:
                raise ValueError("cfg.calibrate_points_fraction must be set when initial_sampler='sobol'")
        elif self.cfg.initial_sampler == "grid":
            for field_name, pts_dict in (
                ("grid_points_initial", self.cfg.grid_points_initial),
                ("grid_points_calibrate", self.cfg.grid_points_calibrate),
            ):
                missing_pts = sorted(param_names - set(pts_dict.keys()))
                if missing_pts:
                    raise ValueError(
                        f"cfg.{field_name} is missing entries for parameters: {missing_pts}. "
                        f"All model parameters must have grid point counts when initial_sampler='grid'"
                    )
                extra_pts = sorted(set(pts_dict.keys()) - param_names)
                if extra_pts:
                    raise ValueError(f"cfg.{field_name} has entries for unknown parameters: {extra_pts}")
                bad_pts = [name for name, n in pts_dict.items() if n < 1]
                if bad_pts:
                    raise ValueError(f"cfg.{field_name} has point counts < 1 for parameters: {bad_pts}")

        self._param_batch_sizer: Callable[[int], int] | None = None
        self._param_batch_size: int | None = None
        self._hist_params: torch.Tensor = torch.empty(0, device=config.device, dtype=config.dtype)
        self._hist_scores: torch.Tensor = torch.empty(0, device=config.device, dtype=config.dtype)
        self._history_size: int | None = None
        self._search_points: int | None = None
        self._search_points_detailed: list[int] = []

    @property
    def _checked_param_batch_size(self) -> int:
        if self._param_batch_size is None:
            raise RuntimeError("_update_peak_bytes() has not been called yet — call calibrate() first")
        return self._param_batch_size

    @property
    def _checked_history_size(self) -> int:
        if self._history_size is None:
            raise RuntimeError("_update_peak_bytes() has not been called yet — call calibrate() first")
        return self._history_size

    @property
    def _checked_search_points(self) -> int:
        if self._search_points is None:
            raise RuntimeError("_set_explore_plan() has not been called yet")
        return self._search_points

    @contextmanager
    def temporary_state(self) -> Iterator[None]:
        saved_cfg = copy.deepcopy(self.cfg)
        try:
            yield
        finally:
            self.cfg = saved_cfg

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        pass

    def score(self, batch: PreparedBatch) -> float:
        if not self.model.is_initialized():
            raise RuntimeError("score() called on uninitialized model")
        data = TorchPreparedBatch.from_batch(batch)
        d = config.device
        p_cur = torch.tensor([p.value for p in self.model.params], device=d, dtype=config.dtype).unsqueeze(0)
        return float(self._score_params(P=p_cur, data=data)[0].item())

    def _set_explore_plan(self) -> None:
        is_initial_stage = not self.model.is_initialized()
        if self.cfg.initial_sampler == "sobol":
            self._search_points = (
                self.cfg.initial_points
                if is_initial_stage
                else max(1, int(self.cfg.initial_points * self.cfg.calibrate_points_fraction))  # type: ignore[arg-type]
            )
            return

        pts_dict = self.cfg.grid_points_initial if is_initial_stage else self.cfg.grid_points_calibrate
        self._search_points_detailed = [pts_dict[p.name] for p in self.model.params]
        self._search_points = int(np.prod(self._search_points_detailed))

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

    def _update_peak_bytes(self, data: TorchPreparedBatch) -> None:
        self._set_explore_plan()
        n_bucket = int(data.F_t.shape[0])
        if self._param_batch_sizer is not None:
            self._param_batch_size = self._param_batch_sizer(n_bucket)
            return

        d = config.device
        dt = config.dtype
        pmin = torch.tensor([p.min_value for p in self.model.params], device=d, dtype=dt)
        pmax = torch.tensor([p.max_value for p in self.model.params], device=d, dtype=dt)
        make_f = self.probe(data=data, pmin=pmin, pmax=pmax)

        self._param_batch_sizer = probe_param_batch_sizes(
            make_f=make_f,
            max_n=n_bucket,
            max_m=max(1, int(self._checked_search_points * self.cfg.probe_max_candidates_fraction)),
            available_memory_fraction=self.cfg.available_memory_fraction,
        )
        self._param_batch_size = self._param_batch_sizer(n_bucket)
        self._history_size = max(1, int(self._checked_search_points * self.cfg.history_points_fraction))

    def _score_params(self, P: torch.Tensor, data: TorchPreparedBatch) -> torch.Tensor:
        with torch.no_grad():
            preds = self.model.prices_for_param_matrix(data=data, param_matrix=P)
            out = weighted_iv(data=data, pred_prices=preds).detach()
        return torch.nan_to_num(out, nan=1e9, posinf=1e9)

    def _history_merge(self, params: torch.Tensor, scores: torch.Tensor) -> None:
        K = self._checked_history_size
        all_scores = torch.cat([self._hist_scores, scores])
        all_params = torch.cat([self._hist_params, params])
        if all_scores.numel() <= K:
            self._hist_scores = all_scores
            self._hist_params = all_params
        else:
            idx = torch.topk(all_scores, k=K, largest=False, sorted=False).indices
            self._hist_scores = all_scores[idx]
            self._hist_params = all_params[idx]

    def _explore(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        d = config.device
        p_dim = int(p_min.numel())
        N = self._checked_search_points
        bs = self._checked_param_batch_size

        self._hist_params = torch.empty(0, device=d, dtype=config.dtype)
        self._hist_scores = torch.empty(0, device=d, dtype=config.dtype)

        if self.cfg.initial_sampler == "sobol":
            engine = torch.quasirandom.SobolEngine(dimension=p_dim, scramble=True)
            for s in range(0, N, bs):
                U = engine.draw(min(bs, N - s)).to(device=d, dtype=config.dtype)
                candidates = torch.lerp(p_min, p_max, U).clamp(min=p_min, max=p_max)
                scores = self._score_params(candidates, data)
                self._history_merge(candidates, scores)
        else:
            dims = torch.tensor(self._search_points_detailed, device=d, dtype=torch.long)
            steps = (p_max - p_min) / (dims - 1).clamp(min=1).to(config.dtype)
            for s in range(0, N, bs):
                flat_indices = torch.arange(s, s + min(bs, N - s), device=d, dtype=torch.long)
                rem = flat_indices
                multi_indices = []
                for i in range(p_dim - 1, -1, -1):
                    multi_indices.append(rem % dims[i])
                    rem = rem // dims[i]
                multi_indices = multi_indices[::-1]
                candidates = torch.stack(
                    [p_min[i] + multi_indices[i].to(config.dtype) * steps[i] for i in range(p_dim)], dim=1
                )
                scores = self._score_params(candidates, data)
                self._history_merge(candidates, scores)

        best_p = self._hist_params[torch.argmin(self._hist_scores)]
        for i, p in enumerate(self.model.params):
            p.value = float(best_p[i].item())

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        return

    def _run_search_stage(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        self._explore(p_min=p_min, p_max=p_max, data=data)
        self.refine(p_min=p_min, p_max=p_max, data=data)

    def calibrate(self, batch: PreparedBatch) -> None:
        d = config.device
        dt = config.dtype
        data = TorchPreparedBatch.from_batch(batch)

        if not self.model.is_initialized():
            p_min = torch.tensor([p.min_value for p in self.model.params], device=d, dtype=dt)
            p_max = torch.tensor([p.max_value for p in self.model.params], device=d, dtype=dt)
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

        self._update_peak_bytes(data=data)
        self._run_search_stage(p_min=p_min, p_max=p_max, data=data)
