import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    initial_sampler: Literal["grid", "sobol"] = "sobol"
    initial_points: int | None = None
    calibrate_points_fraction: float | None = None
    history_points_fraction: float = 0.1
    available_memory_fraction: float = 0.85
    probe_max_candidates_fraction: float = 0.1
    calibrate_radii: dict[str, float] = field(default_factory=dict)
    grid_points_initial: dict[str, int] = field(default_factory=dict)
    grid_points_calibrate: dict[str, int] = field(default_factory=dict)

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
    def __init__(self, model: TorchParameterizedModel, cfg: GridSearchConfig | None = None):
        super().__init__(model)
        self.cfg = cfg or GridSearchConfig()

        param_names = {p.name for p in model.params}
        radii = self.cfg.calibrate_radii
        missing = sorted(param_names - set(radii.keys()))
        if missing:
            raise ValueError(
                f"cfg.calibrate_radii is missing entries for parameters: {missing}. "
                f"All model parameters must have a calibration radius."
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
                        f"All model parameters must have grid point counts when initial_sampler='grid'."
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
                    P = pmin[None, :] + (pmax - pmin)[None, :] * torch.rand((m, pmin.numel()), device=d, dtype=dt)
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
        d = config.device
        bs = self._checked_param_batch_size
        out = torch.empty((int(P.shape[0]),), device=d, dtype=config.dtype)
        with torch.no_grad():
            for s in range(0, int(P.shape[0]), bs):
                preds = self.model.prices_for_param_matrix(data=data, param_matrix=P[s : s + bs])
                out[s : s + bs] = weighted_iv(data=data, pred_prices=preds).detach()
        return out

    def _history_merge(self, params: torch.Tensor, scores: torch.Tensor) -> None:
        K = self._checked_history_size
        all_scores = torch.cat([self._hist_scores, scores])
        all_params = torch.cat([self._hist_params, params])
        idx = torch.topk(all_scores, k=min(K, int(all_scores.numel())), largest=False, sorted=True).indices
        self._hist_scores = all_scores[idx]
        self._hist_params = all_params[idx]

    def _explore(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        d = config.device
        p_dim = int(p_min.numel())

        self._hist_params = torch.empty(0, device=config.device, dtype=config.dtype)
        self._hist_scores = torch.empty(0, device=config.device, dtype=config.dtype)

        if self.cfg.initial_sampler == "sobol":
            engine = torch.quasirandom.SobolEngine(dimension=p_dim, scramble=True)
            U = engine.draw(self._checked_search_points).to(device=d, dtype=config.dtype)
            candidates = (p_min[None, :] + U * (p_max - p_min)[None, :]).clamp(min=p_min, max=p_max)
        else:
            edges = [
                torch.linspace(
                    p_min[i].item(), p_max[i].item(), self._search_points_detailed[i], device=d, dtype=config.dtype
                )
                for i in range(p_dim)
            ]
            candidates = torch.stack([g.flatten() for g in torch.meshgrid(*edges, indexing="ij")], dim=1)

        scores = self._score_params(candidates, data)
        self._history_merge(candidates, scores)
        best_p = self._hist_params[0]
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
