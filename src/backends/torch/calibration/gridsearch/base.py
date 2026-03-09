from abc import ABC
import time
from typing import Callable, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

from src.abc.parameterized_model import ModelParam, ParameterizedModel
from src.config.config import settings
from src.utils.implied_vol import weighted_iv_l2_from_prices
from src.utils.memory_probation import probe_param_batch_sizes
from src.utils.torch_utils import Batch1D


class GridSearchModelParam(ModelParam):
    points_initial: int | None = Field(default=None, ge=1)
    points_calibrate: int | None = Field(default=None, ge=1)
    radius_calibrate: float | None = Field(default=None, ge=0.0)


class GridSearchSettings(BaseModel):
    initial_sampler: Literal["grid", "sobol"] = "sobol"
    initial_batches: int | None = Field(default=None, ge=1)
    calibrate_batches: int | None = Field(default=None, ge=1)
    history_batches: int = Field(default=8, ge=1)
    available_memory_fraction: float = Field(default=0.85, ge=0.0, le=1.0)
    probe_max_candidates: int = Field(default=4096, ge=1)


class GridSearchModel(ParameterizedModel, ABC):
    def __init__(
        self, param_overrides: dict[str, GridSearchModelParam] | None = None, gs: GridSearchSettings | None = None
    ):
        super().__init__(param_overrides)
        self._param_batch_sizer: Callable[[int], int] | None = None
        self._param_batch_size: int | None = None
        self._hist_params: torch.Tensor | None = None
        self._hist_scores: torch.Tensor | None = None
        self.gs: GridSearchSettings = gs or GridSearchSettings()
        self._history_size: int | None = None
        self._search_batches: int | None = None
        self._search_points_detailed: list[int] | None = None

    def _set_explore_plan(self) -> None:
        is_initial_stage = any(p.value is None for p in self.params)
        if self.gs.initial_sampler == "sobol":
            batches = self.gs.initial_batches if is_initial_stage else self.gs.calibrate_batches
            if batches is None:
                stage_name = "initial" if is_initial_stage else "calibrate"
                raise ValueError(f"gs.{stage_name}_batches must be set when initial_sampler='sobol'")
            self._search_batches = batches
            self._search_points_detailed = None
            return

        self._search_batches = None
        self._search_points_detailed = [
            p.points_initial if is_initial_stage else p.points_calibrate for p in self.params
        ]
        if any(x is None for x in self._search_points_detailed):
            stage_name = "initial" if is_initial_stage else "calibrate"
            raise ValueError(
                f"Grid sampler requires points_{stage_name} for all parameters when initial_sampler='grid'"
            )

    def _update_peak_bytes(self, *, data: Batch1D) -> None:
        self._set_explore_plan()
        n_bucket = int(data.S.shape[0])
        if self._param_batch_sizer is not None:
            self._param_batch_size = self._param_batch_sizer(n_bucket)
            return

        d = settings.device
        dt = settings.dtype
        pmin = torch.tensor([p.min_value for p in self.params], device=d, dtype=dt)
        pmax = torch.tensor([p.max_value for p in self.params], device=d, dtype=dt)
        make_f = self.probe(data=data, pmin=pmin, pmax=pmax)

        self._param_batch_sizer = probe_param_batch_sizes(
            make_f=make_f,
            max_n=n_bucket,
            max_m=self.gs.probe_max_candidates,
            available_memory_fraction=self.gs.available_memory_fraction,
        )
        self._param_batch_size = self._param_batch_sizer(n_bucket)

    def probe(
        self, *, data: Batch1D, pmin: torch.Tensor, pmax: torch.Tensor
    ) -> Callable[[int, int], Callable[[], None]]:
        d = settings.device
        dt = settings.dtype
        p_dim = int(pmin.numel())

        def make_f(m: int, n: int):
            def _run():
                with torch.no_grad():
                    P = pmin[None, :] + (pmax - pmin)[None, :] * torch.rand((m, p_dim), device=d, dtype=dt)
                    _ = self.prices_for_param_matrix(data=data.slice(n), param_matrix=P)
            return _run

        return make_f

    def _score_params(self, P: torch.Tensor, data: Batch1D) -> torch.Tensor:
        d = settings.device
        out = torch.empty((int(P.shape[0]),), device=d, dtype=settings.dtype)
        with torch.no_grad():
            for s in range(0, int(P.shape[0]), self._param_batch_size):
                Pc = P[s : s + self._param_batch_size]
                preds = self.prices_for_param_matrix(data=data, param_matrix=Pc)
                out[s : s + self._param_batch_size] = weighted_iv_l2_from_prices(data=data, pred_prices=preds)
        return out

    @property
    def _explore_budget_points(self) -> int:
        if self.gs.initial_sampler == "sobol":
            return self._search_batches * self._param_batch_size
        return np.prod(self._search_points_detailed)

    def _history_set(self, params: torch.Tensor, scores: torch.Tensor) -> None:
        idx = torch.topk(scores, k=min(self._history_size, int(scores.numel())), largest=False, sorted=False).indices
        self._hist_scores = scores.index_select(0, idx).reshape(-1)
        self._hist_params = params.index_select(0, idx)

    def _history_merge(self, params: torch.Tensor, scores: torch.Tensor) -> None:
        if self._hist_params is None or self._hist_scores is None:
            self._history_set(params, scores)
            return

        self._history_set(
            params=torch.cat([self._hist_params, params], dim=0), scores=torch.cat([self._hist_scores, scores], dim=0)
        )

    def _explore(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: Batch1D) -> None:
        d = settings.device
        p_dim = int(p_min.numel())
        self._history_size = self.gs.history_batches * self._param_batch_size

        # Start a fresh history buffer for this exploration stage.
        self._hist_params = None
        self._hist_scores = None

        if self.gs.initial_sampler == "sobol":
            engine = torch.quasirandom.SobolEngine(dimension=p_dim, scramble=True)
            U = engine.draw(self._explore_budget_points).to(device=d, dtype=settings.dtype)
            P = (p_min[None, :] + U * (p_max - p_min)[None, :]).clamp(min=p_min, max=p_max)
        else:
            edges = [
                torch.linspace(
                    p_min[i].item(), p_max[i].item(), self._search_points_detailed[i], device=d, dtype=settings.dtype
                )
                for i in range(p_dim)
            ]
            idx = torch.arange(0, self._explore_budget_points, device=d, dtype=torch.int64)
            P = torch.empty((self._explore_budget_points, p_dim), device=d, dtype=settings.dtype)
            cur = idx
            for j in range(p_dim - 1, -1, -1):
                L = int(edges[j].numel())
                c = torch.remainder(cur, L).to(torch.int64)
                cur = cur // L
                P[:, j] = edges[j].index_select(0, c).to(dtype=settings.dtype)

        scores = self._score_params(P, data)
        self._history_merge(P, scores)
        best_p = P[int(torch.argmin(scores).item())].detach().clone()
        for i, p in enumerate(self.params):
            p.value = float(best_p[i].item())

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: Batch1D) -> None:
        return

    def _run_search_stage(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: Batch1D) -> None:
        if self._param_batch_size is not None:
            self.save_fact("explored", "batch_capacity", str(self._param_batch_size))

        if self.gs.initial_sampler == "sobol":
            self.save_fact("explored", "budget_batches", str(self._search_batches))
        else:
            self.save_fact("explored", "budget_points", str(self._search_points_detailed))

        t0 = time.perf_counter()
        self._explore(p_min=p_min, p_max=p_max, data=data)
        t1 = time.perf_counter()
        self.save_fact("explored", "elapsed_sec", str(t1 - t0))
        self.save_params("explored")

        t0 = time.perf_counter()
        self.refine(p_min=p_min, p_max=p_max, data=data)
        t1 = time.perf_counter()
        self.save_fact("refined", "elapsed_sec", str(t1 - t0))
        self.save_params("refined")

    def find_initial_params(
        self,
        *,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close: np.ndarray,
        close_IV: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
    ) -> None:
        if any(p.value is not None for p in self.params):
            raise RuntimeError(
                "find_initial_params() cannot be called after parameter values were set. Use calibrate()."
            )

        d = settings.device
        p_min = torch.tensor([p.min_value for p in self.params], device=d, dtype=settings.dtype)
        p_max = torch.tensor([p.max_value for p in self.params], device=d, dtype=settings.dtype)
        data = Batch1D.from_numpy(S=S, K=K, T=T, is_call=is_call, close_IV=close_IV, r=r, w=w)
        self._update_peak_bytes(data=data)
        self._run_search_stage(p_min=p_min, p_max=p_max, data=data)

    def calibrate(
        self,
        *,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        is_call: np.ndarray,
        close: np.ndarray,
        close_IV: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
    ) -> None:
        d = settings.device
        dt = settings.dtype

        if any(p.value is None for p in self.params):
            raise RuntimeError("calibrate() called before find_initial_params()")
        if any(p.radius_calibrate is None for p in self.params):
            raise ValueError("calibrate() requires radius_calibrate for all parameters")

        p_min = torch.tensor([max(p.min_value, p.value - p.radius_calibrate) for p in self.params], device=d, dtype=dt)
        p_max = torch.tensor([min(p.max_value, p.value + p.radius_calibrate) for p in self.params], device=d, dtype=dt)
        data = Batch1D.from_numpy(S=S, K=K, T=T, is_call=is_call, close_IV=close_IV, r=r, w=w)
        self._update_peak_bytes(data=data)
        self._run_search_stage(p_min=p_min, p_max=p_max, data=data)

    def price(self, *, S: np.ndarray, K: np.ndarray, T: np.ndarray, is_call: np.ndarray, r: np.ndarray) -> np.ndarray:
        if any(p.value is None for p in self.params):
            raise RuntimeError("price() called before find_initial_params()")

        P = torch.tensor([[p.value for p in self.params]], device=settings.device, dtype=settings.dtype)
        dummy_w = np.ones_like(S, dtype=np.float64)
        dummy_iv = np.zeros_like(S, dtype=np.float64)
        data = Batch1D.from_numpy(S=S, K=K, T=T, is_call=is_call, close_IV=dummy_iv, r=r, w=dummy_w)
        with torch.no_grad():
            out = self.prices_for_param_matrix(data=data, param_matrix=P)[0]
        return out.detach().cpu().numpy()
