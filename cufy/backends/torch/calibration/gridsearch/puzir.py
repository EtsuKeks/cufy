from abc import ABC
from typing import Literal, Mapping

import numpy as np
import torch
from pydantic import Field

import cufy.config as config
from cufy.backends.torch.calibration.gridsearch.base import GridSearchModel, GridSearchModelParam, GridSearchSettings
from cufy.backends.torch.utils.torch_utils import Batch1D


class PuzirSettings(GridSearchSettings):
    refine_multiplier: float = Field(default=1.0, gt=0.0)
    probe_budget_frac: float = Field(default=0.1, gt=0.0, le=1.0)
    drift_method: Literal["none", "ridge", "cma"] = "ridge"
    drift_mahalanobis_step: float = Field(default=1.0, gt=0.0)
    cma_mu_frac: float = Field(default=0.5, gt=0.0, le=1.0)
    neighbors_k: int = Field(default=25, ge=1)
    elite_power: float = Field(default=2 / 3, gt=0.0, le=1.0)
    jitter_scale: float = Field(default=1.0, ge=0.0)
    damping: float = Field(default=1e-6, gt=0.0)


class PuzirModel(GridSearchModel, ABC):
    gs: PuzirSettings                                        

    def __init__(
        self, param_overrides: Mapping[str, GridSearchModelParam] | None = None, gs: PuzirSettings | None = None
    ):
        super().__init__(param_overrides, gs=gs or PuzirSettings())

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: Batch1D) -> None:
        budget = int(self._explore_budget_points * self.gs.refine_multiplier)
        if budget < 1:
            return

        d = config.device
        dt = config.dtype
        eps = torch.tensor(config.eps, device=d, dtype=dt)

        best_idx = torch.argmin(self._hist_scores)
        best_s = float(self._hist_scores[best_idx].item())
        best_p = self._hist_params[best_idx].detach().clone()
        p_dim = int(best_p.numel())
        self.save_numeric("refined_batch_capacity", float(self._param_batch_size))

        if self.gs.initial_sampler == "sobol":
            self.save_numeric("refined_budget_batches", float(self.gs.refine_multiplier))
        else:
            self.save_string("refined_budget_points", str(self._search_points_detailed))

        p_range = (p_max - p_min).clamp_min(eps)
        if self.gs.initial_sampler == "grid":
            denom = torch.tensor([max(x - 1, 1) for x in self._search_points_detailed if x is not None], device=d, dtype=dt)
            cell = p_range / denom
        else:
            cell = p_range * (self._explore_budget_points ** (-1.0 / p_dim))
        h = (cell * self.gs.jitter_scale).clamp_min(eps)
        I = torch.eye(p_dim, device=d, dtype=dt)
        damping = torch.tensor(self.gs.damping, device=d, dtype=dt)

        def _chol_pd_batch(C: torch.Tensor) -> torch.Tensor:
            E = int(C.shape[0])
            L, info = torch.linalg.cholesky_ex(C + damping * I[None, :, :])
            if bool(info.ne(0).any()):
                cnt = int(torch.sum(info.ne(0)).item())
                raise RuntimeError(
                    f"Cholesky failed for {cnt}/{E} elite covariances with configured damping regularization"
                )
            return L

        while budget > 0:
            m = min(self._param_batch_size, budget)
            budget -= m

            hist_s = self._hist_scores
            hist_p = self._hist_params

            hist_norm = (hist_p - p_min[None, :]) / p_range[None, :]

            score_min = torch.min(hist_s)
                                             
            weights = torch.exp(-(hist_s - score_min)).clamp_min(eps)
            idx = torch.multinomial(weights, num_samples=m, replacement=bool(int(hist_s.numel()) < m))
            base_scores = hist_s.index_select(0, idx)
            base = hist_p.index_select(0, idx)

            elite_cnt = max(1, int(float(m) ** self.gs.elite_power))
            order = torch.argsort(base_scores)
            elite_idx = order[:elite_cnt]
            non_idx = order[elite_cnt:]

            elites_s = base_scores.index_select(0, elite_idx)
            elites = base.index_select(0, elite_idx)
            non = base.index_select(0, non_idx) if int(non_idx.numel()) > 0 else base[:0]

            cand_non = (non + torch.randn_like(non) * h[None, :]).clamp(min=p_min, max=p_max)

            elites_norm = (elites - p_min[None, :]) / p_range[None, :]
            dists = torch.cdist(elites_norm, hist_norm)
            dists = torch.where(dists <= eps, torch.full_like(dists, float("inf")), dists)

            k_min = p_dim + 1
            if self.gs.neighbors_k < k_min:
                raise ValueError(
                    f"self.gs.neighbors_k must be >= (num_params + 1) = {k_min}; got {self.gs.neighbors_k}"
                )

            hist_n = int(hist_p.shape[0])
            if hist_n < k_min:
                raise ValueError(
                    f"History too small for kNN covariance: need at least {k_min} points, got {hist_n}. "
                    f"Increase self.gs.history_batches or initial exploration budget"
                )
            k = min(self.gs.neighbors_k, hist_n)

            nn_idx = torch.topk(dists, k=k, largest=False, sorted=False, dim=1).indices

            neigh = hist_p[nn_idx]
            neigh_s = hist_s[nn_idx]

            X = neigh - elites[:, None, :]
            XtX = torch.bmm(X.transpose(1, 2), X)
            cov = XtX / float(max(1, k - 1))
            L = _chol_pd_batch(cov)

            mu = elites
            if self.gs.drift_method == "ridge":
                y = (neigh_s - elites_s[:, None]).unsqueeze(-1)
                XtY = torch.bmm(X.transpose(1, 2), y)
                g, info = torch.linalg.solve_ex(cov + damping * I[None, :, :], XtY / float(max(1, k - 1)))
                if bool(info.ne(0).any()):
                    cnt = int(torch.sum(info.ne(0)).item())
                    raise RuntimeError(f"Ridge drift solve failed for {cnt}/{elite_cnt} elite neighborhoods")
                g = g.squeeze(-1)

                drift_dir = (-g).to(dtype=dt)
                step_raw = torch.bmm(cov, drift_dir[:, :, None]).squeeze(-1)
                quad = (drift_dir * step_raw).sum(dim=1)
                denom = torch.sqrt(quad.clamp_min(eps))
                step = torch.where(
                    (quad > eps)[:, None],
                    step_raw * (self.gs.drift_mahalanobis_step / denom)[:, None],
                    torch.zeros_like(step_raw),
                )
                mu = (elites + step).clamp(min=p_min, max=p_max)
            elif self.gs.drift_method == "cma":
                mu_n = max(1, int(np.floor(self.gs.cma_mu_frac * k)))

                top_pos = torch.topk(neigh_s, k=mu_n, largest=False, sorted=True, dim=1).indices
                top_p = torch.gather(neigh, 1, top_pos[:, :, None].expand(-1, -1, p_dim))

                rank = torch.arange(1, mu_n + 1, device=d, dtype=dt)
                w = torch.log(torch.as_tensor(mu_n + 0.5, device=d, dtype=dt)) - torch.log(rank)
                w = w / w.sum()

                recomb = (w[None, :, None] * top_p).sum(dim=1)
                delta = recomb - elites

                C = cov + damping * I[None, :, :]
                v, info = torch.linalg.solve_ex(C, delta[:, :, None])
                if bool(info.ne(0).any()):
                    cnt = int(torch.sum(info.ne(0)).item())
                    raise RuntimeError(f"CMA drift solve failed for {cnt}/{elite_cnt} elite neighborhoods")
                v = v.squeeze(-1)

                quad = (delta * v).sum(dim=1)
                mnorm = torch.sqrt(quad.clamp_min(eps))
                delta = torch.where(
                    (quad > eps)[:, None],
                    delta * (self.gs.drift_mahalanobis_step / mnorm)[:, None],
                    torch.zeros_like(delta),
                )
                mu = (elites + delta).clamp(min=p_min, max=p_max)

            z = torch.randn((elite_cnt, p_dim, 1), device=d, dtype=dt)
            noise = torch.bmm(L, z).squeeze(-1)
            cand_elite = (mu + noise).clamp(min=p_min, max=p_max)

            cand = torch.cat([cand_elite, cand_non], dim=0)
            scores = self._score_params(P=cand, data=data)
            self._history_merge(params=cand, scores=scores)

            v, idx0 = torch.min(scores, dim=0)
            if v.item() < best_s:
                best_s = float(v.item())
                best_p = cand[int(idx0.item())].detach().clone()

        for i, p in enumerate(self.params):
            p.value = float(best_p[i].item())
