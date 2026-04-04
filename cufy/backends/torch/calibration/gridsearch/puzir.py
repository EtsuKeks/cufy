from dataclasses import dataclass
from typing import Literal

import numpy as np
import optuna.trial
import torch

import cufy.backends.torch.config as config
from cufy.backends.torch.calibration.gridsearch.base import GridSearchCalibrator, GridSearchConfig
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.backends.torch.utils.torch_utils import TorchPreparedBatch


@dataclass
class PuzirConfig(GridSearchConfig):
    refine_multiplier: float
    drift_method: Literal["none", "ridge", "cma"]
    drift_mahalanobis_step: float | None
    cma_mu_frac: float | None
    neighbors_k: int
    elite_power: float
    jitter_scale: float
    damping: float
    temperature: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.refine_multiplier <= 0.0:
            raise ValueError("refine_multiplier must be > 0")
        if self.drift_method != "none":
            if self.drift_mahalanobis_step is None:
                raise ValueError("drift_mahalanobis_step must be set when drift_method != 'none'")
            if self.drift_mahalanobis_step <= 0.0:
                raise ValueError("drift_mahalanobis_step must be > 0")
        if self.drift_method == "cma":
            if self.cma_mu_frac is None:
                raise ValueError("cma_mu_frac must be set when drift_method == 'cma'")
            if self.cma_mu_frac <= 0.0 or self.cma_mu_frac > 1.0:
                raise ValueError("cma_mu_frac must be in (0.0, 1.0]")
        if self.neighbors_k < 1:
            raise ValueError("neighbors_k must be >= 1")
        if not (0.0 < self.elite_power <= 1.0):
            raise ValueError("elite_power must be in (0.0, 1.0]")
        if self.jitter_scale < 0.0:
            raise ValueError("jitter_scale must be >= 0")
        if self.damping < 0.0:
            raise ValueError("damping must be >= 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


class PuzirCalibrator(GridSearchCalibrator):
    cfg: PuzirConfig

    def __init__(self, model: TorchParameterizedModel, cfg: PuzirConfig):
        super().__init__(model, cfg=cfg)
        k_min = len(model.params) + 1
        if self.cfg.neighbors_k < k_min:
            raise ValueError(f"cfg.neighbors_k must be >= (num_params + 1) = {k_min}; got {self.cfg.neighbors_k}")

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        if self.cfg.damping > 0.0:
            self.cfg.damping = trial.suggest_float("damping", self.cfg.damping * 1e-2, self.cfg.damping * 1e2, log=True)
        self.cfg.temperature = trial.suggest_float(
            "temperature", self.cfg.temperature * 1e-2, self.cfg.temperature * 1e2, log=True
        )

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        k_min = len(self.model.params) + 1
        hist_size = int(self._hist_params.shape[0])
        if hist_size < k_min:
            raise ValueError(
                f"Configured history size ({hist_size}) is too small for kNN covariance. "
                f"Need at least {k_min} points. Increase cfg.history_points_fraction or initial exploration budget."
            )

        cfg = self.cfg
        budget = int(self._checked_search_points * cfg.refine_multiplier)
        if budget < 1:
            return

        d = config.device
        dt = config.dtype
        eps = config.eps

        best_p = self._hist_params[torch.argmin(self._hist_scores)]
        p_dim = len(self.model.params)

        p_range = (p_max - p_min).clamp_min(eps)
        if cfg.initial_sampler == "grid":
            cell = p_range / torch.tensor([max(x - 1, 1) for x in self._search_points_detailed], device=d, dtype=dt)
        else:
            cell = p_range * (self._checked_search_points ** (-1.0 / p_dim))
        h = (cell * cfg.jitter_scale).clamp_min(eps)

        def _chol_pd_batch(C: torch.Tensor) -> torch.Tensor:
            E = int(C.shape[0])
            L, info = torch.linalg.cholesky_ex(C)
            if bool(info.ne(0).any()):
                cnt = int(torch.sum(info.ne(0)).item())
                raise optuna.exceptions.TrialPruned(
                    f"Cholesky failed for {cnt}/{E} elite covariances with configured damping regularization"
                )
            return L

        while budget > 0:
            m = min(self._checked_param_batch_size, budget)
            budget -= m

            hist_s = self._hist_scores
            hist_p = self._hist_params

            score_min = torch.min(self._hist_scores)
            weights = torch.exp(-(hist_s - score_min) / cfg.temperature).clamp_min(eps)
            idx = torch.multinomial(weights, num_samples=m, replacement=True)
            base_scores = hist_s.index_select(0, idx)
            base = hist_p.index_select(0, idx)

            elite_cnt = max(1, int(float(m) ** cfg.elite_power))
            order = torch.argsort(base_scores)
            elite_idx = order[:elite_cnt]
            non_idx = order[elite_cnt:]

            elites_s = base_scores.index_select(0, elite_idx)
            elites = base.index_select(0, elite_idx)
            non = base.index_select(0, non_idx)

            cand_non = (non + torch.randn_like(non) * h[None, :]).clamp(min=p_min, max=p_max)

            elites_norm = (elites - p_min[None, :]) / p_range[None, :]
            hist_norm = (hist_p - p_min[None, :]) / p_range[None, :]
            dists = torch.cdist(elites_norm, hist_norm)
            dists.masked_fill_(dists <= eps, float("inf"))

            k = min(cfg.neighbors_k, int(hist_p.shape[0]))

            nn_idx = torch.topk(dists, k=k, largest=False, sorted=False, dim=1).indices

            neigh = hist_p[nn_idx]
            neigh_s = hist_s[nn_idx]

            X = neigh - elites[:, None, :]
            XtX = torch.bmm(X.transpose(1, 2), X)
            cov = XtX / float(max(1, k - 1))
            if cfg.damping > 0.0:
                diag_reg = (cfg.damping * torch.diagonal(cov, dim1=-2, dim2=-1)).clamp_min(eps)
                cov_reg = cov + torch.diag_embed(diag_reg)
            else:
                cov_reg = cov
            L = _chol_pd_batch(cov_reg)

            mu = elites
            step_raw: torch.Tensor | None = None
            drift_dir: torch.Tensor | None = None
            if cfg.drift_method == "ridge":
                y = (neigh_s - elites_s[:, None]).unsqueeze(-1)
                XtY = torch.bmm(X.transpose(1, 2), y)
                g = torch.cholesky_solve(XtY / float(max(1, k - 1)), L).squeeze(-1)

                drift_dir = -g
                step_raw = torch.bmm(cov_reg, drift_dir[:, :, None]).squeeze(-1)
            elif cfg.drift_method == "cma":
                mu_n = max(1, int(np.floor(cfg.cma_mu_frac * k)))

                top_pos = torch.topk(neigh_s, k=mu_n, largest=False, sorted=True, dim=1).indices
                top_p = torch.gather(neigh, 1, top_pos[:, :, None].expand(-1, -1, p_dim))

                rank = torch.arange(1, mu_n + 1, device=d, dtype=dt)
                w = torch.log(torch.as_tensor(mu_n + 0.5, device=d, dtype=dt)) - torch.log(rank)
                w = w / w.sum()

                step_raw = (w[None, :, None] * top_p).sum(dim=1) - elites
                drift_dir = torch.cholesky_solve(step_raw.unsqueeze(-1), L).squeeze(-1)

            if cfg.drift_method in ("ridge", "cma") and step_raw is not None and drift_dir is not None:
                quad = (step_raw * drift_dir).sum(dim=1)
                denom = torch.sqrt(quad.clamp_min(eps))
                step = torch.where(
                    (quad > eps)[:, None],
                    step_raw * (cfg.drift_mahalanobis_step / denom)[:, None],  # type: ignore[operator]
                    torch.zeros_like(step_raw),
                )
                mu = (elites + step).clamp(min=p_min, max=p_max)

            z = torch.randn((elite_cnt, p_dim, 1), device=d, dtype=dt)
            noise = torch.bmm(L, z).squeeze(-1)
            cand_elite = (mu + noise).clamp(min=p_min, max=p_max)

            cand = torch.cat([cand_elite, cand_non], dim=0)
            scores = self._score_params(P=cand, data=data)
            self._history_merge(params=cand, scores=scores)

        best_p = self._hist_params[torch.argmin(self._hist_scores)]
        for i, p in enumerate(self.model.params):
            p.value = float(best_p[i].item())
