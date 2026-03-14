from abc import ABC
from typing import Callable

import torch
from torch.func import jacfwd, vmap
from pydantic import Field

import cufy.config as config
from cufy.backends.torch.calibration.gridsearch.base import GridSearchModel, GridSearchModelParam, GridSearchSettings
from cufy.backends.torch.utils.torch_utils import Batch1D
from cufy.backends.torch.utils.implied_vol import implied_vol_newton, proxy_dsigma_dprice


class GridSearchLMSettings(GridSearchSettings):
    refine_multiplier: float = Field(default=1.0, gt=0.0)
    lm_steps: int = Field(default=3, gt=0)
    damping_init: float = Field(default=1e-2, gt=0.0)


class GridSearchLMRefinedModel(GridSearchModel, ABC):
    gs: GridSearchLMSettings                                        

    def __init__(
        self, param_overrides: dict[str, GridSearchModelParam] | None = None, gs: GridSearchLMSettings | None = None
    ):
        super().__init__(param_overrides, gs=gs or GridSearchLMSettings())

    def _ensure_prices_has_grad(self, *, data: Batch1D, p_min: torch.Tensor, p_max: torch.Tensor) -> None:
        def dummy_price(p_flat: torch.Tensor) -> torch.Tensor:
            return self.prices_for_param_matrix(data=data.slice(1), param_matrix=p_flat[None, :])[0]

        try:
            with torch.enable_grad():
                J = jacfwd(dummy_price)((0.5 * (p_min + p_max)).detach().clone())
        except Exception as e:
            raise ValueError(f"prices_for_param_matrix does not support forward-mode AD (jacfwd): {e}") from e

        if not torch.isfinite(J).all().item():  # type: ignore[arg-type]
            raise ValueError("prices_for_param_matrix returns non-finite Jacobians via jacfwd")

    def _eval_err(
        self, *, P: torch.Tensor, data: Batch1D, close_IV_t: torch.Tensor, w_sqrt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.prices_for_param_matrix(data=data, param_matrix=P)
        iv_model = implied_vol_newton(price=preds, data=data, sigma_init=close_IV_t)
        err = w_sqrt[None, :] * (iv_model - close_IV_t[None, :])
        E = (err * err).sum(dim=-1)
        return iv_model, err, E

    def _run_lm(self, *, P0: torch.Tensor, data: Batch1D, p_min: torch.Tensor, p_max: torch.Tensor) -> torch.Tensor:
        d = config.device
        dt = config.dtype
        B, p_dim = P0.shape

        p_cur = P0.detach().clone().to(device=d, dtype=dt)
        lam = torch.full((B,), self.gs.damping_init, device=d, dtype=dt)
        close_IV_t = data.close_IV_t
        w_sqrt = torch.sqrt(data.w_t)
        eye_p = torch.eye(p_dim, device=d, dtype=dt)

        with torch.no_grad():
            iv_cur, err_cur, E_cur = self._eval_err(P=p_cur, data=data, close_IV_t=close_IV_t, w_sqrt=w_sqrt)

        for _ in range(self.gs.lm_steps):
            with torch.no_grad():
                didc = proxy_dsigma_dprice(data=data, sigma=iv_cur)

            def price_fn_single(p_flat: torch.Tensor) -> torch.Tensor:
                return self.prices_for_param_matrix(data=data, param_matrix=p_flat[None, :])[0]

            with torch.enable_grad():
                J_price = vmap(jacfwd(price_fn_single))(p_cur)

            J = (w_sqrt[None, :, None] * didc[:, :, None]) * J_price
            JtJ = torch.bmm(J.mT, J)
            Jte = torch.bmm(J.mT, err_cur.unsqueeze(-1))
            lam_I = lam[:, None, None] * eye_p.unsqueeze(0)

            L, info = torch.linalg.cholesky_ex(JtJ + lam_I)
            dP = torch.linalg.cholesky_solve(-Jte, L).squeeze(-1)  # type: ignore[attr-defined]
            dP = torch.where((info == 0).unsqueeze(-1), dP, torch.zeros_like(dP))

            p_prop = (p_cur + dP).clamp(min=p_min, max=p_max)

            with torch.no_grad():
                iv_prop, err_prop, E_prop = self._eval_err(
                    P=p_prop, data=data, close_IV_t=close_IV_t, w_sqrt=w_sqrt
                )

            accept = E_prop < E_cur
            acc = accept.unsqueeze(-1)
            p_cur = torch.where(acc, p_prop, p_cur)
            iv_cur = torch.where(acc, iv_prop, iv_cur)
            err_cur = torch.where(acc, err_prop, err_cur)
            E_cur = torch.where(accept, E_prop, E_cur)
            lam = torch.where(accept, lam / 10.0, lam * 10.0)

        return p_cur

    def probe(
        self, *, data: Batch1D, pmin: torch.Tensor, pmax: torch.Tensor
    ) -> Callable[[int, int], Callable[[], None]]:
        d = config.device
        dt = config.dtype
        p_dim = int(pmin.numel())

        def make_f(m: int, n: int):
            def _run():
                P0 = pmin[None, :] + (pmax - pmin)[None, :] * torch.rand((m, p_dim), device=d, dtype=dt)
                _ = self._run_lm(P0=P0, data=data.slice(n), p_min=pmin, p_max=pmax)
            return _run

        return make_f

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: Batch1D) -> None:
        self._ensure_prices_has_grad(data=data, p_min=p_min, p_max=p_max)
        budget = int(self._explore_budget_points * self.gs.refine_multiplier)
        if budget < 1:
            return

        d = config.device
        dt = config.dtype
        eps = torch.tensor(config.eps, device=d, dtype=dt)

        best_idx = torch.argmin(self._hist_scores)
        best_s = float(self._hist_scores[best_idx].item())
        best_p = self._hist_params[best_idx].detach().clone()

        self.save_numeric("refined_batch_capacity", float(self._param_batch_size))

        if self.gs.initial_sampler == "sobol":
            self.save_numeric("refined_budget_batches", float(self.gs.refine_multiplier))
        else:
            self.save_string("refined_budget_points", str(self._search_points_detailed))

        while budget > 0:
            m = min(self._param_batch_size, budget)
            budget -= m

            hist_p = self._hist_params
            hist_s = self._hist_scores

            score_min = torch.min(hist_s)
            weights = torch.exp(-(hist_s - score_min)).clamp_min(eps)
            idx = torch.multinomial(weights, num_samples=m, replacement=bool(int(hist_s.numel()) < m))
            base = hist_p.index_select(0, idx)

            cand = self._run_lm(P0=base, data=data, p_min=p_min, p_max=p_max)
            scores = self._score_params(P=cand, data=data)
            self._history_merge(params=cand, scores=scores)

            v, idx0 = torch.min(scores, dim=0)
            if v.item() < best_s:
                best_s = float(v.item())
                best_p = cand[int(idx0.item())].detach().clone()

        for i, p in enumerate(self.params):
            p.value = float(best_p[i].item())
