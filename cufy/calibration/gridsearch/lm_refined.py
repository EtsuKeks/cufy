from collections.abc import Callable
from dataclasses import dataclass

import optuna.trial
import torch
from torch.func import jacfwd, vmap

import cufy.config as config
from cufy.calibration.gridsearch.base import GridSearchCalibrator, GridSearchConfig
from cufy.models.base import TorchParameterizedModel
from cufy.utils.implied_vol import implied_vol_newton
from cufy.utils.torch_utils import TorchPreparedBatch


@dataclass
class GridSearchLMConfig(GridSearchConfig):
    refine_multiplier: float
    lm_steps: int
    damping_init: float
    temperature: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.refine_multiplier <= 0.0:
            raise ValueError("refine_multiplier must be > 0")
        if self.lm_steps < 1:
            raise ValueError("lm_steps must be > 0")
        if self.damping_init <= 0.0:
            raise ValueError("damping_init must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


class GridSearchLMRefinedCalibrator(GridSearchCalibrator):
    cfg: GridSearchLMConfig

    def __init__(self, model: TorchParameterizedModel, cfg: GridSearchLMConfig):
        super().__init__(model, cfg=cfg)
        self._grad_checked: bool = False

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        self.cfg.damping_init = trial.suggest_float(
            "damping_init", self.cfg.damping_init * 1e-2, self.cfg.damping_init * 1e2, log=True
        )
        self.cfg.temperature = trial.suggest_float(
            "temperature", self.cfg.temperature * 1e-2, self.cfg.temperature * 1e2, log=True
        )

    def _ensure_prices_have_grad(self, *, data: TorchPreparedBatch, p_min: torch.Tensor, p_max: torch.Tensor) -> None:
        def dummy_price(p_flat: torch.Tensor) -> torch.Tensor:
            return self.model.prices_for_param_matrix(data=data.slice(1), param_matrix=p_flat[None, :])[0]

        try:
            with torch.enable_grad():
                J = jacfwd(dummy_price)(0.5 * (p_min + p_max))
        except Exception as e:
            raise ValueError(f"prices_for_param_matrix does not support forward-mode AD (jacfwd): {e}") from e

        if not torch.isfinite(J).all().item():  # type: ignore[arg-type]
            raise ValueError("prices_for_param_matrix returns non-finite Jacobians via jacfwd")

    def _eval_err(
        self, *, P: torch.Tensor, data: TorchPreparedBatch, close_IV_t: torch.Tensor, w_sqrt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = self.model.prices_for_param_matrix(data=data, param_matrix=P)
        iv_model = implied_vol_newton(price=preds, data=data, sigma_init=close_IV_t)
        err = w_sqrt[None, :] * (iv_model - close_IV_t[None, :])
        E = err.square().sum(dim=-1)
        return iv_model, err, E

    def _run_lm(
        self, *, P0: torch.Tensor, data: TorchPreparedBatch, p_min: torch.Tensor, p_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = config.device
        dt = config.dtype
        B = int(P0.shape[0])

        p_cur = P0.to(device=d, dtype=dt)
        lam = torch.full((B,), self.cfg.damping_init, device=d, dtype=dt)
        nu = torch.full((B,), 2.0, device=d, dtype=dt)
        close_IV_t = data.close_IV_t

        if bool((data.w_t < 0.0).any()):
            raise ValueError("All weights in data.w_t must be non-negative for Levenberg-Marquardt calibration")
        w_sqrt = torch.sqrt(data.w_t)

        with torch.no_grad():
            iv_cur, err_cur, E_cur = self._eval_err(P=p_cur, data=data, close_IV_t=close_IV_t, w_sqrt=w_sqrt)

        def err_fn_single(p_flat: torch.Tensor) -> torch.Tensor:
            preds = self.model.prices_for_param_matrix(data=data, param_matrix=p_flat[None, :])[0]
            iv = implied_vol_newton(price=preds[None, :], data=data, sigma_init=close_IV_t)[0]
            return w_sqrt * (iv - close_IV_t)

        for _ in range(self.cfg.lm_steps):
            with torch.enable_grad():
                J = vmap(jacfwd(err_fn_single))(p_cur)

            JtJ = J.mT @ J
            Jte = J.mT @ err_cur.unsqueeze(-1)

            diag_JtJ = torch.diagonal(JtJ, dim1=-2, dim2=-1).clamp_min(config.eps)
            D_inv = torch.rsqrt(diag_JtJ)
            A_scaled = torch.einsum("bi,bij,bj->bij", D_inv, JtJ, D_inv) + torch.diag_embed(lam)
            Jte_scaled = Jte * D_inv.unsqueeze(-1)
            L, info = torch.linalg.cholesky_ex(A_scaled)
            dP_scaled = -torch.cholesky_solve(Jte_scaled, L).squeeze(-1)
            dP = torch.where((info == 0).unsqueeze(-1), dP_scaled * D_inv, 0.0)

            p_prop = (p_cur + dP).clamp(min=p_min, max=p_max)
            with torch.no_grad():
                iv_prop, err_prop, E_prop = self._eval_err(P=p_prop, data=data, close_IV_t=close_IV_t, w_sqrt=w_sqrt)

            dP_actual = p_prop - p_cur
            JtJ_dP = (JtJ @ dP_actual.unsqueeze(-1)).squeeze(-1)
            dL = -torch.einsum("bd,bd->b", dP_actual, Jte.squeeze(-1) + 0.5 * JtJ_dP)

            dF = 0.5 * (E_cur - E_prop)
            rho = dF / dL.clamp_min(config.eps)
            accept = rho > 0
            acc = accept.unsqueeze(-1)

            p_cur = torch.where(acc, p_prop, p_cur)
            iv_cur = torch.where(acc, iv_prop, iv_cur)
            err_cur = torch.where(acc, err_prop, err_cur)
            E_cur = torch.where(accept, E_prop, E_cur)

            lam_factor_success = (1.0 - (2.0 * rho - 1.0) ** 3).clamp_min(1.0 / 3.0)
            lam = torch.where(accept, lam * lam_factor_success, lam * nu)
            nu = torch.where(accept, 2.0, nu * 2.0)

        return p_cur, torch.nan_to_num(torch.sqrt(E_cur), nan=1e9, posinf=1e9)

    def probe(
        self, *, data: TorchPreparedBatch, pmin: torch.Tensor, pmax: torch.Tensor
    ) -> Callable[[int, int], Callable[[], None]]:
        d = config.device
        dt = config.dtype
        p_dim = int(pmin.numel())

        def make_f(m: int, n: int):
            def _run():
                P0 = torch.lerp(pmin, pmax, torch.rand((m, p_dim), device=d, dtype=dt))
                _, _ = self._run_lm(P0=P0, data=data.slice(n), p_min=pmin, p_max=pmax)

            return _run

        return make_f

    def refine(self, *, p_min: torch.Tensor, p_max: torch.Tensor, data: TorchPreparedBatch) -> None:
        if not self._grad_checked:
            self._ensure_prices_have_grad(data=data, p_min=p_min, p_max=p_max)
            self._grad_checked = True
        budget = int(self._checked_search_points * self.cfg.refine_multiplier)
        if budget < 1:
            return

        eps = config.eps
        while budget > 0:
            m = min(self._checked_param_batch_size, budget)
            budget -= m

            hist_p = self._hist_params
            hist_s = self._hist_scores

            score_min = torch.min(self._hist_scores)
            weights = torch.exp((score_min - hist_s) / self.cfg.temperature).clamp_min(eps)
            if int(hist_s.shape[0]) < m:
                raise ValueError(
                    f"History size ({int(hist_s.shape[0])}) is smaller than batch size ({m}). "
                    f"Levenberg-Marquardt is deterministic and cannot process duplicate points in a batch. "
                    f"Consider increasing cfg.history_points_fraction or initial exploration budget"
                )

            idx = torch.multinomial(weights, num_samples=m, replacement=False)
            base = hist_p[idx]

            cand, scores = self._run_lm(P0=base, data=data, p_min=p_min, p_max=p_max)
            self._history_merge(params=cand, scores=scores)

        best_p = self._hist_params[torch.argmin(self._hist_scores)]
        for i, p in enumerate(self.model.params):
            p.value = float(best_p[i].item())
