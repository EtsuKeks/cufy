from abc import ABC
from typing import Literal, Mapping
import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

import cufy.config as config
from cufy.abc.data import OptionBatch
from cufy.abc.parameterized_model import ModelParam
from cufy.backends.torch.models.base import TorchParameterizedModel
from cufy.abc.telemetry_model import clears_telemetry
from cufy.backends.torch.utils.implied_vol import implied_vol_newton, bs_vega
from cufy.backends.torch.utils.torch_utils import Batch1D


class GradientModelParam(ModelParam):
    pass


class GradientModelSettings(BaseModel):
    class LMSettings(BaseModel):
        steps_initial: int = Field(default=100, ge=0)
        steps_calibrate: int = Field(default=100, ge=0)
        damping_init: float = Field(default=1e-2, gt=0.0)

    class LBFGSSettings(BaseModel):
        max_iter_initial: int = Field(default=100, ge=0)
        max_iter_calibrate: int = Field(default=100, ge=0)
        lr: float = Field(default=1.0, gt=0.0)
        history_size: int = Field(default=100, ge=1)
        line_search_fn: Literal["strong_wolfe"] | None = "strong_wolfe"
        tolerance_grad: float = Field(default=config.eps, gt=0.0)
        tolerance_change: float = Field(default=config.eps, gt=0.0)

    lm: LMSettings | None = None
    lbfgs: LBFGSSettings | None = None

    @model_validator(mode="after")
    def _validate_optimizer_settings(self) -> "GradientModelSettings":
        if self.lm is not None and self.lbfgs is not None:
            raise ValueError("Specify only one of `lm` or `lbfgs` (they are mutually exclusive).")
        if self.lm is None and self.lbfgs is None:
            self.lm = GradientModelSettings.LMSettings()
        return self


class GradientModel(TorchParameterizedModel, ABC):
    def __init__(
        self,
        param_overrides: Mapping[str, GradientModelParam] | None = None,
        grad: GradientModelSettings | None = None,
    ):
        if param_overrides is None:
            raise ValueError(
                "GradientModel requires an explicit initial guess: pass param_overrides with `value` set for EVERY "
                "parameter (no midpoint / fallback initialization)."
            )
        super().__init__(param_overrides=param_overrides)
        self.grad = grad or GradientModelSettings()
        self._prices_has_grad: bool | None = None
        missing = [p.position for p in self.params if p.value is None]
        if missing:
            raise ValueError(
                "GradientModel requires `value` to be set for all parameters in the initial guess. "
                f"Missing positions={missing}"
            )

    def _ensure_prices_has_grad(self, *, data: Batch1D) -> None:
        if self._prices_has_grad is not None:
            return
        try:
            P = torch.tensor(
                [[float(p.value) for p in self.params if p.value is not None]],
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
            preds = self.prices_for_param_matrix(data=data, param_matrix=P)
            g = torch.autograd.grad(preds.sum(), P, allow_unused=True)[0]
            self._prices_has_grad = bool(g is not None and torch.isfinite(g).all().item())  # type: ignore[union-attr,unnecessary-comparison]
        except Exception:
            self._prices_has_grad = False

    def _optimize_lm_theseus(
        self, *, steps: int, start: torch.Tensor, data: Batch1D, p_min: torch.Tensor, p_max: torch.Tensor
    ) -> torch.Tensor:
        import theseus as th  # type: ignore[import-not-found]
        from theseus.core.cost_function import CostFunction  # type: ignore[import-not-found]

        if steps < 1:
            return start.detach()
        self._ensure_prices_has_grad(data=data)
        if self._prices_has_grad is False:
            raise RuntimeError("No gradients through prices_for_param_matrix; cannot use GradientModel")

        market_iv = data.close_IV_t
        w_sqrt = torch.sqrt(data.w_t)

        p_dim = int(start.shape[1])
        params = th.Vector(p_dim, name="params")

        class _ExactIVResidualCost(CostFunction):
            def __init__(self, *, params_var: th.Vector, name: str | None = None):
                super().__init__(th.ScaleCostWeight(1.0), name=name)
                self.register_vars([params_var], is_optim_vars=True)
                self._dim = int(data.F_t.numel())
                self._cache_p: torch.Tensor | None = None
                self._cache_err: torch.Tensor | None = None
                self._cache_didc: torch.Tensor | None = None

            def dim(self) -> int:
                return self._dim

            def _refresh_cache(self, *, Pm: torch.Tensor) -> None:
                if self._cache_p is not None and torch.equal(Pm, self._cache_p):
                    return

                with torch.no_grad():
                    preds = self_outer.prices_for_param_matrix(data=data, param_matrix=Pm)
                    iv_model = implied_vol_newton(price=preds, data=data, sigma_init=market_iv[None, :])
                    err = w_sqrt[None, :] * (iv_model - market_iv[None, :])
                    eps = float(config.eps)
                    didc = 1.0 / bs_vega(data=data, sigma=iv_model).clamp_min(eps)

                self._cache_p = Pm.detach().clone()
                self._cache_err = err
                self._cache_didc = didc

            def error(self) -> torch.Tensor:
                Pm = self.optim_vars[0].tensor
                self._refresh_cache(Pm=Pm)
                assert self._cache_err is not None
                return self._cache_err

            def jacobians(self):
                Pm_full = self.optim_vars[0].tensor
                self._refresh_cache(Pm=Pm_full)
                assert self._cache_err is not None and self._cache_didc is not None

                with torch.enable_grad():
                    p0 = Pm_full.reshape(-1).detach().requires_grad_(True)
                    def price_fn(p_flat: torch.Tensor) -> torch.Tensor:
                        return self_outer.prices_for_param_matrix(data=data, param_matrix=p_flat[None, :])

                    J_price = torch.autograd.functional.jacobian(                          
                        price_fn,
                        p0,
                        create_graph=False,
                        strict=False,
                        vectorize=True,
                        strategy="forward-mode",
                    )

                J = (w_sqrt[None, :, None] * self._cache_didc[:, :, None]) * J_price
                return [J], self._cache_err

            def _copy_impl(self, new_name: str | None = None) -> "_ExactIVResidualCost":
                pv = self.optim_vars[0].copy()
                return _ExactIVResidualCost(params_var=pv, name=new_name)

        self_outer = self
        objective = th.Objective()
        cost = _ExactIVResidualCost(params_var=params)
        objective.add(cost)

        linear_solver_cls = getattr(th, "CholeskyDenseSolver", None)
        opt = (
            th.LevenbergMarquardt(
                objective, linear_solver_cls=linear_solver_cls, max_iterations=int(steps), step_size=1.0, vectorize=True
            )
            if linear_solver_cls is not None
            else th.LevenbergMarquardt(objective, max_iterations=int(steps), step_size=1.0, vectorize=True)
        )
        assert self.grad.lm is not None
        opt.reset(damping=float(self.grad.lm.damping_init), adaptive_damping=True)
        layer = th.TheseusLayer(opt)

        P0 = start.detach().clone().clamp(min=p_min, max=p_max)
        sol, _info = layer.forward(input_tensors={"params": P0})
        return sol["params"].clamp(min=p_min, max=p_max)

    def _optimize_lbfgs(
        self, *, max_iter: int, start: torch.Tensor, data: Batch1D, p_min: torch.Tensor, p_max: torch.Tensor
    ) -> torch.Tensor:
        if max_iter < 1:
            return start.detach()
        self._ensure_prices_has_grad(data=data)
        if self._prices_has_grad is False:
            raise RuntimeError("No gradients through prices_for_param_matrix; cannot use GradientModel")

        assert self.grad.lbfgs is not None
        cfg = self.grad.lbfgs

        market_iv = data.close_IV_t
        w_sqrt = torch.sqrt(data.w_t)

        P = start.detach().clone().requires_grad_(True)
        opt = torch.optim.LBFGS(
            [P],
            lr=float(cfg.lr),
            max_iter=int(max_iter),
            history_size=int(cfg.history_size),
            line_search_fn=cfg.line_search_fn,
            tolerance_grad=float(cfg.tolerance_grad),
            tolerance_change=float(cfg.tolerance_change),
        )

        def closure() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                P.clamp_(min=p_min, max=p_max)

            preds = self.prices_for_param_matrix(data=data, param_matrix=P)
            iv_model = implied_vol_newton(price=preds, data=data, sigma_init=market_iv[None, :])
            resid = w_sqrt[None, :] * (iv_model - market_iv[None, :])
            loss = 0.5 * (resid * resid).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            P.clamp_(min=p_min, max=p_max)
        return P.detach()

    def _optimize(
        self, *, steps: int, start: torch.Tensor, data: Batch1D, p_min: torch.Tensor, p_max: torch.Tensor
    ) -> torch.Tensor:
        if self.grad.lbfgs is None:
            return self._optimize_lm_theseus(steps=steps, start=start, data=data, p_min=p_min, p_max=p_max)
        return self._optimize_lbfgs(max_iter=steps, start=start, data=data, p_min=p_min, p_max=p_max)

    @clears_telemetry
    def find_initial_params(self, batch: OptionBatch) -> None:
        data = self._batch_to_1d(batch)
        p_min = torch.tensor([p.min_value for p in self.params], device=config.device, dtype=config.dtype)
        p_max = torch.tensor([p.max_value for p in self.params], device=config.device, dtype=config.dtype)
        if any(p.value is None for p in self.params):
            raise RuntimeError(
                "GradientModel requires an explicit initial guess: initialize the model with param_overrides "
                "and set `value` for all parameters"
            )
        start = torch.tensor(
            [[float(p.value) for p in self.params if p.value is not None]], device=config.device, dtype=config.dtype
        ).clamp(min=p_min, max=p_max)

        if self.grad.lbfgs is not None:
            steps = int(self.grad.lbfgs.max_iter_initial)
        else:
            assert self.grad.lm is not None
            steps = int(self.grad.lm.steps_initial)
        P = self._optimize(steps=steps, start=start, data=data, p_min=p_min, p_max=p_max)[0]
        for i, p in enumerate(self.params):
            p.value = float(P[i].item())
        self.save_string("stage", "post_fit_initial")

    @clears_telemetry
    def calibrate(self, batch: OptionBatch) -> None:
        if any(p.value is None for p in self.params):
            raise RuntimeError("calibrate() called before find_initial_params()")
        data = self._batch_to_1d(batch)
        p_min = torch.tensor([p.min_value for p in self.params], device=config.device, dtype=config.dtype)
        p_max = torch.tensor([p.max_value for p in self.params], device=config.device, dtype=config.dtype)
        cur = torch.tensor([[float(p.value) for p in self.params if p.value is not None]], device=config.device, dtype=config.dtype)
        start = cur.clamp(min=p_min, max=p_max)

        if self.grad.lbfgs is not None:
            steps = int(self.grad.lbfgs.max_iter_calibrate)
        else:
            assert self.grad.lm is not None
            steps = int(self.grad.lm.steps_calibrate)
        P = self._optimize(steps=steps, start=start, data=data, p_min=p_min, p_max=p_max)[0]
        for i, p in enumerate(self.params):
            p.value = float(P[i].item())
        self.save_string("stage", "post_fit_calibrate")

    def price(self, batch: OptionBatch) -> np.ndarray:
        if any(p.value is None for p in self.params):
            raise RuntimeError("price() called before find_initial_params()")
        P = torch.tensor([[p.value for p in self.params]], device=config.device, dtype=config.dtype)
        data = Batch1D.from_option_batch(batch)
        with torch.no_grad():
            out = self.prices_for_param_matrix(data=data, param_matrix=P)[0].detach().cpu().numpy()
        return out
