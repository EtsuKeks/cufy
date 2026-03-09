from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
from pydantic import BaseModel, PrivateAttr

from src.abc.model import Model
from src.utils.torch_utils import Batch1D


class ModelParam(BaseModel):
    _name: str | None = PrivateAttr(default=None)
    position: int
    value: float | None = None
    min_value: float
    max_value: float


class ParameterizedModel(Model, ABC):
    def __init__(self, param_overrides: dict[str, ModelParam] | None = None):
        self.params = self._collect_params(param_overrides)
        self._saved_params: dict[str, dict[str, float]] = {}
        self._saved_facts: dict[str, dict[str, str]] = {}

    @abstractmethod
    def prices_for_param_matrix(self, *, data: Batch1D, param_matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _collect_params(self, param_overrides: dict[str, ModelParam] | None) -> list[ModelParam]:
        found: dict[str, ModelParam] = {}
        for cls in reversed(self.__class__.mro()):
            if cls in (ParameterizedModel, Model, ABC, object):
                continue
            found.update({k: v for k, v in vars(cls).items() if isinstance(v, ModelParam)})

        overrides = param_overrides or {}
        unknown = sorted(set(overrides.keys()) - set(found.keys()))
        if unknown:
            raise ValueError(f"Unknown parameter overrides: {unknown}. Known parameters: {sorted(found.keys())}")

        bound: dict[str, ModelParam] = {}
        for name, default in found.items():
            p = (overrides[name] if name in overrides else default).model_copy(deep=True)
            p._name = name
            bound[name] = p
            setattr(self, name, p)

        params = sorted(bound.values(), key=lambda p: p.position)
        if not params:
            raise ValueError("No parameter fields found")

        positions: dict[int, ModelParam] = {}
        for p in params:
            if p.position in positions:
                raise ValueError(
                    f"Duplicate parameter.position values. "
                    f"Got {p._name} with {p.position} "
                    f"when {positions[p.position]._name} with {positions[p.position].position} was already defined"
                )
            positions[p.position] = p
        return params

    def save_params(self, label: str) -> None:
        self._saved_params[label] = {p._name: np.nan if p.value is None else p.value for p in self.params}

    def get_params(self) -> dict[str, dict[str, float]]:
        return copy.deepcopy(self._saved_params)

    def save_fact(self, label: str, key: str, value: str) -> None:
        self._saved_facts.setdefault(label, {})[key] = str(value)

    def get_facts(self) -> dict[str, dict[str, str]]:
        return copy.deepcopy(self._saved_facts)
