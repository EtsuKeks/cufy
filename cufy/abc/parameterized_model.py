from abc import ABC
from collections.abc import Mapping
from typing import Any, overload

import numpy as np
from pydantic import BaseModel, PrivateAttr

from cufy.abc.model import Model
from cufy.abc.telemetry_model import TelemetryModel


class ModelParam(BaseModel):
    _name: str = PrivateAttr(default="")
    position: int
    value: float | None = None
    min_value: float
    max_value: float

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    # Descriptor protocol methods
    @overload
    def __get__(self, obj: None, objtype: Any) -> ModelParam: ...

    @overload
    def __get__(self, obj: ParameterizedModel, objtype: Any) -> float | None: ...

    def __get__(self, obj: Any, objtype: Any) -> Any:
        if obj is None:
            # Accessed on the class (e.g. SABR.alpha), return the ModelParam instance
            return self
        # Accessed on an instance (e.g. self.alpha), return the float value
        return self.value

    def __set__(self, obj: Any, value: float | None) -> None:
        if obj is None:
            raise AttributeError("Cannot set parameter value on class, only on instance")
        if value is not None:
            value = float(value)
            if value < self.min_value or value > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}' value {value} is out of bounds [{self.min_value}, {self.max_value}]"
                )
        self.value = value


class ParameterizedModel(TelemetryModel, ABC):
    def __init__(self, param_overrides: Mapping[str, ModelParam] | None = None) -> None:
        super().__init__()
        self.params = self._collect_params(param_overrides)

    def _collect_params(self, param_overrides: Mapping[str, ModelParam] | None) -> list[ModelParam]:
        found: dict[str, ModelParam] = {}
        for cls in reversed(self.__class__.mro()):
            if cls in (ParameterizedModel, TelemetryModel, Model, ABC, object):
                continue
            found.update({k: v for k, v in vars(cls).items() if isinstance(v, ModelParam)})

        overrides = param_overrides or {}
        unknown = sorted(set(overrides.keys()) - set(found.keys()))
        if unknown:
            raise ValueError(f"Unknown parameter overrides: {unknown}. Known parameters: {sorted(found.keys())}")

        bound: dict[str, ModelParam] = {}
        for name, default in found.items():
            p = (overrides[name] if name in overrides else default).model_copy(deep=True)
            p.name = name
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
                    f"Got {p.name} with {p.position} "
                    f"when {positions[p.position].name} with {positions[p.position].position} was already defined"
                )
            positions[p.position] = p
        return params

    def get_numeric(self) -> dict[str, float]:
        param_state = {p.name: np.nan if p.value is None else p.value for p in self.params}
        return {**param_state, **self._numeric}

    def is_initialized(self) -> bool:
        """
        Check if the model has been initialized (i.e. all parameters have values).
        """
        return all(p.value is not None for p in self.params)
