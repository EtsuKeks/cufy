import copy
from abc import ABC
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import optuna.trial

from cufy.core.tuner import TunableModel


@dataclass
class ModelParam:
    position: int
    min_value: float
    max_value: float
    value: float | None = None
    name: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.value is not None and not (self.min_value <= self.value <= self.max_value):
            raise ValueError(
                f"Parameter '{self.name}' value {self.value} is out of bounds [{self.min_value}, {self.max_value}]"
            )


class ParameterizedModel(TunableModel, ABC):
    def __init__(self, param_overrides: Mapping[str, ModelParam] | None = None) -> None:
        super().__init__()
        self.params = self._collect_params(param_overrides)

    def _collect_params(self, param_overrides: Mapping[str, ModelParam] | None) -> list[ModelParam]:
        found: dict[str, ModelParam] = {}
        for cls in reversed(self.__class__.mro()):
            for k, v in vars(cls).items():
                if isinstance(v, ModelParam):
                    found[k] = v
                elif k in found:
                    raise TypeError(
                        f"Class {cls.__name__} overrides parameter '{k}' with a non-ModelParam value "
                        f"of type {type(v).__name__}. Parameters must be overridden with ModelParam instances"
                    )

        if not found:
            raise ValueError("No parameter fields found")

        overrides = param_overrides or {}
        unknown = sorted(set(overrides.keys()) - set(found.keys()))
        if unknown:
            raise ValueError(f"Unknown parameter overrides: {unknown}. Known parameters: {sorted(found.keys())}")

        bound: dict[str, ModelParam] = {}
        for name, default in found.items():
            p = copy.copy(overrides[name] if name in overrides else default)
            p.name = name
            bound[name] = p
            object.__setattr__(self, name, p)

        params = sorted(bound.values(), key=lambda p: p.position)
        positions: dict[int, ModelParam] = {}
        for p in params:
            if p.position in positions:
                raise ValueError(
                    f"Duplicate parameter.position values. Got {p.name} with position {p.position} "
                    f"when {positions[p.position].name} was already defined"
                )
            positions[p.position] = p
        return params

    def is_initialized(self) -> bool:
        return all(p.value is not None for p in self.params)

    @contextmanager
    def temporary_state(self) -> Iterator[None]:
        saved_values = {p.name: p.value for p in self.params}
        try:
            yield
        finally:
            for p in self.params:
                p.value = saved_values[p.name]

    def suggest(self, trial: optuna.trial.BaseTrial) -> None:
        """No model hyperparameters by default. Override to expose tunable hyperparameters to Optuna."""
