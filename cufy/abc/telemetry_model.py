from abc import ABC
from functools import wraps
from typing import Any, Callable, TypeVar

from cufy.abc.model import Model

T = TypeVar("T", bound="TelemetryModel")
F = TypeVar("F", bound=Callable[..., Any])


def clears_telemetry(func: F) -> F:
    """
    Decorator for TelemetryModel methods.
    Ensures that the telemetry state is cleared before the method executes,
    preventing stale metrics from leaking into subsequent calls.
    """
    @wraps(func)
    def wrapper(self: "TelemetryModel", *args: Any, **kwargs: Any) -> Any:
        self.clear_state()
        return func(self, *args, **kwargs)
    
    return wrapper  # type: ignore


class TelemetryModel(Model, ABC):
    def __init__(self) -> None:
        self._numeric: dict[str, float] = {}
        self._string: dict[str, str] = {}

    def save_numeric(self, key: str, value: float) -> None:
        self._numeric[key] = float(value)

    def save_string(self, key: str, value: str) -> None:
        self._string[key] = str(value)

    def get_numeric(self) -> dict[str, float]:
        return dict(self._numeric)

    def get_string(self) -> dict[str, str]:
        return dict(self._string)

    def clear_state(self) -> None:
        self._numeric.clear()
        self._string.clear()
