from dataclasses import dataclass, field
from typing import Callable

from cufy.abc.runner import Runner


@dataclass(frozen=True, slots=True)
class Trial:
    id: str
    make_runner: Callable[[], Runner]
    tags: dict[str, str] = field(default_factory=dict)
