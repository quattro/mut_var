from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RESULTS(str, Enum):
    successful = "successful"
    invalid_input = "invalid_input"
    empty_subset = "empty_subset"
    nonfinite_objective = "nonfinite_objective"
    max_steps_reached = "max_steps_reached"


@dataclass(frozen=True, slots=True)
class Solution:
    value: Any
    result: RESULTS
    stats: dict[str, Any] | None = None
    state: Any = None

    @property
    def ok(self) -> bool:
        return self.result == RESULTS.successful
