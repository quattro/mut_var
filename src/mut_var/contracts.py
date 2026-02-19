from __future__ import annotations

# pattern: Functional Core
from typing import Any

import equinox as eqx
import equinox.internal as eqxi


class RESULTS(eqxi.Enumeration):
    successful = "successful"
    invalid_input = "invalid_input"
    empty_subset = "empty_subset"
    nonfinite_objective = "nonfinite_objective"
    max_steps_reached = "max_steps_reached"


class Solution(eqx.Module):
    value: Any
    result: RESULTS
    stats: dict[str, Any] | None = None
    state: Any = None

    @property
    def ok(self) -> bool:
        r"""Return `True` only when `result` is `RESULTS.successful`."""
        return bool(self.result == RESULTS.successful)
