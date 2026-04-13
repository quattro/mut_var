from __future__ import annotations

# pattern: Functional Core
import enum

from dataclasses import dataclass
from typing import Any


class RESULTS(enum.Enum):
    r"""Canonical status codes for numerics operations.

    Used as the ``result`` field of every :class:`Solution` returned by
    numerics kernels.  Downstream callers should branch on these values rather
    than inspecting ``Solution.value`` directly.
    """

    successful = "successful"
    invalid_input = "invalid_input"
    empty_subset = "empty_subset"
    nonfinite_objective = "nonfinite_objective"
    max_steps_reached = "max_steps_reached"


@dataclass(frozen=True)
class Solution:
    r"""Immutable result container for all numerics operations.

    **Attributes:**

    - ``value``: The computed result (type depends on the calling function).
    - ``result``: Status code from :class:`RESULTS`.
    - ``stats``: Optional diagnostics dict (step counts, objective values, …).
    - ``state``: Reserved; always ``None`` in current implementations.
    """

    value: Any
    result: RESULTS
    stats: dict[str, Any] | None = None
    state: Any = None

    @property
    def ok(self) -> bool:
        r"""Return ``True`` only when ``result`` is ``RESULTS.successful``."""
        return bool(self.result == RESULTS.successful)
