from __future__ import annotations

# pattern: Functional Core
from typing import cast

import numpy as np

from mut_var.contracts import RESULTS

RECOVERABLE_RESULTS = (RESULTS.successful, RESULTS.max_steps_reached)


def is_recoverable_result(result: RESULTS) -> bool:
    r"""Return whether a status is recoverable for staged numerics pipelines."""
    return result in RECOVERABLE_RESULTS


def merge_recoverable_results(*results: RESULTS) -> RESULTS:
    r"""Merge recoverable statuses, preferring `max_steps_reached` when present."""
    if any(result == RESULTS.max_steps_reached for result in results):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.successful)


def is_nonfinite(value: object) -> bool:
    r"""Return ``True`` when any value is non-finite."""
    return not bool(np.isfinite(np.asarray(value)).all())
