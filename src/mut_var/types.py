# pattern: Functional Core
from __future__ import annotations

import enum

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

VarianceLink = Literal["none", "maf_power", "maf_power_shifted"]
AfModel = Literal["uniform", "beta"]
SeModel = Literal["constant", "af_n_scaled"]


class InferenceConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-5
    step_size: float = 0.01
    filter_threshold: float = 1e-8
    penalty: float = 1.0


class SimulationConfig(NamedTuple):
    n_rows: int
    seed: int = 0
    weights: tuple[float, ...] = (0.95, 0.05)
    log_var_scales: tuple[float, ...] = (-8.0, -5.5)
    variance_link: VarianceLink = "maf_power"
    theta: float = 0.5
    link_eps: float = 1e-8
    link_shift: float = 0.0
    af_clip_min: float = 1e-4
    af_model: AfModel = "beta"
    af_uniform_low: float = 0.01
    af_uniform_high: float = 0.5
    af_beta_a: float = 0.4
    af_beta_b: float = 0.4
    se_model: SeModel = "af_n_scaled"
    se_constant: float = 0.02
    sample_size: float = 50000.0
    se_scale: float = 1.0


class RESULTS(enum.Enum):
    r"""Canonical status codes for numerics operations.

    Used as the ``result`` field of every :class:`Solution` returned by
    numerics kernels. Downstream callers should branch on these values rather
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
    - ``stats``: Optional diagnostics dict (step counts, objective values, ...).
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


__all__ = ["InferenceConfig", "SimulationConfig", "RESULTS", "Solution"]
