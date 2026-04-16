from __future__ import annotations

# pattern: Functional Core
from typing import Any, NamedTuple, TYPE_CHECKING

import equinox as eqx
import equinox.internal as eqxi

if TYPE_CHECKING:
    from mut_var.numerics.baseline import BaselineConfig
    from mut_var.numerics.refit import RefitConfig
    from mut_var.numerics.simulate import SimulationNumericsConfig


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


class InferenceConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01
    filter_threshold: float = 1e-8
    penalty: float = 1.0

    def to_baseline_config(self) -> BaselineConfig:
        r"""Convert pipeline controls to baseline-stage solver config."""
        from mut_var.numerics.baseline import BaselineConfig

        return BaselineConfig(
            num_clusters=self.num_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )

    def to_refit_config(self) -> RefitConfig:
        r"""Convert pipeline controls to refit-stage solver config."""
        from mut_var.numerics.refit import RefitConfig

        return RefitConfig(
            penalty=self.penalty,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )


class SimulationPipelineConfig(NamedTuple):
    n_rows: int
    seed: int = 0
    numerics: SimulationNumericsConfig | None = None
