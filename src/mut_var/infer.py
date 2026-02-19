from __future__ import annotations

# pattern: Imperative Shell
from typing import Any, Mapping

import jax.numpy as jnp
import polars as pl

from mut_var.adapters.tabular import build_maf_masks, payload_to_long_dataframe, to_inference_arrays
from mut_var.contracts import RESULTS, Solution
from mut_var.io import validate_maf_grid, validate_numeric_columns, validate_required_columns, validate_sumstats_domain
from mut_var.numerics.pipeline import (
    InferenceArrays,
    InferenceConfig,
    run_inference_pipeline as _run_numerics_inference_pipeline,
)


class InferencePipelineError(RuntimeError):
    result: RESULTS
    stats: dict[str, Any] | None

    def __init__(self, result: RESULTS, reason: str, *, stats: dict[str, Any] | None = None):
        super().__init__(reason)
        self.result = result
        self.stats = stats


def _reason_from_solution(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"inference failed with status '{RESULTS[solution.result]}'."


def _payload_from_solution(solution: Solution) -> Mapping[str, object]:
    if solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        raise InferencePipelineError(
            result=solution.result,
            reason=_reason_from_solution(solution),
            stats=solution.stats if isinstance(solution.stats, dict) else None,
        )

    if not isinstance(solution.value, Mapping):
        raise InferencePipelineError(
            result=RESULTS.invalid_input,
            reason="inference payload must be a mapping.",
            stats=solution.stats if isinstance(solution.stats, dict) else None,
        )

    return solution.value


def run_inference_pipeline(
    df: pl.DataFrame,
    *,
    af_col: str = "effect_allele_frequency",
    beta_col: str = "beta",
    se_col: str = "standard_error",
    lowest: float = 1e-5,
    highest: float = 1e-2,
    num_breaks: int = 10,
    seed: int = 0,
    config: InferenceConfig | None = None,
) -> pl.DataFrame:
    validate_maf_grid(lowest, highest, num_breaks)
    validate_required_columns(df, af_col, beta_col, se_col)
    validate_numeric_columns(df, af_col, beta_col, se_col)
    validate_sumstats_domain(df, af_col, se_col)

    arrays = to_inference_arrays(
        df,
        af_col=af_col,
        beta_col=beta_col,
        se_col=se_col,
    )

    maf_grid = jnp.exp(jnp.linspace(jnp.log(lowest), jnp.log(highest), num_breaks))
    maf_masks = build_maf_masks(arrays.af, maf_grid)
    inference_config = config if config is not None else InferenceConfig(num_clusters=30)
    solution = _run_numerics_inference_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=seed,
        config=inference_config,
    )
    payload = _payload_from_solution(solution)
    return payload_to_long_dataframe(payload)


__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "InferencePipelineError",
    "run_inference_pipeline",
]
