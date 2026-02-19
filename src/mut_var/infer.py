from __future__ import annotations

# pattern: Imperative Shell
import logging

from typing import Mapping

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


def _reason_from_solution(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"inference failed with status '{RESULTS[solution.result]}'."


def _payload_from_solution(solution: Solution) -> Mapping[str, object]:
    if solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        reason = _reason_from_solution(solution)
        if solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
            raise ValueError(reason)
        raise RuntimeError(reason)

    if not isinstance(solution.value, Mapping):
        raise ValueError("inference payload must be a mapping.")

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
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    workflow_log = logging.getLogger(__name__) if log is None else log

    workflow_log.info("inference pipeline: validating input data")
    validate_maf_grid(lowest, highest, num_breaks)
    validate_required_columns(df, af_col, beta_col, se_col)
    validate_numeric_columns(df, af_col, beta_col, se_col)
    validate_sumstats_domain(df, af_col, se_col)
    workflow_log.info("inference pipeline: input validation complete")

    workflow_log.info("inference pipeline: converting tabular data to arrays")
    arrays = to_inference_arrays(
        df,
        af_col=af_col,
        beta_col=beta_col,
        se_col=se_col,
    )

    workflow_log.info("inference pipeline: building maf grid and masks")
    maf_grid = jnp.exp(jnp.linspace(jnp.log(lowest), jnp.log(highest), num_breaks))
    maf_masks = build_maf_masks(arrays.af, maf_grid)
    inference_config = config if config is not None else InferenceConfig(num_clusters=30)

    workflow_log.info("inference pipeline: starting numerics")
    solution = _run_numerics_inference_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=seed,
        config=inference_config,
    )
    workflow_log.info("inference pipeline: numerics completed with result '%s'", RESULTS[solution.result])

    workflow_log.info("inference pipeline: preparing output dataframe")
    payload = _payload_from_solution(solution)
    result_df = payload_to_long_dataframe(payload)
    workflow_log.info("inference pipeline: output dataframe prepared (%d rows)", result_df.height)
    return result_df


__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "run_inference_pipeline",
]
