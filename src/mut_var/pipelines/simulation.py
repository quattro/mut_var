from __future__ import annotations

import logging

import numpy as np
import polars as pl

from mut_var.numerics import simulate_mixture_data, SimulationArrays
from mut_var.pipelines.artifacts import SimulationArtifacts
from mut_var.types import RESULTS, SimulationConfig, Solution


def _pipeline_reason(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"simulation failed with status '{solution.result.value}'."


def _truth_dataframe(arrays: SimulationArrays) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": np.asarray(arrays.row_id, dtype=np.int64),
            "component": np.asarray(arrays.component, dtype=np.int64),
            "beta_true": np.asarray(arrays.beta_true, dtype=np.float64),
            "sigma2": np.asarray(arrays.sigma2, dtype=np.float64),
            "effect_allele_frequency": np.asarray(arrays.af, dtype=np.float64),
        }
    ).select(
        [
            "row_id",
            "component",
            "beta_true",
            "sigma2",
            "effect_allele_frequency",
        ]
    )


def _observed_dataframe(arrays: SimulationArrays) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": np.asarray(arrays.row_id, dtype=np.int64),
            "effect_allele_frequency": np.asarray(arrays.af, dtype=np.float64),
            "beta": np.asarray(arrays.beta_hat, dtype=np.float64),
            "standard_error": np.asarray(arrays.se, dtype=np.float64),
        }
    ).select(
        [
            "row_id",
            "effect_allele_frequency",
            "beta",
            "standard_error",
        ]
    )


def _metadata_dataframe(arrays: SimulationArrays, config: SimulationConfig) -> pl.DataFrame:
    truth_df = _truth_dataframe(arrays)
    with_deciles = truth_df.with_columns(
        (
            ((pl.col("effect_allele_frequency").rank("ordinal") - 1) * 10.0 / pl.len())
            .floor()
            .clip(0, 9)
            .cast(pl.Int64)
            .alias("af_decile")
        )
    )

    diagnostics = with_deciles.group_by("af_decile").agg(
        [
            pl.col("beta_true").var(ddof=0).alias("empirical_var_beta_true"),
            pl.col("sigma2").mean().alias("empirical_mean_sigma2"),
        ]
    )

    all_deciles = pl.DataFrame({"af_decile": list(range(10))})
    merged = all_deciles.join(diagnostics, on="af_decile", how="left").sort("af_decile")
    return merged.with_columns(
        [
            pl.lit(int(config.seed)).alias("seed"),
            pl.lit(int(config.n_rows)).alias("n_rows"),
            pl.lit(int(len(config.weights))).alias("num_components"),
            pl.lit(str(config.variance_link)).alias("variance_link"),
            pl.lit(float(config.theta)).alias("theta"),
        ]
    ).select(
        [
            "seed",
            "n_rows",
            "num_components",
            "variance_link",
            "theta",
            "af_decile",
            "empirical_var_beta_true",
            "empirical_mean_sigma2",
        ]
    )


def run_simulation_pipeline(
    *,
    config: SimulationConfig,
    log: logging.Logger | None = None,
) -> SimulationArtifacts:
    r"""Run simulation numerics and prepare tabular truth/observed/metadata artifacts.

    **Arguments:**

    - `config`: Pipeline simulation controls and numerics configuration.
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**

    - `SimulationArtifacts` containing `truth`, `observed`, and `metadata` dataframes.

    **Raises:**

    - `ValueError`: Invalid simulation config/result status.
    - `RuntimeError`: Non-recoverable simulation failure.
    """
    workflow_log = logging.getLogger(__name__) if log is None else log
    workflow_log.info("simulation pipeline: validating config")

    workflow_log.info("simulation pipeline: running numerics")
    solution = simulate_mixture_data(config=config)

    if solution.result != RESULTS.successful:
        reason = _pipeline_reason(solution)
        if solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
            raise ValueError(reason)
        raise RuntimeError(reason)

    if not isinstance(solution.value, SimulationArrays):
        raise ValueError("simulation payload must be SimulationArrays.")

    workflow_log.info("simulation pipeline: preparing artifacts")
    arrays = solution.value
    truth_df = _truth_dataframe(arrays)
    observed_df = _observed_dataframe(arrays)
    metadata_df = _metadata_dataframe(arrays, config)
    return SimulationArtifacts(
        truth=truth_df,
        observed=observed_df,
        metadata=metadata_df,
    )


__all__ = ["run_simulation_pipeline"]
