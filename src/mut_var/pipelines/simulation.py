from __future__ import annotations

import logging


# pattern: Mixed (unavoidable)
# Reason: this module combines pure dataframe shaping helpers with workflow orchestration/logging.
from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl

from mut_var.numerics import simulate_mixture_data, SimulationArrays
from mut_var.types import RESULTS, SimulationPipelineConfig, Solution

if TYPE_CHECKING:
    from mut_var.numerics.simulate import SimulationNumericsConfig


class SimulationArtifacts(NamedTuple):
    truth: pl.DataFrame
    observed: pl.DataFrame
    metadata: pl.DataFrame


def _to_numpy(values: jnp.ndarray | object, dtype: np.dtype) -> np.ndarray:
    return np.asarray(jnp.asarray(values), dtype=dtype)


class _SimulationColumns(NamedTuple):
    row_id: np.ndarray
    af: np.ndarray
    component: np.ndarray
    beta_true: np.ndarray
    sigma2: np.ndarray
    se: np.ndarray
    beta_hat: np.ndarray


def _simulation_columns(arrays: SimulationArrays) -> _SimulationColumns:
    return _SimulationColumns(
        row_id=_to_numpy(arrays.row_id, np.int64),
        af=_to_numpy(arrays.af, np.float64),
        component=_to_numpy(arrays.component, np.int64),
        beta_true=_to_numpy(arrays.beta_true, np.float64),
        sigma2=_to_numpy(arrays.sigma2, np.float64),
        se=_to_numpy(arrays.se, np.float64),
        beta_hat=_to_numpy(arrays.beta_hat, np.float64),
    )


def _pipeline_reason(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"simulation failed with status '{RESULTS[solution.result]}'."


def _truth_dataframe(columns: _SimulationColumns) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": columns.row_id,
            "component": columns.component,
            "beta_true": columns.beta_true,
            "sigma2": columns.sigma2,
            "effect_allele_frequency": columns.af,
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


def _observed_dataframe(columns: _SimulationColumns) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": columns.row_id,
            "effect_allele_frequency": columns.af,
            "beta": columns.beta_hat,
            "standard_error": columns.se,
        }
    ).select(
        [
            "row_id",
            "effect_allele_frequency",
            "beta",
            "standard_error",
        ]
    )


def _metadata_dataframe(
    truth_df: pl.DataFrame,
    config: SimulationPipelineConfig,
    numerics: SimulationNumericsConfig,
) -> pl.DataFrame:
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
            pl.lit(int(len(numerics.weights))).alias("num_components"),
            pl.lit(str(numerics.variance_link)).alias("variance_link"),
            pl.lit(float(numerics.theta)).alias("theta"),
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
    config: SimulationPipelineConfig,
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

    from mut_var.numerics.simulate import SimulationNumericsConfig

    effective_numerics = (
        config.numerics
        if config.numerics is not None
        else SimulationNumericsConfig(weights=(0.95, 0.05), log_var_scales=(-8.0, -5.5))
    )

    workflow_log.info("simulation pipeline: running numerics")
    solution = simulate_mixture_data(
        n_rows=config.n_rows,
        seed=config.seed,
        config=effective_numerics,
    )

    if solution.result != RESULTS.successful:
        reason = _pipeline_reason(solution)
        if solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
            raise ValueError(reason)
        raise RuntimeError(reason)

    if not isinstance(solution.value, SimulationArrays):
        raise ValueError("simulation payload must be SimulationArrays.")

    workflow_log.info("simulation pipeline: preparing artifacts")
    arrays = solution.value
    columns = _simulation_columns(arrays)
    truth_df = _truth_dataframe(columns)
    observed_df = _observed_dataframe(columns)
    metadata_df = _metadata_dataframe(truth_df, config, effective_numerics)
    return SimulationArtifacts(
        truth=truth_df,
        observed=observed_df,
        metadata=metadata_df,
    )


__all__ = [
    "SimulationArtifacts",
    "run_simulation_pipeline",
]
