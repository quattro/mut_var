from __future__ import annotations

import logging

from collections.abc import Callable

# pattern: Mixed (unavoidable)
# Reason: preserves compatibility while consolidating numerics and orchestration entrypoints.
from typing import Any, Mapping, NamedTuple, TYPE_CHECKING

import jax.numpy as jnp
import jax.random as rdm
import polars as pl

from jaxtyping import ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.io import validate_maf_grid, validate_numeric_columns, validate_required_columns, validate_sumstats_domain

if TYPE_CHECKING:
    from mut_var.numerics.baseline import BaselineConfig, Params
    from mut_var.numerics.refit import RefitConfig


class InferenceArrays(NamedTuple):
    af: ArrayLike
    beta_hat: ArrayLike
    s2: ArrayLike


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


def _filter_components(params: Params, threshold: float) -> Params:
    keep = params.pi > threshold
    keep = keep.at[0].set(True)
    pi = params.pi[keep]
    pi = pi / jnp.sum(pi)
    return params.__class__(
        pi=pi,
        mu_k=params.mu_k[keep[1:]],
        var_k=params.var_k[keep[1:]],
    )


def _build_long_payload(models: list[Params], maf_grid: ArrayLike, af: ArrayLike) -> dict[str, Any]:
    maf_arr = jnp.asarray(maf_grid, dtype=jnp.float64)
    af_arr = jnp.asarray(af, dtype=jnp.float64)

    empirical_min_maf = jnp.minimum(jnp.min(af_arr), 1.0 - jnp.max(af_arr))
    maf_values = jnp.concatenate((jnp.asarray([empirical_min_maf], dtype=jnp.float64), maf_arr))
    names = [f"pi{idx}" for idx in range(len(models))]

    mu0 = jnp.asarray(jnp.pad(models[0].mu_k, (1, 0)), dtype=jnp.float64)
    var0 = jnp.asarray(jnp.pad(models[0].var_k, (1, 0)), dtype=jnp.float64)

    if any(model.pi.shape[0] != mu0.shape[0] for model in models):
        raise ValueError("All models must keep the same number of mixture components.")

    values = jnp.concatenate([jnp.asarray(model.pi, dtype=jnp.float64) for model in models])
    n_comp = int(mu0.shape[0])
    name_values = [name for name in names for _ in range(n_comp)]

    return {
        "mu0": jnp.tile(mu0, len(models)),
        "var0": jnp.tile(var0, len(models)),
        "maf": jnp.repeat(maf_values, n_comp),
        "name": name_values,
        "value": values,
    }


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


def _solver_debug_callback(
    workflow_log: logging.Logger,
    stage: str,
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], None] | None:
    if not workflow_log.isEnabledFor(logging.DEBUG):
        return None

    def _verbose_callback(step_index: ArrayLike, loss: ArrayLike, grad_norm: ArrayLike) -> None:
        workflow_log.debug(
            "inference pipeline: %s solver step=%d loss=%.6g grad_norm=%.6g",
            stage,
            int(step_index),
            float(loss),
            float(grad_norm),
        )

    return _verbose_callback


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
    r"""Run the high-level inference workflow from dataframe ingress to dataframe egress.

    **Arguments:**
    - `df`: Input summary-statistics dataframe.
    - `af_col`: AF column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.
    - `lowest`: Minimum MAF threshold for grid construction.
    - `highest`: Maximum MAF threshold for grid construction.
    - `num_breaks`: Number of MAF grid breakpoints.
    - `seed`: PRNG seed for baseline initialization.
    - `config`: Optional numerics config; defaults to `InferenceConfig(num_clusters=30)`.
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**
    - Long-format inference dataframe suitable for downstream output.

    **Raises:**
    - `ValueError`: Boundary validation failure or invalid numerics result.
    - `RuntimeError`: Non-recoverable numerics failure.
    """
    from mut_var.adapters.tabular import build_maf_masks, payload_to_long_dataframe, to_inference_arrays
    from mut_var.numerics._solver_utils import is_recoverable_result, merge_recoverable_results
    from mut_var.numerics.baseline import fit_baseline
    from mut_var.numerics.refit import fit_refit_grid

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
    beta_hat = jnp.asarray(arrays.beta_hat)
    s2 = jnp.asarray(arrays.s2)

    workflow_log.info("inference pipeline: fitting baseline model")
    baseline_solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(seed),
        config=inference_config.to_baseline_config(),
        verbose_callback=_solver_debug_callback(workflow_log, "baseline"),
    )
    workflow_log.info("inference pipeline: baseline fit completed with result '%s'", RESULTS[baseline_solution.result])

    if not is_recoverable_result(baseline_solution.result):
        solution = baseline_solution
    else:
        workflow_log.info("inference pipeline: filtering baseline components")
        filtered = _filter_components(baseline_solution.value, inference_config.filter_threshold)

        workflow_log.info("inference pipeline: fitting refit grid")
        refit_solution = fit_refit_grid(
            beta_hat=beta_hat,
            s2=s2,
            maf_masks=maf_masks,
            init=filtered,
            config=inference_config.to_refit_config(),
            verbose_callback=_solver_debug_callback(workflow_log, "refit"),
        )
        workflow_log.info("inference pipeline: refit grid completed with result '%s'", RESULTS[refit_solution.result])

        if not is_recoverable_result(refit_solution.result):
            solution = refit_solution
        else:
            workflow_log.info("inference pipeline: building numerics payload")
            models = refit_solution.value
            numerics_payload = _build_long_payload(models, maf_grid=maf_grid, af=arrays.af)
            solution = Solution(
                value=numerics_payload,
                result=merge_recoverable_results(baseline_solution.result, refit_solution.result),
                stats={
                    "num_models": len(models),
                    "num_components": int(models[0].pi.shape[0]),
                    "baseline": baseline_solution.stats,
                    "refit": refit_solution.stats,
                },
                state=None,
            )

    workflow_log.info("inference pipeline: numerics completed with result '%s'", RESULTS[solution.result])

    workflow_log.info("inference pipeline: preparing output dataframe")
    output_payload = _payload_from_solution(solution)
    result_df = payload_to_long_dataframe(output_payload)
    workflow_log.info("inference pipeline: output dataframe prepared (%d rows)", result_df.height)
    return result_df


__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "run_inference_pipeline",
]
