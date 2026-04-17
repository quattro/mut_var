from __future__ import annotations

import logging

from collections.abc import Callable

# pattern: Mixed (unavoidable)
# Reason: preserves compatibility while consolidating numerics and orchestration entrypoints.
from typing import Any, Mapping, TYPE_CHECKING

import jax
import jax.debug as jdb
import jax.numpy as jnp
import numpy as np
import polars as pl

from jaxtyping import ArrayLike

import mut_var.numerics.mixture_fit as mixture_fit

from mut_var.io import (
    load_inference_arrays,
    payload_to_long_dataframe,
    validate_maf_grid,
)
from mut_var.numerics._solver_utils import is_recoverable_result, merge_recoverable_results
from mut_var.types import InferenceConfig, RESULTS, Solution

if TYPE_CHECKING:
    from mut_var.numerics.mixture_fit import FitState, Params


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


def _maf_subset_mask(af: ArrayLike, threshold: float) -> jax.Array:
    af_arr = jnp.asarray(af)
    return jnp.logical_and(af_arr >= threshold, af_arr <= (1.0 - threshold))


def _build_long_dataframe(
    pi_rows: list[ArrayLike],
    maf_grid: ArrayLike,
    af: ArrayLike,
    mu_k: ArrayLike,
    var_k: ArrayLike,
) -> pl.DataFrame:
    maf_arr = jnp.asarray(maf_grid)
    af_arr = jnp.asarray(af)

    empirical_min_maf = float(jnp.minimum(jnp.min(af_arr), 1.0 - jnp.max(af_arr)))
    maf_values = np.concatenate(([empirical_min_maf], np.asarray(maf_arr)))
    mu0 = np.asarray(jnp.pad(jnp.asarray(mu_k), (1, 0)))
    var0 = np.asarray(jnp.pad(jnp.asarray(var_k), (1, 0)))
    n_models = len(pi_rows)

    if any(jnp.asarray(pi).shape[0] != mu0.shape[0] for pi in pi_rows):
        raise ValueError("All models must keep the same number of mixture components.")

    values = np.concatenate([np.asarray(jnp.asarray(pi)) for pi in pi_rows])
    n_comp = int(mu0.shape[0])
    name_values = [f"pi{idx}" for idx in range(n_models) for _ in range(n_comp)]

    return pl.DataFrame(
        {
            "mu0": np.tile(mu0, n_models),
            "var0": np.tile(var0, n_models),
            "maf": np.repeat(maf_values, n_comp),
            "name": name_values,
            "value": values,
        }
    ).select(["mu0", "var0", "maf", "name", "value"])


def _reason_from_solution(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"inference failed with status '{RESULTS[solution.result]}'."


def _payload_from_solution(solution: Solution) -> Mapping[str, object] | pl.DataFrame:
    if solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        reason = _reason_from_solution(solution)
        if solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
            raise ValueError(reason)
        raise RuntimeError(reason)

    if not isinstance(solution.value, (Mapping, pl.DataFrame)):
        raise ValueError("inference payload must be a mapping or dataframe.")

    return solution.value


def _solver_debug_callback(
    workflow_log: logging.Logger,
    stage: str,
) -> bool | Callable[..., None]:
    if not workflow_log.isEnabledFor(logging.DEBUG):
        return False
    """
    Returns a Callable[..., None] that can be called inside JIT code
    and logs its keyword arguments via jax.debug.callback.
    """

    def _verbose(**kwargs: tuple[str, Any]) -> None:
        if not kwargs:
            return

        # We build format + args outside the callback so only
        # concrete runtime values are transferred.
        items = list(kwargs.values())
        fmt = ", ".join(f"{k}: %s" for k, _ in items)
        fmt = f"{stage} | {fmt}"
        args = tuple(v for _, v in items)

        def _log_callback(*cb_args):
            workflow_log.debug(fmt, *cb_args)

        # Pass only runtime values positionally
        jdb.callback(_log_callback, *args)

    return _verbose


def run_inference_pipeline(
    path: str,
    *,
    af_col: str = "effect_allele_frequency",
    beta_col: str = "beta",
    se_col: str = "standard_error",
    lowest: float = 1e-5,
    highest: float = 1e-2,
    num_breaks: int = 10,
    config: InferenceConfig | None = None,
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    r"""Run the high-level inference workflow from path ingress to dataframe egress.

    **Arguments:**

    - `path`: Input summary-statistics TSV path.
    - `af_col`: AF column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.
    - `lowest`: Minimum MAF threshold for grid construction.
    - `highest`: Maximum MAF threshold for grid construction.
    - `num_breaks`: Number of MAF grid breakpoints.
    - `config`: Optional numerics config; defaults to `InferenceConfig(num_clusters=30)`.
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**

    - Long-format inference dataframe suitable for downstream output.

    **Raises:**

    - `ValueError`: Boundary validation failure or invalid numerics result.
    - `RuntimeError`: Non-recoverable numerics failure.
    - `FileNotFoundError`: Input path does not exist.
    """
    workflow_log = logging.getLogger(__name__) if log is None else log

    workflow_log.info("inference pipeline: validating grid parameters")
    validate_maf_grid(lowest, highest, num_breaks)

    workflow_log.info("inference pipeline: loading input data from '%s'", path)
    arrays = load_inference_arrays(path, af_col=af_col, beta_col=beta_col, se_col=se_col)
    workflow_log.info("inference pipeline: input loaded and validated")

    workflow_log.info("inference pipeline: building maf grid and masks")
    maf_grid = jnp.exp(jnp.linspace(jnp.log(lowest), jnp.log(highest), num_breaks))
    workflow_log.info("inference pipeline: starting numerics")
    beta_hat = jnp.asarray(arrays.beta_hat)
    s2 = jnp.asarray(arrays.s2)
    inference_config = config if config is not None else InferenceConfig(num_clusters=30)

    workflow_log.info("inference pipeline: preparing fit state")
    state_solution = mixture_fit.prepare_fit_state(beta_hat, s2, inference_config)
    if not is_recoverable_result(state_solution.result):
        solution = state_solution
    else:
        state: FitState = state_solution.value

        workflow_log.info("inference pipeline: fitting baseline model")
        baseline_solution = mixture_fit.fit_baseline(
            state,
            inference_config,
            verbose=_solver_debug_callback(workflow_log, "baseline"),
        )
        workflow_log.info(
            "inference pipeline: baseline fit completed with result '%s'",
            RESULTS[baseline_solution.result],
        )

        if not is_recoverable_result(baseline_solution.result):
            solution = baseline_solution
        else:
            workflow_log.info("inference pipeline: filtering baseline components")
            filtered = _filter_components(baseline_solution.value, inference_config.filter_threshold)
            keep = baseline_solution.value.pi > inference_config.filter_threshold
            keep = keep.at[0].set(True)
            L_filtered = state.likelihood_matrix[:, keep]

            workflow_log.info("inference pipeline: fitting refit grid")
            pi_rows: list[jax.Array] = [filtered.pi]
            prev_params = filtered
            step_results: list[RESULTS] = []
            solution = baseline_solution
            for threshold in np.asarray(maf_grid):
                mask = _maf_subset_mask(arrays.af, float(threshold))
                step_sol = mixture_fit.fit_refit_masked_step(
                    L_filtered,
                    mask,
                    prev_params,
                    inference_config,
                    verbose=_solver_debug_callback(workflow_log, "refit"),
                )
                step_results.append(step_sol.result)
                if not is_recoverable_result(step_sol.result):
                    solution = step_sol
                    break
                prev_params = step_sol.value
                pi_rows.append(prev_params.pi)
            else:
                refit_result = merge_recoverable_results(*step_results)
                workflow_log.info(
                    "inference pipeline: refit grid completed with result '%s'",
                    RESULTS[refit_result],
                )
                if not is_recoverable_result(refit_result):
                    solution = Solution(
                        value=None,
                        result=refit_result,
                        stats={"reason": f"refit failed with status '{RESULTS[refit_result]}'."},
                        state=None,
                    )
                else:
                    workflow_log.info("inference pipeline: building numerics payload")
                    solution = Solution(
                        value=_build_long_dataframe(
                            pi_rows,
                            maf_grid=maf_grid,
                            af=arrays.af,
                            mu_k=filtered.mu_k,
                            var_k=filtered.var_k,
                        ),
                        result=merge_recoverable_results(baseline_solution.result, refit_result),
                        stats={
                            "num_models": len(pi_rows),
                            "num_components": int(pi_rows[0].shape[0]),
                            "baseline": baseline_solution.stats,
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
    "run_inference_pipeline",
]
