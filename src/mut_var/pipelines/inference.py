# pattern: Imperative Shell
from __future__ import annotations

import logging

from collections.abc import Callable
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import polars as pl

import mut_var.numerics.mixture_fit as mixture_fit_module

from mut_var.io import (
    build_maf_masks,
    load_inference_arrays,
    payload_to_long_dataframe,
    validate_maf_grid,
)
from mut_var.numerics.mixsqp import is_recoverable_result, merge_recoverable_results
from mut_var.types import InferenceConfig, RESULTS, Solution

if TYPE_CHECKING:
    from mut_var.numerics.mixture_fit import Params


def _filter_components(params: Params, threshold: float) -> tuple[Params, np.ndarray]:
    keep = params.pi > threshold
    keep = keep.copy()
    keep[0] = True
    pi = params.pi[keep]
    pi = pi / np.sum(pi)
    return (
        params.__class__(
            pi=pi,
            mu_k=params.mu_k[keep[1:]],
            var_k=params.var_k[keep[1:]],
        ),
        keep,
    )


def _build_long_payload(models: list[Params], maf_grid: np.ndarray, af: np.ndarray) -> dict[str, Any]:
    maf_arr = np.asarray(maf_grid, dtype=float)
    af_arr = np.asarray(af, dtype=float)

    empirical_min_maf = float(np.minimum(np.min(af_arr), 1.0 - np.max(af_arr)))
    maf_values = np.concatenate(([empirical_min_maf], maf_arr))
    names = [f"pi{idx}" for idx in range(len(models))]

    mu0 = np.pad(models[0].mu_k, (1, 0)).astype(float)
    var0 = np.pad(models[0].var_k, (1, 0)).astype(float)

    if any(model.pi.shape[0] != mu0.shape[0] for model in models):
        raise ValueError("All models must keep the same number of mixture components.")

    values = np.concatenate([np.asarray(model.pi, dtype=float) for model in models])
    n_comp = int(mu0.shape[0])
    name_values = [name for name in names for _ in range(n_comp)]

    return {
        "mu0": np.tile(mu0, len(models)),
        "var0": np.tile(var0, len(models)),
        "maf": np.repeat(maf_values, n_comp),
        "name": name_values,
        "value": values,
    }


def _reason_from_solution(solution: Solution) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"inference failed with status '{solution.result.value}'."


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
) -> bool | Callable[..., None]:
    if not workflow_log.isEnabledFor(logging.DEBUG):
        return False

    def _verbose(step: int, obj: float, **_kw: Any) -> None:
        workflow_log.debug("%s | Step: %d, obj: %.6f", stage, step, obj)

    return _verbose


def _run_refit_grid(
    L_full: np.ndarray,
    maf_masks: np.ndarray,
    init: Params,
    config: InferenceConfig,
    workflow_log: logging.Logger,
) -> Solution:
    L_arr = np.asarray(L_full, dtype=float)
    masks_arr = np.asarray(maf_masks, dtype=bool)

    if L_arr.ndim != 2 or masks_arr.ndim != 2 or L_arr.shape[0] != masks_arr.shape[1]:
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "refit likelihood matrix must align with maf masks"},
        )

    active_obs = np.any(masks_arr, axis=0)
    if not active_obs.any():
        return Solution(
            value=[init],
            result=RESULTS.empty_subset,
            stats={"reason": "no observations are active for refit"},
        )

    L_active = L_arr[active_obs]
    masks_active = masks_arr[:, active_obs]

    models: list[Params] = [init]
    any_max_steps = False
    threshold_diagnostics: list[dict[str, Any]] = []

    for idx, mask in enumerate(masks_active):
        workflow_log.info(
            "inference pipeline: refit threshold %d/%d",
            idx + 1,
            masks_active.shape[0],
        )
        n_obs = int(mask.sum())
        if n_obs == 0:
            return Solution(
                value=models,
                result=RESULTS.empty_subset,
                stats={"reason": f"maf mask at index {idx} is empty"},
            )

        step_solution = mixture_fit_module.fit_refit_step(
            L_sub=L_active[mask],
            prev_params=models[-1],
            config=config,
            verbose=_solver_debug_callback(workflow_log, "refit"),
        )
        if not is_recoverable_result(step_solution.result):
            return Solution(
                value=models,
                result=step_solution.result,
                stats=step_solution.stats,
            )
        if step_solution.result == RESULTS.max_steps_reached:
            any_max_steps = True

        threshold_diagnostics.append({"threshold_index": idx, **dict(step_solution.stats or {})})
        models.append(step_solution.value)

    final_result = RESULTS.max_steps_reached if any_max_steps else RESULTS.successful
    return Solution(
        value=models,
        result=final_result,
        stats={
            "num_models": len(models),
            "num_thresholds": int(masks_active.shape[0]),
            "threshold_diagnostics": threshold_diagnostics,
        },
    )


def run_inference_pipeline(
    path: str,
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

    - `path`: Input summary-statistics TSV path.
    - `af_col`: AF column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.
    - `lowest`: Minimum MAF threshold for grid construction.
    - `highest`: Maximum MAF threshold for grid construction.
    - `num_breaks`: Number of MAF grid breakpoints.
    - `seed`: PRNG seed (retained for API compatibility; currently unused).
    - `config`: Optional numerics config; defaults to `InferenceConfig(num_clusters=30)`.
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**

    - Long-format inference dataframe suitable for downstream output.

    **Raises:**

    - `ValueError`: Boundary validation failure or invalid numerics result.
    - `RuntimeError`: Non-recoverable numerics failure.
    """
    workflow_log = logging.getLogger(__name__) if log is None else log
    inference_config = config if config is not None else InferenceConfig(num_clusters=30)

    workflow_log.info("inference pipeline: validating input data")
    validate_maf_grid(lowest, highest, num_breaks)
    arrays = load_inference_arrays(path, af_col, beta_col, se_col)
    workflow_log.info("inference pipeline: input validation complete")
    workflow_log.info("inference pipeline: building maf grid and masks")
    maf_grid = np.exp(np.linspace(np.log(lowest), np.log(highest), num_breaks))
    maf_masks = build_maf_masks(arrays.af, maf_grid)

    workflow_log.info("inference pipeline: starting numerics")
    workflow_log.info("inference pipeline: preparing fit state")
    fit_state_solution = mixture_fit_module.prepare_fit_state(
        beta_hat=arrays.beta_hat,
        s2=arrays.s2,
        config=inference_config,
    )
    workflow_log.info(
        "inference pipeline: fit state prepared with result '%s'",
        fit_state_solution.result.value,
    )

    if fit_state_solution.result != RESULTS.successful:
        solution = fit_state_solution
    else:
        fit_state = fit_state_solution.value
        workflow_log.info("inference pipeline: fitting baseline model with config %s", inference_config)
        baseline_solution = mixture_fit_module.fit_baseline(
            state=fit_state,
            config=inference_config,
            verbose=_solver_debug_callback(workflow_log, "baseline"),
        )
        workflow_log.info(
            "inference pipeline: baseline fit completed with result '%s'",
            baseline_solution.result.value,
        )

        if not is_recoverable_result(baseline_solution.result):
            solution = baseline_solution
        else:
            workflow_log.info("inference pipeline: filtering baseline components")
            filtered, keep = _filter_components(baseline_solution.value, inference_config.filter_threshold)
            filtered_likelihood = fit_state.likelihood_matrix[:, keep]

            workflow_log.info("inference pipeline: fitting refit grid")
            refit_solution = _run_refit_grid(
                filtered_likelihood,
                maf_masks,
                filtered,
                inference_config,
                workflow_log,
            )
            workflow_log.info(
                "inference pipeline: refit grid completed with result '%s'",
                refit_solution.result.value,
            )

            if not is_recoverable_result(refit_solution.result):
                solution = refit_solution
            else:
                workflow_log.info("inference pipeline: building numerics payload")
                models = refit_solution.value
                solution = Solution(
                    value=_build_long_payload(models, maf_grid=maf_grid, af=arrays.af),
                    result=merge_recoverable_results(baseline_solution.result, refit_solution.result),
                    stats={
                        "num_models": len(models),
                        "num_components": int(models[0].pi.shape[0]),
                        "prepare": fit_state_solution.stats,
                        "baseline": baseline_solution.stats,
                        "refit": refit_solution.stats,
                    },
                )

    workflow_log.info(
        "inference pipeline: numerics completed with result '%s'",
        solution.result.value,
    )

    workflow_log.info("inference pipeline: preparing output dataframe")
    output_payload = _payload_from_solution(solution)
    result_df = payload_to_long_dataframe(output_payload)
    workflow_log.info("inference pipeline: output dataframe prepared (%d rows)", result_df.height)
    return result_df


__all__ = ["run_inference_pipeline"]
