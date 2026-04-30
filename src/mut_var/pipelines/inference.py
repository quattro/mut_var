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
    # Drop signal components with negligible baseline weight before the refit
    # cycle, while always retaining the null component.
    keep = params.pi > threshold
    keep[0] = True
    pi = params.pi[keep]
    pi_sum = np.sum(pi)
    pi = pi / pi_sum if pi_sum > 0.0 else np.ones_like(pi) / float(pi.shape[0])
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

    # The baseline row uses the smallest observed MAF as its "threshold" so it
    # sits below the grid when the output is plotted in MAF order.
    per_row_maf = np.minimum(af_arr, 1.0 - af_arr)
    positive_maf = per_row_maf[per_row_maf > 0.0]
    baseline_maf = float(np.min(positive_maf)) if positive_maf.size > 0 else 0.0
    maf_values = np.concatenate(([baseline_maf], maf_arr))
    names = [f"pi{idx}" for idx in range(len(models))]

    # mu_k and var_k are fixed after the baseline fit; only pi varies across
    # MAF thresholds, so models[0] is the canonical source for component means/variances.
    mu0 = np.pad(models[0].mu_k, (1, 0)).astype(float)
    var0 = np.pad(models[0].var_k, (1, 0)).astype(float)

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
    prior: np.ndarray | None = None,
) -> Solution:
    # Compress out observations that pass no MAF threshold; this shrinks every
    # submatrix slice below and keeps the solver from seeing rows that never
    # contribute to any refit.
    active_obs = np.any(maf_masks, axis=0)
    if not active_obs.any():
        return Solution(
            value=[init],
            result=RESULTS.empty_subset,
            stats={"reason": "no observations are active for refit"},
        )

    L_active = L_full[active_obs]
    masks_active = maf_masks[:, active_obs]

    models: list[Params] = [init]
    any_max_steps = False
    threshold_diagnostics: list[dict[str, Any]] = []

    for idx, mask in enumerate(masks_active):
        workflow_log.info(
            "inference pipeline: refit threshold %d/%d",
            idx + 1,
            masks_active.shape[0],
        )
        if not mask.any():
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
            prior=prior,
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
    lowest: float | None = None,
    highest: float = 1e-2,
    num_breaks: int = 10,
    config: InferenceConfig | None = None,
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    r"""Run the high-level inference workflow from dataframe ingress to dataframe egress.

    **Arguments:**

    - `path`: Input summary-statistics TSV path.
    - `af_col`: AF column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.
    - `lowest`: Minimum MAF threshold for grid construction. When `None`
      (default), the lower bound is auto-derived from the data as the minimum
      observed MAF so the baseline point and the first refit threshold share
      the same left edge.
    - `highest`: Maximum MAF threshold for grid construction.
    - `num_breaks`: Number of MAF grid breakpoints.
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
    arrays = load_inference_arrays(path, af_col, beta_col, se_col)

    if lowest is None:
        # Auto-derive the grid floor from the observed MAF distribution so the
        # first refit threshold lands on real data rather than a fixed default
        # that can sit a decade above the smallest observation.
        per_row_maf = np.minimum(arrays.af, 1.0 - arrays.af)
        positive_maf = per_row_maf[per_row_maf > 0.0]
        if positive_maf.size == 0:
            raise ValueError(
                "cannot auto-derive maf grid lower bound: all observations have MAF == 0. " "Pass an explicit `lowest`."
            )
        resolved_lowest = float(np.min(positive_maf))
        workflow_log.info(
            "inference pipeline: auto-derived maf grid lowest=%g from data",
            resolved_lowest,
        )
    else:
        resolved_lowest = float(lowest)

    validate_maf_grid(resolved_lowest, highest, num_breaks)
    workflow_log.info("inference pipeline: input validation complete")
    workflow_log.info("inference pipeline: building maf grid and masks")
    maf_grid = np.exp(np.linspace(np.log(resolved_lowest), np.log(highest), num_breaks))
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
        baseline_prior = np.ones(fit_state.initial_params.pi.shape[0], dtype=float)
        baseline_prior[0] = 10.0
        baseline_solution = mixture_fit_module.fit_baseline(
            state=fit_state,
            config=inference_config,
            verbose=_solver_debug_callback(workflow_log, "baseline"),
            prior=baseline_prior,
        )
        workflow_log.info(
            "inference pipeline: baseline fit completed with result '%s'",
            baseline_solution.result.value,
        )

        if not is_recoverable_result(baseline_solution.result):
            solution = baseline_solution
        else:
            workflow_log.info("inference pipeline: filtering baseline components")
            filtered_baseline, keep = _filter_components(baseline_solution.value, inference_config.filter_threshold)
            workflow_log.info("inference pipeline: %d / %d remain", int(keep.sum()), int(keep.shape[0]))
            filtered_likelihood = fit_state.likelihood_matrix[:, keep]
            filtered_prior = baseline_prior[keep]

            workflow_log.info("inference pipeline: fitting refit grid")
            refit_solution = _run_refit_grid(
                filtered_likelihood,
                maf_masks,
                filtered_baseline,
                inference_config,
                workflow_log,
                prior=filtered_prior,
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
                        "component_keep_mask": keep.tolist(),
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
