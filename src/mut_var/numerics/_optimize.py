from __future__ import annotations

# pattern: Functional Core
from typing import Any, Callable, NamedTuple

import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import ArrayLike

from mut_var.contracts import RESULTS
from mut_var.numerics._solver_utils import is_nonfinite, should_backtrack


class OptimizationLoopConfig(NamedTuple):
    max_iter: int
    tol: float
    step_size: float
    max_backtracks: int = 1


class OptimizationLoopResult(NamedTuple):
    params: Any
    objective: ArrayLike
    epoch_count: int
    converged: bool
    result: RESULTS
    step_size: float
    key: rdm.PRNGKey | None


def _constant_step_size(_epoch: int, base_step_size: float) -> float:
    return base_step_size


def _absolute_progress(diff: ArrayLike, _objective: ArrayLike) -> ArrayLike:
    return diff


def run_iterative_optimization(
    *,
    init_params: Any,
    init_objective: ArrayLike,
    key: rdm.PRNGKey | None,
    config: OptimizationLoopConfig,
    make_epoch_context: Callable[[int, Any, rdm.PRNGKey | None], tuple[Any, rdm.PRNGKey | None]],
    compute_direction: Callable[[Any, Any], Any],
    propose_candidate: Callable[[Any, Any, float], Any],
    evaluate_objective: Callable[[Any, Any], ArrayLike],
    step_size_for_epoch: Callable[[int, float], float] = _constant_step_size,
    should_backtrack_step: Callable[[ArrayLike, ArrayLike], bool] = should_backtrack,
    progress_metric: Callable[[ArrayLike, ArrayLike], ArrayLike] = _absolute_progress,
) -> OptimizationLoopResult:
    params = init_params
    objective = jnp.asarray(init_objective)
    rng_key = key
    epochs = 0
    converged = False
    diff = jnp.asarray(0.0)
    last_step_size = float(config.step_size)
    max_backtracks = max(1, int(config.max_backtracks))

    for epoch in range(config.max_iter):
        epochs = epoch + 1
        context, rng_key = make_epoch_context(epoch, params, rng_key)
        direction = compute_direction(params, context)

        step_size = float(step_size_for_epoch(epoch, config.step_size))
        accepted = False
        candidate = params
        candidate_objective = objective

        for _ in range(max_backtracks):
            candidate = propose_candidate(params, direction, step_size)
            candidate_objective = evaluate_objective(candidate, context)
            diff = jnp.asarray(candidate_objective) - objective
            if should_backtrack_step(diff, candidate_objective):
                step_size *= 0.5
                continue
            accepted = True
            break

        last_step_size = step_size
        if accepted:
            params = candidate
            objective = jnp.asarray(candidate_objective)

        if is_nonfinite(objective):
            return OptimizationLoopResult(
                params=params,
                objective=objective,
                epoch_count=epochs,
                converged=False,
                result=RESULTS.nonfinite_objective,
                step_size=last_step_size,
                key=rng_key,
            )

        progress = jnp.asarray(progress_metric(diff, objective))
        if bool(jnp.abs(progress) < config.tol):
            converged = True
            break

    result = RESULTS.successful if converged else RESULTS.max_steps_reached
    return OptimizationLoopResult(
        params=params,
        objective=objective,
        epoch_count=epochs,
        converged=converged,
        result=result,
        step_size=last_step_size,
        key=rng_key,
    )
