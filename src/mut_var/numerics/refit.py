from __future__ import annotations

# pattern: Functional Core
from time import perf_counter
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp
import optimistix as optx

from jax.scipy.special import xlogy
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics._optimistix_solver import map_optimistix_result, MutVarSolver
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_recoverable_result,
    simplex_tangent_direction,
)
from mut_var.numerics.baseline import Params


class RefitConfig(NamedTuple):
    penalty: float = 1.0
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


pdf = jax.vmap(_pdf, (None, None, 0, 0), 1)


def penalized_objective(
    param: Params,
    likelihoods: ArrayLike,
    weights: ArrayLike,
    alpha: ArrayLike,
    baseline_param: Params,
    penalty: ArrayLike,
):
    r"""Evaluate penalized refit objective over threshold-weighted likelihoods."""
    mixture_pdf = jnp.clip(likelihoods @ param.pi, min=jnp.finfo(float).tiny)
    obj_term = jnp.sum(weights * jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))

    p1 = jnp.sum(nn.relu(baseline_param.pi[1:] * param.pi[:-1] - baseline_param.pi[:-1] * param.pi[1:]))
    rel_point_mass_dist = nn.relu(baseline_param.pi[0] - param.pi[0])

    penalty_term = penalty * (p1 + rel_point_mass_dist)
    return obj_term - log_penalty - penalty_term


def _fit_single_refit(
    likelihoods: ArrayLike,
    weights: ArrayLike,
    init: Params,
    config: RefitConfig,
    obj,
) -> Solution:
    likelihoods_arr = jnp.asarray(likelihoods)
    weights_arr = jnp.asarray(weights)

    if likelihoods_arr.ndim != 2:
        return Solution(
            value=init,
            result=RESULTS.invalid_input,
            stats={"reason": "likelihoods must be a 2D array"},
            state=None,
        )

    if weights_arr.ndim != 1 or weights_arr.shape[0] != likelihoods_arr.shape[0]:
        return Solution(
            value=init,
            result=RESULTS.invalid_input,
            stats={"reason": "weights must be 1D and aligned with likelihood rows"},
            state=None,
        )

    if not bool(jnp.isfinite(likelihoods_arr).all()):
        return Solution(
            value=init,
            result=RESULTS.invalid_input,
            stats={"reason": "likelihoods contain non-finite values"},
            state=None,
        )

    if int(jnp.sum(weights_arr > 0.0)) == 0:
        return Solution(
            value=init,
            result=RESULTS.empty_subset,
            stats={"reason": "all threshold weights are zero"},
            state=None,
        )

    alpha = jnp.array([10.0] + (len(init.pi) - 1) * [1.0])
    params = Params(jnp.asarray(init.pi), jnp.asarray(init.mu_k), jnp.asarray(init.var_k))
    return _fit_single_refit_optimistix(
        likelihoods=likelihoods_arr,
        weights=weights_arr,
        alpha=alpha,
        init=params,
        objective=obj,
        config=config,
    )


def _fit_single_refit_optimistix(
    *,
    likelihoods: ArrayLike,
    weights: ArrayLike,
    alpha: ArrayLike,
    init: Params,
    objective,
    config: RefitConfig,
) -> Solution:
    def _propose_candidate(params_now: Params, direction: Params, step_size: ArrayLike) -> Params:
        tangent_pi = simplex_tangent_direction(params_now.pi, direction.pi)
        pi = exponential_map_simplex(params_now.pi, tangent_pi, step_size)
        return Params(pi, params_now.mu_k, params_now.var_k)

    def _neg_objective(params_now: Params, _unused: Any) -> ArrayLike:
        return -objective(params_now, likelihoods, weights, alpha, init, config.penalty)

    solver = MutVarSolver(
        step_update=_propose_candidate,
        step_size=config.step_size,
        rtol=config.tol,
        atol=config.tol,
    )
    optx_solution = optx.minimise(
        fn=_neg_objective,
        solver=solver,
        y0=init,
        args=None,
        max_steps=config.max_iter,
        throw=False,
    )

    objective_value = objective(optx_solution.value, likelihoods, weights, alpha, init, config.penalty)
    mapped_result = map_optimistix_result(optx_solution.result)
    if not bool(jnp.isfinite(objective_value)):
        return Solution(
            value=optx_solution.value,
            result=RESULTS.nonfinite_objective,
            stats={
                "epoch": int(optx_solution.stats.get("num_steps", 0)),
                "objective": float(objective_value),
            },
            state=None,
        )

    return Solution(
        value=optx_solution.value,
        result=mapped_result,
        stats={
            "epoch_count": int(optx_solution.stats.get("num_steps", 0)),
            "objective": float(objective_value),
            "converged": mapped_result == RESULTS.successful,
            "n_obs_used": int(jnp.sum(weights > 0.0)),
            "likelihood_shape": tuple(int(x) for x in jnp.asarray(likelihoods).shape),
        },
        state=None,
    )


def fit_refit_grid(
    beta_hat: ArrayLike,
    s2: ArrayLike,
    maf_masks: ArrayLike,
    init: Params,
    config: RefitConfig,
) -> Solution:
    r"""Sequentially refit mixture weights across MAF-threshold masks.

    **Arguments:**
    - `beta_hat`: 1D effect-size estimates.
    - `s2`: 1D positive variances aligned with `beta_hat`.
    - `maf_masks`: 2D boolean masks selecting observations per threshold.
    - `init`: Initial parameters from baseline fit.
    - `config`: Refit solver controls.

    **Returns:**
    - `Solution` with one fitted model per threshold (plus initial model).

    **Failure Modes:**
    - `RESULTS.invalid_input` for shape/mask mismatches.
    - `RESULTS.empty_subset` for empty threshold slices.
    - `RESULTS.nonfinite_objective` for non-finite objective values.
    - `RESULTS.max_steps_reached` when any threshold solve does not converge in time.
    """
    beta_hat_arr = jnp.asarray(beta_hat)
    s2_arr = jnp.asarray(s2)
    masks_arr = jnp.asarray(maf_masks, dtype=bool)

    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1 or beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays of equal length"},
            state=None,
        )

    if masks_arr.ndim != 2 or masks_arr.shape[1] != beta_hat_arr.shape[0]:
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "maf_masks must be a 2D mask array over observations"},
            state=None,
        )

    obj = eqx.filter_jit(penalized_objective)

    models = [init]
    any_max_steps = False
    threshold_diagnostics: list[dict[str, int | float | tuple[int, int]]] = []
    total_elapsed = 0.0

    for idx in range(masks_arr.shape[0]):
        weights = masks_arr[idx].astype(jnp.float64)
        n_obs = int(jnp.sum(weights))
        if n_obs == 0:
            return Solution(
                value=models,
                result=RESULTS.empty_subset,
                stats={"reason": f"maf mask at index {idx} is empty"},
                state=None,
            )

        mu_k = jnp.pad(models[-1].mu_k, (1, 0))
        var_k = jnp.pad(models[-1].var_k, (1, 0))
        likelihoods = pdf(beta_hat_arr, s2_arr, mu_k, var_k)

        start = perf_counter()
        fit_solution = _fit_single_refit(likelihoods, weights, models[-1], config, obj)
        elapsed = perf_counter() - start
        total_elapsed += elapsed

        if not is_recoverable_result(fit_solution.result):
            return Solution(
                value=models,
                result=fit_solution.result,
                stats=fit_solution.stats,
                state=fit_solution.state,
            )
        if fit_solution.result == RESULTS.max_steps_reached:
            any_max_steps = True

        diag = dict(fit_solution.stats)
        diag["threshold_index"] = idx
        diag["elapsed_seconds"] = elapsed
        threshold_diagnostics.append(diag)

        models.append(fit_solution.value)

    result = RESULTS.max_steps_reached if any_max_steps else RESULTS.successful
    return Solution(
        value=models,
        result=result,
        stats={
            "num_models": len(models),
            "num_thresholds": int(masks_arr.shape[0]),
            "total_refit_seconds": total_elapsed,
            "threshold_diagnostics": threshold_diagnostics,
        },
        state=None,
    )
