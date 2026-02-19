from __future__ import annotations

# pattern: Functional Core
from time import perf_counter
from typing import NamedTuple

import jax
import jax.nn as nn
import jax.numpy as jnp

from jax.scipy.special import xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics.baseline import Params


class RefitConfig(NamedTuple):
    penalty: float = 1.0
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


pdf = jax.vmap(_pdf, (None, None, 0, 0), 1)


def _exponential_map_simplex(
    pi: Array,
    direction: Array,
    step_size: float,
) -> Array:
    s = jnp.sqrt(jnp.sum(direction**2) / pi)
    c = jnp.cos(0.5 * step_size * s)
    s2 = jnp.sin(0.5 * step_size * s)

    phi = jnp.sqrt(pi)
    step = (direction / (s * phi)) * s2
    phi_new = phi * c + step
    pi_new = phi_new**2

    return pi_new / jnp.sum(pi_new)


def penalized_objective(
    param: Params,
    likelihoods: ArrayLike,
    weights: ArrayLike,
    alpha: ArrayLike,
    baseline_param: Params,
    penalty: ArrayLike,
):
    mixture_pdf = jnp.clip(likelihoods @ param.pi, min=jnp.finfo(float).tiny)
    obj_term = jnp.sum(weights * jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))

    p1 = jnp.sum(
        nn.relu(
            baseline_param.pi[1:] * param.pi[:-1] - baseline_param.pi[:-1] * param.pi[1:]
        )
    )
    rel_point_mass_dist = nn.relu(baseline_param.pi[0] - param.pi[0])

    penalty_term = penalty * (p1 + rel_point_mass_dist)
    return obj_term - log_penalty - penalty_term


def _fit_single_refit(
    likelihoods: ArrayLike,
    weights: ArrayLike,
    init: Params,
    config: RefitConfig,
    vg_f,
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

    ologlike = -1e10
    start = config.step_size
    converged = False
    epochs = 0
    pi = params.pi

    for epoch in range(config.max_iter):
        epochs = epoch + 1
        _, direction = vg_f(params, likelihoods_arr, weights_arr, alpha, init, config.penalty)
        step_size = start
        for _ in range(20):
            tangent_pi = params.pi * (direction.pi - (direction.pi @ params.pi))
            pi = _exponential_map_simplex(params.pi, tangent_pi, step_size)
            candidate = Params(pi, params.mu_k, params.var_k)
            nloglike = obj(candidate, likelihoods_arr, weights_arr, alpha, init, config.penalty)
            diff = nloglike - ologlike
            if bool(diff < 0) or bool(jnp.isnan(nloglike)) or bool(jnp.isinf(nloglike)):
                step_size *= 0.5
            else:
                params = candidate
                ologlike = nloglike
                break

        if bool(jnp.isnan(ologlike)) or bool(jnp.isinf(ologlike)):
            return Solution(
                value=params,
                result=RESULTS.nonfinite_objective,
                stats={"epoch": epochs, "objective": float(ologlike)},
                state={"step_size": float(step_size)},
            )

        if bool(jnp.abs(diff) < config.tol):
            converged = True
            break

    result = RESULTS.successful if converged else RESULTS.max_steps_reached
    return Solution(
        value=Params(pi, init.mu_k, init.var_k),
        result=result,
        stats={
            "epoch_count": epochs,
            "objective": float(ologlike),
            "converged": converged,
            "n_obs_used": int(jnp.sum(weights_arr > 0.0)),
            "likelihood_shape": tuple(int(x) for x in likelihoods_arr.shape),
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

    vg_f = jax.jit(jax.value_and_grad(penalized_objective))
    obj = jax.jit(penalized_objective)

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
        fit_solution = _fit_single_refit(likelihoods, weights, models[-1], config, vg_f, obj)
        elapsed = perf_counter() - start
        total_elapsed += elapsed

        if fit_solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
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
