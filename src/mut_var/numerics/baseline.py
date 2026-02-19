from __future__ import annotations

# pattern: Functional Core
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as rdm

from jax.scipy.special import logsumexp, xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics._optimize import OptimizationLoopConfig, run_iterative_optimization
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_nonfinite,
    MAX_BACKTRACK_STEPS,
    should_backtrack,
    simplex_tangent_direction,
)


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


class BaselineConfig(NamedTuple):
    num_clusters: int
    batch_size: int = 10_000
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01


def _logpdf(beta_hat, s2, mean, var_k):
    return norm.logpdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


logpdf = jax.vmap(_logpdf, (None, None, 0, 0), 1)
pdf = jax.vmap(_pdf, (None, None, 0, 0), 1)


def baseline_objective(
    param: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
):
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))
    pi = param.pi
    log_likelihood = jnp.sum(
        jnp.log(
            pdf(beta_hat, s2, param.mu_k, param.var_k) @ pi[1:]
            + _pdf(beta_hat, s2, 0.0, 0.0) * pi[0]
        )
    )
    return log_likelihood - log_penalty


def baseline_objective_lse(
    param: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
):
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))
    pi = param.pi
    log_likes = jnp.concatenate(
        (
            _logpdf(beta_hat, s2, 0.0, 0.0)[:, jnp.newaxis],
            logpdf(beta_hat, s2, param.mu_k, param.var_k),
        ),
        axis=1,
    )
    lse = logsumexp(log_likes, axis=1, b=pi)
    log_likelihood = jnp.sum(lse)
    return log_likelihood - log_penalty


def _fix_var(var):
    eps = jnp.finfo(float).eps
    inf = ~jnp.isfinite(var)
    zs = var == 0.0
    return jnp.where(jnp.logical_or(inf, zs), eps, var)


def _exponential_map_normal(
    mu0: Array,
    v0: Array,
    mu_direction: Array,
    v_direction: Array,
    step_size: float,
) -> tuple[Array, Array]:
    std_dev = jnp.sqrt(v0)
    theta = jnp.arctan2(v_direction / jnp.sqrt(2.0), mu_direction / std_dev)

    a = step_size / jnp.sqrt(2.0)
    tanh_a = jnp.tanh(a)
    denom = 1.0 - jnp.sin(theta) * tanh_a

    mu_step = jnp.sqrt(2.0) * std_dev * jnp.cos(theta) * tanh_a / denom
    mu = jnp.where(jnp.isnan(mu_step), mu0, mu0 + mu_step)

    denom_sq = jnp.square(jnp.cosh(a) * denom)
    v = _fix_var(v0 / denom_sq)

    return mu, v


def _riemannian_step(
    params: Params,
    direction: Params,
    step_size: float,
) -> Params:
    tangent_pi = simplex_tangent_direction(params.pi, direction.pi)
    pi = exponential_map_simplex(params.pi, tangent_pi, step_size)

    tangent_var_k = 2 * direction.var_k * params.var_k**2
    tangent_mu_k = direction.mu_k * params.var_k
    mu_k, var_k = _exponential_map_normal(
        params.mu_k,
        params.var_k,
        tangent_mu_k,
        tangent_var_k,
        step_size,
    )

    return Params(pi, mu_k, var_k)


def _validate_inputs(beta_hat: ArrayLike, s2: ArrayLike, config: BaselineConfig) -> Solution | None:
    if hasattr(beta_hat, "columns") or hasattr(s2, "columns"):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "baseline kernel expects arrays, not tabular objects"},
            state=None,
        )

    if config.num_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be >= 2"},
            state=None,
        )

    try:
        beta_hat_arr = jnp.asarray(beta_hat)
        s2_arr = jnp.asarray(s2)
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"failed to convert inputs to arrays: {exc}"},
            state=None,
        )

    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1 or beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays of equal length"},
            state=None,
        )

    if beta_hat_arr.shape[0] == 0:
        return Solution(value=None, result=RESULTS.empty_subset, stats={"reason": "no variants available"}, state=None)

    if not bool(jnp.isfinite(beta_hat_arr).all()) or not bool(jnp.isfinite(s2_arr).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "inputs contain non-finite values"},
            state=None,
        )

    if bool((s2_arr <= 0.0).any()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "s2 must be strictly positive"},
            state=None,
        )

    return None


def fit_baseline(
    beta_hat: ArrayLike,
    s2: ArrayLike,
    key: rdm.PRNGKey,
    config: BaselineConfig,
) -> Solution:
    invalid = _validate_inputs(beta_hat, s2, config)
    if invalid is not None:
        return invalid

    beta_hat_arr = jnp.asarray(beta_hat)
    s2_arr = jnp.asarray(s2)

    alpha = jnp.array([10.0] + (config.num_clusters - 1) * [1.0])
    std_err = jnp.sqrt(s2_arr)
    min_val = jnp.min(std_err) / 10
    max_val = jnp.max(beta_hat_arr**2 - s2_arr)
    if max_val < 0.0:
        max_val = 8 * min_val
    else:
        max_val = 2 * jnp.sqrt(max_val)

    if is_nonfinite(max_val) or bool(max_val <= 0.0):
        max_val = 8 * min_val

    params = Params(
        pi=rdm.dirichlet(key, alpha),
        mu_k=jnp.zeros(config.num_clusters - 1),
        var_k=jnp.exp(
            jnp.linspace(jnp.log(min_val), jnp.log(max_val), config.num_clusters - 1)
        )
        ** 2,
    )

    vg_f = jax.jit(jax.value_and_grad(baseline_objective))
    obj = jax.jit(baseline_objective)
    nobs = len(beta_hat_arr)
    using_sgd = config.batch_size < nobs

    def _make_epoch_context(
        _epoch: int,
        _params: Params,
        rng_key: rdm.PRNGKey | None,
    ) -> tuple[tuple[ArrayLike, ArrayLike], rdm.PRNGKey | None]:
        if not using_sgd:
            return (beta_hat_arr, s2_arr), rng_key

        if rng_key is None:
            raise ValueError("sgd mode requires a PRNGKey")
        next_key, sample_key = rdm.split(rng_key)
        idxs = rdm.choice(sample_key, nobs, shape=(config.batch_size,), replace=False)
        return (beta_hat_arr[idxs], s2_arr[idxs]), next_key

    def _compute_direction(params_now: Params, epoch_context: tuple[ArrayLike, ArrayLike]) -> Params:
        beta_now, s2_now = epoch_context
        _, direction = vg_f(params_now, beta_now, s2_now, alpha)
        if using_sgd:
            scale = nobs / config.batch_size
            direction = jax.tree.map(lambda x: scale * x, direction)
        return direction

    def _evaluate_objective(params_now: Params, epoch_context: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
        beta_now, s2_now = epoch_context
        return obj(params_now, beta_now, s2_now, alpha)

    loop_solution = run_iterative_optimization(
        init_params=params,
        init_objective=jnp.asarray(-1e10),
        key=key if using_sgd else None,
        config=OptimizationLoopConfig(
            max_iter=config.max_iter,
            tol=config.tol,
            step_size=config.step_size,
            max_backtracks=1 if using_sgd else MAX_BACKTRACK_STEPS,
        ),
        make_epoch_context=_make_epoch_context,
        compute_direction=_compute_direction,
        propose_candidate=_riemannian_step,
        evaluate_objective=_evaluate_objective,
        step_size_for_epoch=lambda epoch, base_step: float(jnp.power(base_step, epoch)),
        should_backtrack_step=(lambda _diff, _objective: False) if using_sgd else should_backtrack,
        progress_metric=lambda diff, objective: diff / (jnp.abs(objective) + 1e-12),
    )

    if loop_solution.result == RESULTS.nonfinite_objective:
        return Solution(
            value=loop_solution.params,
            result=RESULTS.nonfinite_objective,
            stats={"epoch": loop_solution.epoch_count, "objective": float(loop_solution.objective)},
            state={"step_size": float(loop_solution.step_size)},
        )

    result = loop_solution.result
    return Solution(
        value=loop_solution.params,
        result=result,
        stats={
            "epoch_count": loop_solution.epoch_count,
            "objective": float(loop_solution.objective),
            "converged": loop_solution.converged,
            "num_observations": int(nobs),
        },
        state=None,
    )
