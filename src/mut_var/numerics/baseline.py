from __future__ import annotations

# pattern: Functional Core
from typing import Any, Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import optimistix as optx

from jax.scipy.special import logsumexp, xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics._optimistix_solver import map_optimistix_result, MutVarSolver
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_nonfinite,
    simplex_tangent_direction,
)


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


class BaselineConfig(NamedTuple):
    num_clusters: int
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
    r"""Evaluate the baseline penalized log-likelihood objective.

    **Arguments:**

    - `param`: Current mixture parameters.
    - `beta_hat`: Observed effect-size estimates.
    - `s2`: Observation variances.
    - `alpha`: Dirichlet prior concentration vector for mixture weights.

    **Returns:**

    - Scalar objective value to maximize.
    """
    log_penalty = jnp.sum(xlogy(alpha - 1, param.pi))
    pi = param.pi
    log_likelihood = jnp.sum(
        jnp.log(pdf(beta_hat, s2, param.mu_k, param.var_k) @ pi[1:] + _pdf(beta_hat, s2, 0.0, 0.0) * pi[0])
    )
    return log_likelihood - log_penalty


def baseline_objective_lse(
    param: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
):
    r"""Numerically stable baseline objective variant using log-sum-exp."""
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
    verbose: bool | Callable[..., None] = False,
) -> Solution:
    r"""Fit baseline mixture parameters with Optimistix full-batch descent.

    **Arguments:**

    - `beta_hat`: 1D effect-size estimates.
    - `s2`: 1D positive variances aligned with `beta_hat`.
    - `key`: JAX PRNG key used for simplex initialization.
    - `config`: Baseline solver controls.

    **Returns:**

    - `Solution` carrying fitted `Params` and status diagnostics.

    **Failure Modes:**

    - `RESULTS.invalid_input` for shape/domain violations.
    - `RESULTS.empty_subset` for empty arrays.
    - `RESULTS.nonfinite_objective` when objective evaluation is non-finite.
    - `RESULTS.max_steps_reached` when convergence tolerance is not met.
    """
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
        var_k=jnp.exp(jnp.linspace(jnp.log(min_val), jnp.log(max_val), config.num_clusters - 1)) ** 2,
    )

    obj = eqx.filter_jit(baseline_objective)
    nobs = len(beta_hat_arr)
    return _fit_baseline_optimistix(
        init=params,
        beta_hat=beta_hat_arr,
        s2=s2_arr,
        alpha=alpha,
        objective=obj,
        nobs=nobs,
        config=config,
        verbose=verbose,
    )


def _fit_baseline_optimistix(
    *,
    init: Params,
    beta_hat: ArrayLike,
    s2: ArrayLike,
    alpha: ArrayLike,
    objective,
    nobs: int,
    config: BaselineConfig,
    verbose: bool | Callable[..., None] = False,
) -> Solution:
    def _propose_candidate(params_now: Params, direction: Params, step_size: ArrayLike) -> Params:
        tangent_pi = simplex_tangent_direction(params_now.pi, direction.pi)
        pi = exponential_map_simplex(params_now.pi, tangent_pi, step_size)
        return Params(pi, params_now.mu_k, params_now.var_k)

    def _neg_objective(params_now: Params, _unused: Any) -> ArrayLike:
        return -objective(params_now, beta_hat, s2, alpha)

    solver = MutVarSolver(
        step_update=_propose_candidate,
        step_size=config.step_size,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )
    optx_solution = optx.minimise(
        fn=_neg_objective,
        solver=solver,
        y0=init,
        args=None,
        max_steps=config.max_iter,
        throw=False,
    )

    objective_value = objective(optx_solution.value, beta_hat, s2, alpha)
    if is_nonfinite(objective_value):
        return Solution(
            value=optx_solution.value,
            result=RESULTS.nonfinite_objective,
            stats={
                "epoch": int(optx_solution.stats.get("num_steps", 0)),
                "objective": float(objective_value),
            },
            state=None,
        )

    mapped_result = map_optimistix_result(optx_solution.result)
    return Solution(
        value=optx_solution.value,
        result=mapped_result,
        stats={
            "epoch_count": int(optx_solution.stats.get("num_steps", 0)),
            "objective": float(objective_value),
            "converged": mapped_result == RESULTS.successful,
            "num_observations": int(nobs),
            "used_full_batch_objective": True,
        },
        state=None,
    )
