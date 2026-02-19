from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as rdm

from jax.scipy.special import logsumexp, xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.contracts import RESULTS, Solution


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


@dataclass(frozen=True, slots=True)
class BaselineConfig:
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


def _reimannian_step(
    params: Params,
    direction: Params,
    step_size: float,
) -> Params:
    tangent_pi = params.pi * (direction.pi - (direction.pi @ params.pi))
    pi = _exponential_map_simplex(params.pi, tangent_pi, step_size)

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

    if bool(~jnp.isfinite(max_val)) or bool(max_val <= 0.0):
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

    ologlike = -1e10
    init_ss = config.step_size
    nobs = len(beta_hat_arr)
    using_sgd = config.batch_size < nobs
    converged = False
    epochs = 0

    for epoch in range(config.max_iter):
        epochs = epoch + 1
        step_size = jnp.power(init_ss, epoch)
        if using_sgd:
            key, skey = rdm.split(key)
            idxs = rdm.choice(skey, nobs, shape=(config.batch_size,), replace=False)
            _, direction = vg_f(params, beta_hat_arr[idxs], s2_arr[idxs], alpha)
            direction = jax.tree.map(lambda x: (nobs / config.batch_size) * x, direction)
            params = _reimannian_step(params, direction, step_size)
            nloglike = obj(params, beta_hat_arr[idxs], s2_arr[idxs], alpha)
            diff = nloglike - ologlike
            ologlike = nloglike
        else:
            _, direction = vg_f(params, beta_hat_arr, s2_arr, alpha)
            for inner in range(20):
                candidate = _reimannian_step(params, direction, step_size)
                nloglike = obj(candidate, beta_hat_arr, s2_arr, alpha)
                diff = nloglike - ologlike
                if bool(diff < 0) or bool(jnp.isnan(nloglike)) or bool(jnp.isinf(nloglike)):
                    step_size *= 0.5
                else:
                    params = candidate
                    ologlike = nloglike
                    break

        rel_diff = diff / (jnp.abs(ologlike) + 1e-12)
        if bool(jnp.isnan(ologlike)) or bool(jnp.isinf(ologlike)):
            return Solution(
                value=params,
                result=RESULTS.nonfinite_objective,
                stats={"epoch": epochs, "objective": float(ologlike)},
                state={"step_size": float(step_size)},
            )

        if bool(jnp.abs(rel_diff) < config.tol):
            converged = True
            break

    result = RESULTS.successful if converged else RESULTS.max_steps_reached
    return Solution(
        value=params,
        result=result,
        stats={
            "epoch_count": epochs,
            "objective": float(ologlike),
            "converged": converged,
            "num_observations": int(nobs),
        },
        state=None,
    )
