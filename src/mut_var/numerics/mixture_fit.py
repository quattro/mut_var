from __future__ import annotations

# pattern: Functional Core
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.nn as nn
import jax.numpy as jnp
import optimistix as optx

from jax import core as jax_core
from jax.scipy.special import xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.numerics._optimistix_solver import map_optimistix_result, MutVarSolver
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_nonfinite,
    simplex_tangent_direction,
)
from mut_var.types import InferenceConfig, RESULTS, Solution

_DEFAULT_STEP_SIZE: float = 0.01
_DEFAULT_PENALTY: float = 1.0


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


class FitState(NamedTuple):
    likelihood_matrix: Array
    initial_params: Params


def _is_tracing(*values: object) -> bool:
    leaves = []
    for value in values:
        leaves.extend(jax.tree.leaves(value))
    return any(isinstance(leaf, jax_core.Tracer) for leaf in leaves)


def _logpdf(beta_hat, s2, mean, var_k):
    return norm.logpdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


_pdf_components = jax.vmap(_pdf, (None, None, 0, 0), 1)


def _compute_likelihood_matrix(
    beta_hat: Array,
    s2: Array,
    mu_k: Array,
    var_k: Array,
) -> Array:
    """Build the cached likelihood matrix used by the mixture fit."""
    null_col = _pdf(beta_hat, s2, 0.0, 0.0)[:, jnp.newaxis]
    other_cols = _pdf_components(beta_hat, s2, mu_k, var_k)
    return jnp.concatenate([null_col, other_cols], axis=1)


def _validate_inputs(beta_hat: ArrayLike, s2: ArrayLike, config: InferenceConfig) -> Solution | None:
    beta_hat_arr = jnp.asarray(beta_hat)
    s2_arr = jnp.asarray(s2)

    if config.num_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be at least 2"},
            state=None,
        )
    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays"},
            state=None,
        )
    if beta_hat_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "input arrays are empty"},
            state=None,
        )
    if beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must have the same length"},
            state=None,
        )
    if not bool(jnp.isfinite(beta_hat_arr).all()) or not bool(jnp.isfinite(s2_arr).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be finite"},
            state=None,
        )
    if not bool((s2_arr > 0.0).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "s2 must be strictly positive"},
            state=None,
        )
    return None


def _prepare_fit_state_traced(
    beta_hat: ArrayLike,
    s2: ArrayLike,
    config: InferenceConfig,
) -> Solution:
    beta_hat_arr = jnp.asarray(beta_hat, dtype=jnp.float64)
    s2_arr = jnp.asarray(s2, dtype=jnp.float64)
    invalid = jnp.logical_or(
        jnp.logical_not(jnp.isfinite(beta_hat_arr).all()),
        jnp.logical_not(jnp.isfinite(s2_arr).all()),
    )
    invalid = jnp.logical_or(invalid, jnp.logical_not((s2_arr > 0.0).all()))

    def _invalid(_):
        k = config.num_clusters
        n = beta_hat_arr.shape[0]
        dummy_state = FitState(
            likelihood_matrix=jnp.zeros((n, k), dtype=jnp.float64),
            initial_params=Params(
                pi=jnp.zeros(k, dtype=jnp.float64),
                mu_k=jnp.zeros(k - 1, dtype=jnp.float64),
                var_k=jnp.zeros(k - 1, dtype=jnp.float64),
            ),
        )
        return Solution(value=dummy_state, result=RESULTS.invalid_input, stats=None, state=None)

    def _valid(_):
        std_err = jnp.sqrt(s2_arr)
        min_val = jnp.min(std_err) / 10.0
        max_val = jnp.max(beta_hat_arr**2 - s2_arr)
        max_val = jnp.where(max_val < 0.0, 8.0 * min_val, 2.0 * jnp.sqrt(max_val))
        max_val = jnp.where(~jnp.isfinite(max_val) | (max_val <= 0.0), 8.0 * min_val, max_val)

        k = config.num_clusters
        mu_k = jnp.zeros(k - 1, dtype=jnp.float64)
        var_k = jnp.exp(jnp.linspace(jnp.log(min_val), jnp.log(max_val), k - 1)) ** 2
        likelihood_matrix = _compute_likelihood_matrix(beta_hat_arr, s2_arr, mu_k, var_k)
        pi = jnp.ones(k, dtype=jnp.float64) / k
        state = FitState(
            likelihood_matrix=likelihood_matrix,
            initial_params=Params(pi=pi, mu_k=mu_k, var_k=var_k),
        )
        return Solution(value=state, result=RESULTS.successful, stats=None, state=None)

    return jax.lax.cond(invalid, _invalid, _valid, operand=None)


def _fit_baseline_traced(
    state: FitState,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    L = state.likelihood_matrix
    init_params = state.initial_params
    pi_init = jnp.asarray(init_params.pi, dtype=jnp.float64)

    if L.ndim != 2:
        return Solution(
            value=init_params,
            result=RESULTS.invalid_input,
            stats=None,
            state=None,
        )
    if pi_init.ndim != 1:
        return Solution(
            value=init_params,
            result=RESULTS.invalid_input,
            stats=None,
            state=None,
        )
    if L.shape[1] != pi_init.shape[0]:
        return Solution(
            value=init_params,
            result=RESULTS.invalid_input,
            stats=None,
            state=None,
        )

    invalid = jnp.logical_or(
        jnp.logical_not(jnp.isfinite(L).all()),
        jnp.logical_not(jnp.isfinite(pi_init).all()),
    )
    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)
    obj = eqx.filter_jit(_baseline_objective)

    def _invalid(_):
        return Solution(
            value=init_params,
            result=RESULTS.invalid_input,
            stats=None,
            state=None,
        )

    def _valid(_):
        def _neg_obj(pi, _args):
            val = obj(pi, L, alpha)
            return jnp.where(jnp.isfinite(val), val, jnp.inf)

        solver = MutVarSolver(
            step_update=_pi_step,
            step_size=_DEFAULT_STEP_SIZE,
            rtol=config.tol,
            atol=config.tol,
            verbose=verbose,
        )
        optx_solution = optx.minimise(
            fn=_neg_obj,
            solver=solver,
            y0=pi_init,
            args=None,
            max_steps=config.max_iter,
            throw=False,
        )
        result = map_optimistix_result(optx_solution.result)
        pi_opt = optx_solution.value
        params = Params(pi=pi_opt, mu_k=init_params.mu_k, var_k=init_params.var_k)
        return Solution(value=params, result=result, stats=None, state=None)

    return jax.lax.cond(invalid, _invalid, _valid, operand=None)


def _fit_refit_step_traced(
    L_sub: ArrayLike,
    prev_params: Params,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    L_arr = jnp.asarray(L_sub, dtype=jnp.float64)
    pi_init = jnp.asarray(prev_params.pi, dtype=jnp.float64)

    if L_arr.ndim != 2:
        return Solution(value=prev_params, result=RESULTS.invalid_input, stats=None, state=None)
    if L_arr.shape[0] == 0:
        return Solution(value=prev_params, result=RESULTS.empty_subset, stats=None, state=None)
    if pi_init.ndim != 1:
        return Solution(value=prev_params, result=RESULTS.invalid_input, stats=None, state=None)
    if L_arr.shape[1] != pi_init.shape[0]:
        return Solution(value=prev_params, result=RESULTS.invalid_input, stats=None, state=None)

    invalid = jnp.logical_not(jnp.isfinite(L_arr).all())
    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)
    obj = eqx.filter_jit(_refit_objective)

    def _invalid(_):
        return Solution(value=prev_params, result=RESULTS.invalid_input, stats=None, state=None)

    def _valid(_):
        def _neg_obj(pi, _args):
            val = obj(pi, L_arr, pi_init, alpha, _DEFAULT_PENALTY)
            return jnp.where(jnp.isfinite(val), val, jnp.inf)

        solver = MutVarSolver(
            step_update=_pi_step,
            step_size=_DEFAULT_STEP_SIZE,
            rtol=config.tol,
            atol=config.tol,
            verbose=verbose,
        )
        optx_solution = optx.minimise(
            fn=_neg_obj,
            solver=solver,
            y0=pi_init,
            args=None,
            max_steps=config.max_iter,
            throw=False,
        )
        result = map_optimistix_result(optx_solution.result)
        pi_opt = optx_solution.value
        params = Params(pi=pi_opt, mu_k=prev_params.mu_k, var_k=prev_params.var_k)
        return Solution(value=params, result=result, stats=None, state=None)

    return jax.lax.cond(invalid, _invalid, _valid, operand=None)


def prepare_fit_state(
    beta_hat: ArrayLike,
    s2: ArrayLike,
    config: InferenceConfig,
) -> Solution:
    r"""Validate inputs, build the fixed grid, and pre-compute the likelihood matrix.

    **Arguments:**

    - `beta_hat`: 1D array of effect-size estimates.
    - `s2`: 1D array of observation variances (must be strictly positive).
    - `config`: Inference configuration.

    **Returns:**

    - `Solution` with `FitState` on success. Status codes:
      `RESULTS.successful`, `RESULTS.invalid_input`, `RESULTS.empty_subset`.
    """
    if _is_tracing(beta_hat, s2):
        return _prepare_fit_state_traced(beta_hat, s2, config)

    invalid = _validate_inputs(beta_hat, s2, config)
    if invalid is not None:
        return invalid

    beta_hat_arr = jnp.asarray(beta_hat, dtype=jnp.float64)
    s2_arr = jnp.asarray(s2, dtype=jnp.float64)

    std_err = jnp.sqrt(s2_arr)
    min_val = jnp.min(std_err) / 10.0
    max_val = jnp.max(beta_hat_arr**2 - s2_arr)
    if max_val < 0.0:
        max_val = 8.0 * min_val
    else:
        max_val = 2.0 * jnp.sqrt(max_val)
    if is_nonfinite(max_val) or bool(max_val <= 0.0):
        max_val = 8.0 * min_val

    k = config.num_clusters
    mu_k = jnp.zeros(k - 1, dtype=jnp.float64)
    var_k = jnp.exp(jnp.linspace(jnp.log(min_val), jnp.log(max_val), k - 1)) ** 2

    likelihood_matrix = _compute_likelihood_matrix(beta_hat_arr, s2_arr, mu_k, var_k)
    pi = jnp.ones(k, dtype=jnp.float64) / k

    state = FitState(
        likelihood_matrix=likelihood_matrix,
        initial_params=Params(pi=pi, mu_k=mu_k, var_k=var_k),
    )
    return Solution(value=state, result=RESULTS.successful, stats=None, state=None)


def _pi_step(pi: Array, direction: Array, step_size: ArrayLike) -> Array:
    tangent = simplex_tangent_direction(pi, direction)
    return exponential_map_simplex(pi, tangent, step_size)


def _baseline_objective(pi: Array, L: Array, alpha: Array) -> Array:
    mixture_pdf = jnp.clip(L @ pi, min=jnp.finfo(jnp.float64).tiny)
    log_likelihood = jnp.sum(jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1.0, pi))
    return -(log_likelihood - log_penalty)


def fit_baseline(
    state: FitState,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    r"""Fit baseline mixture weights via Riemannian gradient descent on the simplex.

    **Arguments:**

    - `state`: Pre-computed fit state from `prepare_fit_state`.
    - `config`: Inference configuration.
    - `verbose`: If `True` or a callable, emit solver diagnostics.

    **Returns:**

    - `Solution` with `Params`. Status: `RESULTS.successful`, `RESULTS.max_steps_reached`,
      `RESULTS.invalid_input`, `RESULTS.nonfinite_objective`.
    """
    if _is_tracing(state, config):
        return _fit_baseline_traced(state, config, verbose=verbose)

    L = state.likelihood_matrix
    init_params = state.initial_params
    pi_init = jnp.asarray(init_params.pi, dtype=jnp.float64)
    if L.ndim != 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "likelihood_matrix must be a 2D array"},
            state=None,
        )
    if pi_init.ndim != 1:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "initial_params.pi must be a 1D array"},
            state=None,
        )
    if L.shape[1] != pi_init.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "likelihood_matrix column count must match initial_params.pi length"},
            state=None,
        )
    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)

    obj = eqx.filter_jit(_baseline_objective)

    def _neg_obj(pi, _args):
        val = obj(pi, L, alpha)
        return jnp.where(jnp.isfinite(val), val, jnp.inf)

    solver = MutVarSolver(
        step_update=_pi_step,
        step_size=_DEFAULT_STEP_SIZE,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )
    optx_solution = optx.minimise(
        fn=_neg_obj,
        solver=solver,
        y0=pi_init,
        args=None,
        max_steps=config.max_iter,
        throw=False,
    )
    result = map_optimistix_result(optx_solution.result)
    pi_opt = optx_solution.value
    params = Params(pi=pi_opt, mu_k=init_params.mu_k, var_k=init_params.var_k)
    return Solution(
        value=params,
        result=result,
        stats={"n_steps": optx_solution.stats.get("num_steps", None)},
        state=None,
    )


def _refit_objective(
    pi: Array,
    L_sub: Array,
    prev_pi: Array,
    alpha: Array,
    penalty: float,
) -> Array:
    mixture_pdf = jnp.clip(L_sub @ pi, min=jnp.finfo(jnp.float64).tiny)
    log_likelihood = jnp.sum(jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1.0, pi))
    p1 = jnp.sum(nn.relu(prev_pi[1:] * pi[:-1] - prev_pi[:-1] * pi[1:]))
    rel_point_mass_dist = nn.relu(prev_pi[0] - pi[0])
    ordering_penalty = penalty * (p1 + rel_point_mass_dist)
    return -(log_likelihood - log_penalty - ordering_penalty)


def fit_refit_step(
    L_sub: ArrayLike,
    prev_params: Params,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    r"""Fit one refit step for a MAF-subset likelihood matrix.

    **Arguments:**

    - `L_sub`: Pre-sliced likelihood matrix for this MAF threshold, shape `(n_sub, K)`.
    - `prev_params`: Params from the previous threshold (provides init and ordering anchor).
    - `config`: Inference configuration.
    - `verbose`: If `True` or a callable, emit solver diagnostics.

    **Returns:**

    - `Solution` with `Params`. Status: `RESULTS.successful`, `RESULTS.max_steps_reached`,
      `RESULTS.empty_subset`, `RESULTS.invalid_input`, `RESULTS.nonfinite_objective`.
    """
    if _is_tracing(L_sub, prev_params, config):
        return _fit_refit_step_traced(L_sub, prev_params, config, verbose=verbose)

    L_arr = jnp.asarray(L_sub, dtype=jnp.float64)
    pi_init = jnp.asarray(prev_params.pi, dtype=jnp.float64)

    if L_arr.ndim != 2:
        return Solution(
            value=prev_params,
            result=RESULTS.invalid_input,
            stats={"reason": "L_sub must be a 2D array"},
            state=None,
        )
    if L_arr.shape[0] == 0:
        return Solution(
            value=prev_params,
            result=RESULTS.empty_subset,
            stats={"reason": "L_sub has no rows (empty MAF subset)"},
            state=None,
        )
    if not bool(jnp.isfinite(L_arr).all()):
        return Solution(
            value=prev_params,
            result=RESULTS.invalid_input,
            stats={"reason": "L_sub contains non-finite values"},
            state=None,
        )
    if pi_init.ndim != 1:
        return Solution(
            value=prev_params,
            result=RESULTS.invalid_input,
            stats={"reason": "prev_params.pi must be a 1D array"},
            state=None,
        )
    if L_arr.shape[1] != pi_init.shape[0]:
        return Solution(
            value=prev_params,
            result=RESULTS.invalid_input,
            stats={"reason": "L_sub column count must match prev_params.pi length"},
            state=None,
        )

    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)
    obj = eqx.filter_jit(_refit_objective)

    def _neg_obj(pi, _args):
        val = obj(pi, L_arr, pi_init, alpha, _DEFAULT_PENALTY)
        return jnp.where(jnp.isfinite(val), val, jnp.inf)

    solver = MutVarSolver(
        step_update=_pi_step,
        step_size=_DEFAULT_STEP_SIZE,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )
    optx_solution = optx.minimise(
        fn=_neg_obj,
        solver=solver,
        y0=pi_init,
        args=None,
        max_steps=config.max_iter,
        throw=False,
    )
    result = map_optimistix_result(optx_solution.result)
    pi_opt = optx_solution.value
    params = Params(pi=pi_opt, mu_k=prev_params.mu_k, var_k=prev_params.var_k)
    return Solution(
        value=params,
        result=result,
        stats={"n_steps": optx_solution.stats.get("num_steps", None)},
        state=None,
    )


__all__ = [
    "FitState",
    "Params",
    "fit_baseline",
    "fit_refit_step",
    "prepare_fit_state",
]
