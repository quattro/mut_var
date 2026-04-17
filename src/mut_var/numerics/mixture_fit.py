from __future__ import annotations

# pattern: Functional Core
from typing import Any, cast, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx

from jax.scipy.special import xlogy
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike

from mut_var.numerics._solver import (
    map_optimistix_result,
    RiemannianGradientDescent,
    RiemannianTrustRegion,
)
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_nonfinite,
    simplex_tangent_direction,
)
from mut_var.types import InferenceConfig, RESULTS, Solution

_DEFAULT_STEP_SIZE: float = 0.01
_DEFAULT_PENALTY: float = 1.0


def _pi_step(pi: Array, grad: Array, step_size: ArrayLike) -> Array:
    # Move downhill: flip sign once here so the stored descent state carries
    # the raw gradient and logs read in one consistent direction.
    tangent = simplex_tangent_direction(pi, -grad)
    return exponential_map_simplex(pi, tangent, step_size)


def _build_solver(
    config: InferenceConfig,
    hessian_builder: Any,
    verbose: bool | Any,
) -> optx.AbstractMinimiser:
    if config.solver == "rtr":
        return RiemannianTrustRegion(
            hessian_builder,
            gtol=config.tol,
            verbose=verbose,
        )
    return RiemannianGradientDescent(
        step_update=_pi_step,
        step_size=_DEFAULT_STEP_SIZE,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )


class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


class FitState(NamedTuple):
    likelihood_matrix: Array
    initial_params: Params


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


def _mixture_rgrad_riemannian_hessian(
    y_sphere: Array,
    pi: Array,
    L: Array,
    alpha: Array,
) -> tuple[Array, Array, lx.MatrixLinearOperator]:
    r"""Analytic Riemannian gradient and Hessian on the sphere chart.

    For $F(\pi) = -\sum_n \log(L\pi)_n - \sum_k (\alpha_k-1) \log \pi_k$
    (MAP objective, Dirichlet log-prior with inward sign), the Euclidean
    Hessian factors as a low-rank-plus-diagonal form
    $\mathrm{Hess}_\pi F = L^\top D L + \mathrm{diag}((\alpha-1)/\pi^2)$
    with $D = \mathrm{diag}(1/(L\pi)^2)$. Lifting through $\pi = y^2$ gives a
    dense $k \times k$ sphere-chart Euclidean Hessian, and the projection
    sandwich $\mathrm{Hess}_R f(v) = P_y H_E P_y v - \langle y, \nabla_y f\rangle P_y v$
    collapses to a fully materialised $k \times k$ symmetric matrix. The heavy
    $n$-sized matmul $G = L^\top D L$ happens once per outer step and is shared
    with the gradient ($L^\top r$), so every inner TruncatedCG HVP is an
    $O(k^2)$ dense matvec and no extra autodiff pass is needed for the gradient.
    """
    tiny = jnp.finfo(jnp.float64).tiny
    Lpi = jnp.clip(L @ pi, min=tiny)
    log_Lpi = jnp.log(Lpi)
    f = -(jnp.sum(log_Lpi) + jnp.sum(xlogy(alpha - 1.0, pi)))
    r = 1.0 / Lpi
    D = r * r
    # Gram G = L^T D L is the only expensive $n$-sized matmul per outer step.
    # Identity: L^T r = G @ pi, since L @ pi = 1/r, so G @ pi = L^T (D Lpi)
    # = L^T (r^2 / r) = L^T r. Costs $k^2$ instead of an extra $n k$ matvec.
    G = L.T @ (D[:, None] * L)
    Lt_r = G @ pi
    pen_diag = (alpha - 1.0) / pi
    g_pi = -Lt_r - pen_diag
    ge = 2.0 * y_sphere * g_pi
    rgrad = ge - jnp.dot(y_sphere, ge) * y_sphere
    y_ge_dot = jnp.dot(y_sphere, ge)
    diag_total = -2.0 * Lt_r + 2.0 * pen_diag

    H_E = jnp.diag(diag_total) + 4.0 * (y_sphere[:, None] * G * y_sphere[None, :])
    k = y_sphere.shape[0]
    P_y = jnp.eye(k, dtype=y_sphere.dtype) - jnp.outer(y_sphere, y_sphere)
    H_R = P_y @ H_E @ P_y - y_ge_dot * P_y

    return f, rgrad, lx.MatrixLinearOperator(H_R, tags=frozenset({lx.symmetric_tag}))


def _baseline_hessian_builder(y_sphere: Array, pi: Array, args: Any) -> tuple[Array, Array, lx.AbstractLinearOperator]:
    L, alpha = args
    return _mixture_rgrad_riemannian_hessian(y_sphere, pi, L, alpha)


def _ordering_penalty(pi: Array, prev_pi: Array, penalty: float) -> Array:
    c1 = prev_pi[1:] * pi[:-1] - prev_pi[:-1] * pi[1:]
    p1 = jnp.sum(jax.nn.relu(c1))
    c2 = prev_pi[0] - pi[0]
    rel_point_mass_dist = jax.nn.relu(c2)
    return penalty * (p1 + rel_point_mass_dist)


def _refit_hessian_builder(y_sphere: Array, pi: Array, args: Any) -> tuple[Array, Array, lx.AbstractLinearOperator]:
    # The ReLU ordering penalty is omitted from the Hessian (non-smooth and
    # small away from the active set). We include it in `f` so the TR
    # actual/predicted reduction ratio reflects the full refit objective.
    L_sub, prev_pi, alpha = args
    f_mix, rg, H = _mixture_rgrad_riemannian_hessian(y_sphere, pi, L_sub, alpha)
    f = f_mix + _ordering_penalty(pi, prev_pi, _DEFAULT_PENALTY)
    return f, rg, H


def _result_from_objective_value(
    objective_value: ArrayLike,
    solver_result: optx.RESULTS,
) -> RESULTS:
    objective_is_finite = jnp.all(jnp.isfinite(jnp.asarray(objective_value)))
    return cast(
        RESULTS,
        jax.lax.cond(
            objective_is_finite,
            lambda _: map_optimistix_result(solver_result),
            lambda _: RESULTS.nonfinite_objective,
            operand=None,
        ),
    )


def _baseline_objective(pi: Array, L: Array, alpha: Array) -> Array:
    mixture_pdf = jnp.clip(L @ pi, min=jnp.finfo(jnp.float64).tiny)
    log_likelihood = jnp.sum(jnp.log(mixture_pdf))
    log_prior = jnp.sum(xlogy(alpha - 1.0, pi))
    return -(log_likelihood + log_prior)


_baseline_obj_jit = eqx.filter_jit(_baseline_objective)


def _neg_baseline_obj(pi: Array, args: tuple) -> Array:
    L, alpha = args
    val = _baseline_obj_jit(pi, L, alpha)
    return jnp.where(jnp.isfinite(val), val, jnp.inf)


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
      `RESULTS.nonfinite_objective`.
    """
    L = state.likelihood_matrix
    init_params = state.initial_params
    pi_init = jnp.asarray(init_params.pi, dtype=jnp.float64)
    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)

    solver = _build_solver(config, _baseline_hessian_builder, verbose)
    optx_solution = optx.minimise(
        fn=_neg_baseline_obj,
        solver=solver,
        y0=pi_init,
        args=(L, alpha),
        max_steps=config.max_iter,
        throw=False,
    )
    pi_opt = optx_solution.value
    objective_value = _baseline_obj_jit(pi_opt, L, alpha)
    result = _result_from_objective_value(objective_value, optx_solution.result)
    n_steps = optx_solution.stats.get("num_steps")
    return Solution(
        value=Params(pi=pi_opt, mu_k=init_params.mu_k, var_k=init_params.var_k),
        result=result,
        stats={"n_steps": n_steps},
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
    log_prior = jnp.sum(xlogy(alpha - 1.0, pi))
    return -(log_likelihood + log_prior) + _ordering_penalty(pi, prev_pi, penalty)


_refit_obj_jit = eqx.filter_jit(_refit_objective)


def _neg_refit_obj(pi: Array, args: tuple) -> Array:
    L_sub, prev_pi, alpha = args
    val = _refit_obj_jit(pi, L_sub, prev_pi, alpha, _DEFAULT_PENALTY)
    return jnp.where(jnp.isfinite(val), val, jnp.inf)


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
      `RESULTS.empty_subset`, `RESULTS.nonfinite_objective`.
    """
    L_arr = jnp.asarray(L_sub, dtype=jnp.float64)
    pi_init = jnp.asarray(prev_params.pi, dtype=jnp.float64)

    if L_arr.shape[0] == 0:
        return Solution(
            value=prev_params,
            result=RESULTS.empty_subset,
            stats={"reason": "L_sub has no rows (empty MAF subset)"},
            state=None,
        )

    k = pi_init.shape[0]
    alpha = jnp.array([10.0] + (k - 1) * [1.0], dtype=jnp.float64)

    solver = _build_solver(config, _refit_hessian_builder, verbose)
    optx_solution = optx.minimise(
        fn=_neg_refit_obj,
        solver=solver,
        y0=pi_init,
        args=(L_arr, pi_init, alpha),
        max_steps=config.max_iter,
        throw=False,
    )
    pi_opt = optx_solution.value
    objective_value = _refit_obj_jit(pi_opt, L_arr, pi_init, alpha, _DEFAULT_PENALTY)
    result = _result_from_objective_value(objective_value, optx_solution.result)
    n_steps = optx_solution.stats.get("num_steps")
    return Solution(
        value=Params(pi=pi_opt, mu_k=prev_params.mu_k, var_k=prev_params.var_k),
        result=result,
        stats={"n_steps": n_steps},
        state=None,
    )


__all__ = [
    "FitState",
    "Params",
    "fit_baseline",
    "fit_refit_step",
    "prepare_fit_state",
]
