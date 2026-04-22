from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.scipy.special import xlogy

from mut_var.io import load_inference_arrays
from mut_var.numerics._solver._trust_region import (
    _accept_trust_region_step,
    _boundary_stagnation_converged,
)
from mut_var.numerics._solver_utils import simplex_to_sphere, sphere_to_simplex
from mut_var.numerics.mixture_fit import (
    _baseline_hessian_builder,
    _baseline_objective,
    _refit_hessian_builder,
    _refit_objective,
    fit_baseline,
    fit_refit_masked_step,
    fit_refit_step,
    FitState,
    Params,
    prepare_fit_state,
)
from mut_var.types import InferenceConfig, RESULTS


@pytest.fixture
def synthetic_arrays():
    rng = np.random.default_rng(42)
    n = 100
    beta = rng.normal(0.0, 0.05, n)
    s2 = np.full(n, 0.01)
    return beta, s2


def test_prepare_fit_state_succeeds_on_valid_arrays(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3)

    solution = prepare_fit_state(beta, s2, config)

    assert solution.result == RESULTS.successful
    assert isinstance(solution.value, FitState)
    assert solution.value.likelihood_matrix.shape == (beta.shape[0], 3)


def test_fit_baseline_converges_pi_sums_to_one(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    solution = fit_baseline(state_solution.value, config)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert isinstance(solution.value, Params)
    assert jnp.isclose(jnp.sum(solution.value.pi), 1.0)
    assert jnp.allclose(solution.value.mu_k, state_solution.value.initial_params.mu_k)
    assert jnp.allclose(solution.value.var_k, state_solution.value.initial_params.var_k)


def test_fit_baseline_max_steps_reached_on_max_iter_one(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=1)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    solution = fit_baseline(state_solution.value, config)

    assert solution.result == RESULTS.max_steps_reached
    assert isinstance(solution.value, Params)


def test_fit_refit_step_pi_sums_to_one_and_mu_var_unchanged(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    prev_params = Params(
        pi=jnp.asarray([0.8, 0.15, 0.05], dtype=jnp.float64),
        mu_k=state_solution.value.initial_params.mu_k,
        var_k=state_solution.value.initial_params.var_k,
    )
    l_sub = state_solution.value.likelihood_matrix[:25, :]

    solution = fit_refit_step(l_sub, prev_params, config)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert isinstance(solution.value, Params)
    assert jnp.isclose(jnp.sum(solution.value.pi), 1.0)
    assert jnp.allclose(solution.value.mu_k, prev_params.mu_k)
    assert jnp.allclose(solution.value.var_k, prev_params.var_k)


def test_fit_refit_masked_step_matches_sliced_refit(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    prev_params = Params(
        pi=jnp.asarray([0.8, 0.15, 0.05], dtype=jnp.float64),
        mu_k=state_solution.value.initial_params.mu_k,
        var_k=state_solution.value.initial_params.var_k,
    )
    mask = jnp.asarray([True] * 25 + [False] * (state_solution.value.likelihood_matrix.shape[0] - 25))
    l_sub = state_solution.value.likelihood_matrix[mask, :]

    sliced = fit_refit_step(l_sub, prev_params, config)
    masked = fit_refit_masked_step(state_solution.value.likelihood_matrix, mask, prev_params, config)

    assert sliced.result == masked.result
    assert jnp.allclose(sliced.value.pi, masked.value.pi, rtol=1e-8, atol=1e-8)
    assert jnp.allclose(sliced.value.mu_k, masked.value.mu_k)
    assert jnp.allclose(sliced.value.var_k, masked.value.var_k)


def test_rtr_baseline_makes_progress(synthetic_arrays):
    # Regression: RTR pred_gain sign bug caused the solver to reject every step
    # (radius shrank to zero, objective never changed, Accepted=False throughout).
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=20, solver="rtr")

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    solution = fit_baseline(state_solution.value, config)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert isinstance(solution.value, Params)
    assert jnp.isclose(jnp.sum(solution.value.pi), 1.0)
    # Solver must have accepted at least one step — if pred_gain sign is wrong,
    # n_steps will be maxiter with zero accepted steps and objective unchanged.
    assert solution.stats is not None and solution.stats["n_steps"] > 0


def test_baseline_objective_includes_null_component_dirichlet_penalty():
    l_matrix = jnp.asarray(
        [
            [0.80, 0.20],
            [0.65, 0.35],
            [0.55, 0.45],
        ],
        dtype=jnp.float64,
    )
    pi = jnp.asarray([0.70, 0.30], dtype=jnp.float64)

    objective = _baseline_objective(pi, l_matrix)

    mixture_pdf = l_matrix @ pi
    prior_weights = jnp.asarray([9.0, 0.0], dtype=jnp.float64)
    expected = -jnp.sum(jnp.log(mixture_pdf)) - jnp.sum(xlogy(prior_weights, pi))
    assert jnp.allclose(objective, expected, rtol=1e-12, atol=1e-12)


def test_rtr_baseline_builder_gradient_matches_penalized_objective():
    l_matrix = jnp.asarray(
        [
            [0.80, 0.20],
            [0.65, 0.35],
            [0.55, 0.45],
        ],
        dtype=jnp.float64,
    )
    pi = jnp.asarray([0.70, 0.30], dtype=jnp.float64)
    y = simplex_to_sphere(pi)

    f, rgrad, _hessian = _baseline_hessian_builder(y, pi, (l_matrix,))

    def penalized_baseline_on_sphere(y_sphere):
        pi_sphere = sphere_to_simplex(y_sphere)
        return _baseline_objective(pi_sphere, l_matrix)

    autodiff_grad = jax.grad(penalized_baseline_on_sphere)(y)
    autodiff_rgrad = autodiff_grad - jnp.dot(y, autodiff_grad) * y

    assert jnp.isfinite(f)
    assert jnp.allclose(rgrad, autodiff_rgrad, rtol=1e-8, atol=1e-8)


def test_rtr_refit_builder_gradient_matches_penalized_objective():
    penalty = 100.0
    l_sub = jnp.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
        ],
        dtype=jnp.float64,
    )
    prev_pi = jnp.asarray([0.80, 0.15, 0.05], dtype=jnp.float64)
    pi = jnp.asarray([0.70, 0.10, 0.20], dtype=jnp.float64)
    y = simplex_to_sphere(pi)

    f, rgrad, _hessian = _refit_hessian_builder(y, pi, (l_sub, prev_pi, penalty))

    def penalized_refit_on_sphere(y_sphere):
        pi_sphere = sphere_to_simplex(y_sphere)
        return _refit_objective(pi_sphere, l_sub, prev_pi, penalty)

    autodiff_grad = jax.grad(penalized_refit_on_sphere)(y)
    autodiff_rgrad = autodiff_grad - jnp.dot(y, autodiff_grad) * y

    assert jnp.isfinite(f)
    assert jnp.allclose(rgrad, autodiff_rgrad, rtol=1e-8, atol=1e-8)


def test_rgd_baseline_converges_quickly_on_fixture():
    arrays = load_inference_arrays("tests/fixtures/sumstats_valid.tsv")
    config = InferenceConfig(num_clusters=10, max_iter=300, tol=1e-3, solver="rgd")

    state_solution = prepare_fit_state(arrays.beta_hat, arrays.s2, config)
    assert state_solution.result == RESULTS.successful

    solution = fit_baseline(state_solution.value, config)

    assert solution.result == RESULTS.successful
    assert solution.stats is not None
    # With manifold-step termination, the same fixture still converges well
    # within the configured iteration budget, but no longer stops on a tiny
    # ambient simplex difference.
    assert int(solution.stats["n_steps"]) < 250


def test_rtr_baseline_does_not_stall_on_orthant_crossing_fixture():
    arrays = load_inference_arrays("tests/fixtures/sumstats_valid.tsv")
    config = InferenceConfig(num_clusters=10, max_iter=100, tol=1e-3, solver="rtr")

    state_solution = prepare_fit_state(arrays.beta_hat, arrays.s2, config)
    assert state_solution.result == RESULTS.successful

    solution = fit_baseline(state_solution.value, config)

    assert solution.result == RESULTS.successful
    assert solution.stats is not None
    # Regression: RTR used to reject valid geodesic trials whenever the sphere
    # chart crossed into a negative orthant, causing repeated radius quartering
    # at identical objective values near sparse solutions.
    assert int(solution.stats["n_steps"]) < 10


def test_rtr_accepts_positive_tiny_model_reduction_step():
    accepted = _accept_trust_region_step(
        f=jnp.asarray(6_441_581.815680849, dtype=jnp.float64),
        actual_reduction=jnp.asarray(1e-6, dtype=jnp.float64),
        pred_reduction=jnp.asarray(5e-6, dtype=jnp.float64),
        rho=jnp.asarray(0.2, dtype=jnp.float64),
        eta_accept=0.1,
    )
    assert bool(accepted)

    accepted_subthreshold = _accept_trust_region_step(
        f=jnp.asarray(6_441_581.815680849, dtype=jnp.float64),
        actual_reduction=jnp.asarray(1e-6, dtype=jnp.float64),
        pred_reduction=jnp.asarray(5e-6, dtype=jnp.float64),
        rho=jnp.asarray(0.05, dtype=jnp.float64),
        eta_accept=0.1,
    )
    assert bool(accepted_subthreshold)


def test_rtr_boundary_stagnation_counts_as_converged():
    converged = _boundary_stagnation_converged(
        radius=jnp.asarray(1e-12, dtype=jnp.float64),
        pred_reduction=jnp.asarray(1.1432911006049647e-11, dtype=jnp.float64),
        objective=jnp.asarray(6_441_581.815678639, dtype=jnp.float64),
        on_boundary=jnp.asarray(True),
        min_radius=1e-12,
    )
    assert bool(converged)

    not_converged = _boundary_stagnation_converged(
        radius=jnp.asarray(1e-6, dtype=jnp.float64),
        pred_reduction=jnp.asarray(1e-3, dtype=jnp.float64),
        objective=jnp.asarray(10.0, dtype=jnp.float64),
        on_boundary=jnp.asarray(False),
        min_radius=1e-12,
    )
    assert not bool(not_converged)
