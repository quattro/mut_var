from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mut_var.numerics.mixture_fit import fit_baseline, fit_refit_step, FitState, Params, prepare_fit_state
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


def test_prepare_fit_state_invalid_s2(synthetic_arrays):
    beta, _ = synthetic_arrays
    s2 = np.array([0.01, 0.0] + [0.01] * (beta.shape[0] - 2))

    solution = prepare_fit_state(beta, s2, InferenceConfig(num_clusters=3))

    assert solution.result == RESULTS.invalid_input


def test_prepare_fit_state_empty_arrays():
    solution = prepare_fit_state([], [], InferenceConfig(num_clusters=3))

    assert solution.result == RESULTS.empty_subset


def test_prepare_fit_state_rejects_singleton_num_clusters_under_jit(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=1)

    eager_solution = prepare_fit_state(beta, s2, config)
    traced_solution = jax.jit(prepare_fit_state)(beta, s2, config)

    assert eager_solution.result == RESULTS.invalid_input
    assert traced_solution.result == RESULTS.invalid_input


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


def test_fit_baseline_invalid_input_on_likelihood_parameter_shape_mismatch(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    state = state_solution.value
    mismatched_state = FitState(
        likelihood_matrix=state.likelihood_matrix[:, :2],
        initial_params=state.initial_params,
    )

    solution = fit_baseline(mismatched_state, config)

    assert solution.result == RESULTS.invalid_input
    assert solution.stats == {
        "reason": "likelihood_matrix column count must match initial_params.pi length",
    }


def test_fit_baseline_invalid_input_on_array_like_pi(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    state = state_solution.value
    array_like_state = FitState(
        likelihood_matrix=state.likelihood_matrix,
        initial_params=Params(
            pi=[0.8, 0.2],
            mu_k=state.initial_params.mu_k,
            var_k=state.initial_params.var_k,
        ),
    )

    solution = fit_baseline(array_like_state, config)

    assert solution.result == RESULTS.invalid_input
    assert solution.stats == {
        "reason": "likelihood_matrix column count must match initial_params.pi length",
    }


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


def test_fit_refit_step_invalid_input_on_likelihood_parameter_shape_mismatch(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=10)

    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    prev_params = Params(
        pi=jnp.asarray([0.8, 0.2], dtype=jnp.float64),
        mu_k=state_solution.value.initial_params.mu_k,
        var_k=state_solution.value.initial_params.var_k,
    )
    l_sub = state_solution.value.likelihood_matrix[:25, :]

    solution = fit_refit_step(l_sub, prev_params, config)

    assert solution.result == RESULTS.invalid_input
    assert solution.stats == {
        "reason": "L_sub column count must match prev_params.pi length",
    }


@pytest.mark.parametrize("use_jit", [False, True])
def test_fit_baseline_returns_nonfinite_objective_when_objective_is_nonfinite(
    synthetic_arrays,
    monkeypatch,
    use_jit,
):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=5)
    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    def _always_nonfinite(*args, **kwargs):
        del args, kwargs
        return jnp.asarray(jnp.inf, dtype=jnp.float64)

    from mut_var.numerics import mixture_fit

    monkeypatch.setattr(mixture_fit, "_baseline_objective", _always_nonfinite)

    state = FitState(
        likelihood_matrix=state_solution.value.likelihood_matrix,
        initial_params=Params(
            pi=jnp.zeros_like(state_solution.value.initial_params.pi),
            mu_k=state_solution.value.initial_params.mu_k,
            var_k=state_solution.value.initial_params.var_k,
        ),
    )

    runner = jax.jit(fit_baseline) if use_jit else fit_baseline
    solution = runner(state, config)

    assert solution.result == RESULTS.nonfinite_objective
    jax.clear_caches()


@pytest.mark.parametrize("use_jit", [False, True])
def test_fit_refit_step_returns_nonfinite_objective_when_objective_is_nonfinite(
    synthetic_arrays,
    monkeypatch,
    use_jit,
):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=5)
    state_solution = prepare_fit_state(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    def _always_nonfinite(*args, **kwargs):
        del args, kwargs
        return jnp.asarray(jnp.inf, dtype=jnp.float64)

    from mut_var.numerics import mixture_fit

    monkeypatch.setattr(mixture_fit, "_refit_objective", _always_nonfinite)

    prev_params = Params(
        pi=jnp.zeros_like(state_solution.value.initial_params.pi),
        mu_k=state_solution.value.initial_params.mu_k,
        var_k=state_solution.value.initial_params.var_k,
    )
    l_sub = state_solution.value.likelihood_matrix[:25, :]

    runner = jax.jit(fit_refit_step) if use_jit else fit_refit_step
    solution = runner(l_sub, prev_params, config)

    assert solution.result == RESULTS.nonfinite_objective
    jax.clear_caches()


def test_public_numerics_entrypoints_are_jittable(synthetic_arrays):
    beta, s2 = synthetic_arrays
    config = InferenceConfig(num_clusters=3, max_iter=5)

    jax.clear_caches()
    state_solution = jax.jit(prepare_fit_state)(beta, s2, config)
    assert state_solution.result == RESULTS.successful

    baseline_solution = jax.jit(fit_baseline)(state_solution.value, config)
    assert baseline_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)

    refit_solution = jax.jit(fit_refit_step)(
        state_solution.value.likelihood_matrix[:25, :],
        state_solution.value.initial_params,
        config,
    )
    assert refit_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
