# pattern: Functional Core
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

import mut_var.cli as cli
import mut_var.numerics.mixsqp as mixsqp_module
import mut_var.numerics.mixture_fit as mixture_fit_module

from mut_var.numerics.mixture_fit import fit_baseline, fit_refit_step, prepare_fit_state
from mut_var.types import InferenceConfig, RESULTS, Solution


def test_fit_baseline_returns_structured_solution_on_valid_arrays():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])

    prepared = prepare_fit_state(
        beta_hat=beta_hat,
        s2=s2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )
    solution = fit_baseline(
        state=prepared.value,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    assert prepared.result == RESULTS.successful
    assert isinstance(solution, Solution)
    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    assert isinstance(solution.stats, dict)
    assert "objective" in solution.stats


def test_fit_baseline_returns_controlled_failure_for_empty_input():
    solution = prepare_fit_state(
        beta_hat=np.array([]),
        s2=np.array([]),
        config=InferenceConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.empty_subset


def test_fit_baseline_rejects_tabular_objects():
    df = pl.DataFrame({"beta": [0.1], "s2": [0.01]})

    solution = prepare_fit_state(
        beta_hat=df,
        s2=df,
        config=InferenceConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input
    assert solution.stats is not None
    assert "arrays" in solution.stats["reason"]


def test_cli_no_longer_owns_optimizer_internals():
    assert not hasattr(cli, "baseline_objective")
    assert not hasattr(cli, "fit_baseline_mixture")


def test_fit_baseline_rejects_nonfinite_inputs():
    solution = prepare_fit_state(
        beta_hat=np.array([0.1, np.nan]),
        s2=np.array([0.01, 0.02]),
        config=InferenceConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input


def test_fit_baseline_uses_mix_sqp_solver():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])

    prepared = prepare_fit_state(
        beta_hat=beta_hat,
        s2=s2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )
    solution = fit_baseline(
        state=prepared.value,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    assert prepared.result == RESULTS.successful
    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    params = solution.value
    assert isinstance(params.pi, np.ndarray)
    assert abs(params.pi.sum() - 1.0) < 1e-6


def test_inference_config_no_longer_exposes_solver_backend_knob():
    with pytest.raises(TypeError):
        cast(Any, InferenceConfig)(num_clusters=3, solver_backend="optimistix")


def test_inference_config_no_longer_exposes_legacy_optimization_knobs():
    with pytest.raises(TypeError):
        cast(Any, InferenceConfig)(num_clusters=3, step_size=0.5)
    with pytest.raises(TypeError):
        cast(Any, InferenceConfig)(num_clusters=3, penalty=1.0)


def test_fit_refit_step_returns_updated_params_for_likelihood_subset():
    init = mixture_fit_module.Params(
        pi=np.array([0.8, 0.2]),
        mu_k=np.array([0.0]),
        var_k=np.array([1.0]),
    )

    solution = fit_refit_step(
        L_sub=np.array([[0.8, 0.2], [0.7, 0.3], [0.85, 0.15]]),
        prev_params=init,
        config=InferenceConfig(num_clusters=2, max_iter=5),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert abs(solution.value.pi.sum() - 1.0) < 1e-6


def test_baseline_and_refit_use_same_prior_pseudo_observations(monkeypatch):
    likelihood = np.array([[0.8, 0.2], [0.3, 0.7]])
    prior = np.array([10.0, 2.0])
    params = mixture_fit_module.Params(
        pi=np.array([0.6, 0.4]),
        mu_k=np.array([0.0]),
        var_k=np.array([1.0]),
    )
    captured: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _fake_mix_sqp(L, *, w, **_kwargs):  # noqa: ANN001
        captured["baseline"] = (L.copy(), w.copy())
        return params.pi, {"converged": True, "n_iter": 1, "objective": 0.0}

    def _fake_mix_sqp_ordered(L, *, w, **_kwargs):  # noqa: ANN001
        captured["refit"] = (L.copy(), w.copy())
        return params.pi, {"converged": True, "n_iter": 1, "objective": 0.0}

    monkeypatch.setattr(mixture_fit_module, "mix_sqp", _fake_mix_sqp)
    monkeypatch.setattr(mixture_fit_module, "mix_sqp_ordered", _fake_mix_sqp_ordered)

    baseline_solution = fit_baseline(
        state=mixture_fit_module.FitState(likelihood_matrix=likelihood, initial_params=params),
        config=InferenceConfig(num_clusters=2),
        prior=prior,
    )
    refit_solution = fit_refit_step(
        L_sub=likelihood,
        prev_params=params,
        config=InferenceConfig(num_clusters=2),
        prior=prior,
    )

    expected_likelihood = np.vstack([likelihood, np.eye(2)])
    expected_weights = np.array([1.0, 1.0, 9.0, 1.0])
    assert baseline_solution.result == RESULTS.successful
    assert refit_solution.result == RESULTS.successful
    np.testing.assert_allclose(captured["baseline"][0], expected_likelihood)
    np.testing.assert_allclose(captured["baseline"][1], expected_weights)
    np.testing.assert_allclose(captured["refit"][0], expected_likelihood)
    np.testing.assert_allclose(captured["refit"][1], expected_weights)


def test_constraint_matrix_includes_null_component_cap():
    baseline = np.array([0.6, 0.3, 0.1])

    constraints = mixsqp_module.build_constraints_matrix(baseline, constrain_spike=True)

    assert constraints.shape == (3, 3)
    np.testing.assert_allclose(constraints[0], np.array([0.4, -0.6, -0.6]))
    np.testing.assert_allclose(constraints[1], np.array([0.3, -0.6, 0.0]))
    np.testing.assert_allclose(constraints[2], np.array([0.0, 0.1, -0.3]))
    assert constraints[0] @ np.array([0.7, 0.2, 0.1]) > 0.0
    assert constraints[0] @ np.array([0.6, 0.25, 0.15]) <= 0.0


def test_ordered_qp_solver_tracks_zero_bound_constraints():
    hessian = np.array(
        [
            [9.265236483644882, 2.0170134658972074, -0.6582646779509045, -4.11791022773436, 0.7152169304180868],
            [2.0170134658972074, 2.719504563388768, -0.06023579846176903, 0.06160778642712707, 0.21478247341825352],
            [-0.6582646779509045, -0.06023579846176903, 3.1054449909759607, -0.7437559711761249, -0.8206272061390226],
            [-4.11791022773436, 0.06160778642712707, -0.7437559711761249, 3.821151774139339, -1.1360469632156072],
            [0.7152169304180868, 0.21478247341825352, -0.8206272061390226, -1.1360469632156072, 1.357415677929455],
        ]
    )
    linear = np.array(
        [1.3253230505301357, 0.16236948488296493, -0.6354460697725838, 1.1112524465092426, -0.9833875285453708]
    )
    baseline = np.array(
        [0.1043872229360963, 0.0570553954283856, 0.23167923122797676, 0.6020230798663861, 0.00485507054115538]
    )
    constraints = mixsqp_module.build_constraints_matrix(baseline)
    feasible_better = np.array([0.0, 0.0, 0.03304169878694252, 0.08585950998783105, 0.8162883707070141])

    solution = mixsqp_module.solve_qp_ordered(hessian, linear, constraints, baseline, max_iter=1000, tol=1e-9)

    solution_objective = 0.5 * solution @ hessian @ solution + linear @ solution
    candidate_objective = 0.5 * feasible_better @ hessian @ feasible_better + linear @ feasible_better
    assert np.all(solution >= -1e-8)
    assert np.max(constraints @ solution) <= 1e-7
    assert np.max(constraints @ feasible_better) <= 1e-8
    assert solution_objective <= candidate_objective + 1e-6


def test_fit_refit_step_enforces_previous_null_component_cap():
    init = mixture_fit_module.Params(
        pi=np.array([0.4, 0.3, 0.3]),
        mu_k=np.array([0.0, 0.0]),
        var_k=np.array([1.0, 4.0]),
    )

    solution = fit_refit_step(
        L_sub=np.tile(np.array([[1000.0, 1.0, 1.0]]), (12, 1)),
        prev_params=init,
        config=InferenceConfig(num_clusters=3, max_iter=50, constrain_spike=True),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value.pi[0] <= init.pi[0] + 1e-8


def test_fit_refit_step_allows_null_component_enrichment_without_spike_constraint():
    init = mixture_fit_module.Params(
        pi=np.array([0.4, 0.3, 0.3]),
        mu_k=np.array([0.0, 0.0]),
        var_k=np.array([1.0, 4.0]),
    )

    solution = fit_refit_step(
        L_sub=np.tile(np.array([[1000.0, 1.0, 1.0]]), (12, 1)),
        prev_params=init,
        config=InferenceConfig(num_clusters=3, max_iter=50, constrain_spike=False),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value.pi[0] > init.pi[0] + 0.1


def test_fit_refit_step_allows_later_component_enrichment_under_constraints():
    init = mixture_fit_module.Params(
        pi=np.array([0.4, 0.3, 0.3]),
        mu_k=np.array([0.0, 0.0]),
        var_k=np.array([1.0, 4.0]),
    )

    solution = fit_refit_step(
        L_sub=np.tile(np.array([[1.0, 1.0, 1000.0]]), (12, 1)),
        prev_params=init,
        config=InferenceConfig(num_clusters=3, max_iter=50, constrain_spike=True),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value.pi[0] <= init.pi[0] + 1e-8
    assert solution.value.pi[2] > init.pi[2] + 0.1
