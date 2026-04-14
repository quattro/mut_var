from typing import Any, cast

import numpy as np
import polars as pl
import pytest

import mut_var.cli as cli
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
