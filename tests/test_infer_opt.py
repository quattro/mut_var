from typing import Any, cast

import numpy as np
import polars as pl
import pytest

import mut_var.cli as cli

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics import baseline
from mut_var.numerics.baseline import BaselineConfig, fit_baseline
from mut_var.numerics.refit import fit_refit_grid, RefitConfig


def test_fit_baseline_returns_structured_solution_on_valid_arrays():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])

    solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        config=BaselineConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    assert isinstance(solution, Solution)
    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    assert isinstance(solution.stats, dict)
    assert "objective" in solution.stats


def test_fit_baseline_returns_controlled_failure_for_empty_input():
    solution = fit_baseline(
        beta_hat=np.array([]),
        s2=np.array([]),
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.empty_subset


def test_fit_baseline_rejects_tabular_objects():
    df = pl.DataFrame({"beta": [0.1], "s2": [0.01]})

    solution = fit_baseline(
        beta_hat=df,
        s2=df,
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input
    assert solution.stats is not None
    assert "arrays" in solution.stats["reason"]


def test_cli_no_longer_owns_optimizer_internals():
    assert not hasattr(cli, "baseline_objective")
    assert not hasattr(cli, "fit_baseline_mixture")


def test_fit_baseline_rejects_nonfinite_inputs():
    solution = fit_baseline(
        beta_hat=np.array([0.1, np.nan]),
        s2=np.array([0.01, 0.02]),
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input


def test_fit_baseline_uses_mix_sqp_solver():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])

    solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        config=BaselineConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    params = solution.value
    assert isinstance(params.pi, np.ndarray)
    assert abs(params.pi.sum() - 1.0) < 1e-6


def test_baseline_config_no_longer_exposes_solver_backend_knob():
    with pytest.raises(TypeError):
        cast(Any, BaselineConfig)(num_clusters=3, solver_backend="optimistix")


def test_fit_refit_grid_returns_one_model_per_threshold_plus_init():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])
    maf_masks = np.array([[True, True, True, True, True]], dtype=bool)
    init = baseline.Params(
        pi=np.array([0.8, 0.2]),
        mu_k=np.array([0.0]),
        var_k=np.array([1.0]),
    )

    solution = fit_refit_grid(
        beta_hat=beta_hat,
        s2=s2,
        maf_masks=maf_masks,
        init=init,
        config=RefitConfig(max_iter=5, step_size=0.5),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert len(solution.value) == 2  # baseline + 1 threshold


def test_refit_config_no_longer_exposes_solver_backend_knob():
    with pytest.raises(TypeError):
        cast(Any, RefitConfig)(solver_backend="optimistix")


def test_fit_refit_grid_returns_normalized_weights():
    beta_hat = np.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = np.array([0.01, 0.015, 0.02, 0.013, 0.011])
    maf_masks = np.array(
        [
            [True, True, True, True, True],
            [True, True, False, True, False],
            [True, False, False, True, False],
        ],
        dtype=bool,
    )
    init = baseline.Params(
        pi=np.array([0.8, 0.2]),
        mu_k=np.array([0.0]),
        var_k=np.array([1.0]),
    )

    solution = fit_refit_grid(
        beta_hat=beta_hat,
        s2=s2,
        maf_masks=maf_masks,
        init=init,
        config=RefitConfig(max_iter=5, step_size=0.5),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    # Each model should have weights summing to 1.
    for model in solution.value:
        assert abs(model.pi.sum() - 1.0) < 1e-6
