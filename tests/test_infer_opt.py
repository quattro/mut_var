from typing import Any, cast

import jax.numpy as jnp
import jax.random as rdm
import polars as pl
import pytest

import mut_var.cli as cli

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics import baseline, refit
from mut_var.numerics.baseline import BaselineConfig, fit_baseline
from mut_var.numerics.refit import _fit_single_refit, RefitConfig


def test_fit_baseline_returns_structured_solution_on_valid_arrays():
    beta_hat = jnp.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = jnp.array([0.01, 0.015, 0.02, 0.013, 0.011])

    solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(0),
        config=BaselineConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    assert isinstance(solution, Solution)
    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    assert isinstance(solution.stats, dict)
    assert "objective" in solution.stats


def test_fit_baseline_returns_controlled_failure_for_empty_input():
    solution = fit_baseline(
        beta_hat=jnp.array([]),
        s2=jnp.array([]),
        key=rdm.PRNGKey(0),
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.empty_subset


def test_fit_baseline_rejects_tabular_objects():
    df = pl.DataFrame({"beta": [0.1], "s2": [0.01]})

    solution = fit_baseline(
        beta_hat=df,
        s2=df,
        key=rdm.PRNGKey(0),
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input
    assert "arrays" in solution.stats["reason"]


def test_cli_no_longer_owns_optimizer_internals():
    assert not hasattr(cli, "baseline_objective")
    assert not hasattr(cli, "fit_baseline_mixture")


def test_fit_baseline_rejects_nonfinite_inputs():
    solution = fit_baseline(
        beta_hat=jnp.array([0.1, jnp.nan]),
        s2=jnp.array([0.01, 0.02]),
        key=rdm.PRNGKey(0),
        config=BaselineConfig(num_clusters=3),
    )

    assert solution.result == RESULTS.invalid_input


def test_algorithm_scope_keeps_original_objective_functions():
    assert callable(baseline.baseline_objective)
    assert callable(refit.penalized_objective)


def test_fit_single_refit_returns_controlled_failure_for_nonfinite_objective():
    init = baseline.Params(
        pi=jnp.array([0.7, 0.3]),
        mu_k=jnp.array([0.0]),
        var_k=jnp.array([1.0]),
    )
    likelihoods = jnp.array([[0.2, 0.8], [0.4, 0.6]], dtype=jnp.float64)
    weights = jnp.array([1.0, 1.0], dtype=jnp.float64)

    def _nonfinite_obj(*_args, **_kwargs):
        return jnp.nan

    solution = _fit_single_refit(
        likelihoods=likelihoods,
        weights=weights,
        init=init,
        config=RefitConfig(max_iter=1, step_size=0.75, tol=1e-8),
        obj=_nonfinite_obj,
        verbose=False,
    )

    assert solution.result == RESULTS.nonfinite_objective


def test_fit_baseline_uses_optimistix_solver_path_by_default():
    beta_hat = jnp.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = jnp.array([0.01, 0.015, 0.02, 0.013, 0.011])

    solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(0),
        config=BaselineConfig(
            num_clusters=3,
            max_iter=5,
            step_size=0.5,
        ),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None


def test_baseline_config_no_longer_exposes_solver_backend_knob():
    with pytest.raises(TypeError):
        cast(Any, BaselineConfig)(num_clusters=3, solver_backend="optimistix")


def test_fit_refit_grid_uses_optimistix_solver_path_by_default():
    beta_hat = jnp.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = jnp.array([0.01, 0.015, 0.02, 0.013, 0.011])
    maf_masks = jnp.array([[True, True, True, True, True]], dtype=bool)
    init = baseline.Params(
        pi=jnp.array([0.8, 0.2]),
        mu_k=jnp.array([0.0]),
        var_k=jnp.array([1.0]),
    )

    solution = refit.fit_refit_grid(
        beta_hat=beta_hat,
        s2=s2,
        maf_masks=maf_masks,
        init=init,
        config=RefitConfig(max_iter=5, step_size=0.5),
    )

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert len(solution.value) == 2


def test_refit_config_no_longer_exposes_solver_backend_knob():
    with pytest.raises(TypeError):
        cast(Any, RefitConfig)(solver_backend="optimistix")
