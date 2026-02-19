import jax.numpy as jnp
import jax.random as rdm
import polars as pl

import mut_var.cli as cli

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics import baseline, refit
from mut_var.numerics._optimize import OptimizationLoopConfig, run_iterative_optimization
from mut_var.numerics.baseline import BaselineConfig, fit_baseline
from mut_var.numerics.refit import _fit_single_refit, RefitConfig


def test_fit_baseline_returns_structured_solution_on_valid_arrays():
    beta_hat = jnp.array([0.05, -0.03, 0.01, 0.02, -0.01])
    s2 = jnp.array([0.01, 0.015, 0.02, 0.013, 0.011])

    solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(0),
        config=BaselineConfig(num_clusters=3, batch_size=5, max_iter=5, step_size=0.5),
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


def test_fit_single_refit_returns_last_accepted_params_when_backtracking_never_accepts():
    init = baseline.Params(
        pi=jnp.array([0.7, 0.3]),
        mu_k=jnp.array([0.0]),
        var_k=jnp.array([1.0]),
    )
    likelihoods = jnp.array([[0.2, 0.8], [0.4, 0.6]], dtype=jnp.float64)
    weights = jnp.array([1.0, 1.0], dtype=jnp.float64)

    def _vg_f(*_args, **_kwargs):
        return jnp.array(0.0), baseline.Params(
            pi=jnp.array([0.2, -0.2]),
            mu_k=jnp.array([0.0]),
            var_k=jnp.array([0.0]),
        )

    def _always_reject_obj(*_args, **_kwargs):
        return -jnp.inf

    solution = _fit_single_refit(
        likelihoods=likelihoods,
        weights=weights,
        init=init,
        config=RefitConfig(max_iter=1, step_size=0.75, tol=1e-8),
        vg_f=_vg_f,
        obj=_always_reject_obj,
    )

    assert solution.result == RESULTS.max_steps_reached
    assert bool(jnp.allclose(solution.value.pi, init.pi))


def test_run_iterative_optimization_respects_step_schedule_and_convergence_metric():
    result = run_iterative_optimization(
        init_params=jnp.asarray(0.0),
        init_objective=jnp.asarray(0.0),
        key=None,
        config=OptimizationLoopConfig(max_iter=5, tol=0.3, step_size=1.0, max_backtracks=1),
        make_epoch_context=lambda _epoch, _params, key: (None, key),
        compute_direction=lambda _params, _ctx: jnp.asarray(1.0),
        propose_candidate=lambda params, direction, step_size: params + direction * step_size,
        evaluate_objective=lambda candidate, _ctx: candidate,
        step_size_for_epoch=lambda epoch, base_step: base_step * (0.5**epoch),
        should_backtrack_step=lambda _diff, _objective: False,
        progress_metric=lambda diff, _objective: diff,
    )

    assert result.result == RESULTS.successful
    assert result.converged
    assert result.epoch_count == 3
    assert float(result.params) == 1.75


def test_run_iterative_optimization_keeps_last_accepted_params_when_all_candidates_reject():
    init = jnp.asarray(2.0)

    result = run_iterative_optimization(
        init_params=init,
        init_objective=jnp.asarray(5.0),
        key=None,
        config=OptimizationLoopConfig(max_iter=2, tol=1e-9, step_size=0.5, max_backtracks=3),
        make_epoch_context=lambda _epoch, _params, key: (None, key),
        compute_direction=lambda _params, _ctx: jnp.asarray(1.0),
        propose_candidate=lambda params, direction, step_size: params + direction * step_size,
        evaluate_objective=lambda _candidate, _ctx: jnp.asarray(-jnp.inf),
        step_size_for_epoch=lambda _epoch, base_step: base_step,
        should_backtrack_step=lambda _diff, _objective: True,
        progress_metric=lambda diff, _objective: diff,
    )

    assert result.result == RESULTS.max_steps_reached
    assert float(result.params) == float(init)
