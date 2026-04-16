import importlib.util
import logging

import jax.numpy as jnp
import polars as pl
import pytest

import mut_var
import mut_var.cli as cli
import mut_var.numerics as numerics

from mut_var.infer import InferenceConfig, run_inference_pipeline as run_inference_dataframe_pipeline
from mut_var.io import to_inference_arrays
from mut_var.numerics import SimulationNumericsConfig
from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig
from mut_var.types import RESULTS, Solution


def test_run_inference_pipeline_returns_dataframe(sumstats_valid_df):
    result_df = run_inference_dataframe_pipeline(
        sumstats_valid_df,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]


def test_run_inference_pipeline_logs_numerics_stages(sumstats_valid_df, caplog):
    caplog.set_level(logging.INFO, logger="mut_var.infer")

    run_inference_dataframe_pipeline(
        sumstats_valid_df,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.infer"]
    assert any("fitting baseline model" in message for message in messages)
    assert any("fitting refit grid" in message for message in messages)
    assert any("building numerics payload" in message for message in messages)


def test_run_inference_pipeline_logs_solver_steps_at_debug(sumstats_valid_df, caplog):
    caplog.set_level(logging.DEBUG, logger="mut_var.infer")

    run_inference_dataframe_pipeline(
        sumstats_valid_df,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5, step_size=0.5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.infer"]
    assert any("baseline | Step:" in message for message in messages)
    assert any("refit | Step:" in message for message in messages)


def test_run_inference_pipeline_raises_on_critical_numerics_result(sumstats_valid_df, monkeypatch):
    import mut_var.numerics.baseline as baseline_module

    monkeypatch.setattr(
        baseline_module,
        "fit_baseline",
        lambda **_kwargs: Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "objective became non-finite"},
            state=None,
        ),
    )

    with pytest.raises(RuntimeError) as err:
        run_inference_dataframe_pipeline(
            sumstats_valid_df,
            seed=0,
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=3, max_iter=2, step_size=0.5),
        )

    assert "non-finite" in str(err.value)


def test_run_inference_pipeline_raises_on_empty_subset_result(sumstats_valid_df, monkeypatch):
    import mut_var.numerics.baseline as baseline_module
    import mut_var.numerics.refit as refit_module

    baseline_params = baseline_module.Params(
        pi=jnp.asarray([1.0], dtype=jnp.float64),
        mu_k=jnp.asarray([], dtype=jnp.float64),
        var_k=jnp.asarray([], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        baseline_module,
        "fit_baseline",
        lambda **_kwargs: Solution(
            value=baseline_params,
            result=RESULTS.successful,
            stats={},
            state=None,
        ),
    )
    monkeypatch.setattr(
        refit_module,
        "fit_refit_grid",
        lambda **_kwargs: Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "empty subset"},
            state=None,
        ),
    )

    with pytest.raises(ValueError) as err:
        run_inference_dataframe_pipeline(
            sumstats_valid_df,
            seed=0,
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=1, max_iter=2, step_size=0.5),
        )

    assert "empty subset" in str(err.value)


def test_orchestration_is_separate_from_cli_internals():
    assert not hasattr(numerics, "run_inference_pipeline")
    assert not hasattr(cli, "penalized_objective")
    assert not hasattr(cli, "fit_mixture")


def test_adapters_convert_to_arrays(sumstats_valid_df):
    arrays = to_inference_arrays(sumstats_valid_df, "effect_allele_frequency", "beta", "standard_error")
    assert not hasattr(arrays.beta_hat, "columns")


def test_package_root_exports_canonical_pipeline_entrypoints():
    assert callable(mut_var.run_inference_pipeline)
    assert callable(mut_var.run_curve_pipeline)


def test_numerics_public_surface_does_not_export_profiling_helpers():
    assert not hasattr(numerics, "run_inference_pipeline")
    assert not hasattr(numerics, "run_profiled_inference_pipeline")
    assert not hasattr(numerics, "evaluate_performance_gate")
    assert not hasattr(numerics, "profile_solution_runs")
    assert not hasattr(numerics, "PerformanceGateResult")


def test_numerics_module_owns_numerics_entrypoint():
    import mut_var.infer as infer_module

    assert importlib.util.find_spec("mut_var.numerics.pipeline") is None
    assert not hasattr(infer_module, "run_numerics_inference_pipeline")
    assert infer_module.InferenceArrays is numerics.InferenceArrays
    assert infer_module.InferenceConfig is numerics.InferenceConfig
    assert not hasattr(numerics, "run_inference_pipeline")


def test_simulated_observed_output_is_accepted_by_run_inference_pipeline(monkeypatch):
    import mut_var.numerics.baseline as baseline_module
    import mut_var.numerics.refit as refit_module

    baseline_params = baseline_module.Params(
        pi=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
        mu_k=jnp.asarray([0.0], dtype=jnp.float64),
        var_k=jnp.asarray([1e-4], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        baseline_module,
        "fit_baseline",
        lambda **_kwargs: Solution(
            value=baseline_params,
            result=RESULTS.successful,
            stats={"objective": 0.0},
            state=None,
        ),
    )
    monkeypatch.setattr(
        refit_module,
        "fit_refit_grid",
        lambda **_kwargs: Solution(
            value=[baseline_params, baseline_params, baseline_params],
            result=RESULTS.successful,
            stats={"num_models": 3},
            state=None,
        ),
    )

    artifacts = run_simulation_pipeline(
        config=SimulationPipelineConfig(
            n_rows=128,
            seed=0,
            numerics=SimulationNumericsConfig(weights=(0.9, 0.1), log_var_scales=(-8.0, -5.5)),
        )
    )

    result_df = run_inference_dataframe_pipeline(
        artifacts.observed,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=2, max_iter=5, step_size=0.5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
