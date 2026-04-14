import importlib.util
import logging

import numpy as np
import polars as pl
import pytest

import mut_var
import mut_var.cli as cli
import mut_var.numerics as numerics

from mut_var.io import to_inference_arrays
from mut_var.pipelines import (
    run_inference_pipeline as run_inference_dataframe_pipeline,
    run_simulation_pipeline,
)
from mut_var.types import InferenceConfig, RESULTS, SimulationConfig, Solution


def test_run_inference_pipeline_returns_dataframe(sumstats_valid_df):
    path = "tests/fixtures/sumstats_valid.tsv"
    result_df = run_inference_dataframe_pipeline(
        path,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]


def test_run_inference_pipeline_logs_numerics_stages(sumstats_valid_df, caplog):
    caplog.set_level(logging.INFO, logger="mut_var.pipelines.inference")

    run_inference_dataframe_pipeline(
        "tests/fixtures/sumstats_valid.tsv",
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.pipelines.inference"]
    assert any("fitting baseline model" in message for message in messages)
    assert any("fitting refit grid" in message for message in messages)
    assert any("building numerics payload" in message for message in messages)


def test_run_inference_pipeline_logs_solver_steps_at_debug(sumstats_valid_df, caplog):
    caplog.set_level(logging.DEBUG, logger="mut_var.pipelines.inference")

    run_inference_dataframe_pipeline(
        "tests/fixtures/sumstats_valid.tsv",
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.pipelines.inference"]
    assert any("baseline | Step:" in message for message in messages)
    assert any("refit | Step:" in message for message in messages)


def test_run_inference_pipeline_raises_on_critical_numerics_result(sumstats_valid_df, monkeypatch):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    monkeypatch.setattr(
        mixture_fit_module,
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
            "tests/fixtures/sumstats_valid.tsv",
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=3, max_iter=2),
        )

    assert "non-finite" in str(err.value)


def test_run_inference_pipeline_raises_on_empty_subset_result(sumstats_valid_df, monkeypatch):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    baseline_params = mixture_fit_module.Params(
        pi=np.asarray([1.0], dtype=float),
        mu_k=np.asarray([], dtype=float),
        var_k=np.asarray([], dtype=float),
    )
    fit_state = mixture_fit_module.FitState(
        likelihood_matrix=np.ones((4, 1), dtype=float),
        initial_params=baseline_params,
    )

    monkeypatch.setattr(
        mixture_fit_module,
        "prepare_fit_state",
        lambda **_kwargs: Solution(
            value=fit_state,
            result=RESULTS.successful,
            stats={},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_baseline",
        lambda **_kwargs: Solution(
            value=baseline_params,
            result=RESULTS.successful,
            stats={},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_refit_step",
        lambda **_kwargs: Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "empty subset"},
            state=None,
        ),
    )

    with pytest.raises(ValueError) as err:
        run_inference_dataframe_pipeline(
            "tests/fixtures/sumstats_valid.tsv",
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=1, max_iter=2),
        )

    assert "empty subset" in str(err.value)


def test_orchestration_is_separate_from_cli_internals():
    assert not hasattr(numerics, "run_inference_pipeline")
    assert not hasattr(cli, "penalized_objective")
    assert not hasattr(cli, "fit_mixture")


def test_adapters_convert_to_arrays(sumstats_valid_df):
    arrays = to_inference_arrays(sumstats_valid_df, "effect_allele_frequency", "beta", "standard_error")
    assert not hasattr(arrays.beta_hat, "columns")


def test_run_inference_pipeline_accepts_path_input():
    result_df = run_inference_dataframe_pipeline(
        "tests/fixtures/sumstats_valid.tsv",
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]


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
    import mut_var.types as config_module

    assert importlib.util.find_spec("mut_var.numerics.pipeline") is None
    assert config_module.InferenceConfig is InferenceConfig
    assert not hasattr(numerics, "InferenceArrays")
    assert not hasattr(numerics, "InferenceConfig")
    assert not hasattr(numerics, "run_inference_pipeline")


def test_simulated_observed_output_is_accepted_by_run_inference_pipeline(monkeypatch, tmp_path):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    baseline_params = mixture_fit_module.Params(
        pi=np.asarray([0.9, 0.1], dtype=float),
        mu_k=np.asarray([0.0], dtype=float),
        var_k=np.asarray([1e-4], dtype=float),
    )
    fit_state = mixture_fit_module.FitState(
        likelihood_matrix=np.ones((128, 2), dtype=float),
        initial_params=baseline_params,
    )

    monkeypatch.setattr(
        mixture_fit_module,
        "prepare_fit_state",
        lambda **_kwargs: Solution(
            value=fit_state,
            result=RESULTS.successful,
            stats={},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_baseline",
        lambda **_kwargs: Solution(
            value=baseline_params,
            result=RESULTS.successful,
            stats={"objective": 0.0},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_refit_step",
        lambda **_kwargs: Solution(
            value=baseline_params,
            result=RESULTS.successful,
            stats={"epoch_count": 1},
            state=None,
        ),
    )

    artifacts = run_simulation_pipeline(
        config=SimulationConfig(
            n_rows=128,
            seed=0,
            weights=(0.9, 0.1),
            log_var_scales=(-8.0, -5.5),
        )
    )
    observed_path = tmp_path / "simulated_observed.tsv"
    artifacts.observed.write_csv(observed_path, separator="\t")

    result_df = run_inference_dataframe_pipeline(
        str(observed_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=2, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
