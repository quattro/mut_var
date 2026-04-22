import importlib.util
import logging

import jax.numpy as jnp
import polars as pl
import pytest

import mut_var
import mut_var.cli as cli
import mut_var.numerics as numerics

from mut_var.io import to_inference_arrays
from mut_var.numerics import SimulationNumericsConfig
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.pipelines.simulation import run_simulation_pipeline
from mut_var.types import InferenceConfig, RESULTS, SimulationPipelineConfig, Solution


def test_run_inference_pipeline_returns_dataframe(sumstats_valid_path):
    result_df = run_inference_pipeline(
        str(sumstats_valid_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]


def test_run_inference_pipeline_reports_min_observed_maf_for_baseline(sumstats_valid_path):
    result_df = run_inference_pipeline(
        str(sumstats_valid_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    baseline_maf = result_df.filter(pl.col("name") == "pi0").get_column("maf").unique()

    assert baseline_maf.len() == 1
    assert baseline_maf.item() == pytest.approx(0.1)


def test_run_inference_pipeline_logs_numerics_stages(sumstats_valid_path, caplog):
    caplog.set_level(logging.INFO, logger="mut_var.pipelines.inference")

    run_inference_pipeline(
        str(sumstats_valid_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.pipelines.inference"]
    assert any("fitting baseline model" in message for message in messages)
    assert any("fitting refit grid" in message for message in messages)
    assert any("building numerics payload" in message for message in messages)


def test_run_inference_pipeline_logs_solver_steps_at_debug(sumstats_valid_path, caplog):
    caplog.set_level(logging.DEBUG, logger="mut_var.pipelines.inference")

    run_inference_pipeline(
        str(sumstats_valid_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5),
    )

    messages = [record.getMessage() for record in caplog.records if record.name == "mut_var.pipelines.inference"]
    assert any("baseline | Step:" in message for message in messages)
    assert any("refit 1 | Step:" in message for message in messages)


def test_run_inference_pipeline_raises_on_critical_numerics_result(sumstats_valid_path, monkeypatch):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    monkeypatch.setattr(
        mixture_fit_module,
        "fit_baseline",
        lambda *_args, **_kwargs: Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "objective became non-finite"},
            state=None,
        ),
    )

    with pytest.raises(RuntimeError) as err:
        run_inference_pipeline(
            str(sumstats_valid_path),
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=3, max_iter=2),
        )

    assert "non-finite" in str(err.value)


def test_run_inference_pipeline_raises_on_empty_subset_result(sumstats_valid_path, monkeypatch):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    monkeypatch.setattr(
        mixture_fit_module,
        "fit_refit_masked_step",
        lambda *_args, **_kwargs: Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "empty subset"},
            state=None,
        ),
    )

    with pytest.raises(ValueError) as err:
        run_inference_pipeline(
            str(sumstats_valid_path),
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=2, max_iter=2),
        )

    assert "empty subset" in str(err.value)


def test_run_inference_pipeline_passes_arrays_into_numerics(monkeypatch, sumstats_valid_path):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    seen = {}

    def _capture_prepare_fit_state(beta_hat, s2, config):
        seen["beta_hat_is_dataframe"] = isinstance(beta_hat, pl.DataFrame)
        seen["s2_is_dataframe"] = isinstance(s2, pl.DataFrame)
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "stop after ingress"},
            state=None,
        )

    monkeypatch.setattr(mixture_fit_module, "prepare_fit_state", _capture_prepare_fit_state)

    with pytest.raises(ValueError) as err:
        run_inference_pipeline(
            str(sumstats_valid_path),
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=2, max_iter=2),
        )

    assert "stop after ingress" in str(err.value)
    assert seen == {"beta_hat_is_dataframe": False, "s2_is_dataframe": False}


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
    import mut_var.pipelines.inference as infer_module

    assert importlib.util.find_spec("mut_var.numerics.pipeline") is None
    assert not hasattr(infer_module, "run_numerics_inference_pipeline")
    assert "InferenceArrays" not in getattr(numerics, "__all__", ())
    assert "InferenceConfig" not in getattr(numerics, "__all__", ())
    assert not hasattr(numerics, "InferenceArrays")
    assert not hasattr(numerics, "InferenceConfig")
    assert "InferenceArrays" not in getattr(infer_module, "__all__", ())
    assert "InferenceConfig" not in getattr(infer_module, "__all__", ())
    assert not hasattr(infer_module, "InferenceArrays")
    assert not hasattr(numerics, "run_inference_pipeline")


def test_run_inference_pipeline_raises_file_not_found_for_missing_path(tmp_path):
    missing_path = tmp_path / "missing.tsv"

    with pytest.raises(FileNotFoundError) as err:
        run_inference_pipeline(
            str(missing_path),
            lowest=1e-3,
            highest=5e-3,
            num_breaks=2,
            config=InferenceConfig(num_clusters=2, max_iter=2),
        )

    assert str(missing_path) in str(err.value)


def test_simulated_observed_output_is_accepted_by_run_inference_pipeline(monkeypatch, tmp_path):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    fake_params = mixture_fit_module.Params(
        pi=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
        mu_k=jnp.asarray([0.0], dtype=jnp.float64),
        var_k=jnp.asarray([1e-4], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        mixture_fit_module,
        "fit_baseline",
        lambda *_args, **_kwargs: Solution(
            value=fake_params,
            result=RESULTS.successful,
            stats={"n_steps": 1},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_refit_step",
        lambda *_args, **_kwargs: Solution(
            value=fake_params,
            result=RESULTS.successful,
            stats={"n_steps": 1},
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

    observed_path = tmp_path / "observed.tsv"
    artifacts.observed.write_csv(observed_path, separator="\t")

    result_df = run_inference_pipeline(
        str(observed_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=2, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
