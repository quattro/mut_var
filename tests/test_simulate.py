from __future__ import annotations

import polars as pl
import pytest

import mut_var

from mut_var.config import SimulationConfig
from mut_var.contracts import RESULTS, Solution
from mut_var.pipelines import (
    run_simulation_pipeline,
    SimulationArtifacts,
)


def _pipeline_config(n_rows: int = 200) -> SimulationConfig:
    return SimulationConfig(
        n_rows=n_rows,
        seed=0,
        weights=(0.9, 0.1),
        log_var_scales=(-8.0, -5.5),
        variance_link="maf_power",
        theta=0.5,
        se_model="af_n_scaled",
        sample_size=50000.0,
        se_scale=1.0,
    )


def test_package_root_exports_simulation_pipeline_entrypoint():
    assert callable(mut_var.run_simulation_pipeline)


def test_run_simulation_pipeline_returns_three_dataframe_artifacts():
    artifacts = run_simulation_pipeline(config=_pipeline_config())

    assert isinstance(artifacts, SimulationArtifacts)
    assert isinstance(artifacts.truth, pl.DataFrame)
    assert isinstance(artifacts.observed, pl.DataFrame)
    assert isinstance(artifacts.metadata, pl.DataFrame)
    assert artifacts.truth.columns == ["row_id", "component", "beta_true", "sigma2", "effect_allele_frequency"]
    assert artifacts.observed.columns == ["row_id", "effect_allele_frequency", "beta", "standard_error"]
    assert artifacts.metadata.columns == [
        "seed",
        "n_rows",
        "num_components",
        "variance_link",
        "theta",
        "af_decile",
        "empirical_var_beta_true",
        "empirical_mean_sigma2",
    ]


def test_truth_and_observed_row_ids_align():
    artifacts = run_simulation_pipeline(config=_pipeline_config())
    truth_ids = artifacts.truth.get_column("row_id")
    observed_ids = artifacts.observed.get_column("row_id")

    assert truth_ids.to_list() == observed_ids.to_list()
    assert truth_ids.to_list() == list(range(artifacts.truth.height))


def test_metadata_contains_expected_decile_rows():
    artifacts = run_simulation_pipeline(config=_pipeline_config(n_rows=1000))
    meta = artifacts.metadata.sort("af_decile")

    assert meta.height == 10
    assert meta.get_column("af_decile").to_list() == list(range(10))
    assert bool(meta.get_column("empirical_var_beta_true").is_finite().all())
    assert bool(meta.get_column("empirical_mean_sigma2").is_finite().all())


def test_pipeline_raises_value_error_for_invalid_input_status(monkeypatch):
    def _bad_numerics(*, config):  # noqa: ARG001
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "invalid simulation config"},
            state=None,
        )

    monkeypatch.setattr("mut_var.pipelines.simulation.simulate_mixture_data", _bad_numerics)

    with pytest.raises(ValueError, match="invalid simulation config"):
        run_simulation_pipeline(config=_pipeline_config())


def test_pipeline_raises_runtime_error_for_nonrecoverable_status(monkeypatch):
    def _bad_numerics(*, config):  # noqa: ARG001
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "simulation became non-finite"},
            state=None,
        )

    monkeypatch.setattr("mut_var.pipelines.simulation.simulate_mixture_data", _bad_numerics)

    with pytest.raises(RuntimeError, match="simulation became non-finite"):
        run_simulation_pipeline(config=_pipeline_config())
