import jax.numpy as jnp
import polars as pl
import pytest

import mut_var
import mut_var.cli as cli

from mut_var.adapters.tabular import build_maf_masks, to_inference_arrays
from mut_var.contracts import RESULTS, Solution
from mut_var.infer import run_inference_pipeline as run_inference_dataframe_pipeline
from mut_var.numerics.pipeline import (
    InferenceArrays,
    InferenceConfig,
    run_inference_pipeline as run_inference_numerics_pipeline,
)


def test_run_inference_numerics_pipeline_returns_solution_with_status_and_stats(sumstats_valid_df):
    df = sumstats_valid_df
    arrays = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
    maf_grid = jnp.asarray([1e-3, 5e-3])
    maf_masks = build_maf_masks(arrays.af, maf_grid)

    solution = run_inference_numerics_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=0,
        config=InferenceConfig(num_clusters=3, max_iter=5, batch_size=8, step_size=0.5),
    )

    assert isinstance(solution, Solution)
    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert isinstance(solution.stats, dict)
    assert "num_models" in solution.stats
    assert solution.value is not None
    assert set(solution.value) == {"mu0", "var0", "maf", "name", "value"}


def test_run_inference_pipeline_returns_dataframe(sumstats_valid_df):
    result_df = run_inference_dataframe_pipeline(
        sumstats_valid_df,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=3, max_iter=5, batch_size=8, step_size=0.5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]


def test_run_inference_pipeline_raises_on_critical_numerics_result(sumstats_valid_df, monkeypatch):
    import mut_var.infer as infer_module

    monkeypatch.setattr(
        infer_module,
        "_run_numerics_inference_pipeline",
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
            config=InferenceConfig(num_clusters=3, max_iter=2, batch_size=8, step_size=0.5),
        )

    assert "non-finite" in str(err.value)


def test_orchestration_is_separate_from_cli_internals():
    assert callable(run_inference_numerics_pipeline)
    assert not hasattr(cli, "penalized_objective")
    assert not hasattr(cli, "fit_mixture")


def test_pipeline_rejects_tabular_payloads_and_adapters_convert_to_arrays(sumstats_valid_df):
    df = sumstats_valid_df
    arrays = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
    assert not hasattr(arrays.beta_hat, "columns")

    bad_arrays = InferenceArrays(af=df, beta_hat=df, s2=df)
    maf_grid = jnp.asarray([1e-3, 5e-3])
    maf_masks = jnp.asarray([[True, True, True, True, True, True, True, True]])

    solution = run_inference_numerics_pipeline(
        arrays=bad_arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=0,
        config=InferenceConfig(num_clusters=3, max_iter=2, batch_size=8),
    )

    assert solution.result == RESULTS.invalid_input


def test_pipeline_returns_empty_subset_when_masks_select_nothing(sumstats_valid_df):
    arrays = to_inference_arrays(sumstats_valid_df, "effect_allele_frequency", "beta", "standard_error")
    maf_grid = jnp.asarray([0.49, 0.5])
    maf_masks = jnp.zeros((2, len(arrays.af)), dtype=bool)

    solution = run_inference_numerics_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=0,
        config=InferenceConfig(num_clusters=3, max_iter=2, batch_size=4),
    )

    assert solution.result == RESULTS.empty_subset


def test_package_root_exports_canonical_pipeline_entrypoints():
    assert callable(mut_var.run_inference_pipeline)
    assert callable(mut_var.run_curve_pipeline)
