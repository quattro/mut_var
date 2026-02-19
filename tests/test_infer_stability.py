from __future__ import annotations

import numpy as np
import polars as pl

from benchmarks.infer_runtime import RuntimeBenchmarkConfig, build_report, generate_sumstats
from mut_var.adapters.array_cache import ArrayConversionCache
from mut_var.adapters.tabular import build_maf_masks, to_inference_arrays, to_inference_arrays_cached
from mut_var.contracts import RESULTS, Solution
from mut_var.numerics.pipeline import InferenceConfig, run_inference_pipeline
from mut_var.numerics.profiling import evaluate_performance_gate, profile_solution_runs


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "effect_allele_frequency": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
            "beta": [0.02, -0.03, 0.01, 0.04, -0.01, 0.03, -0.02, 0.01],
            "standard_error": [0.04, 0.03, 0.05, 0.02, 0.04, 0.03, 0.04, 0.02],
        }
    )


def test_array_cache_reuses_and_invalidates_cache():
    df = _sample_df()
    cache = ArrayConversionCache()

    first, first_hit = to_inference_arrays_cached(
        df,
        "effect_allele_frequency",
        "beta",
        "standard_error",
        cache,
    )
    second, second_hit = to_inference_arrays_cached(
        df,
        "effect_allele_frequency",
        "beta",
        "standard_error",
        cache,
    )

    assert not first_hit
    assert second_hit
    assert first is second
    assert cache.size == 1

    cache.invalidate()
    third, third_hit = to_inference_arrays_cached(
        df,
        "effect_allele_frequency",
        "beta",
        "standard_error",
        cache,
    )
    assert not third_hit
    assert third is not second


def test_benchmark_dataset_is_reproducible_with_cache_config():
    config = RuntimeBenchmarkConfig(
        seed=7,
        num_rows=16,
        num_clusters=3,
        max_iter=1,
        num_breaks=4,
        lowest=1e-4,
        highest=1e-2,
        batch_size=16,
        step_size=0.5,
        filter_threshold=1e-8,
        penalty=1.0,
        steady_runs=1,
        legacy_conversion_repeats=2,
        improvement_threshold_percent=20.0,
    )

    left = generate_sumstats(config)
    right = generate_sumstats(config)
    assert left.to_dicts() == right.to_dicts()


def test_refit_retrace_diagnostics_have_stable_likelihood_shapes_retrace():
    df = _sample_df()
    arrays = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
    maf_grid = np.array([1e-3, 5e-3, 1e-2])
    maf_masks = build_maf_masks(arrays.af, maf_grid)

    solution = run_inference_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=0,
        config=InferenceConfig(num_clusters=4, max_iter=2, batch_size=8, step_size=0.5),
    )

    assert solution.result in {RESULTS.successful, RESULTS.max_steps_reached}
    refit_stats = solution.stats["refit"]
    diagnostics = refit_stats["threshold_diagnostics"]
    assert len(diagnostics) == len(maf_grid)

    shapes = {tuple(diag["likelihood_shape"]) for diag in diagnostics}
    assert len(shapes) == 1
    assert {diag["threshold_index"] for diag in diagnostics} == {0, 1, 2}


def test_profile_solution_runs_separates_compile_and_steady_state_profiling():
    def _ok_solution() -> Solution:
        return Solution(value=None, result=RESULTS.successful, stats={}, state=None)

    profile = profile_solution_runs(_ok_solution, steady_runs=2)

    assert set(profile) == {"compile", "steady_state"}
    assert profile["compile"]["count"] == 1
    assert profile["compile"]["elapsed_seconds"] >= 0.0
    assert profile["steady_state"]["runs"] == 2
    assert len(profile["steady_state"]["elapsed_seconds"]) == 2


def test_performance_gate_fails_when_improvement_below_threshold_profiling():
    gate = evaluate_performance_gate(
        baseline_seconds=10.0,
        candidate_seconds=9.0,
        threshold_percent=20.0,
    )

    assert gate.improvement_percent == 10.0
    assert not gate.passed


def test_benchmark_report_contains_compile_and_steady_sections_profiling():
    config = RuntimeBenchmarkConfig(
        seed=11,
        num_rows=64,
        num_clusters=3,
        max_iter=1,
        num_breaks=3,
        lowest=1e-4,
        highest=1e-2,
        batch_size=64,
        step_size=0.5,
        filter_threshold=1e-8,
        penalty=1.0,
        steady_runs=1,
        legacy_conversion_repeats=3,
        improvement_threshold_percent=20.0,
    )

    report = build_report(config)

    assert "compile" in report
    assert "steady_state" in report
    assert "comparison" in report
    assert "passed" in report
    assert report["compile"]["baseline"]["count"] == 1
    assert report["steady_state"]["baseline"]["runs"] == 1
