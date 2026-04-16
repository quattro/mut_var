from __future__ import annotations

import importlib

import pytest


def test_types_module_exposes_shared_contracts():
    from mut_var.types import InferenceConfig, RESULTS, SimulationPipelineConfig, Solution

    assert RESULTS is not None
    assert Solution is not None
    assert InferenceConfig is not None
    assert SimulationPipelineConfig is not None
    assert not hasattr(InferenceConfig, "to_baseline_config")
    assert not hasattr(InferenceConfig, "to_refit_config")


def test_pipelines_package_exports_canonical_entrypoints():
    from mut_var.pipelines import run_curve_pipeline, run_inference_pipeline, run_simulation_pipeline

    assert callable(run_inference_pipeline)
    assert callable(run_curve_pipeline)
    assert callable(run_simulation_pipeline)


def test_io_module_exposes_ingress_helpers():
    from mut_var.io import (
        build_maf_masks,
        InferenceArrays,
        load_inference_arrays,
        payload_to_long_dataframe,
        to_inference_arrays,
    )

    assert InferenceArrays is not None
    assert callable(load_inference_arrays)
    assert callable(to_inference_arrays)
    assert callable(build_maf_masks)
    assert callable(payload_to_long_dataframe)


def test_benchmark_module_imports_with_current_io_surface():
    module = importlib.import_module("benchmarks.infer_runtime")

    assert hasattr(module, "build_report")
    assert hasattr(module, "main")


@pytest.mark.parametrize(
    "module_name",
    [
        "mut_var." + "contracts",
        "mut_var." + "adapters",
        "mut_var.numerics.baseline",
        "mut_var.numerics.refit",
    ],
)
def test_legacy_layout_modules_are_removed(module_name: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
