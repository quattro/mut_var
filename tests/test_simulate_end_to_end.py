import polars as pl

from mut_var.infer import InferenceConfig, run_inference_pipeline
from mut_var.numerics import SimulationNumericsConfig
from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig


def test_run_simulation_pipeline_outputs_feed_inference_pipeline_smoke():
    artifacts = run_simulation_pipeline(
        config=SimulationPipelineConfig(
            n_rows=256,
            seed=0,
            numerics=SimulationNumericsConfig(
                weights=(0.9, 0.1),
                log_var_scales=(-8.0, -5.5),
            ),
        )
    )

    result_df = run_inference_pipeline(
        artifacts.observed,
        seed=0,
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=2, max_iter=5, step_size=0.5),
    )

    assert isinstance(artifacts.truth, pl.DataFrame)
    assert isinstance(artifacts.observed, pl.DataFrame)
    assert isinstance(artifacts.metadata, pl.DataFrame)
    assert artifacts.truth.height > 0
    assert artifacts.observed.height > 0
    assert artifacts.metadata.height == 10
    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
