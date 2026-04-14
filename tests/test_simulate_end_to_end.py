import polars as pl

from mut_var.pipelines import (
    run_inference_pipeline,
    run_simulation_pipeline,
)
from mut_var.types import InferenceConfig, SimulationConfig


def test_run_simulation_pipeline_outputs_feed_inference_pipeline_smoke(tmp_path):
    artifacts = run_simulation_pipeline(
        config=SimulationConfig(
            n_rows=256,
            seed=0,
            weights=(0.9, 0.1),
            log_var_scales=(-8.0, -5.5),
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

    assert isinstance(artifacts.truth, pl.DataFrame)
    assert isinstance(artifacts.observed, pl.DataFrame)
    assert isinstance(artifacts.metadata, pl.DataFrame)
    assert artifacts.truth.height > 0
    assert artifacts.observed.height > 0
    assert artifacts.metadata.height == 10
    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
