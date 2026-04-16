# pattern: Functional Core
from mut_var.pipelines.curve import run_curve_pipeline
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.pipelines.simulation import run_simulation_pipeline, SimulationArtifacts

__all__ = [
    "run_curve_pipeline",
    "run_inference_pipeline",
    "run_simulation_pipeline",
    "SimulationArtifacts",
]
