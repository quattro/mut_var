# SPDX-FileCopyrightText: 2025-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
# pattern: Functional Core

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .pipelines import run_curve_pipeline, run_inference_pipeline, run_simulation_pipeline, SimulationArtifacts
from .types import SimulationPipelineConfig

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "__version__",
    "run_curve_pipeline",
    "run_inference_pipeline",
    "run_simulation_pipeline",
    "SimulationPipelineConfig",
    "SimulationArtifacts",
]
