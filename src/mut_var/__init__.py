# SPDX-FileCopyrightText: 2025-present Nicholas Mancuso <nmancuso@usc.edu>
#
# SPDX-License-Identifier: MIT
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from .contracts import RESULTS, Solution
from .curve import run_curve_workflow
from .infer import InferenceArrays, InferenceConfig, run_inference_pipeline
from .numerics import fit_baseline, fit_curve, fit_refit_grid, run_profiled_inference_pipeline

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "RESULTS",
    "Solution",
    "__version__",
    "fit_baseline",
    "fit_curve",
    "fit_refit_grid",
    "run_curve_workflow",
    "run_inference_pipeline",
    "run_profiled_inference_pipeline",
]
