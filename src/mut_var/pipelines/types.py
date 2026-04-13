from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from mut_var.numerics.simulate import SimulationNumericsConfig


class InferenceArrays(NamedTuple):
    af: np.ndarray
    beta_hat: np.ndarray
    s2: np.ndarray


class InferenceConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-5
    step_size: float = 0.01
    filter_threshold: float = 1e-8
    penalty: float = 1.0

    def to_baseline_config(self):
        r"""Convert pipeline controls to baseline-stage solver config."""
        from mut_var.numerics.baseline import BaselineConfig

        return BaselineConfig(
            num_clusters=self.num_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )

    def to_refit_config(self):
        r"""Convert pipeline controls to refit-stage solver config."""
        from mut_var.numerics.refit import RefitConfig

        return RefitConfig(
            penalty=self.penalty,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )


class SimulationPipelineConfig(NamedTuple):
    n_rows: int
    seed: int = 0
    numerics: SimulationNumericsConfig = SimulationNumericsConfig(
        weights=(0.95, 0.05),
        log_var_scales=(-8.0, -5.5),
    )


class SimulationArtifacts(NamedTuple):
    truth: pl.DataFrame
    observed: pl.DataFrame
    metadata: pl.DataFrame


__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "SimulationArtifacts",
    "SimulationPipelineConfig",
]
