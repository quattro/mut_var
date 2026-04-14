from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl


class InferenceArrays(NamedTuple):
    af: np.ndarray
    beta_hat: np.ndarray
    s2: np.ndarray


class SimulationArtifacts(NamedTuple):
    truth: pl.DataFrame
    observed: pl.DataFrame
    metadata: pl.DataFrame


__all__ = [
    "InferenceArrays",
    "SimulationArtifacts",
]
