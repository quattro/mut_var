from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl

from mut_var.infer import InferenceArrays


def to_inference_arrays(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> InferenceArrays:
    r"""**Arguments:**

    - `df`: Validated summary-statistics dataframe.
    - `af_col`: Effect-allele-frequency column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.

    **Returns:**

    - `InferenceArrays` with numpy arrays for AF, beta, and variance (``se^2``).
    """
    af = np.asarray(df[af_col].to_numpy(), dtype=float)
    beta_hat = np.asarray(df[beta_col].to_numpy(), dtype=float)
    std_err = np.asarray(df[se_col].to_numpy(), dtype=float)
    return InferenceArrays(af=af, beta_hat=beta_hat, s2=std_err**2)


def build_maf_masks(af: np.ndarray, maf_grid: np.ndarray) -> np.ndarray:
    r"""Build per-threshold boolean masks over observations."""
    af_arr = np.asarray(af)
    maf_arr = np.asarray(maf_grid)
    return np.logical_and(
        af_arr[np.newaxis, :] >= maf_arr[:, np.newaxis],
        af_arr[np.newaxis, :] <= (1.0 - maf_arr[:, np.newaxis]),
    )


def payload_to_long_dataframe(payload: Mapping[str, object]) -> pl.DataFrame:
    r"""Normalize numerics payload mappings into the long-format output dataframe."""
    columns = {}
    for name, values in payload.items():
        if isinstance(values, (list, tuple)):
            columns[name] = values
        else:
            columns[name] = np.asarray(values).tolist()
    df = pl.DataFrame(columns)
    return df.select(["mu0", "var0", "maf", "name", "value"])
