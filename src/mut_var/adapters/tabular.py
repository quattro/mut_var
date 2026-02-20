from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
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

    - `InferenceArrays` with JAX arrays for AF, beta, and variance (`se^2`).
    """
    af = jnp.asarray(df[af_col].to_jax())
    beta_hat = jnp.asarray(df[beta_col].to_jax())
    std_err = jnp.asarray(df[se_col].to_jax())
    return InferenceArrays(af=af, beta_hat=beta_hat, s2=std_err**2)


def build_maf_masks(af: jax.Array, maf_grid: jax.Array) -> jax.Array:
    r"""Build per-threshold boolean masks over observations."""
    af_arr = jnp.asarray(af)
    maf_arr = jnp.asarray(maf_grid)
    return jnp.logical_and(
        af_arr[jnp.newaxis, :] >= maf_arr[:, jnp.newaxis],
        af_arr[jnp.newaxis, :] <= (1.0 - maf_arr[:, jnp.newaxis]),
    )


def payload_to_long_dataframe(payload: Mapping[str, object]) -> pl.DataFrame:
    r"""Normalize numerics payload mappings into the long-format output dataframe."""
    columns = {}
    for name, values in payload.items():
        if isinstance(values, (list, tuple)):
            columns[name] = values
        else:
            columns[name] = jnp.asarray(values).tolist()
    df = pl.DataFrame(columns)
    return df.select(["mu0", "var0", "maf", "name", "value"])
