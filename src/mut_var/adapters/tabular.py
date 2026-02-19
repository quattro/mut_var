from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import polars as pl

from mut_var.adapters.array_cache import ArrayConversionCache
from mut_var.numerics.pipeline import InferenceArrays


def to_inference_arrays(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> InferenceArrays:
    af = jnp.asarray(df[af_col].to_jax())
    beta_hat = jnp.asarray(df[beta_col].to_jax())
    std_err = jnp.asarray(df[se_col].to_jax())
    return InferenceArrays(af=af, beta_hat=beta_hat, s2=std_err**2)


def build_maf_masks(af: jax.Array, maf_grid: jax.Array) -> jax.Array:
    af_arr = jnp.asarray(af)
    maf_arr = jnp.asarray(maf_grid)
    return jnp.logical_and(
        af_arr[jnp.newaxis, :] >= maf_arr[:, jnp.newaxis],
        af_arr[jnp.newaxis, :] <= (1.0 - maf_arr[:, jnp.newaxis]),
    )


def to_inference_arrays_cached(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
    cache: ArrayConversionCache,
) -> tuple[InferenceArrays, bool]:
    return cache.get_or_create(
        df=df,
        af_col=af_col,
        beta_col=beta_col,
        se_col=se_col,
        converter=to_inference_arrays,
    )


def payload_to_long_dataframe(payload: Mapping[str, object]) -> pl.DataFrame:
    columns = {}
    for name, values in payload.items():
        if isinstance(values, (list, tuple)):
            columns[name] = values
        else:
            columns[name] = jnp.asarray(values).tolist()
    df = pl.DataFrame(columns)
    return df.select(["mu0", "var0", "maf", "name", "value"])
