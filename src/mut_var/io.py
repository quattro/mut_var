from __future__ import annotations

# pattern: Imperative Shell
import hashlib

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import polars as pl

from jaxtyping import ArrayLike


class InferenceArrays(NamedTuple):
    af: ArrayLike
    beta_hat: ArrayLike
    s2: ArrayLike


def read_sumstats(path: str) -> pl.DataFrame:
    r"""Read tab-delimited summary statistics from disk.

    **Arguments:**

    - `path`: Input TSV path.

    **Returns:**

    - Loaded dataframe.

    **Raises:**

    - `FileNotFoundError`: Path does not exist.
    - `ValueError`: File cannot be parsed as expected TSV input.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")

    try:
        return pl.read_csv(path, separator="\t")
    except Exception as exc:
        raise ValueError(f"could not read summary statistics file '{path}': expected a tab-delimited file.") from exc


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


def load_inference_arrays(
    path: str,
    *,
    af_col: str = "effect_allele_frequency",
    beta_col: str = "beta",
    se_col: str = "standard_error",
) -> InferenceArrays:
    r"""Load and validate summary statistics from a TSV file, returning numeric arrays.

    **Arguments:**

    - `path`: Input TSV path.
    - `af_col`: Effect-allele-frequency column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.

    **Returns:**

    - `InferenceArrays` with JAX arrays for AF, beta, and variance (`se^2`).

    **Raises:**

    - `FileNotFoundError`: Path does not exist.
    - `ValueError`: Validation failed (missing columns, non-numeric values, domain violations).
    """
    df = read_sumstats(path)
    validate_required_columns(df, af_col, beta_col, se_col)
    validate_numeric_columns(df, af_col, beta_col, se_col)
    validate_sumstats_domain(df, af_col, se_col)
    return to_inference_arrays(df, af_col, beta_col, se_col)


def validate_required_columns(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> None:
    r"""Validate presence of required AF, beta, and standard-error columns."""
    required = [af_col, beta_col, se_col]
    missing = [name for name in required if name not in df.columns]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(
            f"Missing required column(s): {cols}. "
            f"Expected columns include AF='{af_col}', beta='{beta_col}', SE='{se_col}'."
        )


def _as_numeric_column(df: pl.DataFrame, col_name: str) -> pl.Series:
    series = df.get_column(col_name)
    if series.len() == 0:
        raise ValueError(f"column '{col_name}' is empty.")

    if series.null_count() > 0:
        raise ValueError(f"column '{col_name}' contains null values.")

    try:
        numeric = series.cast(pl.Float64, strict=True)
    except Exception as exc:
        raise ValueError(f"column '{col_name}' must contain numeric values only.") from exc

    if not bool(numeric.is_finite().all()):
        raise ValueError(f"column '{col_name}' contains non-finite numeric values.")
    return numeric


def validate_numeric_columns(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> None:
    r"""Validate that AF/beta/SE columns are numeric, non-null, and finite."""
    for name in (af_col, beta_col, se_col):
        _as_numeric_column(df, name)


def validate_sumstats_domain(df: pl.DataFrame, af_col: str, se_col: str) -> None:
    r"""Validate domain constraints (`AF in [0, 1]`, `SE > 0`)."""
    af = _as_numeric_column(df, af_col)
    se = _as_numeric_column(df, se_col)

    af_oor = int(((af < 0.0) | (af > 1.0)).sum())
    if af_oor > 0:
        raise ValueError(f"Column '{af_col}' must be within [0, 1]. Found {af_oor} out-of-range row(s).")

    se_nonpositive = int((se <= 0.0).sum())
    if se_nonpositive > 0:
        raise ValueError(f"Column '{se_col}' must be strictly positive. Found {se_nonpositive} non-positive row(s).")


def validate_maf_grid(lowest: Any, highest: Any, num_breaks: Any) -> None:
    r"""Validate MAF grid configuration at ingress boundaries."""
    try:
        lowest_val = float(lowest)
        highest_val = float(highest)
    except (TypeError, ValueError) as exc:
        raise ValueError("maf grid bounds must be numeric: expected 0 < lowest < highest <= 0.5.") from exc

    if not isinstance(num_breaks, int):
        raise ValueError("num_breaks must be an integer with value >= 2.")

    if lowest_val <= 0.0 or highest_val <= 0.0:
        raise ValueError("lowest and highest must both be > 0.")
    if highest_val > 0.5:
        raise ValueError("highest must be <= 0.5.")
    if lowest_val >= highest_val:
        raise ValueError("lowest must be strictly less than highest.")
    if num_breaks < 2:
        raise ValueError("num_breaks must be >= 2.")


def dataframe_fingerprint(df: pl.DataFrame, columns: list[str]) -> str:
    r"""Build a stable content fingerprint for selected dataframe columns."""
    hasher = hashlib.sha256()
    hasher.update(str(df.height).encode("utf-8"))
    hasher.update(str(df.width).encode("utf-8"))
    hasher.update(",".join(columns).encode("utf-8"))

    for col in columns:
        series = df.get_column(col)
        hasher.update(col.encode("utf-8"))
        hasher.update(str(series.dtype).encode("utf-8"))
        if series.dtype.is_numeric():
            arr = series.cast(pl.Float64, strict=False).fill_null(float("nan")).to_jax()
            hasher.update(arr.tobytes())
        else:
            values = series.cast(pl.Utf8, strict=False).fill_null("").to_list()
            for value in values:
                payload = value.encode("utf-8")
                hasher.update(len(payload).to_bytes(8, byteorder="little", signed=False))
                hasher.update(payload)

    return hasher.hexdigest()
