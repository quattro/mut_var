from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class MutVarInputError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class SumstatsReadError(MutVarInputError):
    pass


class MissingColumnsError(MutVarInputError):
    pass


class NumericColumnsError(MutVarInputError):
    pass


class SumstatsDomainError(MutVarInputError):
    pass


class MAFGridError(MutVarInputError):
    pass


def read_sumstats(path: str) -> pl.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise SumstatsReadError(f"Input file does not exist: {path}")

    try:
        return pl.read_csv(path, separator="\t")
    except Exception as exc:
        raise SumstatsReadError(
            f"Could not read summary statistics file '{path}'. Expected a tab-delimited file."
        ) from exc


def validate_required_columns(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> None:
    required = [af_col, beta_col, se_col]
    missing = [name for name in required if name not in df.columns]
    if missing:
        cols = ", ".join(missing)
        raise MissingColumnsError(
            f"Missing required column(s): {cols}. "
            f"Expected columns include AF='{af_col}', beta='{beta_col}', SE='{se_col}'."
        )


def _as_numeric_column(df: pl.DataFrame, col_name: str) -> pl.Series:
    series = df.get_column(col_name)
    if series.len() == 0:
        raise NumericColumnsError(f"Column '{col_name}' is empty.")

    if series.null_count() > 0:
        raise NumericColumnsError(f"Column '{col_name}' contains null values.")

    try:
        numeric = series.cast(pl.Float64, strict=True)
    except Exception as exc:
        raise NumericColumnsError(f"Column '{col_name}' must contain numeric values only.") from exc

    if not bool(numeric.is_finite().all()):
        raise NumericColumnsError(f"Column '{col_name}' contains non-finite numeric values.")
    return numeric


def validate_numeric_columns(
    df: pl.DataFrame,
    af_col: str,
    beta_col: str,
    se_col: str,
) -> None:
    for name in (af_col, beta_col, se_col):
        _as_numeric_column(df, name)


def validate_sumstats_domain(df: pl.DataFrame, af_col: str, se_col: str) -> None:
    af = _as_numeric_column(df, af_col)
    se = _as_numeric_column(df, se_col)

    af_oor = int(((af < 0.0) | (af > 1.0)).sum())
    if af_oor > 0:
        raise SumstatsDomainError(
            f"Column '{af_col}' must be within [0, 1]. Found {af_oor} out-of-range row(s)."
        )

    se_nonpositive = int((se <= 0.0).sum())
    if se_nonpositive > 0:
        raise SumstatsDomainError(
            f"Column '{se_col}' must be strictly positive. Found {se_nonpositive} non-positive row(s)."
        )


def validate_maf_grid(lowest: Any, highest: Any, num_breaks: Any) -> None:
    try:
        lowest_val = float(lowest)
        highest_val = float(highest)
    except (TypeError, ValueError) as exc:
        raise MAFGridError(
            "MAF grid bounds must be numeric: expected 0 < lowest < highest <= 0.5."
        ) from exc

    if not isinstance(num_breaks, int):
        raise MAFGridError("num_breaks must be an integer with value >= 2.")

    if lowest_val <= 0.0 or highest_val <= 0.0:
        raise MAFGridError("lowest and highest must both be > 0.")
    if highest_val > 0.5:
        raise MAFGridError("highest must be <= 0.5.")
    if lowest_val >= highest_val:
        raise MAFGridError("lowest must be strictly less than highest.")
    if num_breaks < 2:
        raise MAFGridError("num_breaks must be >= 2.")


def dataframe_fingerprint(df: pl.DataFrame, columns: list[str]) -> str:
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
