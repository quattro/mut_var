from __future__ import annotations

# pattern: Imperative Shell
from typing import Callable, NamedTuple

import polars as pl

from mut_var.infer import InferenceArrays
from mut_var.io import dataframe_fingerprint


class CacheKey(NamedTuple):
    fingerprint: str
    af_col: str
    beta_col: str
    se_col: str


class ArrayConversionCache:
    def __init__(self) -> None:
        self._entries: dict[CacheKey, InferenceArrays] = {}

    def build_key(self, df: pl.DataFrame, af_col: str, beta_col: str, se_col: str) -> CacheKey:
        r"""**Arguments:**
        - `df`: Input dataframe used for inference.
        - `af_col`: Effect-allele-frequency column name.
        - `beta_col`: Effect-size column name.
        - `se_col`: Standard-error column name.

        **Returns:**
        - Stable cache key for this dataframe/column selection.
        """
        fingerprint = dataframe_fingerprint(df, [af_col, beta_col, se_col])
        return CacheKey(fingerprint=fingerprint, af_col=af_col, beta_col=beta_col, se_col=se_col)

    def get(self, key: CacheKey) -> InferenceArrays | None:
        r"""**Arguments:**
        - `key`: Cache lookup key.

        **Returns:**
        - Cached arrays when present, otherwise `None`.
        """
        return self._entries.get(key)

    def set(self, key: CacheKey, arrays: InferenceArrays) -> None:
        r"""Store converted arrays under a cache key."""
        self._entries[key] = arrays

    def get_or_create(
        self,
        df: pl.DataFrame,
        af_col: str,
        beta_col: str,
        se_col: str,
        converter: Callable[[pl.DataFrame, str, str, str], InferenceArrays],
    ) -> tuple[InferenceArrays, bool]:
        r"""**Arguments:**
        - `df`: Input dataframe.
        - `af_col`: Effect-allele-frequency column name.
        - `beta_col`: Effect-size column name.
        - `se_col`: Standard-error column name.
        - `converter`: Conversion function from dataframe to inference arrays.

        **Returns:**
        - Tuple `(arrays, cache_hit)` where `cache_hit` is `True` when no conversion was needed.
        """
        key = self.build_key(df, af_col, beta_col, se_col)
        cached = self.get(key)
        if cached is not None:
            return cached, True

        arrays = converter(df, af_col, beta_col, se_col)
        self.set(key, arrays)
        return arrays, False

    def invalidate(self, key: CacheKey | None = None) -> None:
        r"""Invalidate one cache key, or all entries when `key` is `None`."""
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    @property
    def size(self) -> int:
        r"""Return the number of cached entries."""
        return len(self._entries)
