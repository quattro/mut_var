from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from mut_var.io import dataframe_fingerprint
from mut_var.numerics.pipeline import InferenceArrays


@dataclass(frozen=True, slots=True)
class CacheKey:
    fingerprint: str
    af_col: str
    beta_col: str
    se_col: str


class ArrayConversionCache:
    def __init__(self) -> None:
        self._entries: dict[CacheKey, InferenceArrays] = {}

    def build_key(self, df: pl.DataFrame, af_col: str, beta_col: str, se_col: str) -> CacheKey:
        fingerprint = dataframe_fingerprint(df, [af_col, beta_col, se_col])
        return CacheKey(fingerprint=fingerprint, af_col=af_col, beta_col=beta_col, se_col=se_col)

    def get(self, key: CacheKey) -> InferenceArrays | None:
        return self._entries.get(key)

    def set(self, key: CacheKey, arrays: InferenceArrays) -> None:
        self._entries[key] = arrays

    def get_or_create(
        self,
        df: pl.DataFrame,
        af_col: str,
        beta_col: str,
        se_col: str,
        converter: Callable[[pl.DataFrame, str, str, str], InferenceArrays],
    ) -> tuple[InferenceArrays, bool]:
        key = self.build_key(df, af_col, beta_col, se_col)
        cached = self.get(key)
        if cached is not None:
            return cached, True

        arrays = converter(df, af_col, beta_col, se_col)
        self.set(key, arrays)
        return arrays, False

    def invalidate(self, key: CacheKey | None = None) -> None:
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    @property
    def size(self) -> int:
        return len(self._entries)
