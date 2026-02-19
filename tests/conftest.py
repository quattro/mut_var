from __future__ import annotations

import jax
import polars as pl
import pytest

from tests.helpers import fixture_path

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sumstats_valid_path():
    return fixture_path("sumstats_valid.tsv")


@pytest.fixture
def sumstats_invalid_path():
    return fixture_path("sumstats_invalid.tsv")


@pytest.fixture
def sumstats_valid_df(sumstats_valid_path):
    return pl.read_csv(sumstats_valid_path, separator="\t")


@pytest.fixture
def sumstats_invalid_df(sumstats_invalid_path):
    return pl.read_csv(sumstats_invalid_path, separator="\t")
