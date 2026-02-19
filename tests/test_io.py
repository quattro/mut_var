import polars as pl
import pytest

from mut_var.io import (
    read_sumstats,
    validate_maf_grid,
    validate_numeric_columns,
    validate_required_columns,
    validate_sumstats_domain,
)
from tests.helpers import assert_no_traceback


def test_read_sumstats_reads_tab_delimited_file(sumstats_valid_path):
    df = read_sumstats(str(sumstats_valid_path))
    assert df.shape == (4, 3)


def test_validate_required_columns_rejects_missing_columns():
    df = pl.DataFrame({"effect_allele_frequency": [0.1], "beta": [0.0]})

    with pytest.raises(ValueError, match="Missing required column"):
        validate_required_columns(df, "effect_allele_frequency", "beta", "standard_error")


def test_validate_numeric_columns_rejects_non_numeric_values():
    df = pl.DataFrame(
        {
            "effect_allele_frequency": [0.1, 0.2],
            "beta": [0.1, "not-a-number"],
            "standard_error": [0.01, 0.02],
        },
        strict=False,
    )

    with pytest.raises(ValueError, match="must contain numeric values"):
        validate_numeric_columns(df, "effect_allele_frequency", "beta", "standard_error")


def test_validate_numeric_columns_rejects_invalid_fixture(sumstats_invalid_df):
    with pytest.raises(ValueError):
        validate_numeric_columns(sumstats_invalid_df, "effect_allele_frequency", "beta", "standard_error")


def test_assert_no_traceback_helper():
    assert_no_traceback("validation failed: missing column")


@pytest.mark.parametrize(
    ("df", "error"),
    [
        (
            pl.DataFrame(
                {
                    "effect_allele_frequency": [0.2, 1.2],
                    "standard_error": [0.03, 0.02],
                }
            ),
            "within \\[0, 1\\]",
        ),
        (
            pl.DataFrame(
                {
                    "effect_allele_frequency": [0.2, 0.3],
                    "standard_error": [0.03, 0.0],
                }
            ),
            "strictly positive",
        ),
    ],
)
def test_validate_sumstats_domain_rejects_invalid_af_and_se(df, error):
    with pytest.raises(ValueError, match=error):
        validate_sumstats_domain(df, "effect_allele_frequency", "standard_error")


@pytest.mark.parametrize(
    ("lowest", "highest", "num_breaks"),
    [
        (0.0, 0.1, 10),
        (0.1, 0.05, 10),
        (0.1, 0.6, 10),
        (0.01, 0.1, 1),
    ],
)
def test_validate_maf_grid_rejects_invalid_configs(lowest, highest, num_breaks):
    with pytest.raises(ValueError):
        validate_maf_grid(lowest, highest, num_breaks)


def test_validate_maf_grid_accepts_valid_config():
    validate_maf_grid(1e-4, 0.1, 12)
