import polars as pl
import pytest

from mut_var.io import (
    MAFGridError,
    MissingColumnsError,
    NumericColumnsError,
    SumstatsDomainError,
    read_sumstats,
    validate_maf_grid,
    validate_numeric_columns,
    validate_required_columns,
    validate_sumstats_domain,
)


def test_read_sumstats_reads_tab_delimited_file(tmp_path):
    path = tmp_path / "sumstats.tsv"
    path.write_text(
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "0.20\t0.1\t0.03\n"
        "0.40\t-0.2\t0.02\n",
        encoding="utf-8",
    )

    df = read_sumstats(str(path))
    assert df.shape == (2, 3)


def test_validate_required_columns_rejects_missing_columns():
    df = pl.DataFrame({"effect_allele_frequency": [0.1], "beta": [0.0]})

    with pytest.raises(MissingColumnsError, match="Missing required column"):
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

    with pytest.raises(NumericColumnsError, match="must contain numeric values"):
        validate_numeric_columns(df, "effect_allele_frequency", "beta", "standard_error")


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
    with pytest.raises(SumstatsDomainError, match=error):
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
    with pytest.raises(MAFGridError):
        validate_maf_grid(lowest, highest, num_breaks)


def test_validate_maf_grid_accepts_valid_config():
    validate_maf_grid(1e-4, 0.1, 12)
