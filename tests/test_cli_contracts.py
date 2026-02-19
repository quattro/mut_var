import sys

from io import StringIO

import polars as pl

import mut_var.cli as cli
import mut_var.numerics.baseline as baseline_module

from tests.helpers import assert_no_traceback, fixture_path


def _guard_numerics(monkeypatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("Numerics should not execute for boundary validation failures.")

    monkeypatch.setattr(baseline_module, "fit_baseline", _unexpected_call)


def _write_sumstats(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _patch_streams(monkeypatch):
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    return stdout, stderr


def test_run_cli_is_canonical_entrypoint(monkeypatch):
    stdout, _ = _patch_streams(monkeypatch)
    code = cli.run_cli(["--help"])

    assert code == 0
    help_text = stdout.getvalue()
    assert "usage:" in help_text
    assert "infer" in help_text
    assert "curve" in help_text


def test_missing_required_columns_returns_nonzero(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    _guard_numerics(monkeypatch)
    path = tmp_path / "sumstats.tsv"
    _write_sumstats(
        path,
        "effect_allele_frequency\tbeta\n0.2\t0.1\n",
    )

    code = cli.run_cli(["infer", str(path)])

    assert code == 2
    err = stderr.getvalue()
    assert "Missing required column" in err
    assert_no_traceback(err)


def test_out_of_range_af_returns_nonzero(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    _guard_numerics(monkeypatch)
    path = tmp_path / "sumstats.tsv"
    _write_sumstats(
        path,
        "effect_allele_frequency\tbeta\tstandard_error\n1.2\t0.1\t0.01\n",
    )

    code = cli.run_cli(["infer", str(path)])

    assert code == 2
    err = stderr.getvalue()
    assert "within [0, 1]" in err
    assert_no_traceback(err)


def test_non_positive_se_returns_nonzero(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    _guard_numerics(monkeypatch)
    path = tmp_path / "sumstats.tsv"
    _write_sumstats(
        path,
        "effect_allele_frequency\tbeta\tstandard_error\n0.2\t0.1\t0.0\n",
    )

    code = cli.run_cli(["infer", str(path)])

    assert code == 2
    err = stderr.getvalue()
    assert "strictly positive" in err
    assert_no_traceback(err)


def test_invalid_maf_grid_returns_nonzero(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    _guard_numerics(monkeypatch)
    path = tmp_path / "sumstats.tsv"
    _write_sumstats(
        path,
        "effect_allele_frequency\tbeta\tstandard_error\n0.2\t0.1\t0.01\n",
    )

    code = cli.run_cli(
        [
            "infer",
            str(path),
            "--lowest",
            "0.1",
            "--highest",
            "0.05",
            "--num_breaks",
            "10",
        ]
    )

    assert code == 2
    err = stderr.getvalue()
    assert "lowest must be strictly less than highest" in err
    assert_no_traceback(err)


def test_cli_infer_success_writes_dataframe(monkeypatch, tmp_path):
    stdout, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_inference_pipeline",
        lambda *_args, **_kwargs: pl.DataFrame(
            {
                "mu0": [0.0],
                "var0": [0.1],
                "maf": [0.001],
                "name": ["pi0"],
                "value": [1.0],
            }
        ),
    )

    code = cli.run_cli(["infer", str(valid_path)])

    assert code == 0
    assert "mu0" in stdout.getvalue()
    err = stderr.getvalue()
    assert "infer: loading data" in err
    assert "infer: writing output" in err
    assert_no_traceback(err)


def test_cli_maps_inference_pipeline_error_status_to_exit(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")

    def _raise_error(*_args, **_kwargs):
        raise RuntimeError("objective became non-finite")

    monkeypatch.setattr(cli, "run_inference_pipeline", _raise_error)

    code = cli.run_cli(["infer", str(valid_path)])

    assert code == 1
    err = stderr.getvalue()
    assert "non-finite" in err
    assert_no_traceback(err)


def test_cli_requires_explicit_subcommand(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")

    code = cli.run_cli([str(valid_path)])

    assert code == 2
    err = stderr.getvalue()
    assert "invalid choice" in err
    assert_no_traceback(err)


def test_curve_subcommand_runs_fit_only_and_writes_coefficients(monkeypatch):
    stdout, stderr = _patch_streams(monkeypatch)

    code = cli.run_cli(["curve", str(fixture_path("curve_small.tsv")), "--fit-only"])

    assert code == 0
    out = stdout.getvalue()
    assert "var0" in out
    assert "coef_left" in out
    err = stderr.getvalue()
    assert "curve: starting curve pipeline" in err
    assert "curve: writing output" in err
    assert_no_traceback(err)
