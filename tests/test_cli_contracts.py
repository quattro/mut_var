import sys

from io import StringIO

import mut_var.cli as cli

from mut_var.contracts import RESULTS, Solution
from tests.helpers import assert_no_traceback, fixture_path


def _guard_numerics(monkeypatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("Numerics should not execute for boundary validation failures.")

    monkeypatch.setattr(cli, "run_inference_pipeline", _unexpected_call)


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
    assert "usage:" in stdout.getvalue()


def test_missing_required_columns_returns_nonzero(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    _guard_numerics(monkeypatch)
    path = tmp_path / "sumstats.tsv"
    _write_sumstats(
        path,
        "effect_allele_frequency\tbeta\n"
        "0.2\t0.1\n",
    )

    code = cli.run_cli([str(path)])

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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "1.2\t0.1\t0.01\n",
    )

    code = cli.run_cli([str(path)])

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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "0.2\t0.1\t0.0\n",
    )

    code = cli.run_cli([str(path)])

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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "0.2\t0.1\t0.01\n",
    )

    code = cli.run_cli(
        [
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


def test_cli_maps_max_steps_status_to_zero_exit(monkeypatch, tmp_path):
    stdout, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_inference_pipeline",
        lambda **_kwargs: Solution(value=None, result=RESULTS.max_steps_reached, stats={}, state=None),
    )

    code = cli.run_cli([str(valid_path)])

    assert code == 0
    assert stdout.getvalue() == ""
    assert_no_traceback(stderr.getvalue())
