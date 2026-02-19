from io import StringIO
from pathlib import Path
import sys

import mut_var.cli as cli


def _guard_numerics(monkeypatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("Numerics should not execute for boundary validation failures.")

    monkeypatch.setattr(cli, "fit_baseline_mixture", _unexpected_call)
    monkeypatch.setattr(cli, "fit_mixture", _unexpected_call)


def _write_sumstats(path: Path, content: str) -> None:
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
    assert "Traceback" not in err


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
    assert "Traceback" not in err


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
    assert "Traceback" not in err


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
    assert "Traceback" not in err
