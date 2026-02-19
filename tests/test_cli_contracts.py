import json
import sys

from io import StringIO
from pathlib import Path

import mut_var.cli as cli

from mut_var.contracts import RESULTS, Solution
from scripts.check_release_gate import evaluate_release_gate
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
        "effect_allele_frequency\tbeta\n"
        "0.2\t0.1\n",
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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "1.2\t0.1\t0.01\n",
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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "0.2\t0.1\t0.0\n",
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
        "effect_allele_frequency\tbeta\tstandard_error\n"
        "0.2\t0.1\t0.01\n",
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


def test_cli_maps_max_steps_status_to_zero_exit(monkeypatch, tmp_path):
    stdout, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_inference_pipeline",
        lambda **_kwargs: Solution(value=None, result=RESULTS.max_steps_reached, stats={}, state=None),
    )

    code = cli.run_cli(["infer", str(valid_path)])

    assert code == 0
    assert stdout.getvalue() == ""
    assert_no_traceback(stderr.getvalue())


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
    assert_no_traceback(stderr.getvalue())


def test_release_gate_fails_when_report_is_missing(tmp_path):
    report_path = tmp_path / "missing.json"
    passed, errors, _ = evaluate_release_gate(report_path)

    assert not passed
    assert any("not found" in err for err in errors)


def test_release_gate_passes_with_valid_report(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "comparison": {
                    "improvement_percent": 25.0,
                    "threshold_percent": 20.0,
                    "passed": True,
                }
            }
        ),
        encoding="utf-8",
    )

    passed, errors, payload = evaluate_release_gate(Path(report_path))

    assert passed
    assert errors == []
    assert payload is not None
