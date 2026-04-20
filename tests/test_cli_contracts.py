import sys

from io import StringIO

import polars as pl

import mut_var.cli as cli
import mut_var.numerics.mixture_fit as mixture_fit_module

from tests.helpers import assert_no_traceback, fixture_path


def _guard_numerics(monkeypatch):
    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("Numerics should not execute for boundary validation failures.")

    monkeypatch.setattr(mixture_fit_module, "fit_baseline", _unexpected_call)


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
    assert "simulate" in help_text


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
            "--num-breaks",
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
    captured = {}
    monkeypatch.setattr(
        cli,
        "run_inference_pipeline",
        lambda path, **kwargs: captured.setdefault(
            "call",
            (path, kwargs["af_col"], kwargs["beta_col"], kwargs["se_col"]),
        )
        and pl.DataFrame(
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
    assert captured["call"] == (str(valid_path), "effect_allele_frequency", "beta", "standard_error")
    assert "mu0" in stdout.getvalue()
    err = stderr.getvalue()
    assert "infer: loading data" in err
    assert "infer: writing output" in err
    assert_no_traceback(err)


def test_cli_infer_passes_atol_rtol_to_pipeline_config(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)
    valid_path = tmp_path / "sumstats.tsv"
    valid_path.write_text(fixture_path("sumstats_valid.tsv").read_text(encoding="utf-8"), encoding="utf-8")
    captured = {}

    def _fake_pipeline(*_args, **kwargs):
        captured["config"] = kwargs["config"]
        return pl.DataFrame(
            {
                "mu0": [0.0],
                "var0": [0.1],
                "maf": [0.001],
                "name": ["pi0"],
                "value": [1.0],
            }
        )

    monkeypatch.setattr(cli, "run_inference_pipeline", _fake_pipeline)

    code = cli.run_cli(["infer", str(valid_path), "--atol", "1e-7", "--rtol", "2e-7"])

    assert code == 0
    assert captured["config"].atol == 1e-7
    assert captured["config"].rtol == 2e-7
    assert_no_traceback(stderr.getvalue())


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


def test_curve_subcommand_runs_fit_only_and_writes_fitted_samples(monkeypatch):
    stdout, stderr = _patch_streams(monkeypatch)

    code = cli.run_cli(["curve", str(fixture_path("curve_small.tsv")), "--fit-only"])

    assert code == 0
    out = stdout.getvalue()
    assert "var0" in out
    assert "param_name" in out
    assert "param_value" in out
    err = stderr.getvalue()
    assert "curve: starting curve pipeline" in err
    assert "curve: writing output" in err
    assert_no_traceback(err)


def test_curve_subcommand_passes_method_to_pipeline(monkeypatch):
    stdout, stderr = _patch_streams(monkeypatch)
    captured = {}

    def _fake_pipeline(path, **kwargs):
        captured["call"] = (path, kwargs["generate_plots"], kwargs["method"])
        return pl.DataFrame(
            {
                "var0": [0.1],
                "method": ["isotonic"],
                "param_name": ["y_0"],
                "param_value": [0.21],
            }
        )

    monkeypatch.setattr(cli, "run_curve_pipeline", _fake_pipeline)

    code = cli.run_cli(["curve", "input.tsv", "--method", "isotonic", "--fit-only"])

    assert code == 0
    assert captured["call"] == ("input.tsv", False, "isotonic")
    assert "param_value" in stdout.getvalue()
    assert_no_traceback(stderr.getvalue())
