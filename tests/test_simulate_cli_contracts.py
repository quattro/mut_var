import sys

from io import StringIO

import polars as pl

import mut_var.cli as cli

from mut_var.pipelines.simulation import SimulationArtifacts
from tests.helpers import assert_no_traceback


def _patch_streams(monkeypatch):
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    return stdout, stderr


def _fake_artifacts() -> SimulationArtifacts:
    return SimulationArtifacts(
        truth=pl.DataFrame(
            {
                "row_id": [0, 1],
                "component": [0, 1],
                "beta_true": [0.1, -0.2],
                "sigma2": [0.01, 0.02],
                "effect_allele_frequency": [0.1, 0.2],
            }
        ),
        observed=pl.DataFrame(
            {
                "row_id": [0, 1],
                "effect_allele_frequency": [0.1, 0.2],
                "beta": [0.11, -0.19],
                "standard_error": [0.01, 0.01],
            }
        ),
        metadata=pl.DataFrame(
            {
                "seed": [0],
                "n_rows": [2],
                "num_components": [2],
                "variance_link": ["maf_power"],
                "theta": [0.5],
                "af_decile": [0],
                "empirical_var_beta_true": [0.01],
                "empirical_mean_sigma2": [0.015],
            }
        ),
    )


def test_run_cli_help_includes_simulate_subcommand(monkeypatch):
    stdout, _ = _patch_streams(monkeypatch)

    code = cli.run_cli(["--help"])

    assert code == 0
    assert "simulate" in stdout.getvalue()


def test_simulate_invalid_weight_vector_returns_exit_2(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)

    code = cli.run_cli(
        [
            "simulate",
            "--output-prefix",
            "sim",
            "--output-dir",
            str(tmp_path),
            "--weights",
            "1.0",
            "--log-var-scales",
            "-8.0",
        ]
    )

    assert code == 2
    err = stderr.getvalue()
    assert "weights" in err
    assert_no_traceback(err)


def test_simulate_mismatched_weights_and_scales_returns_exit_2(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)

    code = cli.run_cli(
        [
            "simulate",
            "--output-prefix",
            "sim",
            "--output-dir",
            str(tmp_path),
            "--weights",
            "0.9,0.1",
            "--log-var-scales",
            "-8.0",
        ]
    )

    assert code == 2
    err = stderr.getvalue()
    assert "same length" in err
    assert_no_traceback(err)


def test_simulate_runtime_failure_maps_to_exit_1(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)

    def _raise_runtime(*_args, **_kwargs):
        raise RuntimeError("simulation became non-finite")

    monkeypatch.setattr(cli, "run_simulation_pipeline", _raise_runtime, raising=False)

    code = cli.run_cli(
        [
            "simulate",
            "--output-prefix",
            "sim",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert code == 1
    err = stderr.getvalue()
    assert "non-finite" in err
    assert_no_traceback(err)


def test_simulate_success_writes_truth_observed_meta_files(monkeypatch, tmp_path):
    stdout, stderr = _patch_streams(monkeypatch)

    monkeypatch.setattr(cli, "run_simulation_pipeline", lambda **_kwargs: _fake_artifacts(), raising=False)

    code = cli.run_cli(
        [
            "simulate",
            "--output-prefix",
            "sim",
            "--output-dir",
            str(tmp_path),
            "--n-rows",
            "2",
        ]
    )

    truth_path = tmp_path / "sim.truth.tsv"
    observed_path = tmp_path / "sim.observed.tsv"
    meta_path = tmp_path / "sim.meta.tsv"

    assert code == 0
    assert stdout.getvalue() == ""
    assert truth_path.exists()
    assert observed_path.exists()
    assert meta_path.exists()
    assert "row_id" in truth_path.read_text(encoding="utf-8")
    assert "standard_error" in observed_path.read_text(encoding="utf-8")
    assert "af_decile" in meta_path.read_text(encoding="utf-8")
    assert_no_traceback(stderr.getvalue())


def test_simulate_logs_stage_markers_without_traceback(monkeypatch, tmp_path):
    _, stderr = _patch_streams(monkeypatch)

    monkeypatch.setattr(cli, "run_simulation_pipeline", lambda **_kwargs: _fake_artifacts(), raising=False)

    code = cli.run_cli(
        [
            "simulate",
            "--output-prefix",
            "sim",
            "--output-dir",
            str(tmp_path),
            "--n-rows",
            "2",
        ]
    )

    assert code == 0
    err = stderr.getvalue()
    assert "simulate: validating args" in err
    assert "simulate: running pipeline" in err
    assert "simulate: writing outputs" in err
    assert_no_traceback(err)
