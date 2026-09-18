from __future__ import annotations

# pattern: Imperative Shell
import subprocess
import sys

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scipy.special import expit

from mut_var.numerics.curve_fit import CurveFitResult, evaluate_curve_fit, fit_curve_model
from mut_var.pipelines import run_curve_pipeline
from mut_var.types import RESULTS


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
@pytest.mark.parametrize("noise", [0.0, 0.002])
@pytest.mark.parametrize("slope", [-0.4, 0.4])
def test_invlog_parameter_recovery(method, noise, slope):
    maf = np.geomspace(1e-6, 0.4, 200)
    coef = np.array([0.3, slope])
    value = coef[0] + coef[1] / np.log(1 / maf)
    if method == "invlog_logit":
        value = expit(value)
    value += np.random.default_rng(42).normal(0, noise, maf.size)

    solution = fit_curve_model(maf, value, method=method)

    assert solution.result == RESULTS.successful
    assert solution.value.method == method
    np.testing.assert_allclose(solution.value.payload, coef, atol=0.01 if noise else 1e-6)
    assert solution.stats["rmse"] < 0.003


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
def test_invlog_prediction_exact_zero_and_vectorized_limit(method):
    fit = CurveFitResult(method=method, payload=np.array([0.2, 0.7]))
    maf = np.array([[0, 1e-3], [1e-12, np.nextafter(0.0, 1.0)]])
    with np.errstate(all="raise"):
        prediction = evaluate_curve_fit(fit, maf)
        scalar = evaluate_curve_fit(fit, np.asarray(0.0))
    endpoint = 0.2 if method == "invlog_linear" else expit(0.2)
    assert prediction.shape == maf.shape
    assert prediction.dtype == np.float64
    assert prediction[0, 0] == endpoint
    assert scalar == endpoint
    assert 0 < prediction[1, 1] - endpoint < prediction[1, 0] - endpoint < prediction[0, 1] - endpoint


def test_invlog_linear_does_not_clip_predictions():
    fit = CurveFitResult(method="invlog_linear", payload=np.array([-0.2, 2.0]))
    prediction = evaluate_curve_fit(fit, np.array([0, 0.4]))
    assert prediction[0] == -0.2
    assert prediction[1] > 1


@pytest.mark.parametrize("values", [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1e-14, 1 - 1e-14, 1]])
def test_invlog_logit_boundary_observations(values):
    maf = np.array([1e-5, 1e-3, 0.01, 0.4])
    solution = fit_curve_model(maf, np.array(values), method="invlog_logit")
    assert solution.result == RESULTS.successful
    assert np.isfinite(solution.value.payload).all()
    prediction = evaluate_curve_fit(solution.value, np.r_[0, maf])
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert np.mean((prediction[1:] - values) ** 2) < 1e-4


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.2, np.inf, np.nan])
def test_invlog_invalid_maf(method, bad):
    solution = fit_curve_model(np.array([bad, 0.1]), np.array([0.2, 0.3]), method=method)
    assert solution.result == RESULTS.invalid_input
    assert solution.value is None
    with pytest.raises(ValueError, match="MAF"):
        evaluate_curve_fit(CurveFitResult(method=method, payload=np.array([0.2, 0.3])), np.array([bad]))


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
@pytest.mark.parametrize(
    "maf,value",
    [
        ([], []),
        ([0.1], [0.2]),
        ([0.1, 0.1], [0.2, 0.3]),
        ([0.1, 0.2], [0.2]),
        ([[0.1, 0.2]], [[0.2, 0.3]]),
        ([0.1, 0.2], [0.2, np.nan]),
    ],
)
def test_invlog_invalid_fitting_inputs(method, maf, value):
    solution = fit_curve_model(np.array(maf), np.array(value), method=method)
    assert solution.result in (RESULTS.invalid_input, RESULTS.empty_subset)
    assert solution.value is None


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
def test_invlog_pipeline_parameters_roundtrip_and_plotting(tmp_path, method):
    # Deliberately non-normalized component limits must survive reporting.
    maf = np.array([0.0, 0.001, 0.01, 0.1, 0.4])
    z = np.r_[0, 1 / np.log(1 / maf[1:])]
    values = 0.3 + 0.4 * z
    if method == "invlog_logit":
        values = expit(values)
    path = tmp_path / "curves.tsv"
    pl.DataFrame(
        {"maf": np.tile(maf, 2), "value": np.tile(values, 2), "var0": np.repeat([0.0, 0.1], len(maf))}
    ).write_csv(path, separator="\t")
    fit_only = run_curve_pipeline(str(path), method=method, generate_plots=False)
    assert not list(tmp_path.glob("*.png"))
    plotted = run_curve_pipeline(str(path), method=method, generate_plots=True)
    assert fit_only.equals(plotted)
    assert fit_only.columns == ["var0", "method", "param_name", "param_value"]
    assert set(fit_only["param_name"]) == {"a", "b"}
    for variance in [0.0, 0.1]:
        coef = fit_only.filter(pl.col("var0") == variance).sort("param_name")["param_value"].to_numpy()
        np.testing.assert_allclose(coef, [0.3, 0.4], atol=1e-6)
        fit = CurveFitResult(method=method, payload=coef)
        np.testing.assert_allclose(evaluate_curve_fit(fit, maf), values, atol=1e-7)
    pngs = list(tmp_path.glob(f"*_{method}_var_*.png"))
    assert len(pngs) == 2
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in pngs)


@pytest.mark.parametrize("method", ["invlog_linear", "invlog_logit"])
def test_invlog_cli_method_selection(tmp_path, method):
    path = tmp_path / "input.tsv"
    output = tmp_path / "parameters.tsv"
    pl.DataFrame({"maf": [0.001, 0.01, 0.1], "value": [0.2, 0.3, 0.4], "var0": [0.1] * 3}).write_csv(
        path, separator="\t"
    )
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("mutvar")),
            "curve",
            str(path),
            "--method",
            method,
            "--fit-only",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = pl.read_csv(output, separator="\t")
    assert result["method"].to_list() == [method, method]
    assert result["param_name"].to_list() == ["a", "b"]
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_invlog_sigmoid_parameter_recovery(direction, noise):
    maf = np.r_[0.0, np.geomspace(1e-8, 0.4, 150)]
    z = np.r_[0.0, 1 / -np.log(maf[1:])]
    coef = np.array([0.12, 0.65, -10 * direction, 60 * direction])
    value = coef[0] + (coef[1] - coef[0]) * expit(coef[2] + coef[3] * z)
    value += np.random.default_rng(15).normal(0, noise, maf.size)
    solution = fit_curve_model(maf, value, method="invlog_sigmoid")
    assert solution.result == RESULTS.successful
    assert solution.value.method == "invlog_sigmoid"
    np.testing.assert_allclose(solution.value.payload, coef, rtol=0.01 if noise else 1e-5, atol=1e-6)


def test_invlog_sigmoid_exact_zero_and_tail():
    fit = CurveFitResult(method="invlog_sigmoid", payload=np.array([0.1, 0.8, -2.0, 3.0]))
    maf = np.array([[0.0, 1e-3], [1e-12, np.nextafter(0.0, 1.0)]])
    with np.errstate(all="raise"):
        prediction = evaluate_curve_fit(fit, maf)
    endpoint = 0.1 + (0.8 - 0.1) * expit(-2.0)
    assert prediction.shape == maf.shape
    assert prediction[0, 0] == endpoint
    assert evaluate_curve_fit(fit, np.asarray(0.0)) == endpoint
    assert 0 < prediction[1, 1] - endpoint < prediction[1, 0] - endpoint < prediction[0, 1] - endpoint
    assert ((prediction >= 0.1) & (prediction <= 0.8)).all()


def test_invlog_sigmoid_recovers_small_nonzero_plateaus():
    maf = np.geomspace(1e-6, 1e-2, 60)
    values = 2.5e-6 + 2e-6 * expit(40 - 500 / -np.log(maf))
    solution = fit_curve_model(maf, values, method="invlog_sigmoid")
    assert solution.result == RESULTS.successful
    np.testing.assert_allclose(evaluate_curve_fit(solution.value, maf), values, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(solution.value.payload, [2.5e-6, 4.5e-6, 40, -500], rtol=1e-4)


@pytest.mark.parametrize("value", [0.0, 0.3, 1.0])
def test_invlog_sigmoid_constant_observations(value):
    maf = np.geomspace(1e-6, 0.1, 10)
    solution = fit_curve_model(maf, np.full(10, value), method="invlog_sigmoid")
    assert solution.result == RESULTS.successful
    np.testing.assert_allclose(evaluate_curve_fit(solution.value, np.r_[0, maf]), value, atol=1e-10)


def test_invlog_sigmoid_exact_boundary_observations():
    maf = np.geomspace(1e-6, 0.4, 20)
    values = np.r_[np.zeros(5), np.linspace(0.01, 0.99, 10), np.ones(5)]
    solution = fit_curve_model(maf, values, method="invlog_sigmoid")
    assert solution.result == RESULTS.successful
    assert np.isfinite(solution.value.payload).all()
    predictions = evaluate_curve_fit(solution.value, np.r_[0.0, maf])
    assert ((predictions >= 0) & (predictions <= 1)).all()
    assert np.sqrt(np.mean((predictions[1:] - values) ** 2)) < 0.06


@pytest.mark.parametrize(
    "maf,values",
    [
        ([0.01, 0.02, 0.03], [0.1, 0.2, 0.3]),
        ([0.01, 0.02, 0.03, 1.0], [0.1] * 4),
        ([0.01, 0.02, 0.03, 0.04], [0.1, 0.2, 0.3, 1.1]),
    ],
)
def test_invlog_sigmoid_invalid_input(maf, values):
    solution = fit_curve_model(np.array(maf), np.array(values), method="invlog_sigmoid")
    assert solution.result == RESULTS.invalid_input


def test_invlog_sigmoid_pipeline_and_cli(tmp_path):
    maf = np.geomspace(1e-6, 0.1, 20)
    values = 0.2 + 0.4 * expit(-10 + 60 / -np.log(maf))
    path = tmp_path / "input.tsv"
    pl.DataFrame({"maf": maf, "value": values, "var0": np.ones(20)}).write_csv(path, separator="\t")
    fit_only = run_curve_pipeline(str(path), method="invlog_sigmoid", generate_plots=False)
    assert not list(tmp_path.glob("*.png"))
    plotted = run_curve_pipeline(str(path), method="invlog_sigmoid", generate_plots=True)
    assert fit_only.equals(plotted)
    assert set(fit_only["param_name"]) == {"lower", "upper", "a", "b"}
    mapping = dict(zip(fit_only["param_name"], fit_only["param_value"]))
    fit = CurveFitResult(method="invlog_sigmoid", payload=np.array([mapping[k] for k in ["lower", "upper", "a", "b"]]))
    np.testing.assert_allclose(evaluate_curve_fit(fit, maf), values, atol=1e-7)
    assert len(list(tmp_path.glob("*.png"))) == 1
    output = tmp_path / "params.tsv"
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("mutvar")),
            "curve",
            str(path),
            "--method",
            "invlog_sigmoid",
            "--fit-only",
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert pl.read_csv(output, separator="\t").equals(fit_only)


def test_invlog_logit_recovers_rare_component_curve():
    maf = np.geomspace(1e-6, 0.01, 20)
    value = expit(-20 + 30 / -np.log(maf))
    solution = fit_curve_model(maf, value, method="invlog_logit")
    assert solution.result == RESULTS.successful
    np.testing.assert_allclose(solution.value.payload, [-20, 30], rtol=1e-8)


def test_invlog_logit_recovers_constant_rare_component():
    maf = np.geomspace(1e-6, 0.01, 20)
    values = np.full(maf.size, 1e-12)
    solution = fit_curve_model(maf, values, method="invlog_logit")
    assert solution.result == RESULTS.successful
    np.testing.assert_allclose(evaluate_curve_fit(solution.value, maf), values, rtol=1e-6, atol=0)
