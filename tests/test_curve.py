from __future__ import annotations

import logging
import shutil
import sys

from pathlib import Path

import numpy as np
import polars as pl

from scipy.special import expit

from mut_var.numerics.curve_fit import CurveFitResult, evaluate_curve_fit, fit_curve_model
from mut_var.pipelines import run_curve_pipeline
from mut_var.types import RESULTS, Solution

FIXTURE = Path(__file__).parent / "fixtures" / "curve_small.tsv"


def _copy_fixture(tmp_path: Path) -> Path:
    data_path = tmp_path / "curve_small.tsv"
    shutil.copy(FIXTURE, data_path)
    return data_path


def _fit_matrix(fit_df: pl.DataFrame) -> np.ndarray:
    ordered = fit_df.sort(["var0", "method", "param_name"])
    selected = ordered.select(["var0", "param_value"]).to_numpy()
    return np.asarray(selected, dtype=float)


def _expected_midpoint(raw_mid: float, maf: np.ndarray) -> float:
    maf_np = np.asarray(maf, dtype=float)
    positive = maf_np[maf_np > 0.0]
    if positive.size == 0:
        log_min = np.log(1e-12)
        log_span = 1.0
    else:
        log_min = float(np.log(np.min(positive) + 1e-12))
        log_span = max(float(np.log(np.max(positive) + 1e-12) - log_min), 1e-6)
    return float(np.exp(log_min + float(expit(np.asarray(raw_mid))) * log_span))


def _fit_sigmoid(maf: np.ndarray, value: np.ndarray) -> Solution:
    return fit_curve_model(maf, value, method="sigmoid")


def _sigmoid_result(coef: np.ndarray) -> CurveFitResult:
    return CurveFitResult(method="sigmoid", payload=np.asarray(coef, dtype=float))


def test_fit_only_numerics_has_no_plotting_dependency():
    sys.modules.pop("matplotlib.pyplot", None)

    maf = np.asarray([0.001, 0.005, 0.01])
    value = np.asarray([0.2, 0.21, 0.22])
    solution = _fit_sigmoid(maf, value)

    assert solution.result == RESULTS.successful
    assert "matplotlib.pyplot" not in sys.modules


def test_fit_only_pipeline_is_deterministic_and_side_effect_free(tmp_path):
    data_path = _copy_fixture(tmp_path)
    sys.modules.pop("mut_var.plotting.curve_plots", None)

    first = run_curve_pipeline(str(data_path), generate_plots=False)
    second = run_curve_pipeline(str(data_path), generate_plots=False)

    assert first.columns == ["var0", "method", "param_name", "param_value"]
    assert second.columns == ["var0", "method", "param_name", "param_value"]
    assert first.height > 0
    assert second.height > 0
    assert set(first["method"].to_list()) == {"sigmoid"}
    assert set(second["method"].to_list()) == {"sigmoid"}
    assert bool(np.allclose(_fit_matrix(first), _fit_matrix(second), rtol=1e-6, atol=1e-6))

    assert "mut_var.plotting.curve_plots" not in sys.modules
    assert list(tmp_path.glob("*.png")) == []


def test_isotonic_method_pipeline_is_deterministic_and_side_effect_free(tmp_path):
    data_path = _copy_fixture(tmp_path)

    first = run_curve_pipeline(str(data_path), generate_plots=False, method="isotonic")
    second = run_curve_pipeline(str(data_path), generate_plots=False, method="isotonic")

    assert first.columns == ["var0", "method", "param_name", "param_value"]
    assert second.columns == ["var0", "method", "param_name", "param_value"]
    assert first.height > 0
    assert second.height > 0
    assert set(first["method"].to_list()) == {"isotonic"}
    assert set(second["method"].to_list()) == {"isotonic"}
    assert bool(np.allclose(_fit_matrix(first), _fit_matrix(second), rtol=1e-6, atol=1e-6))


def test_plotting_mode_writes_png_side_effects(tmp_path):
    data_path = _copy_fixture(tmp_path)

    coef_df = run_curve_pipeline(str(data_path), generate_plots=True)

    assert coef_df.height > 0
    assert set(coef_df["method"].to_list()) == {"sigmoid"}
    png_paths = list(tmp_path.glob("*.png"))
    assert len(png_paths) > 0
    for path in png_paths:
        assert path.exists()
        assert path.suffix == ".png"


def test_plotting_does_not_change_fit_outputs(tmp_path):
    data_path = _copy_fixture(tmp_path)

    fit_only = run_curve_pipeline(str(data_path), generate_plots=False)
    with_plots = run_curve_pipeline(str(data_path), generate_plots=True)

    assert fit_only.height > 0
    assert with_plots.height > 0
    assert bool(np.allclose(_fit_matrix(fit_only), _fit_matrix(with_plots), rtol=1e-6, atol=1e-6))


def test_fit_curve_returns_clean_success_stats(monkeypatch):
    class _FakeSciPySolution:
        x = np.asarray([0.25, 0.75, 1.0, 0.0], dtype=float)
        status = 1
        nfev = 13

    def _fake(*_args, **_kwargs):
        return _FakeSciPySolution()

    monkeypatch.setattr("mut_var.numerics.curve_fit.sco.least_squares", _fake)

    solution = _fit_sigmoid(np.asarray([0.001, 0.005, 0.01]), np.asarray([0.2, 0.21, 0.22]))

    assert solution.result == RESULTS.successful
    assert solution.value is not None
    expected_left = float(expit(0.25))
    expected_right = expected_left + (1.0 - expected_left) * float(expit(0.75))
    expected_midpoint = _expected_midpoint(0.0, np.asarray([0.001, 0.005, 0.01]))
    assert solution.value is not None
    assert bool(
        np.allclose(
            solution.value.payload,
            np.asarray([expected_left, expected_right, 1.0, expected_midpoint]),
            rtol=1e-8,
            atol=1e-8,
        )
    )
    assert solution.stats is not None
    assert solution.stats["n_obs"] == 3
    assert solution.stats["epoch_count"] == 13
    assert solution.stats["converged"] is True
    assert "rmse" in solution.stats
    assert "max_abs_error" in solution.stats
    assert "data_sign_changes" in solution.stats
    assert "poor_fit" in solution.stats
    assert solution.state is None


def test_fit_curve_maps_max_steps_reached(monkeypatch):
    class _FakeSciPySolution:
        x = np.asarray([0.1, 0.2, 0.3, 0.0], dtype=float)
        status = 0
        nfev = 1000

    def _fake(*_args, **_kwargs):
        return _FakeSciPySolution()

    monkeypatch.setattr("mut_var.numerics.curve_fit.sco.least_squares", _fake)

    solution = _fit_sigmoid(np.asarray([0.001, 0.005, 0.01]), np.asarray([0.2, 0.21, 0.22]))

    assert solution.result == RESULTS.max_steps_reached
    assert solution.value is not None
    expected_left = float(expit(0.1))
    expected_right = expected_left + (1.0 - expected_left) * float(expit(0.2))
    expected_midpoint = _expected_midpoint(0.0, np.asarray([0.001, 0.005, 0.01]))
    assert solution.value is not None
    assert bool(
        np.allclose(
            solution.value.payload,
            np.asarray([expected_left, expected_right, 0.3, expected_midpoint]),
            rtol=1e-8,
            atol=1e-8,
        )
    )
    assert solution.stats is not None
    assert solution.stats["n_obs"] == 3
    assert solution.stats["epoch_count"] == 1000
    assert solution.stats["converged"] is False
    assert "rmse" in solution.stats
    assert "max_abs_error" in solution.stats
    assert "data_sign_changes" in solution.stats
    assert "poor_fit" in solution.stats


def test_fit_curve_bounded_coefficients_and_predictions_for_decreasing_step_like_series():
    maf = np.asarray(
        [
            0.0,
            1.0e-5,
            2.1544346900318854e-5,
            4.641588833612779e-5,
            1.0e-4,
            2.154434690031884e-4,
            4.641588833612784e-4,
            1.0e-3,
            2.1544346900318825e-3,
            4.641588833612781e-3,
            1.0e-2,
        ]
    )
    value = np.asarray(
        [
            0.9113987225089466,
            0.8793662020134873,
            0.8798967014997061,
            0.882359372010592,
            0.885495440889938,
            0.8893293270273518,
            0.890440871322699,
            0.8910280241148753,
            0.8899870681412326,
            0.2799740535707099,
            5.783520058402039e-7,
        ]
    )

    solution = _fit_sigmoid(maf, value)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    left, right, _ = [float(x) for x in solution.value.payload[:3]]
    assert 0.0 <= left <= right <= 1.0
    midpoint = float(solution.value.payload[3])
    positive_maf = maf[maf > 0.0]
    assert float(positive_maf.min()) <= midpoint <= float(positive_maf.max())

    fitted = evaluate_curve_fit(solution.value, maf)
    assert bool(np.isfinite(fitted).all())
    assert bool((fitted >= 0.0).all())
    assert bool((fitted <= 1.0).all())


def test_fit_curve_bounded_coefficients_and_predictions_for_increasing_step_like_series():
    maf = np.asarray(
        [
            0.0,
            1.0e-5,
            2.1544346900318854e-5,
            4.641588833612779e-5,
            1.0e-4,
            2.154434690031884e-4,
            4.641588833612784e-4,
            1.0e-3,
            2.1544346900318825e-3,
            4.641588833612781e-3,
            1.0e-2,
        ]
    )
    value = np.asarray(
        [
            0.003559415880098624,
            0.003229568717924408,
            0.003242203573372101,
            0.003293493342508824,
            0.003368829642686811,
            0.0035139123398018938,
            0.003713713605468037,
            0.0038184530049926793,
            0.004134948486398562,
            0.622544470522179,
            0.8853649117801035,
        ]
    )

    solution = _fit_sigmoid(maf, value)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    left, right, _ = [float(x) for x in solution.value.payload[:3]]
    assert 0.0 <= left <= right <= 1.0
    midpoint = float(solution.value.payload[3])
    positive_maf = maf[maf > 0.0]
    assert float(positive_maf.min()) <= midpoint <= float(positive_maf.max())

    fitted = evaluate_curve_fit(solution.value, maf)
    assert bool(np.isfinite(fitted).all())
    assert bool((fitted >= 0.0).all())
    assert bool((fitted <= 1.0).all())


def test_curve_pipeline_accepts_max_steps_reached(monkeypatch, tmp_path):
    data_path = _copy_fixture(tmp_path)

    def _fake_fit_curve_model(maf, value, *, method):  # noqa: ARG001
        del maf, value
        return Solution(
            value=CurveFitResult(method="sigmoid", payload=np.asarray([0.2, 0.8, -2.0, 0.003])),
            result=RESULTS.max_steps_reached,
            stats={"n_obs": 4, "epoch_count": 1000, "converged": False},
            state=None,
        )

    monkeypatch.setattr("mut_var.pipelines.curve.fit_curve_model", _fake_fit_curve_model)

    coef_df = run_curve_pipeline(str(data_path), generate_plots=False)

    assert coef_df.columns == ["var0", "method", "param_name", "param_value"]
    assert coef_df.height == 8
    assert set(coef_df["method"].to_list()) == {"sigmoid"}


def test_curve_pipeline_logs_warning_for_poor_fit(monkeypatch, tmp_path, caplog):
    data_path = _copy_fixture(tmp_path)

    def _fake_fit_curve_model(maf, value, *, method):  # noqa: ARG001
        del maf, value
        return Solution(
            value=CurveFitResult(method="sigmoid", payload=np.asarray([0.2, 0.8, -2.0, 0.003])),
            result=RESULTS.successful,
            stats={
                "n_obs": 4,
                "epoch_count": 1000,
                "converged": True,
                "rmse": 0.2,
                "max_abs_error": 0.4,
                "data_sign_changes": 3,
                "poor_fit": True,
            },
            state=None,
        )

    monkeypatch.setattr("mut_var.pipelines.curve.fit_curve_model", _fake_fit_curve_model)

    with caplog.at_level(logging.WARNING):
        run_curve_pipeline(str(data_path), generate_plots=False)

    assert any("poor-fit diagnostics" in record.message for record in caplog.records)


def test_fit_curve_rate_initialization_follows_endpoint_trend(monkeypatch):
    class _FakeSciPySolution:
        def __init__(self, x0: np.ndarray):
            self.x = x0
            self.status = 0
            self.nfev = 1

    def _fake_least_squares(*args, **_kwargs):
        x0 = args[1]  # second positional arg is the initial params
        return _FakeSciPySolution(x0)

    monkeypatch.setattr("mut_var.numerics.curve_fit.sco.least_squares", _fake_least_squares)

    maf = np.asarray([0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    decreasing = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    increasing = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])

    dec_sol = _fit_sigmoid(maf, decreasing)
    inc_sol = _fit_sigmoid(maf, increasing)

    assert dec_sol.value is not None
    assert inc_sol.value is not None
    assert float(dec_sol.value.payload[2]) > 0.0
    assert float(inc_sol.value.payload[2]) < 0.0


def test_fit_curve_sets_poor_fit_for_step_like_nonmonotone_series():
    maf = np.asarray(
        [
            0.0,
            1.0e-5,
            2.1544346900318854e-5,
            4.641588833612779e-5,
            1.0e-4,
            2.154434690031884e-4,
            4.641588833612784e-4,
            1.0e-3,
            2.1544346900318825e-3,
            4.641588833612781e-3,
            1.0e-2,
        ]
    )
    value = np.asarray(
        [
            0.9113987225089466,
            0.8793662020134873,
            0.8798967014997061,
            0.882359372010592,
            0.885495440889938,
            0.8893293270273518,
            0.890440871322699,
            0.8910280241148753,
            0.8899870681412326,
            0.2799740535707099,
            5.783520058402039e-7,
        ]
    )

    solution = _fit_sigmoid(maf, value)

    assert solution.stats is not None
    assert solution.stats["poor_fit"] is True


def test_fit_curve_sets_non_poor_fit_for_smooth_monotone_series():
    maf = np.asarray([0.001, 0.005, 0.01, 0.02])
    value = np.asarray([0.78, 0.75, 0.73, 0.70])

    solution = _fit_sigmoid(maf, value)

    assert solution.stats is not None
    assert solution.stats["poor_fit"] is False


def test_fit_curve_learns_different_midpoints_for_shifted_curves():
    maf = np.asarray([0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2])
    early_value = evaluate_curve_fit(_sigmoid_result(np.asarray([1.0e-9, 2.0e-4, 6.0, 5.0e-5])), maf)
    late_value = evaluate_curve_fit(_sigmoid_result(np.asarray([1.0e-9, 2.0e-4, 6.0, 2.0e-3])), maf)

    early_solution = _fit_sigmoid(maf, early_value)
    late_solution = _fit_sigmoid(maf, late_value)

    assert early_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert late_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert early_solution.value is not None
    assert late_solution.value is not None
    assert float(early_solution.value.payload[3]) < float(late_solution.value.payload[3])


def test_fit_curve_model_supports_isotonic():
    maf = np.asarray([0.001, 0.002, 0.005, 0.01])
    value = np.asarray([0.2, 0.19, 0.18, 0.17])

    solution = fit_curve_model(maf, value, method="isotonic")

    assert solution.result == RESULTS.successful
    assert solution.value is not None
    fitted = evaluate_curve_fit(solution.value, maf)
    assert fitted.shape == maf.shape
    assert bool(np.all(np.isfinite(fitted)))


def test_curve_pipeline_supports_isotonic_method(tmp_path):
    data_path = _copy_fixture(tmp_path)

    fit_df = run_curve_pipeline(str(data_path), generate_plots=False, method="isotonic")

    assert fit_df.columns == ["var0", "method", "param_name", "param_value"]
    assert fit_df.height > 0
    assert set(fit_df["method"].to_list()) == {"isotonic"}
    assert bool(np.isfinite(_fit_matrix(fit_df)).all())
