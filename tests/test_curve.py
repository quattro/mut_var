from __future__ import annotations

import logging
import shutil
import sys

from pathlib import Path

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import polars as pl

from mut_var.numerics.curve_fit import curve, fit_curve
from mut_var.pipelines.curve import run_curve_pipeline
from mut_var.types import RESULTS, Solution

FIXTURE = Path(__file__).parent / "fixtures" / "curve_small.tsv"


def _copy_fixture(tmp_path: Path) -> Path:
    data_path = tmp_path / "curve_small.tsv"
    shutil.copy(FIXTURE, data_path)
    return data_path


def _coef_matrix(coef_df: pl.DataFrame) -> jnp.ndarray:
    ordered = coef_df.sort("var0")
    selected = ordered.select(["coef_left", "coef_right", "coef_rate", "coef_midpoint"]).to_numpy()
    return jnp.asarray(selected, dtype=float)


def _expected_midpoint(raw_mid: float, maf: jnp.ndarray) -> float:
    maf_np = np.asarray(maf, dtype=float)
    positive = maf_np[maf_np > 0.0]
    if positive.size == 0:
        log_min = np.log(1e-12)
        log_span = 1.0
    else:
        log_min = float(np.log(np.min(positive) + 1e-12))
        log_span = max(float(np.log(np.max(positive) + 1e-12) - log_min), 1e-6)
    return float(np.exp(log_min + float(jnn.sigmoid(jnp.asarray(raw_mid))) * log_span))


def test_fit_only_numerics_has_no_plotting_dependency():
    sys.modules.pop("matplotlib.pyplot", None)

    maf = jnp.asarray([0.001, 0.005, 0.01])
    value = jnp.asarray([0.2, 0.21, 0.22])
    solution = fit_curve(maf, value)

    assert solution.result == RESULTS.successful
    assert "matplotlib.pyplot" not in sys.modules


def test_fit_only_pipeline_is_deterministic_and_side_effect_free(tmp_path):
    data_path = _copy_fixture(tmp_path)
    sys.modules.pop("mut_var.plotting.curve_plots", None)

    first = run_curve_pipeline(str(data_path), generate_plots=False)
    second = run_curve_pipeline(str(data_path), generate_plots=False)

    assert first.columns == ["var0", "coef_left", "coef_right", "coef_rate", "coef_midpoint"]
    assert second.columns == ["var0", "coef_left", "coef_right", "coef_rate", "coef_midpoint"]
    assert first.height > 0
    assert second.height > 0
    assert bool(jnp.allclose(_coef_matrix(first), _coef_matrix(second), rtol=1e-6, atol=1e-6))

    assert "mut_var.plotting.curve_plots" not in sys.modules
    assert list(tmp_path.glob("*.png")) == []


def test_plotting_mode_writes_png_side_effects(tmp_path):
    data_path = _copy_fixture(tmp_path)

    coef_df = run_curve_pipeline(str(data_path), generate_plots=True)

    assert coef_df.height > 0
    png_paths = list(tmp_path.glob("*.png"))
    assert len(png_paths) > 0
    assert {path.name for path in png_paths} == {
        "curve_small.tsv_var_0.png",
        "curve_small.tsv_var_1.png",
    }


def test_plotting_does_not_change_fit_outputs(tmp_path):
    data_path = _copy_fixture(tmp_path)

    fit_only = run_curve_pipeline(str(data_path), generate_plots=False)
    with_plots = run_curve_pipeline(str(data_path), generate_plots=True)

    assert fit_only.height > 0
    assert with_plots.height > 0
    assert bool(jnp.allclose(_coef_matrix(fit_only), _coef_matrix(with_plots), rtol=1e-6, atol=1e-6))


def test_plot_titles_use_component_index_labels(monkeypatch, tmp_path):
    data_path = _copy_fixture(tmp_path)
    captured_titles: list[str] = []
    captured_paths: list[Path] = []

    def _fake_render_curve_plot(*, title, output_path, **kwargs):  # noqa: ANN003
        del kwargs
        captured_titles.append(title)
        captured_paths.append(Path(output_path))
        return Path(output_path)

    monkeypatch.setattr("mut_var.plotting.curve_plots.render_curve_plot", _fake_render_curve_plot)

    run_curve_pipeline(str(data_path), generate_plots=True)

    assert captured_titles == ["var_0 = 0.1", "var_1 = 0.2"]
    assert [path.name for path in captured_paths] == [
        "curve_small.tsv_var_0.png",
        "curve_small.tsv_var_1.png",
    ]


def test_fit_curve_passes_throw_false_and_returns_clean_success_stats(monkeypatch):
    class _FakeOptxSolution:
        value = jnp.asarray([0.25, 0.75, 1.0, 0.0], dtype=float)
        result = optx.RESULTS.successful
        stats = {"num_steps": 13}

    def _fake_least_squares(*args, **kwargs):
        assert kwargs.get("throw") is False
        return _FakeOptxSolution()

    monkeypatch.setattr("mut_var.numerics.curve_fit.optx.least_squares", _fake_least_squares)

    solution = fit_curve(jnp.asarray([0.001, 0.005, 0.01]), jnp.asarray([0.2, 0.21, 0.22]))

    assert solution.result == RESULTS.successful
    assert solution.value is not None
    expected_left = float(jnn.sigmoid(jnp.asarray(0.25)))
    expected_right = expected_left + (1.0 - expected_left) * float(jnn.sigmoid(jnp.asarray(0.75)))
    expected_midpoint = _expected_midpoint(0.0, jnp.asarray([0.001, 0.005, 0.01]))
    assert bool(
        jnp.allclose(
            solution.value,
            jnp.asarray([expected_left, expected_right, 1.0, expected_midpoint]),
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


def test_fit_curve_maps_max_steps_reached_from_optimistix(monkeypatch):
    class _FakeOptxSolution:
        value = jnp.asarray([0.1, 0.2, 0.3, 0.0], dtype=float)
        result = optx.RESULTS.max_steps_reached
        stats = {"num_steps": 1000}

    monkeypatch.setattr("mut_var.numerics.curve_fit.optx.least_squares", lambda *args, **kwargs: _FakeOptxSolution())

    solution = fit_curve(jnp.asarray([0.001, 0.005, 0.01]), jnp.asarray([0.2, 0.21, 0.22]))

    assert solution.result == RESULTS.max_steps_reached
    assert solution.value is not None
    expected_left = float(jnn.sigmoid(jnp.asarray(0.1)))
    expected_right = expected_left + (1.0 - expected_left) * float(jnn.sigmoid(jnp.asarray(0.2)))
    expected_midpoint = _expected_midpoint(0.0, jnp.asarray([0.001, 0.005, 0.01]))
    assert bool(
        jnp.allclose(
            solution.value,
            jnp.asarray([expected_left, expected_right, 0.3, expected_midpoint]),
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
    maf = jnp.asarray(
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
    value = jnp.asarray(
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

    solution = fit_curve(maf, value)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    left, right, _rate = [float(x) for x in solution.value[:3]]
    assert 0.0 <= left <= right <= 1.0
    midpoint = float(solution.value[3])
    positive_maf = maf[maf > 0.0]
    assert float(positive_maf.min()) <= midpoint <= float(positive_maf.max())

    fitted = curve(maf, solution.value)
    assert bool(jnp.isfinite(fitted).all())
    assert bool((fitted >= 0.0).all())
    assert bool((fitted <= 1.0).all())


def test_fit_curve_bounded_coefficients_and_predictions_for_increasing_step_like_series():
    maf = jnp.asarray(
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
    value = jnp.asarray(
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

    solution = fit_curve(maf, value)

    assert solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert solution.value is not None
    left, right, _rate = [float(x) for x in solution.value[:3]]
    assert 0.0 <= left <= right <= 1.0
    midpoint = float(solution.value[3])
    positive_maf = maf[maf > 0.0]
    assert float(positive_maf.min()) <= midpoint <= float(positive_maf.max())

    fitted = curve(maf, solution.value)
    assert bool(jnp.isfinite(fitted).all())
    assert bool((fitted >= 0.0).all())
    assert bool((fitted <= 1.0).all())


def test_curve_pipeline_accepts_max_steps_reached(monkeypatch, tmp_path):
    data_path = _copy_fixture(tmp_path)

    def _fake_fit_curve(maf, value):
        del maf, value
        return Solution(
            value=jnp.asarray([0.2, 0.8, -2.0, 0.003]),
            result=RESULTS.max_steps_reached,
            stats={"n_obs": 4, "epoch_count": 1000, "converged": False},
            state=None,
        )

    monkeypatch.setattr("mut_var.pipelines.curve.fit_curve", _fake_fit_curve)

    coef_df = run_curve_pipeline(str(data_path), generate_plots=False)

    assert coef_df.columns == ["var0", "coef_left", "coef_right", "coef_rate", "coef_midpoint"]
    assert coef_df.height == 2


def test_curve_pipeline_logs_warning_for_poor_fit(monkeypatch, tmp_path, caplog):
    data_path = _copy_fixture(tmp_path)

    def _fake_fit_curve(maf, value):
        del maf, value
        return Solution(
            value=jnp.asarray([0.2, 0.8, -2.0, 0.003]),
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

    monkeypatch.setattr("mut_var.pipelines.curve.fit_curve", _fake_fit_curve)

    with caplog.at_level(logging.WARNING):
        run_curve_pipeline(str(data_path), generate_plots=False)

    assert any("poor-fit diagnostics" in record.message for record in caplog.records)


def test_fit_curve_rate_initialization_follows_endpoint_trend(monkeypatch):
    class _FakeOptxSolution:
        def __init__(self, y0):
            self.value = y0
            self.result = optx.RESULTS.max_steps_reached
            self.stats = {"num_steps": 1}

    def _fake_least_squares(*args, **kwargs):
        return _FakeOptxSolution(kwargs["y0"])

    monkeypatch.setattr("mut_var.numerics.curve_fit.optx.least_squares", _fake_least_squares)

    maf = jnp.asarray([0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    decreasing = jnp.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    increasing = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])

    dec_sol = fit_curve(maf, decreasing)
    inc_sol = fit_curve(maf, increasing)

    assert dec_sol.value is not None
    assert inc_sol.value is not None
    assert float(dec_sol.value[2]) > 0.0
    assert float(inc_sol.value[2]) < 0.0


def test_fit_curve_sets_poor_fit_for_step_like_nonmonotone_series():
    maf = jnp.asarray(
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
    value = jnp.asarray(
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

    solution = fit_curve(maf, value)

    assert solution.stats is not None
    assert solution.stats["poor_fit"] is True


def test_fit_curve_sets_non_poor_fit_for_smooth_monotone_series():
    maf = jnp.asarray([0.001, 0.005, 0.01, 0.02])
    value = jnp.asarray([0.78, 0.75, 0.73, 0.70])

    solution = fit_curve(maf, value)

    assert solution.stats is not None
    assert solution.stats["poor_fit"] is False


def test_fit_curve_learns_different_midpoints_for_shifted_curves():
    maf = jnp.asarray([0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2])
    early_value = curve(maf, jnp.asarray([1.0e-9, 2.0e-4, 6.0, 5.0e-5]))
    late_value = curve(maf, jnp.asarray([1.0e-9, 2.0e-4, 6.0, 2.0e-3]))

    early_solution = fit_curve(maf, early_value)
    late_solution = fit_curve(maf, late_value)

    assert early_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert late_solution.result in (RESULTS.successful, RESULTS.max_steps_reached)
    assert early_solution.value is not None
    assert late_solution.value is not None
    assert float(early_solution.value[3]) < float(late_solution.value[3])
