from __future__ import annotations

import shutil
import sys

from pathlib import Path

import jax.numpy as jnp
import polars as pl

from mut_var.contracts import RESULTS
from mut_var.curve import run_curve_pipeline
from mut_var.numerics.curve_fit import fit_curve

FIXTURE = Path(__file__).parent / "fixtures" / "curve_small.tsv"


def _copy_fixture(tmp_path: Path) -> Path:
    data_path = tmp_path / "curve_small.tsv"
    shutil.copy(FIXTURE, data_path)
    return data_path


def _coef_matrix(coef_df: pl.DataFrame) -> jnp.ndarray:
    ordered = coef_df.sort("var0")
    return jnp.asarray(ordered.select(["coef_left", "coef_right", "coef_rate"]).to_numpy(), dtype=float)


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

    assert first.columns == ["var0", "coef_left", "coef_right", "coef_rate"]
    assert second.columns == ["var0", "coef_left", "coef_right", "coef_rate"]
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
    for path in png_paths:
        assert path.exists()
        assert path.suffix == ".png"


def test_plotting_does_not_change_fit_outputs(tmp_path):
    data_path = _copy_fixture(tmp_path)

    fit_only = run_curve_pipeline(str(data_path), generate_plots=False)
    with_plots = run_curve_pipeline(str(data_path), generate_plots=True)

    assert fit_only.height > 0
    assert with_plots.height > 0
    assert bool(jnp.allclose(_coef_matrix(fit_only), _coef_matrix(with_plots), rtol=1e-6, atol=1e-6))
