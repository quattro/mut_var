from __future__ import annotations

import shutil
import sys

from pathlib import Path

import jax.numpy as jnp

from mut_var.contracts import RESULTS
from mut_var.curve import run_curve_workflow
from mut_var.numerics.curve_fit import fit_curve

FIXTURE = Path(__file__).parent / "fixtures" / "curve_small.tsv"


def _copy_fixture(tmp_path: Path) -> Path:
    data_path = tmp_path / "curve_small.tsv"
    shutil.copy(FIXTURE, data_path)
    return data_path


def _coef_matrix(solution_value: dict) -> jnp.ndarray:
    rows = sorted(solution_value["coefficients"], key=lambda row: row["var0"])
    return jnp.asarray([[r["coef_left"], r["coef_right"], r["coef_rate"]] for r in rows], dtype=float)


def test_fit_only_numerics_has_no_plotting_dependency():
    sys.modules.pop("matplotlib.pyplot", None)

    maf = jnp.asarray([0.001, 0.005, 0.01])
    value = jnp.asarray([0.2, 0.21, 0.22])
    solution = fit_curve(maf, value)

    assert solution.result == RESULTS.successful
    assert "matplotlib.pyplot" not in sys.modules


def test_fit_only_workflow_is_deterministic_and_side_effect_free(tmp_path):
    data_path = _copy_fixture(tmp_path)
    sys.modules.pop("mut_var.plotting.curve_plots", None)

    first = run_curve_workflow(str(data_path), generate_plots=False)
    second = run_curve_workflow(str(data_path), generate_plots=False)

    assert first.result == RESULTS.successful
    assert second.result == RESULTS.successful
    assert first.value["plots"] == []
    assert second.value["plots"] == []
    assert bool(jnp.allclose(_coef_matrix(first.value), _coef_matrix(second.value), rtol=1e-6, atol=1e-6))

    assert "mut_var.plotting.curve_plots" not in sys.modules
    assert list(tmp_path.glob("*.png")) == []


def test_plotting_mode_writes_png_side_effects(tmp_path):
    data_path = _copy_fixture(tmp_path)

    solution = run_curve_workflow(str(data_path), generate_plots=True)

    assert solution.result == RESULTS.successful
    assert solution.stats["plots_generated"] > 0
    assert len(solution.value["plots"]) == solution.stats["plots_generated"]
    for path in solution.value["plots"]:
        p = Path(path)
        assert p.exists()
        assert p.suffix == ".png"


def test_plotting_does_not_change_fit_outputs(tmp_path):
    data_path = _copy_fixture(tmp_path)

    fit_only = run_curve_workflow(str(data_path), generate_plots=False)
    with_plots = run_curve_workflow(str(data_path), generate_plots=True)

    assert fit_only.result == RESULTS.successful
    assert with_plots.result == RESULTS.successful
    assert bool(jnp.allclose(_coef_matrix(fit_only.value), _coef_matrix(with_plots.value), rtol=1e-6, atol=1e-6))
