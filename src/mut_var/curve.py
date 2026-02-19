from __future__ import annotations

# pattern: Imperative Shell
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import polars as pl

from mut_var.contracts import RESULTS
from mut_var.numerics.curve_fit import curve, fit_curve


def _to_scalar_var(variance) -> float:
    if isinstance(variance, tuple):
        return float(variance[0])
    return float(variance)


class CurvePipelineError(RuntimeError):
    result: RESULTS
    stats: dict[str, Any] | None

    def __init__(self, result: RESULTS, reason: str, *, stats: dict[str, Any] | None = None):
        super().__init__(reason)
        self.result = result
        self.stats = stats


def _coefficients_dataframe(coeff_rows: list[dict[str, float]]) -> pl.DataFrame:
    if not coeff_rows:
        return pl.DataFrame(
            schema={
                "var0": pl.Float64,
                "coef_left": pl.Float64,
                "coef_right": pl.Float64,
                "coef_rate": pl.Float64,
            }
        )

    return pl.DataFrame(coeff_rows).select(["var0", "coef_left", "coef_right", "coef_rate"])


def run_curve_pipeline(input_path: str, *, generate_plots: bool) -> pl.DataFrame:
    try:
        df = pl.read_csv(input_path, separator="\t")
    except Exception as exc:
        raise CurvePipelineError(
            result=RESULTS.invalid_input,
            reason=f"could not read curve input file: {exc}",
        ) from exc

    required = {"maf", "value", "var0"}
    missing = required.difference(df.columns)
    if missing:
        raise CurvePipelineError(
            result=RESULTS.invalid_input,
            reason=f"missing required curve columns: {', '.join(sorted(missing))}",
        )

    coeff_rows: list[dict[str, float]] = []

    grouped = df.sort(["var0", "maf"]).group_by("var0", maintain_order=True)
    for variance, df_sub in grouped:
        var0 = _to_scalar_var(variance)
        maf = jnp.asarray(df_sub["maf"].to_jax())
        value = jnp.asarray(df_sub["value"].to_jax())

        fit_solution = fit_curve(maf, value)
        if fit_solution.result != RESULTS.successful:
            details = dict(fit_solution.stats or {})
            details["var0"] = var0
            reason = details.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = f"curve fit failed with status '{RESULTS[fit_solution.result]}' at var0={var0}."
            raise CurvePipelineError(
                result=fit_solution.result,
                reason=reason,
                stats=details,
            )

        coef = fit_solution.value
        coeff_rows.append(
            {
                "var0": var0,
                "coef_left": float(coef[0]),
                "coef_right": float(coef[1]),
                "coef_rate": float(coef[2]),
            }
        )

        if generate_plots:
            from mut_var.plotting.curve_plots import render_curve_plot

            maf_space = jnp.linspace(float(maf.min()), float(maf.max()), 200)
            fitted_values = curve(maf_space, coef)
            out_path = Path(f"{input_path}_{var0:.6g}.png")
            rendered = render_curve_plot(
                maf=maf,
                value=value,
                maf_space=maf_space,
                fitted_values=fitted_values,
                title=f"var0 = {var0}",
                output_path=out_path,
            )
            _ = rendered

    return _coefficients_dataframe(coeff_rows)
