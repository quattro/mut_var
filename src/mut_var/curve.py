from __future__ import annotations

# pattern: Imperative Shell
import logging

from pathlib import Path

import jax.numpy as jnp
import polars as pl

from mut_var.contracts import RESULTS
from mut_var.numerics.curve_fit import curve, fit_curve


def _to_scalar_var(variance) -> float:
    if isinstance(variance, tuple):
        return float(variance[0])
    return float(variance)


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


def run_curve_pipeline(
    input_path: str,
    *,
    generate_plots: bool,
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    r"""Run curve fitting for each `var0` group from tabular input.

    **Arguments:**

    - `input_path`: Tab-delimited input path with `maf`, `value`, and `var0` columns.
    - `generate_plots`: When `True`, render one PNG per `var0` group.
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**

    - Coefficient dataframe (`var0`, `coef_left`, `coef_right`, `coef_rate`).

    **Raises:**

    - `FileNotFoundError`: Input path does not exist.
    - `ValueError`: Input schema/content invalid or invalid-fit status.
    - `RuntimeError`: Non-recoverable fitting failure.
    """
    workflow_log = logging.getLogger(__name__) if log is None else log

    workflow_log.info("curve pipeline: loading input data from '%s'", input_path)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    try:
        df = pl.read_csv(input_path, separator="\t")
    except Exception as exc:
        raise ValueError(f"could not read curve input file: {exc}") from exc
    workflow_log.info("curve pipeline: data loaded (%d rows)", df.height)

    workflow_log.info("curve pipeline: validating required columns")
    required = {"maf", "value", "var0"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing required curve columns: {', '.join(sorted(missing))}")
    workflow_log.info("curve pipeline: input validation complete")

    coeff_rows: list[dict[str, float]] = []

    workflow_log.info("curve pipeline: starting curve fitting")
    grouped = df.sort(["var0", "maf"]).group_by("var0", maintain_order=True)
    for variance, df_sub in grouped:
        var0 = _to_scalar_var(variance)
        workflow_log.debug("curve pipeline: fitting variance bin var0=%s", var0)
        maf = jnp.asarray(df_sub["maf"].to_jax())
        value = jnp.asarray(df_sub["value"].to_jax())

        fit_solution = fit_curve(maf, value)
        if fit_solution.result != RESULTS.successful:
            details = dict(fit_solution.stats or {})
            details["var0"] = var0
            reason = details.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                reason = f"curve fit failed with status '{RESULTS[fit_solution.result]}' at var0={var0}."
            if fit_solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
                raise ValueError(reason)
            raise RuntimeError(reason)

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

            workflow_log.debug("curve pipeline: rendering plot for var0=%s", var0)
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

    workflow_log.info("curve pipeline: curve fitting completed")
    workflow_log.info("curve pipeline: preparing output dataframe")
    coef_df = _coefficients_dataframe(coeff_rows)
    workflow_log.info("curve pipeline: output dataframe prepared (%d rows)", coef_df.height)
    return coef_df
