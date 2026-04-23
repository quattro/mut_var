from __future__ import annotations

# pattern: Imperative Shell
import logging

from pathlib import Path

import numpy as np
import polars as pl

from mut_var.numerics.curve_fit import CurveFitResult, CurveMethod, evaluate_curve_fit, fit_curve_model
from mut_var.types import RESULTS, Solution


def _to_scalar_var(variance) -> float:
    # polars.group_by returns the group key as a tuple of key columns.
    if isinstance(variance, tuple):
        return float(variance[0])
    return float(variance)


def _parameter_rows(var0: float, fit: CurveFitResult) -> list[dict[str, str | float]]:
    if fit.method == "sigmoid":
        names = ("left", "right", "rate", "midpoint")
        values = np.asarray(fit.payload, dtype=float)
        return [
            {
                "var0": var0,
                "method": fit.method,
                "param_name": name,
                "param_value": float(value),
            }
            for name, value in zip(names, values, strict=True)
        ]

    # Isotonic and mono_spline fits share the same tabular shape: a direction
    # flag plus paired (x_i, y_i) rows on the unique MAF support. The downstream
    # consumer reassembles the step function (isotonic) or the PCHIP-interpolated
    # curve (mono_spline) from this representation.
    rows: list[dict[str, str | float]] = [
        {
            "var0": var0,
            "method": fit.method,
            "param_name": "increasing",
            "param_value": float(bool(fit.increasing)),
        }
    ]
    support = np.asarray(fit.support, dtype=float)
    payload = np.asarray(fit.payload, dtype=float)
    for idx, (x_val, y_val) in enumerate(zip(support, payload, strict=True)):
        rows.append(
            {
                "var0": var0,
                "method": fit.method,
                "param_name": f"x_{idx}",
                "param_value": float(x_val),
            }
        )
        rows.append(
            {
                "var0": var0,
                "method": fit.method,
                "param_name": f"y_{idx}",
                "param_value": float(y_val),
            }
        )
    return rows


def _parameters_dataframe(rows: list[dict[str, str | float]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "var0": pl.Float64,
                "method": pl.Utf8,
                "param_name": pl.Utf8,
                "param_value": pl.Float64,
            }
        )
    return pl.DataFrame(rows).select(["var0", "method", "param_name", "param_value"])


def _reason_from_solution(solution: Solution, var0: float) -> str:
    if isinstance(solution.stats, dict):
        reason = solution.stats.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason
    return f"curve fit failed with status '{solution.result.value}' at var0={var0}."


def run_curve_pipeline(
    input_path: str,
    *,
    generate_plots: bool,
    method: CurveMethod = "sigmoid",
    log: logging.Logger | None = None,
) -> pl.DataFrame:
    r"""Run curve fitting for each `var0` group from tabular input.

    **Arguments:**

    - `input_path`: Tab-delimited input path with `maf`, `value`, and `var0` columns.
    - `generate_plots`: When `True`, render one PNG per `var0` group.
    - `method`: Curve method to apply (`sigmoid` or `isotonic`).
    - `log`: Optional logger for workflow diagnostics.

    **Returns:**

    - Method-neutral parameter dataframe (`var0`, `method`, `param_name`, `param_value`).

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
    workflow_log.info("curve pipeline: using method '%s'", method)

    parameter_rows: list[dict[str, str | float]] = []

    workflow_log.info("curve pipeline: starting curve fitting")
    grouped = df.sort(["var0", "maf"]).group_by("var0", maintain_order=True)
    for component_idx, (variance, df_sub) in enumerate(grouped):
        var0 = _to_scalar_var(variance)
        component_label = f"var_{component_idx}"
        workflow_log.debug("curve pipeline: fitting variance bin var0=%s", var0)
        maf = np.asarray(df_sub["maf"].to_numpy(), dtype=float)
        value = np.asarray(df_sub["value"].to_numpy(), dtype=float)

        fit_solution = fit_curve_model(maf, value, method=method)
        fit_stats = dict(fit_solution.stats or {})
        if fit_solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
            reason = _reason_from_solution(fit_solution, var0)
            if fit_solution.result in (RESULTS.invalid_input, RESULTS.empty_subset):
                raise ValueError(reason)
            raise RuntimeError(reason)
        if fit_solution.result == RESULTS.max_steps_reached:
            workflow_log.warning("curve pipeline: max steps reached at var0=%s; using last finite iterate", var0)
        if bool(fit_stats.get("poor_fit")):
            workflow_log.warning(
                "curve pipeline: poor-fit diagnostics at var0=%s (rmse=%.6g, max_abs_error=%.6g, sign_changes=%s)",
                var0,
                float(fit_stats.get("rmse", float("nan"))),
                float(fit_stats.get("max_abs_error", float("nan"))),
                fit_stats.get("data_sign_changes"),
            )

        fit_result = fit_solution.value
        parameter_rows.extend(_parameter_rows(var0, fit_result))

        if generate_plots:
            from mut_var.plotting.curve_plots import render_curve_plot

            workflow_log.debug("curve pipeline: rendering plot for var0=%s", var0)
            positive_maf = maf[maf > 0.0]
            maf_min = float(np.min(positive_maf)) if positive_maf.size > 0 else 1e-12
            maf_max = float(np.max(maf)) if maf.size > 0 else maf_min
            maf_space = np.geomspace(max(maf_min, 1e-12), max(maf_max, maf_min), 200)
            fitted_values = evaluate_curve_fit(fit_result, maf_space)
            out_path = Path(f"{input_path}_{method}_{component_label}.png")
            _ = render_curve_plot(
                maf=maf,
                value=value,
                maf_space=maf_space,
                fitted_values=fitted_values,
                title=f"{method} | {component_label} = {var0:.6g}",
                output_path=out_path,
            )

    workflow_log.info("curve pipeline: curve fitting completed")
    workflow_log.info("curve pipeline: preparing output dataframe")
    param_df = _parameters_dataframe(parameter_rows)
    workflow_log.info("curve pipeline: output dataframe prepared (%d rows)", param_df.height)
    return param_df


__all__ = ["run_curve_pipeline"]
