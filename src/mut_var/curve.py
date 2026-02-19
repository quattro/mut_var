from __future__ import annotations

import argparse as ap
import sys

from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import polars as pl

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics.curve_fit import curve, fit_curve


def _to_scalar_var(variance) -> float:
    if isinstance(variance, tuple):
        return float(variance[0])
    return float(variance)


def run_curve_workflow(input_path: str, *, generate_plots: bool) -> Solution:
    try:
        df = pl.read_csv(input_path, separator="\t")
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"could not read curve input file: {exc}"},
            state=None,
        )

    required = {"maf", "value", "var0"}
    missing = required.difference(df.columns)
    if missing:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"missing required curve columns: {', '.join(sorted(missing))}"},
            state=None,
        )

    coeff_rows: list[dict[str, float]] = []
    plot_paths: list[str] = []

    grouped = df.sort(["var0", "maf"]).group_by("var0", maintain_order=True)
    for variance, df_sub in grouped:
        var0 = _to_scalar_var(variance)
        maf = jnp.asarray(df_sub["maf"].to_numpy())
        value = jnp.asarray(df_sub["value"].to_numpy())

        fit_solution = fit_curve(maf, value)
        if fit_solution.result != RESULTS.successful:
            details = dict(fit_solution.stats or {})
            details["var0"] = var0
            return Solution(
                value=None,
                result=fit_solution.result,
                stats=details,
                state=fit_solution.state,
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
            plot_paths.append(str(rendered))

    return Solution(
        value={"coefficients": coeff_rows, "plots": plot_paths},
        result=RESULTS.successful,
        stats={"num_curves": len(coeff_rows), "plots_generated": len(plot_paths)},
        state=None,
    )


def build_parser() -> ap.ArgumentParser:
    argp = ap.ArgumentParser(description="")
    argp.add_argument("data")
    argp.add_argument("--fit-only", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)
    return argp


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    solution = run_curve_workflow(args.data, generate_plots=not args.fit_only)
    if solution.result != RESULTS.successful:
        reason = None
        if isinstance(solution.stats, dict):
            reason = solution.stats.get("reason")
        if reason is None:
            reason = f"Curve workflow failed with status '{solution.result.value}'."
        print(reason, file=sys.stderr)
        return 2

    coef_df = pl.DataFrame(solution.value["coefficients"])
    coef_df.write_csv(args.output, separator="\t")
    return 0


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
