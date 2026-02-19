from __future__ import annotations

import argparse as ap
import sys

from typing import Sequence

import jax
import jax.numpy as jnp
import polars as pl

from mut_var.adapters.tabular import build_maf_masks, payload_to_long_dataframe, to_inference_arrays
from mut_var.contracts import RESULTS
from mut_var.curve import run_curve_workflow
from mut_var.io import (
    MutVarInputError,
    read_sumstats,
    validate_maf_grid,
    validate_numeric_columns,
    validate_required_columns,
    validate_sumstats_domain,
)
from mut_var.numerics.pipeline import InferenceConfig, run_inference_pipeline

jax.config.update("jax_enable_x64", True)


def _build_infer_subcommand(subparsers: ap._SubParsersAction[ap.ArgumentParser]) -> None:
    infer = subparsers.add_parser("infer", help="Run inference workflow.")
    io_group = infer.add_argument_group("Input/Output")
    io_group.add_argument("sumstats")
    io_group.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    data_group = infer.add_argument_group("Input Columns")
    data_group.add_argument("--af-col", type=str, default="effect_allele_frequency")
    data_group.add_argument("--beta-col", type=str, default="beta")
    data_group.add_argument("--se-col", type=str, default="standard_error")

    model_group = infer.add_argument_group("Model Controls")
    model_group.add_argument("-t", "--maf-threshold", type=float, default=0.01)
    model_group.add_argument("-k", "--num-clusters", type=int, default=30)
    model_group.add_argument("-m", "--max-iter", type=int, default=100)
    model_group.add_argument("-r", "--step-size", type=float, default=0.01)
    model_group.add_argument("-b", "--batch-size", type=int, default=10_000)
    model_group.add_argument("-s", "--seed", type=int, default=0)
    model_group.add_argument("-f", "--filter", type=float, default=1e-8)
    model_group.add_argument("--penalty", type=float, default=1.0)

    grid_group = infer.add_argument_group("MAF Grid")
    grid_group.add_argument("--lowest", type=float, default=1e-5)
    grid_group.add_argument("--highest", type=float, default=1e-2)
    grid_group.add_argument("--num_breaks", type=int, default=10)

    infer.add_argument("-v", "--verbose", action="store_true", default=False)
    infer.set_defaults(func=run_infer_workflow)


def _build_curve_subcommand(subparsers: ap._SubParsersAction[ap.ArgumentParser]) -> None:
    curve = subparsers.add_parser("curve", help="Run curve fitting and optional plotting.")
    io_group = curve.add_argument_group("Input/Output")
    io_group.add_argument("data")
    io_group.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    curve_group = curve.add_argument_group("Curve Options")
    curve_group.add_argument("--fit-only", action="store_true", default=False)
    curve.set_defaults(func=run_curve_cli_workflow)


def build_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_infer_subcommand(subparsers)
    _build_curve_subcommand(subparsers)
    return parser


def _solution_exit_code(result: RESULTS) -> int:
    if result in (RESULTS.successful, RESULTS.max_steps_reached):
        return 0
    if result in (RESULTS.invalid_input, RESULTS.empty_subset):
        return 2
    return 1


def run_infer_workflow(args: ap.Namespace) -> int:
    validate_maf_grid(args.lowest, args.highest, args.num_breaks)

    df = read_sumstats(args.sumstats)
    validate_required_columns(df, args.af_col, args.beta_col, args.se_col)
    validate_numeric_columns(df, args.af_col, args.beta_col, args.se_col)
    validate_sumstats_domain(df, args.af_col, args.se_col)

    arrays = to_inference_arrays(
        df,
        af_col=args.af_col,
        beta_col=args.beta_col,
        se_col=args.se_col,
    )
    maf_grid = jnp.exp(jnp.linspace(jnp.log(args.lowest), jnp.log(args.highest), args.num_breaks))
    maf_masks = build_maf_masks(arrays.af, maf_grid)

    solution = run_inference_pipeline(
        arrays=arrays,
        maf_grid=maf_grid,
        maf_masks=maf_masks,
        seed=args.seed,
        config=InferenceConfig(
            num_clusters=args.num_clusters,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            step_size=args.step_size,
            filter_threshold=args.filter,
            penalty=args.penalty,
        ),
    )

    if solution.value is not None and solution.result in (RESULTS.successful, RESULTS.max_steps_reached):
        payload_to_long_dataframe(solution.value).write_csv(args.output, separator="\t")

    if solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        reason = None
        if isinstance(solution.stats, dict):
            reason = solution.stats.get("reason")
        if reason is None:
            reason = f"Inference failed with status '{solution.result.value}'."
        print(reason, file=sys.stderr)

    return _solution_exit_code(solution.result)


def run_curve_cli_workflow(args: ap.Namespace) -> int:
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


def run_cli(argv: Sequence[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else list(argv)
    try:
        args = build_parser().parse_args(raw_args)
        return args.func(args)
    except MutVarInputError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
