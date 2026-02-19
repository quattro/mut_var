from __future__ import annotations

# pattern: Imperative Shell
import argparse as ap
import logging
import sys

from typing import Sequence, TextIO

import jax

from mut_var.curve import run_curve_pipeline
from mut_var.infer import InferenceConfig, run_inference_pipeline
from mut_var.io import read_sumstats

jax.config.update("jax_enable_x64", True)
FMT = ap.ArgumentDefaultsHelpFormatter


def get_logger(name: str) -> logging.Logger:
    r"""Create or reuse a stderr logger for CLI diagnostics.

    **Arguments:**

    - `name`: Logger namespace.

    **Returns:**

    - Configured logger with stderr stream handler.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        console = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt="[%(asctime)s - %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stderr)
    return logger


def _build_infer_subcommand(subparsers: ap._SubParsersAction[ap.ArgumentParser]) -> None:
    infer = subparsers.add_parser(
        "infer",
        formatter_class=FMT,
        help="Run inference pipeline.",
    )
    io_group = infer.add_argument_group("Input/Output")
    io_group.add_argument("sumstats", help="Input summary statistics TSV path.")
    io_group.add_argument(
        "-o",
        "--output",
        type=ap.FileType("w"),
        default=sys.stdout,
        help="Output destination for inference TSV results.",
    )

    data_group = infer.add_argument_group("Input Columns")
    data_group.add_argument(
        "--af-col",
        type=str,
        default="effect_allele_frequency",
        help="Column name for effect allele frequency values.",
    )
    data_group.add_argument(
        "--beta-col",
        type=str,
        default="beta",
        help="Column name for effect size estimates.",
    )
    data_group.add_argument(
        "--se-col",
        type=str,
        default="standard_error",
        help="Column name for standard errors.",
    )

    model_group = infer.add_argument_group("Model Controls")
    model_group.add_argument(
        "-t",
        "--maf-threshold",
        type=float,
        default=0.01,
        help="Reserved MAF threshold parameter for model controls.",
    )
    model_group.add_argument(
        "-k",
        "--num-clusters",
        type=int,
        default=30,
        help="Number of baseline mixture components.",
    )
    model_group.add_argument(
        "-m",
        "--max-iter",
        type=int,
        default=100,
        help="Maximum optimizer iterations.",
    )
    model_group.add_argument(
        "-r",
        "--step-size",
        type=float,
        default=0.01,
        help="Optimization step size.",
    )
    model_group.add_argument("-s", "--seed", type=int, default=0, help="PRNG seed.")
    model_group.add_argument(
        "-f",
        "--filter",
        type=float,
        default=1e-8,
        help="Weight threshold for post-fit component filtering.",
    )
    model_group.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="Penalty weight for objective regularization.",
    )

    grid_group = infer.add_argument_group("MAF Grid")
    grid_group.add_argument("--lowest", type=float, default=1e-5, help="Minimum MAF grid value.")
    grid_group.add_argument("--highest", type=float, default=1e-2, help="Maximum MAF grid value.")
    grid_group.add_argument("--num_breaks", type=int, default=10, help="Number of MAF grid breakpoints.")

    infer.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    infer.set_defaults(func=run_infer_pipeline)


def _build_curve_subcommand(subparsers: ap._SubParsersAction[ap.ArgumentParser]) -> None:
    curve = subparsers.add_parser(
        "curve",
        formatter_class=FMT,
        help="Run curve fitting pipeline and optional plotting.",
    )
    io_group = curve.add_argument_group("Input/Output")
    io_group.add_argument("data", help="Input TSV from `mutvar infer`.")
    io_group.add_argument(
        "-o",
        "--output",
        type=ap.FileType("w"),
        default=sys.stdout,
        help="Output destination for curve coefficient TSV.",
    )

    curve_group = curve.add_argument_group("Curve Options")
    curve_group.add_argument(
        "--fit-only",
        action="store_true",
        default=False,
        help="Fit curves only and skip plot generation.",
    )
    curve.set_defaults(func=run_curve_cli_pipeline)


def build_parser() -> ap.ArgumentParser:
    r"""Build the `mutvar` CLI parser with `infer` and `curve` subcommands."""
    parser = ap.ArgumentParser(description="", formatter_class=FMT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_infer_subcommand(subparsers)
    _build_curve_subcommand(subparsers)
    return parser


def _output_target(output_stream: TextIO) -> str:
    if output_stream is sys.stdout:
        return "stdout"
    name = getattr(output_stream, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return "stream"


def run_infer_pipeline(args: ap.Namespace, log: logging.Logger) -> int:
    r"""Run the CLI inference workflow.

    **Arguments:**

    - `args`: Parsed CLI arguments for `infer`.
    - `log`: Logger used for diagnostics.

    **Returns:**

    - Exit code (`0` success, `2` usage/input errors, `1` runtime failures).
    """
    try:
        log.info("infer: loading data from '%s'", args.sumstats)
        df = read_sumstats(args.sumstats)
        log.info("infer: data loaded (%d rows)", df.height)

        log.info("infer: starting inference pipeline")
        result_df = run_inference_pipeline(
            df,
            af_col=args.af_col,
            beta_col=args.beta_col,
            se_col=args.se_col,
            lowest=args.lowest,
            highest=args.highest,
            num_breaks=args.num_breaks,
            seed=args.seed,
            config=InferenceConfig(
                num_clusters=args.num_clusters,
                max_iter=args.max_iter,
                step_size=args.step_size,
                filter_threshold=args.filter,
                penalty=args.penalty,
            ),
            log=log,
        )
        log.info("infer: inference pipeline completed")
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    log.info("infer: writing output to '%s'", _output_target(args.output))
    result_df.write_csv(args.output, separator="\t")
    log.info("infer: finished writing output")
    return 0


def run_curve_cli_pipeline(args: ap.Namespace, log: logging.Logger) -> int:
    r"""Run the CLI curve-fitting workflow.

    **Arguments:**

    - `args`: Parsed CLI arguments for `curve`.
    - `log`: Logger used for diagnostics.

    **Returns:**

    - Exit code (`0` success, `2` usage/input errors, `1` runtime failures).
    """
    try:
        log.info("curve: starting curve pipeline")
        coef_df = run_curve_pipeline(args.data, generate_plots=not args.fit_only, log=log)
        log.info("curve: curve pipeline completed")
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    log.info("curve: writing output to '%s'", _output_target(args.output))
    coef_df.write_csv(args.output, separator="\t")
    log.info("curve: finished writing output")
    return 0


def run_cli(argv: Sequence[str] | None = None) -> int:
    r"""CLI entrypoint used by console scripts.

    **Arguments:**

    - `argv`: Optional argument vector; defaults to `sys.argv[1:]`.

    **Returns:**

    - Process exit code.
    """
    raw_args = sys.argv[1:] if argv is None else list(argv)
    log = get_logger(__name__)
    log.setLevel(logging.INFO)
    try:
        args = build_parser().parse_args(raw_args)
        if getattr(args, "verbose", False):
            log.setLevel(logging.DEBUG)
        log.debug("cli: parsed args for command '%s'", getattr(args, "command", "unknown"))
        return args.func(args, log)
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
