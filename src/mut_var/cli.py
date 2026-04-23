from __future__ import annotations

# pattern: Imperative Shell
import argparse as ap
import logging
import sys

from pathlib import Path
from typing import Sequence, TextIO

from mut_var.pipelines import (
    run_curve_pipeline,
    run_inference_pipeline,
    run_simulation_pipeline,
)
from mut_var.types import InferenceConfig, SimulationConfig

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
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute convergence tolerance for mix-SQP outer iterations.",
    )
    model_group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative convergence tolerance for mix-SQP outer iterations.",
    )
    model_group.add_argument(
        "-f",
        "--filter",
        type=float,
        default=1e-8,
        help="Weight threshold for post-fit component filtering.",
    )
    grid_group = infer.add_argument_group("MAF Grid")
    grid_group.add_argument("--lowest", type=float, default=1e-5, help="Minimum MAF grid value.")
    grid_group.add_argument("--highest", type=float, default=1e-2, help="Maximum MAF grid value.")
    grid_group.add_argument("--num-breaks", type=int, default=10, help="Number of MAF grid breakpoints.")

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
        help="Output destination for curve fit TSV.",
    )

    curve_group = curve.add_argument_group("Curve Options")
    curve_group.add_argument(
        "--method",
        type=str,
        choices=("sigmoid", "isotonic", "mono_spline"),
        default="sigmoid",
        help="Curve fitting method.",
    )
    curve_group.add_argument(
        "--fit-only",
        action="store_true",
        default=False,
        help="Fit curves only and skip plot generation.",
    )
    curve.set_defaults(func=run_curve_cli_pipeline)


def _build_simulate_subcommand(subparsers: ap._SubParsersAction[ap.ArgumentParser]) -> None:
    simulate = subparsers.add_parser(
        "simulate",
        formatter_class=FMT,
        help="Simulate mixture summary-stat artifacts.",
    )

    output_group = simulate.add_argument_group("Output")
    output_group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Filename prefix for <prefix>.truth.tsv/.observed.tsv/.meta.tsv outputs.",
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for simulation output files.",
    )

    core_group = simulate.add_argument_group("Core")
    core_group.add_argument(
        "--n-rows",
        type=int,
        default=10000,
        help="Number of rows to simulate.",
    )
    core_group.add_argument("--seed", type=int, default=0, help="PRNG seed.")

    mixture_group = simulate.add_argument_group("Mixture")
    mixture_group.add_argument(
        "--weights",
        type=str,
        default="0.95,0.05",
        help="Comma-delimited mixture weights.",
    )
    mixture_group.add_argument(
        "--log-var-scales",
        type=str,
        default="-8.0,-5.5",
        help="Comma-delimited component log variance scales.",
    )

    link_group = simulate.add_argument_group("Variance Link")
    link_group.add_argument(
        "--variance-link",
        type=str,
        choices=("none", "maf_power", "maf_power_shifted"),
        default="maf_power",
        help="AF-dependent variance link family.",
    )
    link_group.add_argument("--theta", type=float, default=0.5, help="Variance-link exponent.")
    link_group.add_argument("--link-eps", type=float, default=1e-8, help="Positive epsilon for maf_power.")
    link_group.add_argument(
        "--link-shift",
        type=float,
        default=0.0,
        help="Positive shift for maf_power_shifted.",
    )
    link_group.add_argument(
        "--af-clip-min",
        type=float,
        default=1e-4,
        help="AF clipping lower bound before link/SE evaluation.",
    )

    af_group = simulate.add_argument_group("AF Model")
    af_group.add_argument(
        "--af-model",
        type=str,
        choices=("uniform", "beta"),
        default="beta",
        help="Allele-frequency generator family.",
    )
    af_group.add_argument("--af-uniform-low", type=float, default=0.01, help="Uniform AF lower bound.")
    af_group.add_argument("--af-uniform-high", type=float, default=0.5, help="Uniform AF upper bound.")
    af_group.add_argument("--af-beta-a", type=float, default=0.4, help="Beta AF shape parameter a.")
    af_group.add_argument("--af-beta-b", type=float, default=0.4, help="Beta AF shape parameter b.")

    se_group = simulate.add_argument_group("SE Model")
    se_group.add_argument(
        "--se-model",
        type=str,
        choices=("constant", "af_n_scaled"),
        default="af_n_scaled",
        help="Standard-error generator family.",
    )
    se_group.add_argument("--se-constant", type=float, default=0.02, help="Constant SE value.")
    se_group.add_argument("--sample-size", type=float, default=50000.0, help="Effective sample size.")
    se_group.add_argument("--se-scale", type=float, default=1.0, help="SE scaling factor.")

    simulate.add_argument_group("Diagnostics").add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    simulate.set_defaults(func=run_simulate_cli_pipeline)


def build_parser() -> ap.ArgumentParser:
    r"""Build the `mutvar` CLI parser with `infer`, `curve`, and `simulate` subcommands."""
    parser = ap.ArgumentParser(description="", formatter_class=FMT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_infer_subcommand(subparsers)
    _build_curve_subcommand(subparsers)
    _build_simulate_subcommand(subparsers)
    return parser


def _output_target(output_stream: TextIO) -> str:
    if output_stream is sys.stdout:
        return "stdout"
    name = getattr(output_stream, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return "stream"


def _parse_comma_floats(raw: str, field: str) -> tuple[float, ...]:
    text = raw.strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty comma-delimited float list")

    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"{field} must not contain empty comma-delimited entries")

    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{field} must contain only comma-delimited floats") from exc


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
        log.info("infer: starting inference pipeline")
        result_df = run_inference_pipeline(
            args.sumstats,
            af_col=args.af_col,
            beta_col=args.beta_col,
            se_col=args.se_col,
            lowest=args.lowest,
            highest=args.highest,
            num_breaks=args.num_breaks,
            config=InferenceConfig(
                num_clusters=args.num_clusters,
                max_iter=args.max_iter,
                atol=args.atol,
                rtol=args.rtol,
                filter_threshold=args.filter,
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
        fit_df = run_curve_pipeline(
            args.data,
            generate_plots=not args.fit_only,
            method=args.method,
            log=log,
        )
        log.info("curve: curve pipeline completed")
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    log.info("curve: writing output to '%s'", _output_target(args.output))
    fit_df.write_csv(args.output, separator="\t")
    log.info("curve: finished writing output")
    return 0


def run_simulate_cli_pipeline(args: ap.Namespace, log: logging.Logger) -> int:
    r"""Run the CLI simulation workflow.

    **Arguments:**

    - `args`: Parsed CLI arguments for `simulate`.
    - `log`: Logger used for diagnostics.

    **Returns:**

    - Exit code (`0` success, `2` usage/input errors, `1` runtime failures).
    """
    try:
        log.info("simulate: validating args")
        output_prefix = str(args.output_prefix).strip()
        if not output_prefix:
            raise ValueError("output_prefix must be a non-empty string")

        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"output directory not found: {output_dir}")
        if not output_dir.is_dir():
            raise ValueError(f"output_dir must be a directory: {output_dir}")

        weights = _parse_comma_floats(args.weights, "weights")
        log_var_scales = _parse_comma_floats(args.log_var_scales, "log_var_scales")

        simulation_config = SimulationConfig(
            n_rows=args.n_rows,
            seed=args.seed,
            weights=weights,
            log_var_scales=log_var_scales,
            variance_link=args.variance_link,
            theta=args.theta,
            link_eps=args.link_eps,
            link_shift=args.link_shift,
            af_clip_min=args.af_clip_min,
            af_model=args.af_model,
            af_uniform_low=args.af_uniform_low,
            af_uniform_high=args.af_uniform_high,
            af_beta_a=args.af_beta_a,
            af_beta_b=args.af_beta_b,
            se_model=args.se_model,
            se_constant=args.se_constant,
            sample_size=args.sample_size,
            se_scale=args.se_scale,
        )

        log.info("simulate: running pipeline")
        artifacts = run_simulation_pipeline(config=simulation_config, log=log)

        log.info("simulate: writing outputs")
        truth_path = output_dir / f"{output_prefix}.truth.tsv"
        observed_path = output_dir / f"{output_prefix}.observed.tsv"
        meta_path = output_dir / f"{output_prefix}.meta.tsv"
        artifacts.truth.write_csv(truth_path, separator="\t")
        artifacts.observed.write_csv(observed_path, separator="\t")
        artifacts.metadata.write_csv(meta_path, separator="\t")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        log.error(str(exc))
        return 2
    except RuntimeError as exc:
        log.error(str(exc))
        return 1


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
