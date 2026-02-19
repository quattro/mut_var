#! /usr/bin/env python
from __future__ import annotations

import argparse as ap
import sys

import polars as pl

from mut_var.contracts import RESULTS
from mut_var.curve import run_curve_workflow


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("data")
    argp.add_argument("--lowest", type=float, default=1e-10)
    argp.add_argument("--highest", type=float, default=1e-2)
    argp.add_argument("--num-breaks", type=int, default=10)
    argp.add_argument("--fit-only", action="store_true", default=False)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)
    solution = run_curve_workflow(args.data, generate_plots=not args.fit_only)
    if solution.result != RESULTS.successful:
        reason = None
        if isinstance(solution.stats, dict):
            reason = solution.stats.get("reason")
        if reason is None:
            reason = f"Curve workflow failed with status '{solution.result.value}'."
        print(reason, file=sys.stderr)
        return 2

    pl.DataFrame(solution.value["coefficients"]).write_csv(args.output, separator="\t")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
