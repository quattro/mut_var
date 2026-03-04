#!/usr/bin/env python3
"""Run mutvar infer + curve on a simulator output file.

This script only calls the public CLI entrypoint and writes outputs under sims/results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run mutvar infer/curve on observed simulation output.")
    parser.add_argument("--run-id", required=True, help="Run identifier (expects sims/results/<run_id>.observed.tsv)")
    parser.add_argument("--results-dir", default="sims/results", help="Results directory")
    parser.add_argument("--fit-only", action="store_true", default=True, help="Run mutvar curve in fit-only mode")
    return parser.parse_args()


def run_command(command):
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    observed_path = results_dir / f"{args.run_id}.observed.tsv"
    infer_path = results_dir / f"{args.run_id}.infer.tsv"
    curve_path = results_dir / f"{args.run_id}.curve.tsv"
    log_path = results_dir / f"{args.run_id}.mutvar.json"

    if not observed_path.exists():
        raise FileNotFoundError(f"missing observed file: {observed_path}")

    infer_cmd = [
        "mutvar",
        "infer",
        str(observed_path),
        "-o",
        str(infer_path),
    ]
    run_command(infer_cmd)

    curve_cmd = [
        "mutvar",
        "curve",
        str(infer_path),
        "-o",
        str(curve_path),
    ]
    if args.fit_only:
        curve_cmd.append("--fit-only")
    run_command(curve_cmd)

    run_manifest = {
        "run_id": args.run_id,
        "commands": [infer_cmd, curve_cmd],
        "files": {
            "observed": str(observed_path),
            "infer": str(infer_path),
            "curve": str(curve_path),
        },
    }
    log_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"wrote: {log_path}")


if __name__ == "__main__":
    main()
