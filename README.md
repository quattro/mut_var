# mut_var

[![PyPI - Version](https://img.shields.io/pypi/v/mut-var.svg)](https://pypi.org/project/mut-var)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mut-var.svg)](https://pypi.org/project/mut-var)
[![CI](https://github.com/quattro/mut-var/actions/workflows/ci.yml/badge.svg)](https://github.com/quattro/mut-var/actions/workflows/ci.yml)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install mut-var
```

## Validation Contract

`mutvar` validates inputs before running any fitting routine.

- Required columns:
  - `effect_allele_frequency` (or `--af-col`)
  - `beta` (or `--beta-col`)
  - `standard_error` (or `--se-col`)
- Domain rules:
  - `effect_allele_frequency` must be within `[0, 1]`
  - `standard_error` must be strictly greater than `0`
- MAF grid rules:
  - `0 < lowest < highest <= 0.5`
  - `num_breaks >= 2`
- Failure contract:
  - Invalid CLI arguments or invalid input data return a deterministic non-zero exit code.
  - Errors are emitted as actionable stderr messages.

## Architecture Contract

Canonical numerics entrypoints live under `mut_var.numerics`:

- `mut_var.numerics.fit_baseline`
- `mut_var.numerics.fit_refit_grid`
- `mut_var.numerics.run_inference_pipeline`

Core numerics APIs return `mut_var.Solution` with explicit `result` status and diagnostics in
`stats`/`state`.

`mut_var.cli` is the imperative shell (argument parsing, boundary validation, IO orchestration); it
is not the numerics implementation module.

## Curve Workflow Contract

Curve fitting is split into:

- Pure numerics: `mut_var.numerics.curve_fit`
- Optional plotting adapter: `mut_var.plotting.curve_plots`
- Orchestration workflow: `mut_var.curve.run_curve_workflow`

Behavior guarantees:

- Fit-only mode (`generate_plots=False` / `mutvar-curve --fit-only`) does not import plotting
  adapters and produces no PNG side effects.
- Plotting mode consumes precomputed fit outputs and only adds PNG side effects; fitted coefficients
  remain unchanged.

## Benchmark Procedure

Run the reproducible runtime benchmark with:

```console
python benchmarks/infer_runtime.py --config benchmarks/config/runtime_baseline.json --output benchmarks/results/latest.json
```

Output report schema guarantees:

- `compile`: one-time compile-focused timing block.
- `steady_state`: repeated-run timing block for runtime behavior.
- `comparison`: includes `improvement_percent`, threshold, and pass/fail result.

Interpretation rules:

- Treat compile and steady-state metrics independently; do not combine them.
- The acceptance gate requires `comparison.improvement_percent >= 20.0`.
- Benchmark representativeness review is recorded in
  `docs/reviews/benchmark-representativeness.md`.

## CI Gates

Required checks (local and CI must match):

- `ruff check src/mut_var tests`
- `mypy src/mut_var tests`
- `pytest -p no:capture`
- `python benchmarks/infer_runtime.py --config benchmarks/config/runtime_baseline.json --output benchmarks/results/latest.json`

Algorithm-scope constraint:

- Changes must remain targeted to validation/orchestration/performance hardening.
- Wholesale objective/model redesign is explicitly out of scope and requires separate design review.

## License

`mut-var` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
