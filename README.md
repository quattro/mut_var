# mut_var

[![PyPI - Version](https://img.shields.io/pypi/v/mut-var.svg)](https://pypi.org/project/mut-var)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mut-var.svg)](https://pypi.org/project/mut-var)
[![CI](https://github.com/quattro/mut_var/actions/workflows/ci.yml/badge.svg)](https://github.com/quattro/mut_var/actions/workflows/ci.yml)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install mut-var
```

## Development and documentation

```bash
uv sync --locked --extra dev --extra docs
uv run --frozen pytest -p no:capture
uv run --frozen zensical build --strict
uv build
```

See [Contributing](docs/site/contributing.md) for lint/type checks, local docs
preview, native wheel builds, and release configuration.

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
  - `--lowest` defaults to the minimum observed MAF in the input data when not explicitly set.
- Failure contract:
  - Invalid CLI arguments or invalid input data return a deterministic non-zero exit code.
  - Errors are emitted as actionable stderr messages.

## Inference Pipeline

Canonical infer pipeline:

- CLI: `mutvar infer <sumstats.tsv> [options]`
- API (pipeline): `mut_var.run_inference_pipeline(path, ...)` -> long-format `polars.DataFrame`

Canonical simulation pipeline:

- CLI: `mutvar simulate --output-prefix demo --output-dir out [options]`
- API (pipeline): `mut_var.run_simulation_pipeline(config=...)` -> `SimulationArtifacts`
- Output artifacts:
  - `out/demo.truth.tsv`
  - `out/demo.observed.tsv`
  - `out/demo.meta.tsv`
- Inference handoff:
  - `mutvar infer out/demo.observed.tsv -o out/demo.infer.tsv`

Canonical curve pipeline:

- CLI: `mutvar curve <mutvar-output.tsv> [--method sigmoid|isotonic|mono_spline|invlog_linear|invlog_logit|invlog_sigmoid] [--fit-only]`
- API: `mut_var.run_curve_pipeline(input_path, generate_plots=..., method=...)` -> method-neutral parameter `polars.DataFrame`

Python inference example:

```python
from mut_var import InferenceConfig, run_inference_pipeline

result_df = run_inference_pipeline(
    "data/bmi_exwas.tsv.gz",
    config=InferenceConfig(num_clusters=30),
)
```

## Simulation Workflow

`mutvar simulate` generates three aligned artifacts for downstream testing and reproducible demos:

- `*.truth.tsv`: latent component assignments and true effects
- `*.observed.tsv`: summary-stat style inputs accepted by `mutvar infer`
- `*.meta.tsv`: run metadata plus AF-decile diagnostics

Example CLI run:

```console
mutvar simulate \
  --output-dir out \
  --output-prefix demo \
  --n-rows 10000 \
  --seed 0 \
  --weights 0.95,0.05 \
  --log-var-scales -8.0,-5.5
```

Produced files:

- `out/demo.truth.tsv`
- `out/demo.observed.tsv`
- `out/demo.meta.tsv`

Artifact column contracts:

- `truth`: `row_id`, `component`, `beta_true`, `sigma2`, `effect_allele_frequency`
- `observed`: `row_id`, `effect_allele_frequency`, `beta`, `standard_error`
- `meta`: `seed`, `n_rows`, `num_components`, `variance_link`, `theta`, `af_decile`, `empirical_var_beta_true`, `empirical_mean_sigma2`

Inference handoff:

```console
mutvar infer out/demo.observed.tsv -o out/demo.infer.tsv
```

### Simulation Comparison Plots (Inferred vs Simulated Proportions)

You can compare inferred mixture proportions against simulated component proportions by combining:

- `out/demo.truth.tsv` (from `mutvar simulate`)
- `out/demo.infer.tsv` (from `mutvar infer out/demo.observed.tsv ...`)

Recommended outputs for this analysis workflow:

- `component_proportions.tsv` — per-threshold comparison table (`maf`, `component`, simulated/inferred proportions)
- `component_proportions_vs_maf.png` — semilog MAF plot with simulated vs inferred proportion overlays for each component
- `component_proportions_scatter.png` — inferred vs simulated proportion scatter plot per component (use `[0, 1]` axes for calibration-style reading)

Comparison notes:

- Inferred component labels are not guaranteed to match simulated component IDs.
- For component-wise comparisons, match inferred components to simulated components using a stable rule (for example nearest `log(var0)` / variance scale) before summing weights.
- Renormalize inferred weights over the matched non-null component mass when the null component (`pi0` at `var0 == 0`) is excluded from the comparison.

This is an analysis/plotting workflow over existing `simulate` and `infer` outputs; it does not change numerics contracts.

Reusable script:

```console
mkdir -p /tmp/mplcfg && MPLCONFIGDIR=/tmp/mplcfg python scripts/plot_component_proportions.py \
  --truth out/demo.truth.tsv \
  --infer out/demo.infer.tsv \
  --output-dir out/sim_compare_plots \
  --output-prefix component_proportions \
  --axis-min 0 \
  --axis-max 1 \
  --maf-min 1e-3
```

Script outputs:

- `out/sim_compare_plots/component_proportions.tsv`
- `out/sim_compare_plots/component_proportions_summary.tsv`
- `out/sim_compare_plots/component_proportions_vs_maf.png`
- `out/sim_compare_plots/component_proportions_scatter.png`

To plot all inferred non-null components directly (without collapsing to simulated component IDs), use:

```console
mkdir -p /tmp/mplcfg && MPLCONFIGDIR=/tmp/mplcfg python scripts/plot_component_proportions.py \
  --component-mode inferred \
  --infer out/demo.infer.tsv \
  --output-dir out/sim_compare_plots \
  --output-prefix inferred_components \
  --axis-min 0 \
  --axis-max 1 \
  --maf-min 1e-3
```

This produces:

- `out/sim_compare_plots/inferred_components.tsv`
- `out/sim_compare_plots/inferred_components_summary.tsv`
- `out/sim_compare_plots/inferred_components_vs_maf.png`

Python API example:

```python
from mut_var import run_simulation_pipeline, SimulationConfig

artifacts = run_simulation_pipeline(
    config=SimulationConfig(
        n_rows=1000,
        seed=0,
        weights=(0.95, 0.05),
        log_var_scales=(-8.0, -5.5),
    )
)

observed_df = artifacts.observed
```

## Architecture Contract

Canonical numerics entrypoints live under `mut_var.numerics`:

- `mut_var.numerics.prepare_fit_state`
- `mut_var.numerics.fit_baseline`
- `mut_var.numerics.fit_refit_step`
- `mut_var.numerics.fit_curve_model`
- `mut_var.numerics.evaluate_curve_fit`
- `mut_var.numerics.simulate_mixture_data`

Pipeline APIs return dataframe outputs (or dataframe artifact containers for simulation) for downstream consumption and file IO.
Core numerics APIs return `mut_var.types.Solution` with explicit `result` status and diagnostics in `stats`/`state`.

`mut_var.cli` is the imperative shell (argument parsing, boundary validation, IO orchestration); it
is not the numerics implementation module.

## Numerics Failure Status Catalog

`mut_var.types.RESULTS` explicitly encodes pipeline outcomes:

- `successful`
- `invalid_input`
- `empty_subset`
- `nonfinite_objective`
- `max_steps_reached`

`mut_var.types.Solution` carries:

- `value`: result payload (if available)
- `result`: status code from `RESULTS`
- `stats`: diagnostics
- `state`: optional solver state

Empty-subset and non-finite paths are explicit and diagnosable.

### Failure Handling Examples

```python
from mut_var import run_inference_pipeline

result_df = run_inference_pipeline("data/bmi_exwas.tsv.gz")
print(result_df.head())
```

## Curve Pipeline Contract

Curve fitting is split into:

- Pure numerics: `mut_var.numerics.fit_curve_model` + `mut_var.numerics.evaluate_curve_fit`
- Optional plotting adapter: `mut_var.plotting.curve_plots`
- Orchestration pipeline: `mut_var.run_curve_pipeline`
- Method selection: `sigmoid` (default), `isotonic`, `mono_spline`, `invlog_linear`, `invlog_logit`, or `invlog_sigmoid`

Behavior guarantees:

- Fit-only mode (`generate_plots=False` / `mutvar curve --fit-only`) does not import plotting
  adapters and produces no PNG side effects.
- Plotting mode consumes precomputed fit outputs and only adds PNG side effects; fitted outputs
  remain unchanged.
- Curve output is method-neutral and records `var0`, `method`, `param_name`, and `param_value`.

The inverse-log methods fit each component independently using the same supplied MAF points
and equal weight per observation (including repeated thresholds):

- `invlog_linear`: $f(t)=a+b/\log(1/t)$, fitted by linear least squares without clipping predictions.
- `invlog_logit`: $f(t)=\operatorname{expit}(a+b/\log(1/t))$, fitted by nonlinear least squares
  on the probability scale. Exact zero/one observations are used directly. Only the initial
  mean is clipped to $[10^{-9},1-10^{-9}]$ to initialize a finite logit.

Both report `a` and `b` parameter rows and accept finite MAF values in $[0,1)$, with at least
two distinct MAF points required for fitting. Evaluation at zero returns exactly `a` or
`expit(a)`, respectively; no epsilon substitutes for zero. Component limits are not
renormalized and need not sum to one. The initial mixture fits are unchanged.

```bash
mutvar curve results.tsv --method invlog_logit --fit-only
```

Python callers select either method with `fit_curve_model(..., method="invlog_linear")`
or `run_curve_pipeline(..., method="invlog_logit", generate_plots=False)`.

`invlog_sigmoid` adds a four-parameter alternative:
$f(t)=L+(U-L)\operatorname{expit}(a+b/\log(1/t))$ with $0\le L\le U\le1$.
It combines a logistic transition with an inverse-log approach to zero frequency,
and reports `lower`, `upper`, `a`, and `b`. Its exact zero-frequency value is
$L+(U-L)\operatorname{expit}(a)$, **not** $L$. At least four distinct MAF points
are required. Fitting uses probability-scale least squares with four deterministic
starts and a common residual scaling factor to handle very small component weights.
Exact zero/one observations are accepted; constant observations return a flat fit
with `lower == upper` and `a == b == 0`. Flat or incompletely observed transitions
may not identify all four parameters, even when predictions fit well.

```bash
mutvar curve results.tsv --method invlog_sigmoid --fit-only
```

## Performance Profiling Status

Performance profiling is currently out of scope for supported workflows in this repository.

- CI and release-readiness gates do not require benchmark execution.
- The canonical quality gates are linting, type checking, and tests.
- Historical benchmark review documents are retained for recordkeeping only.

## CI Gates

Required checks (local and CI must match):

- `uv run --frozen ruff check src tests scripts setup.py`
- `uv run --frozen ruff format --check src tests scripts setup.py`
- `uv run --frozen ty check src tests scripts`
- `uv run --frozen pytest -p no:capture`

Release-readiness quick check:

- `uv run --frozen ruff check src tests scripts setup.py`
- `uv run --frozen ruff format --check src tests scripts setup.py`
- `uv run --frozen ty check src tests scripts`
- `uv run --frozen pytest -p no:capture`

## License

`mut-var` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
