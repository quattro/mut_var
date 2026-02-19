# mut_var

Last verified: 2026-02-18

## Purpose
Provide reproducible mutation-variance inference pipelines with explicit failure states for both CLI and Python callers.

## Contracts
- **Exposes**:
  - CLI entrypoint: `mutvar` (`infer`, `curve`)
  - Package-root pipeline APIs: `run_inference_pipeline`, `run_curve_pipeline`
  - Numerics APIs: `mut_var.numerics.fit_baseline`, `mut_var.numerics.fit_curve`, `mut_var.numerics.fit_refit_grid`
  - Contract types: `mut_var.contracts.RESULTS`, `mut_var.contracts.Solution`, `mut_var.numerics.InferenceArrays`, `mut_var.numerics.InferenceConfig`
- **Guarantees**:
  - Boundary validation happens at ingress before numerics execute.
  - Pipeline/orchestration APIs accept validated dataframe or array-like inputs and return dataframe outputs for downstream writing/processing.
  - Numerics APIs return `Solution` objects and use `Solution.result` as the canonical status signal.
  - Contracts above numerics do not expose `Solution`; they normalize outputs to tabular/dataframe forms.
  - Baseline/refit numerics run through Optimistix-based optimization with full-batch objective updates only.
  - Numerics objective wrappers use `equinox.filter_jit` for JIT staging.
  - Orchestration/input errors use built-in exception types (`ValueError`, `FileNotFoundError`, `RuntimeError`) instead of custom error hierarchies.
  - High-level workflow paths emit step-level progress logs (load/validate/run/prepare/write) through logging, not ad-hoc prints.
  - Curve fit-only mode (`generate_plots=False` / `mutvar curve --fit-only`) performs no plotting side effects.
- **Expects**:
  - Input data includes required AF/BETA/SE fields (or explicit column overrides).
  - Domain constraints hold (`effect_allele_frequency` in `[0,1]`, `standard_error > 0`).
  - Grid constraints hold (`0 < lowest < highest <= 0.5`, `num_breaks >= 2`).

## Dependencies
- **Uses**: `jax`, `equinox`, `optimistix`, `polars`, `jaxtyping`, `matplotlib` (plotting path).
- **Used by**: CLI users and Python integrations via package-root imports.
- **Boundary**:
  - `mut_var.cli` is imperative-shell orchestration; do not treat it as numerics API surface.
  - Canonical numerics implementations live under `src/mut_var/numerics`.
  - Numerics-specific contracts are documented in `src/mut_var/numerics/AGENTS.md`.
  - Prefer package-root imports over reaching into adapter internals.

## Key Decisions
- Public API is intentionally centralized in `src/mut_var/__init__.py` to keep import contracts stable.
- Status-bearing `Solution` objects are reserved for numerics-facing contracts.
- Numerics optimization is standardized on Optimistix with custom manifold descent modules; legacy native loop orchestration was removed.
- Full-batch-only optimization is the canonical contract; `batch_size` controls were removed from public inference/baseline configs and CLI.
- Contract enums/modules use Equinox primitives (`equinox.internal.Enumeration`, `equinox.Module`) instead of stdlib `Enum`/`dataclass`.
- Pipeline-facing APIs normalize successful outputs to `polars.DataFrame`.
- Plot generation is isolated from curve-fitting numerics so fit outputs remain unchanged by plotting.

## Invariants
- `RESULTS` status codes are explicit and stable (`successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, `max_steps_reached`).
- `Solution` carries `value`, `result`, and optional `stats`/`state`.
- Data-structure rule: data-only contracts use `NamedTuple`; behavior-bearing contracts use `equinox.Module`.
- `BaselineConfig` and `InferenceConfig` no longer include a `batch_size` field.
- Canonical release gates remain aligned between local and CI:
  - `ruff check src/mut_var tests`
  - `mypy src/mut_var tests`
  - `pytest -p no:capture`
  - `python benchmarks/infer_runtime.py --config benchmarks/config/runtime_baseline.json --output benchmarks/results/latest.json`

## Commands
- `pip install -e .`
- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [--fit-only]`
- `ruff check src/mut_var tests`
- `mypy src/mut_var tests`
- `pytest -p no:capture`

## Project Structure
- `src/mut_var/` - package API, CLI shell, pipelines, numerics, plotting adapters.
- `tests/` - behavior and contract regression tests.
- `benchmarks/` - reproducible runtime benchmark harness and configs.
- `docs/` - API surface, design plans, and review artifacts.
- `scripts/` - release-gate and maintenance helpers.

## Gotchas
- Importing internals from `mut_var.cli` is unsupported; use package-root APIs.
- Treat `Solution.result` (not presence of `value`) as the success signal for numerics APIs.
- `mutvar infer` no longer accepts `--batch-size`; numerics are full-batch by contract.
- Keep algorithm changes targeted; broad model redesign requires separate design review.
