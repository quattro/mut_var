# mut_var

Last verified: 2026-04-15

## Purpose
Provide reproducible mutation-variance inference pipelines with explicit failure states for both CLI and Python callers.

## Contracts
- **Exposes**:
  - CLI entrypoint: `mutvar` (`infer`, `curve`, `simulate`)
  - Package-root pipeline APIs: `run_inference_pipeline`, `run_curve_pipeline`, `run_simulation_pipeline`
  - Numerics APIs: `mut_var.numerics.fit_baseline`, `mut_var.numerics.fit_curve`, `mut_var.numerics.fit_refit_grid`, `mut_var.numerics.simulate_mixture_data`
  - Contract types: `mut_var.types.RESULTS`, `mut_var.types.Solution`, `mut_var.io.InferenceArrays`, `mut_var.types.InferenceConfig`, `mut_var.numerics.SimulationArrays`, `mut_var.numerics.SimulationNumericsConfig`, `mut_var.SimulationPipelineConfig`, `mut_var.SimulationArtifacts`
- **Guarantees**:
  - Boundary validation happens at ingress before numerics execute.
  - Pipeline/orchestration APIs accept validated dataframe or array-like inputs and return dataframe outputs for downstream writing/processing.
  - Numerics APIs return `Solution` objects and use `Solution.result` as the canonical status signal.
  - Contracts above numerics do not expose `Solution`; they normalize outputs to tabular/dataframe forms.
  - Simulation pipeline APIs return dataframe artifacts (`truth`, `observed`, `metadata`) and keep file writes in CLI/orchestration shells.
  - Baseline/refit numerics run through Optimistix-based optimization with full-batch objective updates only.
  - Numerics objective wrappers use `equinox.filter_jit` for JIT staging.
  - Orchestration/input errors use built-in exception types (`ValueError`, `FileNotFoundError`, `RuntimeError`) instead of custom error hierarchies.
  - High-level workflow paths emit step-level progress logs (load/validate/run/prepare/write) through logging, not ad-hoc prints.
  - Curve fit-only mode (`generate_plots=False` / `mutvar curve --fit-only`) performs no plotting side effects.
- **Expects**:
  - Input data includes required AF/BETA/SE fields (or explicit column overrides).
  - Domain constraints hold (`effect_allele_frequency` in `[0,1]`, `standard_error > 0`).
  - Grid constraints hold (`0 < lowest < highest <= 0.5`, `num_breaks >= 2`).
  - Simulation config constraints hold (mixture weights/scales align, weights sum to `1`, AF/SE model parameters stay in documented domains).

## Dependencies
- **Uses**: `jax`, `equinox`, `optimistix`, `polars`, `jaxtyping`, `matplotlib` (plotting path).
- **Used by**: CLI users and Python integrations via package-root imports.
- **Boundary**:
  - `mut_var.cli` is imperative-shell orchestration; do not treat it as numerics API surface.
  - Canonical numerics implementations live under `src/mut_var/numerics`.
  - Numerics-specific contracts are documented in `src/mut_var/numerics/AGENTS.md`.
  - Prefer package-root imports and `mut_var.io` over reaching into deleted adapter internals.

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
- Canonical quality gates remain aligned between local and CI:
  - `ruff check src/mut_var tests`
  - `mypy src/mut_var tests`
  - `pytest -p no:capture`

## Commands
- `pip install -e .`
- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [--fit-only]`
- `mutvar simulate --output-prefix <prefix> [options]`
- `ruff check src/mut_var tests`
- `mypy src/mut_var tests`
- `pytest -p no:capture`

## Docstrings
- Use raw docstrings (`r"""..."""`) on public CLI/pipeline/numerics entrypoints.
- Use exact section labels: `**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`.
- Keep one empty line immediately after each section heading and one empty line between section blocks.
- Use LaTeX math notation in docstrings when helpful (`$...$` inline, `$$...$$` display blocks).
- Update docstrings in the same patch when signatures, status semantics, or side effects change.

## Docs Markdown
- Use MkDocs admonitions when they improve clarity in generated docs (for example: `!!! info`, `!!! note`, `!!! warning`, `!!! tip`).
- Keep admonition titles/content concise and technically actionable; avoid decorative callouts.

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
- `mutvar simulate` writes three files (`.truth.tsv`, `.observed.tsv`, `.meta.tsv`) and does not stream tabular output to stdout.
- Keep algorithm changes targeted; broad model redesign requires separate design review.
