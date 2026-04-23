# mut_var

Last verified: 2026-04-23

## Purpose
Provide reproducible mutation-variance inference pipelines with explicit failure states for both CLI and Python callers.

## Contracts
- **Exposes**:
  - CLI entrypoint: `mutvar` (`infer`, `curve`, `simulate`)
  - Package-root pipeline APIs: `run_inference_pipeline`, `run_curve_pipeline`, `run_simulation_pipeline`
  - Numerics APIs: `mut_var.numerics.prepare_fit_state`, `mut_var.numerics.fit_baseline`, `mut_var.numerics.fit_refit_step`, `mut_var.numerics.fit_curve_model`, `mut_var.numerics.evaluate_curve_fit`, `mut_var.numerics.simulate_mixture_data`
  - Contract types: `mut_var.types.RESULTS`, `mut_var.types.Solution`, `mut_var.pipelines.InferenceArrays`, `mut_var.InferenceConfig`, `mut_var.numerics.SimulationArrays`, `mut_var.SimulationConfig`, `mut_var.SimulationArtifacts`
- **Guarantees**:
  - Boundary validation happens at ingress before numerics execute.
  - Inference pipeline/orchestration APIs accept path-based ingress plus explicit column overrides and return dataframe outputs for downstream writing/processing.
  - Numerics APIs return `Solution` objects and use `Solution.result` as the canonical status signal.
  - Contracts above numerics do not expose `Solution`; they normalize outputs to tabular/dataframe forms.
  - Inference numerics prepare a shared likelihood matrix once, then reuse that state across baseline and refit.
  - Simulation pipeline APIs return dataframe artifacts (`truth`, `observed`, `metadata`) and keep file writes in CLI/orchestration shells.
  - Baseline/refit numerics use mix-SQP with full-batch objective updates only.
  - Curve numerics support `sigmoid`, `isotonic`, and `mono_spline` methods, and the curve pipeline returns method-neutral parameter rows.
  - Numerics hot path is Cython-compiled (`_core.pyx`) with BLAS acceleration.
  - Orchestration/input errors use built-in exception types (`ValueError`, `FileNotFoundError`, `RuntimeError`) instead of custom error hierarchies.
  - High-level workflow paths emit step-level progress logs (load/validate/run/prepare/write) through logging, not ad-hoc prints.
  - Curve fit-only mode (`generate_plots=False` / `mutvar curve --fit-only`) performs no plotting side effects.
  - `run_inference_pipeline(lowest=None)` auto-derives the MAF grid lower bound from the minimum observed MAF in the input data; explicit `lowest` overrides this.
- **Expects**:
  - Input data includes required AF/BETA/SE fields (or explicit column overrides).
  - Domain constraints hold (`effect_allele_frequency` in `[0,1]`, `standard_error > 0`).
  - Grid constraints hold (`0 < lowest < highest <= 0.5`, `num_breaks >= 2`).
  - Simulation config constraints hold (mixture weights/scales align, weights sum to `1`, AF/SE model parameters stay in documented domains).

## Dependencies
- **Uses**: `numpy`, `scipy`, `polars`, `matplotlib` (plotting path), `Cython` (build-time).
- **Used by**: CLI users and Python integrations via package-root imports.
- **Boundary**:
  - `mut_var.cli` is imperative-shell orchestration; do not treat it as numerics API surface.
  - Canonical numerics implementations live under `src/mut_var/numerics`.
  - Canonical pipeline/orchestration implementations live under `src/mut_var/pipelines`.
  - Numerics-specific contracts are documented in `src/mut_var/numerics/AGENTS.md`.
  - Prefer package-root imports for public pipeline APIs; use `mut_var.pipelines` for pipeline-specific contract types.

## Key Decisions
- Public API is intentionally centralized in `src/mut_var/__init__.py` to keep import contracts stable.
- Status-bearing `Solution` objects are reserved for numerics-facing contracts.
- Numerics optimization uses mix-SQP (Kim et al. 2020) via a Cython/NumPy stack; JAX/Equinox/Optimistix have been removed.
- Full-batch-only optimization is the canonical contract; `batch_size` controls were removed from public inference/baseline configs and CLI.
- Contract types use stdlib `enum.Enum` and `@dataclass(frozen=True)` instead of Equinox primitives.
- Pipeline-facing APIs normalize successful outputs to `polars.DataFrame`.
- Plot generation is isolated from curve-fitting numerics so fit outputs remain unchanged by plotting.

## Invariants
- `RESULTS` status codes are explicit and stable (`successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, `max_steps_reached`).
- `Solution` carries `value`, `result`, and optional `stats`/`state`.
- Data-structure rule: data-only contracts use `NamedTuple`; behavior-bearing contracts use `@dataclass(frozen=True)`.
- `InferenceConfig` no longer includes a `batch_size` field.
- Canonical quality gates remain aligned between local and CI:
  - `ruff check src/mut_var tests`
  - `mypy src/mut_var tests`
  - `pytest -p no:capture`

## Commands
- `pip install -e .`
- `mutvar infer <sumstats.tsv> [options]`
- `mutvar curve <mutvar-output.tsv> [--method sigmoid|isotonic|mono_spline] [--fit-only]`
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
- `src/mut_var/` - package API, CLI shell, pipelines, numerics, plotting, and ingress validation helpers.
- `src/mut_var/pipelines/` - pipeline-facing configs and orchestration entrypoints.
- `tests/` - behavior and contract regression tests.
- `benchmarks/` - reproducible runtime benchmark harness and configs.
- `docs/` - API surface, design plans, and review artifacts.
- `scripts/` - release-gate and maintenance helpers.

## Gotchas
- Importing internals from `mut_var.cli` is unsupported; use package-root APIs.
- Public workflow config types live at package root (`mut_var.InferenceConfig`, `mut_var.SimulationConfig`); `InferenceArrays` remains under `mut_var.pipelines`.
- Treat `Solution.result` (not presence of `value`) as the success signal for numerics APIs.
- `mutvar infer` no longer accepts `--batch-size`; numerics are full-batch by contract.
- `mutvar simulate` writes three files (`.truth.tsv`, `.observed.tsv`, `.meta.tsv`) and does not stream tabular output to stdout.
- Keep algorithm changes targeted; broad model redesign requires separate design review.
