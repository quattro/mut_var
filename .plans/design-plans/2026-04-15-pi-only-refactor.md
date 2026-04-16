# Pi-Only Inference: Fix mu/var, Riemannian pi solver, pipelines/ reorganization Design

## Status
Approved for Implementation

## Handoff Decision
- Current decision: approved
- Ready for implementation: yes
- Blocking items: none

## Metadata
- Date: 2026-04-15
- Slug: pi-only-refactor
- Artifact Directory: `.plans/design-plans/artifacts/2026-04-15-pi-only-refactor`

## Summary

This refactor restructures the mut-var inference codebase to simplify numerics and reorganize module responsibilities. Currently, the system optimizes three mixture parameters (means `mu_k`, variances `var_k`, and weights `pi`) via Riemannian gradient descent on a product manifold, but in practice `mu_k`/`var_k` remain fixed at their log-spaced grid initialization and dead gradient-computation code clutters the numerics layer. The codebase's I/O, pipeline orchestration, and numerical concerns are scattered across multiple modules in a way that obscures the data flow.

The refactor makes a clear architectural shift: `mu_k`/`var_k` are permanently fixed at initialization, `pi` becomes the sole optimization variable on the probability simplex, and the likelihood matrix is pre-computed once before the solver begins. In parallel, the module layout is reorganized to consolidate all ingress logic into `io.py`, unify type contracts in `types.py`, and introduce a `pipelines/` subdirectory for orchestration modules — bringing the codebase into alignment with patterns proven in the mixsqp branch. The existing Riemannian solver and simplex geometry for `pi` are preserved; only the optimization variable set and module organization change.

## Problem Statement
The main branch currently infers all three mixture parameters — means (`mu_k`), variances (`var_k`), and weights (`pi`) — via Riemannian gradient descent on a product manifold. However, `mu_k`/`var_k` are initialized on a fixed log-spaced grid and the proposal step in practice already leaves them unchanged. The dead parameter-update code (normal-distribution manifold maps, full `Params` pytree differentiation) adds complexity without benefit. Separately, the codebase's pipeline, ingress, and numerics responsibilities are scattered across `infer.py`, `adapters/tabular.py`, and `contracts.py` in a way that makes the layers harder to follow than necessary. The mixsqp branch demonstrates a cleaner organization pattern (consolidated `io.py`, `pipelines/` subdirectory, `types.py`) that main should adopt.

## Definition of Done
1. **Numerics simplified**: `baseline.py` + `refit.py` merged into `numerics/mixture_fit.py`. Likelihood matrix pre-computed once (fixed `mu_k`/`var_k`); Riemannian/Optimistix solver operates on `pi` only. Dead code (`_exponential_map_normal`, `_riemannian_step`, all mu/var gradient paths) removed.
2. **Layout reorganized**: `pipelines/` subdirectory added (`inference.py`, `curve.py`, `simulation.py`, `artifacts.py`). `contracts.py` → `types.py`. `adapters/` deleted. `io.py` consolidates all ingress (read + validate + convert + maf masks + payload→dataframe).
3. **All three subcommands pass existing tests**: `mutvar infer`, `mutvar curve`, `mutvar simulate` all pass `pytest`, `ruff check`, and `mypy` under the new structure.
4. **Public API updated consistently**: `__init__.py` exports the same symbols. `run_inference_pipeline` signature changes from `df: pl.DataFrame` to `path: str` (intentional — avoids holding dataframe and JAX arrays in memory simultaneously). CLI: four solver-internal flags removed (`--step-size`, `--penalty`, `--seed`, `--maf-threshold`); all other flags retained.

## Goals and Non-Goals
### Goals
- Fix `mu_k`/`var_k` at initialization; sole optimization variable is `pi` on the probability simplex.
- Pre-compute the full likelihood matrix once per inference run (moved before the Optimistix loop).
- Merge baseline and refit numerics into a single `numerics/mixture_fit.py` module.
- Consolidate all ingress concerns into `io.py` following the mixsqp branch's `load_inference_arrays` pattern.
- Introduce `pipelines/` subdirectory matching the mixsqp branch's layer structure.
- Rename `contracts.py` to `types.py` and unify config types there.
- Delete `adapters/` directory.

### Non-Goals
- Porting the mix-SQP optimizer from the mixsqp branch (the Riemannian/Optimistix solver is kept).
- Changing the curve-fitting or simulation algorithms.
- Altering CLI flag names or pipeline output schemas.
- Redesigning the refit ordering penalty or MAF-grid logic.

## Existing Patterns

**Functional core / imperative shell** is the established pattern. Pure numerical kernels live under `src/mut_var/numerics/`; orchestration, I/O, and logging live in the pipeline and CLI layers.

**`Solution`/`RESULTS` for numerics status** is the project-wide convention. Every numerics entrypoint returns a `Solution` wrapping a `RESULTS` status; callers branch on `solution.result`, never on `solution.value` directly.

**Riemannian simplex optimization** via `MutVarSolver` (Optimistix + `BacktrackingArmijo`) and `simplex_tangent_direction`/`exponential_map_simplex` is the established solver pattern for `pi`. This design preserves it in `_optimistix_solver.py` and `_solver_utils.py` unchanged.

**Equinox primitives for shared contracts** (`eqxi.Enumeration` for `RESULTS`, `eqx.Module` for `Solution`) is the current convention. This design keeps it in `types.py`.

**mixsqp branch as the organizational template.** The `pipelines/` subpackage layout, consolidated `io.py` (read + validate + convert in one module), and unified `types.py` for shared contracts are all derived from the mixsqp branch and are the target organization for main.

## Model Acquisition Path
- Path: `provided-model`
- Why this path: The mixture model structure (fixed component grid, pi-only inference) is already established in the existing codebase. This refactor restructures and simplifies the implementation without changing the statistical model.
- User selection confirmation: confirmed — user explicitly said to keep the Riemannian solver, only fix mu/var and reorganize.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: n/a
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `src/mut_var/infer.py` | in-repo source | Current inference ingress/orchestration contract; preserves public API while boundaries move. | high |
| SRC-2 | `src/mut_var/numerics/baseline.py` | in-repo source | Baseline numerics source for pi-only simplification and fixed-grid behavior. | high |
| SRC-3 | `src/mut_var/numerics/refit.py` | in-repo source | Refit numerics source for pi-only simplification and MAF-subset behavior. | high |
| SRC-4 | `src/mut_var/numerics/_solver_utils.py` | in-repo source | Simplex geometry primitives used by the existing Optimistix solver. | high |
| SRC-5 | `src/mut_var/numerics/_optimistix_solver.py` | in-repo source | Existing Optimistix solver wrapper and line-search integration. | high |
| SRC-6 | `src/mut_var/simulate.py` | in-repo source | Simulation pipeline contract and output artifacts used as a boundary reference. | high |

## Model Option Analysis (Required When `suggested-model`)
- Not applicable. The selected path is `provided-model`, not `suggested-model`.
- No alternative model family is being introduced; the existing fixed-grid, pi-only inference model is retained and only the module boundaries are reorganized.

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Not applicable. This phase is not importing an external repository or branch into the current project.
- The only continuity requirement is public-import stability while `types.py` and `io.py` absorb the existing contracts and adapter logic.

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- `contracts.py` currently contains `RESULTS` and `Solution`.
- `adapters/tabular.py` currently contains `to_inference_arrays`, `build_maf_masks`, and `payload_to_long_dataframe`.
- `infer.py` currently defines `InferenceArrays` and `InferenceConfig` and lazily imports adapter helpers.
- `simulate.py` currently defines `SimulationPipelineConfig`.
- `numerics/__init__.py` currently re-exports `InferenceArrays` and `InferenceConfig` from `mut_var.infer`.
- These findings support the phase-1 boundary move into `types.py` and `io.py` without algorithm changes.

## External Research Findings (When Triggered)
- Not triggered for this plan. No external paper or web lookup was needed because the refactor uses the existing in-repo model and solver contracts.

## Mathematical Sanity Checks
- The inference model remains a fixed-component mixture with `pi` constrained to the probability simplex.
- Holding `mu_k` and `var_k` fixed at the log-spaced grid preserves the likelihood family and removes only dead optimization degrees of freedom.
- Precomputing the likelihood matrix once is equivalent to recomputing it inside the solver because the component grid and input arrays are constant within a single inference run.
- The solver must continue to preserve `sum(pi) = 1` and non-negativity through the existing simplex geometry.

## Solver Translation Feasibility
- Feasible with the current architecture: `MutVarSolver`, `BacktrackingArmijo`, `simplex_tangent_direction`, and `exponential_map_simplex` already operate on the simplex and can accept `pi` as the sole optimization variable.
- The solver closure can capture the fixed likelihood matrix and fixed component parameters, leaving only `pi` as traced state.
- No new manifold or optimizer family is required for this phase.

## Data Conversion and Copy Strategy
- Ingress remains host-side: read TSV into Polars, validate columns and domains, then convert to array inputs at the boundary.
- `to_inference_arrays` currently converts validated Polars columns with `to_jax()` and `jnp.asarray(...)`; phase 1 keeps that behavior so numerics receive array-backed inputs without carrying the dataframe forward.
- The long-format payload is materialized only at egress, and raw source dataframes are not retained inside numerics.
- No explicit zero-copy guarantee is made; the contract is a single boundary conversion with no dataframe-to-solver leakage.

## Solver Strategy Decision
- User preference: keep the existing Riemannian/Optimistix solver
- Chosen strategy: restrict the Optimistix optimization variable to `pi` only; pre-compute the likelihood matrix once before entering the solver loop
- Why this strategy: the Riemannian simplex geometry for `pi` is already correct and well-tested; removing `mu_k`/`var_k` from the optimization variable eliminates dead gradient computation and removes `_exponential_map_normal`/`_riemannian_step` dead code

## Layer Contracts

### Ingress (`io.py`)
- Contract: accepts file path strings or validated `pl.DataFrame` objects; returns `InferenceArrays` (numpy float64 arrays) after validation + conversion.
- Rejection rules: missing columns, non-numeric/non-finite values, AF outside `[0,1]`, SE ≤ 0, and invalid MAF grid bounds all raise `ValueError` or `FileNotFoundError` before any numerics execute.
- New entrypoint `load_inference_arrays(path, ...) -> InferenceArrays` combines `read_sumstats` + validation + `to_inference_arrays` in a single call.
- `InferenceArrays(af, beta_hat, s2)` lives in `io.py`; `build_maf_masks` and `payload_to_long_dataframe` also live here.

### Numerics (`numerics/mixture_fit.py`)
- Contract: accepts array-like inputs (numpy-compatible), converts to JAX internally; returns `Solution` objects with explicit `RESULTS` status codes.
- No I/O, logging, or dataframe operations.
- Public surface:
  - `prepare_fit_state(beta_hat, s2, config: InferenceConfig) -> Solution` — validates arrays, builds log-spaced `mu_k`/`var_k` grid, pre-computes `(n, K)` likelihood matrix, initializes `pi` uniformly.
  - `fit_baseline(state: FitState, config: InferenceConfig, verbose=False) -> Solution` — Optimistix minimizes `f(pi) = -log-likelihood(pi, L) + dirichlet_penalty(pi)` over the simplex; `L` and `mu_k`/`var_k` are fixed constants in the closure.
  - `fit_refit_step(L_sub, prev_params: Params, config: InferenceConfig, verbose=False) -> Solution` — same pi-only Optimistix on a MAF-subset likelihood matrix plus the ordering penalty.
- `Params(pi, mu_k, var_k)` NamedTuple: `pi` is the optimization variable; `mu_k`/`var_k` are fixed metadata carried for output.
- `FitState(likelihood_matrix, initial_params)` NamedTuple: computed once by `prepare_fit_state`, reused by baseline and refit.
- Internal defaults: `_DEFAULT_STEP_SIZE = 0.01`, `_DEFAULT_PENALTY = 1.0`; these are not exposed in `InferenceConfig` or CLI.

### Pipeline (`pipelines/`)
- Contract: accepts file path strings for all three subcommands; orchestrates ingress → numerics → egress; returns `pl.DataFrame` outputs.
- `run_inference_pipeline(path: str, *, af_col, beta_col, se_col, lowest, highest, num_breaks, config, log) -> pl.DataFrame` — takes a path, calls `io.load_inference_arrays(path, ...)` internally. The intermediate dataframe is never retained in memory alongside the JAX arrays; only `InferenceArrays` (numpy) persists past ingress.
- `run_curve_pipeline(input_path, *, generate_plots, log) -> pl.DataFrame` — unchanged.
- `run_simulation_pipeline(config, log) -> SimulationArtifacts` — unchanged.
- `SimulationArtifacts` lives in `pipelines/simulation.py`.

### Egress (CLI, `cli.py`)
- Contract: wraps pipeline calls, handles file I/O, logging, and exit codes (`0`/`1`/`2`).
- `mutvar infer` removes four solver-internal flags: `--step-size` (`-r`), `--penalty`, `--seed` (`-s`), `--maf-threshold` (`-t`). All other flags (`-k`, `-m`, `-f`, grid flags, column overrides, `-o`, `-v`) are retained unchanged.

## Validation Strategy
- Boundary checks: all validation happens in `io.py` before numerics. `validate_maf_grid` checks grid bounds; `validate_required_columns`/`validate_numeric_columns`/`validate_sumstats_domain` check schema and domain.
- Shape/range/domain checks: `prepare_fit_state` validates that `beta_hat`/`s2` are 1D, equal-length, finite, and strictly positive before building the likelihood matrix.
- Failure semantics: `ValueError` for bad input/schema, `FileNotFoundError` for missing paths, `RuntimeError` for non-recoverable numerics failures. `RESULTS.invalid_input` / `RESULTS.empty_subset` map to `ValueError`; `RESULTS.nonfinite_objective` maps to `RuntimeError`.

## Testing and Verification Strategy
- Regression strategy: all existing tests in `tests/` must continue to pass after each phase. New tests cover the pi-only convergence property (fitted `pi` sums to 1, baseline objective improves monotonically on a synthetic dataset, refit produces pi vectors stochastically ordered across MAF thresholds).
- Verification commands: `ruff check src/mut_var tests`, `mypy src/mut_var tests`, `pytest -p no:capture`

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Type system and ingress consolidation

**Goal:** Centralize shared contracts in `types.py` and consolidate all ingress (file I/O, validation, array conversion, mask building, payload formatting) into `io.py`. Remove `contracts.py` and `adapters/`.

**Components:**
- `src/mut_var/types.py` (new) — contains `RESULTS` (`eqxi.Enumeration`), `Solution` (`eqx.Module`), `InferenceConfig(num_clusters, max_iter, tol, filter_threshold)`, and `SimulationPipelineConfig`. Replaces `contracts.py`. All callers updated.
- `src/mut_var/io.py` (expanded) — absorbs `to_inference_arrays`, `build_maf_masks`, and `payload_to_long_dataframe` from `adapters/tabular.py`; adds `load_inference_arrays(path, ...) -> InferenceArrays` convenience entrypoint; defines `InferenceArrays` NamedTuple. Returns numpy arrays (not JAX) from conversion functions.
- `src/mut_var/contracts.py` — deleted; all imports updated to `mut_var.types`.
- `src/mut_var/adapters/` — deleted; content moved to `io.py`.

**Dependencies:** None (first phase; no logic changes, only reorganization).

**Done when:** `ruff check`, `mypy`, and `pytest` all pass with no regressions. `InferenceArrays` importable from `mut_var.io`. `RESULTS`/`Solution`/`InferenceConfig` importable from `mut_var.types`.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Pi-only numerics (`mixture_fit.py`)

**Goal:** Merge `baseline.py` and `refit.py` into a single `numerics/mixture_fit.py` that optimizes only `pi` over the simplex. Pre-compute the likelihood matrix once in `prepare_fit_state`. Remove dead mu/var manifold code.

**Components:**
- `src/mut_var/numerics/mixture_fit.py` (new) — public surface: `FitState`, `Params`, `prepare_fit_state`, `fit_baseline`, `fit_refit_step`. Internally defines `_baseline_objective(pi, L, alpha)` and `_refit_objective(pi, L_sub, weights, alpha, baseline_pi)` with `_DEFAULT_STEP_SIZE = 0.01` and `_DEFAULT_PENALTY = 1.0`. The Optimistix solver (`MutVarSolver`) receives only `pi` (a JAX array) as `y0`; `L`, `mu_k`, and `var_k` are fixed closures. `pi` initialized uniformly (no random seed).
- `src/mut_var/numerics/baseline.py` and `src/mut_var/numerics/refit.py` — deleted. Dead code removed: `_exponential_map_normal`, `_riemannian_step`, `BaselineConfig`, `RefitConfig`, `baseline_objective_lse`.
- `src/mut_var/numerics/_optimistix_solver.py` and `_solver_utils.py` — unchanged.
- `src/mut_var/numerics/__init__.py` — updated exports.

**Dependencies:** Phase 1 (`types.py` must exist for `InferenceConfig`).

**Done when:** `prepare_fit_state` → `fit_baseline` → `fit_refit_step` round-trip produces fitted `pi` that sums to 1 and improves log-likelihood on a synthetic test case. All existing numerics tests (`tests/test_infer_opt.py`) pass or are updated to match the new API surface. `ruff check`, `mypy`, `pytest` pass.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Pipelines subdirectory

**Goal:** Introduce `pipelines/` subpackage; move orchestration modules (`infer.py`, `curve.py`, `simulate.py`) into it.

**Components:**
- `src/mut_var/pipelines/` (new subpackage) — `__init__.py` re-exports `run_inference_pipeline`, `run_curve_pipeline`, `run_simulation_pipeline`, `SimulationArtifacts`.
- `src/mut_var/pipelines/inference.py` (from `infer.py`) — `run_inference_pipeline(path: str, *, ...) -> pl.DataFrame` calls `io.load_inference_arrays(path, ...)` at the top; no `pl.DataFrame` is held in memory after ingress. Subsequent calls use `io.build_maf_masks`, `io.payload_to_long_dataframe`, and `mixture_fit.prepare_fit_state` / `fit_baseline` / `fit_refit_step`. `InferenceArrays` import from `mut_var.io`. `InferenceConfig` import from `mut_var.types`. No `to_baseline_config()`/`to_refit_config()` methods needed.
- `src/mut_var/pipelines/curve.py` (from `curve.py`) — content identical, import paths updated.
- `src/mut_var/pipelines/simulation.py` (from `simulate.py`) — content identical, import paths updated. `SimulationArtifacts` defined here.
- `src/mut_var/infer.py`, `curve.py`, `simulate.py` (top-level) — deleted.

**Dependencies:** Phase 1 (types and io) and Phase 2 (mixture_fit).

**Done when:** `run_inference_pipeline`, `run_curve_pipeline`, `run_simulation_pipeline` all importable from `mut_var.pipelines`. All pipeline tests (`tests/test_infer.py`, `tests/test_curve.py`, `tests/test_simulate.py`) pass. `ruff check`, `mypy`, `pytest` pass.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: CLI and public API cleanup

**Goal:** Update `cli.py` and `__init__.py` to use the new import paths; remove the four solver-internal CLI flags from `mutvar infer`; update AGENTS.md contracts.

**Components:**
- `src/mut_var/cli.py` — imports updated (`from mut_var.types import InferenceConfig`, `from mut_var.pipelines import ...`). `mutvar infer` subcommand: remove `--step-size` (`-r`), `--penalty`, `--seed` (`-s`), `--maf-threshold` (`-t`) flags. `run_infer_pipeline` handler constructs `InferenceConfig(num_clusters, max_iter, tol, filter_threshold)` with no step_size/penalty/seed fields.
- `src/mut_var/__init__.py` — imports updated from `pipelines/` instead of flat modules. Exported symbols unchanged: `run_inference_pipeline`, `run_curve_pipeline`, `run_simulation_pipeline`, `SimulationPipelineConfig`, `SimulationArtifacts`.
- `AGENTS.md` and `src/mut_var/numerics/AGENTS.md` — updated to reflect new module paths (`mut_var.types` for contracts, `mut_var.io` for `InferenceArrays`, `mut_var.pipelines` for pipeline APIs, `mut_var.numerics.mixture_fit` for numerics surface).

**Dependencies:** Phases 1–3.

**Done when:** Full quality gate passes: `ruff check src/mut_var tests`, `mypy src/mut_var tests`, `pytest -p no:capture`. CLI contract tests (`tests/test_cli_contracts.py`) pass. `mutvar infer --help` no longer shows removed flags.
<!-- END_PHASE_4 -->

## Simulation And Inference-Consistency Validation
- In scope: no
- Rationale: simulation numerics (`numerics/simulate.py`) are unchanged. The statistical model (fixed mu/var grid, pi-only weights) was already present in the simulation path; this refactor makes the inference path consistent with it.

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Uniform pi init may converge slower than Dirichlet random init for some datasets | Low | BacktrackingArmijo adapts step size; if observed in practice, can make init a private parameter | impl |
| R2 | Removing `--seed` / `--step-size` is a minor CLI breaking change for scripts that pass these flags | Low | Document in AGENTS.md; flags were solver internals with reasonable defaults | impl |

## Acceptance Criteria

### pi-only-refactor.AC1: Numerics simplified — pi is the sole optimization variable

- **pi-only-refactor.AC1.1 Success:** `prepare_fit_state` returns a `Solution` with `RESULTS.successful` and a `FitState` whose `likelihood_matrix` has shape `(n, K)` for valid inputs.
- **pi-only-refactor.AC1.2 Success:** `fit_baseline` converges on a synthetic dataset and returns `Params` where `sum(pi) == 1.0` (within float tolerance) and `mu_k`/`var_k` match the values set by `prepare_fit_state`.
- **pi-only-refactor.AC1.3 Success:** `fit_refit_step` returns `Params` with updated `pi` (sum 1) and unchanged `mu_k`/`var_k` relative to `prev_params`.
- **pi-only-refactor.AC1.4 Failure:** `prepare_fit_state` returns `RESULTS.invalid_input` when `s2` contains non-positive values.
- **pi-only-refactor.AC1.5 Failure:** `prepare_fit_state` returns `RESULTS.empty_subset` when input arrays are empty.
- **pi-only-refactor.AC1.6 Edge:** `fit_baseline` returns `RESULTS.max_steps_reached` (not an exception) when `max_iter=1`.

### pi-only-refactor.AC2: Module layout matches the target structure

- **pi-only-refactor.AC2.1 Success:** `from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig` succeeds.
- **pi-only-refactor.AC2.2 Success:** `from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays` succeeds.
- **pi-only-refactor.AC2.3 Success:** `from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline` succeeds.
- **pi-only-refactor.AC2.4 Success:** `from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params` succeeds.
- **pi-only-refactor.AC2.5 Failure:** `import mut_var.contracts` raises `ModuleNotFoundError` (module deleted).
- **pi-only-refactor.AC2.6 Failure:** `import mut_var.adapters` raises `ModuleNotFoundError` (directory deleted).
- **pi-only-refactor.AC2.7 Failure:** `import mut_var.numerics.baseline` raises `ModuleNotFoundError`.
- **pi-only-refactor.AC2.8 Failure:** `import mut_var.numerics.refit` raises `ModuleNotFoundError`.

### pi-only-refactor.AC3: Public API surface and CLI updated correctly

- **pi-only-refactor.AC3.1 Success:** `run_inference_pipeline(path, ...)` (path string) returns a long-format `pl.DataFrame` with columns `mu0`, `var0`, `maf`, `name`, `value` for a valid TSV path.
- **pi-only-refactor.AC3.2 Success:** `run_inference_pipeline` does not retain a `pl.DataFrame` in memory after `load_inference_arrays` returns — only `InferenceArrays` (numpy) persists through the numerics stage.
- **pi-only-refactor.AC3.3 Success:** `mutvar infer <path> -k 10 -m 50 -f 1e-6 --lowest 1e-5 --highest 1e-2 --num-breaks 5` exits with code 0.
- **pi-only-refactor.AC3.4 Success:** `mutvar curve <path>` and `mutvar simulate --output-prefix foo` exit with code 0 using valid inputs.
- **pi-only-refactor.AC3.5 Failure:** `mutvar infer <path> --step-size 0.1` exits with code 2 (unrecognized argument).
- **pi-only-refactor.AC3.6 Failure:** `mutvar infer <path> --seed 42` exits with code 2 (unrecognized argument).
- **pi-only-refactor.AC3.7 Failure:** `mutvar infer missing.tsv` exits with code 2 (file not found).
- **pi-only-refactor.AC3.8 Failure:** `run_inference_pipeline("missing.tsv", ...)` raises `FileNotFoundError`.

### pi-only-refactor.AC4: Quality gate passes end-to-end

- **pi-only-refactor.AC4.1 Success:** `ruff check src/mut_var tests` exits with code 0.
- **pi-only-refactor.AC4.2 Success:** `mypy src/mut_var tests` exits with code 0 (no type errors).
- **pi-only-refactor.AC4.3 Success:** `pytest -p no:capture` exits with code 0 (all tests pass).

## Glossary

- **Probability simplex**: The mathematical space of non-negative vectors that sum to 1, the natural domain for mixture weights (`pi`). Constrained optimization on this space requires Riemannian geometry.
- **Riemannian geometry / Riemannian gradient descent**: A framework for optimization on curved manifolds (like the simplex) rather than Euclidean space. Respects the manifold's intrinsic constraints (e.g., sum-to-1 for probabilities).
- **Exponential map / tangent direction**: In Riemannian geometry, the exponential map projects a direction from the tangent space onto the manifold; tangent directions are unconstrained updates that the map translates into valid moves on the simplex.
- **Product manifold**: A manifold formed by combining multiple simpler manifolds. The original code used a product of three manifolds (one for `mu_k`, one for `var_k`, one for the simplex for `pi`).
- **Likelihood matrix**: A 2D array of shape `(n, K)` where `n` is the number of genetic variants and `K` is the number of mixture components. Each entry is the likelihood of the variant under that component; pre-computed once using fixed `mu_k`/`var_k` values.
- **Optimistix**: A JAX-based optimization library used for the iterative solver. In this refactor, it receives only `pi` as the optimization variable; `mu_k`/`var_k` and the likelihood matrix are fixed constants captured in closures.
- **BacktrackingArmijo**: A line search strategy that adapts the step size during optimization to ensure descent. Used internally by `MutVarSolver`.
- **MAF (Minor Allele Frequency) grid**: A discretization of allele frequencies used to stratify variants for the refit step. Grid logic and the ordering penalty are preserved unchanged.
- **Dirichlet penalty**: A Bayesian regularization term that encourages the inferred `pi` weights toward a prior distribution. Kept as an internal hyperparameter (`_DEFAULT_PENALTY`).
- **Log-spaced grid**: A regularly-spaced set of values on a logarithmic scale, used to initialize the fixed `mu_k` and `var_k` component parameters.
- **InferenceArrays**: A NamedTuple containing validated numeric arrays (`af`, `beta_hat`, `s2`) in numpy format after ingress. Acts as the bridge between file I/O and JAX numerics.
- **FitState / Params**: NamedTuples that encapsulate numerics state. `FitState` holds the pre-computed likelihood matrix and initial parameters; `Params` carries the fitted `pi` and the fixed `mu_k`/`var_k` metadata.
- **Solution / RESULTS**: Equinox-based patterns for numerics returns: `Solution` is a module wrapping a value and a `RESULTS` status enum. All numerics functions return `Solution` objects; callers branch on the status code.
- **Closure**: A programming pattern where a nested function captures external variables (e.g., `L`, `mu_k`, `var_k`) in its scope. Used to pass fixed parameters to the Optimistix solver without including them in the optimization variable.
- **eqx.Module / eqxi.Enumeration**: Equinox (JAX library) primitives — `eqx.Module` for dataclass-like containers, `eqxi.Enumeration` for enums — used for `Solution` and `RESULTS` type contracts.
- **Simplex tangent direction / exponential map simplex**: The established Riemannian solver primitives for `pi`, preserved unchanged from the current implementation.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-04-15 | N/A | Draft | Plan created | |
| 2026-04-15 | Draft | Approved for Implementation | Explicit user approval to proceed | quattro |
