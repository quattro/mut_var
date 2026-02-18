# MutVar Whole Project Hardening Design

## Summary
This plan defines a contract-first hardening of `mut-var` so each workflow has a clean boundary from user input, to validated typed data, to JAX/Equinox numerics, to optional side effects like plots. The core change is operational: numerics stop signaling success/failure implicitly and instead return structured status objects (`Solution` + `RESULTS`), so CLI behavior becomes predictable and diagnosable across invalid input, empty subsets, and non-finite optimization paths.

The implementation strategy is phased to reduce risk: lock down validation and failure contracts first, then modularize baseline/refit/curve kernels behind canonical APIs, then isolate plotting from fit logic, then optimize tracing/runtime behavior with benchmark evidence, and finally enforce the new guarantees in CI and release documentation. The result is a single coherent breaking release where behavior changes are intentional, measurable, and migration-ready.

## Definition of Done
We produce a whole-project redesign plan for `mut-var` covering numerics core, CLI/I/O boundaries, testing/CI, and documentation/API contracts.

Correctness and robustness are first priority, and breaking changes are allowed when they materially improve failure handling, validation, and reliability.

Algorithmic changes are limited to targeted, justified improvements (no wholesale model replacement).

Success requires robustness gains plus measurable performance improvement, with at least a modest floor (about 20%) on representative workloads via reduced retracing/runtime overhead.

## Acceptance Criteria
### mutvar-whole-project-hardening.AC1: Canonical architecture and contracts
- **mutvar-whole-project-hardening.AC1.1 Success:** The redesign defines explicit workflow, adapter, numerics, and plotting boundaries with concrete file ownership.
- **mutvar-whole-project-hardening.AC1.2 Success:** Each workflow has one canonical public entrypoint, with documented input/output contracts.
- **mutvar-whole-project-hardening.AC1.3 Success:** Core numerics entrypoints return structured status results (`value`, `result`, optional `stats/state`) instead of implicit success.
- **mutvar-whole-project-hardening.AC1.4 Success:** Breaking API/CLI behavior changes are explicitly documented with migration guidance.

### mutvar-whole-project-hardening.AC2: Robust boundary validation and deterministic failures
- **mutvar-whole-project-hardening.AC2.1 Failure:** Invalid or non-numeric summary-stat inputs (including AF out of range and non-positive SE) fail at boundary validation before numerics execute.
- **mutvar-whole-project-hardening.AC2.2 Failure:** Invalid MAF grid parameters (`lowest`, `highest`, `num_breaks`) fail fast with actionable errors before fitting starts.
- **mutvar-whole-project-hardening.AC2.3 Success:** CLI maps validation/result states to stable exit codes and clean stderr output (no uncaught traceback contract).
- **mutvar-whole-project-hardening.AC2.4 Failure/Edge:** Empty-subset and non-finite-numerics paths return explicit result codes/messages with enough context to diagnose the failing stage.

### mutvar-whole-project-hardening.AC3: Modularity with targeted algorithm changes only
- **mutvar-whole-project-hardening.AC3.1 Success:** Baseline/refit orchestration is modularized so data adapters, objectives, and optimizer drivers are separated.
- **mutvar-whole-project-hardening.AC3.2 Success:** Tabular data is converted to arrays at ingress; traced numerics run on arrays/PyTrees only.
- **mutvar-whole-project-hardening.AC3.3 Success:** Curve fitting can run in fit-only mode without plotting dependencies or file side effects.
- **mutvar-whole-project-hardening.AC3.4 Success:** Plot generation is isolated to an optional adapter path and does not alter fit outputs.
- **mutvar-whole-project-hardening.AC3.5 Constraint:** Algorithm changes remain targeted and documented; wholesale model/objective redesign is excluded.

### mutvar-whole-project-hardening.AC4: Measurable performance and verification gates
- **mutvar-whole-project-hardening.AC4.1 Success:** A reproducible benchmark harness exists with fixed seeds/config and representative datasets.
- **mutvar-whole-project-hardening.AC4.2 Success:** Benchmark reports compile-related cost and steady-state runtime separately.
- **mutvar-whole-project-hardening.AC4.3 Success:** The redesigned pipeline achieves at least a 20% steady-state runtime improvement on the benchmark workload.
- **mutvar-whole-project-hardening.AC4.4 Success:** CI executes regression and stability tests, including `pytest -p no:capture`, and fails on contract regressions.
- **mutvar-whole-project-hardening.AC4.5 Failure Gate:** Release is blocked if AC4.3 is not met or benchmark evidence is missing.

## Glossary
- **Boundary validation**: Upfront input checks (types, nulls, numeric ranges, domain rules) that run before numerics.
- **Canonical public entrypoint**: The single supported API/command surface for a workflow, with defined I/O and error semantics.
- **Adapter layer**: Modules that handle file/tabular I/O and convert data into array-first forms for numerics.
- **Numerics layer**: Pure computational modules (baseline fit, refit, curve fit) that avoid file/plot side effects.
- **`Solution` contract**: Structured return object carrying computed output plus status and optional diagnostics.
- **`RESULTS` codes**: Enumerated outcome states (for example `invalid_input`, `empty_subset`, `nonfinite_objective`) used for deterministic handling.
- **MAF grid parameters (`lowest`, `highest`, `num_breaks`)**: Controls that define the threshold sequence used for repeated refits.
- **Empty subset**: A filtered stage where zero rows remain, handled as an explicit result state rather than a crash.
- **Non-finite objective**: Objective value becomes `NaN`/`Inf` during optimization, signaling numerical failure.
- **JIT boundary**: Interface where JAX compilation/tracing begins; non-array tabular objects should not cross it.
- **PyTree**: JAX-compatible nested structure of arrays/containers used as model and function inputs.
- **`eqx.error_if`**: Equinox mechanism for JAX-safe runtime error signaling inside traced computations.
- **Retracing**: Recompilation caused by changing static arguments/shapes, often a major performance cost.
- **Steady-state runtime**: Runtime measured after compilation overhead, reflecting ongoing execution cost.
- **Compatibility shim**: Temporary thin wrapper preserving old call patterns while internals are restructured.
- **Big-bang release**: One intentionally breaking release that introduces all contract and behavior changes together.

## Architecture
The redesign uses a strict layered pipeline with explicit contracts at each boundary.

- CLI workflow layer (`src/mut_var/cli.py`) stays thin and handles argument parsing, boundary validation, and exit-code mapping.
- Adapter layer (`src/mut_var/io.py`, plus new `src/mut_var/adapters/`) owns TSV/Polars ingress/egress and converts tabular data to typed arrays at ingress.
- Numerics layer (new `src/mut_var/numerics/`) owns baseline fitting, MAF refits, and curve fitting with array-only inputs/outputs.
- Plotting/output layer (new `src/mut_var/plotting/`) owns side effects such as PNG generation and never runs inside core numerics.

Core numerics entrypoints return structured status results instead of implicit success. Contract shape:

```python
class RESULTS(...):
    successful = ""
    invalid_input = "input validation failed"
    empty_subset = "maf/se filtering removed all rows"
    nonfinite_objective = "objective became nonfinite"
    max_steps_reached = "solver reached max steps"

class Solution(...):
    value: object
    result: RESULTS
    stats: dict[str, object]
    state: object
```

JAX/Equinox constraints for this architecture:
- JIT boundaries only at public numerics entrypoints.
- Non-array configuration is static; dynamic inputs are arrays or PyTrees of arrays.
- Tabular objects do not cross JIT boundaries.
- Runtime checks inside traced kernels use JAX-safe signaling (`result` channels, `eqx.error_if`), while user-facing validation errors are raised at boundary layers.

## Existing Patterns
Investigation found useful patterns worth preserving:

- `src/mut_var/cli.py` already acts as an orchestration layer with `infer` and `fit` subcommands.
- `src/mut_var/io.py` already centralizes file reads and required-column checks.
- `pyproject.toml` already defines one canonical CLI entrypoint (`mut_var.cli:run_cli`) and central dependency/tool configuration.
- Tests are organized by domain (`tests/test_io.py`, `tests/test_infer.py`, `tests/test_infer_opt.py`, `tests/test_infer_stability.py`, `tests/test_curve.py`).

This design intentionally diverges where the current structure blocks robustness and performance:

- `src/mut_var/infer.py` currently mixes data filtering, optimization logic, logging, and orchestration in one module.
- `src/mut_var/curve.py` currently mixes fitting and plotting side effects.
- Input/range validation is currently incomplete in `src/mut_var/io.py` and leaks failures into deeper numerics.

The redesign keeps the existing CLI/user workflow but replaces internal module boundaries and failure contracts in one breaking release.

## Implementation phases
<!-- START_PHASE_1 -->
### Phase 1: Boundary contracts and validation reset
**Goal:** Make boundary validation explicit and deterministic before numerics execution.

**Components:**
- `src/mut_var/io.py` — extend validation to numeric/range/domain checks (`AF`, `beta`, `SE`, MAF range args).
- `src/mut_var/cli.py` — validate CLI ranges and map failure statuses to stable exit codes/stderr.
- `src/mut_var/contracts.py` (new) — define `RESULTS` and `Solution` contracts for numerics workflows.
- `tests/test_io.py` — add failing and passing cases for numeric/range validation.
- `tests/test_cli_contracts.py` (new) — verify parse failures, stderr contract, and exit-code mapping.

**Dependencies:** None.

**Done when:** Invalid input fails before numerics start, error messages are actionable, and tests for `mutvar-whole-project-hardening.AC1.1`, `mutvar-whole-project-hardening.AC1.2`, `mutvar-whole-project-hardening.AC2.1`, and `mutvar-whole-project-hardening.AC2.2` pass.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Numerics core modularization
**Goal:** Separate data adapters from optimization kernels and expose one canonical numerics API per workflow.

**Components:**
- `src/mut_var/numerics/baseline.py` (new) — baseline mixture objective and optimization driver.
- `src/mut_var/numerics/refit.py` (new) — MAF-threshold weight refit objective and driver.
- `src/mut_var/numerics/pipeline.py` (new) — orchestration of baseline fit, component filtering, and threshold refits.
- `src/mut_var/infer.py` — reduced to compatibility shim or thin adapter to the new numerics package.
- `tests/test_infer.py` and `tests/test_infer_opt.py` — updated for new contracts and module boundaries.

**Dependencies:** Phase 1.

**Done when:** Numerics modules accept array-first inputs through adapters, return `Solution`-style status, and tests for `mutvar-whole-project-hardening.AC1.3`, `mutvar-whole-project-hardening.AC3.1`, and `mutvar-whole-project-hardening.AC3.2` pass.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Curve fitting and plotting separation
**Goal:** Decouple pure curve fitting from plotting side effects and align with the same status contract.

**Components:**
- `src/mut_var/numerics/curve_fit.py` (new) — pure curve-fitting kernels and solver status handling.
- `src/mut_var/plotting/curve_plots.py` (new) — optional plotting adapter for PNG output.
- `src/mut_var/curve.py` — thin workflow adapter that calls numerics and plotting modules.
- `tests/test_curve.py` — add coverage for fit-only mode and plotting contract behavior.

**Dependencies:** Phase 1 and Phase 2.

**Done when:** Curve fitting runs without importing plotting dependencies in fit-only paths, plotting is optional side effect, and tests for `mutvar-whole-project-hardening.AC3.3` and `mutvar-whole-project-hardening.AC3.4` pass.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Performance rework with stable tracing
**Goal:** Achieve measurable runtime improvement by reducing retracing and repeated host-device conversion.

**Components:**
- `src/mut_var/numerics/refit.py` — refactor threshold loop for stable PyTree/shape behavior where feasible.
- `src/mut_var/adapters/array_cache.py` (new) — cache ingress conversions and mask-driven reuse across thresholds.
- `src/mut_var/numerics/profiling.py` (new) — collect compile count, compile time, and steady-state runtime.
- `tests/test_infer_stability.py` — extend for retrace-sensitive stability checks under varied thresholds.
- `benchmarks/infer_runtime.py` (new) — reproducible benchmark harness and baseline comparison.

**Dependencies:** Phase 2.

**Done when:** Benchmark harness shows at least 20% runtime improvement on representative datasets and tests for `mutvar-whole-project-hardening.AC4.1`, `mutvar-whole-project-hardening.AC4.2`, and `mutvar-whole-project-hardening.AC4.3` pass.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Verification and CI hardening
**Goal:** Enforce correctness contracts continuously across validation, numerics, and CLI behaviors.

**Components:**
- `tests/test_io.py`, `tests/test_infer.py`, `tests/test_infer_opt.py`, `tests/test_infer_stability.py`, `tests/test_curve.py`, `tests/test_cli_contracts.py` — finalize regression suite.
- `pyproject.toml` — ensure lint/type/test commands and dependency constraints are explicit.
- `.github/workflows/ci.yml` (new) — run lint, type checks, and `pytest -p no:capture`.

**Dependencies:** Phases 1-4.

**Done when:** CI gates pass with the redesigned contracts and tests for `mutvar-whole-project-hardening.AC2.3`, `mutvar-whole-project-hardening.AC3.5`, and `mutvar-whole-project-hardening.AC4.4` pass.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Big-bang release and migration communication
**Goal:** Ship one breaking release with explicit migration guidance and updated user-facing contracts.

**Components:**
- `README.md` — update command contracts, required inputs, failure semantics, and reproducibility notes.
- `src/mut_var/__init__.py` and public module docs — align exposed API with new canonical entrypoints.
- `docs/design-plans/2026-02-18-mutvar-whole-project-hardening.md` — final design record with accepted criteria.
- `CHANGELOG.md` (new) — document breaking changes, removed behaviors, and migration instructions.

**Dependencies:** Phases 1-5.

**Done when:** Breaking release notes are published, old implicit behaviors are removed, and tests for `mutvar-whole-project-hardening.AC1.4`, `mutvar-whole-project-hardening.AC2.4`, and `mutvar-whole-project-hardening.AC4.5` pass.
<!-- END_PHASE_6 -->

## Additional considerations
- Big-bang migration is intentional. No compatibility shims are planned for legacy behavior.
- Targeted algorithmic changes are allowed only when they improve correctness/stability and are backed by regression tests and benchmark evidence.
- Performance acceptance uses representative data and must report compile time and steady-state runtime separately.
- Boundary validation is a first-class feature; deep numerics should never be the first place users discover malformed input.
