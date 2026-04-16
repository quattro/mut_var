# MutVar Shared Optimizer Loop Design

## Summary
This plan defines a shared iterative optimization routine for MutVar numerics so `baseline` and `refit` stop duplicating loop orchestration. The refactor keeps solver math local to each module (objectives, manifold-specific proposal maps, and solver-specific diagnostics) while centralizing common control-flow: epoch iteration, mode handling (`full_batch` and `sgd`), backtracking, convergence checks, non-finite handling, and `Solution` status assembly.

The design prioritizes behavior preservation and clarity. Existing result contracts (`Solution`, `RESULTS`) remain unchanged, baseline continues to support both full-batch and SGD paths, and refit continues to use full-batch with backtracking. The end state is less duplicated code, clearer naming, and one canonical optimization loop primitive that solver modules configure through typed callbacks.

## Definition of Done
A design is documented for introducing one shared optimization loop that serves both baseline and refit numerics without changing external API contracts.

The design must preserve current failure/status behavior and support both baseline SGD/full-batch and refit full-batch flows.

The design must provide clear implementation phases, exact module ownership, and testable completion criteria.

## Acceptance Criteria
### mutvar-shared-optimizer-loop.AC1: Shared optimization core exists with stable contracts
- **mutvar-shared-optimizer-loop.AC1.1 Success:** A dedicated numerics optimization module defines one canonical iterative loop with explicit typed callback contracts.
- **mutvar-shared-optimizer-loop.AC1.2 Success:** Shared loop owns epoch progression, optional backtracking, convergence decisions, and non-finite detection.
- **mutvar-shared-optimizer-loop.AC1.3 Constraint:** Shared loop does not own solver-specific objective math or manifold-specific parameter maps.

### mutvar-shared-optimizer-loop.AC2: Baseline is migrated without behavior regressions
- **mutvar-shared-optimizer-loop.AC2.1 Success:** Baseline full-batch optimization runs through the shared loop.
- **mutvar-shared-optimizer-loop.AC2.2 Success:** Baseline SGD optimization runs through the shared loop with deterministic batch sampling semantics.
- **mutvar-shared-optimizer-loop.AC2.3 Failure:** Baseline still reports `invalid_input`, `empty_subset`, and `nonfinite_objective` states consistently with current contracts.

### mutvar-shared-optimizer-loop.AC3: Refit is migrated without behavior regressions
- **mutvar-shared-optimizer-loop.AC3.1 Success:** Refit threshold fitting runs through the shared loop using full-batch mode.
- **mutvar-shared-optimizer-loop.AC3.2 Success:** Refit returns the last accepted parameter state when backtracking does not accept a candidate.
- **mutvar-shared-optimizer-loop.AC3.3 Failure/Edge:** Empty-threshold mask and non-finite-objective paths remain explicit `Solution.result` outcomes.

### mutvar-shared-optimizer-loop.AC4: Duplication is reduced and naming is clearer
- **mutvar-shared-optimizer-loop.AC4.1 Success:** Duplicated iteration/backtracking status logic is removed from baseline/refit modules.
- **mutvar-shared-optimizer-loop.AC4.2 Success:** Shared control-flow names are consistent and domain-agnostic (for example: step proposal, acceptance, backtrack policy).
- **mutvar-shared-optimizer-loop.AC4.3 Constraint:** Public APIs (`fit_baseline`, `fit_refit_grid`, numerics `run_inference_pipeline`) remain stable.

### mutvar-shared-optimizer-loop.AC5: Verification is comprehensive
- **mutvar-shared-optimizer-loop.AC5.1 Success:** Existing infer/baseline/refit regression tests remain green.
- **mutvar-shared-optimizer-loop.AC5.2 Success:** New tests cover shared-loop mode behavior and backtracking acceptance semantics.
- **mutvar-shared-optimizer-loop.AC5.3 Success:** Project gates pass (`ruff`, `mypy`, `pytest -p no:capture`).

## Glossary
- **Shared optimization loop**: A reusable iterative routine that controls epochs, acceptance, backtracking, and stopping logic.
- **Step function**: Solver-specific callback that computes gradient direction and proposes candidate parameters.
- **Full-batch mode**: Optimization step uses all observations each epoch.
- **SGD mode**: Optimization step uses sampled mini-batches each epoch.
- **Backtracking**: Step-size reduction strategy applied when a proposed candidate worsens objective or yields non-finite values.
- **Accepted iterate**: Last candidate that passed acceptance criteria and became solver state.
- **Recoverable result**: `RESULTS.successful` or `RESULTS.max_steps_reached`, where downstream orchestration can continue.
- **Manifold-specific proposal**: Parameter update map tied to solver geometry (for example simplex-only or simplex+normal-space updates).

## Architecture
The shared routine is introduced as a numerics-internal module that exposes one canonical optimization entrypoint.

Contract-level interfaces (illustrative, not implementation code):

```python
class OptimizationMode(NamedTuple):
    name: Literal["full_batch", "sgd"]

class StepOutcome(NamedTuple):
    candidate: Params
    objective: ArrayLike
    diff: ArrayLike
    accepted: bool

class OptimizationResult(NamedTuple):
    params: Params
    objective: float
    epoch_count: int
    converged: bool
    result: RESULTS
    state: dict[str, object] | None

# Shared routine
run_iterative_optimization(
    *,
    init_params: Params,
    mode: OptimizationMode,
    max_iter: int,
    tol: float,
    step_size: float,
    max_backtracks: int,
    make_epoch_context: Callable[[int, ArrayLike, rdm.PRNGKey], ArrayLike],
    compute_direction: Callable[[Params, ArrayLike], Params],
    propose_candidate: Callable[[Params, Params, float], Params],
    evaluate_objective: Callable[[Params, ArrayLike], ArrayLike],
) -> OptimizationResult
```

Ownership boundaries:
- Shared loop module owns generic iteration control.
- `baseline.py` owns objective definitions, normal-space + simplex proposal map, and SGD context setup.
- `refit.py` owns penalized objective, simplex-only proposal map, and threshold diagnostics assembly.
- `pipeline.py` remains orchestration-only and keeps public numerics API stable.

## Existing Patterns
The current numerics code already provides patterns this design follows:

- `src/mut_var/contracts.py` enforces structured `Solution` + `RESULTS` outcomes as canonical status contracts.
- `src/mut_var/numerics/baseline.py` and `src/mut_var/numerics/refit.py` already separate objective math from orchestration entrypoints.
- `src/mut_var/numerics/pipeline.py` centralizes baseline + refit orchestration behind a single numerics entrypoint.
- `src/mut_var/numerics/_solver_utils.py` already hosts shared low-level helpers (simplex map, non-finite checks, backtracking predicate, recoverable-status helpers).

This design intentionally diverges from the current pattern where each solver owns a full optimization loop. That divergence is justified because those loops now duplicate control-flow concerns while differing only in solver-specific callbacks.

## Implementation phases
<!-- START_PHASE_1 -->
### Phase 1: Shared optimization contracts
**Goal:** Define a reusable optimization loop contract without changing solver behavior.

**Components:**
- `src/mut_var/numerics/_optimize.py` (new) — shared loop entrypoint, mode enum/typing, step/outcome/result types.
- `src/mut_var/numerics/_solver_utils.py` — keep low-level math helpers and shared predicates consumed by `_optimize.py`.

**Dependencies:** None.

**Done when:** Shared loop contracts compile, are documented, and tests for `mutvar-shared-optimizer-loop.AC1.1` and `mutvar-shared-optimizer-loop.AC1.2` pass.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Baseline full-batch integration
**Goal:** Route baseline full-batch iterations through the shared loop.

**Components:**
- `src/mut_var/numerics/baseline.py` — replace full-batch loop body with shared-loop callbacks (`compute_direction`, `propose_candidate`, `evaluate_objective`).
- `tests/test_infer_opt.py` — add/adjust baseline regression tests for convergence/result/stat parity.

**Dependencies:** Phase 1.

**Done when:** Baseline full-batch path uses the shared loop and tests for `mutvar-shared-optimizer-loop.AC2.1` and `mutvar-shared-optimizer-loop.AC4.3` pass.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Baseline SGD integration
**Goal:** Move baseline SGD epoch context and step progression into shared-loop mode handling.

**Components:**
- `src/mut_var/numerics/baseline.py` — supply SGD batch context callback and scaling behavior via shared-loop interfaces.
- `tests/test_infer_opt.py` and `tests/test_infer_stability.py` — verify SGD/full-batch mode correctness and deterministic behavior under fixed seeds.

**Dependencies:** Phase 2.

**Done when:** Baseline SGD uses shared loop with unchanged status semantics and tests for `mutvar-shared-optimizer-loop.AC2.2`, `mutvar-shared-optimizer-loop.AC2.3`, and `mutvar-shared-optimizer-loop.AC5.2` pass.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Refit full-batch integration
**Goal:** Route refit threshold fitting through the same shared loop.

**Components:**
- `src/mut_var/numerics/refit.py` — replace `_fit_single_refit` loop body with shared-loop callbacks and preserve threshold diagnostics.
- `tests/test_infer_opt.py` — retain and extend backtracking acceptance regression coverage.

**Dependencies:** Phase 1.

**Done when:** Refit full-batch path uses shared loop and tests for `mutvar-shared-optimizer-loop.AC3.1`, `mutvar-shared-optimizer-loop.AC3.2`, and `mutvar-shared-optimizer-loop.AC3.3` pass.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Numerics orchestration cleanup
**Goal:** Remove leftover duplication and normalize naming around shared optimization concepts.

**Components:**
- `src/mut_var/numerics/baseline.py` and `src/mut_var/numerics/refit.py` — remove obsolete loop helpers and align naming to shared contracts.
- `src/mut_var/numerics/pipeline.py` — keep orchestration minimal and confirm stable API/contract usage.
- `src/mut_var/numerics/__init__.py` — expose new internal module symbols only if needed.

**Dependencies:** Phases 2-4.

**Done when:** Duplicate loop code is removed and tests for `mutvar-shared-optimizer-loop.AC4.1`, `mutvar-shared-optimizer-loop.AC4.2`, and `mutvar-shared-optimizer-loop.AC4.3` pass.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Full verification and documentation alignment
**Goal:** Confirm project-wide correctness and record shared optimizer design guarantees.

**Components:**
- `tests/test_infer.py`, `tests/test_infer_opt.py`, `tests/test_infer_stability.py` — final regression pass for baseline/refit/pipeline interactions.
- `AGENTS.md` — update numerics contract notes if module ownership or invariants changed.

**Dependencies:** Phases 1-5.

**Done when:** `ruff`, `mypy`, and `pytest -p no:capture` pass and tests for `mutvar-shared-optimizer-loop.AC5.1` and `mutvar-shared-optimizer-loop.AC5.3` pass.
<!-- END_PHASE_6 -->

## Additional considerations
- Scope is intentionally limited to loop orchestration reuse; objective definitions and manifold math stay solver-local.
- Public API compatibility is required. Any contract change to `Solution` shape, `RESULTS` semantics, or public function signatures is out of scope.
- Numerical parity should be validated by result-status and stability tests, not strict coefficient identity across every stochastic path.
- Shared-loop abstractions should remain minimal; avoid introducing framework-style indirection beyond baseline/refit needs.
