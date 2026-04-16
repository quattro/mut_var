# Mixture Simulation Module Implementation Plan

**Goal:** Finalize documentation, contract context, and quality gates for the new simulation workflow.

**Architecture:** Keep user-facing contracts centralized in package-root exports and AGENTS docs. Ensure docs and tests describe simulation outputs as direct input for `mutvar infer` and preserve existing quality gate commands.

**Tech Stack:** Markdown docs, `ruff`, `mypy`, `pytest`.

**Scope:** 4 phases from validated design (this file is phase 4).

**Codebase verified:** 2026-02-21

---

## Acceptance Criteria Coverage

This phase implements and tests:

### mutvar-mixture-simulation.AC1: Canonical numerics and status contracts
- **mutvar-mixture-simulation.AC1.3 Success:** Public docs and project context reflect the new simulation API and status conventions.

### mutvar-mixture-simulation.AC5: Verification and documentation
- **mutvar-mixture-simulation.AC5.1 Success:** New tests cover numerics, pipeline, and CLI paths.
- **mutvar-mixture-simulation.AC5.2 Success:** Project quality gates pass with simulation feature integrated.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Update user-facing and project-context documentation

**Verifies:** mutvar-mixture-simulation.AC1.3

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `src/mut_var/numerics/AGENTS.md`

**Implementation:**
`README.md` updates:
- Add `mutvar simulate` usage example.
- Document multi-artifact outputs (`.truth.tsv`, `.observed.tsv`, `.meta.tsv`).
- Show inference handoff command using `.observed.tsv`.

`AGENTS.md` updates:
- Add new exposed contracts:
  - CLI subcommand: `simulate`
  - pipeline API: `run_simulation_pipeline`
  - numerics API: `mut_var.numerics.simulate_mixture_data`
  - config/contracts: `SimulationPipelineConfig`, `SimulationNumericsConfig`, `SimulationArtifacts`
- Add simulation-specific constraints to Guarantees/Expects where needed.

`src/mut_var/numerics/AGENTS.md` updates:
- Include simulation numerics entrypoint and invariants (`sigma2` finite/positive, reproducibility with seed).

**Verification:**
Run: `rg "simulate|run_simulation_pipeline|simulate_mixture_data" README.md AGENTS.md src/mut_var/numerics/AGENTS.md`
Expected: all references present and consistent.

**Commit:** `docs: add simulation contracts and usage guidance`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add end-to-end smoke test and fixture sanity checks

**Verifies:** mutvar-mixture-simulation.AC5.1

**Files:**
- Create: `tests/test_simulate_end_to_end.py` (integration)

**Implementation:**
Add one high-value smoke test:
- run `run_simulation_pipeline` with small config
- pass `artifacts.observed` into `run_inference_pipeline`
- assert both outputs are non-empty and schema-valid
- assert no exceptions for valid config

Keep runtime low (`n_rows <= 500`) and deterministic (`seed=0`).

**Verification:**
Run: `pytest -p no:capture tests/test_simulate_end_to_end.py`
Expected: test passes quickly (<5s local target).

**Commit:** `test: add simulation-to-inference end-to-end smoke coverage`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Run full quality gates and finalize release-ready state

**Verifies:** mutvar-mixture-simulation.AC5.2

**Files:**
- Modify: none (verification-only unless fixes are required)

**Implementation:**
Execute project gates in canonical order and fix any failures before completion:
1. `ruff check src/mut_var tests`
2. `mypy src/mut_var tests`
3. `pytest -p no:capture`

If any command fails, create targeted follow-up commit(s) and rerun all three commands until green.

**Verification:**
Run: all three commands above.
Expected: zero failures.

**Commit:** `chore: pass full quality gates for simulation module`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->
