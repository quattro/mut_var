# Mixture Simulation Module Implementation Plan

**Goal:** Add `mutvar simulate` CLI workflow that emits multi-artifact outputs with stable logging and error-to-exit-code mapping.

**Architecture:** Follow existing CLI subcommand pattern (`_build_<verb>_subcommand`, `run_<verb>_cli_pipeline`) and keep business logic in pipeline modules. CLI remains imperative shell only: parse arguments, call pipeline, write files, map exceptions.

**Tech Stack:** `argparse`, existing `mut_var.cli` conventions, `polars` dataframe writers.

**Scope:** 4 phases from validated design (this file is phase 3).

**Codebase verified:** 2026-02-21

---

## Acceptance Criteria Coverage

This phase implements and tests:

### mutvar-mixture-simulation.AC2: Boundary validation before simulation
- **mutvar-mixture-simulation.AC2.3 Success:** CLI maps validation/input failures to exit code `2` and runtime failures to exit code `1`.

### mutvar-mixture-simulation.AC4: Multi-artifact outputs
- **mutvar-mixture-simulation.AC4.3 Success:** CLI writes three outputs (`truth`, `observed`, `meta`) for each simulation run.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: Add CLI parser support for `simulate`

**Verifies:** mutvar-mixture-simulation.AC4.3

**Files:**
- Modify: `src/mut_var/cli.py`

**Implementation:**
Add `_build_simulate_subcommand(subparsers)` and register it in `build_parser()`.

Required argument groups:
- `Output`: `--output-prefix` (required), `--output-dir` (default `.`)
- `Core`: `--n-rows`, `--seed`
- `Mixture`: `--weights`, `--log-var-scales`
- `Variance Link`: `--variance-link`, `--theta`, `--link-eps`, `--link-shift`, `--af-clip-min`
- `AF Model`: `--af-model`, `--af-uniform-low`, `--af-uniform-high`, `--af-beta-a`, `--af-beta-b`
- `SE Model`: `--se-model`, `--se-constant`, `--sample-size`, `--se-scale`
- `Diagnostics`: `--verbose`

Set parser dispatch with `simulate.set_defaults(func=run_simulate_cli_pipeline)`.

Use parsing helpers:
- `_parse_comma_floats(raw: str, field: str) -> tuple[float, ...]`

**Testing:**
Tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/cli.py`
Expected: module compiles.

**Commit:** `feat: add simulate subcommand argument parser`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement CLI simulation runner and file writing behavior

**Verifies:** mutvar-mixture-simulation.AC2.3, mutvar-mixture-simulation.AC4.3

**Files:**
- Modify: `src/mut_var/cli.py`

**Implementation:**
Add:

```python
def run_simulate_cli_pipeline(args: ap.Namespace, log: logging.Logger) -> int:
    ...
```

Behavior:
1. Parse comma-delimited `weights` and `log_var_scales`.
2. Build `SimulationNumericsConfig` and `SimulationPipelineConfig`.
3. Call `run_simulation_pipeline(config=..., log=log)`.
4. Write:
   - `<output-prefix>.truth.tsv`
   - `<output-prefix>.observed.tsv`
   - `<output-prefix>.meta.tsv`
   under `output_dir`.
5. Log step markers: `simulate: validating args`, `simulate: running pipeline`, `simulate: writing outputs`.
6. Exception mapping:
   - `ValueError`/`FileNotFoundError` -> return `2`
   - `RuntimeError` -> return `1`
   - success -> return `0`

No stdout dataframe streaming for this command (multi-file artifact mode only).

**Testing:**
Tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/cli.py`
Expected: module compiles.

**Commit:** `feat: add simulate cli execution and artifact writing`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Wire CLI imports and package surface for simulation

**Verifies:** mutvar-mixture-simulation.AC4.3

**Files:**
- Modify: `src/mut_var/cli.py`
- Modify: `src/mut_var/__init__.py`

**Implementation:**
Update imports so CLI uses:
- `from mut_var.simulate import SimulationPipelineConfig, run_simulation_pipeline`
- `from mut_var.numerics import SimulationNumericsConfig`

Ensure package root export includes simulation pipeline symbols to keep root API stable for Python users.

**Testing:**
Covered by Task 4 tests.

**Verification:**
Run: `python -m compileall src/mut_var/cli.py src/mut_var/__init__.py`
Expected: modules compile.

**Commit:** `feat: expose simulation pipeline for cli and python callers`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Add CLI contract tests for simulate command

**Verifies:** mutvar-mixture-simulation.AC2.3, mutvar-mixture-simulation.AC4.3

**Files:**
- Create: `tests/test_simulate_cli_contracts.py` (integration)
- Modify: `tests/test_cli_contracts.py`

**Implementation:**
`tests/test_simulate_cli_contracts.py` tests:
- `test_run_cli_help_includes_simulate_subcommand`
- `test_simulate_invalid_weight_vector_returns_exit_2`
- `test_simulate_mismatched_weights_and_scales_returns_exit_2`
- `test_simulate_runtime_failure_maps_to_exit_1` (monkeypatch `run_simulation_pipeline`)
- `test_simulate_success_writes_truth_observed_meta_files`
- `test_simulate_logs_stage_markers_without_traceback`

`tests/test_cli_contracts.py` adjustment:
- Extend canonical entrypoint assertion to include `simulate` in help output.

Reuse `_patch_streams` and `assert_no_traceback` helper style.

**Verification:**
Run: `pytest -p no:capture tests/test_simulate_cli_contracts.py tests/test_cli_contracts.py`
Expected: all tests pass.

**Commit:** `test: add simulate cli contract coverage`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->
