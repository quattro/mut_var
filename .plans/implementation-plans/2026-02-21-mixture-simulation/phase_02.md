# Mixture Simulation Module Implementation Plan

**Goal:** Add a pipeline-level simulation API that validates ingress, calls numerics, and returns `polars.DataFrame` artifacts (`truth`, `observed`, `metadata`).

**Architecture:** Keep numerics output as `Solution`/arrays and perform all tabular shaping in pipeline adapters. Pipeline uses logging stage markers (`validate/run/prepare`) and raises built-in exceptions (`ValueError`, `RuntimeError`) based on `Solution.result`.

**Tech Stack:** `polars`, existing `mut_var.contracts`, existing pipeline conventions in `infer.py`/`curve.py`.

**Scope:** 4 phases from validated design (this file is phase 2).

**Codebase verified:** 2026-02-21

---

## Acceptance Criteria Coverage

This phase implements and tests:

### mutvar-mixture-simulation.AC1: Canonical numerics and status contracts
- **mutvar-mixture-simulation.AC1.1 Success:** Public pipeline API is exposed at package root as `run_simulation_pipeline`.

### mutvar-mixture-simulation.AC4: Multi-artifact outputs
- **mutvar-mixture-simulation.AC4.1 Success:** `truth`, `observed`, and `metadata` artifacts are returned as dataframes with row-level alignment.
- **mutvar-mixture-simulation.AC4.2 Success:** Metadata includes run configuration and AF-bin diagnostics.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: Create pipeline contracts and entrypoint signature

**Verifies:** mutvar-mixture-simulation.AC1.1

**Files:**
- Create: `src/mut_var/simulate.py`
- Modify: `src/mut_var/__init__.py`

**Implementation:**
Add:

```python
class SimulationPipelineConfig(NamedTuple):
    n_rows: int
    seed: int = 0
    numerics: SimulationNumericsConfig = SimulationNumericsConfig(
        weights=(0.95, 0.05),
        log_var_scales=(-8.0, -5.5),
    )

class SimulationArtifacts(NamedTuple):
    truth: pl.DataFrame
    observed: pl.DataFrame
    metadata: pl.DataFrame


def run_simulation_pipeline(*, config: SimulationPipelineConfig, log: logging.Logger | None = None) -> SimulationArtifacts:
    ...
```

`run_simulation_pipeline` docstring must be raw and include `**Arguments:**`, `**Returns:**`, `**Raises:**`.

Export `run_simulation_pipeline`, `SimulationPipelineConfig`, and `SimulationArtifacts` via `src/mut_var/__init__.py` and include in `__all__`.

**Testing:**
Tests are added in Tasks 3-4.

**Verification:**
Run: `python -m compileall src/mut_var/simulate.py src/mut_var/__init__.py`
Expected: modules compile.

**Commit:** `feat: add simulation pipeline API contracts`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement dataframe artifact builders and metadata diagnostics

**Verifies:** mutvar-mixture-simulation.AC4.1, mutvar-mixture-simulation.AC4.2

**Files:**
- Modify: `src/mut_var/simulate.py`

**Implementation:**
Implement helpers:
- `_truth_dataframe(arrays: SimulationArrays) -> pl.DataFrame`
- `_observed_dataframe(arrays: SimulationArrays) -> pl.DataFrame`
- `_metadata_dataframe(arrays: SimulationArrays, config: SimulationPipelineConfig) -> pl.DataFrame`
- `_pipeline_reason(solution: Solution) -> str`

Required column contracts:
- `truth`: `row_id`, `component`, `beta_true`, `sigma2`, `effect_allele_frequency`
- `observed`: `row_id`, `effect_allele_frequency`, `beta`, `standard_error`
- `metadata`: `seed`, `n_rows`, `num_components`, `variance_link`, `theta`, `af_decile`, `empirical_var_beta_true`, `empirical_mean_sigma2`

AF diagnostic rule:
- Build deciles from `effect_allele_frequency` using rank quantiles.
- Emit one metadata row per decile (`af_decile` 0-9).

Row alignment rule:
- `row_id` must be 0-based contiguous and consistent across truth/observed.

**Testing:**
Tests are added in Tasks 3-4.

**Verification:**
Run: `python -m compileall src/mut_var/simulate.py`
Expected: module compiles.

**Commit:** `feat: build simulation truth observed metadata dataframes`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement pipeline orchestration and status-to-exception mapping

**Verifies:** mutvar-mixture-simulation.AC1.1, mutvar-mixture-simulation.AC4.1

**Files:**
- Modify: `src/mut_var/simulate.py`

**Implementation:**
Inside `run_simulation_pipeline`:
1. Log `simulation pipeline: validating config`.
2. Call `simulate_mixture_data`.
3. If result is `RESULTS.invalid_input` or `RESULTS.empty_subset`, raise `ValueError` with `stats["reason"]` fallback.
4. If result is any other non-success status, raise `RuntimeError`.
5. Log `simulation pipeline: preparing artifacts`.
6. Return `SimulationArtifacts(truth=..., observed=..., metadata=...)`.

Ensure no plotting/file I/O in pipeline function.

**Testing:**
Tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/simulate.py`
Expected: module compiles.

**Commit:** `feat: orchestrate simulation pipeline with explicit failure mapping`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Add pipeline tests and inference-compatibility integration test

**Verifies:** mutvar-mixture-simulation.AC1.1, mutvar-mixture-simulation.AC4.1, mutvar-mixture-simulation.AC4.2

**Files:**
- Create: `tests/test_simulate.py` (integration)
- Modify: `tests/test_infer.py`

**Implementation:**
`tests/test_simulate.py` test set:
- `test_run_simulation_pipeline_returns_three_dataframe_artifacts`
- `test_truth_and_observed_row_ids_align`
- `test_metadata_contains_expected_decile_rows`
- `test_pipeline_raises_value_error_for_invalid_input_status` (monkeypatch numerics)
- `test_pipeline_raises_runtime_error_for_nonrecoverable_status` (monkeypatch numerics)

`tests/test_infer.py` addition:
- `test_simulated_observed_output_is_accepted_by_run_inference_pipeline`:
  - call `run_simulation_pipeline`
  - pass `artifacts.observed` to `run_inference_pipeline` with small config
  - assert returned inference dataframe schema and non-empty output

**Verification:**
Run: `pytest -p no:capture tests/test_simulate.py tests/test_infer.py`
Expected: all targeted tests pass.

**Commit:** `test: add simulation pipeline and inference integration coverage`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->
