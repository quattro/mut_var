# Mixture Simulation Test Requirements

This document maps acceptance criteria to concrete automated tests for execution.

## Criteria To Automated Tests

### mutvar-mixture-simulation.AC1.1
- **Criterion:** Public pipeline API is exposed at package root as `run_simulation_pipeline`.
- **Test type:** Unit
- **Expected test file:** `tests/test_simulate.py`
- **Test cases:**
  - `test_package_root_exports_simulation_pipeline_entrypoint`

### mutvar-mixture-simulation.AC1.2
- **Criterion:** Numerics simulation entrypoint returns `Solution` and uses `Solution.result` as canonical status signal.
- **Test type:** Unit
- **Expected test file:** `tests/test_simulate_numerics.py`
- **Test cases:**
  - `test_simulate_mixture_data_returns_solution_and_arrays_on_valid_config`
  - `test_simulate_mixture_data_rejects_invalid_weight_shapes`

### mutvar-mixture-simulation.AC1.3
- **Criterion:** Public docs and project context reflect the new simulation API and status conventions.
- **Test type:** Integration (documentation consistency)
- **Expected test file:** none (verified via review + grep command in phase 4)
- **Human verification:** required

### mutvar-mixture-simulation.AC2.1
- **Criterion:** Invalid mixture/AF/SE domains fail before random draws with `RESULTS.invalid_input`.
- **Test type:** Unit
- **Expected test file:** `tests/test_simulate_numerics.py`
- **Test cases:**
  - `test_simulate_mixture_data_rejects_invalid_weight_shapes`
  - `test_simulate_mixture_data_rejects_invalid_theta`
  - `test_simulate_mixture_data_rejects_invalid_af_domains`

### mutvar-mixture-simulation.AC2.3
- **Criterion:** CLI maps validation/input failures to exit `2` and runtime failures to exit `1`.
- **Test type:** Integration
- **Expected test file:** `tests/test_simulate_cli_contracts.py`
- **Test cases:**
  - `test_simulate_invalid_weight_vector_returns_exit_2`
  - `test_simulate_mismatched_weights_and_scales_returns_exit_2`
  - `test_simulate_runtime_failure_maps_to_exit_1`

### mutvar-mixture-simulation.AC3.1
- **Criterion:** Variance link families produce finite positive variances.
- **Test type:** Unit/property style
- **Expected test file:** `tests/test_simulate_numerics.py`
- **Test cases:**
  - `test_variance_link_outputs_positive_finite_sigma2_for_all_links`

### mutvar-mixture-simulation.AC3.3
- **Criterion:** Simulation reproducible for fixed seed and config.
- **Test type:** Unit
- **Expected test file:** `tests/test_simulate_numerics.py`
- **Test cases:**
  - `test_simulate_mixture_data_reproducible_for_fixed_seed`

### mutvar-mixture-simulation.AC4.1
- **Criterion:** `truth`, `observed`, and `metadata` artifacts are row-aligned and dataframe-based.
- **Test type:** Integration
- **Expected test file:** `tests/test_simulate.py`
- **Test cases:**
  - `test_run_simulation_pipeline_returns_three_dataframe_artifacts`
  - `test_truth_and_observed_row_ids_align`

### mutvar-mixture-simulation.AC4.2
- **Criterion:** Metadata includes run config and AF-bin diagnostics.
- **Test type:** Integration
- **Expected test file:** `tests/test_simulate.py`
- **Test cases:**
  - `test_metadata_contains_expected_decile_rows`

### mutvar-mixture-simulation.AC4.3
- **Criterion:** CLI writes three outputs for each run.
- **Test type:** Integration
- **Expected test file:** `tests/test_simulate_cli_contracts.py`
- **Test cases:**
  - `test_simulate_success_writes_truth_observed_meta_files`

### mutvar-mixture-simulation.AC5.1
- **Criterion:** New tests cover numerics, pipeline, and CLI paths.
- **Test type:** Meta-check
- **Expected test file:** none
- **Verification command:**
  - `pytest -p no:capture tests/test_simulate_numerics.py tests/test_simulate.py tests/test_simulate_cli_contracts.py tests/test_simulate_end_to_end.py`

### mutvar-mixture-simulation.AC5.2
- **Criterion:** Full project quality gates pass.
- **Test type:** Project gate
- **Expected command set:**
  - `ruff check src/mut_var tests`
  - `mypy src/mut_var tests`
  - `pytest -p no:capture`

## Human Verification Items

- `mutvar-mixture-simulation.AC1.3` (documentation language and contract consistency).
- CLI help text readability for new parameter groups.

