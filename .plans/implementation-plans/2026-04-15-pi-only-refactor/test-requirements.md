# Pi-Only Refactor — Human Test Requirements

This document enumerates human verification steps for each acceptance criterion
in the pi-only-refactor implementation plan
(`.plans/design-plans/2026-04-15-pi-only-refactor.md`). Each entry specifies the
inputs to use, the command or action to perform, and the outcome to verify.

Preconditions for all checks:
- Working copy is at the tip of the pi-only-refactor branch with all four
  phases merged.
- A clean editable install is in place (`pip install -e .`).
- The `mutvar` CLI entrypoint is on `PATH`.
- Fixtures under `tests/fixtures/` are available (in particular
  `sumstats_valid.tsv`).
- Python 3.x with JAX, Equinox, Optimistix, Polars, matplotlib installed per
  `pyproject.toml`.

---

## Traceability Matrix

Each AC case below is implemented (and has automated tests generated) in the
referenced phase and task. Phase numbers are 1-4; task numbers are within the
phase. An AC may span multiple phases when incremental work contributes to it
(e.g., AC4.x quality-gate criteria are re-verified at the end of each phase).

| AC ID                        | Phase(s) | Task(s)                                    | Summary of Implementing Work                                                                 |
|------------------------------|----------|--------------------------------------------|----------------------------------------------------------------------------------------------|
| pi-only-refactor.AC1.1       | 2        | P2.T2                                      | `prepare_fit_state` returns `Solution(successful)` with `(n, K)` likelihood matrix.          |
| pi-only-refactor.AC1.2       | 2        | P2.T2, P2.T4                               | `fit_baseline` converges; `sum(pi)==1`; `mu_k`/`var_k` match initial state.                  |
| pi-only-refactor.AC1.3       | 2        | P2.T2, P2.T4                               | `fit_refit_step` returns updated `pi` (sum 1), unchanged `mu_k`/`var_k`.                     |
| pi-only-refactor.AC1.4       | 2        | P2.T2, P2.T4                               | `prepare_fit_state` returns `RESULTS.invalid_input` on non-positive `s2`.                    |
| pi-only-refactor.AC1.5       | 2        | P2.T2, P2.T4                               | `prepare_fit_state` returns `RESULTS.empty_subset` on empty arrays.                          |
| pi-only-refactor.AC1.6       | 2        | P2.T2, P2.T4                               | `fit_baseline` returns `RESULTS.max_steps_reached` (no exception) on `max_iter=1`.           |
| pi-only-refactor.AC2.1       | 1        | P1.T1, P1.T5                               | `types.py` exposes `RESULTS`, `Solution`, `InferenceConfig`, `SimulationPipelineConfig`.     |
| pi-only-refactor.AC2.2       | 1        | P1.T2, P1.T5                               | `io.py` exposes `load_inference_arrays`, `to_inference_arrays`, `build_maf_masks`, `payload_to_long_dataframe`, `InferenceArrays`. |
| pi-only-refactor.AC2.3       | 3        | P3.T1, P3.T4                               | `pipelines/__init__.py` exposes all three pipeline functions.                                |
| pi-only-refactor.AC2.4       | 2        | P2.T2, P2.T5                               | `numerics/mixture_fit.py` exposes `prepare_fit_state`, `fit_baseline`, `fit_refit_step`, `FitState`, `Params`. |
| pi-only-refactor.AC2.5       | 1        | P1.T5                                      | `mut_var.contracts` deleted.                                                                 |
| pi-only-refactor.AC2.6       | 1        | P1.T5                                      | `mut_var.adapters` deleted.                                                                  |
| pi-only-refactor.AC2.7       | 2        | P2.T5                                      | `mut_var.numerics.baseline` deleted.                                                         |
| pi-only-refactor.AC2.8       | 2        | P2.T5                                      | `mut_var.numerics.refit` deleted.                                                            |
| pi-only-refactor.AC3.1       | 3, 4     | P3.T1, P3.T2, P3.T3, P4.T3, P4.T5          | `run_inference_pipeline(path, ...)` returns long DataFrame with required columns.            |
| pi-only-refactor.AC3.2       | 3        | P3.T1                                      | No intermediate DataFrame retained past `load_inference_arrays`.                             |
| pi-only-refactor.AC3.3       | 4        | P4.T2, P4.T5                               | `mutvar infer` with supported flags (`--num-breaks`) exits 0.                                |
| pi-only-refactor.AC3.4       | 3, 4     | P3.T1, P4.T5                               | `mutvar curve` and `mutvar simulate` exit 0.                                                 |
| pi-only-refactor.AC3.5       | 4        | P4.T2, P4.T5                               | `mutvar infer --step-size 0.1` exits 2 (unknown argument).                                   |
| pi-only-refactor.AC3.6       | 4        | P4.T2, P4.T5                               | `mutvar infer --seed 42` exits 2 (unknown argument).                                         |
| pi-only-refactor.AC3.7       | 4        | P4.T2, P4.T5                               | `mutvar infer missing.tsv` exits 2 (file not found).                                         |
| pi-only-refactor.AC3.8       | 3, 4     | P3.T2, P4.T5                               | `run_inference_pipeline("missing.tsv", ...)` raises `FileNotFoundError`.                     |
| pi-only-refactor.AC4.1       | 1-4      | P1.T5, P2.T5, P3.T4, P4.T5                 | `ruff check src/mut_var tests` passes.                                                       |
| pi-only-refactor.AC4.2       | 1-4      | P1.T5, P2.T5, P3.T4, P4.T5                 | `mypy src/mut_var tests` passes.                                                             |
| pi-only-refactor.AC4.3       | 1-4      | P1.T5, P2.T5, P3.T4, P4.T5                 | `pytest -p no:capture` passes.                                                               |

---

## AC1 — Numerics simplified; `pi` is the sole optimization variable

Verifies the new `numerics/mixture_fit.py` API contract.

### pi-only-refactor.AC1.1 — `prepare_fit_state` succeeds with correct likelihood shape

- What to verify: For valid 1D `beta_hat`/`s2` arrays and an `InferenceConfig`
  with `num_clusters = K`, `prepare_fit_state` returns a `Solution` whose
  `.result` equals `RESULTS.successful` and whose `.value` is a `FitState`
  with `likelihood_matrix.shape == (n, K)`.
- Inputs:
  - `beta_hat = np.random.default_rng(0).normal(0.0, 0.1, size=50)` (shape
    `(50,)`).
  - `s2 = np.full(50, 0.01)` (strictly positive).
  - `config = InferenceConfig(num_clusters=5, max_iter=5)`.
- Action: Call `prepare_fit_state(beta_hat, s2, config)` from a Python shell
  (`from mut_var.numerics.mixture_fit import prepare_fit_state`;
  `from mut_var.types import InferenceConfig`).
- Expected result: `sol.result == RESULTS.successful`;
  `sol.value.likelihood_matrix.shape == (50, 5)`;
  `sol.value.initial_params.pi.shape == (5,)` with values ≈ `1/5`.

### pi-only-refactor.AC1.2 — `fit_baseline` converges; `pi` sums to 1; grid preserved

- What to verify: End-to-end `prepare_fit_state` → `fit_baseline` on a small
  synthetic dataset returns a `Solution` with `RESULTS.successful`, where
  `sum(pi) ≈ 1.0` within float tolerance and `Params.mu_k` / `Params.var_k`
  are bit-identical to the initial `FitState.initial_params.mu_k`/`var_k`.
- Inputs:
  - `beta_hat` drawn from a mixture: e.g.
    `np.concatenate([rng.normal(0, 0.01, 80), rng.normal(0, 0.1, 20)])`.
  - `s2 = np.full(100, 0.01)`.
  - `config = InferenceConfig(num_clusters=5, max_iter=50, tol=1e-3)`.
- Action: Call `prepare_fit_state`, then `fit_baseline(state.value, config)`.
- Expected result:
  - `baseline.result == RESULTS.successful`.
  - `abs(float(jnp.sum(baseline.value.pi)) - 1.0) < 1e-5`.
  - `jnp.allclose(baseline.value.mu_k, state.value.initial_params.mu_k)`.
  - `jnp.allclose(baseline.value.var_k, state.value.initial_params.var_k)`.

### pi-only-refactor.AC1.3 — `fit_refit_step` updates `pi`; leaves `mu_k`/`var_k` unchanged

- What to verify: After a successful baseline fit, `fit_refit_step` on a
  MAF-sliced likelihood matrix returns `Params` whose `pi` sums to 1 and whose
  `mu_k`/`var_k` are identical to the `prev_params` passed in.
- Inputs:
  - Reuse the baseline `Params` from AC1.2.
  - Construct a subset mask covering ~50% of rows; slice the likelihood matrix
    columns to the filtered component set and rows to the subset (using the
    same pattern `L[:, keep][mask, :]` as in `pipelines/inference.py`).
  - Same `InferenceConfig` as AC1.2.
- Action: Call
  `fit_refit_step(L_sub, prev_params=baseline_params, config=config)`.
- Expected result:
  - `refit.result == RESULTS.successful`.
  - `abs(float(jnp.sum(refit.value.pi)) - 1.0) < 1e-5`.
  - `refit.value.mu_k is prev_params.mu_k or jnp.array_equal(...)` — i.e. the
    same grid metadata is carried.
  - `refit.value.var_k` is element-wise identical to `prev_params.var_k`.

### pi-only-refactor.AC1.4 — `prepare_fit_state` rejects non-positive `s2`

- What to verify: A `Solution` with `RESULTS.invalid_input` is returned (not an
  exception) when any element of `s2` is ≤ 0.
- Inputs:
  - `beta_hat = np.zeros(10)`; `s2 = np.array([0.01]*9 + [0.0])` (one zero
    entry).
  - `config = InferenceConfig(num_clusters=3)`.
- Action: Call `prepare_fit_state(beta_hat, s2, config)`.
- Expected result: `sol.result == RESULTS.invalid_input`; `sol.value is None`;
  `sol.stats` contains a reason mentioning `s2`/`positive`. No exception
  raised.

### pi-only-refactor.AC1.5 — `prepare_fit_state` handles empty inputs

- What to verify: Empty input arrays yield `RESULTS.empty_subset` (not an
  exception).
- Inputs:
  - `beta_hat = np.array([], dtype=float)`; `s2 = np.array([], dtype=float)`.
  - `config = InferenceConfig(num_clusters=3)`.
- Action: Call `prepare_fit_state(beta_hat, s2, config)`.
- Expected result: `sol.result == RESULTS.empty_subset`; `sol.value is None`;
  no exception raised.

### pi-only-refactor.AC1.6 — `fit_baseline` surfaces max-steps as a status

- What to verify: With `max_iter=1`, the solver should not raise; it should
  return a `Solution` whose `.result == RESULTS.max_steps_reached`.
- Inputs:
  - Same synthetic dataset as AC1.2.
  - `config = InferenceConfig(num_clusters=5, max_iter=1, tol=1e-12)` (tiny
    tol forces max-iter termination on a non-trivial problem).
- Action: `prepare_fit_state(...)` then `fit_baseline(state.value, config)`.
- Expected result: Call returns cleanly;
  `baseline.result == RESULTS.max_steps_reached`; `baseline.value` is a
  `Params` object (the last iterate) with `pi` still summing approximately
  to 1.

---

## AC2 — Module layout matches target structure

Verifies the on-disk reorganization. Each check is a one-line import in a
fresh Python process.

### pi-only-refactor.AC2.1 — `mut_var.types` exposes shared contracts

- What to verify: `from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig` succeeds.
- Action:
  `python -c "from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig; print('OK')"`.
- Expected result: Prints `OK`; exit code 0; no `ImportError` or
  `ModuleNotFoundError`.

### pi-only-refactor.AC2.2 — `mut_var.io` exposes ingress surface

- What to verify: `from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays` succeeds.
- Action:
  `python -c "from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays; print('OK')"`.
- Expected result: Prints `OK`; exit code 0.

### pi-only-refactor.AC2.3 — `mut_var.pipelines` exposes pipeline API

- What to verify: `from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline` succeeds.
- Action:
  `python -c "from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline; print('OK')"`.
- Expected result: Prints `OK`; exit code 0.

### pi-only-refactor.AC2.4 — `mut_var.numerics.mixture_fit` exposes new numerics surface

- What to verify: `from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params` succeeds.
- Action:
  `python -c "from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params; print('OK')"`.
- Expected result: Prints `OK`; exit code 0.

### pi-only-refactor.AC2.5 — `mut_var.contracts` is gone

- What to verify: `import mut_var.contracts` raises `ModuleNotFoundError`.
- Action:
  `python -c "import mut_var.contracts" ; echo $?`.
- Expected result: Non-zero exit (typically 1); traceback contains
  `ModuleNotFoundError: No module named 'mut_var.contracts'`.

### pi-only-refactor.AC2.6 — `mut_var.adapters` is gone

- What to verify: `import mut_var.adapters` raises `ModuleNotFoundError`.
- Action: `python -c "import mut_var.adapters" ; echo $?`.
- Expected result: Non-zero exit; traceback contains
  `ModuleNotFoundError: No module named 'mut_var.adapters'`.

### pi-only-refactor.AC2.7 — `mut_var.numerics.baseline` is gone

- What to verify: `import mut_var.numerics.baseline` raises
  `ModuleNotFoundError`.
- Action: `python -c "import mut_var.numerics.baseline" ; echo $?`.
- Expected result: Non-zero exit; traceback contains
  `ModuleNotFoundError: No module named 'mut_var.numerics.baseline'`.

### pi-only-refactor.AC2.8 — `mut_var.numerics.refit` is gone

- What to verify: `import mut_var.numerics.refit` raises
  `ModuleNotFoundError`.
- Action: `python -c "import mut_var.numerics.refit" ; echo $?`.
- Expected result: Non-zero exit; traceback contains
  `ModuleNotFoundError: No module named 'mut_var.numerics.refit'`.

---

## AC3 — Public API surface and CLI updated correctly

Verifies end-to-end public contracts for the Python API and the `mutvar` CLI.

### pi-only-refactor.AC3.1 — `run_inference_pipeline(path, ...)` returns the canonical long-format DataFrame

- What to verify: The pipeline entrypoint accepts a path string and returns a
  `pl.DataFrame` containing (at minimum) the columns `mu0`, `var0`, `maf`,
  `name`, `value`.
- Inputs:
  - `path = "tests/fixtures/sumstats_valid.tsv"` (or any valid fixture TSV).
  - `config = InferenceConfig(num_clusters=3, max_iter=5)`.
  - Grid: `lowest=1e-3`, `highest=5e-3`, `num_breaks=2`.
- Action (Python):
  ```python
  from mut_var.pipelines import run_inference_pipeline
  from mut_var.types import InferenceConfig
  df = run_inference_pipeline(
      "tests/fixtures/sumstats_valid.tsv",
      lowest=1e-3, highest=5e-3, num_breaks=2,
      config=InferenceConfig(num_clusters=3, max_iter=5),
  )
  ```
- Expected result: `df` is a `polars.DataFrame` with `df.height > 0` and
  `set(["mu0", "var0", "maf", "name", "value"]).issubset(df.columns)`.

### pi-only-refactor.AC3.2 — No intermediate DataFrame retained past ingress

- What to verify: After `load_inference_arrays` returns, the pipeline holds
  only `InferenceArrays` (numpy-backed) in memory, never the original
  `pl.DataFrame`.
- Inputs: Same as AC3.1.
- Action: Code review of `src/mut_var/pipelines/inference.py` — the body of
  `run_inference_pipeline` must:
  1. Call `load_inference_arrays(path, ...)` at the top and assign to a single
     local (`arrays`).
  2. Never bind a `pl.DataFrame` returned by `read_sumstats` to a long-lived
     local.
  3. Pass only `arrays.beta_hat`, `arrays.s2`, `arrays.af` to subsequent
     numerics/mask/payload calls.
  Optionally verify at runtime with `gc.get_referrers(pl.DataFrame)`-style
  inspection or by adding a temporary `del` assertion while debugging, but
  the primary verification is source-level.
- Expected result: No `pl.DataFrame` local variable exists in
  `run_inference_pipeline` between ingress and numerics. The only surviving
  references after ingress are to `arrays` and grid/config scalars.

### pi-only-refactor.AC3.3 — `mutvar infer` exits 0 on supported flag set

- What to verify: The command-line surface accepts the retained flags (`-k`,
  `-m`, `-f`, `--lowest`, `--highest`, `--num-breaks`) and produces a
  zero-exit success.
- Inputs:
  - A valid TSV path (copy `tests/fixtures/sumstats_valid.tsv` to
    `/tmp/sumstats_valid.tsv`).
  - A writable output path (e.g. `/tmp/out.tsv`).
- Action:
  ```
  mutvar infer /tmp/sumstats_valid.tsv \
      -k 10 -m 50 -f 1e-6 \
      --lowest 1e-5 --highest 1e-2 --num-breaks 5 \
      -o /tmp/out.tsv
  echo $?
  ```
- Expected result: Exit code `0`; `/tmp/out.tsv` exists and is a non-empty TSV
  with the long-format columns from AC3.1.

### pi-only-refactor.AC3.4 — `mutvar curve` and `mutvar simulate` exit 0

- What to verify: Both other subcommands work end-to-end.
- Inputs:
  - For `curve`: a prior `mutvar infer` output TSV (reuse `/tmp/out.tsv`
    from AC3.3).
  - For `simulate`: `--output-prefix /tmp/sim`.
- Action:
  ```
  mutvar curve /tmp/out.tsv --fit-only
  echo $?

  mutvar simulate --output-prefix /tmp/sim
  echo $?
  ls /tmp/sim.truth.tsv /tmp/sim.observed.tsv /tmp/sim.meta.tsv
  ```
- Expected result:
  - `mutvar curve` exits 0.
  - `mutvar simulate` exits 0 and produces the three artifacts
    (`/tmp/sim.truth.tsv`, `/tmp/sim.observed.tsv`, `/tmp/sim.meta.tsv`).

### pi-only-refactor.AC3.5 — `mutvar infer --step-size` is rejected

- What to verify: The removed `--step-size` / `-r` flag is no longer
  recognized; argparse returns exit code 2.
- Inputs: Any valid TSV path.
- Action:
  ```
  mutvar infer /tmp/sumstats_valid.tsv --step-size 0.1
  echo $?
  ```
- Expected result: Exit code `2`; stderr contains `unrecognized arguments:
  --step-size`.

### pi-only-refactor.AC3.6 — `mutvar infer --seed` is rejected

- What to verify: The removed `--seed` / `-s` flag is rejected.
- Inputs: Any valid TSV path.
- Action:
  ```
  mutvar infer /tmp/sumstats_valid.tsv --seed 42
  echo $?
  ```
- Expected result: Exit code `2`; stderr contains `unrecognized arguments:
  --seed`.

### pi-only-refactor.AC3.7 — `mutvar infer` on a missing file exits 2

- What to verify: Missing input paths are caught at ingress and mapped to
  exit code 2.
- Inputs: A path that does not exist, e.g. `/tmp/does_not_exist.tsv`.
- Action:
  ```
  mutvar infer /tmp/does_not_exist.tsv
  echo $?
  ```
- Expected result: Exit code `2`; stderr includes a message identifying the
  missing file (e.g. `FileNotFoundError` or an equivalent user-facing error
  mentioning the path).

### pi-only-refactor.AC3.8 — Python API raises `FileNotFoundError` for missing paths

- What to verify: The Python-facing `run_inference_pipeline` raises a plain
  `FileNotFoundError` rather than swallowing the error or raising a custom
  type.
- Inputs: `"missing.tsv"`.
- Action (Python):
  ```python
  from mut_var.pipelines import run_inference_pipeline
  try:
      run_inference_pipeline("missing.tsv", lowest=1e-3, highest=5e-3, num_breaks=2)
  except FileNotFoundError as exc:
      print("OK:", exc)
  ```
- Expected result: The `FileNotFoundError` branch is taken; the exception type
  is `builtins.FileNotFoundError` (not a subclass of a custom error).

---

## AC4 — Quality gate passes end-to-end

Verifies the three canonical repo checks. These should be run from the repo
root.

### pi-only-refactor.AC4.1 — `ruff check` clean

- What to verify: No lint violations remain in source or tests.
- Inputs: Repository working tree at tip of the refactor branch.
- Action: `ruff check src/mut_var tests ; echo $?`.
- Expected result: Exit code `0`; no violations reported.

### pi-only-refactor.AC4.2 — `mypy` clean

- What to verify: No type errors after the reorganization and API changes.
- Inputs: Same working tree.
- Action: `mypy src/mut_var tests ; echo $?`.
- Expected result: Exit code `0`; `Success: no issues found`.

### pi-only-refactor.AC4.3 — `pytest` green

- What to verify: All automated tests (existing regressions plus the new
  `tests/test_infer_opt.py` coverage for `prepare_fit_state` / `fit_baseline` /
  `fit_refit_step`) pass under the new module layout.
- Inputs: Same working tree.
- Action: `pytest -p no:capture ; echo $?`.
- Expected result: Exit code `0`; test summary shows `passed` with no failures,
  errors, or collection failures (in particular, no `ModuleNotFoundError` from
  residual imports of `mut_var.contracts`, `mut_var.adapters`,
  `mut_var.numerics.baseline`, or `mut_var.numerics.refit`).
