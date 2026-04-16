# Pi-Only Refactor Implementation Plan — Phase 3

**Goal:** Introduce `pipelines/` subpackage. Move orchestration modules into it. Change `run_inference_pipeline` signature from `df: pl.DataFrame` to `path: str`. Delete top-level `infer.py`, `curve.py`, `simulate.py`. Update `cli.py` to use path-based invocation.

**Architecture:** Three new modules (`pipelines/inference.py`, `pipelines/curve.py`, `pipelines/simulation.py`) plus a `pipelines/__init__.py`. The only behavioral change is `run_inference_pipeline` accepting a path string and calling `io.load_inference_arrays` at ingress instead of receiving a pre-read DataFrame. All other logic moves verbatim. `cli.py` import paths update and the now-removed `InferenceConfig` fields (`step_size`, `penalty`) are dropped from the constructor call (flags retained in argparse until Phase 4).

**Tech Stack:** Python, JAX, Polars — all in-tree.

**Scope:** Phase 3 of 4 from design plan.

**Codebase verified:** 2026-04-15

---

## Acceptance Criteria Coverage

### pi-only-refactor.AC2: Module layout matches the target structure

- **pi-only-refactor.AC2.3 Success:** `from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline` succeeds.

### pi-only-refactor.AC3: Public API surface and CLI updated correctly

- **pi-only-refactor.AC3.1 Success:** `run_inference_pipeline(path, ...)` (path string) returns a long-format `pl.DataFrame` with columns `mu0`, `var0`, `maf`, `name`, `value` for a valid TSV path.
- **pi-only-refactor.AC3.2 Success:** `run_inference_pipeline` does not retain a `pl.DataFrame` in memory after `load_inference_arrays` returns — only `InferenceArrays` (numpy) persists through the numerics stage.
- **pi-only-refactor.AC3.4 Success:** `mutvar curve <path>` and `mutvar simulate --output-prefix foo` exit with code 0 using valid inputs.
- **pi-only-refactor.AC3.8 Failure:** `run_inference_pipeline("missing.tsv", ...)` raises `FileNotFoundError`.

### pi-only-refactor.AC4: Quality gate passes end-to-end

- **pi-only-refactor.AC4.1 Success:** `ruff check src/mut_var tests` exits with code 0.
- **pi-only-refactor.AC4.2 Success:** `mypy src/mut_var tests` exits with code 0 (no type errors).
- **pi-only-refactor.AC4.3 Success:** `pytest -p no:capture` exits with code 0 (all tests pass).

---

## Design Adjustments

1. **`cli.py` partially updated here.** Phase 3 changes cli.py to import from `pipelines/` and to pass `args.sumstats` (path string) directly to `run_inference_pipeline`. The CLI flags `--step-size`, `--penalty`, `--seed`, `--maf-threshold` are removed from argparse in Phase 4 — they remain in Phase 3 but their values are silently ignored when constructing `InferenceConfig`.

2. **`SimulationArtifacts` defined in `pipelines/simulation.py`.** Consistent with design. `__init__.py` re-exports it.

3. **`run_curve_pipeline` and `run_simulation_pipeline` are content-identical copies** (only import paths change). No logic changes.

---

## Codebase Verification Findings

- ✓ `pipelines/` directory already exists (empty, with `__pycache__`).
- ✓ `conftest.py` has `sumstats_valid_path` fixture returning a path string — tests switch from `sumstats_valid_df` to this.
- ✓ `run_curve_pipeline` already accepts `input_path: str`; tests already pass `str(path)`.
- ✓ `run_simulation_pipeline` uses config-based invocation; no signature change.
- + `cli.py` imports from `mut_var.curve`, `mut_var.infer`, `mut_var.simulate` (lines 13–17) — all update.
- + `cli.py` calls `read_sumstats(args.sumstats)` then passes `df` to pipeline — Phase 3 removes this; passes path directly.
- + `benchmarks/infer_runtime.py` also imports from `mut_var.infer` — update import but outside quality gate scope.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Create `src/mut_var/pipelines/` — all three pipeline modules and `__init__.py`

**Verifies:** pi-only-refactor.AC2.3

**Files:**
- Create: `src/mut_var/pipelines/__init__.py`
- Create: `src/mut_var/pipelines/inference.py`
- Create: `src/mut_var/pipelines/curve.py`
- Create: `src/mut_var/pipelines/simulation.py`

**Implementation:**

**`src/mut_var/pipelines/__init__.py`:**
```python
from mut_var.pipelines.curve import run_curve_pipeline
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.pipelines.simulation import run_simulation_pipeline, SimulationArtifacts

__all__ = [
    "run_curve_pipeline",
    "run_inference_pipeline",
    "run_simulation_pipeline",
    "SimulationArtifacts",
]
```

**`src/mut_var/pipelines/curve.py`:**

Copy `src/mut_var/curve.py` verbatim. Change the single import:
```python
# Before:
from mut_var.contracts import RESULTS
# After:
from mut_var.types import RESULTS
```
All other content is identical.

**`src/mut_var/pipelines/simulation.py`:**

Copy `src/mut_var/simulate.py` verbatim. Change imports:
```python
# Before:
from mut_var.contracts import RESULTS, Solution
# After:
from mut_var.types import RESULTS, Solution, SimulationPipelineConfig
```
Remove the `class SimulationPipelineConfig(NamedTuple)` definition (now imported from `types`). All other content is identical (keep `SimulationArtifacts` defined here).

Update `__all__` at bottom to reflect new location:
```python
__all__ = [
    "SimulationArtifacts",
    "run_simulation_pipeline",
]
```

**`src/mut_var/pipelines/inference.py`:**

This is a modified version of `src/mut_var/infer.py` with the following changes:

1. **Signature change:** `df: pl.DataFrame` → `path: str`
2. **Ingress change:** Replace the inline validation + `to_inference_arrays` block with `io.load_inference_arrays(path, ...)`. The `validate_maf_grid` call stays (it validates grid params, not the data).
3. **Remove `seed` parameter** (removed in Phase 2).
4. **Import changes:**
   - Keep: `import polars as pl` (needed for the return type annotation `-> pl.DataFrame`)
   - Change: `from mut_var.types import RESULTS, Solution, InferenceConfig` (was `mut_var.contracts`)
   - Change: `from mut_var.io import InferenceArrays, build_maf_masks, load_inference_arrays, payload_to_long_dataframe, validate_maf_grid` (was `mut_var.adapters.tabular` + `mut_var.io`)
   - Change: `from mut_var.numerics.mixture_fit import fit_baseline, fit_refit_step, prepare_fit_state` (Phase 2 API)
5. **Remove `rdm` import** (no seed).
6. **Keep all helper functions:** `_filter_components`, `_build_long_payload`, `_payload_from_solution`, `_reason_from_solution`, `_solver_debug_callback`.

The updated `run_inference_pipeline` signature:
```python
def run_inference_pipeline(
    path: str,
    *,
    af_col: str = "effect_allele_frequency",
    beta_col: str = "beta",
    se_col: str = "standard_error",
    lowest: float = 1e-5,
    highest: float = 1e-2,
    num_breaks: int = 10,
    config: InferenceConfig | None = None,
    log: logging.Logger | None = None,
) -> pl.DataFrame:
```

The ingress section becomes:
```python
workflow_log.info("inference pipeline: validating grid parameters")
validate_maf_grid(lowest, highest, num_breaks)

workflow_log.info("inference pipeline: loading input data from '%s'", path)
arrays = load_inference_arrays(path, af_col=af_col, beta_col=beta_col, se_col=se_col)
workflow_log.info("inference pipeline: input loaded and validated")
```

(Removes the previous `validate_required_columns`, `validate_numeric_columns`, `validate_sumstats_domain`, and `to_inference_arrays` calls — these are now inside `load_inference_arrays`.)

All subsequent pipeline logic is identical to the post-Phase-2 `infer.py`.

`__all__`:
```python
__all__ = [
    "InferenceArrays",
    "InferenceConfig",
    "run_inference_pipeline",
]
```

**Verification:**

```bash
python -c "
from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline
print('AC2.3 OK: pipelines imports OK')
"
```

**Commit:** `feat: add pipelines/ subpackage with inference.py, curve.py, simulation.py`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `src/mut_var/__init__.py` and `src/mut_var/cli.py`; delete top-level orchestration modules

**Verifies:** pi-only-refactor.AC3.1, AC3.4, AC3.8

**Files:**
- Modify: `src/mut_var/__init__.py`
- Modify: `src/mut_var/cli.py`
- Delete: `src/mut_var/infer.py`
- Delete: `src/mut_var/curve.py`
- Delete: `src/mut_var/simulate.py`
- Modify (optional): `benchmarks/infer_runtime.py`

**Implementation:**

**`src/mut_var/__init__.py`** — update the three pipeline imports (lines 8–10):
```python
# Before:
from .curve import run_curve_pipeline
from .infer import run_inference_pipeline
from .simulate import run_simulation_pipeline, SimulationArtifacts, SimulationPipelineConfig

# After:
from .pipelines.curve import run_curve_pipeline
from .pipelines.inference import run_inference_pipeline
from .pipelines.simulation import run_simulation_pipeline, SimulationArtifacts
from .types import SimulationPipelineConfig
```

The `__all__` list stays unchanged.

**`src/mut_var/cli.py`** — three changes:

1. Update imports (lines 13–17):
```python
# Before:
from mut_var.curve import run_curve_pipeline
from mut_var.infer import InferenceConfig, run_inference_pipeline
from mut_var.io import read_sumstats
from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig

# After:
from mut_var.pipelines.curve import run_curve_pipeline
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.pipelines.simulation import run_simulation_pipeline
from mut_var.types import InferenceConfig, SimulationPipelineConfig
```
Remove the `from mut_var.io import read_sumstats` import (no longer needed in CLI; path goes directly to pipeline).

2. Update the `run_infer` handler — replace `read_sumstats` + `run_inference_pipeline(df, ...)` with path-based invocation. Remove `seed=args.seed`. Remove `step_size` and `penalty` from `InferenceConfig` construction (these fields no longer exist in `InferenceConfig` after Phase 2). The CLI flags `--step-size`, `--penalty`, `--seed`, `--maf-threshold` stay in argparse until Phase 4 (they will be orphaned in Phase 3 but cause no test failures since tests don't pass them):

```python
# Before (in run_infer handler):
log.info("infer: loading data from '%s'", args.sumstats)
df = read_sumstats(args.sumstats)
log.info("infer: data loaded (%d rows)", df.height)

log.info("infer: starting inference pipeline")
result_df = run_inference_pipeline(
    df,
    af_col=args.af_col,
    beta_col=args.beta_col,
    se_col=args.se_col,
    lowest=args.lowest,
    highest=args.highest,
    num_breaks=args.num_breaks,
    seed=args.seed,
    config=InferenceConfig(
        num_clusters=args.num_clusters,
        max_iter=args.max_iter,
        step_size=args.step_size,
        filter_threshold=args.filter,
        penalty=args.penalty,
    ),
    log=log,
)

# After:
log.info("infer: starting inference pipeline")
result_df = run_inference_pipeline(
    args.sumstats,
    af_col=args.af_col,
    beta_col=args.beta_col,
    se_col=args.se_col,
    lowest=args.lowest,
    highest=args.highest,
    num_breaks=args.num_breaks,
    config=InferenceConfig(
        num_clusters=args.num_clusters,
        max_iter=args.max_iter,
        filter_threshold=args.filter,
    ),
    log=log,
)
```

3. Also update the `test_cli_contracts.py` import patch (if not already fixed in Phase 2): verify that the monkeypatch target `mut_var.numerics.baseline` import is replaced with `mut_var.numerics.mixture_fit`. Check and fix `tests/test_cli_contracts.py` line 8:
```python
# Before:
import mut_var.numerics.baseline as baseline_module
...
monkeypatch.setattr(baseline_module, "fit_baseline", _unexpected_call)

# After:
import mut_var.numerics.mixture_fit as mixture_fit_module
...
monkeypatch.setattr(mixture_fit_module, "prepare_fit_state", _unexpected_call)
```

**Delete top-level orchestration modules:**
```bash
rm src/mut_var/infer.py
rm src/mut_var/curve.py
rm src/mut_var/simulate.py
```

**Update `benchmarks/infer_runtime.py`** if it imports from `mut_var.infer`:
```python
# Before:
from mut_var.infer import InferenceConfig, run_inference_pipeline
# After:
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.types import InferenceConfig
```

**Verify no remaining references:**
```bash
grep -r "from mut_var\.infer\|from mut_var\.curve\|from mut_var\.simulate\|from \.infer\|from \.curve\|from \.simulate" src/mut_var tests --include="*.py"
```
Expected: no output (except possibly `benchmarks/` which is outside quality gate).

**Verification — pipeline API:**

```bash
python -c "
from tests.helpers import fixture_path
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.types import InferenceConfig

path = str(fixture_path('sumstats_valid.tsv'))
result = run_inference_pipeline(
    path, lowest=1e-3, highest=5e-3, num_breaks=2,
    config=InferenceConfig(num_clusters=3, max_iter=5),
)
import polars as pl
assert isinstance(result, pl.DataFrame)
assert result.height > 0
assert set(['mu0', 'var0', 'maf', 'name', 'value']).issubset(result.columns)
print('AC3.1 OK')
"
```

```bash
python -c "
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.types import InferenceConfig
try:
    run_inference_pipeline('missing.tsv', lowest=1e-3, highest=5e-3, num_breaks=2)
except FileNotFoundError:
    print('AC3.8 OK: FileNotFoundError raised for missing path')
"
```

**Full quality gate:**

```bash
ruff check src/mut_var tests
mypy src/mut_var tests
pytest -p no:capture
```
Expected: all pass.

**Commit:** `refactor: move orchestration to pipelines/; update __init__.py and cli.py`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->

<!-- START_TASK_3 -->
### Task 3: Update test imports — switch from top-level modules to pipelines/

**Verifies:** pi-only-refactor.AC4.3

**Files:**
- Modify: `tests/test_infer.py`
- Modify: `tests/test_simulate.py`
- Modify: `tests/test_curve.py` (if needed)

**Implementation:**

**`tests/test_infer.py`:**

1. Change top-level imports:
```python
# Before:
from mut_var.adapters.tabular import to_inference_arrays  # (already mut_var.io after Phase 1)
from mut_var.contracts import RESULTS, Solution  # (already mut_var.types after Phase 1)
from mut_var.infer import InferenceConfig, run_inference_pipeline as run_inference_dataframe_pipeline
from mut_var.numerics import SimulationNumericsConfig
from mut_var.simulate import run_simulation_pipeline, SimulationPipelineConfig

# After:
from mut_var.io import to_inference_arrays
from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig
from mut_var.pipelines.inference import run_inference_pipeline
from mut_var.pipelines.simulation import run_simulation_pipeline
from mut_var.numerics import SimulationNumericsConfig
```

2. Change all calls from `run_inference_dataframe_pipeline(sumstats_valid_df, ...)` to `run_inference_pipeline(sumstats_valid_path, ...)`. The `sumstats_valid_path` fixture already exists in `conftest.py`. Replace the `sumstats_valid_df` fixture parameter with `sumstats_valid_path` in each test function signature that currently uses it.

3. Line 169 (`test_numerics_module_owns_numerics_entrypoint`): update `infer_module` to reference `mut_var.pipelines.inference`:
```python
# Before:
import mut_var.infer as infer_module
# After:
import mut_var.pipelines.inference as infer_module
```
`pipelines/inference.py` re-exports `InferenceArrays` and `InferenceConfig` in its `__all__`, so the identity assertions still hold.

4. Rewrite `test_simulated_observed_output_is_accepted_by_run_inference_pipeline` (lines 174–224). After Phase 3, `run_inference_pipeline` accepts a path string, not a DataFrame. Write `artifacts.observed` to a temp TSV file first:
```python
def test_simulated_observed_output_is_accepted_by_run_inference_pipeline(monkeypatch, tmp_path):
    import mut_var.numerics.mixture_fit as mixture_fit_module

    fake_params = mixture_fit_module.Params(
        pi=jnp.asarray([0.9, 0.1], dtype=jnp.float64),
        mu_k=jnp.asarray([0.0], dtype=jnp.float64),
        var_k=jnp.asarray([1e-4], dtype=jnp.float64),
    )

    monkeypatch.setattr(
        mixture_fit_module,
        "fit_baseline",
        lambda *_args, **_kwargs: Solution(
            value=fake_params,
            result=RESULTS.successful,
            stats={"n_steps": 1},
            state=None,
        ),
    )
    monkeypatch.setattr(
        mixture_fit_module,
        "fit_refit_step",
        lambda *_args, **_kwargs: Solution(
            value=fake_params,
            result=RESULTS.successful,
            stats={"n_steps": 1},
            state=None,
        ),
    )

    artifacts = run_simulation_pipeline(
        config=SimulationPipelineConfig(
            n_rows=128,
            seed=0,
            numerics=SimulationNumericsConfig(weights=(0.9, 0.1), log_var_scales=(-8.0, -5.5)),
        )
    )

    observed_path = tmp_path / "observed.tsv"
    artifacts.observed.write_csv(observed_path, separator="\t")

    result_df = run_inference_pipeline(
        str(observed_path),
        lowest=1e-3,
        highest=5e-3,
        num_breaks=2,
        config=InferenceConfig(num_clusters=2, max_iter=5),
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.height > 0
    assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
```

**`tests/test_simulate.py`:**

Change imports:
```python
# Before:
from mut_var.simulate import run_simulation_pipeline, SimulationArtifacts, SimulationPipelineConfig

# After:
from mut_var.pipelines.simulation import run_simulation_pipeline, SimulationArtifacts
from mut_var.types import SimulationPipelineConfig
```

**`tests/test_curve.py`:**

Change imports:
```python
# Before:
from mut_var.curve import run_curve_pipeline

# After:
from mut_var.pipelines.curve import run_curve_pipeline
```

(The `from mut_var.contracts import RESULTS, Solution` change was already done in Phase 1.)

**Verification:**

```bash
pytest tests/test_infer.py tests/test_simulate.py tests/test_curve.py -p no:capture
```
Expected: all tests pass.

**Commit:** `refactor: update test imports to use pipelines/ and types`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Final quality gate

**Verifies:** pi-only-refactor.AC2.3, AC3.1, AC3.4, AC3.8, AC4.1, AC4.2, AC4.3

**Files:** No changes — verification only.

**Verification — module layout:**
```bash
python -c "
from mut_var.pipelines import run_inference_pipeline, run_curve_pipeline, run_simulation_pipeline
print('AC2.3 OK')
"
```

**Full quality gate:**
```bash
ruff check src/mut_var tests
```
Expected: code 0.

```bash
mypy src/mut_var tests
```
Expected: code 0.

```bash
pytest -p no:capture
```
Expected: all tests pass.

**Commit:** `chore: verify Phase 3 quality gate — pipelines reorganization complete`

(Only create this commit if there were any outstanding nits to fix during the verification run.)
<!-- END_TASK_4 -->

<!-- END_SUBCOMPONENT_B -->
