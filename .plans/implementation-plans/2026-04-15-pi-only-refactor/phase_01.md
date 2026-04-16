# Pi-Only Refactor Implementation Plan — Phase 1

**Goal:** Centralize shared type contracts in `types.py` and consolidate all ingress logic into `io.py`. Remove `contracts.py` and `adapters/`. No algorithm changes.

**Architecture:** Pure reorganization. New `types.py` holds `RESULTS`, `Solution`, `InferenceConfig`, `SimulationPipelineConfig`. Expanded `io.py` absorbs `InferenceArrays` and the three adapter functions, plus a new `load_inference_arrays` convenience entrypoint. Existing logic is unchanged; only import paths move.

**Tech Stack:** Python, equinox, jax, polars — all already in-tree.

**Scope:** Phase 1 of 4 from design plan.

**Codebase verified:** 2026-04-15

---

## Acceptance Criteria Coverage

This phase implements and tests:

### pi-only-refactor.AC2: Module layout matches the target structure

- **pi-only-refactor.AC2.1 Success:** `from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig` succeeds.
- **pi-only-refactor.AC2.2 Success:** `from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays` succeeds.
- **pi-only-refactor.AC2.5 Failure:** `import mut_var.contracts` raises `ModuleNotFoundError` (module deleted).
- **pi-only-refactor.AC2.6 Failure:** `import mut_var.adapters` raises `ModuleNotFoundError` (directory deleted).

### pi-only-refactor.AC4: Quality gate passes end-to-end

- **pi-only-refactor.AC4.1 Success:** `ruff check src/mut_var tests` exits with code 0.
- **pi-only-refactor.AC4.2 Success:** `mypy src/mut_var tests` exits with code 0 (no type errors).
- **pi-only-refactor.AC4.3 Success:** `pytest -p no:capture` exits with code 0 (all tests pass).

---

## Design Adjustments

The following deviations from the design spec are intentional for Phase 1:

1. **`InferenceConfig` keeps `step_size` and `penalty` fields.** The design lists `InferenceConfig(num_clusters, max_iter, tol, filter_threshold)` but Phase 1 is "no logic changes, only reorganization." The current `InferenceConfig` has `step_size` and `penalty` actively used by `baseline.py` and `refit.py` via `to_baseline_config()`/`to_refit_config()`. Removing these fields in Phase 1 would break existing numerics. Phase 2 strips them when merging `baseline.py`/`refit.py` into `mixture_fit.py`.

2. **`to_inference_arrays` keeps JAX array return type.** The design says `io.py` should return numpy arrays. Phase 1 keeps JAX to avoid breaking `baseline.py`/`refit.py` which consume JAX arrays directly. Phase 2 changes the return to numpy when `mixture_fit.py` adds internal `jnp.asarray()` conversion.

3. **`SimulationArtifacts` stays in `simulate.py`.** The design moves it to `pipelines/simulation.py` in Phase 3. Phase 1 does not touch it.

---

## Codebase Verification Findings

- ✓ `contracts.py` has exactly `RESULTS` and `Solution` — 15 files import from it (9 source, 6 test).
- ✓ `adapters/tabular.py` has `to_inference_arrays`, `build_maf_masks`, `payload_to_long_dataframe`.
- ✓ `InferenceArrays` and `InferenceConfig` are defined in `infer.py` (not `contracts.py`).
- ✓ `SimulationPipelineConfig` is defined in `simulate.py`.
- + `adapters/tabular.py` imports `InferenceArrays` from `mut_var.infer` — circular dependency resolved by moving `InferenceArrays` to `io.py`.
- + `numerics/__init__.py` re-exports `InferenceArrays, InferenceConfig` from `mut_var.infer` — needs updating to new locations.
- + `infer.py` imports adapters lazily (local import inside function body at line 191) — replace with top-level import from `mut_var.io`.
- + `test_infer.py` imports `to_inference_arrays` from `mut_var.adapters.tabular` — update to `mut_var.io`.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Create `src/mut_var/types.py`

**Verifies:** pi-only-refactor.AC2.1

**Files:**
- Create: `src/mut_var/types.py`

**Implementation:**

Copy `RESULTS` and `Solution` verbatim from `src/mut_var/contracts.py` (lines 1–28). Then copy `InferenceConfig` from `src/mut_var/infer.py` (lines 32–60, including `to_baseline_config` and `to_refit_config` methods with their lazy internal imports). Finally copy `SimulationPipelineConfig` from `src/mut_var/simulate.py` (lines 18–24).

The resulting `types.py` module pattern comment is `# pattern: Functional Core`.

Required imports at the top:
```python
from __future__ import annotations

# pattern: Functional Core
from typing import Any, NamedTuple, TYPE_CHECKING

import equinox as eqx
import equinox.internal as eqxi

if TYPE_CHECKING:
    from mut_var.numerics.baseline import BaselineConfig
    from mut_var.numerics.refit import RefitConfig
    from mut_var.numerics.simulate import SimulationNumericsConfig
```

**`SimulationPipelineConfig` — avoid runtime import cycle.**

`numerics/simulate.py` imports `RESULTS`/`Solution` from `mut_var.types` (after Task 3). If `types.py` also imports `SimulationNumericsConfig` from `mut_var.numerics.simulate` at module level, a circular import results. Avoid this by:

- Keeping `SimulationNumericsConfig` under `TYPE_CHECKING` only (annotation purposes).
- Changing the `numerics` field default from `SimulationNumericsConfig(...)` to `None`.

The resulting `SimulationPipelineConfig` in `types.py`:
```python
class SimulationPipelineConfig(NamedTuple):
    n_rows: int
    seed: int = 0
    numerics: SimulationNumericsConfig | None = None
```

In `src/mut_var/simulate.py` (Task 3), update `run_simulation_pipeline` to handle `config.numerics is None`:
```python
from mut_var.numerics.simulate import SimulationNumericsConfig
effective_numerics = config.numerics if config.numerics is not None else SimulationNumericsConfig()
```
Use `effective_numerics` wherever `config.numerics` was previously accessed. All CLI and test callers always pass `numerics=...` explicitly, so this only affects the default-construction path.


**Verification:**

```bash
python -c "from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig; print('OK')"
```
Expected: prints `OK` with no errors.

**Commit:** `refactor: add types.py with RESULTS, Solution, InferenceConfig, SimulationPipelineConfig`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Expand `src/mut_var/io.py` with `InferenceArrays`, adapter functions, and `load_inference_arrays`

**Verifies:** pi-only-refactor.AC2.2

**Files:**
- Modify: `src/mut_var/io.py`

**Implementation:**

Add the following to `src/mut_var/io.py` after the existing imports and before `read_sumstats`. The file currently has only validation and read functions. Add:

New imports needed at top of `io.py`:
```python
from typing import Mapping, NamedTuple
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
```

**`InferenceArrays` NamedTuple** (copied from `src/mut_var/infer.py` lines 26–29):
```python
class InferenceArrays(NamedTuple):
    af: ArrayLike
    beta_hat: ArrayLike
    s2: ArrayLike
```
Place this after imports, before `read_sumstats`.

**Three adapter functions** (copied verbatim from `src/mut_var/adapters/tabular.py`):
- `to_inference_arrays(df, af_col, beta_col, se_col) -> InferenceArrays`
- `build_maf_masks(af, maf_grid) -> jax.Array`
- `payload_to_long_dataframe(payload) -> pl.DataFrame`

These keep their existing JAX array return types (numpy change is Phase 2).

**New `load_inference_arrays` convenience function:**
```python
def load_inference_arrays(
    path: str,
    *,
    af_col: str = "effect_allele_frequency",
    beta_col: str = "beta",
    se_col: str = "standard_error",
) -> InferenceArrays:
    r"""Load and validate summary statistics from a TSV file, returning numeric arrays.

    **Arguments:**

    - `path`: Input TSV path.
    - `af_col`: Effect-allele-frequency column name.
    - `beta_col`: Effect-size column name.
    - `se_col`: Standard-error column name.

    **Returns:**

    - `InferenceArrays` with JAX arrays for AF, beta, and variance (`se^2`).

    **Raises:**

    - `FileNotFoundError`: Path does not exist.
    - `ValueError`: Validation failed (missing columns, non-numeric values, domain violations).
    """
    df = read_sumstats(path)
    validate_required_columns(df, af_col, beta_col, se_col)
    validate_numeric_columns(df, af_col, beta_col, se_col)
    validate_sumstats_domain(df, af_col, se_col)
    return to_inference_arrays(df, af_col, beta_col, se_col)
```

Update `io.py`'s module-level comment to `# pattern: Imperative Shell` (it already has this).

**Verification:**

```bash
python -c "from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays; print('OK')"
```
Expected: prints `OK` with no errors.

**Commit:** `refactor: expand io.py with InferenceArrays, adapter functions, load_inference_arrays`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

<!-- START_SUBCOMPONENT_B (tasks 3-5) -->

<!-- START_TASK_3 -->
### Task 3: Update imports in all 9 source files — `mut_var.contracts` → `mut_var.types`

**Verifies:** prerequisite for AC2.5

**Files:**
- Modify: `src/mut_var/simulate.py`
- Modify: `src/mut_var/curve.py`
- Modify: `src/mut_var/numerics/baseline.py`
- Modify: `src/mut_var/numerics/refit.py`
- Modify: `src/mut_var/numerics/curve_fit.py`
- Modify: `src/mut_var/numerics/simulate.py`
- Modify: `src/mut_var/numerics/_optimistix_solver.py`
- Modify: `src/mut_var/numerics/_solver_utils.py`
- Modify: `src/mut_var/infer.py`

**Implementation:**

In each of the 8 files listed above (excluding `infer.py` for now), change:
```python
from mut_var.contracts import RESULTS, Solution
# or
from mut_var.contracts import RESULTS
```
to:
```python
from mut_var.types import RESULTS, Solution
# or
from mut_var.types import RESULTS
```

For `src/mut_var/simulate.py`: additionally:
1. Remove the `SimulationPipelineConfig` class definition (lines 18–24).
2. Add an import from `mut_var.types` so it remains importable via `mut_var.simulate.SimulationPipelineConfig` (required by `mut_var/__init__.py` until Phase 4):
   ```python
   from mut_var.types import SimulationPipelineConfig
   ```
3. In `run_simulation_pipeline`, handle `config.numerics is None` by constructing a default locally:
   ```python
   from mut_var.numerics.simulate import SimulationNumericsConfig
   effective_numerics = config.numerics if config.numerics is not None else SimulationNumericsConfig()
   ```
   Replace all subsequent uses of `config.numerics` with `effective_numerics`.
(Keep `SimulationArtifacts` defined in `simulate.py` — it moves in Phase 3.)

For `src/mut_var/infer.py`:
1. Change `from mut_var.contracts import RESULTS, Solution` → `from mut_var.types import RESULTS, Solution, InferenceConfig`
2. Remove the `InferenceArrays` class definition (lines 26–29)
3. Remove the `InferenceConfig` class definition and its methods (lines 32–60)
4. Add at the top-level imports: `from mut_var.io import InferenceArrays, build_maf_masks, payload_to_long_dataframe, to_inference_arrays`
5. Remove the lazy local import inside the function body (line 191): `from mut_var.adapters.tabular import build_maf_masks, payload_to_long_dataframe, to_inference_arrays`
6. Keep `"InferenceArrays"` and `"InferenceConfig"` in `infer.py`'s `__all__` (they are still re-exported for backward compat via the new imports).

For `src/mut_var/numerics/__init__.py`:
- Change `from mut_var.infer import InferenceArrays, InferenceConfig` to:
  ```python
  from mut_var.io import InferenceArrays
  from mut_var.types import InferenceConfig
  ```

**Verification:**

```bash
python -c "import mut_var.infer; import mut_var.simulate; import mut_var.curve; import mut_var.numerics; print('OK')"
```
Expected: all import cleanly.

**Commit:** `refactor: update all source-file imports to mut_var.types and mut_var.io`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Update imports in all 6 test files — `mut_var.contracts` → `mut_var.types`

**Verifies:** prerequisite for AC4.3

**Files:**
- Modify: `tests/test_contracts.py`
- Modify: `tests/test_curve.py`
- Modify: `tests/test_simulate.py`
- Modify: `tests/test_infer.py`
- Modify: `tests/test_simulate_numerics.py`
- Modify: `tests/test_infer_opt.py`

**Implementation:**

In each file change:
```python
from mut_var.contracts import RESULTS, Solution
```
to:
```python
from mut_var.types import RESULTS, Solution
```

For `tests/test_infer.py` additionally:
1. Change `from mut_var.adapters.tabular import to_inference_arrays` → `from mut_var.io import to_inference_arrays`
2. The import `from mut_var.infer import InferenceConfig, run_inference_pipeline as run_inference_dataframe_pipeline` stays valid because `infer.py` still re-exports `InferenceConfig`.
3. On line 169: `assert infer_module.InferenceArrays is numerics.InferenceArrays` — both now import from `mut_var.io`, so the identity check still holds. No change needed.

**Verification:**

```bash
ruff check src/mut_var tests
```
Expected: exits code 0.

**Commit:** `refactor: update test imports from mut_var.contracts to mut_var.types`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Delete `contracts.py` and `adapters/`; run full quality gate

**Verifies:** pi-only-refactor.AC2.5, pi-only-refactor.AC2.6, pi-only-refactor.AC4.1, pi-only-refactor.AC4.2, pi-only-refactor.AC4.3

**Files:**
- Delete: `src/mut_var/contracts.py`
- Delete: `src/mut_var/adapters/tabular.py`
- Delete: `src/mut_var/adapters/__init__.py`
- Delete: `src/mut_var/adapters/` (directory)

**Implementation:**

```bash
rm src/mut_var/contracts.py
rm -r src/mut_var/adapters/
```

Verify deletion is clean (no remaining imports):
```bash
grep -r "mut_var.contracts\|mut_var.adapters" src/mut_var tests --include="*.py"
```
Expected: no output.

**Verification — module layout:**
```bash
python -c "
from mut_var.types import RESULTS, Solution, InferenceConfig, SimulationPipelineConfig
from mut_var.io import load_inference_arrays, to_inference_arrays, build_maf_masks, payload_to_long_dataframe, InferenceArrays
print('imports OK')
"
python -c "
try:
    import mut_var.contracts
    print('FAIL: contracts still importable')
except ModuleNotFoundError:
    print('AC2.5 OK: contracts deleted')
"
python -c "
try:
    import mut_var.adapters
    print('FAIL: adapters still importable')
except ModuleNotFoundError:
    print('AC2.6 OK: adapters deleted')
"
```

**Verification — full quality gate:**

```bash
ruff check src/mut_var tests
```
Expected: exits code 0.

```bash
mypy src/mut_var tests
```
Expected: exits code 0.

```bash
pytest -p no:capture
```
Expected: all tests pass.

**Commit:** `refactor: delete contracts.py and adapters/; types and io consolidation complete`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->
