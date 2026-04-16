# Pi-Only Refactor Implementation Plan — Phase 2

**Goal:** Merge `baseline.py` and `refit.py` into `numerics/mixture_fit.py` with `pi` as the sole Optimistix optimization variable. Pre-compute the likelihood matrix once in `prepare_fit_state`. Remove dead mu/var manifold code and solver-control fields from `InferenceConfig`.

**Architecture:** New `mixture_fit.py` defines `FitState`/`Params` NamedTuples and three public functions (`prepare_fit_state`, `fit_baseline`, `fit_refit_step`). The Riemannian simplex solver (`MutVarSolver`, `simplex_tangent_direction`, `exponential_map_simplex`) is unchanged. `infer.py` is updated to call the new API. `baseline.py` and `refit.py` are deleted.

**Tech Stack:** JAX, Equinox, Optimistix — all in-tree.

**Scope:** Phase 2 of 4 from design plan.

**Codebase verified:** 2026-04-15

---

## Acceptance Criteria Coverage

### pi-only-refactor.AC1: Numerics simplified — pi is the sole optimization variable

- **pi-only-refactor.AC1.1 Success:** `prepare_fit_state` returns a `Solution` with `RESULTS.successful` and a `FitState` whose `likelihood_matrix` has shape `(n, K)` for valid inputs.
- **pi-only-refactor.AC1.2 Success:** `fit_baseline` converges on a synthetic dataset and returns `Params` where `sum(pi) == 1.0` (within float tolerance) and `mu_k`/`var_k` match the values set by `prepare_fit_state`.
- **pi-only-refactor.AC1.3 Success:** `fit_refit_step` returns `Params` with updated `pi` (sum 1) and unchanged `mu_k`/`var_k` relative to `prev_params`.
- **pi-only-refactor.AC1.4 Failure:** `prepare_fit_state` returns `RESULTS.invalid_input` when `s2` contains non-positive values.
- **pi-only-refactor.AC1.5 Failure:** `prepare_fit_state` returns `RESULTS.empty_subset` when input arrays are empty.
- **pi-only-refactor.AC1.6 Edge:** `fit_baseline` returns `RESULTS.max_steps_reached` (not an exception) when `max_iter=1`.

### pi-only-refactor.AC2: Module layout matches the target structure

- **pi-only-refactor.AC2.4 Success:** `from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params` succeeds.
- **pi-only-refactor.AC2.7 Failure:** `import mut_var.numerics.baseline` raises `ModuleNotFoundError`.
- **pi-only-refactor.AC2.8 Failure:** `import mut_var.numerics.refit` raises `ModuleNotFoundError`.

### pi-only-refactor.AC4: Quality gate passes end-to-end

- **pi-only-refactor.AC4.1 Success:** `ruff check src/mut_var tests` exits with code 0.
- **pi-only-refactor.AC4.2 Success:** `mypy src/mut_var tests` exits with code 0 (no type errors).
- **pi-only-refactor.AC4.3 Success:** `pytest -p no:capture` exits with code 0 (all tests pass).

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `pi` only as optimization variable | `mu_k`/`var_k` were never updated (the proposal step in `_propose_candidate` already only updated `pi`); removing them from `y0` eliminates dead gradient computation |
| Likelihood matrix pre-computed once | `pdf(beta_hat, s2, mu_k, var_k)` is the expensive part; fixed `mu_k`/`var_k` mean it never needs recomputation |
| Uniform `pi` init | No random seed needed; `BacktrackingArmijo` adapts step size; removes `seed` from pipeline |
| `FitState` NamedTuple | Carries likelihood matrix + initial params between `prepare_fit_state` and the fit functions |
| Column-slice for filtered likelihood matrix | After `_filter_components` reduces K→K', slice `L[:, keep_mask]` instead of recomputing PDF |
| `InferenceConfig` drops `step_size`/`penalty` | These become `_DEFAULT_STEP_SIZE = 0.01`, `_DEFAULT_PENALTY = 1.0` in `mixture_fit.py` |

---

## Codebase Verification Findings

- ✓ `MutVarSolver` accepts any pytree `Y` as `y0`; passing `pi` (1D array) requires no solver changes.
- ✓ `_propose_candidate` in refit.py already only updates `pi` — confirms that mu/var were never optimized.
- ✓ `simplex_tangent_direction` and `exponential_map_simplex` in `_solver_utils.py` are unchanged.
- ✓ `pdf = jax.vmap(_pdf, (None, None, 0, 0), 1)` gives shape `(n, K-1)`; null column added separately.
- ✓ Ordering penalty: `relu(baseline.pi[1:] * pi[:-1] - baseline.pi[:-1] * pi[1:])` + `relu(baseline.pi[0] - pi[0])`.
- ✓ `alpha = [10.0] + (K-1) * [1.0]` (Dirichlet concentration — kept as internal constant).
- + `_filter_components` reduces K→K'. Pipeline must compute `keep` mask and slice `L[:, keep]`.
- + `infer.py` currently uses `seed` and `to_baseline_config()`/`to_refit_config()` — both removed in this phase.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Simplify `InferenceConfig` in `types.py` — remove `step_size`, `penalty`, and conversion methods

**Verifies:** prerequisite for AC1 (InferenceConfig no longer exposes solver internals)

**Files:**
- Modify: `src/mut_var/types.py`

**Implementation:**

Remove the `step_size` and `penalty` fields from `InferenceConfig` and delete `to_baseline_config` and `to_refit_config` methods entirely. The resulting class must be:

```python
class InferenceConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-3
    filter_threshold: float = 1e-8
```

Also remove from `types.py`:
- The `TYPE_CHECKING` block that imported `BaselineConfig` and `RefitConfig`
- Any imports from `mut_var.numerics.baseline` or `mut_var.numerics.refit`

Update `SimulationPipelineConfig` default — verify it does not use `step_size`/`penalty` (it does not; leave it unchanged).

**Verification:**

```bash
python -c "
from mut_var.types import InferenceConfig
c = InferenceConfig(num_clusters=10)
assert hasattr(c, 'num_clusters')
assert hasattr(c, 'max_iter')
assert hasattr(c, 'tol')
assert hasattr(c, 'filter_threshold')
assert not hasattr(c, 'step_size')
assert not hasattr(c, 'penalty')
print('InferenceConfig OK')
"
```

**Commit:** `refactor: simplify InferenceConfig — remove step_size, penalty, conversion methods`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Create `src/mut_var/numerics/mixture_fit.py`

**Verifies:** pi-only-refactor.AC1.1, AC1.4, AC1.5, AC2.4

**Files:**
- Create: `src/mut_var/numerics/mixture_fit.py`

**Implementation:**

The module pattern is `# pattern: Functional Core`.

**Imports needed:**
```python
from __future__ import annotations
# pattern: Functional Core
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as nn
import optimistix as optx
from jax.scipy.stats import norm
from jax.scipy.special import xlogy
from jaxtyping import Array, ArrayLike

from mut_var.types import RESULTS, Solution, InferenceConfig
from mut_var.numerics._optimistix_solver import map_optimistix_result, MutVarSolver
from mut_var.numerics._solver_utils import (
    exponential_map_simplex,
    is_nonfinite,
    simplex_tangent_direction,
)
```

**Internal constants:**
```python
_DEFAULT_STEP_SIZE: float = 0.01
_DEFAULT_PENALTY: float = 1.0
```

**NamedTuples:**
```python
class Params(NamedTuple):
    pi: Array
    mu_k: Array
    var_k: Array


class FitState(NamedTuple):
    likelihood_matrix: Array   # shape (n, K)
    initial_params: Params
```

**Internal PDF utilities** (copied from `baseline.py` lines 38–49):
```python
def _logpdf(beta_hat, s2, mean, var_k):
    return norm.logpdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


def _pdf(beta_hat, s2, mean, var_k):
    return norm.pdf(beta_hat, loc=mean, scale=jnp.sqrt(s2 + var_k))


# vmapped over (mu_k, var_k) components, output shape (n, K-1)
_pdf_components = jax.vmap(_pdf, (None, None, 0, 0), 1)
```

**Internal likelihood matrix builder:**
```python
def _compute_likelihood_matrix(
    beta_hat: Array,
    s2: Array,
    mu_k: Array,
    var_k: Array,
) -> Array:
    """Build (n, K) likelihood matrix. Column 0 is the null component."""
    null_col = _pdf(beta_hat, s2, 0.0, 0.0)[:, jnp.newaxis]   # (n, 1)
    other_cols = _pdf_components(beta_hat, s2, mu_k, var_k)     # (n, K-1)
    return jnp.concatenate([null_col, other_cols], axis=1)       # (n, K)
```

**Internal validation** (adapted from `baseline.py::_validate_inputs`):
```python
def _validate_inputs(beta_hat: ArrayLike, s2: ArrayLike, config: InferenceConfig) -> Solution | None:
    beta_hat_arr = jnp.asarray(beta_hat)
    s2_arr = jnp.asarray(s2)

    if config.num_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be at least 2"},
            state=None,
        )
    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays"},
            state=None,
        )
    if beta_hat_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "input arrays are empty"},
            state=None,
        )
    if beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must have the same length"},
            state=None,
        )
    if not bool(jnp.isfinite(beta_hat_arr).all()) or not bool(jnp.isfinite(s2_arr).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be finite"},
            state=None,
        )
    if not bool((s2_arr > 0.0).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "s2 must be strictly positive"},
            state=None,
        )
    return None
```

**`prepare_fit_state`** (public):
```python
def prepare_fit_state(
    beta_hat: ArrayLike,
    s2: ArrayLike,
    config: InferenceConfig,
) -> Solution:
    r"""Validate inputs, build the fixed grid, and pre-compute the likelihood matrix.

    **Arguments:**

    - `beta_hat`: 1D array of effect-size estimates.
    - `s2`: 1D array of observation variances (must be strictly positive).
    - `config`: Inference configuration.

    **Returns:**

    - `Solution` with `FitState` on success. Status codes:
      `RESULTS.successful`, `RESULTS.invalid_input`, `RESULTS.empty_subset`.
    """
    invalid = _validate_inputs(beta_hat, s2, config)
    if invalid is not None:
        return invalid

    beta_hat_arr = jnp.asarray(beta_hat, dtype=jnp.float64)
    s2_arr = jnp.asarray(s2, dtype=jnp.float64)

    std_err = jnp.sqrt(s2_arr)
    min_val = jnp.min(std_err) / 10.0
    max_val = jnp.max(beta_hat_arr ** 2 - s2_arr)
    if max_val < 0.0:
        max_val = 8.0 * min_val
    else:
        max_val = 2.0 * jnp.sqrt(max_val)
    if is_nonfinite(max_val) or bool(max_val <= 0.0):
        max_val = 8.0 * min_val

    K = config.num_clusters
    mu_k = jnp.zeros(K - 1, dtype=jnp.float64)
    var_k = jnp.exp(
        jnp.linspace(jnp.log(min_val), jnp.log(max_val), K - 1)
    ) ** 2

    L = _compute_likelihood_matrix(beta_hat_arr, s2_arr, mu_k, var_k)
    pi = jnp.ones(K, dtype=jnp.float64) / K

    state = FitState(
        likelihood_matrix=L,
        initial_params=Params(pi=pi, mu_k=mu_k, var_k=var_k),
    )
    return Solution(value=state, result=RESULTS.successful, stats=None, state=None)
```

**Internal pi-only step update function:**
```python
def _pi_step(pi: Array, direction: Array, step_size: ArrayLike) -> Array:
    tangent = simplex_tangent_direction(pi, direction)
    return exponential_map_simplex(pi, tangent, step_size)
```

**Internal baseline objective:**
```python
def _baseline_objective(pi: Array, L: Array, alpha: Array) -> Array:
    """Penalized negative log-likelihood for baseline fit."""
    mixture_pdf = jnp.clip(L @ pi, min=jnp.finfo(float).tiny)
    log_likelihood = jnp.sum(jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1.0, pi))
    return -(log_likelihood - log_penalty)
```

**`fit_baseline`** (public):
```python
def fit_baseline(
    state: FitState,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    r"""Fit baseline mixture weights via Riemannian gradient descent on the simplex.

    **Arguments:**

    - `state`: Pre-computed fit state from `prepare_fit_state`.
    - `config`: Inference configuration.
    - `verbose`: If `True` or a callable, emit solver diagnostics.

    **Returns:**

    - `Solution` with `Params`. Status: `RESULTS.successful`, `RESULTS.max_steps_reached`,
      `RESULTS.nonfinite_objective`.
    """
    L = state.likelihood_matrix
    init_params = state.initial_params
    K = init_params.pi.shape[0]
    alpha = jnp.array([10.0] + (K - 1) * [1.0], dtype=jnp.float64)

    obj = eqx.filter_jit(_baseline_objective)

    def _neg_obj(pi, _args):
        val = obj(pi, L, alpha)
        if is_nonfinite(val):
            return jnp.inf, None
        return val, None

    solver = MutVarSolver(
        step_update=_pi_step,
        step_size=_DEFAULT_STEP_SIZE,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )
    optx_solution = optx.minimise(
        fn=_neg_obj,
        solver=solver,
        y0=init_params.pi,
        max_steps=config.max_iter,
        throw=False,
    )
    result = map_optimistix_result(optx_solution.result)
    pi_opt = optx_solution.value
    params = Params(pi=pi_opt, mu_k=init_params.mu_k, var_k=init_params.var_k)
    return Solution(
        value=params,
        result=result,
        stats={"n_steps": optx_solution.stats.get("num_steps", None)},
        state=None,
    )
```

**Internal refit objective:**
```python
def _refit_objective(
    pi: Array,
    L_sub: Array,
    prev_pi: Array,
    alpha: Array,
    penalty: float,
) -> Array:
    """Penalized refit objective with ordering constraint."""
    mixture_pdf = jnp.clip(L_sub @ pi, min=jnp.finfo(float).tiny)
    log_likelihood = jnp.sum(jnp.log(mixture_pdf))
    log_penalty = jnp.sum(xlogy(alpha - 1.0, pi))
    p1 = jnp.sum(nn.relu(prev_pi[1:] * pi[:-1] - prev_pi[:-1] * pi[1:]))
    rel_point_mass_dist = nn.relu(prev_pi[0] - pi[0])
    ordering_penalty = penalty * (p1 + rel_point_mass_dist)
    return -(log_likelihood - log_penalty - ordering_penalty)
```

**`fit_refit_step`** (public):
```python
def fit_refit_step(
    L_sub: ArrayLike,
    prev_params: Params,
    config: InferenceConfig,
    verbose: bool | Any = False,
) -> Solution:
    r"""Fit one refit step for a MAF-subset likelihood matrix.

    **Arguments:**

    - `L_sub`: Pre-sliced likelihood matrix for this MAF threshold, shape `(n_sub, K)`.
    - `prev_params`: Params from the previous threshold (provides init and ordering anchor).
    - `config`: Inference configuration.
    - `verbose`: If `True` or a callable, emit solver diagnostics.

    **Returns:**

    - `Solution` with `Params`. Status: `RESULTS.successful`, `RESULTS.max_steps_reached`,
      `RESULTS.empty_subset`, `RESULTS.invalid_input`, `RESULTS.nonfinite_objective`.
    """
    L_arr = jnp.asarray(L_sub, dtype=jnp.float64)
    pi_init = jnp.asarray(prev_params.pi, dtype=jnp.float64)

    if L_arr.ndim != 2:
        return Solution(value=prev_params, result=RESULTS.invalid_input,
                        stats={"reason": "L_sub must be a 2D array"}, state=None)
    if L_arr.shape[0] == 0:
        return Solution(value=prev_params, result=RESULTS.empty_subset,
                        stats={"reason": "L_sub has no rows (empty MAF subset)"}, state=None)
    if not bool(jnp.isfinite(L_arr).all()):
        return Solution(value=prev_params, result=RESULTS.invalid_input,
                        stats={"reason": "L_sub contains non-finite values"}, state=None)

    K = pi_init.shape[0]
    alpha = jnp.array([10.0] + (K - 1) * [1.0], dtype=jnp.float64)
    obj = eqx.filter_jit(_refit_objective)

    def _neg_obj(pi, _args):
        val = obj(pi, L_arr, pi_init, alpha, _DEFAULT_PENALTY)
        if is_nonfinite(val):
            return jnp.inf, None
        return val, None

    solver = MutVarSolver(
        step_update=_pi_step,
        step_size=_DEFAULT_STEP_SIZE,
        rtol=config.tol,
        atol=config.tol,
        verbose=verbose,
    )
    optx_solution = optx.minimise(
        fn=_neg_obj,
        solver=solver,
        y0=pi_init,
        max_steps=config.max_iter,
        throw=False,
    )
    result = map_optimistix_result(optx_solution.result)
    pi_opt = optx_solution.value
    params = Params(pi=pi_opt, mu_k=prev_params.mu_k, var_k=prev_params.var_k)
    return Solution(
        value=params,
        result=result,
        stats={"n_steps": optx_solution.stats.get("num_steps", None)},
        state=None,
    )
```

**`__all__`:**
```python
__all__ = [
    "FitState",
    "Params",
    "fit_baseline",
    "fit_refit_step",
    "prepare_fit_state",
]
```

**Verification:**

```bash
python -c "
from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params
print('imports OK')
"
```

```bash
python -c "
import numpy as np
from mut_var.types import InferenceConfig
from mut_var.numerics.mixture_fit import prepare_fit_state

np.random.seed(0)
n = 50
beta = np.random.randn(n) * 0.1
s2 = np.ones(n) * 0.01

config = InferenceConfig(num_clusters=5, max_iter=5)
sol = prepare_fit_state(beta, s2, config)
print('result:', sol.result)
print('L shape:', sol.value.likelihood_matrix.shape)
assert sol.value.likelihood_matrix.shape == (n, 5), f'Expected (50, 5), got {sol.value.likelihood_matrix.shape}'
print('AC1.1 OK')
"
```

**Commit:** `feat: add numerics/mixture_fit.py — pi-only Optimistix solver with pre-computed likelihood matrix`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

<!-- START_SUBCOMPONENT_B (tasks 3-5) -->

<!-- START_TASK_3 -->
### Task 3: Update `infer.py` to use the new numerics API; remove `seed` parameter

**Verifies:** prerequisite for AC4.3 (existing pipeline tests pass)

**Files:**
- Modify: `src/mut_var/infer.py`

**Implementation:**

1. **Update imports at top of file:**
   - Remove `import jax.random as rdm` (no more random seed usage)
   - Change TYPE_CHECKING block: replace `from mut_var.numerics.baseline import BaselineConfig, Params` and `from mut_var.numerics.refit import RefitConfig` with `from mut_var.numerics.mixture_fit import FitState, Params`
   - Add runtime import: `from mut_var.numerics.mixture_fit import fit_baseline, fit_refit_step, prepare_fit_state`
   - Remove: `from mut_var.numerics import fit_baseline, fit_refit_grid` (old imports)
   - Keep: `from mut_var.numerics._solver_utils import is_recoverable_result, is_nonfinite, merge_recoverable_results`

2. **Update `run_inference_pipeline` signature** — remove `seed: int = 0` parameter and its docstring line.

3. **Replace the numerics section** (currently lines ~216–265) with the new API:

```python
workflow_log.info("inference pipeline: starting numerics")
beta_hat = jnp.asarray(arrays.beta_hat, dtype=jnp.float64)
s2 = jnp.asarray(arrays.s2, dtype=jnp.float64)
inference_config = config if config is not None else InferenceConfig(num_clusters=30)

workflow_log.info("inference pipeline: preparing fit state")
state_solution = prepare_fit_state(beta_hat, s2, inference_config)
if not is_recoverable_result(state_solution.result):
    solution = state_solution
else:
    state = state_solution.value  # FitState

    workflow_log.info("inference pipeline: fitting baseline model")
    baseline_solution = fit_baseline(
        state, inference_config,
        verbose=_solver_debug_callback(workflow_log, "baseline"),
    )
    workflow_log.info(
        "inference pipeline: baseline fit completed with result '%s'",
        RESULTS[baseline_solution.result],
    )

    if not is_recoverable_result(baseline_solution.result):
        solution = baseline_solution
    else:
        workflow_log.info("inference pipeline: filtering baseline components")
        keep = baseline_solution.value.pi > inference_config.filter_threshold
        keep = keep.at[0].set(True)
        filtered = _filter_components(baseline_solution.value, inference_config.filter_threshold)

        # Slice likelihood matrix columns to match filtered component count
        L_filtered = state.likelihood_matrix[:, keep]

        workflow_log.info("inference pipeline: fitting refit grid")
        models: list[Params] = [filtered]
        prev_params = filtered
        step_results = []
        for i in range(int(maf_masks.shape[0])):
            mask = maf_masks[i].astype(bool)
            L_sub = L_filtered[mask, :]
            step_sol = fit_refit_step(
                L_sub, prev_params, inference_config,
                verbose=_solver_debug_callback(workflow_log, "refit"),
            )
            step_results.append(step_sol.result)
            if is_recoverable_result(step_sol.result):
                prev_params = step_sol.value
            models.append(prev_params)

        refit_result = merge_recoverable_results(*step_results)
        workflow_log.info(
            "inference pipeline: refit grid completed with result '%s'",
            RESULTS[refit_result],
        )

        if not is_recoverable_result(refit_result):
            solution = Solution(
                value=None, result=refit_result,
                stats={"reason": f"refit failed with status '{RESULTS[refit_result]}'"},
                state=None,
            )
        else:
            workflow_log.info("inference pipeline: building numerics payload")
            numerics_payload = _build_long_payload(models, maf_grid=maf_grid, af=arrays.af)
            solution = Solution(
                value=numerics_payload,
                result=merge_recoverable_results(baseline_solution.result, refit_result),
                stats={
                    "num_models": len(models),
                    "num_components": int(models[0].pi.shape[0]),
                    "baseline": baseline_solution.stats,
                },
                state=None,
            )
```

4. **Remove** the `to_baseline_config()` / `to_refit_config()` usage (already replaced above).
5. **Keep** `_filter_components`, `_build_long_payload`, `_payload_from_solution`, `_solver_debug_callback` unchanged.
6. Remove `InferenceConfig` from `infer.py`'s `__all__` only if it's no longer defined there (it was re-exported via import from types — keep it in `__all__` for backward compat).

**Verification:**

```bash
python -c "import mut_var.infer; print('import OK')"
```

**Commit:** `refactor: update infer.py to use prepare_fit_state/fit_baseline/fit_refit_step; remove seed`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Update `numerics/__init__.py`; update test files for removed `seed`/`step_size`; fix baseline/refit monkeypatches

> **CRITICAL:** Task 5 deletes `baseline.py` and `refit.py`. Any test that imports these modules must be fixed in this task, before Task 5 runs. Failing to do so causes `ModuleNotFoundError` during pytest collection in Task 5.

**Verifies:** prerequisite for AC4.3

**Files:**
- Modify: `src/mut_var/numerics/__init__.py`
- Modify: `tests/test_infer.py`
- Modify: `tests/test_cli_contracts.py`
- Modify: `tests/test_infer_opt.py`

**Implementation:**

**`src/mut_var/numerics/__init__.py`** — replace current contents with:

```python
from mut_var.io import InferenceArrays
from mut_var.types import InferenceConfig

from .mixture_fit import fit_baseline, fit_refit_step, FitState, Params, prepare_fit_state
from .curve_fit import curve, fit_curve
from .simulate import simulate_mixture_data, SimulationArrays, SimulationNumericsConfig

__all__ = [
    "FitState",
    "InferenceArrays",
    "InferenceConfig",
    "Params",
    "curve",
    "fit_baseline",
    "fit_curve",
    "fit_refit_step",
    "prepare_fit_state",
    "simulate_mixture_data",
    "SimulationArrays",
    "SimulationNumericsConfig",
]
```

(Removes: `BaselineConfig`, `RefitConfig`, `fit_refit_grid`.)

**`tests/test_infer.py`** — make the following changes:

1. Remove `seed=0` from all `run_inference_pipeline` / `run_inference_dataframe_pipeline` calls (lines 26, 43, 61, 90, 134, 219).
2. Change all `InferenceConfig(num_clusters=N, max_iter=M, step_size=X)` to `InferenceConfig(num_clusters=N, max_iter=M)` — remove `step_size` kwarg.
3. Line 169: `assert infer_module.InferenceArrays is numerics.InferenceArrays` and `assert infer_module.InferenceConfig is numerics.InferenceConfig` — these still hold since both import from `mut_var.io` and `mut_var.types` respectively. No change needed.
4. Rewrite `test_run_inference_pipeline_raises_on_critical_numerics_result` (lines 69–93) — change import and monkeypatch target from `baseline_module.fit_baseline` to `mixture_fit_module.fit_baseline`:
   ```python
   def test_run_inference_pipeline_raises_on_critical_numerics_result(sumstats_valid_df, monkeypatch):
       import mut_var.numerics.mixture_fit as mixture_fit_module

       monkeypatch.setattr(
           mixture_fit_module,
           "fit_baseline",
           lambda *_args, **_kwargs: Solution(
               value=None,
               result=RESULTS.nonfinite_objective,
               stats={"reason": "objective became non-finite"},
               state=None,
           ),
       )

       with pytest.raises(RuntimeError) as err:
           run_inference_dataframe_pipeline(
               sumstats_valid_df,
               lowest=1e-3,
               highest=5e-3,
               num_breaks=2,
               config=InferenceConfig(num_clusters=3, max_iter=2),
           )

       assert "non-finite" in str(err.value)
   ```
5. Rewrite `test_run_inference_pipeline_raises_on_empty_subset_result` (lines 96–137) — there is no `fit_refit_grid` after Phase 2; patch `fit_refit_step` instead to return `RESULTS.empty_subset`:
   ```python
   def test_run_inference_pipeline_raises_on_empty_subset_result(sumstats_valid_df, monkeypatch):
       import mut_var.numerics.mixture_fit as mixture_fit_module

       monkeypatch.setattr(
           mixture_fit_module,
           "fit_refit_step",
           lambda *_args, **_kwargs: Solution(
               value=None,
               result=RESULTS.empty_subset,
               stats={"reason": "empty subset"},
               state=None,
           ),
       )

       with pytest.raises(ValueError) as err:
           run_inference_dataframe_pipeline(
               sumstats_valid_df,
               lowest=1e-3,
               highest=5e-3,
               num_breaks=2,
               config=InferenceConfig(num_clusters=3, max_iter=2),
           )

       assert "empty" in str(err.value).lower()
   ```
6. Rewrite `test_simulated_observed_output_is_accepted_by_run_inference_pipeline` (lines 174–224) — change imports and both monkeypatch targets to `mixture_fit_module`:
   ```python
   def test_simulated_observed_output_is_accepted_by_run_inference_pipeline(monkeypatch):
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

       result_df = run_inference_dataframe_pipeline(
           artifacts.observed,
           lowest=1e-3,
           highest=5e-3,
           num_breaks=2,
           config=InferenceConfig(num_clusters=2, max_iter=5),
       )

       assert isinstance(result_df, pl.DataFrame)
       assert result_df.height > 0
       assert result_df.columns == ["mu0", "var0", "maf", "name", "value"]
   ```

**`tests/test_cli_contracts.py`** — make the following changes to prevent `ModuleNotFoundError` after Task 5 deletes `baseline.py`:

1. Line 8 — change import:
   ```python
   # Before:
   import mut_var.numerics.baseline as baseline_module
   # After:
   import mut_var.numerics.mixture_fit as mixture_fit_module
   ```
2. `_guard_numerics` — change monkeypatch target:
   ```python
   # Before:
   monkeypatch.setattr(baseline_module, "fit_baseline", _unexpected_call)
   # After:
   monkeypatch.setattr(mixture_fit_module, "prepare_fit_state", _unexpected_call)
   ```

**`tests/test_infer_opt.py`** — rewrite to test the new API:

The existing tests target `fit_baseline(beta_hat, s2, key, config, ...)` and `fit_refit_grid(...)`. These are removed. Replace the entire file with tests that target the new `prepare_fit_state`, `fit_baseline`, `fit_refit_step` surface, verifying the ACs:

New test structure:
- `test_prepare_fit_state_succeeds_on_valid_arrays` — calls `prepare_fit_state`; checks `RESULTS.successful`, `FitState`, `likelihood_matrix.shape == (n, K)` (AC1.1).
- `test_prepare_fit_state_invalid_s2` — non-positive `s2`; checks `RESULTS.invalid_input` (AC1.4).
- `test_prepare_fit_state_empty_arrays` — empty arrays; checks `RESULTS.empty_subset` (AC1.5).
- `test_fit_baseline_converges_pi_sums_to_one` — round-trip `prepare_fit_state` → `fit_baseline`; asserts `sum(pi) ≈ 1.0` and `mu_k`/`var_k` match `state.initial_params` (AC1.2).
- `test_fit_baseline_max_steps_reached_on_max_iter_one` — `max_iter=1`; asserts `RESULTS.max_steps_reached` not an exception (AC1.6).
- `test_fit_refit_step_pi_sums_to_one_and_mu_var_unchanged` — runs `fit_refit_step`; asserts `sum(pi) ≈ 1.0`, `mu_k == prev_params.mu_k`, `var_k == prev_params.var_k` (AC1.3).

Use a fixture providing a small synthetic dataset:
```python
@pytest.fixture
def synthetic_arrays():
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    beta = rng.normal(0, 0.05, n)
    s2 = np.full(n, 0.01)
    return beta, s2
```

**Verification:**

```bash
ruff check src/mut_var tests
```
Expected: code 0.

```bash
pytest tests/test_infer_opt.py tests/test_infer.py tests/test_cli_contracts.py -p no:capture
```
Expected: all tests pass.

**Commit:** `refactor: update numerics/__init__.py and tests for new mixture_fit API`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Delete `baseline.py` and `refit.py`; run full quality gate

**Verifies:** pi-only-refactor.AC2.7, AC2.8, AC4.1, AC4.2, AC4.3

**Files:**
- Delete: `src/mut_var/numerics/baseline.py`
- Delete: `src/mut_var/numerics/refit.py`

**Implementation:**

```bash
rm src/mut_var/numerics/baseline.py
rm src/mut_var/numerics/refit.py
```

Verify no remaining references:
```bash
grep -r "numerics.baseline\|numerics.refit\|from .baseline\|from .refit\|BaselineConfig\|RefitConfig\|fit_refit_grid\|baseline_objective_lse\|_exponential_map_normal\|_riemannian_step" src/mut_var tests --include="*.py"
```
Expected: no output.

**Verification — module layout:**

```bash
python -c "
try:
    import mut_var.numerics.baseline
    print('FAIL: baseline still importable')
except ModuleNotFoundError:
    print('AC2.7 OK: baseline deleted')

try:
    import mut_var.numerics.refit
    print('FAIL: refit still importable')
except ModuleNotFoundError:
    print('AC2.8 OK: refit deleted')

from mut_var.numerics.mixture_fit import prepare_fit_state, fit_baseline, fit_refit_step, FitState, Params
print('AC2.4 OK: mixture_fit imports OK')
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

**Commit:** `refactor: delete baseline.py and refit.py; pi-only numerics complete`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->
