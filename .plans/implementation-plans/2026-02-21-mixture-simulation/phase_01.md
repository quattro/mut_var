# Mixture Simulation Module Implementation Plan

**Goal:** Add a numerics simulation kernel for K-component zero-mean mixture-of-normals data generation with AF-dependent variance links and explicit `Solution.result` failure states.

**Architecture:** Keep numerics array-only and side-effect-free under `src/mut_var/numerics`. The numerics entrypoint returns `Solution` and never returns dataframes. AF-dependent variance is modeled via component log-scales times a shared AF link function so component separation and AF-shape controls remain interpretable.

**Tech Stack:** `jax`, `equinox`, `jaxtyping`, `polars` (tests only), existing `mut_var.contracts`.

**Scope:** 4 phases from validated design (this file is phase 1).

**Codebase verified:** 2026-02-21

---

## Acceptance Criteria Coverage

This phase implements and tests:

### mutvar-mixture-simulation.AC1: Canonical numerics and status contracts
- **mutvar-mixture-simulation.AC1.2 Success:** Numerics simulation entrypoint returns `Solution` and uses `Solution.result` as the canonical status signal.

### mutvar-mixture-simulation.AC2: Boundary validation before simulation
- **mutvar-mixture-simulation.AC2.1 Failure:** Invalid mixture/AF/SE parameter domains fail before random draws with `RESULTS.invalid_input`.

### mutvar-mixture-simulation.AC3: AF-dependent variance parameterization
- **mutvar-mixture-simulation.AC3.1 Success:** Variance link families (`none`, `maf_power`, `maf_power_shifted`) produce finite strictly positive per-row variances.
- **mutvar-mixture-simulation.AC3.3 Success:** Simulation is reproducible for fixed seed and config.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: Define numerics simulation contracts

**Verifies:** mutvar-mixture-simulation.AC1.2

**Files:**
- Create: `src/mut_var/numerics/simulate.py`
- Modify: `src/mut_var/numerics/__init__.py`

**Implementation:**
Add numerics-only types and public entrypoint signatures:

```python
class SimulationArrays(NamedTuple):
    row_id: ArrayLike
    af: ArrayLike
    component: ArrayLike
    sigma2: ArrayLike
    beta_true: ArrayLike
    se: ArrayLike
    beta_hat: ArrayLike

class SimulationNumericsConfig(NamedTuple):
    weights: tuple[float, ...]
    log_var_scales: tuple[float, ...]
    variance_link: Literal["none", "maf_power", "maf_power_shifted"] = "maf_power"
    theta: float = 0.5
    link_eps: float = 1e-8
    link_shift: float = 0.0
    af_clip_min: float = 1e-4
    af_model: Literal["uniform", "beta"] = "beta"
    af_uniform_low: float = 0.01
    af_uniform_high: float = 0.5
    af_beta_a: float = 0.4
    af_beta_b: float = 0.4
    se_model: Literal["constant", "af_n_scaled"] = "af_n_scaled"
    se_constant: float = 0.02
    sample_size: float = 50000.0
    se_scale: float = 1.0


def simulate_mixture_data(*, n_rows: int, seed: int, config: SimulationNumericsConfig) -> Solution:
    ...
```

Export `SimulationArrays`, `SimulationNumericsConfig`, and `simulate_mixture_data` from `src/mut_var/numerics/__init__.py`.

Use raw docstrings with exact section labels (`**Arguments:**`, `**Returns:**`, `**Failure Modes:**`).

**Testing:**
No test file in this task; tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/numerics/simulate.py`
Expected: module compiles.

**Commit:** `feat: add numerics simulation contracts`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement ingress validation with explicit status failures

**Verifies:** mutvar-mixture-simulation.AC2.1

**Files:**
- Modify: `src/mut_var/numerics/simulate.py`

**Implementation:**
Implement `_validate_simulation_inputs(n_rows: int, config: SimulationNumericsConfig) -> Solution | None` that returns `RESULTS.invalid_input` with reason strings for:
- `n_rows < 1`
- `len(weights) < 2`
- `len(weights) != len(log_var_scales)`
- non-finite weights/scales
- any weight `<= 0`
- weights not summing to 1 within tolerance (`abs(sum-1) > 1e-8`)
- invalid `theta` range (`0 <= theta <= 1.5`)
- invalid AF generator domains (`uniform`: `0 < low < high <= 1`, `beta`: `a>0, b>0`)
- invalid clipping (`0 < af_clip_min < 0.5`)
- invalid `se_model` params (`se_constant > 0`, `sample_size > 0`, `se_scale > 0`)

Keep this function side-effect free and call it before any sampling.

**Testing:**
Tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/numerics/simulate.py`
Expected: module compiles.

**Commit:** `feat: add simulation numerics validation guards`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement AF variance-link and sampling kernel

**Verifies:** mutvar-mixture-simulation.AC3.1, mutvar-mixture-simulation.AC3.3

**Files:**
- Modify: `src/mut_var/numerics/simulate.py`

**Implementation:**
Implement helpers:
- `_sample_af(key, n_rows, config) -> ArrayLike`
- `_variance_link(af, config) -> ArrayLike`
- `_se_from_af(af, config) -> ArrayLike`

Required formulas:
- `p = clip(af, af_clip_min, 1 - af_clip_min)`
- `maf_term = 2 * p * (1 - p)`
- link families:
  - `none`: `link = 1`
  - `maf_power`: `link = (maf_term + link_eps) ** (-theta)`
  - `maf_power_shifted`: `link = (maf_term + link_shift) ** (-theta)` with `link_shift > 0`
- per-row component variance: `sigma2_i = exp(alpha_{z_i}) * link_i`
- latent effect: `beta_true_i ~ Normal(0, sqrt(sigma2_i))`
- observed effect: `beta_hat_i ~ Normal(beta_true_i, se_i)`

Implement `simulate_mixture_data` flow:
1. Validate inputs with `_validate_simulation_inputs`.
2. Build PRNG key tree from `seed`.
3. Sample AF, mixture component index, sigma2, beta_true, se, beta_hat.
4. Return `Solution(value=SimulationArrays(...), result=RESULTS.successful, stats={...})`.
5. If any simulated array contains non-finite values, return `RESULTS.nonfinite_objective` with reason.

Include stats keys: `n_rows`, `num_components`, `variance_link`, `theta`, `seed`.

**Testing:**
Tests are added in Task 4.

**Verification:**
Run: `python -m compileall src/mut_var/numerics/simulate.py`
Expected: module compiles.

**Commit:** `feat: implement mixture simulation numerics kernel`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Add numerics contract tests for validation, reproducibility, and link behavior

**Verifies:** mutvar-mixture-simulation.AC1.2, mutvar-mixture-simulation.AC2.1, mutvar-mixture-simulation.AC3.1, mutvar-mixture-simulation.AC3.3

**Files:**
- Create: `tests/test_simulate_numerics.py` (unit)

**Implementation:**
Add tests:
- `test_simulate_mixture_data_returns_solution_and_arrays_on_valid_config`
- `test_simulate_mixture_data_rejects_invalid_weight_shapes`
- `test_simulate_mixture_data_rejects_invalid_theta`
- `test_simulate_mixture_data_reproducible_for_fixed_seed`
- `test_variance_link_outputs_positive_finite_sigma2_for_all_links`
- `test_component_index_respects_component_range`

Use small `n_rows` for fast checks and one large run (`n_rows=20000`) for stable positivity/finite assertions.

**Testing:**
Tests must assert on `Solution.result`, not on presence of `value` alone.

**Verification:**
Run: `pytest -p no:capture tests/test_simulate_numerics.py`
Expected: all tests pass.

**Commit:** `test: add numerics simulation contract coverage`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->
