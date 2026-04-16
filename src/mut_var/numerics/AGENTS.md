# Numerics Domain

Last verified: 2026-04-15

## Purpose
Provide array-only numerical kernels for mutation-variance estimation with explicit solver status channels that remain stable under JAX transforms.

## Contracts
- **Exposes**:
  - `prepare_fit_state(beta_hat, s2, config) -> Solution`
  - `fit_baseline(state: FitState, config: InferenceConfig, verbose=False) -> Solution`
  - `fit_refit_step(L_sub: Array, prev_params: Params, config: InferenceConfig, verbose=False) -> Solution`
  - `fit_curve(maf, value) -> Solution`
  - `simulate_mixture_data(n_rows, seed, config) -> Solution`
- **Guarantees**:
  - Public numerics entrypoints return `Solution` with status in `result` for non-success paths and are JIT-safe under `jax.jit` for valid array inputs.
  - Baseline/refit optimization is full-batch and routed through Optimistix (`MutVarSolver` with backtracking line search).
  - Baseline and refit objectives are JIT-staged with `equinox.filter_jit`.
  - Optional `verbose` controls (bool/callable) emit solver diagnostics from the Optimistix-compatible solver step path.
  - Recoverable statuses are merged via `merge_recoverable_results`; `max_steps_reached` propagates without raising.
  - `simulate_mixture_data` validates simulation domains before random draws and returns `SimulationArrays` payloads on `RESULTS.successful`.
- **Expects**:
  - Array-like inputs only (`jnp.asarray`-compatible), not dataframe objects.
  - `beta_hat` and `s2` are finite 1D arrays with equal length and strictly positive `s2`.
  - `InferenceConfig` is treated as a static tracing input so its fields remain available as Python values under JIT.
  - `prepare_fit_state` builds a 2D likelihood matrix over observations/components; `fit_refit_step` receives one threshold-sliced likelihood matrix whose column count matches `prev_params.pi`.
  - Simulation configs provide aligned mixture parameter lengths, valid AF generator domains, and positive SE controls.

## Dependencies
- **Uses**: `jax`, `equinox`, `optimistix`, `jaxtyping`, `mut_var.types` (for shared `RESULTS` and `Solution` contracts).
- **Used by**: `src/mut_var/pipelines/`, package-root pipeline exports, and CLI orchestration.
- **Boundary**:
  - No file I/O, CLI parsing, logging, or dataframe conversion in this domain.
  - User-facing validation belongs at higher-level adapters before numerics execution.
  - Keep public numerics surface exported through `src/mut_var/numerics/__init__.py`.

## Key Decisions
- Shared Optimistix adapter (`_optimistix_solver.py`) replaces legacy `_optimize.py` loops.
- SGD/minibatch paths were removed; full-batch is the only supported optimization mode.
- `MutVarSolver` centralizes result mapping from `optx.RESULTS` to `mut_var.types.RESULTS`.
- `baseline.py` and `refit.py` were merged into `mixture_fit.py`; the likelihood matrix is pre-computed once in `prepare_fit_state`.
- Solver-step diagnostics are surfaced through explicit Optimistix-style `verbose` controls on solver APIs.

## Invariants
- `Params.pi` remains simplex-normalized after each Riemannian update.
- `fit_refit_step` returns the next refit model for one thresholded likelihood subset; callers compose these stepwise results across thresholds when building the refit sequence.
- Solver outputs always report one canonical status: `successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, or `max_steps_reached`.
- Successful simulation outputs have finite arrays with strictly positive `sigma2`.
- `simulate_mixture_data` is reproducible for a fixed `(seed, config, n_rows)` tuple.

## Key Files
- `src/mut_var/numerics/mixture_fit.py` - pi-only solver: `prepare_fit_state`, `fit_baseline`, `fit_refit_step`, `FitState`, `Params`.
- `src/mut_var/numerics/_optimistix_solver.py` - shared Optimistix descent/solver adapters.
- `src/mut_var/numerics/curve_fit.py` - curve least-squares fitting kernel.
- `src/mut_var/numerics/simulate.py` - mixture simulation validation and sampling kernel.

## Gotchas
- `RESULTS.max_steps_reached` is intentionally treated as recoverable by numerics pipeline utilities.
- Changing `Solution.stats` keys can break regression assertions that inspect diagnostics.
- `simulate_mixture_data` callers must check `Solution.result` (not just `value`) before reading simulation arrays.
