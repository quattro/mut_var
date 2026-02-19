# Numerics Domain

Last verified: 2026-02-19

## Purpose
Provide array-only numerical kernels for mutation-variance estimation with explicit solver status channels that remain stable under JAX transforms.

## Contracts
- **Exposes**:
  - `fit_baseline(beta_hat, s2, key, config, verbose_callback=None) -> Solution`
  - `fit_refit_grid(beta_hat, s2, maf_masks, init, config, verbose_callback=None) -> Solution`
  - `fit_curve(maf, value) -> Solution`
- **Guarantees**:
  - Public numerics entrypoints return `Solution` with status in `result` for non-success paths.
  - Baseline/refit optimization is full-batch and routed through Optimistix (`MutVarSolver` with backtracking line search).
  - Baseline and refit objectives are JIT-staged with `equinox.filter_jit`.
  - Optional `verbose_callback` hooks receive `(step_index, loss, grad_norm)` diagnostics without adding logging side effects inside numerics.
  - Recoverable statuses are merged via `merge_recoverable_results`; `max_steps_reached` propagates without raising.
- **Expects**:
  - Array-like inputs only (`jnp.asarray`-compatible), not dataframe objects.
  - `beta_hat` and `s2` are finite 1D arrays with equal length and strictly positive `s2`.
  - `maf_masks` is a 2D boolean-aligned mask over observations.

## Dependencies
- **Uses**: `jax`, `equinox`, `optimistix`, `jaxtyping`, `mut_var.contracts`.
- **Used by**: `src/mut_var/infer.py` and `src/mut_var/curve.py`.
- **Boundary**:
  - No file I/O, CLI parsing, logging, or dataframe conversion in this domain.
  - User-facing validation belongs at higher-level adapters before numerics execution.
  - Keep public numerics surface exported through `src/mut_var/numerics/__init__.py`.

## Key Decisions
- Shared Optimistix adapter (`_optimistix_solver.py`) replaces legacy `_optimize.py` loops.
- SGD/minibatch paths were removed; full-batch is the only supported optimization mode.
- `MutVarSolver` centralizes result mapping from `optx.RESULTS` to `mut_var.contracts.RESULTS`.
- Solver-step debug telemetry is surfaced through explicit callback parameters instead of in-domain logging.

## Invariants
- `Params.pi` remains simplex-normalized after each Riemannian update.
- `fit_refit_grid` returns one model per threshold plus the initial model on successful/recoverable runs.
- Solver outputs always report one canonical status: `successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, or `max_steps_reached`.

## Key Files
- `src/mut_var/numerics/baseline.py` - baseline mixture fitting objective and solver wiring.
- `src/mut_var/numerics/refit.py` - grid refit objective and sequential threshold updates.
- `src/mut_var/numerics/_optimistix_solver.py` - shared Optimistix descent/solver adapters.
- `src/mut_var/numerics/curve_fit.py` - curve least-squares fitting kernel.

## Gotchas
- `RESULTS.max_steps_reached` is intentionally treated as recoverable by numerics pipeline utilities.
- Changing `Solution.stats` keys can break regression assertions that inspect diagnostics.
