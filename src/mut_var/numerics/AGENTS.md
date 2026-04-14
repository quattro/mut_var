# Numerics Domain

Last verified: 2026-04-13

## Purpose
Provide array-only numerical kernels for mutation-variance estimation with explicit solver status channels.

## Contracts
- **Exposes**:
  - `prepare_fit_state(beta_hat, s2, config) -> Solution`
  - `fit_baseline(state, config, verbose=False) -> Solution`
  - `fit_refit_step(L_sub, prev_params, config, verbose=False) -> Solution`
  - `fit_curve(maf, value) -> Solution`
  - `simulate_mixture_data(config) -> Solution`
- **Guarantees**:
  - Public numerics entrypoints return `Solution` with status in `result` for non-success paths.
  - `prepare_fit_state` validates array inputs once, constructs the fixed component grid, and materializes the shared likelihood matrix.
  - Baseline optimization uses mix-SQP on a fixed log-spaced variance grid (optimizes `pi` only).
  - Refit optimization consumes precomputed likelihood matrices and uses mix-SQP-ordered with bidiagonal ordering constraint `A π ≤ 0`.
  - Optional `verbose` callable `(step, obj) -> None` emits per-step diagnostics.
  - Recoverable statuses are merged via `merge_recoverable_results`; `max_steps_reached` propagates without raising.
  - `simulate_mixture_data` validates simulation domains before random draws and returns `SimulationArrays` payloads on `RESULTS.successful`.
- **Expects**:
  - Array-like inputs only (NumPy-compatible), not dataframe objects.
  - `beta_hat` and `s2` are finite 1D arrays with equal length and strictly positive `s2`.
  - `maf_masks` is a 2D boolean-aligned mask over observations.
  - Simulation configs provide aligned mixture parameter lengths, valid AF generator domains, and positive SE controls.

## Dependencies
- **Uses**: `numpy`, `scipy`, `mut_var.types`, `mut_var.numerics._core` (Cython), `mut_var.numerics.mixsqp`.
- **Used by**: `src/mut_var/pipelines/inference.py`, `src/mut_var/pipelines/curve.py`, and `src/mut_var/pipelines/simulation.py`.
- **Boundary**:
  - No file I/O, CLI parsing, logging, or dataframe conversion in this domain.
  - User-facing validation belongs at higher-level adapters before numerics execution.
  - Keep public numerics surface exported through `src/mut_var/numerics/__init__.py`.

## Key Decisions
- mix-SQP (Kim et al. 2020) replaces Optimistix for baseline and refit fitting.
- Cython hot path (`_core.pyx`) implements `compute_grad_hess`, `compute_objective`, `line_search` using BLAS.
- `mixsqp.py` co-locates the outer SQP loop, active-set inner QP solvers, and recoverable-status utilities for the mix-SQP stack.
- Ordering constraint `A π ≤ 0` (hard constraint) replaces the soft penalty term in refit.
- `simulate_mixture_data` uses `numpy.random.default_rng(config.seed)` for reproducibility.
- Curve fitting uses `scipy.optimize.least_squares(method='lm')`.
- JAX, Equinox, and Optimistix have been fully removed from the numerics stack.

## Invariants
- `Params.pi` sums to 1.0 (normalized) after mix-SQP convergence.
- `FitState.likelihood_matrix` is aligned to the full observation set and reused across baseline/refit stages.
- Solver outputs always report one canonical status: `successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, or `max_steps_reached`.
- Successful simulation outputs have finite arrays with strictly positive `sigma2`.
- `simulate_mixture_data` is reproducible for a fixed `SimulationConfig`.

## Key Files
- `src/mut_var/numerics/mixture_fit.py` - fit-state preparation, baseline fitting, and likelihood-driven refit kernels.
- `src/mut_var/numerics/mixsqp.py` - mix-SQP outer loop, active-set QP solvers, ordering matrix construction, and recoverable-status helpers.
- `src/mut_var/numerics/_core.pyx` - Cython BLAS hot path.
- `src/mut_var/numerics/curve_fit.py` - curve least-squares fitting kernel.
- `src/mut_var/numerics/simulate.py` - mixture simulation validation and sampling kernel.

## Gotchas
- `RESULTS.max_steps_reached` is intentionally treated as recoverable by numerics pipeline utilities.
- Changing `Solution.stats` keys can break regression assertions that inspect diagnostics.
- `simulate_mixture_data` callers must check `Solution.result` (not just `value`) before reading simulation arrays.
- The Cython `_core` extension must be compiled before importing (`pip install -e .` or `python setup.py build_ext --inplace`).
