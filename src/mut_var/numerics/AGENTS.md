# Numerics Domain

Last verified: 2026-04-13

## Purpose
Provide array-only numerical kernels for mutation-variance estimation with explicit solver status channels.

## Contracts
- **Exposes**:
  - `fit_baseline(beta_hat, s2, config, verbose=False) -> Solution`
  - `fit_refit_grid(beta_hat, s2, maf_masks, init, config, verbose=False) -> Solution`
  - `fit_curve(maf, value) -> Solution`
  - `simulate_mixture_data(n_rows, seed, config) -> Solution`
- **Guarantees**:
  - Public numerics entrypoints return `Solution` with status in `result` for non-success paths.
  - Baseline optimization uses mix-SQP on a fixed log-spaced variance grid (optimizes `pi` only).
  - Refit grid optimization uses mix-SQP-ordered with bidiagonal ordering constraint `A π ≤ 0`.
  - Optional `verbose` callable `(step, obj) -> None` emits per-step diagnostics.
  - Recoverable statuses are merged via `merge_recoverable_results`; `max_steps_reached` propagates without raising.
  - `simulate_mixture_data` validates simulation domains before random draws and returns `SimulationArrays` payloads on `RESULTS.successful`.
- **Expects**:
  - Array-like inputs only (NumPy-compatible), not dataframe objects.
  - `beta_hat` and `s2` are finite 1D arrays with equal length and strictly positive `s2`.
  - `maf_masks` is a 2D boolean-aligned mask over observations.
  - Simulation configs provide aligned mixture parameter lengths, valid AF generator domains, and positive SE controls.

## Dependencies
- **Uses**: `numpy`, `scipy`, `mut_var.contracts`, `mut_var._core` (Cython), `mut_var.solver`, `mut_var.active_set`.
- **Used by**: `src/mut_var/infer.py`, `src/mut_var/curve.py`, and `src/mut_var/simulate.py`.
- **Boundary**:
  - No file I/O, CLI parsing, logging, or dataframe conversion in this domain.
  - User-facing validation belongs at higher-level adapters before numerics execution.
  - Keep public numerics surface exported through `src/mut_var/numerics/__init__.py`.

## Key Decisions
- mix-SQP (Kim et al. 2020) replaces Optimistix for baseline and refit fitting.
- Cython hot path (`_core.pyx`) implements `compute_grad_hess`, `compute_objective`, `line_search` using BLAS.
- Active-set QP (`active_set.py`) implements the inner QP for both unconstrained (nonneg) and ordered variants.
- Ordering constraint `A π ≤ 0` (hard constraint) replaces the soft penalty term in refit.
- `simulate_mixture_data` uses `numpy.random.default_rng(seed)` for reproducibility.
- Curve fitting uses `scipy.optimize.least_squares(method='lm')`.
- JAX, Equinox, and Optimistix have been fully removed from the numerics stack.

## Invariants
- `Params.pi` sums to 1.0 (normalized) after mix-SQP convergence.
- `fit_refit_grid` returns one model per threshold plus the initial model on successful/recoverable runs.
- Solver outputs always report one canonical status: `successful`, `invalid_input`, `empty_subset`, `nonfinite_objective`, or `max_steps_reached`.
- Successful simulation outputs have finite arrays with strictly positive `sigma2`.
- `simulate_mixture_data` is reproducible for a fixed `(seed, config, n_rows)` tuple.

## Key Files
- `src/mut_var/numerics/baseline.py` - baseline mixture fitting with mix-SQP.
- `src/mut_var/numerics/refit.py` - grid refit with mix-SQP-ordered.
- `src/mut_var/solver.py` - outer SQP loop (`mix_sqp`, `mix_sqp_ordered`, `build_ordering_matrix`).
- `src/mut_var/active_set.py` - NumPy active-set inner QP (`solve_qp_nonneg`, `solve_qp_ordered`).
- `src/mut_var/_core.pyx` - Cython BLAS hot path.
- `src/mut_var/numerics/curve_fit.py` - curve least-squares fitting kernel.
- `src/mut_var/numerics/simulate.py` - mixture simulation validation and sampling kernel.

## Gotchas
- `RESULTS.max_steps_reached` is intentionally treated as recoverable by numerics pipeline utilities.
- Changing `Solution.stats` keys can break regression assertions that inspect diagnostics.
- `simulate_mixture_data` callers must check `Solution.result` (not just `value`) before reading simulation arrays.
- The Cython `_core` extension must be compiled before importing (`pip install -e .` or `python setup.py build_ext --inplace`).
