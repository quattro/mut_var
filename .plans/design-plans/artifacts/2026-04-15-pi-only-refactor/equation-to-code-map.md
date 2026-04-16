# Equation to Code Map

## Likelihood Precomputation

- Mathematical form: build the component likelihood matrix once from fixed `mu_k` and `var_k`.
- Code path: `prepare_fit_state(beta_hat, s2, config)` in `src/mut_var/numerics/mixture_fit.py`.
- Purpose: move all fixed component math out of the solver loop.

## Baseline Objective

- Mathematical form: minimize `f(pi) = -log L(pi) + penalty(pi)` over the simplex.
- Code path: `_baseline_objective(pi, L, alpha)` captured inside the Optimistix closure.
- Solver contract: `MutVarSolver` receives only `pi` as the optimization variable.

## Refit Objective

- Mathematical form: optimize `pi` on a MAF-subset likelihood matrix with the ordering penalty.
- Code path: `_refit_objective(pi, L_sub, weights, alpha, baseline_pi)`.
- Output contract: `Params(pi, mu_k, var_k)` preserves fixed component metadata.

## Simplex Geometry

- Mathematical form: updates remain on the probability simplex, preserving non-negativity and unit sum.
- Code path: `simplex_tangent_direction` and `exponential_map_simplex` in the numerics layer.
- Solver contract: no new manifold family is introduced for this phase.

## Ingress Conversion

- Mathematical form: convert validated tabular columns into array-backed ingress data.
- Code path: `load_inference_arrays(...)` and `to_inference_arrays(...)` in `src/mut_var/io.py`.
- Boundary rule: validation happens before numerics; conversion is host-side and happens once.
