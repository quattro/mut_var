# Solver Feasibility Matrix

| Solver Aspect | Feasibility | Basis | Notes |
| --- | --- | --- | --- |
| Single-variable pi optimization | Yes | Existing simplex solver already optimizes `pi` on the probability simplex. | No new optimizer family required. |
| Fixed `mu_k` / `var_k` | Yes | Component grid is already initialized deterministically. | Keep values as output metadata only. |
| Precomputed likelihood matrix | Yes | Inputs and component grid are constant during one inference run. | Compute once before Optimistix loop. |
| Optimistix integration | Yes | Current solver path already uses Optimistix and BacktrackingArmijo. | Capture fixed data in closure, pass `pi` only. |
| Failure signaling | Yes | Existing `RESULTS` / `Solution` contract is already the numerics status channel. | Preserve explicit non-success statuses. |
| JAX tracing stability | Yes | Array inputs can be normalized at ingress before solver execution. | Keep dataframe objects out of the numerics stage. |
| New manifold / geometry work | No | Existing simplex geometry is sufficient. | Do not introduce a new manifold layer. |
