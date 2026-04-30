# pattern: Functional Core
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mut_var.numerics._core import compute_grad_hess, compute_objective, line_search
from mut_var.types import RESULTS

RECOVERABLE_RESULTS = (RESULTS.successful, RESULTS.max_steps_reached)


def solve_qp_nonneg(
    H: np.ndarray,
    a: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""Solve ``min 1/2 y'Hy + y'a  s.t.  y >= 0`` by active-set (N&W Alg 16.3).

    **Arguments:**

    - `H`: ``(m, m)`` symmetric positive-semidefinite Hessian.
    - `a`: ``(m,)`` linear term.
    - `x0`: ``(m,)`` feasible starting point (nonneg).
    - `max_iter`: Maximum active-set iterations.
    - `tol`: KKT tolerance for declaring optimality.

    **Returns:**

    - ``y*``: ``(m,)`` optimal nonneg solution.
    """
    # Active-set strategy: maintain W, the set of variables currently fixed to
    # zero (the "active" bound constraints). At each iteration we work only in
    # the subspace F = {0..m-1} \ W of "free" variables, where the bound is
    # inactive, and compute a Newton step restricted to that subspace.
    m = H.shape[0]
    y = np.maximum(x0.astype(float, copy=True), 0.0)
    # Initialise W to the variables already at zero in the starting point.
    W: set[int] = set(int(i) for i in np.where(y <= 0.0)[0])

    for _ in range(max_iter):
        g = H @ y + a  # gradient of 1/2 y'Hy + a'y at current y
        F = [i for i in range(m) if i not in W]

        if not F:
            # All variables are on the zero bound. We're at a vertex of the
            # feasible region. Check KKT: the multiplier for variable i in W
            # is g[i]. If all multipliers are >= 0 the KKT conditions hold and
            # this is the optimum. Otherwise release the most-negative one.
            neg = [(i, float(g[i])) for i in W if g[i] < -tol]
            if not neg:
                break
            W.discard(min(neg, key=lambda t: t[1])[0])
            continue

        # Compute the Newton step in the free subspace. This solves the
        # equality-constrained problem obtained by holding all active variables
        # fixed at zero: H_FF h_F = -g_F.
        F_arr = np.asarray(F, dtype=int)
        H_FF = H[np.ix_(F_arr, F_arr)]
        g_F = g[F_arr]

        try:
            h_F = np.linalg.solve(H_FF, -g_F)
        except np.linalg.LinAlgError:
            # Fall back to least-squares if H_FF is singular (can happen when
            # multiple components share identical likelihood columns).
            h_F, *_ = np.linalg.lstsq(H_FF, -g_F, rcond=None)

        # Embed the free-subspace step back into the full space.
        h = np.zeros(m)
        h[F_arr] = h_F

        if np.max(np.abs(h)) < tol:
            # Step is negligible: we're at a constrained stationary point
            # within the current active set. Check KKT multipliers to decide
            # whether we can drop a constraint and improve the objective.
            neg = [(i, float(g[i])) for i in W if g[i] < -tol]
            if not neg:
                break  # All multipliers non-negative: KKT satisfied.
            W.discard(min(neg, key=lambda t: t[1])[0])
            continue

        # Ratio test: find the largest step alpha in [0, 1] before any free
        # variable hits the zero bound (the "blocking" variable). alpha = 1
        # means the full Newton step is feasible.
        alpha = 1.0
        blocking: int | None = None
        for i in F:
            if h[i] < -tol:  # variable i would go negative
                ratio = float(-y[i] / h[i])
                if ratio < alpha:
                    alpha = ratio
                    blocking = i

        y = y + alpha * h
        y = np.maximum(y, 0.0)  # numerical clean-up

        # If a variable hit zero, add it to the active set.
        if blocking is not None:
            W.add(blocking)

    return y


def solve_qp_ordered(
    H: np.ndarray,
    a: np.ndarray,
    A: np.ndarray,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""Solve ``min 1/2 y'Hy + y'a  s.t.  y >= 0, Ay <= 0`` by extended active-set.

    **Arguments:**

    - `H`: ``(m, m)`` symmetric positive-semidefinite Hessian.
    - `a`: ``(m,)`` linear term.
    - `A`: ``(p, m)`` linear constraint matrix; ``Ay <= 0`` at solution.
    - `x0`: ``(m,)`` feasible starting point (``x0 >= 0``, ``A x0 <= 0``).
    - `max_iter`: Maximum active-set iterations.
    - `tol`: KKT tolerance.

    **Returns:**

    - ``y*``: ``(m,)`` optimal solution satisfying both constraint families.
    """
    m = H.shape[0]
    p = A.shape[0]

    # With no linear constraints this reduces to the simpler bound-only problem.
    if p == 0:
        return solve_qp_nonneg(H, a, x0, max_iter, tol)

    # Two active sets for the two constraint families:
    #   W_bound  — bound constraints y_i >= 0 that are currently active (y_i = 0).
    #   W_ord    — linear constraints (Ay)_j <= 0 that are currently active.
    y = np.maximum(x0.astype(float, copy=True), 0.0)
    W_bound: set[int] = set(int(i) for i in np.where(y <= tol)[0])
    W_ord: set[int] = set(int(i) for i in np.where(A @ y >= -tol)[0])

    for _ in range(max_iter):
        W_ord_list = sorted(W_ord)
        F = [i for i in range(m) if i not in W_bound]  # free (non-zero-bound) variables
        nF = len(F)
        nO = len(W_ord_list)

        g = H @ y + a  # gradient at current y

        if nF == 0:
            # All variables on the zero bound; check bound KKT multipliers.
            neg = [(i, float(g[i])) for i in W_bound if g[i] < -tol]
            if not neg:
                break
            W_bound.discard(min(neg, key=lambda t: t[1])[0])
            continue

        F_arr = np.asarray(F, dtype=int)

        if nO > 0:
            # Build and solve the KKT system for the current active set.
            # Restricting to free variables and active linear constraints:
            #
            #   [ H_FF   A_WF' ] [ h_F ]   [ -g_F ]
            #   [ A_WF   0     ] [ nu  ] = [  0   ]
            #
            # where nu are the Lagrange multipliers for the active linear
            # constraints. The lower block enforces A_WF h_F = 0 (the active
            # linear constraints don't change within this step).
            A_WF = A[np.ix_(W_ord_list, F_arr)]
            H_FF = H[np.ix_(F_arr, F_arr)]

            KKT = np.zeros((nF + nO, nF + nO))
            KKT[:nF, :nF] = H_FF
            KKT[:nF, nF:] = A_WF.T
            KKT[nF:, :nF] = A_WF

            rhs = np.zeros(nF + nO)
            rhs[:nF] = -g[F_arr]

            sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            h_F = sol[:nF]
            nu = sol[nF:]  # multipliers for active linear constraints
        else:
            # No active linear constraints: pure Newton step in free subspace.
            H_FF = H[np.ix_(F_arr, F_arr)]
            h_F, *_ = np.linalg.lstsq(H_FF, -g[F_arr], rcond=None)
            nu = np.zeros(0)

        # Embed free-subspace step into full space (active-bound variables stay at 0).
        h = np.zeros(m)
        h[F_arr] = h_F

        if np.max(np.abs(h)) < tol:
            # Step is negligible: check KKT conditions for both constraint families.
            # The multiplier for bound constraint i is lam[i] = (g + A_W' nu)[i].
            # The multiplier for linear constraint j (active) is nu[j].
            # KKT requires lam[i] >= 0 for all i in W_bound and nu[j] >= 0 for
            # all j in W_ord. A negative multiplier identifies a constraint that
            # is holding the optimum back and should be dropped from the active set.
            if nO > 0:
                A_W_full = A[W_ord_list, :]
                lam = g + A_W_full.T @ nu
            else:
                lam = g.copy()

            neg_b = [(i, float(lam[i])) for i in W_bound if lam[i] < -tol]
            neg_o = [(W_ord_list[k], float(nu[k])) for k in range(nO) if nu[k] < -tol]

            if not neg_b and not neg_o:
                break  # All KKT conditions satisfied.

            # Drop whichever constraint has the most-negative multiplier.
            min_b = min((v for _, v in neg_b), default=0.0)
            min_o = min((v for _, v in neg_o), default=0.0)

            if neg_o and min_o <= min_b:
                W_ord.discard(min(neg_o, key=lambda t: t[1])[0])
            else:
                W_bound.discard(min(neg_b, key=lambda t: t[1])[0])
            continue

        # Ratio test for both constraint families. Find the longest step alpha
        # in [0, 1] before any constraint is violated, and record which
        # constraint becomes active (the "blocking" constraint).
        alpha = 1.0
        blocking_b: int | None = None
        blocking_o: int | None = None

        # Bound constraints: variable i hits zero when y[i] + alpha*h[i] = 0.
        for i in F:
            if h[i] < -tol:
                ratio = float(-y[i] / h[i])
                if ratio < alpha:
                    alpha = ratio
                    blocking_b = i
                    blocking_o = None  # bound blocks before any linear constraint

        # Linear constraints: constraint j becomes active when (Ay + alpha*Ah)_j = 0.
        # Only check inactive linear constraints; active ones are already at equality.
        Ah = A @ h
        Ay = A @ y
        for j in range(p):
            if j not in W_ord and Ah[j] > tol:  # constraint j moving toward violation
                ratio = float(-Ay[j] / Ah[j]) if Ay[j] < 0.0 else 0.0
                if ratio < alpha:
                    alpha = ratio
                    blocking_o = j
                    blocking_b = None  # linear constraint blocks before any bound

        y = y + alpha * h
        y = np.maximum(y, 0.0)  # numerical clean-up for bound constraints

        # Add whichever constraint became active to its respective active set.
        if blocking_b is not None:
            W_bound.add(blocking_b)
        W_bound.update(int(i) for i in np.where(y <= tol)[0])
        if blocking_o is not None:
            W_ord.add(blocking_o)

    return y


def build_constraints_matrix(baseline: np.ndarray, *, constrain_spike: bool = False) -> np.ndarray:
    r"""Build refit constraints from baseline proportions.

    **Arguments:**

    - `baseline`: ``(m,)`` baseline mixture proportions.
    - `constrain_spike`: If true, include constraints involving the null/spike component.

    **Returns:**

    - ``A``: Constraint matrix; ``A pi <= 0`` encodes adjacent-pair ordering.
      When `constrain_spike` is true, it also encodes the null-weight cap and
      adjacent ordering between the spike and first signal component.

    When `constrain_spike` is true, row 0 encodes the null-weight cap:

    $$\pi_0 \leq b_0$$

    in the homogeneous form used by mix-SQP:

    $$(1-b_0)\pi_0 - b_0 \sum_{j>0}\pi_j \leq 0.$$

    The remaining rows encode, for each adjacent pair of constrained components ``i`` and
    ``i+1``:

    $$b_{i+1} \pi_i - b_i \pi_{i+1} \leq 0
    \implies \frac{\pi_i}{b_i} \leq \frac{\pi_{i+1}}{b_{i+1}}$$

    where ``b`` is the normalised baseline proportion vector. This enforces
    nondecreasing enrichment ``\pi_i / b_i`` across constrained component
    indices, with spike constraints included only when requested.
    """
    b = np.asarray(baseline, dtype=float)
    m = len(b)
    s = b.sum()
    b = b / s if s > 0 else np.ones(m) / m

    start_pair = 0 if constrain_spike else 1
    n_pair_rows = max((m - 1) - start_pair, 0)
    n_rows = n_pair_rows + (1 if constrain_spike else 0)
    A = np.zeros((n_rows, m))
    offset = 0
    if constrain_spike:
        b0 = b[0]
        A[0, :] = -b0
        A[0, 0] = 1.0 - b0
        offset = 1
    for row, i in enumerate(range(start_pair, m - 1), start=offset):
        # Constrain the cross-ratio of adjacent components i and i+1.
        A[row, i] = b[i + 1]
        A[row, i + 1] = -b[i]
    return A


def is_recoverable_result(result: RESULTS) -> bool:
    r"""Return whether a status is recoverable for staged numerics pipelines."""
    return result in RECOVERABLE_RESULTS


def merge_recoverable_results(*results: RESULTS) -> RESULTS:
    r"""Merge recoverable statuses, preferring `max_steps_reached` when present."""
    if any(result == RESULTS.max_steps_reached for result in results):
        return RESULTS.max_steps_reached
    return RESULTS.successful


def _make_log_fn(verbose: bool | Callable[..., Any]) -> Callable[..., None] | None:
    if callable(verbose):
        return verbose
    if verbose is True:

        def _default(step: int, obj: float, **_kw: Any) -> None:
            print(f"Step: {step}, obj: {obj:.6f}")

        return _default
    return None


def _init_weights(x0: np.ndarray | None, m: int) -> np.ndarray:
    if x0 is None:
        return np.ones(m) / m
    x = np.asarray(x0, dtype=float).copy()
    x = np.maximum(x, 0.0)
    s = x.sum()
    return x / s if s > 0 else np.ones(m) / m


def _normalise(x: np.ndarray) -> np.ndarray:
    # mix-SQP works in a nonneg (not simplex) reformulation; the sum(x) penalty
    # drives the iterate close to the simplex, and this projection pins it.
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def _prepare_mixsqp_inputs(
    L: np.ndarray,
    w: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Row-scaling: dividing each row of L by its max does not change the
    # optimal pi (the scaling cancels in the log-likelihood ratio) but keeps
    # the individual likelihoods in a range where BLAS-level arithmetic stays
    # well-conditioned.
    row_max = L.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).tiny)
    L_f = np.asfortranarray(L / row_max)

    if w is None:
        w_arr = np.ones(L.shape[0], dtype=float)
    else:
        w_arr = np.ascontiguousarray(w, dtype=float)
        if w_arr.shape != (L.shape[0],):
            raise ValueError(f"w must have shape ({L.shape[0]},), got {w_arr.shape}")
    w_sum = float(w_arr.sum())
    if w_sum <= 0.0:
        raise ValueError("sum(w) must be positive")
    return L_f, w_arr, w_sum


def mix_sqp(
    L: np.ndarray,
    x0: np.ndarray | None = None,
    w: np.ndarray | None = None,
    max_iter: int = 100,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    inner_max_iter: int = 200,
    verbose: bool | Callable[..., Any] = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Maximum-likelihood estimation of mixture proportions via mix-SQP.

    Maximises the weighted log-likelihood ``(1/W) sum_j w_j log((Lx)_j)``
    subject to ``x >= 0, sum(x) = 1`` using the mix-SQP algorithm (Kim et
    al. 2020), where ``W = sum(w)``. The sum-to-one constraint is handled
    implicitly: the nonneg-reformulated objective adds a ``sum(x)`` penalty
    that drives the simplex normalisation at convergence.

    **Arguments:**

    - `L`: ``(n, m)`` likelihood matrix; ``L[j, k]`` is the likelihood of
      observation ``j`` under component ``k``.
    - `x0`: Optional warm-start weights (normalised to sum to 1).
    - `w`: Optional ``(n,)`` per-row nonneg weights; defaults to ones.
    - `max_iter`: Maximum outer SQP iterations.
    - `atol`: Absolute convergence tolerance for accepted outer steps.
    - `rtol`: Relative convergence tolerance for accepted outer steps.
    - `inner_max_iter`: Maximum iterations for the inner active-set QP solver.
    - `verbose`: If ``True``, print per-step progress. Accepts a callable
      ``(step, obj) -> None`` for custom logging.

    **Returns:**

    - ``(pi, info)``: Normalised mixture proportions and a diagnostics dict
      with keys ``converged``, ``n_iter``, and ``objective``.
    """
    L = np.asarray(L, dtype=float)
    _n, m = L.shape
    L_f, w_arr, w_sum = _prepare_mixsqp_inputs(L, w)

    x = _init_weights(x0, m)

    # Preallocate workspaces passed into the Cython hot path to avoid repeated
    # allocation inside the loop. g and H receive gradient/Hessian in-place;
    # q and B are intermediate buffers reused across iterations.
    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(L.shape[0])
    B = np.zeros((L.shape[0], m), order="F")
    x_try = np.zeros(m)

    f = compute_objective(L_f, x, w_arr, w_sum, q.copy())
    log_fn = _make_log_fn(verbose)

    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        # Outer SQP step: approximate the objective locally by a quadratic
        # using the current gradient g and Hessian H, then solve that QP.
        compute_grad_hess(L_f, x, w_arr, w_sum, g, H, q, B)

        # The local QP is: min 1/2 y'Hy + y'a  s.t.  y >= 0
        # where a = g - Hx shifts the quadratic so the unconstrained minimiser
        # is at y = x (i.e. we're computing a displacement from x).
        a = g - H @ x
        y_star = solve_qp_nonneg(H, a, x.copy(), max_iter=inner_max_iter)
        p = y_star - x  # SQP search direction

        # Armijo backtracking line search along p to ensure sufficient decrease.
        alpha = line_search(L_f, x, w_arr, w_sum, p, f, g, q.copy(), x_try.copy())
        x_new = np.maximum(x + alpha * p, 0.0)
        f_new = compute_objective(L_f, x_new, w_arr, w_sum, q.copy())

        if log_fn is not None:
            log_fn(step=iteration + 1, obj=float(f_new))

        # Two-pronged convergence: the iterate stops moving AND the objective
        # stops improving. Either alone is insufficient (a degenerate step can
        # be small even when the objective is still descending, and vice versa).
        step_norm = np.max(np.abs(x_new - x))
        f_diff = abs(f_new - f)
        step_converged = step_norm <= (atol + rtol)
        objective_converged = f_diff <= (atol + rtol * max(1.0, abs(f_new)))
        x = x_new
        f = f_new
        n_iter = iteration + 1

        if step_converged and objective_converged:
            converged = True
            break

    pi = _normalise(x)
    return pi, {"converged": converged, "n_iter": n_iter, "objective": float(f)}


def mix_sqp_ordered(
    L: np.ndarray,
    A: np.ndarray,
    baseline: np.ndarray,
    x0: np.ndarray | None = None,
    w: np.ndarray | None = None,
    max_iter: int = 100,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    inner_max_iter: int = 200,
    verbose: bool | Callable[..., Any] = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""mix-SQP with refit constraints ``A pi <= 0``.

    Identical to :func:`mix_sqp` except the inner QP is solved subject to the
    additional linear inequality ``A y <= 0``, which is used in the refit stage
    to cap the null component and preserve the relative component ordering from
    the previous fit. See :func:`build_constraints_matrix` for how ``A`` is
    constructed.

    **Arguments:**

    - `L`: ``(n, m)`` likelihood matrix for the current MAF-threshold subset.
    - `A`: Constraint matrix from :func:`build_constraints_matrix`.
    - `baseline`: ``(m,)`` baseline proportions used to initialise ``x`` when
      no ``x0`` is provided.
    - `x0`: Optional warm-start weights.
    - `w`: Optional ``(n,)`` per-row nonneg weights; defaults to ones.
    - `max_iter`: Maximum outer SQP iterations.
    - `atol`: Absolute convergence tolerance for accepted outer steps.
    - `rtol`: Relative convergence tolerance for accepted outer steps.
    - `inner_max_iter`: Maximum iterations for the inner active-set QP solver.
    - `verbose`: Progress callback; same semantics as :func:`mix_sqp`.

    **Returns:**

    - ``(pi, info)``: Normalised mixture proportions and a diagnostics dict
      with keys ``converged``, ``n_iter``, and ``objective``.
    """
    L = np.asarray(L, dtype=float)
    A = np.asarray(A, dtype=float)
    _n, m = L.shape
    L_f, w_arr, w_sum = _prepare_mixsqp_inputs(L, w)

    # Default initialisation from the baseline proportions so the starting
    # point is consistent with the refit constraints.
    if x0 is None:
        b = np.asarray(baseline, dtype=float)
        s = b.sum()
        x = b / s if s > 0 else np.ones(m) / m
    else:
        x = _init_weights(x0, m)

    # Preallocated workspaces; same layout as mix_sqp.
    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(L.shape[0])
    B = np.zeros((L.shape[0], m), order="F")
    x_try = np.zeros(m)

    f = compute_objective(L_f, x, w_arr, w_sum, q.copy())
    log_fn = _make_log_fn(verbose)

    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        compute_grad_hess(L_f, x, w_arr, w_sum, g, H, q, B)

        # Local QP with both bound and refit constraints:
        #   min 1/2 y'Hy + y'a   s.t.   y >= 0,  Ay <= 0
        a = g - H @ x
        y_star = solve_qp_ordered(H, a, A, x.copy(), max_iter=inner_max_iter)
        p = y_star - x

        alpha = line_search(L_f, x, w_arr, w_sum, p, f, g, q.copy(), x_try.copy())
        x_new = np.maximum(x + alpha * p, 0.0)
        f_new = compute_objective(L_f, x_new, w_arr, w_sum, q.copy())

        if log_fn is not None:
            log_fn(step=iteration + 1, obj=float(f_new))

        step_norm = np.max(np.abs(x_new - x))
        f_diff = abs(f_new - f)
        step_converged = step_norm <= (atol + rtol)
        objective_converged = f_diff <= (atol + rtol * max(1.0, abs(f_new)))
        x = x_new
        f = f_new
        n_iter = iteration + 1

        if step_converged and objective_converged:
            converged = True
            break

    pi = _normalise(x)
    return pi, {"converged": converged, "n_iter": n_iter, "objective": float(f)}


__all__ = [
    "build_constraints_matrix",
    "is_recoverable_result",
    "merge_recoverable_results",
    "mix_sqp",
    "mix_sqp_ordered",
    "solve_qp_nonneg",
    "solve_qp_ordered",
]
