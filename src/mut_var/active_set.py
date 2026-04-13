from __future__ import annotations

import numpy as np

"""
NumPy active-set inner QP solver for mix-SQP.

Solves the convex QP subproblem arising at each SQP outer iteration:

    min  1/2 y' H y + y' a
    s.t. y >= 0                    (bound constraints)
         A y <= 0  (optional)      (ordering constraints)

The active-set logic is inherently combinatorial — dynamic working set sizes,
control flow on multiplier signs, changing KKT matrix shapes — so it lives in
Python/NumPy rather than being JIT-compiled.  The arrays are m-sized (number
of mixture components), so np.linalg.solve on the KKT system is microseconds.

References:
    Nocedal & Wright (2006) *Numerical Optimization* 2nd ed., Algorithm 16.3.
"""


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
    m = H.shape[0]
    y = np.maximum(x0.astype(float, copy=True), 0.0)
    W: set[int] = set(int(i) for i in np.where(y <= 0.0)[0])

    for _ in range(max_iter):
        g = H @ y + a  # QP gradient at y
        F = [i for i in range(m) if i not in W]

        if not F:
            # Every variable at bound; check bound multipliers.
            neg = [(i, float(g[i])) for i in W if g[i] < -tol]
            if not neg:
                break
            W.discard(min(neg, key=lambda t: t[1])[0])
            continue

        F_arr = np.asarray(F, dtype=int)
        H_FF = H[np.ix_(F_arr, F_arr)]
        g_F = g[F_arr]

        try:
            h_F = np.linalg.solve(H_FF, -g_F)
        except np.linalg.LinAlgError:
            h_F, *_ = np.linalg.lstsq(H_FF, -g_F, rcond=None)

        h = np.zeros(m)
        h[F_arr] = h_F

        if np.max(np.abs(h)) < tol:
            # Stationary on free subspace; check bound multipliers.
            neg = [(i, float(g[i])) for i in W if g[i] < -tol]
            if not neg:
                break
            W.discard(min(neg, key=lambda t: t[1])[0])
            continue

        # Compute step length: largest alpha keeping y + alpha*h >= 0.
        alpha = 1.0
        blocking: int | None = None
        for i in F:
            if h[i] < -tol:
                ratio = float(-y[i] / h[i])
                if ratio < alpha:
                    alpha = ratio
                    blocking = i

        y = y + alpha * h
        y = np.maximum(y, 0.0)

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

    Extends N&W Algorithm 16.3 to include homogeneous linear inequality
    constraints ``Ay <= 0``.  The working set tracks both bound constraints
    (``W_bound``) and ordering constraints (``W_ord``).

    Warm-start initialization per the mix-SQP notes: ``W_bound = empty``,
    ``W_ord = all rows``.  The solver drops ordering constraints that are
    non-binding at optimum in the first few inner iterations.

    **Arguments:**

    - `H`: ``(m, m)`` symmetric positive-semidefinite Hessian.
    - `a`: ``(m,)`` linear term.
    - `A`: ``(p, m)`` ordering constraint matrix; ``Ay <= 0`` at solution.
    - `x0`: ``(m,)`` feasible starting point (``x0 >= 0``, ``A x0 <= 0``).
    - `max_iter`: Maximum active-set iterations.
    - `tol`: KKT tolerance.

    **Returns:**

    - ``y*``: ``(m,)`` optimal solution satisfying both constraint families.
    """
    m = H.shape[0]
    p = A.shape[0]

    if p == 0:
        return solve_qp_nonneg(H, a, x0, max_iter, tol)

    y = np.maximum(x0.astype(float, copy=True), 0.0)

    # Initial working sets: all ordering constraints active (notes §Initialization).
    W_bound: set[int] = set()
    W_ord: set[int] = set(range(p))

    for _ in range(max_iter):
        W_ord_list = sorted(W_ord)
        F = [i for i in range(m) if i not in W_bound]
        nF = len(F)
        nO = len(W_ord_list)

        g = H @ y + a  # gradient of QP objective at y

        if nF == 0:
            # All variables at bound; check bound multipliers (no nu term here).
            neg = [(i, float(g[i])) for i in W_bound if g[i] < -tol]
            if not neg:
                break
            W_bound.discard(min(neg, key=lambda t: t[1])[0])
            continue

        F_arr = np.asarray(F, dtype=int)

        if nO > 0:
            # KKT system for equality-constrained step on free variables:
            #
            #   [ H_FF    A_WF' ] [ h_F ]   [ -g_F ]
            #   [ A_WF    0     ] [  nu ]   [   0  ]
            #
            # (A_WF h_F = 0 keeps active ordering constraints satisfied.)
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
            nu = sol[nF:]
        else:
            H_FF = H[np.ix_(F_arr, F_arr)]
            h_F, *_ = np.linalg.lstsq(H_FF, -g[F_arr], rcond=None)
            nu = np.zeros(0)

        h = np.zeros(m)
        h[F_arr] = h_F

        if np.max(np.abs(h)) < tol:
            # Near-stationary: check KKT multipliers for all active constraints.
            # Bound multipliers: (H y + a + A_W' nu)[i] for i in W_bound.
            if nO > 0:
                A_W_full = A[W_ord_list, :]
                lam = g + A_W_full.T @ nu
            else:
                lam = g.copy()

            neg_b = [(i, float(lam[i])) for i in W_bound if lam[i] < -tol]
            neg_o = [(W_ord_list[k], float(nu[k])) for k in range(nO) if nu[k] < -tol]

            if not neg_b and not neg_o:
                break  # KKT conditions satisfied

            # Drop the constraint with the most negative multiplier.
            min_b = min((v for _, v in neg_b), default=0.0)
            min_o = min((v for _, v in neg_o), default=0.0)

            if neg_o and min_o <= min_b:
                W_ord.discard(min(neg_o, key=lambda t: t[1])[0])
            else:
                W_bound.discard(min(neg_b, key=lambda t: t[1])[0])
            continue

        # Find the largest step keeping all inactive constraints feasible.
        alpha = 1.0
        blocking_b: int | None = None
        blocking_o: int | None = None

        # Inactive bound constraints: y[i] + alpha * h[i] >= 0 for i in F.
        for i in F:
            if h[i] < -tol:
                ratio = float(-y[i] / h[i])
                if ratio < alpha:
                    alpha = ratio
                    blocking_b = i
                    blocking_o = None

        # Inactive ordering constraints: (Ay + alpha*Ah)[j] <= 0 for j not in W_ord.
        Ah = A @ h
        Ay = A @ y
        for j in range(p):
            if j not in W_ord and Ah[j] > tol:
                # Need alpha <= -Ay[j] / Ah[j].  Ay[j] < 0 for inactive feasible row.
                ratio = float(-Ay[j] / Ah[j]) if Ay[j] < 0.0 else 0.0
                if ratio < alpha:
                    alpha = ratio
                    blocking_o = j
                    blocking_b = None

        y = y + alpha * h
        y = np.maximum(y, 0.0)

        if blocking_b is not None:
            W_bound.add(blocking_b)
        if blocking_o is not None:
            W_ord.add(blocking_o)

    return y
