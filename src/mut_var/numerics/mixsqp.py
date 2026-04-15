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
    m = H.shape[0]
    y = np.maximum(x0.astype(float, copy=True), 0.0)
    W: set[int] = set(int(i) for i in np.where(y <= 0.0)[0])

    for _ in range(max_iter):
        g = H @ y + a
        F = [i for i in range(m) if i not in W]

        if not F:
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
            neg = [(i, float(g[i])) for i in W if g[i] < -tol]
            if not neg:
                break
            W.discard(min(neg, key=lambda t: t[1])[0])
            continue

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
    W_bound: set[int] = set()
    W_ord: set[int] = set(range(p))

    for _ in range(max_iter):
        W_ord_list = sorted(W_ord)
        F = [i for i in range(m) if i not in W_bound]
        nF = len(F)
        nO = len(W_ord_list)

        g = H @ y + a

        if nF == 0:
            neg = [(i, float(g[i])) for i in W_bound if g[i] < -tol]
            if not neg:
                break
            W_bound.discard(min(neg, key=lambda t: t[1])[0])
            continue

        F_arr = np.asarray(F, dtype=int)

        if nO > 0:
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
            if nO > 0:
                A_W_full = A[W_ord_list, :]
                lam = g + A_W_full.T @ nu
            else:
                lam = g.copy()

            neg_b = [(i, float(lam[i])) for i in W_bound if lam[i] < -tol]
            neg_o = [(W_ord_list[k], float(nu[k])) for k in range(nO) if nu[k] < -tol]

            if not neg_b and not neg_o:
                break

            min_b = min((v for _, v in neg_b), default=0.0)
            min_o = min((v for _, v in neg_o), default=0.0)

            if neg_o and min_o <= min_b:
                W_ord.discard(min(neg_o, key=lambda t: t[1])[0])
            else:
                W_bound.discard(min(neg_b, key=lambda t: t[1])[0])
            continue

        alpha = 1.0
        blocking_b: int | None = None
        blocking_o: int | None = None

        for i in F:
            if h[i] < -tol:
                ratio = float(-y[i] / h[i])
                if ratio < alpha:
                    alpha = ratio
                    blocking_b = i
                    blocking_o = None

        Ah = A @ h
        Ay = A @ y
        for j in range(p):
            if j not in W_ord and Ah[j] > tol:
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


def build_ordering_matrix(baseline: np.ndarray) -> np.ndarray:
    r"""Build the bidiagonal ordering-constraint matrix from baseline proportions.

    **Arguments:**

    - `baseline`: ``(m,)`` baseline mixture proportions (need not sum to 1).

    **Returns:**

    - ``A``: ``(m-1, m)`` constraint matrix; ``A pi <= 0`` encodes the ordering.
    """
    b = np.asarray(baseline, dtype=float)
    m = len(b)
    A = np.zeros((m - 1, m))
    for i in range(m - 1):
        A[i, i] = -b[i + 1]
        A[i, i + 1] = b[i]
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
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def mix_sqp(
    L: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    inner_max_iter: int = 200,
    verbose: bool | Callable[..., Any] = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Maximum-likelihood estimation of mixture proportions via mix-SQP."""
    L = np.asarray(L, dtype=float)
    _n, m = L.shape

    row_max = L.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).tiny)
    L_f = np.asfortranarray(L / row_max)

    x = _init_weights(x0, m)

    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(L.shape[0])
    B = np.zeros((L.shape[0], m), order="F")
    x_try = np.zeros(m)

    f = compute_objective(L_f, x, q.copy())
    log_fn = _make_log_fn(verbose)

    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        compute_grad_hess(L_f, x, g, H, q, B)
        a = g - H @ x
        y_star = solve_qp_nonneg(H, a, x.copy(), max_iter=inner_max_iter)
        p = y_star - x

        alpha = line_search(L_f, x, p, f, g, q.copy(), x_try.copy())
        x_new = np.maximum(x + alpha * p, 0.0)
        f_new = compute_objective(L_f, x_new, q.copy())

        if log_fn is not None:
            log_fn(step=iteration + 1, obj=float(f_new))

        rel_change = abs(f_new - f) / (1.0 + abs(f))
        x = x_new
        f = f_new
        n_iter = iteration + 1

        if rel_change < tol:
            converged = True
            break

    pi = _normalise(x)
    return pi, {"converged": converged, "n_iter": n_iter, "objective": float(f)}


def mix_sqp_ordered(
    L: np.ndarray,
    A: np.ndarray,
    baseline: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    inner_max_iter: int = 200,
    verbose: bool | Callable[..., Any] = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""mix-SQP with ordering constraints ``A pi <= 0``."""
    L = np.asarray(L, dtype=float)
    A = np.asarray(A, dtype=float)
    _n, m = L.shape

    row_max = L.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).tiny)
    L_f = np.asfortranarray(L / row_max)

    if x0 is None:
        b = np.asarray(baseline, dtype=float)
        s = b.sum()
        x = b / s if s > 0 else np.ones(m) / m
    else:
        x = _init_weights(x0, m)

    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(L.shape[0])
    B = np.zeros((L.shape[0], m), order="F")
    x_try = np.zeros(m)

    f = compute_objective(L_f, x, q.copy())
    log_fn = _make_log_fn(verbose)

    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        compute_grad_hess(L_f, x, g, H, q, B)
        a = g - H @ x
        y_star = solve_qp_ordered(H, a, A, x.copy(), max_iter=inner_max_iter)
        p = y_star - x

        alpha = line_search(L_f, x, p, f, g, q.copy(), x_try.copy())
        x_new = np.maximum(x + alpha * p, 0.0)
        f_new = compute_objective(L_f, x_new, q.copy())

        if log_fn is not None:
            log_fn(step=iteration + 1, obj=float(f_new))

        rel_change = abs(f_new - f) / (1.0 + abs(f))
        x = x_new
        f = f_new
        n_iter = iteration + 1

        if rel_change < tol:
            converged = True
            break

    pi = _normalise(x)
    return pi, {"converged": converged, "n_iter": n_iter, "objective": float(f)}


__all__ = [
    "build_ordering_matrix",
    "is_recoverable_result",
    "merge_recoverable_results",
    "mix_sqp",
    "mix_sqp_ordered",
    "solve_qp_nonneg",
    "solve_qp_ordered",
]
