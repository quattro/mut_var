from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mut_var.numerics._core import compute_grad_hess, compute_objective, line_search
from mut_var.numerics.active_set import solve_qp_nonneg, solve_qp_ordered

"""
Outer SQP loop for mix-SQP mixture proportion estimation.

Public API:
    mix_sqp          — standard mix-SQP (simplex constraint only)
    mix_sqp_ordered  — mix-SQP with ordering constraints  A pi <= 0
    build_ordering_matrix — construct bidiagonal A from baseline proportions

References:
    Kim, Carbonetto, Stephens & Anitescu (2020). "A Fast Algorithm for Maximum
    Likelihood Estimation of Mixture Proportions Using Sequential Quadratic
    Programming." JCGS 29(2), 261–273.
"""


def build_ordering_matrix(baseline: np.ndarray) -> np.ndarray:
    r"""Build the bidiagonal ordering-constraint matrix from baseline proportions.

    Encodes the constraint that ``pi / baseline`` is nonincreasing:

    $$b_i \,\pi_{i+1} - b_{i+1}\,\pi_i \le 0 \quad \forall\, i$$

    where ``b = baseline``.

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


def _make_log_fn(verbose: bool | Callable[..., Any]) -> Callable[..., None] | None:
    if callable(verbose):
        return verbose
    if verbose is True:

        def _default(step: int, obj: float, **_kw: Any) -> None:
            print(f"Step: {step}, obj: {obj:.6f}")

        return _default
    return None


def mix_sqp(
    L: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    inner_max_iter: int = 200,
    verbose: bool | Callable[..., Any] = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Maximum-likelihood estimation of mixture proportions via mix-SQP.

    Solves the nonneg-reformulated objective (Proposition 3.2):

    $$\min_{x \ge 0} \tilde{f}(x) = -\frac{1}{n}\sum_j \log\!\left(\sum_k L_{jk}\,x_k\right) + \sum_k x_k$$

    At the optimum ``x`` automatically normalises to a valid simplex point.

    **Arguments:**

    - `L`: ``(n, m)`` likelihood matrix; ``L[j,k] = p(data_j | component k)``.
    - `x0`: Initial weights; defaults to uniform ``1/m``.
    - `max_iter`: Maximum SQP outer iterations.
    - `tol`: Relative-objective convergence tolerance.
    - `inner_max_iter`: Maximum active-set inner iterations per SQP step.
    - `verbose`: ``True`` prints progress; a callable receives ``(step, obj)``.

    **Returns:**

    - ``(pi, info)``: Normalised mixture proportions and a diagnostics dict.
    """
    L = np.asarray(L, dtype=float)
    n, m = L.shape

    # Row-normalise for numerical stability (scale-invariant objective).
    row_max = L.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).tiny)
    L_f = np.asfortranarray(L / row_max)

    x = _init_weights(x0, m)

    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(n)
    B = np.zeros((n, m), order="F")
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
    r"""mix-SQP with ordering constraints ``A pi <= 0``.

    Extends ``mix_sqp`` with homogeneous linear inequality constraints that
    enforce ``pi / baseline`` to be nonincreasing across components.

    **Arguments:**

    - `L`: ``(n, m)`` likelihood matrix.
    - `A`: ``(p, m)`` constraint matrix from :func:`build_ordering_matrix`.
    - `baseline`: ``(m,)`` baseline proportions; used as the default ``x0``.
    - `x0`: Initial weights; defaults to ``baseline / sum(baseline)``.
    - `max_iter`: Maximum SQP outer iterations.
    - `tol`: Relative-objective convergence tolerance.
    - `inner_max_iter`: Maximum active-set inner iterations per SQP step.
    - `verbose`: ``True`` prints progress; a callable receives ``(step, obj)``.

    **Returns:**

    - ``(pi, info)``: Normalised mixture proportions and a diagnostics dict.
    """
    L = np.asarray(L, dtype=float)
    A = np.asarray(A, dtype=float)
    n, m = L.shape

    # Row-normalise for numerical stability.
    row_max = L.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).tiny)
    L_f = np.asfortranarray(L / row_max)

    # Default init: proportional to baseline → A x0 = 0 (all ordering constraints
    # hold at equality), giving the warm-start described in the design notes.
    if x0 is None:
        b = np.asarray(baseline, dtype=float)
        s = b.sum()
        x = b / s if s > 0 else np.ones(m) / m
    else:
        x = _init_weights(x0, m)

    g = np.zeros(m)
    H = np.zeros((m, m), order="F")
    q = np.zeros(n)
    B = np.zeros((n, m), order="F")
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

        rel_change = abs(f_new - f)  # / (1.0 + abs(f))
        x = x_new
        f = f_new
        n_iter = iteration + 1

        if rel_change < tol:
            converged = True
            break

    pi = _normalise(x)
    return pi, {"converged": converged, "n_iter": n_iter, "objective": float(f)}


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
