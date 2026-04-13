# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython hot path for mix-SQP: gradient, Hessian, objective, and line search.

All functions that scale with n (number of observations) live here.
BLAS routines via scipy.linalg.cython_blas with preallocated Fortran-order
workspaces. All hot-path functions are nogil.

L must always be passed as a Fortran-order (column-major) float64 array.
"""

cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport log
from scipy.linalg.cython_blas cimport dgemv, dsyrk

cnp.import_array()

DTYPE = np.float64


def compute_grad_hess(
    cnp.ndarray[DTYPE_t, ndim=2] L,
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=1] g,
    cnp.ndarray[DTYPE_t, ndim=2] H,
    cnp.ndarray[DTYPE_t, ndim=1] q,
    cnp.ndarray[DTYPE_t, ndim=2] B,
):
    r"""Compute gradient and Hessian of the nonneg-reformulated objective f̃.

    **Arguments:**

    - `L`: ``(n, m)`` Fortran-order likelihood matrix.
    - `x`: ``(m,)`` current nonneg weights.
    - `g`: ``(m,)`` pre-allocated gradient output; overwritten in place.
    - `H`: ``(m, m)`` Fortran-order pre-allocated Hessian output; overwritten.
    - `q`: ``(n,)`` workspace; holds ``1/(Lx)`` after the call.
    - `B`: ``(n, m)`` Fortran-order workspace; holds ``diag(d) @ L`` after call.

    **Returns:**

    - Nothing; ``g`` and ``H`` are updated in place.

    Gradient:  ``g_k = -(1/n) sum_j L[j,k] / (Lx)[j] + 1``
    Hessian:   ``H_kl = (1/n) sum_j L[j,k] L[j,l] / (Lx)[j]^2``
    """
    cdef int n = L.shape[0]
    cdef int m = L.shape[1]
    cdef int i, k
    cdef double inv_n = 1.0 / n
    cdef double neg_inv_n = -1.0 / n
    cdef double one_d = 1.0
    cdef double zero_d = 0.0
    cdef int one_i = 1
    cdef char trans_N = b'N'
    cdef char trans_T = b'T'
    cdef char uplo_U = b'U'

    # q = L @ x  (Fortran-order L: leading dim = n)
    dgemv(&trans_N, &n, &m, &one_d, &L[0, 0], &n, &x[0], &one_i,
          &zero_d, &q[0], &one_i)

    # q <- 1/q  (d_j = 1 / (Lx)_j)
    with nogil:
        for i in range(n):
            q[i] = 1.0 / q[i]

    # g = -(1/n) L.T @ d + 1
    dgemv(&trans_T, &n, &m, &neg_inv_n, &L[0, 0], &n, &q[0], &one_i,
          &zero_d, &g[0], &one_i)
    with nogil:
        for k in range(m):
            g[k] += 1.0

    # B[i,k] = d[i] * L[i,k]  (scale rows of L by d = 1/(Lx))
    with nogil:
        for i in range(n):
            for k in range(m):
                B[i, k] = q[i] * L[i, k]

    # H = (1/n) B.T @ B  via dsyrk (fills upper triangle only)
    dsyrk(&uplo_U, &trans_T, &m, &n, &inv_n, &B[0, 0], &n,
          &zero_d, &H[0, 0], &m)

    # Reflect upper triangle -> lower triangle (symmetrise)
    with nogil:
        for i in range(m):
            for k in range(i):
                H[i, k] = H[k, i]


def compute_objective(
    cnp.ndarray[DTYPE_t, ndim=2] L,
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=1] q,
) -> double:
    r"""Evaluate the nonneg-reformulated objective f̃ at x.

    **Arguments:**

    - `L`: ``(n, m)`` Fortran-order likelihood matrix.
    - `x`: ``(m,)`` nonneg weights.
    - `q`: ``(n,)`` workspace; overwritten with ``L @ x``.

    **Returns:**

    - ``f̃(x) = -(1/n) sum_j log((Lx)_j) + sum_k x_k``
    """
    cdef int n = L.shape[0]
    cdef int m = L.shape[1]
    cdef double one_d = 1.0
    cdef double zero_d = 0.0
    cdef int one_i = 1
    cdef char trans_N = b'N'
    cdef double obj = 0.0
    cdef int i, k

    dgemv(&trans_N, &n, &m, &one_d, &L[0, 0], &n, &x[0], &one_i,
          &zero_d, &q[0], &one_i)

    with nogil:
        for i in range(n):
            obj -= log(q[i])
        obj /= n
        for k in range(m):
            obj += x[k]

    return obj


def line_search(
    cnp.ndarray[DTYPE_t, ndim=2] L,
    cnp.ndarray[DTYPE_t, ndim=1] x,
    cnp.ndarray[DTYPE_t, ndim=1] p,
    double f0,
    cnp.ndarray[DTYPE_t, ndim=1] g,
    cnp.ndarray[DTYPE_t, ndim=1] q,
    cnp.ndarray[DTYPE_t, ndim=1] x_try,
    double alpha = 1.0,
    double rho = 0.5,
    double c = 1e-4,
    int max_iter = 50,
) -> double:
    r"""Armijo backtracking line search for the nonneg-reformulated objective.

    **Arguments:**

    - `L`: ``(n, m)`` Fortran-order likelihood matrix.
    - `x`: ``(m,)`` current point.
    - `p`: ``(m,)`` search direction.
    - `f0`: Objective value at ``x``.
    - `g`: ``(m,)`` gradient at ``x``.
    - `q`: ``(n,)`` workspace.
    - `x_try`: ``(m,)`` workspace for trial point.
    - `alpha`: Initial step size (default 1.0).
    - `rho`: Step-size reduction factor (default 0.5).
    - `c`: Armijo sufficient-decrease constant (default 1e-4).
    - `max_iter`: Maximum backtracking iterations (default 50).

    **Returns:**

    - Step size satisfying the Armijo condition, or the smallest tried.
    """
    cdef int n = L.shape[0]
    cdef int m = L.shape[1]
    cdef double gTp = 0.0
    cdef double f_try, qi_val
    cdef int i, k, _iter
    cdef int one_i = 1
    cdef double one_d = 1.0
    cdef double zero_d = 0.0
    cdef char trans_N = b'N'

    with nogil:
        for k in range(m):
            gTp += g[k] * p[k]

    for _iter in range(max_iter):
        with nogil:
            for k in range(m):
                x_try[k] = x[k] + alpha * p[k]
                if x_try[k] < 0.0:
                    x_try[k] = 0.0

        dgemv(&trans_N, &n, &m, &one_d, &L[0, 0], &n, &x_try[0], &one_i,
              &zero_d, &q[0], &one_i)

        f_try = 0.0
        with nogil:
            for i in range(n):
                qi_val = q[i]
                if qi_val <= 0.0:
                    qi_val = 1e-300
                f_try -= log(qi_val)
            f_try /= n
            for k in range(m):
                f_try += x_try[k]

        if f_try <= f0 + c * alpha * gTp:
            return alpha

        alpha *= rho

    return alpha
