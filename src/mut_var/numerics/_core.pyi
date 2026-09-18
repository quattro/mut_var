# pattern: Functional Core
"""Typing interface for the compiled Cython/BLAS kernels."""

import numpy as np

def compute_grad_hess(
    L: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    w_sum: float,
    g: np.ndarray,
    H: np.ndarray,
    q: np.ndarray,
    B: np.ndarray,
) -> None: ...
def compute_objective(
    L: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    w_sum: float,
    q: np.ndarray,
) -> float: ...
def line_search(
    L: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    w_sum: float,
    p: np.ndarray,
    f0: float,
    g: np.ndarray,
    q: np.ndarray,
    x_try: np.ndarray,
    alpha: float = ...,
    rho: float = ...,
    c: float = ...,
    max_iter: int = ...,
) -> float: ...
