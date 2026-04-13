from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from scipy.stats import norm as scipy_norm

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics.solver import mix_sqp


class Params(NamedTuple):
    pi: np.ndarray
    mu_k: np.ndarray
    var_k: np.ndarray


class BaselineConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-3
    # step_size kept for API compatibility; not used by mix-SQP.
    step_size: float = 0.01


def _build_likelihood_matrix(
    beta_hat: np.ndarray,
    s2: np.ndarray,
    mu_k: np.ndarray,
    var_k: np.ndarray,
) -> np.ndarray:
    """Build ``(n, K)`` likelihood matrix for a zero-mean normal mixture.

    Column 0 is the null component ``N(beta; 0, s2[j])``.
    Column ``k > 0`` is ``N(beta; 0, s2[j] + var_k[k-1])``.
    """
    n = len(beta_hat)
    K = len(var_k) + 1
    L = np.empty((n, K), dtype=float)
    # Null component: variance = s2 (pure noise, zero effect).
    L[:, 0] = scipy_norm.pdf(beta_hat, loc=0.0, scale=np.sqrt(s2))
    for k in range(1, K):
        L[:, k] = scipy_norm.pdf(beta_hat, loc=float(mu_k[k - 1]), scale=np.sqrt(s2 + var_k[k - 1]))
    return L


def _validate_inputs(
    beta_hat: Any,
    s2: Any,
    config: BaselineConfig,
) -> Solution | None:
    if hasattr(beta_hat, "columns") or hasattr(s2, "columns"):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "baseline kernel expects arrays, not tabular objects"},
        )

    if config.num_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be >= 2"},
        )

    try:
        beta_hat_arr = np.asarray(beta_hat, dtype=float)
        s2_arr = np.asarray(s2, dtype=float)
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"failed to convert inputs to arrays: {exc}"},
        )

    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1 or beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays of equal length"},
        )

    if beta_hat_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "no variants available"},
        )

    if not np.isfinite(beta_hat_arr).all() or not np.isfinite(s2_arr).all():
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "inputs contain non-finite values"},
        )

    if (s2_arr <= 0.0).any():
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "s2 must be strictly positive"},
        )

    return None


def fit_baseline(
    beta_hat: Any,
    s2: Any,
    config: BaselineConfig,
    verbose: bool | Callable[..., None] = False,
) -> Solution:
    r"""Fit baseline mixture weights via mix-SQP on a fixed variance grid.

    Component means are zero; variances are fixed on a log-spaced grid
    derived from the data range.  Only the mixture weights ``pi`` are
    optimised using the mix-SQP algorithm.

    **Arguments:**

    - `beta_hat`: 1D effect-size estimates.
    - `s2`: 1D positive observation variances aligned with `beta_hat`.
    - `config`: Baseline solver controls.
    - `verbose`: ``False`` for silent; ``True`` prints each SQP step;
      a callable receives ``(step, obj)`` keyword arguments.

    **Returns:**

    - `Solution` carrying fitted `Params` and status diagnostics.

    **Failure Modes:**

    - `RESULTS.invalid_input` for shape/domain violations.
    - `RESULTS.empty_subset` for empty arrays.
    - `RESULTS.nonfinite_objective` when the likelihood matrix is non-finite.
    - `RESULTS.max_steps_reached` when mix-SQP does not converge in `max_iter`.
    """
    invalid = _validate_inputs(beta_hat, s2, config)
    if invalid is not None:
        return invalid

    beta_hat_arr = np.asarray(beta_hat, dtype=float)
    s2_arr = np.asarray(s2, dtype=float)

    # Build log-spaced variance grid for the K-1 non-null components.
    std_err = np.sqrt(s2_arr)
    min_val = float(np.min(std_err)) / 10.0
    max_candidate = float(np.max(beta_hat_arr**2 - s2_arr))
    if max_candidate <= 0.0 or not np.isfinite(max_candidate):
        max_val = 8.0 * min_val
    else:
        max_val = 2.0 * np.sqrt(max_candidate)
    if not np.isfinite(max_val) or max_val <= 0.0:
        max_val = 8.0 * min_val

    var_k = np.exp(np.linspace(np.log(min_val), np.log(max_val), config.num_clusters - 1)) ** 2
    mu_k = np.zeros(config.num_clusters - 1)

    L = _build_likelihood_matrix(beta_hat_arr, s2_arr, mu_k, var_k)

    if not np.isfinite(L).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "likelihood matrix contains non-finite values"},
        )

    # Replace any zero rows (all-zero likelihoods) with a tiny floor so the
    # log-objective remains defined.  This can happen for extreme beta values.
    row_sums = L.sum(axis=1)
    zero_rows = row_sums == 0.0
    if zero_rows.any():
        L[zero_rows, :] = np.finfo(float).tiny

    try:
        pi, info = mix_sqp(
            L,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=verbose,
        )
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"mix-SQP failed: {exc}"},
        )

    result = RESULTS.successful if info["converged"] else RESULTS.max_steps_reached
    return Solution(
        value=Params(pi=pi, mu_k=mu_k, var_k=var_k),
        result=result,
        stats={
            "epoch_count": info["n_iter"],
            "objective": info["objective"],
            "converged": info["converged"],
            "num_observations": int(len(beta_hat_arr)),
            "used_full_batch_objective": True,
        },
    )
