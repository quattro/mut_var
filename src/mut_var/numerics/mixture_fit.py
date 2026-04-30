# pattern: Functional Core
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from scipy.stats import norm as scipy_norm

from mut_var.numerics.mixsqp import (
    build_constraints_matrix,
    mix_sqp,
    mix_sqp_ordered,
)
from mut_var.types import InferenceConfig, RESULTS, Solution


class Params(NamedTuple):
    pi: np.ndarray
    mu_k: np.ndarray
    var_k: np.ndarray


class FitState(NamedTuple):
    likelihood_matrix: np.ndarray
    initial_params: Params


def _build_likelihood_matrix(
    beta_hat: np.ndarray,
    s2: np.ndarray,
    mu_k: np.ndarray,
    var_k: np.ndarray,
) -> np.ndarray:
    r"""Build ``(n, K)`` likelihood matrix for fixed mixture components.

    Column 0 is the point-mass-at-zero component (normal with variance ``s2``);
    columns ``1..K-1`` cover the continuous components with mean ``mu_k[k-1]``
    and total variance ``s2 + var_k[k-1]``. Sampling variance ``s2`` is folded
    into every component so the likelihood is expressed directly in the
    observed ``beta_hat`` space.
    """
    n = len(beta_hat)
    K = len(var_k) + 1
    L = np.empty((n, K), dtype=float)
    L[:, 0] = scipy_norm.pdf(beta_hat, loc=0.0, scale=np.sqrt(s2))
    for k in range(1, K):
        L[:, k] = scipy_norm.pdf(
            beta_hat,
            loc=float(mu_k[k - 1]),
            scale=np.sqrt(s2 + var_k[k - 1]),
        )
    return L


def _augment_with_prior(L: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(L_aug, w_aug)`` encoding a Dirichlet(prior) log-prior on pi.

    Matches ashr's ``rbind(diag(k), matrix_lik)`` trick: for each component ``k``
    append a single ``e_k`` row and set its weight to ``prior_k - 1``. Rows with
    zero weight (``prior_k == 1``) are dropped so the solver skips them.

    Only real ``prior_k >= 1`` is supported. Values in ``(0, 1)`` flip the sign
    of the log-barrier and break convexity of the solver's objective.
    """
    K = L.shape[1]
    weights = prior - 1.0
    if (weights < 0.0).any():
        raise ValueError("prior entries must be >= 1")
    n = L.shape[0]
    data_w = np.ones(n, dtype=float)
    keep = weights > 0.0
    if not keep.any():
        return L, data_w
    aug_rows = np.eye(K, dtype=float)[keep]
    aug_w = weights[keep]
    return np.vstack([L, aug_rows]), np.concatenate([data_w, aug_w])


def _floor_zero_rows(L: np.ndarray) -> np.ndarray:
    # Rows where every component has likelihood zero would make log(Lx) = -inf
    # anywhere in the simplex. Floor them so those observations contribute a
    # negligible-but-finite term to the objective.
    row_sums = L.sum(axis=1)
    zero_rows = row_sums == 0.0
    if zero_rows.any():
        L = L.copy()
        L[zero_rows, :] = np.finfo(float).tiny
    return L


def _validate_array_inputs(beta_hat: Any, s2: Any) -> Solution:
    # Boundary check for the sole array entrypoint (`prepare_fit_state`).
    # Downstream numerics trust the FitState produced here and do not re-check.
    if hasattr(beta_hat, "columns") or hasattr(s2, "columns"):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "mixture kernel expects arrays, not tabular objects"},
        )

    beta_hat_arr = np.asarray(beta_hat, dtype=float)
    s2_arr = np.asarray(s2, dtype=float)

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

    return Solution(value=(beta_hat_arr, s2_arr), result=RESULTS.successful)


def _build_baseline_params(beta_hat: np.ndarray, s2: np.ndarray, config: InferenceConfig) -> Params:
    # Component grid spans from well below the noise floor to above the
    # largest plausible true variance, on a log scale. Using a method-of-moments
    # upper bound keeps the grid adaptive to the data without overshooting.
    std_err = np.sqrt(s2)
    min_val = float(np.min(std_err)) / 10.0
    # max(beta^2 - s2) is a method-of-moments upper bound on the true effect variance.
    max_candidate = float(np.max(beta_hat**2 - s2))
    if max_candidate <= 0.0 or not np.isfinite(max_candidate):
        # Fallback when all signal is noise: span 3 orders of magnitude above min_val.
        max_val = 8.0 * min_val
    else:
        # 2× the MOM estimate gives comfortable headroom above the observed signal.
        max_val = 2.0 * np.sqrt(max_candidate)
    if not np.isfinite(max_val) or max_val <= 0.0:
        max_val = 8.0 * min_val

    # Squared because var_k stores variances, not standard deviations.
    var_k = np.exp(np.linspace(np.log(min_val), np.log(max_val), config.num_clusters - 1)) ** 2
    mu_k = np.zeros(config.num_clusters - 1)
    pi = np.ones(config.num_clusters, dtype=float) / float(config.num_clusters)
    return Params(pi=pi, mu_k=mu_k, var_k=var_k)


def prepare_fit_state(
    beta_hat: Any,
    s2: Any,
    config: InferenceConfig,
) -> Solution:
    r"""Construct baseline fit state from validated array-like inputs.

    **Arguments:**

    - `beta_hat`: Effect-size estimate array.
    - `s2`: Sampling-variance array.
    - `config`: Inference numerics configuration.

    **Returns:**

    - `Solution` whose `value` is a `FitState` on success.

    **Failure Modes:**

    - Returns `RESULTS.invalid_input` for non-array, non-finite, or shape-invalid
      inputs, or when `config.num_clusters < 2`.
    - Returns `RESULTS.empty_subset` when no observations are available.
    - Returns `RESULTS.nonfinite_objective` when the likelihood matrix is non-finite.
    """
    if config.num_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be >= 2"},
        )

    validated = _validate_array_inputs(beta_hat, s2)
    if not validated.ok:
        return validated
    beta_hat_arr, s2_arr = validated.value

    params = _build_baseline_params(beta_hat_arr, s2_arr, config)
    L = _build_likelihood_matrix(beta_hat_arr, s2_arr, params.mu_k, params.var_k)
    if not np.isfinite(L).all():
        # Non-finite likelihoods indicate a broken component grid or extreme
        # inputs that slipped past the boundary checks; escalate as a numerics
        # failure rather than silently masking.
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "likelihood matrix contains non-finite values"},
        )
    L = _floor_zero_rows(L)

    return Solution(
        value=FitState(likelihood_matrix=L, initial_params=params),
        result=RESULTS.successful,
        stats={
            "num_observations": int(beta_hat_arr.shape[0]),
            "num_components": int(params.pi.shape[0]),
        },
    )


def fit_baseline(
    state: FitState,
    config: InferenceConfig,
    verbose: bool | Callable[..., None] = False,
    prior: np.ndarray | None = None,
) -> Solution:
    r"""Fit baseline mixture weights via mix-SQP on a prepared likelihood matrix.

    **Arguments:**

    - `state`: Prepared fit state containing a likelihood matrix and initial component grid.
    - `config`: Inference numerics configuration.
    - `verbose`: Optional mix-SQP progress callback.
    - `prior`: Optional per-component Dirichlet prior vector; defaults to ones (no penalty).

    **Returns:**

    - `Solution` whose `value` is fitted `Params` on recoverable or successful runs.

    **Failure Modes:**

    - Returns `RESULTS.nonfinite_objective` when mix-SQP fails.
    """
    L = state.likelihood_matrix
    prior_arr = np.ones(L.shape[1], dtype=float) if prior is None else np.asarray(prior, dtype=float)
    L_aug, w_aug = _augment_with_prior(L, prior_arr)
    try:
        pi, info = mix_sqp(
            L_aug,
            w=w_aug,
            max_iter=config.max_iter,
            atol=config.atol,
            rtol=config.rtol,
            verbose=verbose,
        )
    except Exception as exc:
        # mix-SQP can raise on genuinely non-finite iterates (e.g. divergent
        # line search); surface as a recoverable-looking numerics failure.
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"mix-SQP failed: {exc}"},
        )

    result = RESULTS.successful if info["converged"] else RESULTS.max_steps_reached
    fitted = Params(pi=pi, mu_k=state.initial_params.mu_k, var_k=state.initial_params.var_k)
    return Solution(
        value=fitted,
        result=result,
        stats={
            "epoch_count": info["n_iter"],
            "objective": info["objective"],
            "converged": info["converged"],
            "num_observations": int(L.shape[0]),
            "used_full_batch_objective": True,
        },
    )


def fit_refit_step(
    L_sub: Any,
    prev_params: Params,
    config: InferenceConfig,
    verbose: bool | Callable[..., None] = False,
    prior: np.ndarray | None = None,
) -> Solution:
    r"""Fit one ordered refit step on a likelihood submatrix.

    **Arguments:**

    - `L_sub`: Likelihood submatrix for one threshold subset.
    - `prev_params`: Previous threshold parameters used as ordered baseline.
    - `config`: Inference numerics configuration.
    - `verbose`: Optional mix-SQP-ordered progress callback.
    - `prior`: Optional per-component Dirichlet prior vector; defaults to ones (no penalty).

    **Returns:**

    - `Solution` whose `value` is the updated `Params` on recoverable or successful runs.

    **Failure Modes:**

    - Returns `RESULTS.empty_subset` when no observations are selected.
    - Returns `RESULTS.nonfinite_objective` when ordered mix-SQP fails.
    """
    L_sub_arr = np.asarray(L_sub, dtype=float)
    if L_sub_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "refit subset is empty"},
        )

    A = build_constraints_matrix(prev_params.pi, constrain_spike=config.constrain_spike)

    prior_arr = np.ones(L_sub_arr.shape[1], dtype=float) if prior is None else np.asarray(prior, dtype=float)
    L_aug, w_aug = _augment_with_prior(L_sub_arr, prior_arr)
    try:
        pi, info = mix_sqp_ordered(
            L_aug,
            A=A,
            baseline=prev_params.pi,
            w=w_aug,
            max_iter=config.max_iter,
            atol=config.atol,
            rtol=config.rtol,
            verbose=verbose,
        )
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"mix-SQP-ordered failed: {exc}"},
        )

    result = RESULTS.successful if info["converged"] else RESULTS.max_steps_reached
    return Solution(
        value=Params(pi=pi, mu_k=prev_params.mu_k, var_k=prev_params.var_k),
        result=result,
        stats={
            "epoch_count": info["n_iter"],
            "objective": info["objective"],
            "converged": info["converged"],
            "n_obs_used": int(L_sub_arr.shape[0]),
            "likelihood_shape": (
                int(L_sub_arr.shape[0]),
                int(L_sub_arr.shape[1]),
            ),
        },
    )


__all__ = ["FitState", "Params", "fit_baseline", "fit_refit_step", "prepare_fit_state"]
