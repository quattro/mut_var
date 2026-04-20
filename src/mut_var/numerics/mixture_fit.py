# pattern: Functional Core
from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from scipy.stats import norm as scipy_norm

from mut_var.numerics.mixsqp import (
    build_ordering_matrix,
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
    r"""Build ``(n, K)`` likelihood matrix for fixed mixture components."""
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


def _floor_zero_rows(L: np.ndarray) -> np.ndarray:
    row_sums = L.sum(axis=1)
    zero_rows = row_sums == 0.0
    if zero_rows.any():
        L = L.copy()
        L[zero_rows, :] = np.finfo(float).tiny
    return L


def _validate_array_inputs(
    beta_hat: Any,
    s2: Any,
    *,
    require_min_clusters: int | None = None,
) -> Solution:
    if hasattr(beta_hat, "columns") or hasattr(s2, "columns"):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "mixture kernel expects arrays, not tabular objects"},
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

    if require_min_clusters is not None and require_min_clusters < 2:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "num_clusters must be >= 2"},
        )

    return Solution(value=(beta_hat_arr, s2_arr), result=RESULTS.successful)


def _build_baseline_params(beta_hat: np.ndarray, s2: np.ndarray, config: InferenceConfig) -> Params:
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

    var_k = np.exp(np.linspace(np.log(min_val), np.log(max_val), config.num_clusters - 1)) ** 2
    mu_k = np.zeros(config.num_clusters - 1)
    pi = np.ones(config.num_clusters, dtype=float) / float(config.num_clusters)
    return Params(pi=pi, mu_k=mu_k, var_k=var_k)


def _validate_likelihood_matrix(L: Any, *, name: str = "likelihood matrix") -> Solution:
    try:
        L_arr = np.asarray(L, dtype=float)
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"failed to convert {name} to array: {exc}"},
        )

    if L_arr.ndim != 2 or L_arr.shape[0] == 0 or L_arr.shape[1] == 0:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"{name} must be a non-empty 2D array"},
        )

    if not np.isfinite(L_arr).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"{name} contains non-finite values"},
        )

    return Solution(value=_floor_zero_rows(L_arr), result=RESULTS.successful)


def _validate_params(params: Params, *, name: str = "params") -> Solution | None:
    if not isinstance(params, Params):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"{name} must be Params"},
        )

    n_components = int(params.pi.shape[0])
    if params.mu_k.shape[0] != n_components - 1 or params.var_k.shape[0] != n_components - 1:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"{name} arrays are not component-aligned"},
        )

    if n_components == 0 or not np.isfinite(params.pi).all() or not np.isfinite(params.var_k).all():
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"{name} contains invalid values"},
        )

    return None


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

    - Returns `RESULTS.invalid_input` for non-array, non-finite, or shape-invalid inputs.
    - Returns `RESULTS.empty_subset` when no observations are available.
    - Returns `RESULTS.nonfinite_objective` when the likelihood matrix is non-finite.
    """
    validated = _validate_array_inputs(beta_hat, s2, require_min_clusters=config.num_clusters)
    if not validated.ok:
        return validated
    beta_hat_arr, s2_arr = validated.value

    params = _build_baseline_params(beta_hat_arr, s2_arr, config)
    L = _build_likelihood_matrix(beta_hat_arr, s2_arr, params.mu_k, params.var_k)
    validated_likelihood = _validate_likelihood_matrix(L)
    if not validated_likelihood.ok:
        return validated_likelihood

    return Solution(
        value=FitState(likelihood_matrix=validated_likelihood.value, initial_params=params),
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
) -> Solution:
    r"""Fit baseline mixture weights via mix-SQP on a prepared likelihood matrix.

    **Arguments:**

    - `state`: Prepared fit state containing a likelihood matrix and initial component grid.
    - `config`: Inference numerics configuration.
    - `verbose`: Optional mix-SQP progress callback.

    **Returns:**

    - `Solution` whose `value` is fitted `Params` on recoverable or successful runs.

    **Failure Modes:**

    - Returns `RESULTS.invalid_input` when the fit state is malformed.
    - Returns `RESULTS.nonfinite_objective` when mix-SQP fails.
    """
    if not isinstance(state, FitState):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "fit_baseline expects a FitState"},
        )

    invalid_params = _validate_params(state.initial_params, name="initial_params")
    if invalid_params is not None:
        return invalid_params

    validated_L = _validate_likelihood_matrix(state.likelihood_matrix)
    if not validated_L.ok:
        return validated_L
    L = validated_L.value

    if L.shape[1] != state.initial_params.pi.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "likelihood matrix columns must align with initial_params"},
        )

    try:
        pi, info = mix_sqp(
            L,
            max_iter=config.max_iter,
            atol=config.atol,
            rtol=config.rtol,
            verbose=verbose,
        )
    except Exception as exc:
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
) -> Solution:
    r"""Fit one ordered refit step on a likelihood submatrix.

    **Arguments:**

    - `L_sub`: Likelihood submatrix for one threshold subset.
    - `prev_params`: Previous threshold parameters used as ordered baseline.
    - `config`: Inference numerics configuration.
    - `verbose`: Optional mix-SQP-ordered progress callback.

    **Returns:**

    - `Solution` whose `value` is the updated `Params` on recoverable or successful runs.

    **Failure Modes:**

    - Returns `RESULTS.invalid_input` when `L_sub` or `prev_params` is malformed.
    - Returns `RESULTS.empty_subset` when no observations are selected.
    - Returns `RESULTS.nonfinite_objective` when ordered mix-SQP fails.
    """
    invalid_params = _validate_params(prev_params, name="prev_params")
    if invalid_params is not None:
        return invalid_params

    validated_likelihood = _validate_likelihood_matrix(L_sub, name="L_sub")
    if not validated_likelihood.ok:
        return validated_likelihood
    L_sub_arr = validated_likelihood.value

    if L_sub_arr.shape[1] != prev_params.pi.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "L_sub columns must align with prev_params"},
        )

    if L_sub_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "refit subset is empty"},
        )

    A = build_ordering_matrix(prev_params.pi)

    try:
        pi, info = mix_sqp_ordered(
            L_sub_arr,
            A=A,
            baseline=prev_params.pi,
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
