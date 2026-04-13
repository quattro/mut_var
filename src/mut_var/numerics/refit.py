from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from scipy.stats import norm as scipy_norm

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics._solver_utils import is_recoverable_result
from mut_var.numerics.baseline import Params
from mut_var.numerics.solver import build_ordering_matrix, mix_sqp_ordered


class RefitConfig(NamedTuple):
    penalty: float = 1.0  # Kept for API compatibility; ordering constraints are now hard.
    max_iter: int = 100
    tol: float = 1e-5
    step_size: float = 0.01  # Kept for API compatibility; unused in mix-SQP.


def _build_likelihood_matrix(
    beta_hat: np.ndarray,
    s2: np.ndarray,
    mu_k: np.ndarray,
    var_k: np.ndarray,
) -> np.ndarray:
    """Build ``(n, K)`` likelihood matrix using fixed component parameters.

    Identical layout to the baseline: column 0 is the null component, columns
    1..K-1 use ``mu_k`` and ``var_k`` from the fitted baseline model.
    """
    n = len(beta_hat)
    K = len(var_k) + 1
    L = np.empty((n, K), dtype=float)
    L[:, 0] = scipy_norm.pdf(beta_hat, loc=0.0, scale=np.sqrt(s2))
    for k in range(1, K):
        L[:, k] = scipy_norm.pdf(beta_hat, loc=float(mu_k[k - 1]), scale=np.sqrt(s2 + var_k[k - 1]))
    return L


def fit_refit_grid(
    beta_hat: Any,
    s2: Any,
    maf_masks: Any,
    init: Params,
    config: RefitConfig,
    verbose: bool | Callable[..., None] = False,
) -> Solution:
    r"""Sequentially refit mixture weights across MAF-threshold masks via mix-SQP-ordered.

    Component means and variances are fixed from ``init`` (baseline output).
    For each MAF threshold slice, mix-SQP-ordered optimises the mixture weights
    subject to the ordering constraint that ``pi / pi_prev`` is nonincreasing,
    warm-started from the previous threshold's solution.

    **Arguments:**

    - `beta_hat`: 1D effect-size estimates.
    - `s2`: 1D positive variances aligned with `beta_hat`.
    - `maf_masks`: 2D boolean masks selecting observations per threshold.
    - `init`: Initial parameters from the baseline fit.
    - `config`: Refit solver controls.
    - `verbose`: ``False`` for silent; ``True`` prints each SQP step;
      a callable receives ``(step, obj)`` keyword arguments.

    **Returns:**

    - `Solution` with one fitted model per threshold plus the initial model.

    **Failure Modes:**

    - `RESULTS.invalid_input` for shape/mask mismatches or non-finite likelihoods.
    - `RESULTS.empty_subset` for empty threshold slices.
    - `RESULTS.nonfinite_objective` for non-finite objective values.
    - `RESULTS.max_steps_reached` when any threshold solve does not converge.
    """
    beta_hat_arr = np.asarray(beta_hat, dtype=float)
    s2_arr = np.asarray(s2, dtype=float)
    masks_arr = np.asarray(maf_masks, dtype=bool)

    if beta_hat_arr.ndim != 1 or s2_arr.ndim != 1 or beta_hat_arr.shape[0] != s2_arr.shape[0]:
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "beta_hat and s2 must be 1D arrays of equal length"},
        )

    if masks_arr.ndim != 2 or masks_arr.shape[1] != beta_hat_arr.shape[0]:
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "maf_masks must be a 2D mask array over observations"},
        )

    # Restrict to observations that appear in at least one threshold mask.
    active_obs = np.any(masks_arr, axis=0)
    beta_hat_active = beta_hat_arr[active_obs]
    s2_active = s2_arr[active_obs]
    masks_active = masks_arr[:, active_obs]

    # Pre-compute likelihoods once; component parameters are fixed across all thresholds.
    L_full = _build_likelihood_matrix(beta_hat_active, s2_active, init.mu_k, init.var_k)

    if not np.isfinite(L_full).all():
        return Solution(
            value=[init],
            result=RESULTS.invalid_input,
            stats={"reason": "likelihood matrix contains non-finite values"},
        )

    # Replace zero rows with tiny floor.
    row_sums = L_full.sum(axis=1)
    zero_rows = row_sums == 0.0
    if zero_rows.any():
        L_full[zero_rows, :] = np.finfo(float).tiny

    models: list[Params] = [init]
    any_max_steps = False
    threshold_diagnostics: list[dict[str, Any]] = []

    for idx in range(masks_arr.shape[0]):
        mask = masks_active[idx]
        n_obs = int(mask.sum())

        if n_obs == 0:
            return Solution(
                value=models,
                result=RESULTS.empty_subset,
                stats={"reason": f"maf mask at index {idx} is empty"},
            )

        L_sub = L_full[mask]

        # Replace zero rows in this subset.
        sub_row_sums = L_sub.sum(axis=1)
        sub_zero = sub_row_sums == 0.0
        if sub_zero.any():
            L_sub = L_sub.copy()
            L_sub[sub_zero, :] = np.finfo(float).tiny

        prev_params = models[-1]
        A = build_ordering_matrix(prev_params.pi)

        try:
            pi, info = mix_sqp_ordered(
                L_sub,
                A=A,
                baseline=prev_params.pi,
                max_iter=config.max_iter,
                tol=config.tol,
                verbose=verbose,
            )
        except Exception as exc:
            return Solution(
                value=models,
                result=RESULTS.nonfinite_objective,
                stats={"reason": f"mix-SQP-ordered failed at threshold {idx}: {exc}"},
            )

        result = RESULTS.successful if info["converged"] else RESULTS.max_steps_reached
        if not is_recoverable_result(result):
            return Solution(
                value=models,
                result=result,
                stats={"reason": f"non-recoverable result at threshold {idx}"},
            )
        if result == RESULTS.max_steps_reached:
            any_max_steps = True

        threshold_diagnostics.append(
            {
                "threshold_index": idx,
                "epoch_count": info["n_iter"],
                "objective": info["objective"],
                "converged": info["converged"],
                "n_obs_used": n_obs,
                "likelihood_shape": (int(L_sub.shape[0]), int(L_sub.shape[1])),
            }
        )
        new_params = Params(pi=pi, mu_k=prev_params.mu_k, var_k=prev_params.var_k)
        models.append(new_params)

    final_result = RESULTS.max_steps_reached if any_max_steps else RESULTS.successful
    return Solution(
        value=models,
        result=final_result,
        stats={
            "num_models": len(models),
            "num_thresholds": int(masks_arr.shape[0]),
            "threshold_diagnostics": threshold_diagnostics,
        },
    )
