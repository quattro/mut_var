from __future__ import annotations

# pattern: Functional Core
from typing import Literal, NamedTuple

import numpy as np
import scipy.optimize as sco

from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression
from scipy.special import expit, logit

from mut_var.types import RESULTS, Solution

_MAF_EPS = 1e-12
_PARAM_EPS = 1e-9
_LOG_RESIDUAL_WEIGHT = 2e-3

_POOR_FIT_RMSE_THRESHOLD = 3e-2
_POOR_FIT_MAX_ABS_THRESHOLD = 1.5e-1
_POOR_FIT_NONMONOTONE_SIGN_CHANGES = 2
_POOR_FIT_NONMONOTONE_MAX_ABS_THRESHOLD = 2e-2

CurveMethod = Literal["sigmoid", "isotonic", "mono_spline"]


class CurveFitResult(NamedTuple):
    r"""Method-neutral fitted curve representation.

    For ``method="sigmoid"``, ``payload`` holds the decoded 4-vector
    ``(left, right, rate, midpoint)`` and ``support``/``increasing`` are unused.
    For ``method="isotonic"``, ``payload`` holds the fitted step-function levels
    on ``support`` (the unique MAF grid), and ``increasing`` records the
    monotonic direction. For ``method="mono_spline"``, ``payload`` holds the
    monotone knot levels on ``support`` (the unique MAF grid), and ``increasing``
    records the direction; evaluation interpolates these with a PCHIP spline in
    log-MAF space.

    **Arguments:**

    - `method`: Curve-fitting method name.
    - `payload`: Method-specific fitted state.
    - `support`: Unique MAF support for isotonic/mono_spline fits; ``None`` for sigmoid.
    - `increasing`: Monotonic direction flag for isotonic/mono_spline; ``None`` for sigmoid.

    """

    method: CurveMethod
    payload: np.ndarray
    support: np.ndarray | None = None
    increasing: bool | None = None


# Sigmoid fitting works in a latent unconstrained space, then decodes back to
# bounded asymptotes and a positive midpoint on the observed MAF range.
def _midpoint_bounds(maf: np.ndarray) -> tuple[float, float]:
    positive = maf[maf > 0.0]
    if positive.size == 0:
        return np.log(_MAF_EPS), 1.0
    log_min = np.log(np.min(positive) + _MAF_EPS)
    log_max = np.log(np.max(positive) + _MAF_EPS)
    log_span = max(log_max - log_min, 1e-6)
    return log_min, log_span


def _decode_latent(
    latent_coef: np.ndarray,
    log_mid_min: float,
    log_mid_span: float,
) -> np.ndarray:
    raw_left, raw_span, raw_rate, raw_mid = latent_coef
    left_asym = expit(raw_left)
    span = expit(raw_span)
    right_asym = left_asym + (1.0 - left_asym) * span
    midpoint = np.exp(log_mid_min + expit(raw_mid) * log_mid_span)
    return np.array([left_asym, right_asym, raw_rate, midpoint], dtype=float)


def _init_latent_parameters(
    maf: np.ndarray,
    value: np.ndarray,
    log_mid_min: float,
    log_mid_span: float,
) -> np.ndarray:
    left_init = np.clip(np.min(value), _PARAM_EPS, 1.0 - _PARAM_EPS)
    right_floor = min(left_init + _PARAM_EPS, 1.0 - _PARAM_EPS)
    right_init = np.clip(np.max(value), right_floor, 1.0 - _PARAM_EPS)
    span_init = np.clip((right_init - left_init) / (1.0 - left_init), _PARAM_EPS, 1.0 - _PARAM_EPS)

    log_maf = np.log(np.clip(maf, _MAF_EPS, None))
    x_centered = log_maf - np.mean(log_maf)
    y_centered = value - np.mean(value)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom > 0.0:
        corr = np.sum(x_centered * y_centered) / denom
        slope_sign = -1.0 if corr >= 0.0 else 1.0
    else:
        slope_sign = -1.0 if value[-1] >= value[0] else 1.0

    target = left_init + 0.5 * (right_init - left_init)
    closest_index = int(np.argmin(np.abs(value - target)))

    midpoint_low = np.exp(log_mid_min)
    midpoint_high = np.exp(log_mid_min + log_mid_span)
    midpoint_init = maf[closest_index]
    if not np.isfinite(midpoint_init) or midpoint_init <= 0.0:
        midpoint_init = np.sqrt(midpoint_low * midpoint_high)
    midpoint_init = np.clip(midpoint_init, midpoint_low, midpoint_high)

    midpoint_fraction = np.clip(
        (np.log(midpoint_init + _MAF_EPS) - log_mid_min) / log_mid_span,
        _PARAM_EPS,
        1.0 - _PARAM_EPS,
    )

    return np.array(
        [logit(left_init), logit(span_init), 2.0 * slope_sign, logit(midpoint_fraction)],
        dtype=float,
    )


def _count_sign_changes(value: np.ndarray) -> int:
    diffs = np.diff(value)
    signs = np.sign(diffs)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def _fit_diagnostics(value: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | bool]:
    # These diagnostics feed warning-level workflow logs, so keep them simple
    # and interpretable rather than solver-specific.
    abs_error = np.abs(prediction - value)
    rmse = np.sqrt(np.mean((prediction - value) ** 2))
    max_abs_error = np.max(abs_error)
    data_sign_changes = _count_sign_changes(value)
    poor_fit = bool(
        (rmse > _POOR_FIT_RMSE_THRESHOLD)
        or (max_abs_error > _POOR_FIT_MAX_ABS_THRESHOLD)
        or (
            data_sign_changes >= _POOR_FIT_NONMONOTONE_SIGN_CHANGES
            and max_abs_error > _POOR_FIT_NONMONOTONE_MAX_ABS_THRESHOLD
        )
    )
    return {
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "data_sign_changes": data_sign_changes,
        "poor_fit": poor_fit,
    }


def _evaluate_sigmoid_curve(maf: np.ndarray, coef: np.ndarray) -> np.ndarray:
    left_asym, right_asym, rate, midpoint = coef
    ratio = (np.clip(maf, 0.0, None) + _MAF_EPS) / (midpoint + _MAF_EPS)
    return left_asym + (right_asym - left_asym) / (1.0 + np.power(ratio, rate))


def _evaluate_isotonic_curve(fit: CurveFitResult, maf: np.ndarray) -> np.ndarray:
    # The isotonic fit is a step function on the unique support grid; for each
    # query MAF, find the largest support value <= query and return its level.
    support = fit.support
    assert support is not None  # enforced by _fit_isotonic_curve_model
    indices = np.searchsorted(support, maf, side="right") - 1
    indices = np.clip(indices, 0, support.size - 1)
    return fit.payload[indices]


def _evaluate_mono_spline_curve(fit: CurveFitResult, maf: np.ndarray) -> np.ndarray:
    # The fit is a monotone PCHIP through the isotonic-regressed levels in
    # log-MAF space. Queries outside the fitted support are clamped to the
    # nearest edge level rather than extrapolated (PCHIP extrapolation is
    # cubic and not guaranteed to stay monotone).
    support = fit.support
    assert support is not None  # enforced by _fit_mono_spline_curve_model
    levels = fit.payload
    if support.size < 2:
        return np.full(maf.shape, float(levels[0]))
    log_support = np.log(np.clip(support, _MAF_EPS, None))
    log_maf = np.log(np.clip(maf, _MAF_EPS, None))
    pchip = PchipInterpolator(log_support, levels, extrapolate=False)
    evaluated = pchip(log_maf)
    return np.where(
        log_maf <= log_support[0],
        levels[0],
        np.where(log_maf >= log_support[-1], levels[-1], evaluated),
    )


def evaluate_curve_fit(fit: CurveFitResult, maf: np.ndarray) -> np.ndarray:
    r"""Evaluate a fitted curve model on MAF inputs.

    **Arguments:**

    - `fit`: Method-neutral fitted curve result.
    - `maf`: MAF values at which to evaluate the fit.

    **Returns:**

    - Fitted values on `maf`.
    """
    if fit.method == "sigmoid":
        return _evaluate_sigmoid_curve(maf, fit.payload)
    if fit.method == "mono_spline":
        return _evaluate_mono_spline_curve(fit, maf)
    return _evaluate_isotonic_curve(fit, maf)


def _fit_sigmoid_curve_model(maf: np.ndarray, value: np.ndarray) -> Solution:
    n_obs = int(maf.size)

    log_mid_min, log_mid_span = _midpoint_bounds(maf)
    init = _init_latent_parameters(maf, value, log_mid_min, log_mid_span)

    def residuals(latent: np.ndarray) -> np.ndarray:
        prediction = _evaluate_sigmoid_curve(maf, _decode_latent(latent, log_mid_min, log_mid_span))
        raw_res = prediction - value
        # A small log-scale residual term helps near-zero regions without
        # turning the objective into a fully relative-error loss.
        log_res = np.log(prediction + _MAF_EPS) - np.log(value + _MAF_EPS)
        return np.concatenate([raw_res, _LOG_RESIDUAL_WEIGHT * log_res])

    try:
        result = sco.least_squares(
            residuals,
            init,
            method="lm",
            max_nfev=1000,
        )
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"curve fit failed: {exc}"},
        )

    coef = _decode_latent(result.x, log_mid_min, log_mid_span)
    prediction = _evaluate_sigmoid_curve(maf, coef)
    if not np.isfinite(prediction).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "curve fit produced non-finite predictions"},
        )

    converged = result.status > 0
    mapped_result = RESULTS.successful if converged else RESULTS.max_steps_reached
    diagnostics = _fit_diagnostics(value, prediction)
    return Solution(
        value=CurveFitResult(method="sigmoid", payload=coef),
        result=mapped_result,
        stats={
            "n_obs": n_obs,
            "epoch_count": int(result.nfev),
            "converged": converged,
            **diagnostics,
        },
    )


def _fit_isotonic_curve_model(maf: np.ndarray, value: np.ndarray) -> Solution:
    n_obs = int(maf.size)
    order = np.argsort(maf, kind="mergesort")
    maf_sorted = maf[order]
    value_sorted = value[order]

    # Collapse duplicate MAF support points before isotonic regression so the
    # fitted step function is represented on a stable unique grid.
    unique_maf, inverse = np.unique(maf_sorted, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    summed_values = np.bincount(inverse, weights=value_sorted)
    averaged_values = summed_values / counts

    if unique_maf.size == 1:
        increasing = True
    else:
        log_maf = np.log(np.clip(unique_maf, _MAF_EPS, None))
        x_centered = log_maf - np.mean(log_maf)
        y_centered = averaged_values - np.mean(averaged_values)
        denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denom > 0.0:
            corr = np.sum(x_centered * y_centered) / denom
            increasing = bool(corr >= 0.0)
        else:
            increasing = bool(averaged_values[-1] >= averaged_values[0])

    # SciPy returns fitted levels on the unique support grid; `inverse`
    # expands them back to the observation-aligned prediction vector.
    fitted_unique = isotonic_regression(averaged_values, weights=counts, increasing=increasing).x
    prediction = fitted_unique[inverse]
    if not np.isfinite(prediction).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "isotonic fitted values are non-finite"},
        )

    diagnostics = _fit_diagnostics(value, prediction)
    return Solution(
        value=CurveFitResult(
            method="isotonic",
            payload=fitted_unique,
            support=unique_maf,
            increasing=increasing,
        ),
        result=RESULTS.successful,
        stats={
            "n_obs": n_obs,
            "n_unique": int(unique_maf.size),
            "epoch_count": int(unique_maf.size),
            "converged": True,
            **diagnostics,
        },
    )


def _fit_mono_spline_curve_model(maf: np.ndarray, value: np.ndarray) -> Solution:
    n_obs = int(maf.size)
    order = np.argsort(maf, kind="mergesort")
    maf_sorted = maf[order]
    value_sorted = value[order]

    # Aggregate duplicate MAFs the same way isotonic does so the spline knots
    # sit on a strictly increasing support with well-defined target values.
    unique_maf, inverse = np.unique(maf_sorted, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    summed_values = np.bincount(inverse, weights=value_sorted)
    averaged_values = summed_values / counts

    log_unique = np.log(np.clip(unique_maf, _MAF_EPS, None))
    if unique_maf.size == 1:
        increasing = True
    else:
        # Direction is picked from the log-MAF vs value correlation since the
        # spline itself is fit in log-MAF space.
        x_centered = log_unique - np.mean(log_unique)
        y_centered = averaged_values - np.mean(averaged_values)
        denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denom > 0.0:
            corr = np.sum(x_centered * y_centered) / denom
            increasing = bool(corr >= 0.0)
        else:
            increasing = bool(averaged_values[-1] >= averaged_values[0])

    # Enforce monotonicity on the knot levels via weighted isotonic regression.
    # PCHIP then gives a smooth monotone cubic interpolant through those levels.
    fitted_unique = isotonic_regression(averaged_values, weights=counts, increasing=increasing).x

    if unique_maf.size < 2:
        prediction_sorted = fitted_unique[inverse]
    else:
        log_maf_sorted = np.log(np.clip(maf_sorted, _MAF_EPS, None))
        pchip = PchipInterpolator(log_unique, fitted_unique, extrapolate=False)
        evaluated = pchip(log_maf_sorted)
        prediction_sorted = np.where(
            log_maf_sorted <= log_unique[0],
            fitted_unique[0],
            np.where(log_maf_sorted >= log_unique[-1], fitted_unique[-1], evaluated),
        )

    # Return prediction to the caller's original sample order before scoring.
    prediction = np.empty_like(prediction_sorted)
    prediction[order] = prediction_sorted
    if not np.isfinite(prediction).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "mono_spline fitted values are non-finite"},
        )

    diagnostics = _fit_diagnostics(value, prediction)
    return Solution(
        value=CurveFitResult(
            method="mono_spline",
            payload=fitted_unique,
            support=unique_maf,
            increasing=increasing,
        ),
        result=RESULTS.successful,
        stats={
            "n_obs": n_obs,
            "n_unique": int(unique_maf.size),
            "epoch_count": int(unique_maf.size),
            "converged": True,
            **diagnostics,
        },
    )


def fit_curve_model(maf: np.ndarray, value: np.ndarray, *, method: CurveMethod = "sigmoid") -> Solution:
    r"""Fit a method-neutral curve model.

    **Arguments:**

    - `maf`: 1D MAF values.
    - `value`: 1D target values aligned with `maf`.
    - `method`: Curve-fitting method (`sigmoid`, `isotonic`, or `mono_spline`).
      The `mono_spline` method fits a monotone cubic (PCHIP) spline through
      isotonic-regressed knot levels in log-MAF space; the caller passes raw
      MAF values and the log-transform is applied internally.

    **Returns:**

    - `Solution` with a method-neutral `CurveFitResult` in `value`.

    **Failure Modes:**

    - `RESULTS.nonfinite_objective` for solver or fit failures.
    - `RESULTS.max_steps_reached` when the sigmoid solver does not converge.
    """
    if method == "sigmoid":
        return _fit_sigmoid_curve_model(maf, value)
    if method == "mono_spline":
        return _fit_mono_spline_curve_model(maf, value)
    return _fit_isotonic_curve_model(maf, value)


__all__ = [
    "CurveFitResult",
    "CurveMethod",
    "evaluate_curve_fit",
    "fit_curve_model",
]
