from __future__ import annotations

# pattern: Functional Core
from typing import get_args, Literal, NamedTuple

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

CurveMethod = Literal["sigmoid", "isotonic", "mono_spline", "invlog_linear", "invlog_logit", "invlog_sigmoid"]


class CurveFitResult(NamedTuple):
    r"""Method-neutral fitted curve representation.

    For ``method="sigmoid"``, ``payload`` holds the decoded 4-vector
    ``(left, right, rate, midpoint)`` and ``support``/``increasing`` are unused.
    For ``method="isotonic"``, ``payload`` holds the fitted step-function levels
    on ``support`` (the unique MAF grid), and ``increasing`` records the
    monotonic direction. For ``method="mono_spline"``, ``payload`` holds the
    monotone knot levels on ``support`` (the unique MAF grid), and ``increasing``
    records the direction; evaluation interpolates these with a PCHIP spline in
    log-MAF space. For the inverse-log methods, ``payload`` is ``(a, b)``
    in $a + b / \log(1/t)$, on the probability or logit scale respectively;
    ``support`` and ``increasing`` are unused. For ``invlog_sigmoid``, payload is
    ``(lower, upper, a, b)`` in $L+(U-L)\operatorname{expit}(a+b/\log(1/t))$.

    **Arguments:**

    - `method`: Curve-fitting method name.
    - `payload`: Method-specific fitted state.
    - `support`: Unique MAF support for isotonic/mono_spline fits; otherwise ``None``.
    - `increasing`: Monotonic direction for isotonic/mono_spline; otherwise ``None``.

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
    # Round-tripping through exp() at fit time and log() here can collapse
    # adjacent knots that were distinct in log space at fit time; PCHIP
    # requires strictly increasing x, so drop any collapsed pairs and their
    # (redundant) levels before building the interpolator.
    log_support = np.log(np.clip(support, _MAF_EPS, None))
    keep = np.concatenate(([True], np.diff(log_support) > 0.0))
    log_support = log_support[keep]
    levels = levels[keep]
    if log_support.size < 2:
        return np.full(maf.shape, float(levels[0]))
    log_maf = np.log(np.clip(maf, _MAF_EPS, None))
    pchip = PchipInterpolator(log_support, levels, extrapolate=False)
    evaluated = pchip(log_maf)
    return np.where(
        log_maf <= log_support[0],
        levels[0],
        np.where(log_maf >= log_support[-1], levels[-1], evaluated),
    )


def _inverse_log_maf(maf: np.ndarray) -> np.ndarray:
    """Return inverse log frequency, with its exact limit at zero."""
    maf = np.asarray(maf, dtype=float)
    if not np.isfinite(maf).all() or np.any((maf < 0) | (maf >= 1)):
        raise ValueError("inverse-log MAF values must be finite and within [0, 1)")
    transformed = np.zeros_like(maf)
    positive = maf > 0
    # -log(t) equals log(1/t), without overflowing 1/t for subnormal MAF.
    transformed[positive] = 1.0 / -np.log(maf[positive])
    return transformed


def evaluate_curve_fit(fit: CurveFitResult, maf: np.ndarray) -> np.ndarray:
    r"""Evaluate a fitted curve model on MAF inputs.

    **Arguments:**

    - `fit`: Method-neutral fitted curve result.
    - `maf`: MAF values at which to evaluate the fit.

    **Returns:**

    - Fitted values on `maf`.

    Two-parameter inverse-log methods evaluate zero exactly as $a$ or $\operatorname{expit}(a)$.
    For `invlog_sigmoid`, zero is exactly $L+(U-L)\operatorname{expit}(a)$.
    Linear predictions are not clipped.

    **Raises:**

    - `ValueError`: Unknown method, or inverse-log MAF inputs are nonfinite
      or outside $[0,1)$.
    """
    if fit.method not in get_args(CurveMethod):
        raise ValueError(f"unknown curve method: {fit.method!r}")
    if fit.method == "invlog_sigmoid":
        lower, upper, a, b = fit.payload
        return lower + (upper - lower) * expit(a + b * _inverse_log_maf(maf))
    if fit.method in ("invlog_linear", "invlog_logit"):
        a, b = fit.payload
        prediction = a + b * _inverse_log_maf(maf)
        return expit(prediction) if fit.method == "invlog_logit" else prediction
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

    # SciPy returns fitted levels on the unique support grid; `inverse` expands
    # them along the sorted observation axis, and we then map back to the
    # caller's original sample order so diagnostics pair like-with-like.
    fitted_unique = isotonic_regression(averaged_values, weights=counts, increasing=increasing).x
    prediction_sorted = fitted_unique[inverse]
    prediction = np.empty_like(prediction_sorted)
    prediction[order] = prediction_sorted
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

    # Aggregate in log-MAF space (not raw MAF) so the spline knot support is
    # strictly increasing in log space — PchipInterpolator requires that, and
    # bit-different MAFs that land sub-ULP apart in log space would otherwise
    # pass as distinct MAFs but collapse to identical log-MAFs and crash PCHIP.
    log_maf_sorted = np.log(np.clip(maf_sorted, _MAF_EPS, None))
    unique_log_maf, inverse = np.unique(log_maf_sorted, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    summed_values = np.bincount(inverse, weights=value_sorted)
    averaged_values = summed_values / counts
    unique_maf = np.exp(unique_log_maf)

    if unique_log_maf.size == 1:
        increasing = True
    else:
        # Direction is picked from the log-MAF vs value correlation since the
        # spline itself is fit in log-MAF space.
        x_centered = unique_log_maf - np.mean(unique_log_maf)
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

    if unique_log_maf.size < 2:
        prediction_sorted = fitted_unique[inverse]
    else:
        pchip = PchipInterpolator(unique_log_maf, fitted_unique, extrapolate=False)
        evaluated = pchip(log_maf_sorted)
        prediction_sorted = np.where(
            log_maf_sorted <= unique_log_maf[0],
            fitted_unique[0],
            np.where(log_maf_sorted >= unique_log_maf[-1], fitted_unique[-1], evaluated),
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


def _fit_invlog_sigmoid(z: np.ndarray, value: np.ndarray) -> Solution:
    """Fit bounded levels and a logistic transition in inverse-log frequency."""
    n_obs = int(value.size)
    if np.ptp(value) == 0:
        # Constant data cannot identify a transition; use a canonical flat fit.
        return Solution(
            CurveFitResult("invlog_sigmoid", np.array([value[0], value[0], 0.0, 0.0])),
            RESULTS.successful,
            stats={"n_obs": n_obs, "epoch_count": 0, "converged": True, **_fit_diagnostics(value, value)},
        )
    # Center inverse-log frequency and scale its largest absolute deviation to
    # one. The observed x range then spans between one and two units, giving
    # the initial slopes below a comparable meaning across different MAF grids.
    center = float(np.mean(z))
    scale = float(np.max(np.abs(z - center)))
    x = (z - center) / scale
    # A common residual scale leaves the probability-scale objective unchanged,
    # but prevents tiny component weights from triggering premature convergence.
    residual_scale = float(np.ptp(value))
    low = float(np.clip(np.min(value) - 0.05 * residual_scale, 1e-12, 1 - 1e-12))
    high = float(np.clip(np.max(value) + 0.05 * residual_scale, low + 1e-12, 1 - 1e-12))
    initial_levels = [logit(low), logit(np.clip((high - low) / (1 - low), 1e-12, 1 - 1e-12))]
    middle = float(x[np.argmin(np.abs(value - 0.5 * (low + high)))])

    def decode(q: np.ndarray) -> tuple[float, float]:
        lower = expit(q[0])
        span = (1 - lower) * expit(q[1])
        return lower, span

    def residuals(q: np.ndarray) -> np.ndarray:
        lower, span = decode(q)
        return (lower + span * expit(q[2] + q[3] * x) - value) / residual_scale

    def jacobian(q: np.ndarray) -> np.ndarray:
        lower, span = decode(q)
        fraction = expit(q[1])
        p = expit(q[2] + q[3] * x)
        transition = span * p * (1 - p)
        return (
            np.column_stack(
                (
                    lower * (1 - lower) * (1 - fraction * p),
                    span * (1 - fraction) * p,
                    transition,
                    transition * x,
                )
            )
            / residual_scale
        )

    try:
        # These deterministic slopes are heuristic starting guesses, not values
        # derived from population genetics or selected by systematic tuning.
        # For expit(slope * (x - middle)), the 10%-90% transition width is
        # 2 * log(9) / abs(slope): about 1.10 x units for abs(slope)=4 and
        # 0.27 for abs(slope)=16. On the scaled predictor these cover broad and
        # sharper transitions; both signs allow increasing or decreasing curves.
        # The initial intercept -slope * middle centers each transition near
        # the observed midpoint. Multiple starts reduce sensitivity to local
        # minima and saturated regions with small gradients, but do not
        # guarantee a global optimum. Every run optimizes all four parameters
        # freely (including slope); we retain the lowest-cost fit below.
        candidates = [
            sco.least_squares(
                residuals,
                np.array([*initial_levels, -slope * middle, slope]),
                jac=jacobian,
                x_scale="jac",
                max_nfev=2000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            for slope in (-16.0, -4.0, 4.0, 16.0)
        ]
        result = min(candidates, key=lambda candidate: candidate.cost)
        lower, span = decode(result.x)
        b = result.x[3] / scale
        a = result.x[2] - b * center
        payload = np.array([lower, lower + span, a, b])
        prediction = lower + span * expit(a + b * z)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return Solution(None, RESULTS.nonfinite_objective, stats={"reason": f"inverse-log sigmoid fit failed: {exc}"})
    if not np.isfinite(payload).all() or not np.isfinite(prediction).all():
        return Solution(None, RESULTS.nonfinite_objective, stats={"reason": "nonfinite inverse-log sigmoid fit"})
    converged = result.status > 0
    return Solution(
        CurveFitResult("invlog_sigmoid", payload),
        RESULTS.successful if converged else RESULTS.max_steps_reached,
        stats={
            "n_obs": n_obs,
            "epoch_count": sum(int(c.nfev) for c in candidates),
            "converged": converged,
            **_fit_diagnostics(value, prediction),
        },
    )


def _fit_invlog_curve_model(maf: np.ndarray, value: np.ndarray, method: CurveMethod) -> Solution:
    z = _inverse_log_maf(maf)
    required = 4 if method == "invlog_sigmoid" else 2
    if np.unique(z).size < required:
        return Solution(None, RESULTS.invalid_input, stats={"reason": f"{method} needs {required} distinct MAF points"})
    if method == "invlog_sigmoid":
        return _fit_invlog_sigmoid(z, value)

    # Each row retains unit weight, including repeated thresholds. Scaling the
    # design improves conditioning without changing the least-squares objective.
    center = float(np.mean(z))
    scale = float(np.max(np.abs(z - center)))
    design = np.column_stack((np.ones_like(z), (z - center) / scale))
    try:
        if method == "invlog_linear":
            coef = np.linalg.lstsq(design, value, rcond=None)[0]
            converged, nfev = True, 1
        else:
            # Only the initial mean is clipped; observations and probability-
            # scale residuals retain exact zeros/ones. No observed logit is used.
            initial = np.array([logit(np.clip(np.mean(value), _PARAM_EPS, 1 - _PARAM_EPS)), 0.0])

            # Common scaling preserves the least-squares minimizer while making
            # the gradient tolerance meaningful for rare component weights.
            mean_value = float(np.mean(value))
            residual_scale = float(np.ptp(value)) or min(mean_value, 1 - mean_value) or 1.0

            def residuals(coef: np.ndarray) -> np.ndarray:
                return (expit(design @ coef) - value) / residual_scale

            def jacobian(coef: np.ndarray) -> np.ndarray:
                prediction = expit(design @ coef)
                return (prediction * (1 - prediction))[:, None] * design / residual_scale

            result = sco.least_squares(
                residuals,
                initial,
                jac=jacobian,
                max_nfev=1000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            coef = result.x
            converged, nfev = result.status > 0, int(result.nfev)
        b = coef[1] / scale
        fit = CurveFitResult(method=method, payload=np.array([coef[0] - b * center, b]))
        prediction = evaluate_curve_fit(fit, maf)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return Solution(None, RESULTS.nonfinite_objective, stats={"reason": f"inverse-log fit failed: {exc}"})
    if not np.isfinite(fit.payload).all() or not np.isfinite(prediction).all():
        return Solution(None, RESULTS.nonfinite_objective, stats={"reason": "nonfinite inverse-log fit"})
    return Solution(
        fit,
        RESULTS.successful if converged else RESULTS.max_steps_reached,
        stats={
            "n_obs": int(maf.size),
            "epoch_count": nfev,
            "converged": converged,
            **_fit_diagnostics(value, prediction),
        },
    )


def fit_curve_model(maf: np.ndarray, value: np.ndarray, *, method: CurveMethod = "sigmoid") -> Solution:
    r"""Fit a method-neutral curve model.

    **Arguments:**

    - `maf`: Aligned finite 1D MAF values within $[0,1)$.
    - `value`: Finite 1D target values aligned with `maf`, within $[0,1]$
      except for unrestricted `invlog_linear` observations.
    - `method`: `sigmoid`, `isotonic`, `mono_spline`, `invlog_linear`, `invlog_logit`,
      or `invlog_sigmoid`.
      The `mono_spline` method fits a monotone cubic (PCHIP) spline through
      isotonic-regressed knot levels in log-MAF space; the caller passes raw
      MAF values and the log-transform is applied internally.
      Two-parameter inverse-log methods fit $a+b/\log(1/t)$ with equal weight per observation,
      including duplicate thresholds, and require two distinct MAFs in $[0,1)$.
      `invlog_linear` uses unconstrained linear least squares; `invlog_logit`
      uses probability-scale nonlinear least squares with stable expit, accepting
      exact zero/one observations without transforming or clipping them.
      Only its initial mean is clipped to $[10^{-9},1-10^{-9}]$ for a finite logit.
      `invlog_sigmoid` fits $L+(U-L)\operatorname{expit}(a+b/\log(1/t))$ with
      $0\le L\le U\le1$, four distinct MAFs, and probability-scale least squares.
      It uses four deterministic starts and a common residual scale for numerical
      conditioning. Constant observations return $L=U$ with $a=b=0$.

    **Returns:**

    - `Solution` with a method-neutral `CurveFitResult` in `value`.

    **Failure Modes:**

    - `RESULTS.nonfinite_objective` for solver or fit failures.
    - `RESULTS.max_steps_reached` when a nonlinear solver does not converge.
    - `RESULTS.invalid_input` for unknown methods, invalid arrays/domains, or fewer
      than two distinct MAF points (four for `invlog_sigmoid`);
      `RESULTS.empty_subset` for empty inputs.
    """
    if method not in get_args(CurveMethod):
        return Solution(None, RESULTS.invalid_input, stats={"reason": f"unknown curve method: {method!r}"})
    try:
        maf = np.asarray(maf, dtype=float)
        value = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        return Solution(None, RESULTS.invalid_input, stats={"reason": f"invalid curve arrays: {exc}"})
    if maf.ndim != 1 or value.ndim != 1 or maf.shape != value.shape:
        return Solution(None, RESULTS.invalid_input, stats={"reason": "expected aligned 1D observations"})
    if maf.size == 0:
        return Solution(None, RESULTS.empty_subset, stats={"reason": "no curve observations"})
    if not np.isfinite(maf).all() or np.any((maf < 0) | (maf >= 1)):
        return Solution(None, RESULTS.invalid_input, stats={"reason": "MAF must be finite and within [0, 1)"})
    if not np.isfinite(value).all():
        return Solution(None, RESULTS.invalid_input, stats={"reason": "curve observations must be finite"})
    if method != "invlog_linear" and np.any((value < 0) | (value > 1)):
        return Solution(
            None, RESULTS.invalid_input, stats={"reason": "bounded curve observations must be within [0, 1]"}
        )
    if method in ("invlog_linear", "invlog_logit", "invlog_sigmoid"):
        return _fit_invlog_curve_model(maf, value, method)
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
