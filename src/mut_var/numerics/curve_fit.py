from __future__ import annotations

# pattern: Functional Core
import numpy as np
import scipy.optimize as sco

from scipy.special import expit  # sigmoid

from mut_var.contracts import RESULTS, Solution

_MAF_EPS = 1e-12
_PARAM_EPS = 1e-9
_LOG_RESIDUAL_WEIGHT = 2e-3

_POOR_FIT_RMSE_THRESHOLD = 3e-2
_POOR_FIT_MAX_ABS_THRESHOLD = 1.5e-1
_POOR_FIT_NONMONOTONE_SIGN_CHANGES = 2
_POOR_FIT_NONMONOTONE_MAX_ABS_THRESHOLD = 2e-2


def _logit(prob: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=float), _PARAM_EPS, 1.0 - _PARAM_EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return expit(np.asarray(x, dtype=float))


def _midpoint_bounds(maf: np.ndarray) -> tuple[float, float]:
    positive = maf[maf > 0.0]
    if positive.size == 0:
        return float(np.log(_MAF_EPS)), 1.0
    log_min = float(np.log(np.min(positive) + _MAF_EPS))
    log_max = float(np.log(np.max(positive) + _MAF_EPS))
    log_span = max(log_max - log_min, 1e-6)
    return log_min, float(log_span)


def _decode_latent(
    latent_coef: np.ndarray,
    log_mid_min: float,
    log_mid_span: float,
) -> np.ndarray:
    raw_left, raw_span, raw_rate, raw_mid = latent_coef
    left_asym = float(_sigmoid(raw_left))
    span = float(_sigmoid(raw_span))
    right_asym = left_asym + (1.0 - left_asym) * span
    midpoint = float(np.exp(log_mid_min + float(_sigmoid(raw_mid)) * log_mid_span))
    return np.array([left_asym, right_asym, raw_rate, midpoint], dtype=float)


def _init_latent_parameters(
    maf: np.ndarray,
    value: np.ndarray,
    log_mid_min: float,
    log_mid_span: float,
) -> np.ndarray:
    left_init = float(np.clip(np.min(value), _PARAM_EPS, 1.0 - _PARAM_EPS))
    right_floor = min(left_init + _PARAM_EPS, 1.0 - _PARAM_EPS)
    right_init = float(np.clip(np.max(value), right_floor, 1.0 - _PARAM_EPS))
    span_init = float(np.clip((right_init - left_init) / (1.0 - left_init), _PARAM_EPS, 1.0 - _PARAM_EPS))

    log_maf = np.log(np.clip(maf, _MAF_EPS, None))
    x_centered = log_maf - np.mean(log_maf)
    y_centered = value - np.mean(value)
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom > 0.0:
        corr = float(np.sum(x_centered * y_centered) / denom)
        slope_sign = -1.0 if corr >= 0.0 else 1.0
    else:
        slope_sign = -1.0 if value[-1] >= value[0] else 1.0

    target = left_init + 0.5 * (right_init - left_init)
    closest_index = int(np.argmin(np.abs(value - target)))

    midpoint_low = float(np.exp(log_mid_min))
    midpoint_high = float(np.exp(log_mid_min + log_mid_span))
    midpoint_init = float(maf[closest_index])
    if not np.isfinite(midpoint_init) or midpoint_init <= 0.0:
        midpoint_init = float(np.sqrt(midpoint_low * midpoint_high))
    midpoint_init = float(np.clip(midpoint_init, midpoint_low, midpoint_high))

    midpoint_fraction = float(
        np.clip(
            (np.log(midpoint_init + _MAF_EPS) - log_mid_min) / log_mid_span,
            _PARAM_EPS,
            1.0 - _PARAM_EPS,
        )
    )

    return np.array(
        [_logit(left_init), _logit(span_init), 2.0 * slope_sign, _logit(midpoint_fraction)],
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
    abs_error = np.abs(prediction - value)
    rmse = float(np.sqrt(np.mean((prediction - value) ** 2)))
    max_abs_error = float(np.max(abs_error))
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


def curve(maf: np.ndarray, coef: np.ndarray) -> np.ndarray:
    r"""Evaluate the bounded four-parameter curve model over MAF inputs."""
    left_asym, right_asym, rate, midpoint = coef
    ratio = (np.clip(np.asarray(maf, dtype=float), 0.0, None) + _MAF_EPS) / (midpoint + _MAF_EPS)
    return left_asym + (right_asym - left_asym) / (1.0 + np.power(ratio, rate))


def fit_curve(maf: np.ndarray, value: np.ndarray) -> Solution:
    r"""Fit curve coefficients with Levenberg-Marquardt least squares.

    **Arguments:**

    - `maf`: 1D MAF values.
    - `value`: 1D target values aligned with `maf`.

    **Returns:**

    - `Solution` with fitted coefficients in `value`.

    **Failure Modes:**

    - `RESULTS.nonfinite_objective` for solver failures or non-finite residuals.
    - `RESULTS.max_steps_reached` when the solver does not converge.
    """
    maf_arr = np.asarray(maf, dtype=float)
    value_arr = np.asarray(value, dtype=float)
    n_obs = int(maf_arr.size)

    log_mid_min, log_mid_span = _midpoint_bounds(maf_arr)
    init = _init_latent_parameters(maf_arr, value_arr, log_mid_min, log_mid_span)

    def residuals(latent: np.ndarray) -> np.ndarray:
        prediction = curve(maf_arr, _decode_latent(latent, log_mid_min, log_mid_span))
        raw_res = prediction - value_arr
        log_res = np.log(prediction + _MAF_EPS) - np.log(value_arr + _MAF_EPS)
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
    if not np.isfinite(coef).all():
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "curve coefficients are non-finite"},
        )

    residual = curve(maf_arr, coef) - value_arr
    if not np.isfinite(residual).all():
        return Solution(
            value=coef,
            result=RESULTS.nonfinite_objective,
            stats={
                "reason": "curve residuals are non-finite",
                "n_obs": n_obs,
                "epoch_count": int(result.nfev),
                "converged": False,
            },
        )

    # scipy.optimize.least_squares success: status > 0
    converged = result.status > 0
    mapped_result = RESULTS.successful if converged else RESULTS.max_steps_reached
    diagnostics = _fit_diagnostics(value_arr, curve(maf_arr, coef))
    return Solution(
        value=coef,
        result=mapped_result,
        stats={
            "n_obs": n_obs,
            "epoch_count": int(result.nfev),
            "converged": converged,
            **diagnostics,
        },
    )
