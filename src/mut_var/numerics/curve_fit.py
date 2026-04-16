from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from jaxtyping import ArrayLike

from mut_var.numerics._optimistix_solver import map_optimistix_result
from mut_var.types import RESULTS, Solution

_MAF_EPS = 1e-12
_PARAM_EPS = 1e-9
_LOG_RESIDUAL_WEIGHT = 2e-3

_POOR_FIT_RMSE_THRESHOLD = 3e-2
_POOR_FIT_MAX_ABS_THRESHOLD = 1.5e-1
_POOR_FIT_NONMONOTONE_SIGN_CHANGES = 2
_POOR_FIT_NONMONOTONE_MAX_ABS_THRESHOLD = 2e-2


def _logit(prob: ArrayLike):
    prob_arr = jnp.asarray(prob)
    clipped = jnp.clip(prob_arr, _PARAM_EPS, 1.0 - _PARAM_EPS)
    return jnp.log(clipped) - jnp.log1p(-clipped)


def _sigmoid(logit_value: ArrayLike):
    return jax.nn.sigmoid(jnp.asarray(logit_value))


def _midpoint_bounds(maf: ArrayLike) -> tuple[float, float]:
    maf_arr = np.asarray(maf, dtype=float)
    positive = maf_arr[maf_arr > 0.0]
    if positive.size == 0:
        return float(np.log(_MAF_EPS)), 1.0

    log_min = float(np.log(np.min(positive) + _MAF_EPS))
    log_max = float(np.log(np.max(positive) + _MAF_EPS))
    log_span = max(log_max - log_min, 1e-6)
    return log_min, float(log_span)


def _decode_latent(latent_coef: ArrayLike, log_mid_min: float, log_mid_span: float):
    raw_left, raw_span, raw_rate, raw_mid = jnp.asarray(latent_coef)
    left_asym = _sigmoid(raw_left)
    span = _sigmoid(raw_span)
    right_asym = left_asym + (1.0 - left_asym) * span
    midpoint = jnp.exp(log_mid_min + _sigmoid(raw_mid) * log_mid_span)
    return jnp.asarray([left_asym, right_asym, raw_rate, midpoint], dtype=float)


def _init_latent_parameters(maf: ArrayLike, value: ArrayLike, log_mid_min: float, log_mid_span: float):
    maf_arr = jnp.asarray(maf)
    value_arr = jnp.asarray(value)
    left_init = jnp.clip(jnp.min(value_arr), _PARAM_EPS, 1.0 - _PARAM_EPS)
    right_floor = jnp.minimum(left_init + _PARAM_EPS, 1.0 - _PARAM_EPS)
    right_init = jnp.clip(jnp.max(value_arr), right_floor, 1.0 - _PARAM_EPS)
    span_init = jnp.clip((right_init - left_init) / (1.0 - left_init), _PARAM_EPS, 1.0 - _PARAM_EPS)

    log_maf = jnp.log(jnp.clip(maf_arr, _MAF_EPS))
    x_centered = log_maf - jnp.mean(log_maf)
    y_centered = value_arr - jnp.mean(value_arr)
    denom = jnp.sqrt(jnp.sum(x_centered * x_centered) * jnp.sum(y_centered * y_centered))
    corr = jnp.where(denom > 0.0, jnp.sum(x_centered * y_centered) / denom, jnp.nan)

    endpoint_sign = jnp.where(value_arr[-1] >= value_arr[0], -1.0, 1.0)
    corr_sign = jnp.where(corr >= 0.0, -1.0, 1.0)
    slope_sign = jnp.where(jnp.isfinite(corr), corr_sign, endpoint_sign)

    target = left_init + 0.5 * (right_init - left_init)
    closest_index = int(np.argmin(np.abs(np.asarray(value_arr, dtype=float) - float(target))))

    midpoint_low = float(np.exp(log_mid_min))
    midpoint_high = float(np.exp(log_mid_min + log_mid_span))
    midpoint_init = float(np.asarray(maf_arr, dtype=float)[closest_index])
    if not np.isfinite(midpoint_init) or midpoint_init <= 0.0:
        midpoint_init = float(np.sqrt(midpoint_low * midpoint_high))
    midpoint_init = float(np.clip(midpoint_init, midpoint_low, midpoint_high))

    midpoint_fraction = np.clip(
        (np.log(midpoint_init + _MAF_EPS) - log_mid_min) / log_mid_span,
        _PARAM_EPS,
        1.0 - _PARAM_EPS,
    )

    return jnp.asarray(
        [_logit(left_init), _logit(span_init), 2.0 * slope_sign, _logit(midpoint_fraction)],
        dtype=float,
    )


def _curve_status_from_optx(optx_result) -> RESULTS:
    mapped = map_optimistix_result(optx_result)
    if mapped != RESULTS.nonfinite_objective:
        return mapped

    if hasattr(optx.RESULTS, "singular") and optx_result == optx.RESULTS.singular:
        return cast(RESULTS, RESULTS.max_steps_reached)
    if hasattr(optx.RESULTS, "breakdown") and optx_result == optx.RESULTS.breakdown:
        return cast(RESULTS, RESULTS.max_steps_reached)
    return mapped


def _count_sign_changes(value: ArrayLike) -> int:
    diffs = np.diff(np.asarray(value, dtype=float))
    signs = np.sign(diffs)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def _fit_diagnostics(value: ArrayLike, prediction: ArrayLike) -> dict[str, float | int | bool]:
    value_arr = np.asarray(value, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    abs_error = np.abs(pred_arr - value_arr)
    rmse = float(np.sqrt(np.mean((pred_arr - value_arr) ** 2)))
    max_abs_error = float(np.max(abs_error))
    data_sign_changes = _count_sign_changes(value_arr)
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


def curve(maf: ArrayLike, coef: ArrayLike):
    r"""Evaluate bounded four-parameter curve model over MAF inputs."""
    left_asym, right_asym, rate, midpoint = coef
    ratio = (jnp.clip(jnp.asarray(maf), 0.0) + _MAF_EPS) / (midpoint + _MAF_EPS)
    return left_asym + (right_asym - left_asym) / (1.0 + jnp.power(ratio, rate))


def _objective(coef, args):
    maf, value, log_mid_min, log_mid_span = args
    prediction = curve(maf, _decode_latent(coef, log_mid_min, log_mid_span))
    raw_residual = prediction - value
    log_residual = jnp.log(prediction + _MAF_EPS) - jnp.log(value + _MAF_EPS)
    return jnp.concatenate([raw_residual, _LOG_RESIDUAL_WEIGHT * log_residual], axis=0)


def _epoch_count(optx_solution) -> int:
    stats = getattr(optx_solution, "stats", None)
    if isinstance(stats, dict):
        return int(stats.get("num_steps", 0))
    return 0


def fit_curve(maf: ArrayLike, value: ArrayLike) -> Solution:
    r"""Fit curve coefficients with Levenberg-Marquardt least squares.

    **Arguments:**

    - `maf`: 1D MAF values.
    - `value`: 1D target values aligned with `maf`.

    **Returns:**

    - `Solution` with fitted coefficients in `value`.

    **Failure Modes:**

    - `RESULTS.nonfinite_objective` for solver failures or non-finite coefficients/residuals.
    - `RESULTS.max_steps_reached` when Optimistix does not converge in `max_steps`.
    """
    maf_arr = jnp.asarray(maf)
    value_arr = jnp.asarray(value)
    n_obs = int(maf_arr.size)

    log_mid_min, log_mid_span = _midpoint_bounds(maf_arr)
    solver = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-3)
    init = _init_latent_parameters(maf_arr, value_arr, log_mid_min, log_mid_span)
    try:
        result = optx.least_squares(
            _objective,
            solver,
            y0=init,
            args=(maf_arr, value_arr, log_mid_min, log_mid_span),
            max_steps=1000,
            throw=False,
        )
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"curve fit failed: {exc}"},
            state=None,
        )

    mapped_result = _curve_status_from_optx(result.result)
    coef = _decode_latent(result.value, log_mid_min, log_mid_span)
    if not bool(jnp.isfinite(coef).all()):
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "curve coefficients are non-finite"},
            state=None,
        )

    residual = curve(maf_arr, coef) - value_arr
    if not bool(jnp.isfinite(residual).all()):
        return Solution(
            value=coef,
            result=RESULTS.nonfinite_objective,
            stats={
                "reason": "curve residuals are non-finite",
                "n_obs": n_obs,
                "epoch_count": _epoch_count(result),
                "converged": False,
            },
            state=None,
        )

    diagnostics = _fit_diagnostics(value_arr, curve(maf_arr, coef))
    return Solution(
        value=coef,
        result=mapped_result,
        stats={
            "n_obs": n_obs,
            "epoch_count": _epoch_count(result),
            "converged": bool(mapped_result == RESULTS.successful),
            **diagnostics,
        },
        state=None,
    )
