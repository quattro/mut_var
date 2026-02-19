from __future__ import annotations

import jax.numpy as jnp
import optimistix as optx

from jaxtyping import ArrayLike

from mut_var.contracts import RESULTS, Solution


def curve(maf: ArrayLike, coef: ArrayLike):
    left_asym, right_asym, rate = coef
    return left_asym + (right_asym - left_asym) / (1.0 + jnp.power(maf, rate))


def _objective(coef, args):
    maf, value = args
    return curve(maf, coef) - value


def fit_curve(maf: ArrayLike, value: ArrayLike) -> Solution:
    try:
        maf_arr = jnp.asarray(maf)
        value_arr = jnp.asarray(value)
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": f"failed to convert curve inputs to arrays: {exc}"},
            state=None,
        )

    if maf_arr.ndim != 1 or value_arr.ndim != 1 or maf_arr.shape[0] != value_arr.shape[0]:
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "maf and value must be 1D arrays of equal length"},
            state=None,
        )

    if maf_arr.shape[0] == 0:
        return Solution(
            value=None,
            result=RESULTS.empty_subset,
            stats={"reason": "curve fit received empty input"},
            state=None,
        )

    if not bool(jnp.isfinite(maf_arr).all()) or not bool(jnp.isfinite(value_arr).all()):
        return Solution(
            value=None,
            result=RESULTS.invalid_input,
            stats={"reason": "maf/value must be finite"},
            state=None,
        )

    solver = optx.LevenbergMarquardt(rtol=1e-3, atol=1e-3)
    try:
        result = optx.least_squares(
            _objective,
            solver,
            y0=jnp.ones(3),
            args=(maf_arr, value_arr),
            max_steps=1000,
        )
    except Exception as exc:
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": f"curve fit failed: {exc}"},
            state=None,
        )

    coef = result.value
    if not bool(jnp.isfinite(coef).all()):
        return Solution(
            value=None,
            result=RESULTS.nonfinite_objective,
            stats={"reason": "curve coefficients are non-finite"},
            state={"solver_result": str(result.result)},
        )

    return Solution(
        value=coef,
        result=RESULTS.successful,
        stats={"n_obs": int(maf_arr.shape[0]), "solver_result": str(result.result)},
        state=None,
    )
