from __future__ import annotations

# pattern: Functional Core
from typing import Literal, NamedTuple

import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import ArrayLike

from mut_var.types import RESULTS, Solution

VarianceLink = Literal["none", "maf_power", "maf_power_shifted"]
AfModel = Literal["uniform", "beta"]
SeModel = Literal["constant", "af_n_scaled"]


class SimulationArrays(NamedTuple):
    row_id: ArrayLike
    af: ArrayLike
    component: ArrayLike
    sigma2: ArrayLike
    beta_true: ArrayLike
    se: ArrayLike
    beta_hat: ArrayLike


class SimulationNumericsConfig(NamedTuple):
    weights: tuple[float, ...]
    log_var_scales: tuple[float, ...]
    variance_link: VarianceLink = "maf_power"
    theta: float = 0.5
    link_eps: float = 1e-8
    link_shift: float = 0.0
    af_clip_min: float = 1e-4
    af_model: AfModel = "beta"
    af_uniform_low: float = 0.01
    af_uniform_high: float = 0.5
    af_beta_a: float = 0.4
    af_beta_b: float = 0.4
    se_model: SeModel = "af_n_scaled"
    se_constant: float = 0.02
    sample_size: float = 50000.0
    se_scale: float = 1.0


def _invalid(reason: str) -> Solution:
    return Solution(
        value=None,
        result=RESULTS.invalid_input,
        stats={"reason": reason},
        state=None,
    )


def _nonfinite(reason: str) -> Solution:
    return Solution(
        value=None,
        result=RESULTS.nonfinite_objective,
        stats={"reason": reason},
        state=None,
    )


def _validate_simulation_inputs(n_rows: int, config: SimulationNumericsConfig) -> Solution | None:
    if not isinstance(n_rows, int):
        return _invalid("n_rows must be an integer >= 1")
    if n_rows < 1:
        return _invalid("n_rows must be >= 1")

    num_components = len(config.weights)
    if num_components < 2:
        return _invalid("weights must contain at least 2 components")
    if num_components != len(config.log_var_scales):
        return _invalid("weights and log_var_scales must have the same length")

    weights = jnp.asarray(config.weights, dtype=jnp.float64)
    log_var_scales = jnp.asarray(config.log_var_scales, dtype=jnp.float64)

    if weights.ndim != 1 or log_var_scales.ndim != 1:
        return _invalid("weights and log_var_scales must be 1D sequences")
    if not bool(jnp.isfinite(weights).all()) or not bool(jnp.isfinite(log_var_scales).all()):
        return _invalid("weights and log_var_scales must be finite")
    if bool((weights <= 0.0).any()):
        return _invalid("all mixture weights must be strictly positive")
    if abs(float(jnp.sum(weights)) - 1.0) > 1e-8:
        return _invalid("mixture weights must sum to 1")

    if config.variance_link not in ("none", "maf_power", "maf_power_shifted"):
        return _invalid("variance_link must be one of: none, maf_power, maf_power_shifted")
    if not jnp.isfinite(config.theta):
        return _invalid("theta must be finite")
    if config.theta < 0.0 or config.theta > 1.5:
        return _invalid("theta must satisfy 0 <= theta <= 1.5")

    if not jnp.isfinite(config.af_clip_min):
        return _invalid("af_clip_min must be finite")
    if config.af_clip_min <= 0.0 or config.af_clip_min >= 0.5:
        return _invalid("af_clip_min must satisfy 0 < af_clip_min < 0.5")

    if config.variance_link == "maf_power":
        if not jnp.isfinite(config.link_eps) or config.link_eps <= 0.0:
            return _invalid("link_eps must be finite and > 0 for maf_power")
    if config.variance_link == "maf_power_shifted":
        if not jnp.isfinite(config.link_shift) or config.link_shift <= 0.0:
            return _invalid("link_shift must be finite and > 0 for maf_power_shifted")

    if config.af_model not in ("uniform", "beta"):
        return _invalid("af_model must be one of: uniform, beta")
    if config.af_model == "uniform":
        if not jnp.isfinite(config.af_uniform_low) or not jnp.isfinite(config.af_uniform_high):
            return _invalid("af_uniform_low and af_uniform_high must be finite")
        if config.af_uniform_low <= 0.0 or config.af_uniform_high <= 0.0:
            return _invalid("uniform AF bounds must be > 0")
        if config.af_uniform_low >= config.af_uniform_high:
            return _invalid("af_uniform_low must be < af_uniform_high")
        if config.af_uniform_high > 1.0:
            return _invalid("af_uniform_high must be <= 1")
    if config.af_model == "beta":
        if not jnp.isfinite(config.af_beta_a) or not jnp.isfinite(config.af_beta_b):
            return _invalid("af_beta_a and af_beta_b must be finite")
        if config.af_beta_a <= 0.0 or config.af_beta_b <= 0.0:
            return _invalid("beta AF shape parameters must be > 0")

    if config.se_model not in ("constant", "af_n_scaled"):
        return _invalid("se_model must be one of: constant, af_n_scaled")
    if config.se_model == "constant":
        if not jnp.isfinite(config.se_constant) or config.se_constant <= 0.0:
            return _invalid("se_constant must be finite and > 0")
    if config.se_model == "af_n_scaled":
        if not jnp.isfinite(config.sample_size) or config.sample_size <= 0.0:
            return _invalid("sample_size must be finite and > 0")
        if not jnp.isfinite(config.se_scale) or config.se_scale <= 0.0:
            return _invalid("se_scale must be finite and > 0")

    return None


def _sample_af(key: rdm.PRNGKey, n_rows: int, config: SimulationNumericsConfig) -> ArrayLike:
    if config.af_model == "uniform":
        return rdm.uniform(
            key,
            shape=(n_rows,),
            minval=float(config.af_uniform_low),
            maxval=float(config.af_uniform_high),
            dtype=jnp.float64,
        )
    return rdm.beta(
        key,
        a=float(config.af_beta_a),
        b=float(config.af_beta_b),
        shape=(n_rows,),
        dtype=jnp.float64,
    )


def _variance_link(af: ArrayLike, config: SimulationNumericsConfig) -> ArrayLike:
    p = jnp.clip(jnp.asarray(af, dtype=jnp.float64), config.af_clip_min, 1.0 - config.af_clip_min)
    maf_term = 2.0 * p * (1.0 - p)
    if config.variance_link == "none":
        return jnp.ones_like(maf_term)
    if config.variance_link == "maf_power":
        return (maf_term + config.link_eps) ** (-config.theta)
    return (maf_term + config.link_shift) ** (-config.theta)


def _se_from_af(af: ArrayLike, config: SimulationNumericsConfig) -> ArrayLike:
    if config.se_model == "constant":
        return jnp.full_like(jnp.asarray(af, dtype=jnp.float64), config.se_constant, dtype=jnp.float64)

    p = jnp.clip(jnp.asarray(af, dtype=jnp.float64), config.af_clip_min, 1.0 - config.af_clip_min)
    maf_term = 2.0 * p * (1.0 - p)
    return config.se_scale / jnp.sqrt(maf_term * config.sample_size)


def _all_finite(arrays: SimulationArrays) -> bool:
    return bool(
        jnp.isfinite(arrays.row_id).all()
        and jnp.isfinite(arrays.af).all()
        and jnp.isfinite(arrays.component).all()
        and jnp.isfinite(arrays.sigma2).all()
        and jnp.isfinite(arrays.beta_true).all()
        and jnp.isfinite(arrays.se).all()
        and jnp.isfinite(arrays.beta_hat).all()
    )


def simulate_mixture_data(
    *,
    n_rows: int,
    seed: int,
    config: SimulationNumericsConfig,
) -> Solution:
    r"""Simulate latent and observed summary-stat data under a zero-mean mixture model.

    **Arguments:**

    - `n_rows`: Number of variants to simulate.
    - `seed`: PRNG seed used to initialize JAX key splitting.
    - `config`: Mixture, AF generator, variance-link, and SE model controls.

    **Returns:**

    - `Solution` with `SimulationArrays` payload on success.

    **Failure Modes:**

    - `RESULTS.invalid_input` for domain or shape violations.
    - `RESULTS.nonfinite_objective` when generated arrays are non-finite.
    """
    invalid = _validate_simulation_inputs(n_rows, config)
    if invalid is not None:
        return invalid

    key = rdm.PRNGKey(seed)
    key_af, key_comp, key_beta_true, key_beta_hat = rdm.split(key, 4)

    af = _sample_af(key_af, n_rows, config)
    link = _variance_link(af, config)

    weights = jnp.asarray(config.weights, dtype=jnp.float64)
    logits = jnp.log(weights)
    component = rdm.categorical(key_comp, logits=logits, shape=(n_rows,))

    component_scales = jnp.exp(jnp.asarray(config.log_var_scales, dtype=jnp.float64))
    sigma2 = component_scales[component] * link
    if bool((sigma2 <= 0.0).any()):
        return _nonfinite("generated sigma2 contains non-positive values")

    se = _se_from_af(af, config)
    beta_true = rdm.normal(key_beta_true, shape=(n_rows,), dtype=jnp.float64) * jnp.sqrt(sigma2)
    beta_hat = beta_true + rdm.normal(key_beta_hat, shape=(n_rows,), dtype=jnp.float64) * se

    arrays = SimulationArrays(
        row_id=jnp.arange(n_rows, dtype=jnp.int64),
        af=af,
        component=component,
        sigma2=sigma2,
        beta_true=beta_true,
        se=se,
        beta_hat=beta_hat,
    )

    if not _all_finite(arrays):
        return _nonfinite("generated arrays contain non-finite values")

    stats = {
        "n_rows": int(n_rows),
        "num_components": int(len(config.weights)),
        "variance_link": config.variance_link,
        "theta": float(config.theta),
        "seed": int(seed),
    }
    return Solution(value=arrays, result=RESULTS.successful, stats=stats, state=None)


__all__ = [
    "SimulationArrays",
    "SimulationNumericsConfig",
    "simulate_mixture_data",
]
