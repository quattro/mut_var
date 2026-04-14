from __future__ import annotations

# pattern: Functional Core
from typing import NamedTuple

import numpy as np

from mut_var.config import SimulationConfig
from mut_var.contracts import RESULTS, Solution


class SimulationArrays(NamedTuple):
    row_id: np.ndarray
    af: np.ndarray
    component: np.ndarray
    sigma2: np.ndarray
    beta_true: np.ndarray
    se: np.ndarray
    beta_hat: np.ndarray


def _invalid(reason: str) -> Solution:
    return Solution(value=None, result=RESULTS.invalid_input, stats={"reason": reason})


def _nonfinite(reason: str) -> Solution:
    return Solution(value=None, result=RESULTS.nonfinite_objective, stats={"reason": reason})


def _validate_simulation_inputs(config: SimulationConfig) -> Solution | None:
    if not isinstance(config.n_rows, int):
        return _invalid("n_rows must be an integer >= 1")
    if config.n_rows < 1:
        return _invalid("n_rows must be >= 1")

    num_components = len(config.weights)
    if num_components < 2:
        return _invalid("weights must contain at least 2 components")
    if num_components != len(config.log_var_scales):
        return _invalid("weights and log_var_scales must have the same length")

    weights = np.asarray(config.weights, dtype=float)
    log_var_scales = np.asarray(config.log_var_scales, dtype=float)

    if not np.isfinite(weights).all() or not np.isfinite(log_var_scales).all():
        return _invalid("weights and log_var_scales must be finite")
    if (weights <= 0.0).any():
        return _invalid("all mixture weights must be strictly positive")
    if abs(float(weights.sum()) - 1.0) > 1e-8:
        return _invalid("mixture weights must sum to 1")

    if config.variance_link not in ("none", "maf_power", "maf_power_shifted"):
        return _invalid("variance_link must be one of: none, maf_power, maf_power_shifted")
    if not np.isfinite(config.theta):
        return _invalid("theta must be finite")
    if config.theta < 0.0 or config.theta > 1.5:
        return _invalid("theta must satisfy 0 <= theta <= 1.5")

    if not np.isfinite(config.af_clip_min):
        return _invalid("af_clip_min must be finite")
    if config.af_clip_min <= 0.0 or config.af_clip_min >= 0.5:
        return _invalid("af_clip_min must satisfy 0 < af_clip_min < 0.5")

    if config.variance_link == "maf_power":
        if not np.isfinite(config.link_eps) or config.link_eps <= 0.0:
            return _invalid("link_eps must be finite and > 0 for maf_power")
    if config.variance_link == "maf_power_shifted":
        if not np.isfinite(config.link_shift) or config.link_shift <= 0.0:
            return _invalid("link_shift must be finite and > 0 for maf_power_shifted")

    if config.af_model not in ("uniform", "beta"):
        return _invalid("af_model must be one of: uniform, beta")
    if config.af_model == "uniform":
        if not np.isfinite(config.af_uniform_low) or not np.isfinite(config.af_uniform_high):
            return _invalid("af_uniform_low and af_uniform_high must be finite")
        if config.af_uniform_low <= 0.0 or config.af_uniform_high <= 0.0:
            return _invalid("uniform AF bounds must be > 0")
        if config.af_uniform_low >= config.af_uniform_high:
            return _invalid("af_uniform_low must be < af_uniform_high")
        if config.af_uniform_high > 1.0:
            return _invalid("af_uniform_high must be <= 1")
    if config.af_model == "beta":
        if not np.isfinite(config.af_beta_a) or not np.isfinite(config.af_beta_b):
            return _invalid("af_beta_a and af_beta_b must be finite")
        if config.af_beta_a <= 0.0 or config.af_beta_b <= 0.0:
            return _invalid("beta AF shape parameters must be > 0")

    if config.se_model not in ("constant", "af_n_scaled"):
        return _invalid("se_model must be one of: constant, af_n_scaled")
    if config.se_model == "constant":
        if not np.isfinite(config.se_constant) or config.se_constant <= 0.0:
            return _invalid("se_constant must be finite and > 0")
    if config.se_model == "af_n_scaled":
        if not np.isfinite(config.sample_size) or config.sample_size <= 0.0:
            return _invalid("sample_size must be finite and > 0")
        if not np.isfinite(config.se_scale) or config.se_scale <= 0.0:
            return _invalid("se_scale must be finite and > 0")

    return None


def _sample_af(rng: np.random.Generator, n_rows: int, config: SimulationConfig) -> np.ndarray:
    if config.af_model == "uniform":
        return rng.uniform(
            low=float(config.af_uniform_low),
            high=float(config.af_uniform_high),
            size=n_rows,
        ).astype(float)
    return rng.beta(
        a=float(config.af_beta_a),
        b=float(config.af_beta_b),
        size=n_rows,
    ).astype(float)


def _variance_link(af: np.ndarray, config: SimulationConfig) -> np.ndarray:
    p = np.clip(af, config.af_clip_min, 1.0 - config.af_clip_min)
    maf_term = 2.0 * p * (1.0 - p)
    if config.variance_link == "none":
        return np.ones_like(maf_term)
    if config.variance_link == "maf_power":
        return (maf_term + config.link_eps) ** (-config.theta)
    return (maf_term + config.link_shift) ** (-config.theta)


def _se_from_af(af: np.ndarray, config: SimulationConfig) -> np.ndarray:
    if config.se_model == "constant":
        return np.full(len(af), config.se_constant, dtype=float)
    p = np.clip(af, config.af_clip_min, 1.0 - config.af_clip_min)
    maf_term = 2.0 * p * (1.0 - p)
    return config.se_scale / np.sqrt(maf_term * config.sample_size)


def _all_finite(arrays: SimulationArrays) -> bool:
    return bool(
        np.isfinite(arrays.row_id).all()
        and np.isfinite(arrays.af).all()
        and np.isfinite(arrays.component).all()
        and np.isfinite(arrays.sigma2).all()
        and np.isfinite(arrays.beta_true).all()
        and np.isfinite(arrays.se).all()
        and np.isfinite(arrays.beta_hat).all()
    )


def simulate_mixture_data(
    *,
    config: SimulationConfig,
) -> Solution:
    r"""Simulate latent and observed summary-stat data under a zero-mean mixture model.

    **Arguments:**

    - `config`: Simulation workflow controls including row count, seed,
      mixture, AF generator, variance-link, and SE model parameters.

    **Returns:**

    - `Solution` with `SimulationArrays` payload on success.

    **Failure Modes:**

    - `RESULTS.invalid_input` for domain or shape violations.
    - `RESULTS.nonfinite_objective` when generated arrays are non-finite.
    """
    invalid = _validate_simulation_inputs(config)
    if invalid is not None:
        return invalid

    rng = np.random.default_rng(config.seed)

    af = _sample_af(rng, config.n_rows, config)
    link = _variance_link(af, config)

    weights = np.asarray(config.weights, dtype=float)
    component = rng.choice(len(weights), size=config.n_rows, p=weights / weights.sum()).astype(np.int64)

    component_scales = np.exp(np.asarray(config.log_var_scales, dtype=float))
    sigma2 = component_scales[component] * link
    if (sigma2 <= 0.0).any():
        return _nonfinite("generated sigma2 contains non-positive values")

    se = _se_from_af(af, config)
    beta_true = rng.standard_normal(size=config.n_rows) * np.sqrt(sigma2)
    beta_hat = beta_true + rng.standard_normal(size=config.n_rows) * se

    arrays = SimulationArrays(
        row_id=np.arange(config.n_rows, dtype=np.int64),
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
        "n_rows": int(config.n_rows),
        "num_components": int(len(config.weights)),
        "variance_link": config.variance_link,
        "theta": float(config.theta),
        "seed": int(config.seed),
    }
    return Solution(value=arrays, result=RESULTS.successful, stats=stats)


__all__ = [
    "SimulationArrays",
    "simulate_mixture_data",
]
