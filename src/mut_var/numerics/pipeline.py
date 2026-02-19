from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as rdm
import numpy as np

from jaxtyping import ArrayLike

from mut_var.contracts import RESULTS, Solution
from mut_var.numerics.baseline import BaselineConfig, Params, fit_baseline
from mut_var.numerics.refit import RefitConfig, fit_refit_grid


@dataclass(frozen=True, slots=True)
class InferenceArrays:
    af: ArrayLike
    beta_hat: ArrayLike
    s2: ArrayLike


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    num_clusters: int
    batch_size: int = 10_000
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01
    filter_threshold: float = 1e-8
    penalty: float = 1.0


def _filter_components(params: Params, threshold: float) -> Params:
    keep = params.pi > threshold
    keep = keep.at[0].set(True)
    pi = params.pi[keep]
    pi = pi / jnp.sum(pi)
    return Params(
        pi=pi,
        mu_k=params.mu_k[keep[1:]],
        var_k=params.var_k[keep[1:]],
    )


def _build_long_payload(models: list[Params], maf_grid: ArrayLike, af: ArrayLike) -> dict[str, Any]:
    maf_arr = np.asarray(maf_grid, dtype=float)
    af_arr = np.asarray(af, dtype=float)

    empirical_min_maf = float(min(np.min(af_arr), 1.0 - np.max(af_arr)))
    maf_values = np.concatenate(([empirical_min_maf], maf_arr))
    names = np.array([f"pi{idx}" for idx in range(len(models))], dtype=object)

    mu0 = np.asarray(jnp.pad(models[0].mu_k, (1, 0)), dtype=float)
    var0 = np.asarray(jnp.pad(models[0].var_k, (1, 0)), dtype=float)

    if any(len(np.asarray(model.pi)) != len(mu0) for model in models):
        raise ValueError("All models must keep the same number of mixture components.")

    values = np.concatenate([np.asarray(model.pi, dtype=float) for model in models])
    n_comp = len(mu0)

    return {
        "mu0": np.tile(mu0, len(models)),
        "var0": np.tile(var0, len(models)),
        "maf": np.repeat(maf_values, n_comp),
        "name": np.repeat(names, n_comp),
        "value": values,
    }


def run_inference_pipeline(
    arrays: InferenceArrays,
    maf_grid: ArrayLike,
    maf_masks: ArrayLike,
    seed: int,
    config: InferenceConfig,
) -> Solution:
    beta_hat = jnp.asarray(arrays.beta_hat)
    s2 = jnp.asarray(arrays.s2)

    baseline_solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(seed),
        config=BaselineConfig(
            num_clusters=config.num_clusters,
            batch_size=config.batch_size,
            max_iter=config.max_iter,
            tol=config.tol,
            step_size=config.step_size,
        ),
    )
    if baseline_solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        return baseline_solution

    filtered = _filter_components(baseline_solution.value, config.filter_threshold)

    refit_solution = fit_refit_grid(
        beta_hat=beta_hat,
        s2=s2,
        maf_masks=maf_masks,
        init=filtered,
        config=RefitConfig(
            penalty=config.penalty,
            max_iter=config.max_iter,
            tol=config.tol,
            step_size=config.step_size,
        ),
    )
    if refit_solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
        return refit_solution

    models: list[Params] = refit_solution.value
    payload = _build_long_payload(models, maf_grid=maf_grid, af=arrays.af)

    result = RESULTS.successful
    if baseline_solution.result == RESULTS.max_steps_reached or refit_solution.result == RESULTS.max_steps_reached:
        result = RESULTS.max_steps_reached

    return Solution(
        value=payload,
        result=result,
        stats={
            "num_models": len(models),
            "num_components": int(len(np.asarray(models[0].pi))),
            "baseline": baseline_solution.stats,
            "refit": refit_solution.stats,
        },
        state=None,
    )
