from __future__ import annotations

# pattern: Functional Core
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import ArrayLike

from mut_var.contracts import Solution
from mut_var.numerics._solver_utils import is_recoverable_result, merge_recoverable_results
from mut_var.numerics.baseline import BaselineConfig, fit_baseline, Params
from mut_var.numerics.profiling import profile_solution_runs
from mut_var.numerics.refit import fit_refit_grid, RefitConfig


class InferenceArrays(NamedTuple):
    af: ArrayLike
    beta_hat: ArrayLike
    s2: ArrayLike


class InferenceConfig(NamedTuple):
    num_clusters: int
    max_iter: int = 100
    tol: float = 1e-3
    step_size: float = 0.01
    filter_threshold: float = 1e-8
    penalty: float = 1.0

    def to_baseline_config(self) -> BaselineConfig:
        r"""Convert pipeline controls to baseline-stage solver config."""
        return BaselineConfig(
            num_clusters=self.num_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )

    def to_refit_config(self) -> RefitConfig:
        r"""Convert pipeline controls to refit-stage solver config."""
        return RefitConfig(
            penalty=self.penalty,
            max_iter=self.max_iter,
            tol=self.tol,
            step_size=self.step_size,
        )


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
    maf_arr = jnp.asarray(maf_grid, dtype=jnp.float64)
    af_arr = jnp.asarray(af, dtype=jnp.float64)

    empirical_min_maf = jnp.minimum(jnp.min(af_arr), 1.0 - jnp.max(af_arr))
    maf_values = jnp.concatenate((jnp.asarray([empirical_min_maf], dtype=jnp.float64), maf_arr))
    names = [f"pi{idx}" for idx in range(len(models))]

    mu0 = jnp.asarray(jnp.pad(models[0].mu_k, (1, 0)), dtype=jnp.float64)
    var0 = jnp.asarray(jnp.pad(models[0].var_k, (1, 0)), dtype=jnp.float64)

    if any(model.pi.shape[0] != mu0.shape[0] for model in models):
        raise ValueError("All models must keep the same number of mixture components.")

    values = jnp.concatenate([jnp.asarray(model.pi, dtype=jnp.float64) for model in models])
    n_comp = int(mu0.shape[0])
    name_values = [name for name in names for _ in range(n_comp)]

    return {
        "mu0": jnp.tile(mu0, len(models)),
        "var0": jnp.tile(var0, len(models)),
        "maf": jnp.repeat(maf_values, n_comp),
        "name": name_values,
        "value": values,
    }


def run_inference_pipeline(
    arrays: InferenceArrays,
    maf_grid: ArrayLike,
    maf_masks: ArrayLike,
    seed: int,
    config: InferenceConfig,
) -> Solution:
    r"""Run numerics-only inference (baseline -> refit -> payload build).

    **Arguments:**
    - `arrays`: Array inputs for AF, beta, and variance.
    - `maf_grid`: MAF threshold grid.
    - `maf_masks`: Boolean mask matrix aligned with observations.
    - `seed`: PRNG seed for baseline initialization.
    - `config`: Pipeline solver controls.

    **Returns:**
    - `Solution` carrying long-format payload mapping and status diagnostics.
    """
    beta_hat = jnp.asarray(arrays.beta_hat)
    s2 = jnp.asarray(arrays.s2)

    baseline_solution = fit_baseline(
        beta_hat=beta_hat,
        s2=s2,
        key=rdm.PRNGKey(seed),
        config=config.to_baseline_config(),
    )
    if not is_recoverable_result(baseline_solution.result):
        return baseline_solution

    filtered = _filter_components(baseline_solution.value, config.filter_threshold)

    refit_solution = fit_refit_grid(
        beta_hat=beta_hat,
        s2=s2,
        maf_masks=maf_masks,
        init=filtered,
        config=config.to_refit_config(),
    )
    if not is_recoverable_result(refit_solution.result):
        return refit_solution

    models: list[Params] = refit_solution.value
    payload = _build_long_payload(models, maf_grid=maf_grid, af=arrays.af)

    return Solution(
        value=payload,
        result=merge_recoverable_results(baseline_solution.result, refit_solution.result),
        stats={
            "num_models": len(models),
            "num_components": int(models[0].pi.shape[0]),
            "baseline": baseline_solution.stats,
            "refit": refit_solution.stats,
        },
        state=None,
    )


def run_profiled_inference_pipeline(
    arrays: InferenceArrays,
    maf_grid: ArrayLike,
    maf_masks: ArrayLike,
    seed: int,
    config: InferenceConfig,
    steady_runs: int = 3,
) -> dict[str, object]:
    r"""Profile compile and steady-state timings for inference numerics pipeline."""
    return profile_solution_runs(
        lambda: run_inference_pipeline(
            arrays=arrays,
            maf_grid=maf_grid,
            maf_masks=maf_masks,
            seed=seed,
            config=config,
        ),
        steady_runs=steady_runs,
    )
