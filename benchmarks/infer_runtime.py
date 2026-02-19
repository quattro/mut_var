from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from time import sleep
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as rdm
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mut_var.adapters.array_cache import ArrayConversionCache
from mut_var.adapters.tabular import build_maf_masks, to_inference_arrays, to_inference_arrays_cached
from mut_var.contracts import RESULTS
from mut_var.numerics.pipeline import InferenceConfig, run_inference_pipeline
from mut_var.numerics.profiling import evaluate_performance_gate, profile_solution_runs


class RuntimeBenchmarkConfig(NamedTuple):
    seed: int
    num_rows: int
    num_clusters: int
    max_iter: int
    num_breaks: int
    lowest: float
    highest: float
    batch_size: int
    step_size: float
    filter_threshold: float
    penalty: float
    steady_runs: int
    legacy_conversion_repeats: int
    improvement_threshold_percent: float
    legacy_conversion_delay_seconds: float = 0.0
    reviewer_name: str = "unassigned"
    review_date: str = "1970-01-01"
    signoff_decision: str = "pending"


def load_config(path: Path) -> RuntimeBenchmarkConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeBenchmarkConfig(**payload)


def generate_sumstats(config: RuntimeBenchmarkConfig) -> pl.DataFrame:
    key = rdm.PRNGKey(config.seed)
    key_af, key_beta, key_se = rdm.split(key, 3)
    af = rdm.uniform(key_af, shape=(config.num_rows,), minval=0.01, maxval=0.49)
    beta = 0.05 * rdm.normal(key_beta, shape=(config.num_rows,))
    se = rdm.uniform(key_se, shape=(config.num_rows,), minval=0.01, maxval=0.06)
    return pl.DataFrame(
        {
            "effect_allele_frequency": af.tolist(),
            "beta": beta.tolist(),
            "standard_error": se.tolist(),
        }
    )


def _maf_grid(config: RuntimeBenchmarkConfig) -> jax.Array:
    return jnp.exp(jnp.linspace(jnp.log(config.lowest), jnp.log(config.highest), config.num_breaks))


def _inference_config(config: RuntimeBenchmarkConfig) -> InferenceConfig:
    return InferenceConfig(
        num_clusters=config.num_clusters,
        batch_size=config.batch_size,
        max_iter=config.max_iter,
        step_size=config.step_size,
        filter_threshold=config.filter_threshold,
        penalty=config.penalty,
    )


def profile_path(
    df: pl.DataFrame,
    config: RuntimeBenchmarkConfig,
    *,
    use_cache: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    cache = ArrayConversionCache()
    cache_stats = {"hits": 0, "misses": 0}
    maf_grid = _maf_grid(config)
    infer_config = _inference_config(config)

    def run_once():
        if use_cache:
            arrays, hit = to_inference_arrays_cached(
                df,
                "effect_allele_frequency",
                "beta",
                "standard_error",
                cache,
            )
            if hit:
                cache_stats["hits"] += 1
            else:
                cache_stats["misses"] += 1
        else:
            arrays = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
            cache_stats["misses"] += 1
            for _ in range(max(0, config.legacy_conversion_repeats - 1)):
                # Simulates pre-refactor repeated host->device conversion in threshold workflows.
                _ = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
            if config.legacy_conversion_delay_seconds > 0:
                sleep(config.legacy_conversion_delay_seconds)

        maf_masks = build_maf_masks(arrays.af, maf_grid)
        solution = run_inference_pipeline(
            arrays=arrays,
            maf_grid=maf_grid,
            maf_masks=maf_masks,
            seed=config.seed,
            config=infer_config,
        )
        if solution.result not in (RESULTS.successful, RESULTS.max_steps_reached):
            raise RuntimeError(f"Pipeline failed during benchmark path: {RESULTS[solution.result]}")
        return solution

    return profile_solution_runs(run_once, steady_runs=config.steady_runs), cache_stats


def build_report(config: RuntimeBenchmarkConfig) -> dict[str, Any]:
    df = generate_sumstats(config)

    baseline_profile, baseline_cache = profile_path(df, config, use_cache=False)
    candidate_profile, candidate_cache = profile_path(df, config, use_cache=True)

    baseline_mean = float(baseline_profile["steady_state"]["mean_seconds"])
    candidate_mean = float(candidate_profile["steady_state"]["mean_seconds"])
    gate = evaluate_performance_gate(
        baseline_seconds=baseline_mean,
        candidate_seconds=candidate_mean,
        threshold_percent=config.improvement_threshold_percent,
    )

    return {
        "config": dict(config._asdict()),
        "compile": {
            "baseline": baseline_profile["compile"],
            "candidate": candidate_profile["compile"],
        },
        "steady_state": {
            "baseline": baseline_profile["steady_state"],
            "candidate": candidate_profile["steady_state"],
        },
        "cache": {
            "baseline": baseline_cache,
            "candidate": candidate_cache,
        },
        "comparison": gate.to_dict(),
        "passed": gate.passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = build_report(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
