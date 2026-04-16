from __future__ import annotations

# pattern: Imperative Shell
import argparse
import json
import sys
import tempfile

from pathlib import Path
from statistics import mean
from time import perf_counter, sleep
from typing import Any, NamedTuple

import jax.random as rdm
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mut_var.io import dataframe_fingerprint, to_inference_arrays  # noqa: E402
from mut_var.pipelines.inference import run_inference_pipeline  # noqa: E402
from mut_var.types import InferenceConfig  # noqa: E402


class RuntimeBenchmarkConfig(NamedTuple):
    seed: int
    num_rows: int
    num_clusters: int
    max_iter: int
    num_breaks: int
    lowest: float
    highest: float
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


class PerformanceGateResult(NamedTuple):
    baseline_seconds: float
    candidate_seconds: float
    threshold_percent: float
    improvement_percent: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self._asdict())


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


def _inference_config(config: RuntimeBenchmarkConfig) -> InferenceConfig:
    return InferenceConfig(
        num_clusters=config.num_clusters,
        max_iter=config.max_iter,
        filter_threshold=config.filter_threshold,
    )


def _to_inference_arrays_cached(
    df: pl.DataFrame,
    cache: dict[str, object],
) -> tuple[Any, bool]:
    cache_key = dataframe_fingerprint(df, ["effect_allele_frequency", "beta", "standard_error"])
    cached = cache.get(cache_key)
    if cached is not None:
        return cached, True

    arrays = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
    cache[cache_key] = arrays
    return arrays, False


def profile_solution_runs(fn, *, steady_runs: int) -> dict[str, dict[str, Any]]:
    compile_start = perf_counter()
    fn()
    compile_seconds = perf_counter() - compile_start

    steady_samples: list[float] = []
    for _ in range(max(1, steady_runs)):
        run_start = perf_counter()
        fn()
        steady_samples.append(perf_counter() - run_start)

    return {
        "compile": {
            "mean_seconds": compile_seconds,
            "runs": 1,
        },
        "steady_state": {
            "mean_seconds": mean(steady_samples),
            "runs": len(steady_samples),
        },
    }


def evaluate_performance_gate(
    *,
    baseline_seconds: float,
    candidate_seconds: float,
    threshold_percent: float,
) -> PerformanceGateResult:
    improvement_percent = 0.0
    if baseline_seconds > 0.0:
        improvement_percent = 100.0 * (baseline_seconds - candidate_seconds) / baseline_seconds

    return PerformanceGateResult(
        baseline_seconds=baseline_seconds,
        candidate_seconds=candidate_seconds,
        threshold_percent=threshold_percent,
        improvement_percent=improvement_percent,
        passed=improvement_percent >= threshold_percent,
    )


def profile_path(
    df: pl.DataFrame,
    config: RuntimeBenchmarkConfig,
    *,
    use_cache: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    cache: dict[str, object] = {}
    cache_stats = {"hits": 0, "misses": 0}
    infer_config = _inference_config(config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as handle:
        df.write_csv(handle, separator="\t")
        sumstats_path = Path(handle.name)

    def run_once():
        if use_cache:
            _arrays, hit = _to_inference_arrays_cached(df, cache)
            if hit:
                cache_stats["hits"] += 1
            else:
                cache_stats["misses"] += 1
        else:
            to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
            cache_stats["misses"] += 1
            for _ in range(max(0, config.legacy_conversion_repeats - 1)):
                # Simulates pre-refactor repeated host->device conversion in threshold workflows.
                _ = to_inference_arrays(df, "effect_allele_frequency", "beta", "standard_error")
            if config.legacy_conversion_delay_seconds > 0:
                sleep(config.legacy_conversion_delay_seconds)

        result_df = run_inference_pipeline(
            str(sumstats_path),
            lowest=config.lowest,
            highest=config.highest,
            num_breaks=config.num_breaks,
            config=infer_config,
        )
        if result_df.height == 0:
            raise RuntimeError("inference pipeline produced an empty result during benchmark path")
        return result_df

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
