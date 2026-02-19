from __future__ import annotations

# pattern: Imperative Shell
from statistics import mean
from time import perf_counter
from typing import Callable

import equinox as eqx

from mut_var.contracts import RESULTS, Solution


class PerformanceGateResult(eqx.Module):
    baseline_seconds: float
    candidate_seconds: float
    improvement_percent: float
    threshold_percent: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        r"""Serialize performance-gate metrics to primitive dictionary values."""
        return {
            "baseline_seconds": self.baseline_seconds,
            "candidate_seconds": self.candidate_seconds,
            "improvement_percent": self.improvement_percent,
            "threshold_percent": self.threshold_percent,
            "passed": self.passed,
        }


def evaluate_performance_gate(
    baseline_seconds: float,
    candidate_seconds: float,
    threshold_percent: float = 20.0,
) -> PerformanceGateResult:
    r"""Compute percent-improvement gate result between baseline and candidate timings."""
    if baseline_seconds <= 0.0:
        improvement = 0.0
    else:
        improvement = ((baseline_seconds - candidate_seconds) / baseline_seconds) * 100.0
    return PerformanceGateResult(
        baseline_seconds=baseline_seconds,
        candidate_seconds=candidate_seconds,
        improvement_percent=improvement,
        threshold_percent=threshold_percent,
        passed=improvement >= threshold_percent,
    )


def profile_solution_runs(
    run_once: Callable[[], Solution],
    steady_runs: int = 3,
) -> dict[str, object]:
    r"""Profile compile and steady-state runtime of a `Solution`-returning callable."""
    if steady_runs < 1:
        raise ValueError("steady_runs must be >= 1")

    compile_start = perf_counter()
    compile_solution = run_once()
    compile_elapsed = perf_counter() - compile_start

    steady_elapsed: list[float] = []
    steady_results: list[str] = []
    for _ in range(steady_runs):
        start = perf_counter()
        solution = run_once()
        steady_elapsed.append(perf_counter() - start)
        steady_results.append(RESULTS[solution.result])

    return {
        "compile": {
            "count": 1,
            "elapsed_seconds": compile_elapsed,
            "result": RESULTS[compile_solution.result],
        },
        "steady_state": {
            "runs": steady_runs,
            "elapsed_seconds": steady_elapsed,
            "mean_seconds": mean(steady_elapsed),
            "min_seconds": min(steady_elapsed),
            "max_seconds": max(steady_elapsed),
            "results": steady_results,
        },
    }
