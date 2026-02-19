from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mut_var.contracts import RESULTS


REQUIRED_FAILURE_STATES = {
    RESULTS[RESULTS.empty_subset],
    RESULTS[RESULTS.nonfinite_objective],
}


def evaluate_release_gate_payload(payload: dict[str, object]) -> tuple[bool, list[str]]:
    errors: list[str] = []

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return False, ["Report must contain a 'comparison' object."]

    improvement = comparison.get("improvement_percent")
    threshold = comparison.get("threshold_percent", 20.0)
    passed = comparison.get("passed")

    try:
        improvement_value = float(improvement)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        errors.append("Comparison metrics must include numeric improvement_percent and threshold_percent.")
        improvement_value = 0.0
        threshold_value = 20.0

    if improvement_value < threshold_value:
        errors.append(
            f"Steady-state improvement {improvement_value:.3f}% is below required {threshold_value:.3f}%."
        )

    if passed is not True:
        errors.append("Benchmark report marks comparison.passed as false.")

    available_states = set(RESULTS._index_to_message)
    missing_states = REQUIRED_FAILURE_STATES.difference(available_states)
    if missing_states:
        errors.append(
            "Failure status catalog is missing required states: "
            + ", ".join(sorted(missing_states))
        )

    return (len(errors) == 0), errors


def evaluate_release_gate(report_path: Path) -> tuple[bool, list[str], dict[str, object] | None]:
    if not report_path.exists():
        return False, [f"Benchmark report artifact not found: {report_path}"], None

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Benchmark report is not valid JSON: {exc}"], None

    if not isinstance(payload, dict):
        return False, ["Benchmark report root must be an object."], None

    passed, errors = evaluate_release_gate_payload(payload)
    return passed, errors, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    passed, errors, payload = evaluate_release_gate(args.report)

    print("Release Gate Criteria")
    print(f"- Report path exists: {args.report.exists()}")

    if payload and isinstance(payload.get("comparison"), dict):
        comparison = payload["comparison"]
        print(f"- Improvement percent: {comparison.get('improvement_percent')}")
        print(f"- Threshold percent: {comparison.get('threshold_percent')}")
        print(f"- Report passed flag: {comparison.get('passed')}")
    print(
        "- Required failure states present: "
        + ", ".join(sorted(REQUIRED_FAILURE_STATES))
    )

    if passed:
        print("Release gate decision: PASS")
        return 0

    print("Release gate decision: FAIL")
    for item in errors:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
