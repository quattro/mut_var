# pattern: Imperative Shell
import importlib
import json
import subprocess
import sys

from pathlib import Path

import pytest


def test_release_gate_accepts_valid_report():
    gate = importlib.import_module("scripts.check_release_gate")
    passed, errors = gate.evaluate_release_gate_payload(
        {
            "comparison": {"improvement_percent": 25.0, "threshold_percent": 20.0, "passed": True},
        }
    )
    assert passed
    assert errors == []


@pytest.mark.parametrize(
    "comparison",
    [
        {"improvement_percent": 10.0, "threshold_percent": 20.0, "passed": True},
        {"improvement_percent": 25.0, "threshold_percent": 20.0, "passed": False},
        {"improvement_percent": "invalid", "passed": True},
        None,
    ],
)
def test_release_gate_rejects_invalid_report(comparison):
    gate = importlib.import_module("scripts.check_release_gate")
    passed, errors = gate.evaluate_release_gate_payload({"comparison": comparison})
    assert not passed
    assert errors


@pytest.mark.parametrize("field", ["improvement_percent", "threshold_percent"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_release_gate_rejects_nonfinite_metrics(field, value):
    gate = importlib.import_module("scripts.check_release_gate")
    comparison = {"improvement_percent": 25.0, "threshold_percent": 20.0, "passed": True}
    comparison[field] = value
    passed, errors = gate.evaluate_release_gate_payload({"comparison": comparison})
    assert not passed
    assert errors


def test_release_gate_help_runs():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_release_gate.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--report" in result.stdout


@pytest.mark.parametrize("improvement, expected_code", [(25.0, 0), (10.0, 1)])
def test_release_gate_reads_report_file(tmp_path, improvement, expected_code):
    gate = importlib.import_module("scripts.check_release_gate")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "comparison": {"improvement_percent": improvement, "threshold_percent": 20.0, "passed": True},
            }
        )
    )
    assert gate.main(["--report", str(report)]) == expected_code
