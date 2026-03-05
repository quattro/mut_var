#!/usr/bin/env python3
"""Run simulator-side contract checks for zero-gate behavior.

This script keeps validation under ``sims/`` and does not depend on pytest.
It exercises the same contract checks that were previously covered in tests:
- config defaults for zero-gate controls,
- bounds validation,
- latent zero-behavior from DFE zero-mass and trait-null gates,
- zero-source provenance consistency.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIMULATOR_PATH = ROOT / "sims" / "scripts" / "simulate_dfe.py"
SCENARIO_CONFIG_PATH = ROOT / "sims" / "config" / "dfe_scenarios.json"


def _load_simulator_module():
    spec = importlib.util.spec_from_file_location("simulate_dfe_script", SIMULATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load simulator module from {SIMULATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scenario_payload() -> dict[str, object]:
    return json.loads(SCENARIO_CONFIG_PATH.read_text(encoding="utf-8"))


def _write_payload(payload: dict[str, object], directory: Path) -> Path:
    path = directory / "scenario.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _assert_raises_value_error(fn, expected_substring: str) -> None:
    try:
        fn()
    except ValueError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(f"expected '{expected_substring}' in error, got: {exc}") from exc
        return
    raise AssertionError("expected ValueError but call succeeded")


def run_checks() -> None:
    simulator = _load_simulator_module()

    with tempfile.TemporaryDirectory(prefix="zero_gate_contract_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)

        payload = _scenario_payload()
        payload["default"]["dfe"].pop("point_mass_zero", None)
        cfg = _write_payload(payload, tmp_dir)
        scenario = simulator._read_scenario(cfg, "default")
        if not np.isclose(scenario.point_mass_zero, 0.0):
            raise AssertionError("default point_mass_zero should be 0.0 when missing")

        payload = _scenario_payload()
        payload["default"]["effect"].pop("trait_null_fraction", None)
        cfg = _write_payload(payload, tmp_dir)
        scenario = simulator._read_scenario(cfg, "default")
        if not np.isclose(scenario.trait_null_fraction, 0.0):
            raise AssertionError("default trait_null_fraction should be 0.0 when missing")

        payload = _scenario_payload()
        payload["default"]["dfe"]["point_mass_zero"] = -0.01
        cfg = _write_payload(payload, tmp_dir)
        _assert_raises_value_error(lambda: simulator._read_scenario(cfg, "default"), "point_mass_zero")

        payload = _scenario_payload()
        payload["default"]["dfe"]["point_mass_zero"] = 1.0
        cfg = _write_payload(payload, tmp_dir)
        _assert_raises_value_error(lambda: simulator._read_scenario(cfg, "default"), "point_mass_zero")

        payload = _scenario_payload()
        payload["default"]["effect"]["trait_null_fraction"] = -0.01
        cfg = _write_payload(payload, tmp_dir)
        _assert_raises_value_error(lambda: simulator._read_scenario(cfg, "default"), "trait_null_fraction")

        payload = _scenario_payload()
        payload["default"]["effect"]["trait_null_fraction"] = 1.01
        cfg = _write_payload(payload, tmp_dir)
        _assert_raises_value_error(lambda: simulator._read_scenario(cfg, "default"), "trait_null_fraction")

        payload = _scenario_payload()
        payload["default"]["n_ascertained"] = 1000
        payload["default"]["dfe"]["point_mass_zero"] = 0.3
        cfg = _write_payload(payload, tmp_dir)
        scenario = simulator._read_scenario(cfg, "default")
        draws = simulator._generate(scenario, n_target=1000, seed=scenario.seed)
        zero_fraction = float(np.mean(draws["beta_s_true"] == 0.0))
        if not (0.1 < zero_fraction < 0.5):
            raise AssertionError(f"point-mass zero fraction out of expected range: {zero_fraction}")

        payload = _scenario_payload()
        payload["default"]["n_ascertained"] = 500
        payload["default"]["dfe"]["point_mass_zero"] = 0.0
        payload["default"]["effect"]["trait_null_fraction"] = 1.0
        cfg = _write_payload(payload, tmp_dir)
        scenario = simulator._read_scenario(cfg, "default")
        draws = simulator._generate(scenario, n_target=500, seed=scenario.seed)
        if not np.all(draws["beta_s_true"] == 0.0):
            raise AssertionError("trait_null_fraction=1.0 should force all beta_s_true to zero")
        if not np.all(draws["beta_zero_from_trait_null"]):
            raise AssertionError("trait-null source flag should be true for all draws when fraction=1.0")
        if np.any(draws["beta_zero_from_dfe_point_mass"]):
            raise AssertionError("DFE point-mass source flag should be false when point_mass_zero=0.0")

        payload = _scenario_payload()
        payload["default"]["n_ascertained"] = 1500
        payload["default"]["dfe"]["point_mass_zero"] = 0.2
        payload["default"]["effect"]["trait_null_fraction"] = 0.3
        cfg = _write_payload(payload, tmp_dir)
        scenario = simulator._read_scenario(cfg, "default")
        draws = simulator._generate(scenario, n_target=1500, seed=scenario.seed)

        from_dfe = draws["beta_zero_from_dfe_point_mass"]
        from_trait = draws["beta_zero_from_trait_null"]
        any_zero = draws["beta_zero_any"]
        beta_zero = draws["beta_s_true"] == 0.0

        if not np.all(any_zero == (from_dfe | from_trait)):
            raise AssertionError("beta_zero_any must equal logical OR of source flags")
        if np.any(from_dfe & from_trait):
            raise AssertionError("source flags should be disjoint by construction")
        if not np.all(beta_zero == any_zero):
            raise AssertionError("beta_s_true==0 should match beta_zero_any")

    print("zero-gate simulator contract checks: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulate_dfe zero-gate contracts (sims-only).")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    run_checks()


if __name__ == "__main__":
    main()
