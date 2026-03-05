from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
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


def _write_config(tmp_path: Path, scenario_payload: dict[str, object]) -> Path:
    config_path = tmp_path / "scenario.json"
    config_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
    return config_path


def _scenario_payload() -> dict[str, object]:
    return json.loads(SCENARIO_CONFIG_PATH.read_text(encoding="utf-8"))


def test_point_mass_zero_defaults_to_zero_when_not_provided(tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["dfe"].pop("point_mass_zero", None)
    cfg = _write_config(tmp_path, payload)

    scenario = module._read_scenario(cfg, "default")

    assert scenario.point_mass_zero == pytest.approx(0.0)


def test_trait_null_fraction_defaults_to_zero_when_not_provided(tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["effect"].pop("trait_null_fraction", None)
    cfg = _write_config(tmp_path, payload)

    scenario = module._read_scenario(cfg, "default")

    assert scenario.trait_null_fraction == pytest.approx(0.0)


@pytest.mark.parametrize("point_mass_zero", [-0.01, 1.0])
def test_point_mass_zero_bounds_validation(point_mass_zero: float, tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["dfe"]["point_mass_zero"] = point_mass_zero
    cfg = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="point_mass_zero"):
        module._read_scenario(cfg, "default")


def test_nonzero_point_mass_injects_latent_zero_effects(tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["n_ascertained"] = 1000
    payload["default"]["dfe"]["point_mass_zero"] = 0.3
    cfg = _write_config(tmp_path, payload)

    scenario = module._read_scenario(cfg, "default")
    draws = module._generate(scenario, n_target=1000, seed=scenario.seed)
    zero_fraction = float(np.mean(draws["beta_s_true"] == 0.0))

    assert zero_fraction > 0.1
    assert zero_fraction < 0.5


@pytest.mark.parametrize("trait_null_fraction", [-0.01, 1.01])
def test_trait_null_fraction_bounds_validation(trait_null_fraction: float, tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["effect"]["trait_null_fraction"] = trait_null_fraction
    cfg = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="trait_null_fraction"):
        module._read_scenario(cfg, "default")


def test_trait_null_fraction_allows_one_and_zeros_all_effects(tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["n_ascertained"] = 500
    payload["default"]["dfe"]["point_mass_zero"] = 0.0
    payload["default"]["effect"]["trait_null_fraction"] = 1.0
    cfg = _write_config(tmp_path, payload)

    scenario = module._read_scenario(cfg, "default")
    draws = module._generate(scenario, n_target=500, seed=scenario.seed)

    assert np.all(draws["beta_s_true"] == 0.0)
    assert np.all(draws["beta_zero_from_trait_null"])
    assert not np.any(draws["beta_zero_from_dfe_point_mass"])


def test_combined_zero_sources_are_tracked_and_consistent(tmp_path: Path):
    module = _load_simulator_module()
    payload = _scenario_payload()
    payload["default"]["n_ascertained"] = 1500
    payload["default"]["dfe"]["point_mass_zero"] = 0.2
    payload["default"]["effect"]["trait_null_fraction"] = 0.3
    cfg = _write_config(tmp_path, payload)

    scenario = module._read_scenario(cfg, "default")
    draws = module._generate(scenario, n_target=1500, seed=scenario.seed)

    from_dfe = draws["beta_zero_from_dfe_point_mass"]
    from_trait = draws["beta_zero_from_trait_null"]
    any_zero = draws["beta_zero_any"]
    beta_zero = draws["beta_s_true"] == 0.0

    assert np.all(any_zero == (from_dfe | from_trait))
    assert not np.any(from_dfe & from_trait)
    assert np.all(beta_zero == any_zero)
