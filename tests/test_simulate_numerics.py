from __future__ import annotations

import numpy as np

from mut_var.numerics import simulate_mixture_data
from mut_var.types import RESULTS, SimulationConfig, Solution


def _valid_config(**overrides) -> SimulationConfig:
    base = SimulationConfig(
        n_rows=256,
        seed=0,
        weights=(0.9, 0.1),
        log_var_scales=(-8.0, -5.5),
        variance_link="maf_power",
        theta=0.5,
        link_eps=1e-8,
        link_shift=1e-3,
        af_clip_min=1e-4,
        af_model="beta",
        af_beta_a=0.4,
        af_beta_b=0.4,
        se_model="af_n_scaled",
        sample_size=50000.0,
        se_scale=1.0,
    )
    values = base._asdict()
    values.update(overrides)
    return SimulationConfig(**values)


def test_simulate_mixture_data_returns_solution_and_arrays_on_valid_config():
    solution = simulate_mixture_data(config=_valid_config(n_rows=256, seed=0))

    assert isinstance(solution, Solution)
    assert solution.result == RESULTS.successful
    assert solution.value is not None
    assert solution.stats is not None
    arrays = solution.value
    assert arrays.row_id.shape == (256,)
    assert arrays.af.shape == (256,)
    assert arrays.component.shape == (256,)
    assert arrays.sigma2.shape == (256,)
    assert arrays.beta_true.shape == (256,)
    assert arrays.se.shape == (256,)
    assert arrays.beta_hat.shape == (256,)


def test_simulate_mixture_data_rejects_invalid_weight_shapes():
    config = _valid_config(weights=(1.0,), log_var_scales=(-8.0,))

    solution = simulate_mixture_data(config=config._replace(n_rows=32, seed=0))

    assert solution.result == RESULTS.invalid_input
    assert solution.stats is not None
    assert "weights" in str(solution.stats.get("reason", ""))


def test_simulate_mixture_data_rejects_invalid_theta():
    config = _valid_config(theta=-0.1)

    solution = simulate_mixture_data(config=config._replace(n_rows=32, seed=0))

    assert solution.result == RESULTS.invalid_input
    assert solution.stats is not None
    assert "theta" in str(solution.stats.get("reason", ""))


def test_simulate_mixture_data_reproducible_for_fixed_seed():
    config = _valid_config()
    first = simulate_mixture_data(config=config._replace(n_rows=512, seed=123))
    second = simulate_mixture_data(config=config._replace(n_rows=512, seed=123))

    assert first.result == RESULTS.successful
    assert second.result == RESULTS.successful
    first_arrays = first.value
    second_arrays = second.value
    assert np.array_equal(first_arrays.row_id, second_arrays.row_id)
    assert np.array_equal(first_arrays.af, second_arrays.af)
    assert np.array_equal(first_arrays.component, second_arrays.component)
    assert np.array_equal(first_arrays.sigma2, second_arrays.sigma2)
    assert np.array_equal(first_arrays.beta_true, second_arrays.beta_true)
    assert np.array_equal(first_arrays.se, second_arrays.se)
    assert np.array_equal(first_arrays.beta_hat, second_arrays.beta_hat)


def test_variance_link_outputs_positive_finite_sigma2_for_all_links():
    for link in ("none", "maf_power", "maf_power_shifted"):
        config = _valid_config(variance_link=link, link_shift=1e-3 if link == "maf_power_shifted" else 0.0)
        solution = simulate_mixture_data(config=config._replace(n_rows=20000, seed=0))

        assert solution.result == RESULTS.successful
        sigma2 = solution.value.sigma2
        assert bool(np.all(np.isfinite(sigma2)))
        assert bool(np.all(sigma2 > 0.0))


def test_component_index_respects_component_range():
    config = _valid_config(weights=(0.6, 0.3, 0.1), log_var_scales=(-8.0, -6.0, -4.0))
    solution = simulate_mixture_data(config=config._replace(n_rows=4096, seed=7))

    assert solution.result == RESULTS.successful
    component = solution.value.component
    assert bool(np.all(component >= 0))
    assert bool(np.all(component < 3))
