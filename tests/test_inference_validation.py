import numpy as np
import pytest

from mut_var import run_inference_pipeline
from mut_var.io import validate_maf_grid
from mut_var.numerics import prepare_fit_state
from mut_var.types import InferenceConfig, RESULTS


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_clusters", 2.5),
        ("num_clusters", True),
        ("max_iter", 0),
        ("max_iter", 1.5),
        ("atol", np.inf),
        ("rtol", np.nan),
        ("atol", -1),
        ("filter_threshold", np.nan),
        ("filter_threshold", -0.1),
        ("filter_threshold", 1.1),
        ("constrain_spike", "false"),
    ],
)
def test_invalid_config_rejected_before_inference(field, value):
    config = InferenceConfig(3)._replace(**{field: value})
    solution = prepare_fit_state(np.array([0.1]), np.array([1.0]), config)
    assert solution.result is RESULTS.invalid_input
    assert field in solution.stats["reason"]
    with pytest.raises(ValueError, match=field):
        run_inference_pipeline("missing-input.tsv", config=config)


@pytest.mark.parametrize("lowest,highest", [(np.nan, 0.01), (1e-6, np.nan), (np.inf, 0.01)])
def test_nonfinite_grid_rejected(lowest, highest):
    with pytest.raises(ValueError, match="finite"):
        validate_maf_grid(lowest, highest, 10)


@pytest.mark.parametrize("excess", [0.0, 1e-6, 0.01])
def test_near_noise_variance_grid_is_strictly_increasing(excess):
    solution = prepare_fit_state(np.full(100, np.sqrt(1 + excess)), np.ones(100), InferenceConfig(4))
    assert solution.result is RESULTS.successful
    variances = solution.value.initial_params.var_k
    assert np.all(variances > 0)
    assert np.all(np.diff(variances) > 0)
    np.testing.assert_allclose(np.diff(np.log(variances)), np.diff(np.log(variances))[0])
