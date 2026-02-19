from dataclasses import FrozenInstanceError

import pytest

from mut_var import RESULTS, Solution


def test_contracts_exported_from_package_root():
    expected = {
        "successful",
        "invalid_input",
        "empty_subset",
        "nonfinite_objective",
        "max_steps_reached",
    }
    assert expected == {status.value for status in RESULTS}


def test_solution_preserves_status_and_diagnostics():
    stats = {"n_steps": 12, "objective": -1.2}
    state = {"pi0": 0.9}
    solution = Solution(
        value=1.23,
        result=RESULTS.invalid_input,
        stats=stats,
        state=state,
    )

    assert solution.value == 1.23
    assert solution.result == RESULTS.invalid_input
    assert solution.stats == stats
    assert solution.state == state
    assert not solution.ok


def test_solution_is_immutable_and_ok_on_success():
    solution = Solution(value={"k": 2}, result=RESULTS.successful, stats={}, state=None)

    assert solution.ok
    with pytest.raises(FrozenInstanceError):
        solution.result = RESULTS.max_steps_reached
