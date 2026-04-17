from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import optimistix as optx

from mut_var.types import RESULTS


# Pulled from optimistix for consistent solver-step diagnostics.
def default_verbose(verbose: bool | Callable[..., None]) -> Callable[..., None]:
    if callable(verbose):
        return verbose
    elif verbose is True:
        return _default_verbose
    elif verbose is False:
        return _default_no_verbose
    else:
        raise ValueError(
            f"Unrecognized `verbose` of type {type(verbose)}. Accepted types are either booleans or callables."
        )


def _default_verbose(**kwargs: tuple[str, Any]) -> None:
    string_pieces = []
    arg_pieces = []
    for name, value in kwargs.values():
        string_pieces.append(name + ": {}")
        arg_pieces.append(value)
    if len(string_pieces) > 0:
        string = ", ".join(string_pieces)
        jax.debug.print(string, *arg_pieces)


def _default_no_verbose(**kwargs):
    del kwargs


def map_optimistix_result(result: optx.RESULTS) -> RESULTS:
    r"""Map Optimistix solver statuses to mut_var contract statuses."""
    code = jnp.asarray(result._value, dtype=jnp.int32)
    code = jnp.where(code == optx.RESULTS.successful._value, 0, code)
    code = jnp.where(
        (code == optx.RESULTS.max_steps_reached._value) | (code == optx.RESULTS.nonlinear_max_steps_reached._value),
        1,
        code,
    )
    code = jnp.where((code == 0) | (code == 1), code, 2)
    return cast(
        RESULTS,
        jax.lax.switch(
            code,
            (
                lambda _: RESULTS.successful,
                lambda _: RESULTS.max_steps_reached,
                lambda _: RESULTS.nonfinite_objective,
            ),
            operand=None,
        ),
    )
