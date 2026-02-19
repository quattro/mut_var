from __future__ import annotations

# pattern: Functional Core
from typing import cast

import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from mut_var.contracts import RESULTS

RECOVERABLE_RESULTS = (RESULTS.successful, RESULTS.max_steps_reached)


def is_recoverable_result(result: RESULTS) -> bool:
    return result in RECOVERABLE_RESULTS


def merge_recoverable_results(*results: RESULTS) -> RESULTS:
    if any(result == RESULTS.max_steps_reached for result in results):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.successful)


def is_nonfinite(value: ArrayLike) -> bool:
    return not bool(jnp.isfinite(jnp.asarray(value)).all())


def simplex_tangent_direction(pi: Array, direction: Array) -> Array:
    return pi * (direction - (direction @ pi))


def exponential_map_simplex(
    pi: Array,
    tangent_direction: Array,
    step_size: float,
) -> Array:
    s = jnp.sqrt(jnp.sum(tangent_direction**2) / pi)
    c = jnp.cos(0.5 * step_size * s)
    s2 = jnp.sin(0.5 * step_size * s)

    phi = jnp.sqrt(pi)
    step = (tangent_direction / (s * phi)) * s2
    phi_new = phi * c + step
    pi_new = phi_new**2

    return pi_new / jnp.sum(pi_new)
