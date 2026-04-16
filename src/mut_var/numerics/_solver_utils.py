from __future__ import annotations

# pattern: Functional Core
from typing import cast

import jax.numpy as jnp

from jaxtyping import Array, ArrayLike

from mut_var.types import RESULTS

RECOVERABLE_RESULTS = (RESULTS.successful, RESULTS.max_steps_reached)


def is_recoverable_result(result: RESULTS) -> bool:
    r"""Return whether a status is recoverable for staged numerics pipelines."""
    return result in RECOVERABLE_RESULTS


def merge_recoverable_results(*results: RESULTS) -> RESULTS:
    r"""Merge recoverable statuses, preferring `max_steps_reached` when present."""
    if any(result == RESULTS.max_steps_reached for result in results):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.successful)


def is_nonfinite(value: ArrayLike) -> bool:
    r"""Return `True` when any value is non-finite."""
    return not bool(jnp.isfinite(jnp.asarray(value)).all())


def simplex_to_sphere(pi: Array, eps: float = 1e-12) -> Array:
    r"""Map the simplex to the positive orthant of the unit sphere via $y = \sqrt{\pi}$."""
    x = jnp.maximum(jnp.asarray(pi), eps)
    x = x / jnp.sum(x)
    return jnp.sqrt(x)


def sphere_to_simplex(y: Array, eps: float = 1e-12) -> Array:
    r"""Inverse chart map: $\pi = y^2$ (renormalised)."""
    pi = jnp.asarray(y) ** 2
    pi = jnp.maximum(pi, eps)
    return pi / jnp.sum(pi)


def project_tangent_sphere(y: Array, v: Array) -> Array:
    r"""Project $v$ onto the tangent space of the unit sphere at $y$."""
    return v - jnp.dot(y, v) * y


def sphere_expmap(y: Array, eta: Array, eps: float = 1e-15) -> Array:
    r"""Exponential map on the unit sphere: geodesic from $y$ in direction $\eta$."""
    eta = project_tangent_sphere(y, eta)
    norm_eta = jnp.linalg.norm(eta)
    cos_theta = jnp.cos(norm_eta)
    sin_over_norm = jnp.where(norm_eta > eps, jnp.sin(norm_eta) / norm_eta, 1.0)
    y_new = cos_theta * y + sin_over_norm * eta
    return y_new / jnp.linalg.norm(y_new)


def simplex_tangent_direction(pi: Array, direction: Array) -> Array:
    r"""Project an unconstrained direction onto the simplex Fisher-Rao tangent."""
    return pi * (direction - (direction @ pi))


def exponential_map_simplex(
    pi: Array,
    tangent_direction: Array,
    step_size: float,
) -> Array:
    r"""Move simplex parameters along a Fisher-Rao tangent via the unit-sphere chart.

    The Fisher-Rao tangent vector $v$ in simplex coordinates corresponds to
    $\eta = v / (2y)$ in the sphere chart $y = \sqrt{\pi}$; the geodesic step
    of length `step_size` is taken on the sphere and pulled back to the simplex.
    """
    y = simplex_to_sphere(pi)
    eta = step_size * tangent_direction / (2.0 * y)
    return sphere_to_simplex(sphere_expmap(y, eta))
