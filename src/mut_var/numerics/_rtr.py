from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jax import Array

from mut_var.numerics._optimistix_solver import default_verbose

# =============================================================================
# Geometry: simplex <-> positive orthant of sphere(radius=2)
# =============================================================================

_RADIUS = 2.0


def _normalize_simplex(x: Array, eps: float = 1e-12) -> Array:
    x = jnp.asarray(x)
    x = jnp.maximum(x, eps)
    return x / jnp.sum(x)


def _simplex_to_sphere(x: Array) -> Array:
    x = _normalize_simplex(x)
    return _RADIUS * jnp.sqrt(x)


def _sphere_to_simplex(y: Array, eps: float = 1e-12) -> Array:
    x = (jnp.asarray(y) ** 2) / (_RADIUS**2)
    x = jnp.maximum(x, eps)
    return x / jnp.sum(x)


def _project_tangent_sphere(y: Array, v: Array) -> Array:
    return v - (jnp.dot(y, v) / (_RADIUS**2)) * y


def _sphere_expmap(y: Array, eta: Array, eps: float = 1e-15) -> Array:
    eta = _project_tangent_sphere(y, eta)
    norm_eta = jnp.linalg.norm(eta)
    theta = norm_eta / _RADIUS
    cos_theta = jnp.cos(theta)
    sin_over_norm = jnp.where(norm_eta > eps, jnp.sin(theta) / norm_eta, 1.0 / _RADIUS)
    y_new = cos_theta * y + (_RADIUS * sin_over_norm) * eta
    return _RADIUS * y_new / jnp.linalg.norm(y_new)


# =============================================================================
# Objectives
# =============================================================================


def _mixture_loglik_from_L(x: Array, L: Array, eps: float = 1e-32) -> Array:
    mix = L @ x
    mix = jnp.maximum(mix, eps)
    return jnp.sum(jnp.log(mix))


def _dirichlet_logterm(x: Array, alpha: Array, eps: float = 1e-32) -> Array:
    x = jnp.maximum(x, eps)
    return jnp.sum((alpha - 1.0) * jnp.log(x))


def _adjacent_pair_violations(x_star: Array, x_fixed: Array) -> Array:
    return x_star[1:] * x_fixed[:-1] - x_star[:-1] * x_fixed[1:]


def _ordered_softplus_penalty(x_star: Array, x_fixed: Array, tau: float) -> Array:
    c = _adjacent_pair_violations(x_star, x_fixed)
    pair_pen = jnp.sum(jax.nn.softplus(tau * c) / tau)
    null_pen = jax.nn.softplus(tau * (x_fixed[0] - x_star[0])) / tau
    return pair_pen + null_pen


class _BaselineObjective(eqx.Module):
    L: Array
    alpha: Array
    min_x: float = eqx.field(static=True, default=1e-12)

    def __call__(self, y: Array) -> Array:
        x = _sphere_to_simplex(y, eps=self.min_x)
        return _mixture_loglik_from_L(x, self.L) + _dirichlet_logterm(x, self.alpha)


class _OrderedObjective(eqx.Module):
    L: Array
    alpha: Array
    x_fixed: Array
    eta_penalty: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    min_x: float = eqx.field(static=True, default=1e-12)

    def __call__(self, y: Array) -> Array:
        x_star = _sphere_to_simplex(y, eps=self.min_x)
        x_fixed = _normalize_simplex(self.x_fixed, eps=self.min_x)
        return (
            _mixture_loglik_from_L(x_star, self.L)
            + _dirichlet_logterm(x_star, self.alpha)
            - self.eta_penalty * _ordered_softplus_penalty(x_star, x_fixed, self.tau)
        )


# =============================================================================
# Manifold gradient / HVP on the sphere
# =============================================================================


def _make_rgrad_and_rhvp(objective: eqx.Module):
    grad_fn = jax.grad(objective)

    def rgrad(y: Array) -> Array:
        ge = grad_fn(y)
        return _project_tangent_sphere(y, ge)

    def rhvp(y: Array, eta: Array) -> Array:
        eta = _project_tangent_sphere(y, eta)
        ge = grad_fn(y)
        _, He_eta = jax.jvp(grad_fn, (y,), (eta,))
        term1 = _project_tangent_sphere(y, He_eta)
        correction = (jnp.dot(y, ge) / (_RADIUS**2)) * eta
        return term1 - correction

    return rgrad, rhvp


# =============================================================================
# Trust-region subproblem: truncated CG for maximization
# =============================================================================


def _tau_to_boundary(s: Array, d: Array, radius: Array) -> Array:
    a = jnp.dot(d, d)
    b = 2.0 * jnp.dot(s, d)
    c = jnp.dot(s, s) - radius**2
    disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    return (-b + jnp.sqrt(disc)) / (2.0 * a)


def _truncated_cg_trust_region_max(
    g: Array,
    hvp,
    radius: Array,
    *,
    maxiter: int = 100,
    tol: float = 1e-10,
) -> tuple[Array, Array]:
    c = -g

    def B(v: Array) -> Array:
        return -hvp(v)

    def cond_fun(state):
        i, s, r, d, done, pred = state
        return (i < maxiter) & (~done)

    def body_fun(state):
        i, s, r, d, done, pred = state
        Bd = B(d)
        dBd = jnp.dot(d, Bd)
        rr = jnp.dot(r, r)

        def negative_curvature(_):
            tau = _tau_to_boundary(s, d, radius)
            s_new = s + tau * d
            pred_new = jnp.dot(g, s_new) + 0.5 * jnp.dot(s_new, hvp(s_new))
            return i + 1, s_new, r, d, jnp.array(True), pred_new

        def regular_step(_):
            alpha_cg = rr / dBd
            s_next = s + alpha_cg * d

            def hit_boundary(__):
                tau = _tau_to_boundary(s, d, radius)
                s_new = s + tau * d
                pred_new = jnp.dot(g, s_new) + 0.5 * jnp.dot(s_new, hvp(s_new))
                return i + 1, s_new, r, d, jnp.array(True), pred_new

            def interior(__):
                r_next = r + alpha_cg * Bd

                def converged(___):
                    pred_new = jnp.dot(g, s_next) + 0.5 * jnp.dot(s_next, hvp(s_next))
                    return i + 1, s_next, r_next, d, jnp.array(True), pred_new

                def continue_cg(___):
                    beta = jnp.dot(r_next, r_next) / rr
                    d_next = -r_next + beta * d
                    return i + 1, s_next, r_next, d_next, jnp.array(False), pred

                return jax.lax.cond(
                    jnp.linalg.norm(r_next) < tol,
                    converged,
                    continue_cg,
                    operand=None,
                )

            return jax.lax.cond(
                jnp.linalg.norm(s_next) >= radius,
                hit_boundary,
                interior,
                operand=None,
            )

        return jax.lax.cond(dBd <= 0.0, negative_curvature, regular_step, operand=None)

    init = (
        jnp.array(0),
        jnp.zeros_like(g),
        c,
        -c,
        jnp.linalg.norm(c) < tol,
        jnp.array(0.0),
    )
    _, s, _, _, _, pred = jax.lax.while_loop(cond_fun, body_fun, init)
    return s, pred


# =============================================================================
# RTR state / config
# =============================================================================


class RTRState(NamedTuple):
    objective: Array
    grad_norm: Array
    radius: Array
    accepted: Array


class RTRResult(NamedTuple):
    x_opt: Array
    y_opt: Array
    objective: Array
    grad_norm: Array
    iterations: Array
    converged: Array


@dataclass
class RTRConfig:
    maxiter: int = 200
    gtol: float = 1e-7
    initial_radius: float = 0.25
    max_radius: float = 2.0
    eta_accept: float = 0.1
    eta_expand: float = 0.75
    shrink_factor: float = 0.25
    expand_factor: float = 2.0
    subproblem_maxiter: int = 100
    subproblem_tol: float = 1e-10
    min_x: float = 1e-12


# =============================================================================
# RTR core
# =============================================================================


def _make_rtr_step(objective: eqx.Module, config: RTRConfig):
    rgrad_fn, rhvp_fn = _make_rgrad_and_rhvp(objective)
    # rgrad_fn is returned so _run_rtr can compute the final grad norm without
    # creating a second rgrad function.

    @eqx.filter_jit
    def step(y: Array, radius: Array):
        f = objective(y)
        g = rgrad_fn(y)
        gnorm = jnp.linalg.norm(g)

        def hvp_local(v: Array) -> Array:
            return rhvp_fn(y, v)

        eta, pred_gain = _truncated_cg_trust_region_max(
            g,
            hvp_local,
            radius,
            maxiter=config.subproblem_maxiter,
            tol=config.subproblem_tol,
        )

        def reject_bad_model(_):
            return y, radius * config.shrink_factor, jnp.array(False), f, gnorm

        def try_step(_):
            y_trial = _sphere_expmap(y, eta)
            valid = jnp.all(y_trial > 0.0)

            def reject_orthant(__):
                return y, radius * config.shrink_factor, jnp.array(False), f, gnorm

            def assess(__):
                f_trial = objective(y_trial)
                actual_gain = f_trial - f
                rho = actual_gain / pred_gain
                accepted = rho > config.eta_accept
                y_new = jnp.where(accepted, y_trial, y)

                radius_small = jnp.maximum(radius * config.shrink_factor, 1e-12)
                radius_large = jnp.minimum(radius * config.expand_factor, config.max_radius)
                hit_boundary = jnp.abs(jnp.linalg.norm(eta) - radius) < 1e-8

                radius_new = jax.lax.cond(
                    rho < 0.25,
                    lambda: radius_small,
                    lambda: jax.lax.cond(
                        (rho > config.eta_expand) & hit_boundary,
                        lambda: radius_large,
                        lambda: radius,
                    ),
                )
                return y_new, radius_new, accepted, f, gnorm

            return jax.lax.cond(valid, assess, reject_orthant, operand=None)

        y_new, radius_new, accepted, f_out, gnorm_out = jax.lax.cond(
            pred_gain <= 0.0,
            reject_bad_model,
            try_step,
            operand=None,
        )

        state = RTRState(
            objective=f_out,
            grad_norm=gnorm_out,
            radius=radius,
            accepted=accepted,
        )
        return y_new, radius_new, state

    return step, rgrad_fn


def _run_rtr(
    objective: eqx.Module,
    x0: Array,
    config: RTRConfig,
    verbose: bool | Callable[..., Any] = False,
) -> RTRResult:
    # The outer loop is Python so that `step` (which includes the inner CG
    # while_loop) compiles once via filter_jit and is reused each iteration.
    # A JAX while_loop here would force XLA to compile the outer loop together
    # with the inner one, producing a deeply nested compilation that is very slow.
    step, rgrad_fn = _make_rtr_step(objective, config)
    verbose_fn = default_verbose(verbose)

    y = _simplex_to_sphere(_normalize_simplex(x0, eps=config.min_x))
    radius = jnp.asarray(config.initial_radius)

    iterations_run = 0
    for it in range(config.maxiter):
        y_new, radius_new, state = step(y, radius)
        verbose_fn(
            iteration=("Iteration", jnp.array(it)),
            objective=("Objective", state.objective),
            grad_norm=("Grad norm", state.grad_norm),
            accepted=("Accepted", state.accepted),
            radius=("Radius", state.radius),
        )
        y, radius = y_new, radius_new
        iterations_run = it + 1
        if float(state.grad_norm) < config.gtol:
            break

    f_final = objective(y)
    gnorm_final = jnp.linalg.norm(rgrad_fn(y))

    return RTRResult(
        x_opt=_sphere_to_simplex(y, eps=config.min_x),
        y_opt=y,
        objective=f_final,
        grad_norm=gnorm_final,
        iterations=jnp.array(iterations_run),
        converged=gnorm_final < config.gtol,
    )


# =============================================================================
# Public solvers
# =============================================================================


def baseline_rtr(
    x0: Array,
    L: Array,
    alpha: Array,
    *,
    config: RTRConfig | None = None,
    verbose: bool | Callable[..., Any] = False,
) -> RTRResult:
    if config is None:
        config = RTRConfig()
    objective = _BaselineObjective(L=L, alpha=alpha, min_x=config.min_x)
    return _run_rtr(objective, x0, config, verbose=verbose)


def ordered_rtr(
    x0: Array,
    L: Array,
    alpha: Array,
    x_fixed: Array,
    *,
    eta_penalty: float,
    tau: float = 1.0,
    config: RTRConfig | None = None,
    verbose: bool | Callable[..., Any] = False,
) -> RTRResult:
    if config is None:
        config = RTRConfig()
    objective = _OrderedObjective(
        L=L,
        alpha=alpha,
        x_fixed=_normalize_simplex(x_fixed, eps=config.min_x),
        eta_penalty=eta_penalty,
        tau=tau,
        min_x=config.min_x,
    )
    return _run_rtr(objective, x0, config, verbose=verbose)


__all__ = [
    "RTRConfig",
    "RTRResult",
    "RTRState",
    "baseline_rtr",
    "ordered_rtr",
]
