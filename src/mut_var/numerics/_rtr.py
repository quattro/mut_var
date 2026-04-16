from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from jax import Array

from mut_var.numerics._optimistix_solver import default_verbose
from mut_var.numerics._solver_utils import (
    project_tangent_sphere,
    simplex_to_sphere,
    sphere_expmap,
    sphere_to_simplex,
)

# =============================================================================
# Trust-region subproblem: truncated CG for maximization (scan-based)
# =============================================================================


def _find_boundary_tau(s: Array, d: Array, r: Array, dBd: Array, radius: Array) -> Array:
    r"""Pick the trust-region boundary intersection minimising the local model.

    Roots of $\|s + \tau d\| = $ radius are $\tau_\pm$. The model difference
    $m(\tau) - m(0) = \tau\, d\!\cdot\! r + \tfrac{1}{2}\tau^2\, d^\top B d$
    is evaluated at both roots and the smaller is returned. Avoids relying on
    the CG conjugacy identity $d\!\cdot\! r = -\|r\|^2$, which can drift after
    several iterations or after a residual restart.
    """
    a = jnp.dot(d, d)
    sd = jnp.dot(s, d)
    c0 = jnp.dot(s, s) - radius**2
    disc = jnp.maximum(sd * sd - a * c0, 0.0)
    sqrt_disc = jnp.sqrt(disc)
    tau_plus = (-sd + sqrt_disc) / a
    tau_minus = (-sd - sqrt_disc) / a
    dr = jnp.dot(d, r)
    m_plus = tau_plus * dr + 0.5 * tau_plus**2 * dBd
    m_minus = tau_minus * dr + 0.5 * tau_minus**2 * dBd
    return jnp.where(m_plus <= m_minus, tau_plus, tau_minus)


def _truncated_cg_trust_region_max(
    g: Array,
    hvp: Callable[[Array], Array],
    radius: Array,
    *,
    maxiter: int = 25,
    tol: float = 1e-10,
    restart_every: int = 10,
) -> tuple[Array, Array]:
    """Truncated CG for the trust-region subproblem (maximization).

    Finds eta maximizing g·eta + ½ eta·H·eta subject to ‖eta‖ ≤ radius.
    Uses jax.lax.scan (fixed iterations) to avoid a nested while_loop inside
    Optimistix's outer while_loop.  The `done` flag short-circuits updates via
    jnp.where once a boundary or convergence condition is hit.

    Refreshes the residual r ← B·s − b from scratch every `restart_every`
    iterations to combat floating-point drift in the r ← r + α B·d recurrence.
    """
    b = g  # CG minimises ½sᵀBs − bᵀs with B = −H (≡ maximises g·s + ½sᵀHs)

    def B(v: Array) -> Array:
        return -hvp(v)

    def step_fn(carry: tuple, i: Array) -> tuple[tuple, None]:
        s, r, d, done, pred = carry

        # Residual restart: refresh r from scratch every `restart_every` steps
        # while still live.  lax.cond evaluates only the chosen branch, so the
        # extra B(s) call is paid only on restart iterations.
        do_restart = (i > 0) & (i % restart_every == 0) & jnp.invert(done)
        r = jax.lax.cond(do_restart, lambda _: B(s) - b, lambda _: r, operand=None)

        Bd = B(d)
        dBd = jnp.dot(d, Bd)
        rr = jnp.dot(r, r)

        def negative_curvature(_: None) -> tuple:
            tau = _find_boundary_tau(s, d, r, dBd, radius)
            s_new = s + tau * d
            pred_new = jnp.dot(g, s_new) + 0.5 * jnp.dot(s_new, hvp(s_new))
            return s_new, r, d, jnp.array(True), pred_new

        def regular_step(_: None) -> tuple:
            alpha_cg = rr / dBd
            s_next = s + alpha_cg * d

            def hit_boundary(__: None) -> tuple:
                tau = _find_boundary_tau(s, d, r, dBd, radius)
                s_new = s + tau * d
                pred_new = jnp.dot(g, s_new) + 0.5 * jnp.dot(s_new, hvp(s_new))
                return s_new, r, d, jnp.array(True), pred_new

            def interior(__: None) -> tuple:
                r_next = r + alpha_cg * Bd

                def converged(___: None) -> tuple:
                    pred_new = jnp.dot(g, s_next) + 0.5 * jnp.dot(s_next, hvp(s_next))
                    return s_next, r_next, d, jnp.array(True), pred_new

                def continue_cg(___: None) -> tuple:
                    beta = jnp.dot(r_next, r_next) / rr
                    d_next = -r_next + beta * d
                    return s_next, r_next, d_next, jnp.array(False), pred

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

        s_new, r_new, d_new, done_new, pred_new = jax.lax.cond(
            dBd <= 0.0, negative_curvature, regular_step, operand=None
        )

        # Short-circuit: freeze state once done (boundary hit or converged).
        s_out = jnp.where(done, s, s_new)
        r_out = jnp.where(done, r, r_new)
        d_out = jnp.where(done, d, d_new)
        done_out = done | done_new
        pred_out = jnp.where(done, pred, pred_new)

        return (s_out, r_out, d_out, done_out, pred_out), None

    init_done = jnp.linalg.norm(b) < tol
    init = (jnp.zeros_like(g), -b, b, init_done, jnp.array(0.0))
    (s, _, _, _, pred), _ = jax.lax.scan(step_fn, init, jnp.arange(maxiter))
    return s, pred


# =============================================================================
# RTR Optimistix solver
# =============================================================================


class _RTRSolverState(eqx.Module):
    radius: Array
    gnorm: Array


class RiemannianTrustRegion(optx.AbstractMinimiser):
    r"""Riemannian trust-region minimiser for simplex-constrained objectives.

    Maintains the positive-orthant sphere parameterisation internally and
    exposes the simplex iterate `pi` as the Optimistix `y`.  One `step` call
    runs one outer RTR iteration (objective evaluation + Riemannian gradient +
    Hessian-vector products via JVP + truncated-CG subproblem + accept/reject).

    **Arguments:**

    - `gtol`: Gradient-norm tolerance for convergence.
    - `initial_radius`: Starting trust-region radius.
    - `max_radius`: Maximum trust-region radius.
    - `eta_accept`: Minimum rho to accept a step.
    - `eta_expand`: Minimum rho to expand the radius.
    - `shrink_factor`: Radius shrink multiplier on rejection.
    - `expand_factor`: Radius expand multiplier on good steps.
    - `subproblem_maxiter`: Fixed CG iterations (scan length).
    - `subproblem_tol`: CG residual tolerance (short-circuits via flag).
    - `min_x`: Floor applied when mapping sphere ↔ simplex.
    - `rtol`, `atol`, `norm`: Required by the Optimistix interface; RTR uses
      gradient-norm termination instead of Cauchy conditions.
    - `verbose`: If `True` or a callable, emit per-step diagnostics.
    """

    rtol: float
    atol: float
    norm: Callable
    gtol: float = eqx.field(static=True)
    initial_radius: float = eqx.field(static=True)
    max_radius: float = eqx.field(static=True)
    eta_accept: float = eqx.field(static=True)
    eta_expand: float = eqx.field(static=True)
    shrink_factor: float = eqx.field(static=True)
    expand_factor: float = eqx.field(static=True)
    subproblem_maxiter: int = eqx.field(static=True)
    subproblem_tol: float = eqx.field(static=True)
    min_x: float = eqx.field(static=True)
    verbose: Callable[..., None]

    def __init__(
        self,
        *,
        gtol: float = 1e-7,
        initial_radius: float = 0.125,
        max_radius: float = 1.0,
        eta_accept: float = 0.1,
        eta_expand: float = 0.75,
        shrink_factor: float = 0.25,
        expand_factor: float = 2.0,
        subproblem_maxiter: int = 25,
        subproblem_tol: float = 1e-10,
        min_x: float = 1e-12,
        rtol: float = 0.0,
        atol: float = 0.0,
        norm: Callable = optx.max_norm,
        verbose: bool | Callable[..., Any] = False,
    ):
        self.gtol = gtol
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta_accept = eta_accept
        self.eta_expand = eta_expand
        self.shrink_factor = shrink_factor
        self.expand_factor = expand_factor
        self.subproblem_maxiter = subproblem_maxiter
        self.subproblem_tol = subproblem_tol
        self.min_x = min_x
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.verbose = default_verbose(verbose)

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        del fn, y, args, options, f_struct, aux_struct, tags
        return _RTRSolverState(
            radius=jnp.asarray(self.initial_radius, dtype=jnp.float64),
            gnorm=jnp.asarray(jnp.inf, dtype=jnp.float64),
        )

    def step(self, fn, y, args, options, state, tags):
        del options, tags
        min_x = self.min_x
        y_sphere = simplex_to_sphere(y, eps=min_x)

        # Build sphere-space objective from the Optimistix minimisation fn.
        # fn(pi, args) -> (f_min, aux); negate to get a maximisation objective.
        def sphere_obj(y_s: Array) -> Array:
            pi = sphere_to_simplex(y_s, eps=min_x)
            f_min, _ = fn(pi, args)
            return -f_min

        # Euclidean gradient and value in one forward+backward pass.
        f, ge = jax.value_and_grad(sphere_obj)(y_sphere)
        rgrad = project_tangent_sphere(y_sphere, ge)
        gnorm = jnp.linalg.norm(rgrad)

        # Riemannian Hessian-vector product (projection + curvature correction).
        def rhvp(v: Array) -> Array:
            v_proj = project_tangent_sphere(y_sphere, v)
            _, He_v = jax.jvp(jax.grad(sphere_obj), (y_sphere,), (v_proj,))
            term1 = project_tangent_sphere(y_sphere, He_v)
            correction = jnp.dot(y_sphere, ge) * v_proj
            return term1 - correction

        eta, pred_gain = _truncated_cg_trust_region_max(
            rgrad,
            rhvp,
            state.radius,
            maxiter=self.subproblem_maxiter,
            tol=self.subproblem_tol,
        )

        min_radius = jnp.asarray(1e-12, dtype=jnp.float64)

        def reject_bad_model(_: None) -> tuple[Array, Array]:
            return y, jnp.maximum(state.radius * self.shrink_factor, min_radius)

        def try_step(_: None) -> tuple[Array, Array]:
            y_trial_sphere = sphere_expmap(y_sphere, eta)
            valid = jnp.all(y_trial_sphere > 0.0)

            def reject_orthant(_: None) -> tuple[Array, Array]:
                return y, jnp.maximum(state.radius * self.shrink_factor, min_radius)

            def assess(_: None) -> tuple[Array, Array]:
                f_trial = sphere_obj(y_trial_sphere)
                actual_gain = f_trial - f
                rho = actual_gain / pred_gain
                accepted = rho > self.eta_accept
                y_new_sphere = jnp.where(accepted, y_trial_sphere, y_sphere)
                y_new = sphere_to_simplex(y_new_sphere, eps=min_x)

                radius_small = jnp.maximum(state.radius * self.shrink_factor, min_radius)
                radius_large = jnp.minimum(state.radius * self.expand_factor, self.max_radius)
                hit_boundary = jnp.abs(jnp.linalg.norm(eta) - state.radius) < 1e-8

                radius_new = jax.lax.cond(
                    rho < 0.25,
                    lambda: radius_small,
                    lambda: jax.lax.cond(
                        (rho > self.eta_expand) & hit_boundary,
                        lambda: radius_large,
                        lambda: state.radius,
                    ),
                )
                return y_new, radius_new

            return jax.lax.cond(valid, assess, reject_orthant, operand=None)

        y_new, radius_new = jax.lax.cond(pred_gain <= 0.0, reject_bad_model, try_step, operand=None)

        self.verbose(
            objective=("Objective", -f),
            grad_norm=("Grad norm", gnorm),
            radius=("Radius", state.radius),
        )

        new_state = _RTRSolverState(radius=radius_new, gnorm=gnorm)
        return y_new, new_state, None

    def terminate(self, fn, y, args, options, state, tags):
        # Return RESULTS.successful while iterating; the framework sets
        # nonlinear_max_steps_reached automatically when the step budget runs out.
        del fn, y, args, options, tags
        converged = state.gnorm < self.gtol
        return converged, optx.RESULTS.successful

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result, state
        return y, aux, {}


__all__ = ["RiemannianTrustRegion"]
