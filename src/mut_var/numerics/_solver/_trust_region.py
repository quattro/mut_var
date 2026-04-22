from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx

from jax import Array

from mut_var.numerics._solver._common import default_verbose
from mut_var.numerics._solver._truncated_cg import TruncatedCG
from mut_var.numerics._solver_utils import (
    simplex_to_sphere,
    sphere_expmap,
    sphere_to_simplex,
)

# =============================================================================
# Riemannian trust-region solver (sphere-chart, analytic Hessian only)
# =============================================================================

# Builder contract: ``builder(y_sphere, pi, args) -> (f, rgrad, hessian)``.
# All three are computed analytically once per outer step so the solver never
# needs to autodiff the objective — the caller supplies the problem-specific
# closed-form Riemannian gradient, objective value, and Hessian operator.
RiemannianHessianBuilder = Callable[[Array, Array, Any], tuple[Array, Array, lx.AbstractLinearOperator]]


class _RTRSolverState(eqx.Module):
    first_step: Array
    accepted_step: Array
    radius: Array
    gnorm: Array
    step_norm: Array
    pred_reduction: Array
    objective: Array
    on_boundary: Array
    f_diff: Array


def _accept_trust_region_step(
    *,
    f: Array,
    actual_reduction: Array,
    pred_reduction: Array,
    rho: Array,
    eta_accept: float,
) -> Array:
    """Accept positive-decrease steps when the model reduction is numerically tiny."""
    objective_scale = jnp.maximum(jnp.asarray(1.0, dtype=f.dtype), jnp.abs(f))
    tiny_model_reduction = pred_reduction <= (1e-10 * objective_scale)
    return (rho > eta_accept) | (tiny_model_reduction & (actual_reduction > 0.0))


def _boundary_stagnation_converged(
    *,
    radius: Array,
    pred_reduction: Array,
    objective: Array,
    on_boundary: Array,
    min_radius: float,
) -> Array:
    """Detect boundary-limited stagnation when the model cannot reduce meaningfully."""
    objective_scale = jnp.maximum(jnp.asarray(1.0, dtype=objective.dtype), jnp.abs(objective))
    tiny_model_reduction = pred_reduction <= (1e-12 * objective_scale)
    tiny_radius = radius <= jnp.asarray(min_radius, dtype=radius.dtype)
    return on_boundary & tiny_radius & tiny_model_reduction


class RiemannianTrustRegion(optx.AbstractMinimiser):
    r"""Riemannian trust-region minimiser for simplex-constrained mixture fits.

    Parameterises $\pi \in \Delta^{k-1}$ through $y = \sqrt{\pi}$ on the
    positive-orthant unit sphere. Each outer iteration delegates objective,
    gradient, and Hessian evaluation to a caller-supplied analytic
    ``riemannian_hessian_builder``, approximately solves the trust-region
    subproblem with
    :class:`~mut_var.numerics._solver._truncated_cg.TruncatedCG`, retracts
    along the sphere geodesic, and accepts/rejects via the actual/predicted
    reduction ratio.

    **Arguments:**

    - `riemannian_hessian_builder`: Callable returning
      ``(f, rgrad, hessian)`` on the sphere chart at ``y_sphere``; the Hessian
      is a :class:`lineax.AbstractLinearOperator` already in its
      projection-sandwiched Riemannian form.
    - `initial_radius`: Starting trust-region radius.
    - `max_radius`: Maximum trust-region radius.
    - `eta_accept`: Minimum rho to accept a step.
    - `eta_expand`: Minimum rho to expand the radius.
    - `shrink_factor`: Radius shrink multiplier on rejection.
    - `expand_factor`: Radius expand multiplier on good steps.
    - `subproblem_maxiter`: Hard cap on inner CG iterations per outer step.
    - `subproblem_rtol`: Upper bound for the Eisenstat-Walker forcing sequence
      used inside TruncatedCG. The per-call tolerance is
      $\min(\texttt{subproblem\_rtol}, \sqrt{\|g\|})$ so the inner solve
      tightens automatically as the outer iterate converges.
    - `min_x`: Floor applied when mapping sphere $\leftrightarrow$ simplex.
    - `rtol`, `atol`, `norm`: Convergence settings, aligned with
      :class:`RiemannianGradientDescent` through manifold-step/objective
      termination on accepted updates.
    - `verbose`: If `True` or a callable, emit per-step diagnostics.
    """

    rtol: float
    atol: float
    norm: Callable
    initial_radius: float = eqx.field(static=True)
    max_radius: float = eqx.field(static=True)
    eta_accept: float = eqx.field(static=True)
    eta_expand: float = eqx.field(static=True)
    shrink_factor: float = eqx.field(static=True)
    expand_factor: float = eqx.field(static=True)
    subproblem_maxiter: int = eqx.field(static=True)
    subproblem_rtol: float = eqx.field(static=True)
    min_x: float = eqx.field(static=True)
    verbose: Callable[..., None]
    riemannian_hessian_builder: RiemannianHessianBuilder = eqx.field(static=True)

    def __init__(
        self,
        riemannian_hessian_builder: RiemannianHessianBuilder,
        *,
        initial_radius: float = 0.125,
        max_radius: float = 1.0,
        eta_accept: float = 0.1,
        eta_expand: float = 0.75,
        shrink_factor: float = 0.25,
        expand_factor: float = 2.0,
        subproblem_maxiter: int = 25,
        subproblem_rtol: float = 0.5,
        min_x: float = 1e-12,
        rtol: float = 0.0,
        atol: float = 0.0,
        norm: Callable = optx.max_norm,
        verbose: bool | Callable[..., Any] = False,
    ):
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta_accept = eta_accept
        self.eta_expand = eta_expand
        self.shrink_factor = shrink_factor
        self.expand_factor = expand_factor
        self.subproblem_maxiter = subproblem_maxiter
        self.subproblem_rtol = subproblem_rtol
        self.min_x = min_x
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.verbose = default_verbose(verbose)
        self.riemannian_hessian_builder = riemannian_hessian_builder

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        del fn, args, options, f_struct, aux_struct, tags
        return _RTRSolverState(
            first_step=jnp.asarray(True),
            accepted_step=jnp.asarray(False),
            radius=jnp.asarray(self.initial_radius, dtype=jnp.float64),
            gnorm=jnp.asarray(jnp.inf, dtype=jnp.float64),
            step_norm=jnp.asarray(jnp.inf, dtype=jnp.float64),
            pred_reduction=jnp.asarray(jnp.inf, dtype=jnp.float64),
            objective=jnp.asarray(jnp.inf, dtype=jnp.float64),
            on_boundary=jnp.asarray(False),
            f_diff=jnp.asarray(jnp.inf, dtype=jnp.float64),
        )

    def step(self, fn, y, args, options, state, tags):
        del fn, options, tags
        min_x = self.min_x
        y_sphere = simplex_to_sphere(y, eps=min_x)
        pi = sphere_to_simplex(y_sphere, eps=min_x)

        f, rgrad, hessian = self.riemannian_hessian_builder(y_sphere, pi, args)
        gnorm = jnp.linalg.norm(rgrad)

        # Eisenstat-Walker forcing sequence so the inner solve tightens as the
        # outer iterate converges. Mirrors SteihaugCGDescent.step.
        ew_rtol = jnp.minimum(jnp.asarray(self.subproblem_rtol), jnp.sqrt(gnorm))

        # Solve  H p = -g  s.t. ||p|| <= radius. TruncatedCG handles negative
        # curvature and boundary crossings internally and returns the projected
        # step directly; no extra manifold-aware CG loop is needed.
        subproblem_out = lx.linear_solve(
            hessian,
            jnp.negative(rgrad),
            TruncatedCG(
                rtol=self.subproblem_rtol,
                atol=0.0,
                max_steps=self.subproblem_maxiter,
            ),
            options={"delta": state.radius, "rtol": ew_rtol},
            throw=False,
        )
        eta = subproblem_out.value
        on_boundary = subproblem_out.stats["hit_boundary"] | subproblem_out.stats["negative_curvature"]

        # Predicted reduction of the quadratic model:
        #   m(0) - m(eta) = -(g . eta + 1/2 eta . H eta).
        Heta = hessian.mv(eta)
        pred_reduction = -(jnp.dot(rgrad, eta) + 0.5 * jnp.dot(eta, Heta))

        min_radius = jnp.asarray(1e-12, dtype=jnp.float64)

        def reject_bad_model(_: None) -> tuple[Array, Array, Array, Array]:
            return (
                y,
                jnp.maximum(state.radius * self.shrink_factor, min_radius),
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=f.dtype),
            )

        def try_step(_: None) -> tuple[Array, Array, Array, Array]:
            y_trial_sphere = sphere_expmap(y_sphere, eta)
            # The sphere chart is sign-symmetric for simplex parameters:
            # pi = y^2 / sum(y^2). A trial geodesic may cross into another
            # orthant while still representing a valid simplex point, so do not
            # reject purely on the sign of y_trial_sphere.
            pi_trial = sphere_to_simplex(y_trial_sphere, eps=min_x)
            f_trial, _rg_trial, _H_trial = self.riemannian_hessian_builder(y_trial_sphere, pi_trial, args)
            actual_reduction = f - f_trial
            rho = actual_reduction / pred_reduction
            accepted = _accept_trust_region_step(
                f=f,
                actual_reduction=actual_reduction,
                pred_reduction=pred_reduction,
                rho=rho,
                eta_accept=self.eta_accept,
            )
            y_new_sphere = jnp.where(accepted, y_trial_sphere, y_sphere)
            y_new = sphere_to_simplex(y_new_sphere, eps=min_x)

            radius_small = jnp.maximum(state.radius * self.shrink_factor, min_radius)
            radius_large = jnp.minimum(state.radius * self.expand_factor, self.max_radius)

            radius_new = jax.lax.cond(
                rho < 0.25,
                lambda: radius_small,
                lambda: jax.lax.cond(
                    (rho > self.eta_expand) & on_boundary,
                    lambda: radius_large,
                    lambda: state.radius,
                ),
            )
            f_diff = jnp.where(accepted, jnp.abs(actual_reduction), jnp.asarray(0.0, dtype=f.dtype))
            return y_new, radius_new, accepted, f_diff

        y_new, radius_new, accepted_step, f_diff = jax.lax.cond(
            pred_reduction <= 0.0,
            reject_bad_model,
            try_step,
            operand=None,
        )

        self.verbose(
            objective=("Objective", f),
            grad_norm=("Grad norm", gnorm),
            radius=("Radius", state.radius),
            step_norm=("Step norm", jnp.linalg.norm(eta)),
            predicted_reduction=("Predicted reduction", pred_reduction),
            on_boundary=("Boundary/NC", on_boundary),
        )

        new_state = _RTRSolverState(
            first_step=jnp.asarray(False),
            accepted_step=accepted_step,
            radius=radius_new,
            gnorm=gnorm,
            step_norm=jnp.linalg.norm(eta),
            pred_reduction=pred_reduction,
            objective=f,
            on_boundary=on_boundary,
            f_diff=f_diff,
        )
        return y_new, new_state, None

    def terminate(self, fn, y, args, options, state, tags):
        del fn, y, args, options, tags
        step_converged = state.step_norm <= jnp.asarray(self.atol + self.rtol, dtype=state.step_norm.dtype)
        objective_converged = state.f_diff <= (
            self.atol
            + self.rtol
            * jnp.maximum(
                jnp.asarray(1.0, dtype=state.objective.dtype),
                jnp.abs(state.objective),
            )
        )
        step_objective_converged = step_converged & objective_converged
        step_objective_converged = jnp.where(
            state.first_step | ~state.accepted_step,
            jnp.asarray(False),
            step_objective_converged,
        )
        converged = step_objective_converged | _boundary_stagnation_converged(
            radius=state.radius,
            pred_reduction=state.pred_reduction,
            objective=state.objective,
            on_boundary=state.on_boundary,
            min_radius=1e-12,
        )
        return converged, optx.RESULTS.successful

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result, state
        return y, aux, {}


__all__ = ["RiemannianHessianBuilder", "RiemannianTrustRegion"]
