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
    project_tangent_sphere,
    simplex_to_sphere,
    sphere_expmap,
    sphere_to_simplex,
)

# =============================================================================
# Riemannian gradient + lazy Hessian operator (sphere chart)
# =============================================================================


def _riemannian_grad_hessian(
    sphere_obj: Callable[[Array], Array],
    y_sphere: Array,
    *,
    autodiff_mode: str = "bwd",
) -> tuple[Array, lx.FunctionLinearOperator]:
    r"""Return $(\nabla_R f, \mathrm{Hess}_R f)$ at ``y_sphere`` as an HVP operator.

    Mirrors ``optimistix.second_order._grad_hessian`` but emits tangent-projected
    outputs for the unit-sphere chart. The forward-over-reverse AD linearisation
    is shared across every HVP in the inner TruncatedCG solve, so the Euclidean
    Hessian is never materialised.

    **Arguments:**

    - `sphere_obj`: scalar-valued minimisation objective on the sphere chart.
    - `y_sphere`: base point on the positive-orthant unit sphere.
    - `autodiff_mode`: ``"bwd"`` (reverse-mode gradient, default) or ``"fwd"``.

    **Returns:**

    A tuple ``(rgrad, hessian)`` where ``rgrad`` is the Riemannian gradient and
    ``hessian`` is a :class:`lineax.FunctionLinearOperator` implementing the
    Riemannian Hessian-vector product.
    """
    if autodiff_mode == "bwd":
        grad_fn = jax.grad(sphere_obj)
    elif autodiff_mode == "fwd":
        grad_fn = jax.jacfwd(sphere_obj)
    else:
        raise ValueError(f"Unknown `autodiff_mode`={autodiff_mode!r}; expected 'fwd' or 'bwd'.")

    ge, hvp_euclid = jax.linearize(grad_fn, y_sphere)
    rgrad = project_tangent_sphere(y_sphere, ge)
    y_ge_dot = jnp.dot(y_sphere, ge)

    def rhvp(v: Array) -> Array:
        # Projection sandwich + curvature correction give the Riemannian
        # Hessian on an embedded sphere: P(H_E v) - <y, g_E> v.
        v_proj = project_tangent_sphere(y_sphere, v)
        term1 = project_tangent_sphere(y_sphere, hvp_euclid(v_proj))
        return term1 - y_ge_dot * v_proj

    hessian = lx.FunctionLinearOperator(
        rhvp,
        jax.eval_shape(lambda: y_sphere),
        frozenset({lx.symmetric_tag}),
    )
    return rgrad, hessian


# =============================================================================
# RTR Optimistix solver
# =============================================================================


class _RTRSolverState(eqx.Module):
    radius: Array
    gnorm: Array


class RiemannianTrustRegion(optx.AbstractMinimiser):
    r"""Riemannian trust-region minimiser for simplex-constrained objectives.

    Parameterises $\pi \in \Delta^{k-1}$ through $y = \sqrt{\pi}$ on the
    positive-orthant unit sphere. Each outer iteration builds the Riemannian
    gradient and a lazy Hessian-vector-product operator via JAX automatic
    differentiation, approximately solves the trust-region subproblem with
    :class:`~mut_var.numerics._solver._truncated_cg.TruncatedCG`, retracts along the
    sphere geodesic, and accepts/rejects based on the actual vs. predicted
    objective reduction.

    Conceptually this matches :class:`optimistix.TrustNewton` with
    :class:`optimistix.SteihaugCGDescent`: the Hessian is exposed as a
    :class:`lineax.FunctionLinearOperator` so no dense matrix is ever
    materialised, HVPs share a single forward-over-reverse AD linearisation,
    and forward- or reverse-mode gradients are both supported.

    **Arguments:**

    - `gtol`: Riemannian gradient-norm tolerance for convergence.
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
    - `rtol`, `atol`, `norm`: Required by the Optimistix interface; RTR uses
      gradient-norm termination instead of Cauchy conditions.
    - `verbose`: If `True` or a callable, emit per-step diagnostics.

    Supports the following ``options`` key:

    - ``autodiff_mode``: ``"fwd"`` or ``"bwd"`` (default). Controls whether the
      gradient is computed via forward- or reverse-mode autodifferentiation;
      HVPs always use forward-over-{fwd|bwd} AD off the linearised gradient.
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
    subproblem_rtol: float = eqx.field(static=True)
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
        subproblem_rtol: float = 0.5,
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
        self.subproblem_rtol = subproblem_rtol
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
        del tags
        autodiff_mode = options.get("autodiff_mode", "bwd")
        min_x = self.min_x
        y_sphere = simplex_to_sphere(y, eps=min_x)

        # Minimisation objective on the sphere chart.
        def sphere_obj(y_s: Array) -> Array:
            pi = sphere_to_simplex(y_s, eps=min_x)
            f_min, _ = fn(pi, args)
            return f_min

        f = sphere_obj(y_sphere)
        rgrad, hessian = _riemannian_grad_hessian(sphere_obj, y_sphere, autodiff_mode=autodiff_mode)
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

        def reject_bad_model(_: None) -> tuple[Array, Array]:
            return y, jnp.maximum(state.radius * self.shrink_factor, min_radius)

        def try_step(_: None) -> tuple[Array, Array]:
            y_trial_sphere = sphere_expmap(y_sphere, eta)
            valid = jnp.all(y_trial_sphere > 0.0)

            def reject_orthant(_: None) -> tuple[Array, Array]:
                return y, jnp.maximum(state.radius * self.shrink_factor, min_radius)

            def assess(_: None) -> tuple[Array, Array]:
                f_trial = sphere_obj(y_trial_sphere)
                actual_reduction = f - f_trial
                rho = actual_reduction / pred_reduction
                accepted = rho > self.eta_accept
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
                return y_new, radius_new

            return jax.lax.cond(valid, assess, reject_orthant, operand=None)

        y_new, radius_new = jax.lax.cond(pred_reduction <= 0.0, reject_bad_model, try_step, operand=None)

        self.verbose(
            objective=("Objective", f),
            grad_norm=("Grad norm", gnorm),
            radius=("Radius", state.radius),
        )

        new_state = _RTRSolverState(radius=radius_new, gnorm=gnorm)
        return y_new, new_state, None

    def terminate(self, fn, y, args, options, state, tags):
        # Gradient-norm termination; the framework signals
        # nonlinear_max_steps_reached automatically once the step budget runs out.
        del fn, y, args, options, tags
        converged = state.gnorm < self.gtol
        return converged, optx.RESULTS.successful

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result, state
        return y, aux, {}


__all__ = ["RiemannianTrustRegion"]
