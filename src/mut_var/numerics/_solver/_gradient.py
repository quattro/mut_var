from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, cast, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree, Scalar
from optimistix import FunctionInfo
from optimistix._custom_types import Aux, Fn
from optimistix._misc import (
    filter_cond,
    lin_to_grad,
    tree_where,
)

from mut_var.numerics._solver._common import default_verbose

Y = TypeVar("Y")


class _WarmStartArmijoState(eqx.Module):
    step_size: Array


class WarmStartBacktrackingArmijo(
    optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, _WarmStartArmijoState]
):
    """Backtracking Armijo search that warm-starts from the last accepted step size."""

    decrease_factor: Scalar = 0.5
    slope: Scalar = 0.1
    step_init: Scalar = 1.0

    def init(self, y: Y, f_info_struct: optx.FunctionInfo.EvalGrad) -> _WarmStartArmijoState:
        del y, f_info_struct
        return _WarmStartArmijoState(step_size=jnp.asarray(self.step_init))

    def step(
        self,
        first_step: Array,
        y: Y,
        y_eval: Y,
        f_info: optx.FunctionInfo.EvalGrad,
        f_eval_info: optx.FunctionInfo.Eval,
        state: _WarmStartArmijoState,
    ) -> tuple[Scalar, Array, optx.RESULTS, _WarmStartArmijoState]:
        y_diff = (y_eval**ω - y**ω).ω
        predicted_reduction = f_info.compute_grad_dot(y_diff)
        f_min = f_info.as_min()
        f_min_eval = f_eval_info.as_min()
        f_min_diff = f_min_eval - f_min
        satisfies_armijo = f_min_diff <= self.slope * predicted_reduction
        has_reduction = predicted_reduction <= 0

        accept = first_step | (satisfies_armijo & has_reduction)
        next_step_size = jnp.where(accept, state.step_size, self.decrease_factor * state.step_size)
        next_step_size = cast(Scalar, next_step_size)
        return (
            next_step_size,
            accept,
            optx.RESULTS.successful,
            _WarmStartArmijoState(step_size=next_step_size),
        )


class RiemannianSteepestDescentState(eqx.Module, Generic[Y]):
    params: Y
    grad: Y
    step_index: Array


class RiemannianSteepestDescent(optx.AbstractDescent[Y, optx.FunctionInfo.EvalGrad, RiemannianSteepestDescentState[Y]]):
    r"""Retraction-parameterised steepest-descent direction for manifold-constrained problems.

    Wraps an unconstrained Euclidean gradient into a manifold-aware proposal
    using a user-supplied ``step_update`` (the retraction). The retraction
    receives the current gradient directly; it is responsible for any metric
    conversion, tangent projection, and sign convention needed to move the
    parameter *downhill* by ``step_size``.

    This is an :class:`optimistix.AbstractDescent` primitive; compose it with a
    line search and termination rule via :class:`RiemannianGradientDescent` to
    obtain a full :class:`optimistix.AbstractMinimiser`.
    """

    step_update: Callable[[Y, Y, ArrayLike], Y] = eqx.field(static=True)

    def __init__(self, *, step_update: Callable[[Y, Y, ArrayLike], Y]):
        self.step_update = step_update

    def init(
        self,
        params: Y,
        f_info_struct: optx.FunctionInfo.EvalGrad,
    ) -> RiemannianSteepestDescentState[Y]:
        r"""Initialize descent state with a zero gradient matching the parameter structure."""
        del f_info_struct
        zeros = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), params)
        return RiemannianSteepestDescentState(params=params, grad=zeros, step_index=jnp.asarray(0, dtype=jnp.int32))

    def query(
        self,
        params: Y,
        f_info: optx.FunctionInfo.EvalGrad,
        state: RiemannianSteepestDescentState[Y],
    ) -> RiemannianSteepestDescentState[Y]:
        r"""Update the stored gradient from current function information."""
        if not isinstance(f_info, optx.FunctionInfo.EvalGrad):
            raise ValueError("RiemannianSteepestDescent requires gradient information")
        step_index = state.step_index + jnp.asarray(1, dtype=jnp.int32)
        return RiemannianSteepestDescentState(params=params, grad=f_info.grad, step_index=step_index)

    def step(
        self,
        step_size: ArrayLike,
        state: RiemannianSteepestDescentState[Y],
    ) -> tuple[Y, optx.RESULTS]:
        r"""Apply one manifold-aware parameter proposal step."""
        next_params = self.step_update(state.params, state.grad, step_size)
        delta = jax.tree.map(lambda x_new, x_old: x_new - x_old, next_params, state.params)
        return delta, optx.RESULTS.successful


class RiemannianGradientDescent(optx.AbstractGradientDescent[Y, Any]):
    r"""Riemannian gradient-descent minimiser: steepest descent + backtracking line search.

    Composes :class:`RiemannianSteepestDescent` with a warm-started Armijo
    backtracking search so ``optx.minimise`` can drive the solver end-to-end:
    compute the Euclidean gradient via ``jax.linearize``, pick a step size by
    backtracking on the Armijo condition, retract onto the manifold through the
    user-supplied ``step_update``, and terminate when an accepted manifold step
    is tiny and the corresponding objective improvement is also tiny.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree[Array]], Array]
    step_norm: Callable[[Y, Y, ArrayLike], Array]
    descent: RiemannianSteepestDescent[Y]
    search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any]
    verbose: Callable[..., None]

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
        step_norm: Callable[[Y, Y, ArrayLike], Array],
        step_size: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Array] = optx.max_norm,
        search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any] | None = None,
        verbose: bool | Callable[..., None] = False,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.step_norm = step_norm
        self.descent = RiemannianSteepestDescent(step_update=step_update)
        self.search = search if search is not None else WarmStartBacktrackingArmijo(step_init=step_size)
        self.verbose = default_verbose(verbose)

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
    ) -> tuple[Y, Any, Aux]:
        del tags
        autodiff_mode = options.get("autodiff_mode", "bwd")
        f_eval, lin_fn, aux_eval = jax.linearize(lambda _y: fn(_y, args), state.y_eval, has_aux=True)
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            FunctionInfo.Eval(f_eval),
            state.search_state,
        )

        def accepted(descent_state):
            grad = lin_to_grad(lin_fn, state.y_eval, autodiff_mode, f_eval.dtype)

            f_eval_info = FunctionInfo.EvalGrad(f_eval, grad)
            descent_state = self.descent.query(state.y_eval, f_eval_info, descent_state)
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            accepted_step_norm = self.step_norm(y, grad, step_size)
            step_converged = accepted_step_norm <= jnp.asarray(self.atol + self.rtol, dtype=accepted_step_norm.dtype)
            objective_scale = self.atol + self.rtol * jnp.maximum(
                jnp.asarray(1.0, dtype=f_eval.dtype),
                jnp.abs(f_eval),
            )
            objective_converged = jnp.abs(f_diff) <= objective_scale
            terminate = step_converged & objective_converged
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)
            return state.y_eval, f_eval_info, aux_eval, descent_state, terminate

        def rejected(descent_state):
            return y, state.f_info, state.aux, descent_state, jnp.array(False)

        y, f_info, aux, descent_state, terminate = filter_cond(accept, accepted, rejected, state.descent_state)

        self.verbose(
            num_steps=("Step", descent_state.step_index),
            accepted=("Accepted", accept),
            loss_this_step=("Loss on this step", f_eval),
            loss_last_accepted_step=("Loss on the last accepted step", state.f_info.f),
            step_size=("Step size", step_size),
        )

        y_descent, descent_result = self.descent.step(step_size, descent_state)
        y_eval = (y**ω + y_descent**ω).ω
        result = optx.RESULTS.where(search_result == optx.RESULTS.successful, descent_result, search_result)

        prev_aux = tree_where(state.first_step, aux, state.aux)
        state = state.__class__(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=f_info,
            aux=aux,
            descent_state=descent_state,
            terminate=terminate,
            result=result,
        )
        return y, state, prev_aux


__all__ = [
    "RiemannianGradientDescent",
    "RiemannianSteepestDescent",
    "RiemannianSteepestDescentState",
    "WarmStartBacktrackingArmijo",
]
