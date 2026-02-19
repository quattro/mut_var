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
    cauchy_termination,
    filter_cond,
    lin_to_grad,
    tree_where,
)

from mut_var.contracts import RESULTS

Y = TypeVar("Y")


class MutVarDescentState(eqx.Module, Generic[Y]):
    params: Y
    direction: Y
    step_index: Array


def _ascent_direction(grad: Y) -> Y:
    return jax.tree.map(lambda leaf: -leaf, grad)


# pulled from optimistix...
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


class MutVarDescent(optx.AbstractDescent[Y, optx.FunctionInfo.EvalGrad, MutVarDescentState[Y]]):
    step_update: Callable[[Y, Y, ArrayLike], Y] = eqx.field(static=True)
    direction_transform: Callable[[Y], Y] = eqx.field(static=True)

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
        direction_transform: Callable[[Y], Y] = _ascent_direction,
    ):
        self.step_update = step_update
        self.direction_transform = direction_transform

    def init(
        self,
        params: Y,
        f_info_struct: optx.FunctionInfo.EvalGrad,
    ) -> MutVarDescentState[Y]:
        r"""Initialize descent state with zero direction matching parameter structure."""
        del f_info_struct
        zeros = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), params)
        return MutVarDescentState(params=params, direction=zeros, step_index=jnp.asarray(0, dtype=jnp.int32))

    def query(
        self,
        params: Y,
        f_info: optx.FunctionInfo.EvalGrad,
        state: MutVarDescentState[Y],
    ) -> MutVarDescentState[Y]:
        r"""Update descent direction from current gradient information."""
        if not isinstance(f_info, optx.FunctionInfo.EvalGrad):
            raise ValueError("mut_var optimistix descent requires gradient information")
        direction = self.direction_transform(f_info.grad)
        step_index = state.step_index + jnp.asarray(1, dtype=jnp.int32)
        return MutVarDescentState(params=params, direction=direction, step_index=step_index)

    def step(
        self,
        step_size: ArrayLike,
        state: MutVarDescentState[Y],
    ) -> tuple[Y, optx.RESULTS]:
        r"""Apply one manifold-aware parameter proposal step."""
        next_params = self.step_update(state.params, state.direction, step_size)
        delta = jax.tree.map(lambda x_new, x_old: x_new - x_old, next_params, state.params)
        return delta, optx.RESULTS.successful


class MutVarSolver(optx.AbstractGradientDescent[Y, Any]):
    rtol: float
    atol: float
    norm: Callable[[PyTree[Array]], Array]
    descent: MutVarDescent[Y]
    search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any]
    verbose: Callable[..., None]

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
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
        self.descent = MutVarDescent(step_update=step_update)
        self.search = search if search is not None else optx.BacktrackingArmijo(step_init=step_size)
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
            y_diff = (state.y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(self.rtol, self.atol, self.norm, state.y_eval, y_diff, f_eval, f_diff)
            terminate = jnp.where(state.first_step, jnp.array(False), terminate)  # Skip termination on first step
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
            # y=("y", state.y_eval),
            # y_last_accepted_step=("y on the last accepted step", y),
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


def map_optimistix_result(result: optx.RESULTS) -> RESULTS:
    r"""Map Optimistix solver statuses to mut_var contract statuses."""
    if result == optx.RESULTS.successful:
        return cast(RESULTS, RESULTS.successful)
    if result in (optx.RESULTS.max_steps_reached, optx.RESULTS.nonlinear_max_steps_reached):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.nonfinite_objective)
