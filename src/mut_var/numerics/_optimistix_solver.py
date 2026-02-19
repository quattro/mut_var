from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, cast, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from jaxtyping import Array, ArrayLike, PyTree

from mut_var.contracts import RESULTS

Y = TypeVar("Y")


class MutVarDescentState(eqx.Module, Generic[Y]):
    params: Y
    direction: Y


def _ascent_direction(grad: Y) -> Y:
    return jax.tree.map(lambda leaf: -leaf, grad)


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
        del f_info_struct
        zeros = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), params)
        return MutVarDescentState(params=params, direction=zeros)

    def query(
        self,
        params: Y,
        f_info: optx.FunctionInfo.EvalGrad,
        state: MutVarDescentState[Y],
    ) -> MutVarDescentState[Y]:
        del state
        if not isinstance(f_info, optx.FunctionInfo.EvalGrad):
            raise ValueError("mut_var optimistix descent requires gradient information")
        direction = self.direction_transform(f_info.grad)
        return MutVarDescentState(params=params, direction=direction)

    def step(
        self,
        step_size: ArrayLike,
        state: MutVarDescentState[Y],
    ) -> tuple[Y, optx.RESULTS]:
        next_params = self.step_update(state.params, state.direction, step_size)
        delta = jax.tree.map(lambda x_new, x_old: x_new - x_old, next_params, state.params)
        return delta, optx.RESULTS.successful


class MutVarSolver(optx.AbstractGradientDescent[Y, Any]):
    rtol: float
    atol: float
    norm: Callable[[PyTree[Array]], Array]
    descent: MutVarDescent[Y]
    search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any]

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
        step_size: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Array] = optx.max_norm,
        search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any] | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = MutVarDescent(step_update=step_update)
        self.search = search if search is not None else optx.BacktrackingArmijo(step_init=step_size)


def map_optimistix_result(result: optx.RESULTS) -> RESULTS:
    if result == optx.RESULTS.successful:
        return cast(RESULTS, RESULTS.successful)
    if result in (optx.RESULTS.max_steps_reached, optx.RESULTS.nonlinear_max_steps_reached):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.nonfinite_objective)
