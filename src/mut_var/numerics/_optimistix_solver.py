from __future__ import annotations

# pattern: Functional Core
from collections.abc import Callable
from typing import Any, cast, Generic, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx

from jaxtyping import Array, ArrayLike, PyTree

from mut_var.contracts import RESULTS

Y = TypeVar("Y")
VerboseCallback: TypeAlias = Callable[[ArrayLike, ArrayLike, ArrayLike], None]


class MutVarDescentState(eqx.Module, Generic[Y]):
    params: Y
    direction: Y
    step_index: Array


def _ascent_direction(grad: Y) -> Y:
    return jax.tree.map(lambda leaf: -leaf, grad)


class MutVarDescent(optx.AbstractDescent[Y, optx.FunctionInfo.EvalGrad, MutVarDescentState[Y]]):
    step_update: Callable[[Y, Y, ArrayLike], Y] = eqx.field(static=True)
    direction_transform: Callable[[Y], Y] = eqx.field(static=True)
    verbose_callback: VerboseCallback | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
        direction_transform: Callable[[Y], Y] = _ascent_direction,
        verbose_callback: VerboseCallback | None = None,
    ):
        self.step_update = step_update
        self.direction_transform = direction_transform
        self.verbose_callback = verbose_callback

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
        if self.verbose_callback is not None:
            grad_norm_sq = jtu.tree_reduce(
                lambda acc, leaf: acc + jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float64))),
                f_info.grad,
                initializer=jnp.asarray(0.0, dtype=jnp.float64),
            )
            grad_norm = jnp.sqrt(grad_norm_sq)
            jax.debug.callback(self.verbose_callback, step_index, f_info.f, grad_norm)
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

    def __init__(
        self,
        *,
        step_update: Callable[[Y, Y, ArrayLike], Y],
        step_size: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Array] = optx.max_norm,
        search: optx.AbstractSearch[Y, optx.FunctionInfo.EvalGrad, optx.FunctionInfo.Eval, Any] | None = None,
        verbose_callback: VerboseCallback | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = MutVarDescent(step_update=step_update, verbose_callback=verbose_callback)
        self.search = search if search is not None else optx.BacktrackingArmijo(step_init=step_size)


def map_optimistix_result(result: optx.RESULTS) -> RESULTS:
    r"""Map Optimistix solver statuses to mut_var contract statuses."""
    if result == optx.RESULTS.successful:
        return cast(RESULTS, RESULTS.successful)
    if result in (optx.RESULTS.max_steps_reached, optx.RESULTS.nonlinear_max_steps_reached):
        return cast(RESULTS, RESULTS.max_steps_reached)
    return cast(RESULTS, RESULTS.nonfinite_objective)
