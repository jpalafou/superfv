from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Callable, Literal, Optional, Tuple, Union, cast

from .axes import AXIS_TO_DIM
from .field import MultivarField
from .mesh import UniformFVMesh
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap, crop


class BC(Enum):
    PERIODIC = 0
    DIRICHLET = 1
    FREE = 2
    SYMMETRIC = 3
    REFLECTIVE = 4
    ZEROS = 5
    ONES = 6
    PATCH = 7
    NONE = 8
    IC = 9  # gets switched to DIRICHLET in the solver


@dataclass
class BCcontext:
    axis: int
    lower: bool
    nghost: int
    f: Optional[MultivarField] = None
    variable_index_map: Optional[VariableIndexMap] = None
    mesh: Optional[UniformFVMesh] = None
    t: Optional[float] = None
    p: Optional[int] = None
    xp: Optional[ModuleType] = None


PatchBC = Callable[[ArrayLike, BCcontext], None]


def apply_bc(
    xp: ModuleType,
    _u_: ArrayLike,
    nghost: int,
    bcx: Tuple[BC, BC],
    bcy: Tuple[BC, BC],
    bcz: Tuple[BC, BC],
    bcx_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
    bcx_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
    bcy_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
    bcy_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
    bcz_callable_lower: Optional[Union[MultivarField, PatchBC]] = None,
    bcz_callable_upper: Optional[Union[MultivarField, PatchBC]] = None,
    variable_index_map: Optional[VariableIndexMap] = None,
    mesh: Optional[UniformFVMesh] = None,
    t: Optional[float] = None,
    p: Optional[int] = None,
):
    for i, (dim, modelr) in enumerate(zip(["x", "y", "z"], [bcx, bcy, bcz])):
        for j, bound in enumerate(("lower", "upper")):
            mode = modelr[j]
            bc_callable = [
                bcx_callable_lower,
                bcx_callable_upper,
                bcy_callable_lower,
                bcy_callable_upper,
                bcz_callable_lower,
                bcz_callable_upper,
            ][2 * i + j]

            if mode == BC.NONE or nghost == 0:
                continue

            context = BCcontext(
                axis=i + 1,
                lower=(j == 0),
                nghost=nghost,
                f=None if mode != BC.DIRICHLET else cast(MultivarField, bc_callable),
                variable_index_map=variable_index_map,
                mesh=mesh,
                t=t,
                p=p,
                xp=xp,
            )

            match mode:
                case BC.PERIODIC:
                    apply_periodic_bc(_u_, context)
                case BC.DIRICHLET:
                    apply_dirichlet_bc(_u_, context)
                case BC.FREE:
                    apply_free_bc(_u_, context)
                case BC.SYMMETRIC:
                    apply_symmetric_bc(_u_, context)
                case BC.REFLECTIVE:
                    apply_reflective_bc(_u_, context)
                case BC.ZEROS:
                    apply_uniform_bc(_u_, context, 0.0)
                case BC.ONES:
                    apply_uniform_bc(_u_, context, 1.0)
                case BC.PATCH:
                    patch = cast(PatchBC, bc_callable)
                    patch(_u_, context)
                case _:
                    raise ValueError(
                        f"Boundary condition '{mode}' not implemented for {bound} {dim} boundary."
                    )


def apply_periodic_bc(_u_: ArrayLike, context: BCcontext):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    if lower:
        outer_slice = crop(axis, (None, nghost))
        inner_slice = crop(axis, (-2 * nghost, -nghost))
    else:
        outer_slice = crop(axis, (-nghost, None))
        inner_slice = crop(axis, (nghost, 2 * nghost))
    _u_[outer_slice] = _u_[inner_slice]


def apply_dirichlet_bc(_u_: ArrayLike, context: BCcontext):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    f = cast(MultivarField, context.f)
    idx = cast(VariableIndexMap, context.variable_index_map)
    mesh = cast(UniformFVMesh, context.mesh)
    t = context.t
    p = cast(int, context.p)
    xp = cast(ModuleType, context.xp)

    if lower:
        outer_slice = crop(axis, (None, nghost), ndim=4)
    else:
        outer_slice = crop(axis, (-nghost, None), ndim=4)

    slab_region = AXIS_TO_DIM[axis] + ("l" if lower else "r")

    f_eval = xp.empty_like(_u_[outer_slice])
    mesh.perform_GaussLegendre_quadrature(
        lambda X, Y, Z: f(idx, X, Y, Z, t, xp=xp),
        f_eval,
        mesh_region=cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], slab_region),
        cell_region="interior",
        p=p,
    )
    _u_[outer_slice] = f_eval


def apply_free_bc(_u_: ArrayLike, context: BCcontext):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    if lower:
        outer_slice = crop(axis, (None, nghost))
        inner_slice = crop(axis, (nghost, nghost + 1))
    else:
        outer_slice = crop(axis, (-nghost, None))
        inner_slice = crop(axis, (-nghost - 1, -nghost))
    _u_[outer_slice] = _u_[inner_slice]


def apply_symmetric_bc(_u_: ArrayLike, context: BCcontext):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    flipper_slice = crop(axis, (None, None), step=-1)
    if lower:
        outer_slice = crop(axis, (None, nghost))
        inner_slice = crop(axis, (nghost, 2 * nghost))
    else:
        outer_slice = crop(axis, (-nghost, None))
        inner_slice = crop(axis, (-2 * nghost, -nghost))
    _u_[outer_slice] = _u_[inner_slice][flipper_slice]


def apply_reflective_bc(_u_: ArrayLike, context: BCcontext):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    idx = cast(VariableIndexMap, context.variable_index_map)

    outer_slice = crop(axis, (None, nghost) if lower else (-nghost, None))
    dim = AXIS_TO_DIM[axis]

    velocity = "v" + dim
    if velocity not in idx.var_names:
        raise ValueError(
            "VariableIndexMap must contain 'v' variable for reflective boundary conditions."
        )

    apply_symmetric_bc(_u_, context)
    _u_[outer_slice][idx(velocity)] *= -1


def apply_uniform_bc(_u_: ArrayLike, context: BCcontext, value: float):
    nghost = context.nghost
    axis = context.axis
    lower = context.lower

    if lower:
        _u_[crop(axis, (None, nghost))] = value
    else:
        _u_[crop(axis, (-nghost, None))] = value
