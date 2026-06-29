from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np

from .axes import AXIS_TO_DIM
from .field import MultivarField
from .hydro import prim_to_cons
from .mesh import UniformFiniteVolumeMesh
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import crop
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


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


MESH_REGION_LOOKUP: Dict[
    Literal["x", "y", "z"],
    Dict[
        Literal["l", "r"], Literal["xl_slab", "xr_slab", "yl_slab", "yr_slab", "zl_slab", "zr_slab"]
    ],
] = {
    "x": {"l": "xl_slab", "r": "xr_slab"},
    "y": {"l": "yl_slab", "r": "yr_slab"},
    "z": {"l": "zl_slab", "r": "zr_slab"},
}


@dataclass
class BCcontext:
    axis: int
    lower: bool
    nghost: int
    f: Optional[MultivarField] = None
    variable_index_map: Optional[VariableIndexMap] = None
    mesh: Optional[UniformFiniteVolumeMesh] = None
    t: Optional[float] = None
    sampling_p: Optional[int] = None
    gamma: Optional[float] = None


PatchBC = Callable[[ArrayLike, BCcontext], None]


def apply_bc(
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
    mesh: Optional[UniformFiniteVolumeMesh] = None,
    t: Optional[float] = None,
    sampling_p: Optional[int] = None,
    gamma: Optional[float] = None,
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
                f=cast(MultivarField, bc_callable) if mode == BC.DIRICHLET else None,
                variable_index_map=variable_index_map,
                mesh=mesh,
                t=t,
                sampling_p=sampling_p,
                gamma=gamma,
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
    xp = cp if CUPY_AVAILABLE and isinstance(_u_, cp.ndarray) else np

    nghost = context.nghost
    axis = context.axis
    lower = context.lower
    f = context.f
    idx = context.variable_index_map
    mesh = context.mesh
    t = context.t
    sampling_p = context.sampling_p
    gamma = context.gamma

    if f is None:
        raise ValueError("Dirichlet boundary condition requires a callable function.")
    if idx is None:
        raise ValueError("Dirichlet boundary condition requires a VariableIndexMap.")
    if mesh is None:
        raise ValueError("Dirichlet boundary condition requires a mesh.")
    if t is None:
        raise ValueError("Dirichlet boundary condition requires a time value.")
    if sampling_p is None:
        raise ValueError("Dirichlet boundary condition requires a quadrature order.")
    if gamma is None:
        raise ValueError("Dirichlet boundary condition requires gamma.")

    if lower:
        outer_slice = crop(axis, (None, nghost), ndim=4)
    else:
        outer_slice = crop(axis, (-nghost, None), ndim=4)

    def conservative_boundary(X, Y, Z):
        w = f(idx, X, Y, Z, t, xp=xp)
        u = xp.empty_like(w)
        prim_to_cons(w, u, idx, gamma)
        return u

    _u_[outer_slice] = mesh.perform_GaussLegendre_quadrature(
        conservative_boundary,
        sampling_p,
        MESH_REGION_LOOKUP[AXIS_TO_DIM[axis]][("l" if lower else "r")],
    )


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
    idx = context.variable_index_map

    if idx is None:
        raise ValueError("Reflective boundary condition requires a VariableIndexMap.")

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
