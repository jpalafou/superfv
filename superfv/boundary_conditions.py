from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Literal, Optional, Tuple, Union, cast

from .field import MultivarField
from .fv import AXIS_TO_DIM
from .mesh import UniformFVMesh, xyz_tup
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap, crop


@dataclass
class BCcontext:
    axis: int
    left: bool
    slab_thickness: int
    f: Optional[MultivarField] = None
    variable_index_map: Optional[VariableIndexMap] = None
    mesh: Optional[UniformFVMesh] = None
    t: Optional[float] = None
    p: Optional[int] = None
    xp: Optional[ModuleType] = None


BCs = Literal[
    "none",
    "periodic",
    "dirichlet",
    "free",
    "symmetric",
    "reflective",
    "zeros",
    "ones",
    "patch",
]
PatchBC = Callable[[ArrayLike, BCcontext], None]
CallableBC = Union[MultivarField, PatchBC]


def apply_bc(
    xp: ModuleType,
    _u_: ArrayLike,
    pad_width: Tuple[int, int, int],
    mode: Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]],
    f: Tuple[
        Tuple[Optional[CallableBC], Optional[CallableBC]],
        Tuple[Optional[CallableBC], Optional[CallableBC]],
        Tuple[Optional[CallableBC], Optional[CallableBC]],
    ] = ((None, None), (None, None), (None, None)),
    variable_index_map: Optional[VariableIndexMap] = None,
    mesh: Optional[UniformFVMesh] = None,
    t: Optional[float] = None,
    p: Optional[int] = None,
):
    """
    Apply boundary conditions to the array _u_.

    Args:
        xp: Array module (e.g., numpy or cupy) for array operations.
        _u_: Array to which the boundary conditions are applied.
        pad_width: Tuple specifying the thickness of the boundary slabs in each
            dimension (x, y, z). The same thickness is applied to both sides of each
            dimension.
        mode: Tuple specifying the type of boundary condition for each side of each
            dimension. Supported types are:
            - "none": No boundary condition applied.
            - "periodic": Periodic boundary condition.
            - "dirichlet": Dirichlet boundary condition, requires a function in `f`
                to specify the boundary values.
            - "free": Free boundary condition, no constraints applied.
            - "symmetric": Symmetric boundary condition.
            - "reflective": Reflective boundary condition, requires 'v' variable in
                `variable_index_map`.
            - "zeros": Sets the boundary values to zero.
            - "ones": Sets the boundary values to one.
            - "patch": Custom boundary condition, requires a function in `f` to
                specify the boundary values.
        f: Tuple of tuples containing functions for Dirichlet or patch boundary
            conditions. Each function should match the `MultivarField` or `PatchBC`
            protocol. The structure of `f` should correspond to `mode`.
        variable_index_map: Optional VariableIndexMap object for indexing variables
            in _u_. Required for Dirichlet and reflective boundary conditions.
        mesh: Optional UniformFVMesh object representing the computational mesh.
            Required for Dirichlet boundary conditions.
        t: Optional time value to be passed to boundary condition functions.
        p: Optional polynomial order for numerical integration in Dirichlet boundary
            conditions.
    """
    for i, dim in enumerate(xyz_tup):
        for j, pos in enumerate(("l", "r")):
            if mode[i][j] == "none" or pad_width[i] == 0:
                continue

            context = BCcontext(
                axis=i + 1,
                left=(j == 0),
                slab_thickness=pad_width[i],
                f=None if mode[i][j] == "patch" else cast(MultivarField, f[i][j]),
                variable_index_map=variable_index_map,
                mesh=mesh,
                t=t,
                p=p,
                xp=xp,
            )

            match mode[i][j]:
                case "periodic":
                    apply_periodic_bc(_u_, context)
                case "dirichlet":
                    apply_dirichlet_bc(_u_, context)
                case "free":
                    apply_free_bc(_u_, context)
                case "symmetric":
                    apply_symmetric_bc(_u_, context)
                case "reflective":
                    apply_reflective_bc(_u_, context)
                case "zeros":
                    apply_uniform_bc(_u_, context, 0.0)
                case "ones":
                    apply_uniform_bc(_u_, context, 1.0)
                case "patch":
                    if f[i][j] is None:
                        raise ValueError(
                            "Patch boundary condition function is required to be passed"
                            " to `f` for 'patch' mode."
                        )
                    patch = cast(PatchBC, f[i][j])
                    patch(_u_, context)
                case _:
                    raise ValueError(
                        f"Boundary condition '{mode[i][j]}' not implemented for {dim}{pos} boundary."
                    )


def apply_periodic_bc(_u_: ArrayLike, context: BCcontext):
    """
    Apply periodic boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left
    if left:
        outer_slice = crop(axis, (None, slab_thickness))
        inner_slice = crop(axis, (-2 * slab_thickness, -slab_thickness))
    else:
        outer_slice = crop(axis, (-slab_thickness, None))
        inner_slice = crop(axis, (slab_thickness, 2 * slab_thickness))
    _u_[outer_slice] = _u_[inner_slice]


def apply_dirichlet_bc(_u_: ArrayLike, context: BCcontext):
    """
    Apply Dirichlet boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left
    f = cast(MultivarField, context.f)
    idx = cast(VariableIndexMap, context.variable_index_map)
    mesh = cast(UniformFVMesh, context.mesh)
    t = context.t
    p = cast(int, context.p)
    xp = cast(ModuleType, context.xp)

    if left:
        outer_slice = crop(axis, (None, slab_thickness), ndim=4)
    else:
        outer_slice = crop(axis, (-slab_thickness, None), ndim=4)

    slab_region = AXIS_TO_DIM[axis] + ("l" if left else "r")

    f_eval = mesh.perform_GaussLegendre_quadrature(
        lambda X, Y, Z: f(idx, X, Y, Z, t, xp=xp),
        node_axis=4,
        mesh_region=cast(Literal["xl", "xr", "yl", "yr", "zl", "zr"], slab_region),
        cell_region="interior",
        p=p,
    )
    _u_[outer_slice] = f_eval


def apply_free_bc(_u_: ArrayLike, context: BCcontext):
    """
    Apply free boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left
    if left:
        outer_slice = crop(axis, (None, slab_thickness))
        inner_slice = crop(axis, (slab_thickness, slab_thickness + 1))
    else:
        outer_slice = crop(axis, (-slab_thickness, None))
        inner_slice = crop(axis, (-slab_thickness - 1, -slab_thickness))
    _u_[outer_slice] = _u_[inner_slice]


def apply_symmetric_bc(_u_: ArrayLike, context: BCcontext):
    """
    Apply symmetric boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left
    flipper_slice = crop(axis, (None, None), step=-1)
    if left:
        outer_slice = crop(axis, (None, slab_thickness))
        inner_slice = crop(axis, (slab_thickness, 2 * slab_thickness))
    else:
        outer_slice = crop(axis, (-slab_thickness, None))
        inner_slice = crop(axis, (-2 * slab_thickness, -slab_thickness))
    _u_[outer_slice] = _u_[inner_slice][flipper_slice]


def apply_reflective_bc(_u_: ArrayLike, context: BCcontext):
    """
    Apply reflective boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left
    idx = cast(VariableIndexMap, context.variable_index_map)

    outer_slice = crop(
        axis, (None, slab_thickness) if left else (-slab_thickness, None)
    )
    dim = AXIS_TO_DIM[axis]

    velocity = "v" + dim
    if velocity not in idx.var_names:
        raise ValueError(
            "VariableIndexMap must contain 'v' variable for reflective boundary conditions."
        )

    apply_symmetric_bc(_u_, context)
    _u_[outer_slice][idx(velocity)] *= -1


def apply_uniform_bc(_u_: ArrayLike, context: BCcontext, value: float):
    """
    Apply uniform boundary conditions to the array _u_ along the specified axis.

    Args:
        _u_: Array to which the boundary conditions are applied.
        context: BCcontext object containing parameters for applying the BC.
        value: Uniform value to set in the boundary region.
    """
    slab_thickness = context.slab_thickness
    axis = context.axis
    left = context.left

    if left:
        _u_[crop(axis, (None, slab_thickness))] = value
    else:
        _u_[crop(axis, (-slab_thickness, None))] = value
