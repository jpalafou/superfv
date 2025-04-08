from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver


def _upwind(yl: ArrayLike, yr: ArrayLike, v: ArrayLike) -> ArrayLike:
    """
    Upwinding operator for states yl and yr with velocity v.

    Args:
        yl (ArrayLike): Left state. Has shape (nx, ny, nz, ...).
        yr (ArrayLike): Right state. Has shape (nx, ny, nz, ...).
        v (ArrayLike): Velocity. Has shape (nx, ny, nz, ...).

    Returns:
        ArrayLike: Flux. Has shape (nx, ny, nz, ...).
    """
    return v * np.where(v > 0, yl, np.where(v < 0, yr, 0))


def advection_upwind(
    array_slicer: ArraySlicer,
    yl: ArrayLike,
    yr: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Upwinding Riemann solver for the advection equation.

    Args:
        array_slicer (ArraySlicer): Array slicer object.
        yl (ArrayLike): Left state. Has shape (nvars, nx, ny, nz, ...).
        yr (ArrayLike): Right state. Has shape (nvars, nx, ny, nz, ...).
        dim (Literal["x", "y", "z"]): Dimension.

    Returns:
        ArrayLike: Flux. Has shape (nvars, nx, ny, nz, ...).
    """
    # get the velocity
    _slc = array_slicer
    vl, vr = yl[_slc(f"v{dim}")], yr[_slc(f"v{dim}")]
    v = np.where(np.abs(vl) > np.abs(vr), vl, vr)

    # compute the density flux
    out = np.zeros_like(yl)
    out[_slc("rho")] = _upwind(yl[_slc("rho")], yr[_slc("rho")], v)

    # handle passives
    if "user_defined_passives" in _slc.group_names:
        out[_slc("user_defined_passives")] = _upwind(
            yl[_slc("user_defined_passives")], yr[_slc("user_defined_passives")], v
        )
    return out


def call_riemann_solver(
    fv_solver: FiniteVolumeSolver,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
    primitives: bool,
    name: Literal["llf", "hllc"],
) -> ArrayLike:
    """
    Call the Riemann solver for the given dimension.

    Args:
        fv_solver (FiniteVolumeSolver): Finite volume solver object.
        wl (ArrayLike): Left state. Has shape (nvars, nx, ny, nz, ...).
        wr (ArrayLike): Right state. Has shape (nvars, nx, ny, nz, ...).
        dim (Literal["x", "y", "z"]): Dimension.
        gamma (float): Ratio of specific heats.
        primitives (bool): Whether the input states are in primitive variables. If
            False, the input states are in conservative variables.
        name (Literal["llf"]): Name of the Riemann solver.

    Returns:
        ArrayLike: Flux. Has shape (nvars, nx, ny, nz, ...).
    """
    # Get relevant variables
    _slc = fv_solver.array_slicer
    gamma = fv_solver.gamma
    hydro = fv_solver.hydro
    HAS_PASSIVES = "user_defined_passives" in _slc.group_names

    # Get the principal dimension and the other two transverse dimensions
    dim1, (dim2, dim3) = dim, {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    # Preallocate the flux array
    F = np.empty_like(wl)

    # Call the Riemann solver
    (
        F[_slc("rho")],
        F[_slc("m" + dim1)],
        F[_slc("m" + dim2)],
        F[_slc("m" + dim3)],
        F[_slc("E")],
        F_passives,
    ) = getattr(hydro, name)(
        rho_L=wl[_slc("rho")],
        v1_L=wl[_slc("v" + dim1)],
        v2_L=wl[_slc("v" + dim2)],
        v3_L=wl[_slc("v" + dim3)],
        P_L=wl[_slc("P")],
        rho_R=wr[_slc("rho")],
        v1_R=wr[_slc("v" + dim1)],
        v2_R=wr[_slc("v" + dim2)],
        v3_R=wr[_slc("v" + dim3)],
        P_R=wr[_slc("P")],
        gamma=gamma,
        passives_L=(
            cast(ArrayLike, wl)[_slc("user_defined_passives")] if HAS_PASSIVES else None
        ),
        passives_R=(
            cast(ArrayLike, wr)[_slc("user_defined_passives")] if HAS_PASSIVES else None
        ),
        primitives=primitives,
    )

    # Handle passives
    if HAS_PASSIVES:
        F[_slc("user_defined_passives")] = F_passives
    return F
