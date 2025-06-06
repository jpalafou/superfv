from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .tools.array_management import ArrayLike, VariableIndexMap

if TYPE_CHECKING:
    from superfv.euler_solver import EulerSolver


def _upwind(yl: ArrayLike, yr: ArrayLike, v: ArrayLike) -> ArrayLike:
    """
    Upwinding operator for states yl and yr with velocity v.

    Args:
        yl: Left state. Has shape (nx, ny, nz, ...).
        yr: Right state. Has shape (nx, ny, nz, ...).
        v: Velocity. Has shape (nx, ny, nz, ...).

    Returns:
        Flux array. Has shape (nx, ny, nz, ...).
    """
    return v * np.where(v > 0, yl, np.where(v < 0, yr, 0))


def advection_upwind(
    idx: VariableIndexMap,
    yl: ArrayLike,
    yr: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Upwinding Riemann solver for the advection equation.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        yl: Left state. Has shape (nvars, nx, ny, nz, ...).
        yr: Right state. Has shape (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".

    Returns:
        Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    # get the velocity
    vl, vr = yl[idx("v" + dim)], yr[idx("v" + dim)]
    v = np.where(np.abs(vl) > np.abs(vr), vl, vr)

    # compute the density flux
    out = np.zeros_like(yl)
    out[idx("rho")] = _upwind(yl[idx("rho")], yr[idx("rho")], v)

    # handle passives
    if "user_defined_passives" in idx.group_names:
        out[idx("user_defined_passives")] = _upwind(
            yl[idx("user_defined_passives")], yr[idx("user_defined_passives")], v
        )
    return out


def call_riemann_solver(
    euler_solver: EulerSolver,
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
        euler_solver: Euler solver object.
        wl: Left state array. Has shape (nvars, nx, ny, nz, ...).
        wr: Right state array. Has shape (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        gamma: Ratio of specific heats.
        primitives: Whether the input states are in primitive variables. If False, the
            input states are in conservative variables.
        name: Name of the Riemann solver to use. Can be "llf" or "hllc".

    Returns:
        Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    # Get relevant variables
    idx = euler_solver.variable_index_map
    hydro = euler_solver.hydro
    HAS_PASSIVES = "user_defined_passives" in idx.group_names

    # Get the principal dimension and the other two transverse dimensions
    dim1, (dim2, dim3) = dim, {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    # Preallocate the flux array
    F = np.empty_like(wl)

    # Call the Riemann solver
    (
        F[idx("rho")],
        F[idx("m" + dim1)],
        F[idx("m" + dim2)],
        F[idx("m" + dim3)],
        F[idx("E")],
        F_passives,
    ) = getattr(hydro, name)(
        rho_L=wl[idx("rho")],
        v1_L=wl[idx("v" + dim1)],
        v2_L=wl[idx("v" + dim2)],
        v3_L=wl[idx("v" + dim3)],
        P_L=wl[idx("P")],
        rho_R=wr[idx("rho")],
        v1_R=wr[idx("v" + dim1)],
        v2_R=wr[idx("v" + dim2)],
        v3_R=wr[idx("v" + dim3)],
        P_R=wr[idx("P")],
        gamma=gamma,
        passives_L=(
            cast(ArrayLike, wl)[idx("user_defined_passives")] if HAS_PASSIVES else None
        ),
        passives_R=(
            cast(ArrayLike, wr)[idx("user_defined_passives")] if HAS_PASSIVES else None
        ),
        primitives=primitives,
    )

    # Handle passives
    if HAS_PASSIVES:
        F[idx("user_defined_passives")] = F_passives
    return F
