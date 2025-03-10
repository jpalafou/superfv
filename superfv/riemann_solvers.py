from typing import Any, Literal, Optional, cast

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer


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
    if "passives" in _slc.group_names:
        out[_slc("passives")] = _upwind(yl[_slc("passives")], yr[_slc("passives")], v)
    return out


def llf(
    hydro: Any,
    array_slicer: ArraySlicer,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
    ul: Optional[ArrayLike] = None,
    ur: Optional[ArrayLike] = None,
):
    """
    Compute the Lax-Friedrichs Riemann flux for the Euler equations.

    Args:
        hydro (Any): Hydro namespace.
        array_slicer (ArraySlicer): Array slicer object.
        wl (ArrayLike): Left state. Has shape (nvars, nx, ny, nz, ...).
        wr (ArrayLike): Right state. Has shape (nvars, nx, ny, nz, ...).
        dim (Literal["x", "y", "z"]): Dimension.
        gamma (float): Adiabatic index.
        ul (Optional[ArrayLike]): Left conserved variables. Has shape
            (nvars, nx, ny, nz, ...).
        ur (Optional[ArrayLike]): Right conserved variables. Has shape
            (nvars, nx, ny, nz, ...).

    Returns:
        F (ArrayLike): Flux. Has shape (nvars, nx, ny, nz, ...).
    """
    _slc = array_slicer
    HAS_PASSIVES = "passives" in _slc.group_names
    dim1, (dim2, dim3) = dim, {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]
    F = np.empty_like(wl)
    (
        F[_slc("rho")],
        F[_slc("m" + dim1)],
        F[_slc("m" + dim2)],
        F[_slc("m" + dim3)],
        F[_slc("E")],
        F_passives,
    ) = hydro.llf(
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
        m1_L=cast(ArrayLike, ul)[_slc("m" + dim1)],
        m2_L=cast(ArrayLike, ul)[_slc("m" + dim2)],
        m3_L=cast(ArrayLike, ul)[_slc("m" + dim3)],
        E_L=cast(ArrayLike, ul)[_slc("E")],
        m1_R=cast(ArrayLike, ur)[_slc("m" + dim1)],
        m2_R=cast(ArrayLike, ur)[_slc("m" + dim2)],
        m3_R=cast(ArrayLike, ur)[_slc("m" + dim3)],
        E_R=cast(ArrayLike, ur)[_slc("E")],
        passives_L=cast(ArrayLike, wl)[_slc("passives")] if HAS_PASSIVES else None,
        passives_R=cast(ArrayLike, wr)[_slc("passives")] if HAS_PASSIVES else None,
        conserved_passives_L=(
            cast(ArrayLike, ul)[_slc("passives")] if HAS_PASSIVES else None
        ),
        conserved_passives_R=(
            cast(ArrayLike, ur)[_slc("passives")] if HAS_PASSIVES else None
        ),
    )
    if HAS_PASSIVES:
        F[_slc("passives")] = F_passives
    return F
