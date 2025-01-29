from typing import Literal

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
    array_slicer: ArraySlicer, yl: ArrayLike, yr: ArrayLike, dim: Literal["x", "y", "z"]
) -> ArrayLike:
    """
    Upwinding Riemann solver for the advection equation.

    Args:
        array_slicer (ArraySlicer): ArraySlicer object.
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
