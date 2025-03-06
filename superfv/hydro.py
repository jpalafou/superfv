import numpy as np
import wtflux.hydro as hydro

from superfv.tools.array_management import ArrayLike, ArraySlicer


def conservatives_from_primitives(
    array_slicer: ArraySlicer, w: ArrayLike, gamma: float
):
    """
    Compute the conservative variables from the primitive variables.

    Args:
        array_slicer (ArraySlicer): Slicer object that defines the mapping between
            variable names and indices in the arrays.
        w (ArrayLike): Array of primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        gamma (float): Adiabatic index.

    Returns:
        u (ArrayLike): Array with conservative variables. Has shape
            (nvars, nx, ny, nz, ...).
    """
    _slc = array_slicer
    u = np.empty_like(w)
    HAS_PASSIVES = "passives" in array_slicer.group_names
    (
        u[_slc("rho")],
        u[_slc("mx")],
        u[_slc("my")],
        u[_slc("mz")],
        u[_slc("E")],
        passives,
    ) = hydro.conservatives_from_primitives(
        rho=w[_slc("rho")],
        v1=w[_slc("vx")],
        v2=w[_slc("vy")],
        v3=w[_slc("vz")],
        P=w[_slc("P")],
        gamma=gamma,
        passives=w[_slc("passives")] if HAS_PASSIVES else None,
    )
    if HAS_PASSIVES:
        u[_slc("passives")] = passives
    return u


def primitives_from_conservatives(
    array_slicer: ArraySlicer, u: ArrayLike, gamma: float
):
    """
    Compute the primitive variables from the conservative variables.

    Args:
        array_slicer (ArraySlicer): Slicer object that defines the mapping between
            variable names and indices in the arrays.
        u (ArrayLike): Array of conservative variables. Has shape
            (nvars, nx, ny, nz, ...).
        gamma (float): Adiabatic index.

    Returns:
        w (ArrayLike): Array with primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
    """
    _slc = array_slicer
    w = np.empty_like(u)
    HAS_PASSIVES = "passives" in array_slicer.group_names
    (
        w[_slc("rho")],
        w[_slc("mx")],
        w[_slc("my")],
        w[_slc("mz")],
        w[_slc("E")],
        passives,
    ) = hydro.primitives_from_conservatives(
        rho=u[_slc("rho")],
        m1=u[_slc("vx")],
        m2=u[_slc("vy")],
        m3=u[_slc("vz")],
        E=u[_slc("P")],
        gamma=gamma,
        conserved_passives=u[_slc("passives")] if HAS_PASSIVES else None,
    )
    if HAS_PASSIVES:
        w[_slc("passives")] = passives
    return w
