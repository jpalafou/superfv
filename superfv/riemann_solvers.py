from types import ModuleType
from typing import Literal, Tuple

from .hydro import fluxes, prim_to_cons, sound_speed
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap


def advection_upwind(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Compute the upwind advection flux for a given dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension for the advection flux ("x", "y", or "z").

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    vl = wl[idx("v" + dim)]
    vr = wr[idx("v" + dim)]
    v = xp.where(xp.abs(vl) > xp.abs(vr), vl, vr)

    F = xp.empty_like(wl)

    F[idx("rho")] = v * xp.where(v > 0, wl[idx("rho")], wr[idx("rho")])
    F[idx("vx")] = 0.0
    F[idx("vy")] = 0.0
    F[idx("vz")] = 0.0
    if "passives" in idx.group_var_map:
        F[idx("passives")] = v * xp.where(
            v > 0, wl[idx("passives")], wr[idx("passives")]
        )

    return F


def llf(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    gamma: float,
) -> ArrayLike:
    """
    Compute the LLF flux for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        active_dims: Tuple of active dimensions (e.g., ("x", "y", "z")).
        gamma: Adiabatic index.

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    ul = prim_to_cons(xp, idx, wl, active_dims, gamma)
    ur = prim_to_cons(xp, idx, wr, active_dims, gamma)

    cl = sound_speed(xp, idx, wl, gamma) + xp.abs(wl[idx("v" + dim)])
    cr = sound_speed(xp, idx, wr, gamma) + xp.abs(wr[idx("v" + dim)])
    cmax = xp.maximum(cl, cr)

    Fl = fluxes(xp, idx, wl, dim, active_dims, gamma)
    Fr = fluxes(xp, idx, wr, dim, active_dims, gamma)

    F = 0.5 * (Fl + Fr - cmax * (ur - ul))

    return F


def hllc(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    gamma: float,
) -> ArrayLike:
    """
    Compute the HLLC flux for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        active_dims: Tuple of active dimensions (e.g., ("x", "y", "z")).
        gamma: Adiabatic index.

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    return

    # d1 = dim
    # d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    # rhol = wl[idx("rho")]
    # v1l = wl[idx("v" + d1)]
    # v2l = wl[idx("v" + d2)] if "y" in active_dims else 0.0
    # v3l = wl[idx("v" + d3)] if "z" in active_dims else 0.0
    # Pl = wl[idx("P")]
    # rhor = wr[idx("rho")]
    # v1r = wr[idx("v" + d1)]
    # v2r = wr[idx("v" + d2)] if "y" in active_dims else 0.0
    # v3r = wr[idx("v" + d3)] if "z" in active_dims else 0.0
    # Pr = wr[idx("P")]

    # cl = sound_speed(xp, idx, wl, gamma)
    # cr = sound_speed(xp, idx, wr, gamma)
    # cmax = xp.maximum(cl, cr)

    # sl = xp.minimum(v1l, v1r) - cmax
    # sr = xp.maximum(v1l, v1r) + cmax

    # rcl = rhol * (v1l - sl)
    # rcr = rhor * (sr - v1r)

    # star_denom = rcl + rcr
    # vstar = (rcr * v1r + rcl * v1l + (Pl - Pr)) / star_denom
    # Pstar = (rcr * Pl + rcl * Pr + rcl * rcr * (v1l - v1r)) / star_denom

    # rhostarl_denom = sl - v1l
    # rhostarr_denom = sr - v1r
    # Estarl_denom = sl - vstar
    # Estarr_denom = sr - vstar
    # rhostarl = rhol * (sl - v1l) / rhostarl_denom
    # rhostarr = rhor * (sr - v1r) / rhostarr_denom
