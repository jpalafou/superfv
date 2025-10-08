from types import ModuleType
from typing import Literal

from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap


def sound_speed(
    xp: ModuleType, idx: VariableIndexMap, w: ArrayLike, gamma: float
) -> ArrayLike:
    """
    Compute the sound speed from primitive variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        cs: Sound speed array. Has shape (1, nx, ny, nz, ...).
    """
    rho = w[idx("rho", keepdims=True)]
    P = w[idx("P", keepdims=True)]
    cs2 = gamma * P / rho
    cs = xp.sqrt(xp.maximum(cs2, 0.0))
    return cs


def prim_to_cons(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    gamma: float,
) -> ArrayLike:
    """
    Convert primitive variables to conservative variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        u: Array with conservative variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = w[idx("rho")]
    vx = w[idx("vx")]
    vy = w[idx("vy")]
    vz = w[idx("vz")]
    P = w[idx("P")]

    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    u = xp.empty_like(w)

    u[idx("rho")] = rho
    u[idx("mx")] = rho * vx
    u[idx("my")] = rho * vy
    u[idx("mz")] = rho * vz
    u[idx("E")] = K + P / (gamma - 1)

    if "passives" in idx:
        u[idx("passives")] = rho * w[idx("passives")]

    return u


def cons_to_prim(
    xp: ModuleType,
    idx: VariableIndexMap,
    u: ArrayLike,
    gamma: float,
) -> ArrayLike:
    """
    Convert conservative variables to primitive variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        u: Array of conservative variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        w: Array with primitive variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = u[idx("rho")]
    mx = u[idx("mx")]
    my = u[idx("my")]
    mz = u[idx("mz")]
    E = u[idx("E")]

    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    w = xp.empty_like(u)

    w[idx("rho")] = rho
    w[idx("vx")] = vx
    w[idx("vy")] = vy
    w[idx("vz")] = vz
    w[idx("P")] = (gamma - 1) * (E - K)

    if "passives" in idx:
        w[idx("passives")] = u[idx("passives")] / rho

    return w


def fluxes(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
) -> ArrayLike:
    """
    Compute the fluxes for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        gamma: Adiabatic index.

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    d1 = dim
    d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    rho = w[idx("rho")]
    v1 = w[idx("v" + d1)]
    v2 = w[idx("v" + d2)]
    v3 = w[idx("v" + d3)]
    P = w[idx("P")]

    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)

    F = xp.empty_like(w)

    F[idx("rho")] = rho * v1
    F[idx("m" + d1)] = rho * v1**2 + P
    F[idx("m" + d2)] = rho * v1 * v2
    F[idx("m" + d3)] = rho * v1 * v3
    F[idx("E")] = (K + P / (gamma - 1) + P) * v1

    if "passives" in idx:
        F[idx("passives")] = rho * v1 * w[idx("passives")]

    return F
