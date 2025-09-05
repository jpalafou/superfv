from types import ModuleType
from typing import Literal, Tuple

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
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    gamma: float,
) -> ArrayLike:
    """
    Convert primitive variables to conservative variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        active_dims: Tuple of active dimensions (e.g., ("x", "y", "z")).
        gamma: Adiabatic index.

    Returns:
        u: Array with conservative variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = w[idx("rho")]
    vx = w[idx("vx")] if "x" in active_dims else 0.0
    vy = w[idx("vy")] if "y" in active_dims else 0.0
    vz = w[idx("vz")] if "z" in active_dims else 0.0
    P = w[idx("P")]

    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    u = xp.empty_like(w)

    u[idx("rho")] = rho
    u[idx("mx")] = rho * vx if "x" in active_dims else 0.0
    u[idx("my")] = rho * vy if "y" in active_dims else 0.0
    u[idx("mz")] = rho * vz if "z" in active_dims else 0.0
    u[idx("E")] = K + P / (gamma - 1)

    if "passives" in idx:
        u[idx("passives")] = rho * w[idx("passives")]

    return u


def cons_to_prim(
    xp: ModuleType,
    idx: VariableIndexMap,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    gamma: float,
) -> ArrayLike:
    """
    Convert conservative variables to primitive variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        u: Array of conservative variables. Has shape (nvars, nx, ny, nz, ...).
        active_dims: Tuple of active dimensions (e.g., ("x", "y", "z")).
        gamma: Adiabatic index.

    Returns:
        w: Array with primitive variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = u[idx("rho")]
    mx = u[idx("mx")]
    my = u[idx("my")]
    mz = u[idx("mz")]
    E = u[idx("E")]

    vx = mx / rho if "x" in active_dims else 0.0
    vy = my / rho if "y" in active_dims else 0.0
    vz = mz / rho if "z" in active_dims else 0.0
    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    w = xp.empty_like(u)

    w[idx("rho")] = rho
    w[idx("vx")] = vx if "x" in active_dims else 0.0
    w[idx("vy")] = vy if "y" in active_dims else 0.0
    w[idx("vz")] = vz if "z" in active_dims else 0.0
    w[idx("P")] = (gamma - 1) * (E - K)

    if "passives" in idx:
        w[idx("passives")] = u[idx("passives")] / rho

    return w


def fluxes(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    gamma: float,
) -> ArrayLike:
    """
    Compute the fluxes for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        active_dims: Tuple of active dimensions (e.g., ("x", "y", "z")).
        gamma: Adiabatic index.

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    if dim not in active_dims:
        raise ValueError(f"Dimension '{dim}' is not active in the mesh.")

    d1 = dim
    d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    rho = w[idx("rho")]
    v1 = w[idx("v" + d1)] if d1 in active_dims else 0.0
    v2 = w[idx("v" + d2)] if d2 in active_dims else 0.0
    v3 = w[idx("v" + d3)] if d3 in active_dims else 0.0
    P = w[idx("P")]

    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)

    F = xp.empty_like(w)

    F[idx("rho")] = rho * v1
    F[idx("m" + d1)] = rho * v1**2 + P
    F[idx("m" + d2)] = rho * v1 * v2 if d2 in active_dims else 0.0
    F[idx("m" + d3)] = rho * v1 * v3 if d3 in active_dims else 0.0
    F[idx("E")] = (K + P / (gamma - 1) + P) * v1

    if "passives" in idx:
        F[idx("passives")] = rho * v1 * w[idx("passives")]

    return F
