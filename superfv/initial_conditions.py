from types import ModuleType
from typing import Literal, Optional, Tuple

import numpy as np

from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap


def parse_xyz(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> str:
    """
    Returns a string with the dimensions of the input arrays.

    Args:
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).

    Returns:
        str: String with the dimensions of the input arrays (some combination of "x",
            "y", and "z").
    """
    if not x.shape == y.shape == z.shape:
        raise ValueError("x, y, and z must have the same shape.")
    return "".join(dim for dim, size in zip("xyz", x.shape) if size > 1)


def _uninitialized(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: float = 0.0,
    *,
    xp: ModuleType,
) -> ArrayLike:
    raise NotImplementedError(
        "`_uninitialized` cannot be called and is meant to be used as a placeholder"
        " for pickling."
    )


def sinus(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: float = 0.0,
    bounds: Tuple[float, float] = (0, 1),
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    P: float = 1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for sinusoidal initial condition that is periodic on the interval
    [0, 1] in each dimension.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Time variable.
        bounds: Bounds of the density sinusoidal function.
        vx: Uniform velocity in the x-direction.
        vy: Uniform velocity in the y-direction.
        vz: Uniform velocity in the z-direction.
        P: Pressure.
        xp: NumPy namespace module (e.g., `np` or `cupy`).s
    """
    dims = parse_xyz(x, y, z)
    out = xp.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMaps
    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        r = xp.zeros_like(x)
        if "x" in dims:
            r += x - vx * t
        if "y" in dims:
            r += y - vy * t
        if "z" in dims:
            r += z - vz * t
        out[idx("rho")] = (
            0.5 * (bounds[1] - bounds[0]) * xp.sin(2 * np.pi * r) + 0.5 + bounds[0]
        )
        out[idx("vx")] = vx
        out[idx("vy")] = vy
        out[idx("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    if "P" in idx.var_names:
        out[idx("P")] = P
    return out


def square(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: float = 0.0,
    bounds: Tuple[float, float] = (0, 1),
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    P: float = 1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the square wave initial condition that is periodic on the
    interval [0, 1] in each dimension.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Time variable.
        bounds: Bounds of the density square function.
        vx: Uniform velocity in the x-direction.
        vy: Uniform velocity in the y-direction.
        vz: Uniform velocity in the z-direction.
        P: Pressure.
        xp: NumPy namespace module (e.g., `np` or `cupy`).

    Returns:
        ArrayLike: Array with the initial conditions for the hydro variables.
    """
    dims = parse_xyz(x, y, z)
    out = xp.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        r = xp.ones_like(x).astype(bool)
        if "x" in dims:
            xa = (x - vx * t) % 1
            r &= (xa >= 0.25) & (xa <= 0.75)
        if "y" in dims:
            ya = (y - vy * t) % 1
            r &= (ya >= 0.25) & (ya <= 0.75)
        if "z" in dims:
            za = (z - vz * t) % 1
            r &= (za >= 0.25) & (za <= 0.75)
        r = r.astype(float)
        out[idx("rho")] = (bounds[1] - bounds[0]) * r + bounds[0]
        out[idx("vx")] = vx
        out[idx("vy")] = vy
        out[idx("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    if "P" in idx.var_names:
        out[idx("P")] = P
    return out


def slotted_disk(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: float = 0.0,
    rho_min_max: Tuple[float, float] = (0.0, 1.0),
    rotation: Literal["cw", "ccw"] = "ccw",
    P: float = 1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the slotted disk initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Time variable.
        rho_min_max: Minimum and maximum values of the density.
        rotation: Rotation direction of the disk: "cw" for clockwise, "ccw" for
            counter-clockwise.
        P: Pressure.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    out = xp.empty((len(idx.idxs), *x.shape))

    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # angular velocity
        omega = 1.0
        theta = -omega * t if rotation == "ccw" else omega * t

        # rotate coordinates backward in time
        x0, y0 = x - 0.5, y - 0.5
        x_rot = xp.cos(theta) * x0 - xp.sin(theta) * y0
        y_rot = xp.sin(theta) * x0 + xp.cos(theta) * y0

        # define disk in rotated coordinates
        rsq = xp.sqrt(x_rot**2 + (y_rot - 0.25) ** 2)
        inside_disk = rsq < 0.15
        inside_disk &= ~((xp.abs(x_rot) < 0.025) & (y_rot < 0.35))

        out[idx("rho")] = xp.where(inside_disk, rho_min_max[1], rho_min_max[0])
        out[idx("vx")] = -y0 if rotation == "ccw" else y0
        out[idx("vy")] = x0 if rotation == "ccw" else -x0
        out[idx("vz")] = 0.0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    if "P" in idx.var_names:
        out[idx("P")] = P

    return out


def sod_shock_tube_1d(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    pos1: float = 0.5,
    rhol: float = 1.0,
    rhor: float = 0.125,
    vl: float = 0,
    vr: float = 0,
    pl: float = 1.0,
    pr: float = 0.1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Sod shock tube initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        pos1: Position of the discontinuity.
        rhol: Density on the left side of the discontinuity.
        rhor: Density on the right side of the discontinuity.
        vl: Velocity on the left side of the discontinuity.
        vr: Velocity on the right side of the discontinuity.
        pl: Pressure on the left side of the discontinuity.
        pr: Pressure on the right side of the discontinuity.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    out = xp.empty((len(idx.idxs), *x.shape))

    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition only works in 1D.")

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dims]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dims]
        out[idx("rho")] = xp.where(r < pos1, rhol, rhor)
        out[idx("v" + dims)] = xp.where(r < pos1, vl, vr)
        out[idx("v" + orth_dim1)] = 0
        out[idx("v" + orth_dim2)] = 0
        out[idx("P")] = xp.where(r < pos1, pl, pr)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out


def velocity_ramp(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    rho0: float = 1,
    P0: float = 1,
    H0: float = 1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the velocity ramp initial condition with a uniform density and
    pressure.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        rho0: Initial uniform density.
        P0: Initial uniform pressure.
        H0: Initial uniform velocity gradient.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    out = xp.empty((len(idx.idxs), *x.shape))

    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition only works in 1D.")

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dims]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dims]
        out[idx("rho")] = rho0
        out[idx("v" + dims)] = H0 * (r - 0.5)
        out[idx("v" + orth_dim1)] = 0
        out[idx("v" + orth_dim2)] = 0
        out[idx("P")] = P0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out


def sedov(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    gamma: Optional[float] = None,
    h: Optional[float] = None,
    rho0: float = 1,
    P0: float = 0,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Sedov blast wave initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        gamma: Ratio of specific heats.
        h: Mesh size. Assumed to be the same in all dimensions.
        rho0: Initial uniform density.
        P0: Background pressure.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    if gamma is None or h is None:
        raise ValueError("Sedov initial condition requires `gamma` and `h` to be set.")

    dims = parse_xyz(x, y, z)
    out = xp.zeros((len(idx.idxs), *x.shape))

    inside_blast_cell = xp.ones_like(x, dtype=bool)
    if "x" in dims:
        inside_blast_cell &= xp.abs(x) < h
    if "y" in dims:
        inside_blast_cell &= xp.abs(y) < h
    if "z" in dims:
        inside_blast_cell &= xp.abs(z) < h

    P_blast_cell = (gamma - 1) * 1 / ((2 * h) ** len(dims))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        out[idx("rho")] = rho0
        out[idx("P")] = xp.where(inside_blast_cell, P_blast_cell, P0)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out
