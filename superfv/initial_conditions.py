from typing import Literal, Tuple

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
    bounds: Tuple[float, float] = (0, 1),
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    P: float = 1,
) -> ArrayLike:
    """
    Returns array for sinusoidal initial condition that is periodic on the interval
    [0, 1] in each dimension.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        bounds: Bounds of the density sinusoidal function.
        vx: Uniform velocity in the x-direction.
        vy: Uniform velocity in the y-direction.
        vz: Uniform velocity in the z-direction.
        P: Pressure.
    """
    dims = parse_xyz(x, y, z)
    out = np.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMaps
    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        r = int("x" in dims) * x + int("y" in dims) * y + int("z" in dims) * z
        out[idx("rho")] = (
            0.5 * (bounds[1] - bounds[0]) * np.sin(2 * np.pi * r) + 0.5 + bounds[0]
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
    bounds: Tuple[float, float] = (0, 1),
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    P: float = 1,
) -> ArrayLike:
    """
    Returns array for the square wave initial condition that is periodic on the
    interval [0, 1] in each dimension.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        bounds: Bounds of the density square function.
        vx: Uniform velocity in the x-direction.
        vy: Uniform velocity in the y-direction.
        vz: Uniform velocity in the z-direction.
        P: Pressure.
    """
    dims = parse_xyz(x, y, z)
    out = np.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        r = np.ones_like(x).astype(bool)
        if "x" in dims:
            r &= (x >= 0.25) & (x <= 0.75)
        if "y" in dims:
            r &= (y >= 0.25) & (y <= 0.75)
        if "z" in dims:
            r &= (z >= 0.25) & (z <= 0.75)
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
    rho_min_max: Tuple[float, float] = (0.0, 1.0),
    rotation: Literal["cw", "ccw"] = "ccw",
) -> ArrayLike:
    """
    Returns array for the slotted disk initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        rho_min_max: Minimum and maximum values of the density.
        rotation: Rotation direction of the disk: "cw" for clockwise, "ccw" for
            counter-clockwise.
    """
    out = np.empty((len(idx.idxs), *x.shape))

    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        xc, yc = x - 0.5, y - 0.5
        rsq = np.sqrt(xc**2 + (y - 0.75) ** 2)
        inside_disk = rsq < 0.15
        inside_disk &= np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))

        out[idx("rho")] = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
        out[idx("vx")] = -yc if rotation == "ccw" else yc
        out[idx("vy")] = xc if rotation == "ccw" else -xc
        out[idx("vz")] = 0.0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )

    return out


def sod_shock_tube_1d(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    pos1: float = 0.5,
    rhol: float = 1.0,
    rhor: float = 0.125,
    vl: float = 0,
    vr: float = 0,
    pl: float = 1.0,
    pr: float = 0.1,
) -> ArrayLike:
    """
    Returns array for the Sod shock tube initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        pos1: Position of the discontinuity.
        rhol: Density on the left side of the discontinuity.
        rhor: Density on the right side of the discontinuity.
        vl: Velocity on the left side of the discontinuity.
        vr: Velocity on the right side of the discontinuity.
        pl: Pressure on the left side of the discontinuity.
        pr: Pressure on the right side of the discontinuity.
    """
    dims = parse_xyz(x, y, z)
    out = np.empty((len(idx.idxs), *x.shape))

    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition only works in 1D.")

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dims]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dims]
        out[idx("rho")] = np.where(r < pos1, rhol, rhor)
        out[idx("v" + dims)] = np.where(r < pos1, vl, vr)
        out[idx("v" + orth_dim1)] = 0
        out[idx("v" + orth_dim2)] = 0
        out[idx("P")] = np.where(r < pos1, pl, pr)
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
    rho0: float = 1,
    P0: float = 1,
    H0: float = 1,
) -> ArrayLike:
    """
    Returns array for the velocity ramp initial condition with a uniform density and
    pressure.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        rho0: Initial uniform density.
        P0: Initial uniform pressure.
        H0: Initial uniform velocity gradient.
    """
    dims = parse_xyz(x, y, z)
    out = np.empty((len(idx.idxs), *x.shape))

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
    gamma: float,
    h: float,
    rho0: float = 1,
    P0: float = 0,
) -> ArrayLike:
    """
    Returns array for the Sedov blast wave initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        gamma: Ratio of specific heats.
        h: Mesh size. Assumed to be the same in all dimensions.
        rho0: Initial uniform density.
        P0: Background pressure.
    """
    dims = parse_xyz(x, y, z)
    out = np.zeros((len(idx.idxs), *x.shape))

    inside_blast_cell = np.ones_like(x, dtype=bool)
    if "x" in dims:
        inside_blast_cell &= np.abs(x) < h
    if "y" in dims:
        inside_blast_cell &= np.abs(y) < h
    if "z" in dims:
        inside_blast_cell &= np.abs(z) < h

    E_blast_cell = 1 / (2 ** len(dims) * h ** len(dims))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        out[idx("rho")] = rho0
        out[idx("P")] = np.where(inside_blast_cell, (gamma - 1) * E_blast_cell, P0)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out
