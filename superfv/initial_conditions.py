from types import ModuleType
from typing import Literal, Optional, Tuple

import numpy as np

from .mesh import xyz_tup
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap


def parse_xyz(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> Tuple[Literal["x", "y", "z"], ...]:
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
    return tuple([dim for dim, size in zip(xyz_tup, x.shape) if size > 1])


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
        out[idx("rho")] = (bounds[1] - bounds[0]) * (
            0.5 * xp.sin(2 * np.pi * r) + 0.5
        ) + bounds[0]
        out[idx("vx")] = vx
        out[idx("vy")] = vy
        out[idx("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz'}."
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
            "Required variables: {'rho', 'vx', 'vy', 'vz'}."
        )
    if "P" in idx.var_names:
        out[idx("P")] = P
    return out


def composite(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: float = 0.0,
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    P: float = 1,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the 1D composite profile.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Time variable. Unused.
        vx: Uniform velocity in the x-direction.
        vy: Uniform velocity in the y-direction.
        vz: Uniform velocity in the z-direction.
        P: Pressure.
        xp: NumPy namespace module (e.g., `np` or `cupy`).

    Returns:
        ArrayLike: Array with the initial conditions for the hydro variables.
    """
    dims = parse_xyz(x, y, z)
    if len(dims) > 1:
        raise ValueError("Composite profile only defined in 1D.")
    dim = dims[0]

    out = xp.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz"} <= idx.var_names:
        # advection case
        r = {"x": x, "y": y, "z": z}[dim]
        u = xp.zeros_like(r)

        gauss_part = (
            1
            / 6
            * (
                xp.exp(-xp.log(2) / 36 / 0.0025**2 * (r - 0.0025 - 0.15) ** 2)
                + xp.exp(-xp.log(2) / 36 / 0.0025**2 * (r + 0.0025 - 0.15) ** 2)
                + 4 * xp.exp(-xp.log(2) / 36 / 0.0025**2 * (r - 0.15) ** 2)
            )
        )
        square_part = 0.75
        triangle_part = 1 - xp.abs(20 * (r - 0.55))
        ellipse_part = (
            1
            / 6
            * (
                xp.sqrt(xp.maximum(1 - (20 * (r - 0.75 - 0.0025)) ** 2, 0))
                + xp.sqrt(xp.maximum(1 - (20 * (r - 0.75 + 0.0025)) ** 2, 0))
                + 4 * xp.sqrt(xp.maximum(1 - (20 * (r - 0.75)) ** 2, 0))
            )
        )

        u = xp.where(xp.logical_and(r >= 0.1, r <= 0.2), gauss_part, u)
        u = xp.where(xp.logical_and(r >= 0.3, r <= 0.4), square_part, u)
        u = xp.where(xp.logical_and(r >= 0.5, r <= 0.6), triangle_part, u)
        u = xp.where(xp.logical_and(r >= 0.7, r <= 0.8), ellipse_part, u)

        out[idx("rho")] = u
        out[idx("vx")] = vx
        out[idx("vy")] = vy
        out[idx("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz'}."
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
            "Required variables: {'rho', 'vx', 'vy', 'vz'}."
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
    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition is only defined in 1D.")
    dim1 = dims[0]

    out = xp.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim1]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dim1]
        out[idx("rho")] = xp.where(r < pos1, rhol, rhor)
        out[idx("v" + dim1)] = xp.where(r < pos1, vl, vr)
        out[idx("v" + orth_dim1)] = 0
        out[idx("v" + orth_dim2)] = 0
        out[idx("P")] = xp.where(r < pos1, pl, pr)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
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
    if len(dims) != 1:
        raise ValueError("Velocity ramp initial condition is only defined in 1D.")
    dim1 = dims[0]

    out = xp.empty((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim1]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dim1]
        out[idx("rho")] = rho0
        out[idx("v" + dim1)] = H0 * (r - 0.5)
        out[idx("v" + orth_dim1)] = 0
        out[idx("v" + orth_dim2)] = 0
        out[idx("P")] = P0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
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
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def toro1(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Toro 1 shock tube initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    if len(dims) != 1:
        raise ValueError("Toro initial condition is only defined in 1D.")
    dim = dims[0]

    out = xp.zeros((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim]

        out[idx("rho")] = xp.where(r < 0.3, 1.0, 0.125)
        out[idx("v" + dim)] = xp.where(r < 0.3, 0.75, 0.0)
        out[idx("P")] = xp.where(r < 0.3, 1.0, 0.1)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def toro2(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Toro 2 shock tube initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    if len(dims) != 1:
        raise ValueError("Toro initial condition is only defined in 1D.")
    dim = dims[0]

    out = xp.zeros((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim]

        out[idx("rho")] = 1
        out[idx("v" + dim)] = xp.where(r < 0.5, -2, 2)
        out[idx("P")] = 0.4
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def toro3(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Toro 3 shock tube initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    if len(dims) != 1:
        raise ValueError("Toro initial condition is only defined in 1D.")
    dim = dims[0]

    out = xp.zeros((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim]

        out[idx("rho")] = 1
        out[idx("P")] = xp.where(r < 0.5, 1000, 0.01)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def shu_osher(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Shu-Osher problem initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    if len(dims) != 1:
        raise ValueError("Shu-Osher initial condition is only defined in 1D.")
    dim = dims[0]

    out = xp.zeros((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim]

        density_wave = 1 + 0.2 * xp.sin(2 * np.pi * 8 * r)
        out[idx("rho")] = xp.where(r < 0.125, 3.857143, density_wave)
        out[idx("v" + dim)] = xp.where(r < 0.125, 2.629369, 0)
        out[idx("P")] = xp.where(r < 0.125, 10.33333, 1)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def interacting_blast_wave_1d(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the interacting blast wave initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    dims = parse_xyz(x, y, z)
    if len(dims) != 1:
        raise ValueError("Toro initial condition is only defined in 1D.")
    dim = dims[0]

    out = xp.zeros((len(idx.idxs), *x.shape))

    # Validate variables in VariableIndexMap
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dim]

        out[idx("rho")] = 1
        out[idx("P")] = xp.where(r < 0.1, 1000, np.where(r < 0.9, 0.01, 100))
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {idx.var_names}. "
            "Required variables: {'rho', 'vx', 'vy', 'vz', 'P'}."
        )
    return out


def kelvin_helmholtz_2d(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the Kelvin-Helmholtz instability initial condition.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).
    """
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names:
        raise ValueError(
            "Kelvin-Helmholtz initial condition requires all hydro variables."
        )

    dims = parse_xyz(x, y, z)
    if len(dims) != 2:
        raise ValueError("Kelvin-Helmholtz initial condition is only defined in 2D.")
    dim1 = dims[0]
    dim2 = dims[1]

    d1 = {"x": x, "y": y, "z": z}[dim1]
    d2 = {"x": x, "y": y, "z": z}[dim2]

    inner_region = xp.logical_and(d2 > 0.25, d2 < 0.75)

    w0 = 0.1
    sigma = 0.05 * np.sqrt(2)

    rho = xp.where(inner_region, 2.0, 1.0)
    v1 = xp.where(inner_region, 0.5, -0.5)
    v2 = (
        w0
        * xp.sin(4 * np.pi * d1)
        * (
            xp.exp(-((d2 - 0.25) ** 2) / ((2 * sigma) ** 2))
            + xp.exp(-((d2 - 0.75) ** 2) / ((2 * sigma) ** 2))
        )
    )
    P = 2.5

    out = xp.zeros((len(idx.idxs), *x.shape))

    out[idx("rho")] = rho
    out[idx("v" + dim1)] = v1
    out[idx("v" + dim2)] = v2
    out[idx("P")] = P

    return out


def double_mach_reflection(
    idx: VariableIndexMap,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    t: Optional[float] = None,
    *,
    xp: ModuleType,
) -> ArrayLike:
    """
    Returns array for the double Mach reflection initial condition on the domain [0,4]
    in the first active dimension (x, for example) and [0,1] in the second active
    dimension (y, for example).

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        x: x-coordinate array. Has shape (nx, ny, nz).
        y: y-coordinate array. Has shape (nx, ny, nz).
        z: z-coordinate array. Has shape (nx, ny, nz).
        t: Optional time variable.
        xp: NumPy namespace module (e.g., `np` or `cupy`).

    Returns:
        ArrayLike: Array with the initial conditions for the hydro variables.
    """
    if {"rho", "vx", "vy", "vz", "P"} - idx.var_names:
        raise ValueError(
            "Kelvin-Helmholtz initial condition requires all hydro variables."
        )

    dims = parse_xyz(x, y, z)
    if set(dims) != {"x", "y"}:
        raise ValueError(
            "Double Mach reflection initial condition is only defined in 2D for x and y."
        )

    dx = x - y / np.tan(np.pi / 3)

    rho = xp.where(dx < 1 / 6, 8.0, 1.4)
    vx = xp.where(dx < 1 / 6, 7.145, 0.0)
    vy = xp.where(dx < 1 / 6, -4.125, 0.0)
    P = xp.where(dx < 1 / 6, 116.5, 1.0)

    out = xp.zeros((len(idx.idxs), *x.shape))

    out[idx("rho")] = rho
    out[idx("vx")] = vx
    out[idx("vy")] = vy
    out[idx("P")] = P

    return out
