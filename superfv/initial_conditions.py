from typing import Literal, Tuple

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer


def parse_xyz(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> str:
    """
    Returns a string with the dimensions of the input arrays.

    Args:
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).

    Returns:
        str: String with the dimensions of the input arrays (some combination of "x",
            "y", and "z").
    """
    if not x.shape == y.shape == z.shape:
        raise ValueError("x, y, and z must have the same shape.")
    return "".join(dim for dim, size in zip("xyz", x.shape) if size > 1)


def sinus(
    array_slicer: ArraySlicer,
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
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        bounds (Tuple[float, float]): Bounds of the density sinusoidal function.
        vx (float): x-component of the velocity.
        vy (float): y-component of the velocity.
        vz (float): z-component of the velocity.
        P (float): Pressure.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)
    out = np.empty((len(_slc.idxs), *x.shape))

    # Validate variables in ArraySlicers
    if {"rho", "vx", "vy", "vz"} <= _slc.var_names:
        # advection case
        r = int("x" in dims) * x + int("y" in dims) * y + int("z" in dims) * z
        out[_slc("rho")] = (
            0.5 * (bounds[1] - bounds[0]) * np.sin(2 * np.pi * r) + 0.5 + bounds[0]
        )
        out[_slc("vx")] = vx
        out[_slc("vy")] = vy
        out[_slc("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    if "P" in _slc.var_names:
        out[_slc("P")] = P
    return out


def square(
    array_slicer: ArraySlicer,
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
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        bounds (Tuple[float, float]): Bounds of the density square function.
        vx (float): x-component of the velocity.
        vy (float): y-component of the velocity.
        vz (float): z-component of the velocity.
        P (float): Pressure.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)
    out = np.empty((len(_slc.idxs), *x.shape))

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz"} <= _slc.var_names:
        # advection case
        r = np.ones_like(x).astype(bool)
        if "x" in dims:
            r &= (x >= 0.25) & (x <= 0.75)
        if "y" in dims:
            r &= (y >= 0.25) & (y <= 0.75)
        if "z" in dims:
            r &= (z >= 0.25) & (z <= 0.75)
        r = r.astype(float)
        out[_slc("rho")] = (bounds[1] - bounds[0]) * r + bounds[0]
        out[_slc("vx")] = vx
        out[_slc("vy")] = vy
        out[_slc("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    if "P" in _slc.var_names:
        out[_slc("P")] = P
    return out


def slotted_disk(
    array_slicer: ArraySlicer,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    rho_min_max: Tuple[float, float] = (0.0, 1.0),
    rotation: Literal["cw", "ccw"] = "ccw",
) -> ArrayLike:
    """
    Returns array for the slotted disk initial condition.

    Args:
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        rho_min_max (Tuple[float, float]): Minimum and maximum values of the density.
        rotation (Literal["cw", "ccw"]): Rotation direction of the disk.
    """
    _slc = array_slicer
    out = np.empty((len(_slc.idxs), *x.shape))

    if {"rho", "vx", "vy", "vz"} <= _slc.var_names:
        # advection case
        xc, yc = x - 0.5, y - 0.5
        rsq = np.sqrt(xc**2 + (y - 0.75) ** 2)
        inside_disk = rsq < 0.15
        inside_disk &= np.logical_not(np.logical_and(np.abs(xc) < 0.025, y < 0.85))

        out = np.empty((4, *x.shape))
        out[_slc("rho")] = np.where(inside_disk, rho_min_max[1], rho_min_max[0])
        out[_slc("vx")] = -yc if rotation == "ccw" else yc
        out[_slc("vy")] = xc if rotation == "ccw" else -xc
        out[_slc("vz")] = 0.0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )

    return out


def sod_shock_tube_1d(
    array_slicer: ArraySlicer,
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
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        pos1 (float): Position of the discontinuity.
        rhol (float): Density on the left side of the discontinuity.
        rhor (float): Density on the right side of the discontinuity.
        vl (float): Velocity on the left side of the discontinuity.
        vr (float): Velocity on the right side of the discontinuity.
        pl (float): Pressure on the left side of the discontinuity.
        pr (float): Pressure on the right side of the discontinuity.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)
    out = np.empty((len(_slc.idxs), *x.shape))

    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition only works in 1D.")

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz", "P"} - _slc.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dims]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dims]
        out[_slc("rho")] = np.where(r < pos1, rhol, rhor)
        out[_slc("v" + dims)] = np.where(r < pos1, vl, vr)
        out[_slc("v" + orth_dim1)] = 0
        out[_slc("v" + orth_dim2)] = 0
        out[_slc("P")] = np.where(r < pos1, pl, pr)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out


def velocity_ramp(
    array_slicer: ArraySlicer,
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
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        rho0 (float): Initial uniform density.
        P0 (float): Initial uniform pressure.
        H0 (float): Initial uniform velocity gradient.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)
    out = np.empty((len(_slc.idxs), *x.shape))

    if len(dims) != 1:
        raise ValueError("Sod shock tube initial condition only works in 1D.")

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz", "P"} - _slc.var_names != {}:
        r = {"x": x, "y": y, "z": z}[dims]
        orth_dim1, orth_dim2 = [dim for dim in "xyz" if dim != dims]
        out[_slc("rho")] = rho0
        out[_slc("v" + dims)] = H0 * (r - 0.5)
        out[_slc("v" + orth_dim1)] = 0
        out[_slc("v" + orth_dim2)] = 0
        out[_slc("P")] = P0
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out


def sedov(
    array_slicer: ArraySlicer,
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
        array_slicer (ArraySlicer): Array slicer object. Defines the variables used in
            the initial condition.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        gamma (float): Ratio of specific heats.
        h (float): Mesh size. Assumed to be the same in all dimensions.
        rho0 (float): Initial uniform density.
        P0 (float): Background pressure.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)
    out = np.zeros((len(_slc.idxs), *x.shape))

    inside_blast_cell = np.ones_like(x, dtype=bool)
    if "x" in dims:
        inside_blast_cell &= np.abs(x) < h
    if "y" in dims:
        inside_blast_cell &= np.abs(y) < h
    if "z" in dims:
        inside_blast_cell &= np.abs(z) < h
    E_blast_cell = (0.5 / h) ** (len(dims))

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz", "P"} - _slc.var_names != {}:
        out[_slc("rho")] = rho0
        out[_slc("P")] = np.where(inside_blast_cell, (gamma - 1) * E_blast_cell, P0)
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out
