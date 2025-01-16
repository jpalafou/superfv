from typing import Tuple

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
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    bounds: Tuple[float, float] = (0, 1),
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
        vx (float): x-component of the velocity.
        vy (float): y-component of the velocity.
        vz (float): z-component of the velocity.
        bounds (Tuple[float, float]): Bounds of the sinusoidal function.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz"} - _slc.var_names == set():
        # advection case
        out = np.zeros((4, *x.shape))
        r = int("x" in dims) * x + int("y" in dims) * y + int("z" in dims) * z
        out[_slc("rho")] = (bounds[1] - bounds[0]) * np.sin(2 * np.pi * r) + bounds[0]
        out[_slc("vx")] = vx
        out[_slc("vy")] = vy
        out[_slc("vz")] = vz
    else:
        raise NotImplementedError(
            f"Initial condition not implemented for variables: {_slc.var_names}. "
            "Supported variables: {'u', 'vx', 'vy', 'vz'}."
        )
    return out


def square(
    array_slicer: ArraySlicer,
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    vx: float = 0,
    vy: float = 0,
    vz: float = 0,
    bounds: Tuple[float, float] = (0, 1),
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
        vx (float): x-component of the velocity.
        vy (float): y-component of the velocity.
        vz (float): z-component of the velocity.
        bounds (Tuple[float, float]): Bounds of the square function.
    """
    _slc = array_slicer
    dims = parse_xyz(x, y, z)

    # Validate variables in ArraySlicer
    if {"rho", "vx", "vy", "vz"} - _slc.var_names == set():
        # advection case
        out = np.zeros((4, *x.shape))
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
    return out
