from itertools import product
from typing import Tuple

import numpy as np

from superfv.tools.array_management import ArrayLike


def minmod(du_left: ArrayLike, du_right: ArrayLike, tol: float = 1e-16) -> ArrayLike:
    """
    Args:
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
        tol (float): Tolerance.
    Returns:
        ArrayLike: Minmod of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    ratio = du_right / np.where(
        du_left > 0,
        np.where(du_left > tol, du_left, tol),
        np.where(du_left < -tol, du_left, -tol),
    )
    ratio = np.where(ratio < 1, ratio, 1)
    return np.where(ratio > 0, ratio, 0) * du_left


def moncen(du_left: ArrayLike, du_right: ArrayLike) -> ArrayLike:
    """
    Args:
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
    Returns:
        ArrayLike: Moncen of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)


def compute_dmp(
    arr: ArrayLike, dim: str, include_corners: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes the discrete maximum principle (DMP) for an array along the 2nd, 3rd,
    and/or 4th axes.

    Args:
        arr (ArrayLike): Array. First axis is assumed to be variable axis. Has shape
            (nvars, nx, ny, nz, ...).
        dim (str): Dimension to check. Must be a subset of "xyz".
        include_corners (bool): Whether to include corners.
    Returns:
        Tuple[ArrayLike, ArrayLike]: Minimum and maximum values of the array along the
            specified dimension.
    """
    dim_map = {"x": 1, "y": 2, "z": 3}
    active_dims = [dim_map[d] for d in dim]

    _slice = [slice(None)] * arr.ndim
    for d in active_dims:
        _slice[d] = slice(1, -1)
    dmp_min = arr[tuple(_slice)].copy()
    dmp_max = arr[tuple(_slice)].copy()

    for offsets in product([-1, 0, 1], repeat=len(active_dims)):
        if not include_corners and all(offsets):  # skip the corners
            continue
        if not any(offsets):  # skip the center
            continue
        for d, offset in zip(active_dims, offsets):
            _slice[d] = {-1: slice(2, None), 0: slice(1, -1), 1: slice(None, -2)}[
                offset
            ]
        dmp_min[...] = np.minimum(dmp_min, arr[tuple(_slice)])
        dmp_max[...] = np.maximum(dmp_max, arr[tuple(_slice)])

    return dmp_min, dmp_max
