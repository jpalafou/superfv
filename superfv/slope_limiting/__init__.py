from itertools import product
from typing import Any, Callable, Literal, Tuple

import numpy as np

from superfv.stencil import get_symmetric_slices
from superfv.tools.device_management import ArrayLike


def minmod(
    xp: Any, du_left: ArrayLike, du_right: ArrayLike, tol: float = 1e-16
) -> ArrayLike:
    """
    Args:
        xp (Any): `np` namespace.
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
        tol (float): Tolerance.
    Returns:
        ArrayLike: Minmod of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    ratio = du_right / np.where(
        du_left > 0,
        xp.where(du_left > tol, du_left, tol),
        xp.where(du_left < -tol, du_left, -tol),
    )
    ratio = xp.where(ratio < 1, ratio, 1)
    return xp.where(ratio > 0, ratio, 0) * du_left


def moncen(xp: Any, du_left: ArrayLike, du_right: ArrayLike) -> ArrayLike:
    """
    Args:
        xp (Any): `np` namespace.
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
    Returns:
        ArrayLike: Moncen of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    du_central = 0.5 * (du_left + du_right)
    slope = xp.minimum(xp.abs(2 * du_left), xp.abs(2 * du_right))
    slope = xp.sign(du_central) * xp.minimum(slope, xp.abs(du_central))
    return xp.where(du_left * du_right >= 0, slope, 0)


def compute_dmp(
    xp: Any, arr: ArrayLike, dims: str, include_corners: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes the discrete maximum principle (DMP) for an array along the 2nd, 3rd,
    and/or 4th axes.

    Args:
        xp (Any): `np` namespace.
        arr (ArrayLike): Array. First axis is assumed to be variable axis. Has shape
            (nvars, nx, ny, nz, ...).
        dims (str): Dimension to check. Must be a subset of "xyz".
        include_corners (bool): Whether to include corners.
    Returns:
        Tuple[ArrayLike, ArrayLike]: Minimum and maximum values of the array along the
            specified dimension.
    """
    dim_map = {"x": 1, "y": 2, "z": 3}
    active_dims = [dim_map[d] for d in dims]

    slices = [slice(None)] * arr.ndim
    for d in active_dims:
        slices[d] = slice(1, -1)
    dmp_min = arr[tuple(slices)].copy()
    dmp_max = arr[tuple(slices)].copy()

    for offsets in product([-1, 0, 1], repeat=len(active_dims)):
        if (
            not include_corners and all(offsets) and len(active_dims) > 1
        ):  # skip the corners
            continue
        if not any(offsets):  # skip the center
            continue
        slices = [slice(None)] * arr.ndim
        for d, offset in zip(active_dims, offsets):
            slices[d] = {-1: slice(2, None), 0: slice(1, -1), 1: slice(None, -2)}[
                offset
            ]
        dmp_min[...] = xp.minimum(dmp_min, arr[tuple(slices)])
        dmp_max[...] = xp.maximum(dmp_max, arr[tuple(slices)])

    return dmp_min, dmp_max


def muscl(
    xp: Any,
    averages: ArrayLike,
    dim: Literal["x", "y", "z"],
    slope_limiter: Callable[[Any, ArrayLike, ArrayLike], ArrayLike],
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Computes the MUSCL reconstruction of an array along the 2nd, 3rd, and/or 4th axes.

    Args:
        xp (Any): `np` namespace.
        averages (ArrayLike): Array. First axis is assumed to be variable axis. Has
            shape (nvars, nx, ny, nz).
        dim (str): Dimension to check. Must be a subset of "xyz".
        slope_limiter (Callable[[ArrayLike, ArrayLike], ArrayLike]): Slope
            limiter. Must take two arguments (left and right differences) and
            return an array of the same shape.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Left and right face values. Each has shape
            (nvars, <=nx, <=ny, <=nz, 1) where the axis corresponding to `dim` is
            shortened by 2 (1 on each side).
    """
    l_slc, c_slc, r_slc = get_symmetric_slices(
        ndim=4, nslices=3, axis="xyz".index(dim) + 1
    )
    l_daverages = averages[c_slc] - averages[l_slc]
    r_daverages = averages[r_slc] - averages[c_slc]
    limited_daverages = slope_limiter(xp, l_daverages, r_daverages)
    left_face = averages[c_slc] - 0.5 * limited_daverages
    right_face = averages[c_slc] + 0.5 * limited_daverages
    return left_face[..., np.newaxis], right_face[..., np.newaxis]
