from itertools import product
from typing import Any, Callable, Literal, Tuple

import numpy as np

from superfv.fv import DIM_TO_AXIS
from superfv.stencil import get_symmetric_slices
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices


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
    xp: Any,
    arr: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    include_corners: bool,
) -> Tuple[slice, ...]:
    """
    Compute the minimum and maximum values of each point and its neighbors along
    specified dimensions.

    Args:
        xp (Any): `np` namespace.
        arr (ArrayLike): Array. First axis is assumed to be variable axis. Has shape
            (nvars, nx, ny, nz).
        active_dims (Tuple[Literal["x", "y", "z"], ...]): Dimensions to check.
        out (ArrayLike): Output array to store the results. Should have shape
            (nvars, nx, ny, nz, >=2).
        include_corners (bool): Whether to include corners.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    ndim = len(active_dims)
    axes = tuple(DIM_TO_AXIS[dim] for dim in active_dims)

    # gather slices
    if include_corners:
        all_slices = []
        for offset in product([-1, 0, 1], repeat=ndim):
            if offset == (0,) * ndim:
                continue
            all_slices.append(
                merge_slices(
                    *[
                        crop(i, (1 + shift, -1 + shift), ndim=4)
                        for i, shift in zip(axes, offset)
                    ]
                )
            )
    else:
        all_slices = []
        for ax in axes:
            for shift in [-1, 1]:
                all_slices.append(
                    merge_slices(
                        *[
                            crop(
                                i,
                                (1 + shift, -1 + shift) if ax == i else (1, -1),
                                ndim=4,
                            )
                            for i in axes
                        ]
                    )
                )

    # store inner slices first
    inner_slice = merge_slices(*[crop(i, (1, -1), ndim=4) for i in axes])
    all_slices.insert(0, inner_slice)

    # stack views of neighbors
    all_views = [arr[slc] for slc in all_slices]
    stacked = xp.stack(all_views, axis=0)

    # compute min an max
    out[inner_slice + (0,)] = xp.min(stacked, axis=0)
    out[inner_slice + (1,)] = xp.max(stacked, axis=0)

    # return inner slice
    return inner_slice + (slice(None, 2),)


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
