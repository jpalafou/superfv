from functools import lru_cache
from itertools import product
from types import ModuleType
from typing import List, Literal, Tuple, cast

from superfv.axes import DIM_TO_AXIS
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices


def gather_neighbor_slices(
    active_dims: Tuple[Literal["x", "y", "z"], ...], include_corners: bool
) -> List[Tuple[slice, ...]]:
    """
    Returns a list of slice objects for gathering neighbors in up to 3 dimensions with
    the center slice as the first element.

    Args:
        active_dims (Tuple[Literal["x", "y", "z"], ...]): Active dimensions for
        interpolation.
        include_corners (bool): Whether to include corner neighbors.

    Returns:
        List[Tuple[slice, ...]]: List of slice objects for gathering neighbors.
    """
    return _gather_neighbor_slices(active_dims, include_corners)


@lru_cache(maxsize=None)
def _gather_neighbor_slices(
    active_dims: Tuple[Literal["x", "y", "z"], ...], include_corners: bool
) -> List[Tuple[slice, ...]]:
    ndim = len(active_dims)
    axes = tuple(DIM_TO_AXIS[dim] for dim in active_dims)

    # gather all slices excluding the center
    all_slices: List[Tuple[slice, ...]] = []
    if include_corners:
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

    # insert inner slices in beginning
    inner_slice = merge_slices(*[crop(i, (1, -1), ndim=4) for i in axes])
    all_slices.insert(0, inner_slice)

    return all_slices


def compute_dmp(
    xp: ModuleType,
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
        xp: `np` namespace.
        arr: Array. First axis is assumed to be variable axis. Has shape
            (nvars, nx, ny, nz).
        active_dims: Dimensions to check. Must be a subset of ("x", "y", "z").
        out: Output array to store the results. Should have shape
            (nvars, nx, ny, nz, >=2). The DMP min will be stored in the last axis at
            index 0, and the DMP max at index 1.
        include_corners: Whether to include corners.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    all_slices = gather_neighbor_slices(active_dims, include_corners)
    inner_slice = all_slices[0]

    # stack views of neighbors
    all_views = [arr[slc] for slc in all_slices]
    stacked = xp.stack(all_views, axis=0)

    # compute min an max
    out[insert_slice(inner_slice, 4, 0)] = xp.min(stacked, axis=0)
    out[insert_slice(inner_slice, 4, 1)] = xp.max(stacked, axis=0)

    # return inner slice
    modified = cast(Tuple[slice, ...], insert_slice(inner_slice, 4, slice(None, 2)))
    return modified


def compute_vis(
    xp: ModuleType, dmp: ArrayLike, rtol: float, atol: float, *, out: ArrayLike
):
    """
    Compute a boolean array indicating where the local DMP spread is significant and
    should be visualized. A cell is flagged True where:

    |dmp_max - dmp_min| > atol + rtol * max(|dmp_max|, |dmp_min|)

    Args:
        xp: `np` namespace.
        dmp: Array containing the discrete maximum principle values. Has shape
            (nvars, nx, ny, nz, >=2), where the last axis contains the min and max
            values.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        out: Output boolean array to store the results. Should have shape
            (nvars, nx, ny, nz).
    """
    m = dmp[..., 0]
    M = dmp[..., 1]
    out[...] = M - m > atol + rtol * xp.maximum(xp.abs(m), xp.abs(M))
