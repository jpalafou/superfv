from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .device_management import ArrayLike

IndexLike = Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]
SliceBounds = Tuple[Union[None, int], Union[None, int]]


def crop(
    axis: Union[int, Tuple[int, ...]],
    cut: Tuple[Optional[int], Optional[int]],
    step: Optional[int] = None,
    ndim: Optional[int] = None,
) -> Tuple[slice, ...]:
    """
    Create an array slice for a given axis or axes.

    Args:
        axis: Axis or axes along which to slice.
        cut: Start and stop indices for the slice.
        step: Step size for the slice. Default is None.
        ndim: Number of dimensions of the array. If None, uses the maximum axis value
            to determine the number of dimensions.

    Returns:
        Tuple of slices for the given axis or axes.
    """
    return _crop(axis, cut, step, ndim)


@lru_cache(maxsize=None)
def _crop(
    axis: Union[int, Tuple[int, ...]],
    cut: Tuple[Optional[int], Optional[int]],
    step: Optional[int] = None,
    ndim: Optional[int] = None,
) -> Tuple[slice, ...]:
    if isinstance(axis, int):
        axis = (axis,)
    rank = ndim if ndim is not None else max(axis) + 1
    out = [slice(None)] * rank
    for ax in axis:
        if ax < 0 or ax >= rank:
            raise ValueError(f"Axis {ax} is out of bounds for array with {len(out)} dimensions.")
        out[ax] = slice(cut[0] or None, cut[1] or None, step)
    return tuple(out)


def crop_to_center(
    arr: ArrayLike,
    target_shape: Tuple[int, ...],
    ignore_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> ArrayLike:
    """
    Crop an array to a target shape by removing an equal amount from both ends along each axis.

    Args:
        arr: The input array to be cropped.
        target_shape: The desired shape of the output array.
        ignore_axes: Axes to ignore when cropping.

    Returns:
        A cropped version of the input array with the target shape.
    """
    slices = _crop_to_center(arr.shape, target_shape, ignore_axes)
    return arr[slices]


@lru_cache(maxsize=None)
def _crop_to_center(
    in_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    ignore_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[slice, ...]:
    """
    Create an array slice to crop an input array to a target shape by removing an equal
    amount from both ends along each axis.

    Args:
        in_shape: The shape of the input array.
        target_shape: The desired shape of the output array.
        ignore_axes: Axes to ignore when cropping. If None, all axes are considered. If
            an int, it is treated as a single axis. If a tuple, it contains multiple
            axes to ignore.

    Returns:
        Tuple of slices that can be used to crop the input array to the target shape.
    """
    out = [slice(None)] * len(in_shape)
    if ignore_axes is None:
        ignore_axes = tuple()
    elif isinstance(ignore_axes, int):
        ignore_axes = (ignore_axes,)
    for i, (dim_length, target_length) in enumerate(zip(in_shape, target_shape)):
        if i in ignore_axes:
            out[i] = slice(None)
            continue
        if target_length > dim_length:
            raise ValueError(
                f"Target shape {target_shape} must be less than or equal to the input array's shape {in_shape} in all dimensions."
            )
        elif (dim_length - target_length) % 2 == 0:
            margin = (dim_length - target_length) // 2
            out[i] = slice(margin or None, -margin or None)
        else:
            raise ValueError(f"Cannot evenly crop dimension from {dim_length} to {target_length}.")
    return tuple(out)


def intersection_shape(*args: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the intersection of the shapes of multiple arrays.

    Args:
        *args: Tuple of shapes.

    Returns:
        Intersection shape.
    """
    return tuple(min(s) for s in zip(*args))


def merge_indices(
    *slices: Union[int, slice, Tuple[int, ...], List[int], np.ndarray[Any, np.dtype[np.int_]]],
    as_array: bool = False,
) -> IndexLike:
    """
    Merge indices, slices, or sequences of integers into a union of indices as a slice
    if they form a contiguous range, or as a numpy array otherwise.

    Args:
        *slices: Indices, slices, or sequences of integers to merge.
        as_array: If True, return a numpy array instead of a slice, even if the indices
            are contiguous.

    Returns:
        Merged slice if merged indices form a contiguous range, otherwise a numpy array
            of indices.
    """
    idxs = []
    for s in slices:
        if isinstance(s, int):
            if s < 0:
                raise ValueError(f"Index must be non-negative, got {s}.")
            idxs.append(s)
        elif isinstance(s, slice):
            if s.start is not None and s.start < 0:
                raise ValueError(f"Slice start cannot be negative, got {s.start}.")
            if s.stop is None or s.stop < 0:
                raise ValueError("Slice stop must be specified and non-negative.")
            if s.step is not None and s.step != 1:
                raise ValueError("Only slices with step=1 are supported.")
            idxs.extend(range(s.start if s.start is not None else 0, s.stop))
        elif isinstance(s, tuple) or isinstance(s, list) or isinstance(s, np.ndarray):
            if len(s) > 0:
                s_arr = np.asarray(s)
                if s_arr.ndim != 1:
                    raise ValueError("Only 1D arrays/lists of integers are supported.")
                if not np.issubdtype(s_arr.dtype, np.integer):
                    raise ValueError(
                        f"Expected 1D sequence of integers, got array with dtype {s_arr.dtype}."
                    )
                idxs.extend([int(i) for i in s_arr])
            else:
                s_arr = np.array([], dtype=np.int_)
        else:
            raise TypeError(
                f"Unsupported type {type(s)} for merging indices. Expected int, slice, or numpy array."
            )
    idxs = sorted(set(idxs))
    if not as_array:
        if len(idxs) == 0:
            return slice(0, 0)
        if idxs == list(range(idxs[0], idxs[-1] + 1)):
            return slice(idxs[0], idxs[-1] + 1)
    return np.array(idxs, dtype=np.int_)


def merge_slices(*args: Tuple[slice, ...], union: bool = False) -> Tuple[slice, ...]:
    """
    Merge multiple N-dimensional slices into a single tuple of slices that covers the
    intersection or union of all input slices along each axis.

    Args:
        *args: Tuples of slices (e.g., for indexing a multi-dimensional array).
        union: If True, compute the union of the slices instead of the intersection.

    Returns:
        Tuple of slices that represent the merged slices along each axis.
    """
    return _merge_slices(*args, union=union)


@lru_cache(maxsize=None)
def _merge_slices(*args: Tuple[slice, ...], union: bool = False) -> Tuple[slice, ...]:
    n = max(len(s) for s in args)
    result = []
    for i in range(n):
        starts_raw = [s[i].start if i < len(s) else None for s in args]
        stops_raw = [s[i].stop if i < len(s) else None for s in args]

        starts = [x for x in starts_raw if x is not None]
        stops = [x for x in stops_raw if x is not None]

        if union:
            start = min(starts) if len(starts) == len(starts_raw) else None
            stop = max(stops) if len(stops) == len(stops_raw) else None
        else:
            start = max(starts) if starts else None
            stop = min(stops) if stops else None

        result.append(slice(start, stop))
    return tuple(result)


def replace_slice(
    slc: Tuple[Union[int, slice], ...], axis: int, new_slice: Union[int, slice]
) -> Tuple[Union[int, slice], ...]:
    """
    Adjust a slice tuple by replacing the slice at a specific axis with a new slice.

    Args:
        slc: Tuple of slices.
        axis: Axis index to adjust.
        new_slice: New slice/index to replace the existing one at the specified axis.

    Returns:
        A new tuple of slices with the specified axis replaced by the new slice.
    """
    return _replace_slice(slc, axis, new_slice)


@lru_cache(maxsize=None)
def _replace_slice(
    slc: Tuple[Union[int, slice], ...], axis: int, new_slice: slice
) -> Tuple[Union[int, slice], ...]:
    if axis < 0 or axis >= len(slc):
        raise IndexError(f"Axis {axis} is out of bounds for slice tuple of length {len(slc)}.")
    return slc[:axis] + (new_slice,) + slc[axis + 1 :]


def insert_slice(
    slc: Tuple[Union[int, slice], ...],
    axis: int,
    new_slice: Union[int, slice],
) -> Tuple[Union[int, slice], ...]:
    """
    Insert a new slice into a tuple of slices at a specified axis.

    Args:
        slc: Tuple of slices.
        axis: Axis index to assign new slice.
        new_slice: Slice to insert at the specified axis.

    Returns:
        A new tuple of slices with an additional slice inserted at the specified axis.
    """
    return _insert_slice(slc, axis, new_slice)


@lru_cache(maxsize=None)
def _insert_slice(
    slc: Tuple[Union[int, slice], ...], axis: int, new_slice: Union[int, slice]
) -> Tuple[Union[int, slice], ...]:
    if axis < 0 or axis > len(slc):
        raise IndexError(f"Axis {axis} is out of bounds for slice tuple of length {len(slc)}.")
    return slc[:axis] + (new_slice,) + slc[axis:]
