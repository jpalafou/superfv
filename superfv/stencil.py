from functools import lru_cache, wraps
from types import ModuleType
from typing import Any, Callable, List, Literal, Sequence, Tuple, Union, cast

import numpy as np
from stencilpal import conservative_interpolation_stencil, uniform_quadrature
from stencilpal.stencil import Stencil

from .tools.array_management import ArrayLike, crop

Coordinate = Union[int, float, Literal["l", "c", "r"]]


def canonicalize_interp_coord(
    func: Callable[[int, Coordinate], np.ndarray],
) -> Callable[[int, Coordinate], np.ndarray]:
    """
    Decorator for `conservative_interpolation_weights` to ensure that the
    interpolation coordinate `x` is always an integer if it is equivalent to an integer
    value.
    """

    @wraps(func)
    def wrapper(xp, p: int, x: Coordinate) -> np.ndarray:
        if isinstance(x, float) and x in (-1.0, 0.0, 1.0):
            x = int(x)
        return func(xp, p, x)

    return wrapper


@canonicalize_interp_coord
@lru_cache(maxsize=None)
def conservative_interpolation_weights(xp, p: int, x: Coordinate) -> np.ndarray:
    """
    Returns the weights of the conservative interpolation stencil for a given
    polynomial degree.

    Args:
        p: The polynomial degree.
        x: interpolation point on the interval [-1, 1] as a number or alias:
            - "l": alias for the leftmost point of the cell (-1).
            - "c": alias for the center of the cell (0).
            - "r": alias for the rightmost point of the cell (1).

    Returns:
        Array of weights of the conservative interpolation stencil.
            - If `x` is a string or integer, the stencil is returned with rational
            weights and the weights are the pure division of the numerators by the
            denominator.
            - If `x` is a float, the returned array has a float data type, and the weights
            are normalized to sum to 1.

    Notes:
        - Equivalent floating-point and integer values for the position `x` produce the
        same hash. As a result, the type of the cached output depends on which call is
        made first. For example, if `conservative_interpolation_weights(3, -1.0)` is
        called before `conservative_interpolation_weights(3, -1)`, the latter will
        return a float array, not an integer array.
    """
    stencil = conservative_interpolation_stencil(p, x)
    resize_stencil(stencil, stencil_size(p))
    if stencil.rational:
        numerators = stencil.asnumpy()
        denominator = np.sum(numerators)
        return xp.asarray(numerators / denominator)
    return xp.asarray(cast(np.ndarray, stencil.w))


@lru_cache(maxsize=None)
def stencil_size(p: int) -> int:
    """
    Returns the size of the stencil for a given polynomial degree.

    Args:
        p: The polynomial degree.
    """
    return -2 * (-p // 2) + 1


def resize_stencil(stencil: Stencil, target_size: int):
    """
    Resizes a stencil to the target size by expanding the stencil to the left and right
    with zeros.

    Args:
        stencil: Stencil object to resize.
        target_size: The desired size of the stencil.
    """
    while stencil.size < target_size:
        stencil.rescope(
            np.append(np.append(stencil.x[0] - 1, stencil.x), stencil.x[-1] + 1),
            inplace=True,
        )
        if stencil.size > target_size:
            raise ValueError(f"Failed to find stencil of size {target_size}.")


@lru_cache(maxsize=None)
def uniform_quadrature_weights(xp, p: int) -> np.ndarray:
    """
    Returns the weights of the uniform quadrature stencil for a given polynomial
    degree.

    Args:
        p: The polynomial degree.

    Returns:
        Array of weights of the uniform quadrature stencil.
            - The returned array has an integer data type, with elements representing
            the numerators of the rational weights after a common denominator is applied.
            In this case, the weights do not necessarily sum to 1.

    """
    stencil = uniform_quadrature(p)
    resize_stencil(stencil, stencil_size(p))
    if stencil.rational:
        numerators = stencil.asnumpy()
        denominator = np.sum(numerators)
        return xp.asarray(numerators / denominator)
    else:
        return xp.asarray(cast(np.ndarray, stencil.w))


def inplace_stencil_sweep(
    xp: ModuleType,
    arr: ArrayLike,
    stencil_weights: Union[Sequence[float], ArrayLike],
    axis: int,
    out: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Apply a symmetric stencil along a given axis and accumulate the result into the
    central region.

    Args:
        xp: Array namespace (e.g., `np` or `cupy`).
        arr: Input array of field values to apply the stencil to.
        stencil_weights: Sequence or array of stencil weights. Expected to have odd
            length.
        axis: Axis along which to apply the stencil.
        out: Array to store the result. Expected to have the same shape as `arr`.

    Returns:
        A slice object specifying the region of `out` that was modified.
    """
    slices = get_symmetric_slices(arr.ndim, len(stencil_weights), axis)
    modified = slices[len(slices) // 2]
    out[modified] = 0.0
    for w, s in zip(stencil_weights, slices):
        xp.add(out[modified], xp.multiply(arr[s], w), out=out[modified])
    return modified


def stencil_sweep(
    xp: Any, arr: ArrayLike, stencil_weights: np.ndarray, axis: int
) -> ArrayLike:
    """
    Perform a stencil sweep on a field.

    Args:
        xp: `np` namespace.
        y: Array of field values to sweep. Must have shape (nvars, nx, ny, nz, ...).
        stencil_weights: Array of stencil weights to apply. Has shape (nweights,).
        axis: The axis along which to sweep.

    Returns:
        Array of field values after the stencil sweep. Has shape
            (nvars, <= nx, <= ny, <= nz, ...).
    """
    # reshape weights to broadcast with y
    weights = stencil_weights.T.reshape((len(stencil_weights),) + (1,) * arr.ndim)

    # get slices
    slices = get_symmetric_slices(arr.ndim, len(stencil_weights), axis)

    # initialize output array
    out = xp.zeros_like(arr[slices[0]])

    # sweep
    for w, s in zip(weights, slices):
        xp.add(out, xp.multiply(arr[s], w), out)

    return out


@lru_cache(maxsize=None)
def get_symmetric_slices(ndim: int, nslices: int, axis: int) -> List[Tuple[slice, ...]]:
    """
    Returns a list of slices that divide an array into `nslices` symmetric slices along
    the given axis.

    Args:
        ndim: The number of dimensions of the array.
        nslices: The number of slices to create.
        axis: The axis along which to slice.

    Returns:
        A list of slices that divide an array into `nslices`.
    """
    return [crop(axis, (i, -(nslices - 1) + i), ndim=ndim) for i in range(nslices)]
