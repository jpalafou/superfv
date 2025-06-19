from functools import lru_cache
from typing import Any, List, Literal, Tuple, Union, cast

import numpy as np
from stencilpal import conservative_interpolation_stencil, uniform_quadrature
from stencilpal.stencil import Stencil

from .tools.array_management import ArrayLike, crop


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
def conservative_interpolation_weights(
    p: int, x: Union[Literal["l", "c", "r"], int, float]
) -> np.ndarray:
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
            - If `x` is a string or integer, the returned array has an integer data type,
            with elements representing the numerators of the rational weights after a
            common denominator is applied. In this case, the weights do not necessarily
            sum to 1.
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
        return numerators / denominator
    else:
        return cast(np.ndarray, stencil.w)


@lru_cache(maxsize=None)
def uniform_quadrature_weights(p: int) -> np.ndarray:
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
        return numerators / denominator
    else:
        return cast(np.ndarray, stencil.w)


@lru_cache(maxsize=None)
def get_symmetric_slices(
    ndim: int, nslices: int, axis: int
) -> List[Union[slice, int, np.ndarray, Tuple[Union[slice, int, np.ndarray], ...]]]:
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
