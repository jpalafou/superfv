from functools import lru_cache
from typing import List, Literal, Tuple, Union

import numpy as np
from stencilpal import conservative_interpolation_stencil, uniform_quadrature
from stencilpal.stencil import Stencil

from .tools.array_management import ArrayLike, ArraySlicer


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
        stencil (Stencil): The stencil to resize.
        target_size (int): The desired size of the stencil.
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
        x (Union[str, int, float]): interpolation point on the interval [-1, 1] (the
            cell with center 0).
            - "l": alias for the leftmost point of the cell (-1).
            - "c": alias for the center of the cell (0).
            - "r": alias for the rightmost point of the cell (1).

    Returns:
        np.ndarray: The weights of the conservative interpolation stencil.
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
        return stencil.w


@lru_cache(maxsize=None)
def uniform_quadrature_weights(p: int) -> np.ndarray:
    """
    Returns the weights of the uniform quadrature stencil for a given polynomial
    degree.

    Args:
        p: The polynomial degree.

    Returns:
        np.ndarray: The weights of the uniform quadrature stencil.
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
        return stencil.w


@lru_cache(maxsize=None)
def get_symmetric_slices(
    ndim: int, nslices: int, axis: int
) -> List[Union[slice, int, np.ndarray, Tuple[Union[slice, int, np.ndarray], ...]]]:
    """
    Returns a list of slices that divide an array into `nslices` symmetric slices along
    the given axis.

    Args:
        ndim (int): The number of dimensions of the array.
        nslices (int): The number of slices to create.
        axis (int): The axis along which to slice.

    Returns:
        List[Union[slice, int, np.ndarray, Tuple[Union[slice, int, np.ndarray], ...]]]:
            A list of slices that divide an array into `nslices`.
    """
    slicer = ArraySlicer({}, ndim)
    return [slicer(axis=axis, cut=(i, -(nslices - 1) + i)) for i in range(nslices)]


def stencil_sweep(y: ArrayLike, stencil_weights: np.ndarray, axis: int) -> ArrayLike:
    """
    Perform a stencil sweep on a field.

    Args:
        y (ArrayLike): The field to sweep. Must have shape (nvars, nx, ny, nz, ...).
        stencil_weights (np.ndarray): The weights of the stencil to apply. Has shape
            (nweights,).
        axis (int): The axis along which to sweep.

    Returns:
        ArrayLike: The field after the stencil sweep. Has shape
            (nvars, <= nx, <= ny, <= nz, ...).
    """
    # reshape weights to broadcast with y
    weights = stencil_weights.T.reshape((len(stencil_weights),) + (1,) * y.ndim)

    # get slices
    slices = get_symmetric_slices(y.ndim, len(stencil_weights), axis)

    # initialize output array
    out = np.zeros_like(y[slices[0]])

    # sweep
    for w, s in zip(weights, slices):
        out += w * y[s]

    return out


@lru_cache(maxsize=None)
def gauss_legendre(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Gauss-Legendre quadrature points and weights for a given polynomial
    degree.

    Args:
        p: The polynomial degree.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The Gauss-Legendre quadrature points and weights.
    """
    return np.polynomial.legendre.leggauss(p + 1)
