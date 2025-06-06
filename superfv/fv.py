from typing import Callable, Tuple

import numpy as np

from .tools.array_management import ArrayLike


def fv_average(
    f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    h: Tuple[float, float, float],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> ArrayLike:
    """
    Compute finite volume average of f over 3D domain.

    Args:
        f: Function at which to evaluate quadrature points.
        x: x-coordinates, has shape (nx, ny, nz).
        y: y-coordinates, has shape (nx, ny, nz).
        z: z-coordinates, has shape (nx, ny, nz).
        h: Mesh spacings (hx, hy, hz).
        p: Polynomial degree of quadrature rule in each dimension.

    Returns:
        ArrayLike: Finite volume average.
    """
    # mesh spacings
    hx, hy, hz = h

    # quadrature points and weights
    unscaled_points_and_weights = [
        np.polynomial.legendre.leggauss(-(-(_p + 1) // 2)) for _p in p
    ]

    # rescale to have sum of exactly 1
    error_factors = [1 / np.sum(weights) for _, weights in unscaled_points_and_weights]
    points_and_weights = [
        [points * factor, weights * factor]
        for (points, weights), factor in zip(unscaled_points_and_weights, error_factors)
    ]

    # find cell averages
    out = np.zeros_like(f(x, y, z))
    for xp, xw in zip(*points_and_weights[0]):
        for yp, yw in zip(*points_and_weights[1]):
            for zp, zw in zip(*points_and_weights[2]):
                weight = xw * yw * zw
                out += weight * f(x + xp * hx, y + yp * hy, z + zp * hz)

    return out
