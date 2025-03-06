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
        f (Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]): Function to
            average.
        x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
        y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
        z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
        h (Tuple[float, float, float]): Mesh spacings (hx, hy, hz).
        p (Tuple[int, int, int]): Polynomial degree of quadrature rule in each
            dimension.

    Returns:
        ArrayLike: Finite volume average.
    """
    # mesh spacings
    hx, hy, hz = h

    # quadrature points and weights
    points_and_weights = [
        [q / 2 for q in np.polynomial.legendre.leggauss(-(-(_p + 1) // 2))] for _p in p
    ]

    # find cell averages
    out = np.zeros_like(f(x, y, z))
    for xp, xw in zip(*points_and_weights[0]):
        for yp, yw in zip(*points_and_weights[1]):
            for zp, zw in zip(*points_and_weights[2]):
                weight = xw * yw * zw
                out += weight * f(x + xp * hx, y + yp * hy, z + zp * hz)

    return out
