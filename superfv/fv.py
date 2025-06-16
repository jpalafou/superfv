from functools import lru_cache
from typing import Callable, Tuple

import numpy as np

from .tools.array_management import ArrayLike


def _scaled_gauss_legendre_points_and_weights(p: int) -> Tuple[ArrayLike, ArrayLike]:
    unscaled_points, unscaled_weights = np.polynomial.legendre.leggauss(
        -(-(p + 1) // 2)
    )
    scaling = np.sum(unscaled_weights)
    return unscaled_points / scaling, unscaled_weights / scaling


@lru_cache(maxsize=None)
def _gauss_legendre_mesh(
    px: int, py: int, pz: int
) -> Tuple[
    Tuple[ArrayLike, ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike, ArrayLike]
]:
    x_pts, x_wts = _scaled_gauss_legendre_points_and_weights(px)
    y_pts, y_wts = _scaled_gauss_legendre_points_and_weights(py)
    z_pts, z_wts = _scaled_gauss_legendre_points_and_weights(pz)
    xp, yp, zp = np.meshgrid(x_pts, y_pts, z_pts, indexing="ij")
    xw, yw, zw = np.meshgrid(x_wts, y_wts, z_wts, indexing="ij")
    return (xp, yp, zp), (xw, yw, zw)


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
    hx, hy, hz = h
    px, py, pz = p
    (xp, yp, zp), (xw, yw, zw) = _gauss_legendre_mesh(px, py, pz)
    na = np.newaxis

    x_eval = x[..., na, na, na] + xp[na, na, na, :, :, :] * hx
    y_eval = y[..., na, na, na] + yp[na, na, na, :, :, :] * hy
    z_eval = z[..., na, na, na] + zp[na, na, na, :, :, :] * hz

    # Evaluate f on the whole grid at once (result shape: (px, py, pz, nx, ny, nz))
    vals = f(x_eval, y_eval, z_eval)

    weights = xw * yw * zw
    weights = weights[na, na, na, :, :, :]

    # Weighted sum over quadrature points
    return np.sum(weights * vals, axis=(3, 4, 5))
