from functools import lru_cache
from typing import Callable, Tuple

import numpy as np

from .tools.array_management import ArrayLike


def _scaled_gauss_legendre_points_and_weights(p: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute Gauss-Legendre quadrature points and weights scaled to the interval
    [-0.5, 0.5].

    Args:
        p: Polynomial degree of quadrature rule.

    Returns:
        x: Quadrature points, has shape (n,).
        w: Quadrature weights, has shape (n,).
    """
    unscaled_points, unscaled_weights = np.polynomial.legendre.leggauss(
        -(-(p + 1) // 2)
    )
    scaling = np.sum(unscaled_weights)
    return unscaled_points / scaling, unscaled_weights / scaling


@lru_cache(maxsize=None)
def _gauss_legendre_for_finite_volume(
    px: int, py: int, pz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_pts, x_wts = _scaled_gauss_legendre_points_and_weights(px)
    y_pts, y_wts = _scaled_gauss_legendre_points_and_weights(py)
    z_pts, z_wts = _scaled_gauss_legendre_points_and_weights(pz)
    xp, yp, zp = np.meshgrid(x_pts, y_pts, z_pts, indexing="ij")
    xw, yw, zw = np.meshgrid(x_wts, y_wts, z_wts, indexing="ij")
    xp = xp.flatten()
    yp = yp.flatten()
    zp = zp.flatten()
    w = (xw * yw * zw).flatten()
    return xp, yp, zp, w


def gauss_legendre_for_finite_volume(
    px: int, py: int, pz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gauss-Legendre quadrature points and weights for a finite volume with up to
    three dimensions, where the quadrature points are scaled to the 3D unit cube
    [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5].

    Args:
        px: Polynomial degree of quadrature rule in x dimension.
        py: Polynomial degree of quadrature rule in y dimension.
        pz: Polynomial degree of quadrature rule in z dimension.

    Returns:
        xp, yp, zp: Quadrature points in x, y, and z dimensions. Each has shape
            (n_quadrature), where `n_quadrature` is the total number of quadrature
            points flattened across the three dimensions.
        w: Weights for the quadrature points, has shape (n_quadrature,).
    """
    return _gauss_legendre_for_finite_volume(px, py, pz)


def gauss_legendre_mesh(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    h: Tuple[float, float, float],
    p: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Gauss-Legendre quadrature points and weights for a finite volume mesh.

    Args:
        x: x-coordinates, has shape (nx, ny, nz).
        y: y-coordinates, has shape (nx, ny, nz).
        z: z-coordinates, has shape (nx, ny, nz).
        h: Mesh spacings (hx, hy, hz).
        p: Polynomial degree of quadrature rule in each dimension (px, py, pz).

    Returns:
        xp, yp, zp: Quadrature points in x, y, and z dimensions. Each has shape
            (nx, ny, nz, n_quadrature), where `n_quadrature` is the total number of
            quadrature points flattened across the three dimensions.
        w: Weights for the quadrature points, has shape (1, 1, 1, n_quadrature).
    """
    hx, hy, hz = h
    px, py, pz = p
    xp, yp, zp, w = gauss_legendre_for_finite_volume(px, py, pz)

    # Compute the evaluation points for the quadrature rule
    na = np.newaxis
    x_eval = x[..., na] + xp[na, na, na, :] * hx
    y_eval = y[..., na] + yp[na, na, na, :] * hy
    z_eval = z[..., na] + zp[na, na, na, :] * hz

    return x_eval, y_eval, z_eval, w


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
        f: Function at which to evaluate quadrature points. It should accept the
            following arguments:
            - x: x-coordinate array. Has shape (nx, ny, nz).
            - y: y-coordinate array. Has shape (nx, ny, nz).
            - z: z-coordinate array. Has shape (nx, ny, nz).
            and return an array of shape (nvar, nx, ny, nz).
        x: x-coordinates, has shape (nx, ny, nz).
        y: y-coordinates, has shape (nx, ny, nz).
        z: z-coordinates, has shape (nx, ny, nz).
        h: Mesh spacings (hx, hy, hz).
        p: Polynomial degree of quadrature rule in each dimension.

    Returns:
        ArrayLike: Finite volume average.
    """
    x_eval, y_eval, z_eval, weights = gauss_legendre_mesh(x, y, z, h, p)
    vals = f(x_eval, y_eval, z_eval)
    return np.sum(weights * vals, axis=4)
