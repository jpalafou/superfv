import numpy as np

from .tools.array_management import ArrayLike


def minmod(du_left: ArrayLike, du_right: ArrayLike, tol: float = 1e-16) -> ArrayLike:
    """
    Args:
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
        tol (float): Tolerance.
    Returns:
        ArrayLike: Minmod of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    ratio = du_right / np.where(
        du_left > 0,
        np.where(du_left > tol, du_left, tol),
        np.where(du_left < -tol, du_left, -tol),
    )
    ratio = np.where(ratio < 1, ratio, 1)
    return np.where(ratio > 0, ratio, 0) * du_left


def moncen(du_left: ArrayLike, du_right: ArrayLike) -> ArrayLike:
    """
    Args:
        du_left (ArrayLike): Left differences. Has shape (nvars, nx, ny, nz, ...).
        du_right (ArrayLike): Right difference. Has shape (nvars, nx, ny, nz, ...).
    Returns:
        ArrayLike: Moncen of left and right differences. Has shape
            (nvars, nx, ny, nz, ...).
    """
    du_central = 0.5 * (du_left + du_right)
    slope = np.minimum(np.abs(2 * du_left), np.abs(2 * du_right))
    slope = np.sign(du_central) * np.minimum(slope, np.abs(du_central))
    return np.where(du_left * du_right >= 0, slope, 0)
