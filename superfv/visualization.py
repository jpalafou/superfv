from typing import Tuple

import numpy as np

from .tools.device_management import CUPY_AVAILABLE, ArrayLike

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def shell_average(
    q: ArrayLike,
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
    center: Tuple[float, float, float],
    r_edges: ArrayLike,
    convert_to_numpy: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the shell average of a quantity q and the radial centers of the shells with support for
    both NumPy and CuPy arrays.

    Args:
        q: The quantity to average. Has shape (nx, ny, nz).
        X: The x-coordinates of the grid. Has shape (nx, ny, nz).
        Y: The y-coordinates of the grid. Has shape (nx, ny, nz).
        Z: The z-coordinates of the grid. Has shape (nx, ny, nz).
        center: The center of the shells (x_center, y_center, z_center).
        r_edges: The radial edges of the shells. Has shape (n_edges,).
        convert_to_numpy: Convert returned arrays to NumPy arrays if True.

    Returns:
        averages: The shell averages. Has shape (n_edges - 1,).
        r_centers: The radial centers of the shells. Has shape (n_edges - 1,).
    """
    using_cupy = False

    if CUPY_AVAILABLE and isinstance(q, cp.ndarray):
        using_cupy = True
        xp = cp

        if not isinstance(r_edges, cp.ndarray):
            raise ValueError("r_edges must be a CuPy array if q is a CuPy array")
        if not isinstance(X, cp.ndarray):
            raise ValueError("X must be a CuPy array if q is a CuPy array")
        if not isinstance(Y, cp.ndarray):
            raise ValueError("Y must be a CuPy array if q is a CuPy array")
        if not isinstance(Z, cp.ndarray):
            raise ValueError("Z must be a CuPy array if q is a CuPy array")
    else:
        xp = np

        if not isinstance(r_edges, np.ndarray):
            raise ValueError("r_edges must be a NumPy array if q is a NumPy array")
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array if q is a NumPy array")
        if not isinstance(Y, np.ndarray):
            raise ValueError("Y must be a NumPy array if q is a NumPy array")
        if not isinstance(Z, np.ndarray):
            raise ValueError("Z must be a NumPy array if q is a NumPy array")

    R = xp.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    k = xp.digitize(R.ravel(), r_edges)
    ok = (0 < k) & (k < len(r_edges))

    sums = xp.bincount(k[ok] - 1, weights=q.ravel()[ok], minlength=len(r_edges) - 1)
    counts = xp.bincount(k[ok] - 1, minlength=len(r_edges) - 1)

    averages = sums / counts
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    if using_cupy and convert_to_numpy:
        return cp.asnumpy(averages), cp.asnumpy(r_centers)
    else:
        return averages, r_centers
