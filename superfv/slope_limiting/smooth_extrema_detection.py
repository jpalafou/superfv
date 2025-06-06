from typing import Any, Tuple, Union

import numpy as np

from superfv.tools.array_management import ArrayLike, crop


def central_difference(u: ArrayLike, axis: int) -> ArrayLike:
    """
    Compute second order central difference of array u along a specified dimension,
    scaled by the uniform grid spacing.

    Args:
        u: Array of shape (nvars, nx, ny, nz, ...)
        axis: Axis along which to compute the derivative.

    Returns:
        Second order central difference of u along the specified axis. The returned
        array is shorter along the specified axis by 2 elements.
    """
    return 0.5 * (u[crop(axis, (2, 0))] - u[crop(axis, (0, -2))])


def avoid_0(x: np.ndarray, eps: float, postive_at_0: bool = True) -> np.ndarray:
    """
    Args:
        x: Array of arbitrary shape.
        eps: Small positive number to avoid division by zero.
        positive_at_0: Whether to use positive eps where x is 0.

    Returns:
        Array with near-zero elements rounded to +eps or -eps depending on sign.
    """
    if postive_at_0:
        negative_eps = np.logical_and(x > -eps, x < 0.0)
        positive_eps = np.logical_and(x >= 0.0, x < eps)
    else:
        negative_eps = np.logical_and(x > -eps, x <= 0.0)
        positive_eps = np.logical_and(x > 0.0, x < eps)
    return np.where(positive_eps, eps, np.where(negative_eps, -eps, x))


def compute_1d_smooth_extrema_detector(
    xp: Any, u: ArrayLike, axis: int, eps: float = 1e-16
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha along specified direction.

    Args:
        xp: `np` namespace.
        u: Array of artbitrary shape.
        axis: Axis along which to compute the smooth extrema detector.
        eps: How close to 0 dv is permitted to reach.

    Returns:
        Array of values for alpha, the smooth extrema detector value. The returned
        array is shorter along the specified axis by 6 elements.
    """
    du = central_difference(u, axis)
    dv = 0.5 * central_difference(du, axis)
    dv_safe = np.where(
        np.logical_and(dv > -eps, dv < eps), np.where(dv < 0, -eps, eps), dv
    )

    # left detector
    v_l = du[crop(axis, (0, -2))] - du[crop(axis, (1, -1))]
    alpha_l = -xp.where(dv_safe < 0, xp.maximum(v_l, 0), xp.minimum(v_l, 0)) / dv_safe
    alpha_l[...] = xp.minimum(alpha_l, 1)

    # right detector
    v_r = du[crop(axis, (2, 0))] - du[crop(axis, (1, -1))]
    alpha_r = xp.where(dv_safe > 0, xp.maximum(v_r, 0), xp.minimum(v_r, 0)) / dv_safe
    alpha_r[...] = xp.minimum(alpha_r, alpha_l)

    # take local minimum
    alpha = xp.minimum(alpha_l, alpha_r)
    out = xp.minimum.reduce(
        [
            alpha[crop(axis, (2, 0))],
            alpha[crop(axis, (1, -1))],
            alpha[crop(axis, (0, -2))],
        ]
    )
    return out


def compute_2d_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    axes: Tuple[int, int],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha in x and y directions.

    Args:
        xp: `np` namespace.
        u: Array of artbitrary shape.
        axes: Two axes along which to compute the smooth extrema detector.
        eps: How close to 0 dv is permitted to reach.

    Returns:
        Array of values for alpha, the smooth extrema detector value. The returned
        array is shorter along the specified axes by 6 elements.
    """
    axis1, axis2 = axes
    alpha_dim1 = compute_1d_smooth_extrema_detector(xp, u, axis1, eps=eps)[
        crop(axis2, (3, -3))
    ]
    alpha_dim2 = compute_1d_smooth_extrema_detector(xp, u, axis2, eps=eps)[
        crop(axis1, (3, -3))
    ]
    out = xp.minimum(alpha_dim1, alpha_dim2)
    return out


def compute_3d_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    axes: Tuple[int, int, int],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha in x, y, and z directions.

    Args:
        xp: `np` namespace.
        u: Array of artbitrary shape.
        axes: Three axes along which to compute the smooth extrema detector.
        eps: How close to 0 dv is permitted to reach.

    Returns:
        Array of values for alpha, the smooth extrema detector value. The returned
        array is shorter along the specified axes by 6 elements.
    """
    axis1, axis2, axis3 = axes
    alpha_dim1 = compute_1d_smooth_extrema_detector(xp, u, axis1, eps=eps)
    alpha_dim2 = compute_1d_smooth_extrema_detector(xp, u, axis2, eps=eps)
    alpha_dim3 = compute_1d_smooth_extrema_detector(xp, u, axis3, eps=eps)
    out = xp.minimum.reduce(
        [
            alpha_dim1[crop((axis2, axis3), (3, -3))],
            alpha_dim2[crop((axis1, axis3), (3, -3))],
            alpha_dim3[crop((axis1, axis2), (3, -3))],
        ]
    )
    return out


def compute_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    axes: Union[int, Tuple[int, ...]],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha along specified directions.

    Args:
        xp: `np` namespace.
        u: Array of artbitrary shape.
        axes:  Axes along which to compute the smooth extrema detector.
        eps: How close to 0 dv is permitted to reach.

    Returns:
        Array of values for alpha, the smooth extrema detector value. The returned
        array is shorter along the specified axes by 6 elements.
    """
    if isinstance(axes, int) or len(axes) == 1:
        return compute_1d_smooth_extrema_detector(
            xp, u, axes if isinstance(axes, int) else axes[0], eps=eps
        )
    elif len(axes) == 2:
        return compute_2d_smooth_extrema_detector(xp, u, axes, eps=eps)
    elif len(axes) == 3:
        return compute_3d_smooth_extrema_detector(xp, u, axes, eps=eps)
    raise ValueError("Axes must be int or tuple of ints with length 1, 2, or 3.")
