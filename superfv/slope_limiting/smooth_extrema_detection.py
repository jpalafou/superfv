from functools import partial
from typing import Any, Tuple, Union

import numpy as np

from superfv.tools.array_management import ArrayLike, ArraySlicer


def central_difference(array_slicer: ArraySlicer, u: ArrayLike, axis: int) -> ArrayLike:
    """
    Compute second order central difference of array u along a specified dimension,
    scaled by the uniform grid spacing.

    Args:
        array_slicer (ArraySlicer) : Array slicer object.
        u (ArrayLike) : Array of shape (nvars, nx, ny, nz, ...)
        axis (int) : Axis along which to compute the derivative.
    """
    _slc = partial(array_slicer, axis=axis)
    return 0.5 * (u[_slc(cut=(2, 0))] - u[_slc(cut=(0, -2))])


def chopchop(array_slicer, u, chop_size, axis):
    return u[array_slicer(cut=(chop_size[0], -chop_size[1]), axis=axis)]


def avoid_0(x: np.ndarray, eps: float, postive_at_0: bool = True) -> np.ndarray:
    """
    args:
        x:              array
        eps:            tolerance
        positive_at_0:  whether to use positive eps where x is 0
    returns:
        x with near-zero elements rounded to +eps or -eps depending on sign
    """
    if postive_at_0:
        negative_eps = np.logical_and(x > -eps, x < 0.0)
        positive_eps = np.logical_and(x >= 0.0, x < eps)
    else:
        negative_eps = np.logical_and(x > -eps, x <= 0.0)
        positive_eps = np.logical_and(x > 0.0, x < eps)
    return np.where(positive_eps, eps, np.where(negative_eps, -eps, x))


def compute_1d_smooth_extrema_detector(
    xp: Any, array_slicer: ArraySlicer, u: ArrayLike, axis: int, eps: float = 1e-16
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha along specified direction.

    Args:
        xp (Any) : `np` namespace.
        array_slicer (ArraySlicer) : Array slicer object.
        u (ArrayLike) : Array of artbitrary shape.
        axis (int) : Axis along which to compute the derivative.
        eps (float) : How close to 0 dv is permitted to reach.
    Returns:
        out (ArrayLike) : Smooth extrema detector alpha. Shorter along the specified
            axis by 6 elements.
    """
    _slc = partial(array_slicer, axis=axis)

    du = central_difference(array_slicer, u, axis)
    dv = 0.5 * central_difference(array_slicer, du, axis)
    dv_safe = xp.where(xp.abs(dv) < eps, xp.sign(dv) * eps, dv)

    # left detector
    v_l = du[_slc(cut=(0, -2))] - du[_slc(cut=(1, -1))]
    alpha_l = -xp.where(dv_safe < 0, xp.maximum(v_l, 0), xp.minimum(v_l, 0)) / dv_safe
    alpha_l[...] = xp.minimum(alpha_l, 1)

    # right detector
    v_r = du[_slc(cut=(2, 0))] - du[_slc(cut=(1, -1))]
    alpha_r = xp.where(dv_safe > 0, xp.maximum(v_r, 0), xp.minimum(v_r, 0)) / dv_safe
    alpha_r[...] = xp.minimum(alpha_r, alpha_l)

    # take local minimum
    alpha = xp.minimum(alpha_l, alpha_r)
    out = xp.minimum.reduce(
        [
            alpha[_slc(cut=(2, 0))],
            alpha[_slc(cut=(1, -1))],
            alpha[_slc(cut=(0, -2))],
        ]
    )
    return out


def compute_2d_smooth_extrema_detector(
    xp: Any,
    array_slicer: ArraySlicer,
    u: ArrayLike,
    axes: Tuple[int, int],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha in x and y directions.

    Args:
        xp (Any) : `np` namespace.
        array_slicer (ArraySlicer) : Array slicer object.
        u (ArrayLike) : Array of arbitrary shape.
        axes (Tuple[int, int]) : Axes along which to compute the detector.
        eps (float) : How close to 0 dv is permitted to reach.
    Returns:
        out (ArrayLike) : Smooth extrema detector alpha. Shorter along the specified
            axes by 6 elements.
    """
    axis1, axis2 = axes
    _slc1, _slc2 = partial(array_slicer, axis=axis1), partial(array_slicer, axis=axis2)
    alpha_dim1 = compute_1d_smooth_extrema_detector(
        xp, array_slicer, u, axis1, eps=eps
    )[_slc2(cut=(3, -3))]
    alpha_dim2 = compute_1d_smooth_extrema_detector(
        xp, array_slicer, u, axis2, eps=eps
    )[_slc1(cut=(3, -3))]
    out = xp.minimum(alpha_dim1, alpha_dim2)
    return out


def compute_3d_smooth_extrema_detector(
    xp: Any,
    array_slicer: ArraySlicer,
    u: ArrayLike,
    axes: Tuple[int, int, int],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha in x, y, and z directions.

    Args:
        xp (Any) : `np` namespace.
        array_slicer (ArraySlicer) : Array slicer object.
        u (ArrayLike) : Array of arbitrary shape.
        axes (Tuple[int, int, int]) : Axes along which to compute the detector.
        eps (float) : How close to 0 dv is permitted to reach.
    Returns:
        out (ArrayLike) : Smooth extrema detector alpha. Shorter along the specified
            axes by 6 elements.
    """
    _slc = [partial(array_slicer, axis=axis) for axis in axes]
    alpha_dim1 = compute_1d_smooth_extrema_detector(
        xp, array_slicer, u, axes[0], eps=eps
    )
    alpha_dim2 = compute_1d_smooth_extrema_detector(
        xp, array_slicer, u, axes[1], eps=eps
    )
    alpha_dim3 = compute_1d_smooth_extrema_detector(
        xp, array_slicer, u, axes[2], eps=eps
    )
    out = xp.minimum.reduce(
        [
            alpha_dim1[_slc[1](cut=(3, -3))][_slc[2](cut=(3, -3))],
            alpha_dim2[_slc[0](cut=(3, -3))][_slc[2](cut=(3, -3))],
            alpha_dim3[_slc[0](cut=(3, -3))][_slc[1](cut=(3, -3))],
        ]
    )
    return out


def compute_smooth_extrema_detector(
    xp: Any,
    array_slicer: ArraySlicer,
    u: ArrayLike,
    axes: Union[int, Tuple[int, ...]],
    eps: float = 1e-16,
) -> ArrayLike:
    """
    Compute smooth extrema detector alpha along specified directions.

    Args:
        xp (Any) : `np` namespace.
        array_slicer (ArraySlicer) : Array slicer object.
        u (ArrayLike) : Array of arbitrary shape.
        axes (Union[int, Tuple[int, ...]]) : Axes along which to compute the detector.
        eps (float) : How close to 0 dv is permitted to reach.

    Returns:
        out (ArrayLike) : Smooth extrema detector alpha. Shorter along the specified
            axes by 6 elements.
    """
    if isinstance(axes, int) or len(axes) == 1:
        return compute_1d_smooth_extrema_detector(
            xp, array_slicer, u, axes if isinstance(axes, int) else axes[0], eps=eps
        )
    elif len(axes) == 2:
        return compute_2d_smooth_extrema_detector(xp, array_slicer, u, axes, eps=eps)
    elif len(axes) == 3:
        return compute_3d_smooth_extrema_detector(xp, array_slicer, u, axes, eps=eps)
