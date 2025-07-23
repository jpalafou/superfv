from typing import Any, Literal, Tuple, Union

import numpy as np

from superfv.fv import DIM_TO_AXIS
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices, modify_slices


def inplace_central_difference(u: ArrayLike, axis: int, out: ArrayLike):
    """
    Compute 1D central difference, ignoring mesh size.

    Args:
        u: Array of data to differentiate.
        axis: Axis along which to compute the central difference.
        out: Array to which the output is assigned.
    """
    out[crop(axis, (1, -1))] = 0.5 * (u[crop(axis, (2, 0))] - u[crop(axis, (0, -2))])


def inplace_1d_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    dim: Literal["x", "y", "z"],
    buffer: ArrayLike,
    out: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the 1D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        dim: Dimension along which to compute the smooth extrema detector: "x", "y",
            "z".
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=4).
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """

    # retrieve axis
    axis = DIM_TO_AXIS[dim]

    # allocate arrays
    du = buffer[..., 0]
    dv = buffer[..., 1]
    vl = buffer[..., 2]
    vr = buffer[..., 3]

    # compute derivatives
    inplace_central_difference(u, axis, du)
    inplace_central_difference(du, axis, dv)
    dv[...] = 0.5 * dv
    dv[...] = xp.where(
        np.logical_and(dv > -eps, dv < eps), np.where(dv < 0, -eps, eps), dv
    )

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (0, -2))] - du[crop(axis, (1, -1))]
    alpha_l = -xp.where(dv < 0, xp.maximum(vl, 0), xp.minimum(vl, 0)) / dv
    alpha_l[...] = xp.minimum(alpha_l, 1)

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, 0))] - du[crop(axis, (1, -1))]
    alpha_r = xp.where(dv > 0, xp.maximum(vr, 0), xp.minimum(vr, 0)) / dv
    alpha_r[...] = xp.minimum(alpha_r, alpha_l)

    # take local minimum
    alpha = xp.minimum(alpha_l, alpha_r)

    # take min of neighbors and return
    out_modified = modify_slices(crop(axis, (3, -3), ndim=5), axis=4, new_slice=0)
    out[out_modified] = xp.minimum(
        alpha[crop(axis, (4, -2))], alpha[crop(axis, (3, -3))]
    )
    out[out_modified] = xp.minimum(alpha[crop(axis, (2, -4))], out[out_modified])
    modified = modify_slices(out_modified, axis=4, new_slice=slice(None, 1))
    return modified


def inplace_2d_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    buffer: ArrayLike,
    out: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the 2D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple of two dimensions along which to compute the smooth extrema
            detector: ("x", "y"), ("x", "z"), or ("y", "z").
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=6).
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    dim1, dim2 = active_dims

    alpha_dim1 = buffer[..., 4:5]
    alpha_dim2 = buffer[..., 5:6]

    modified1 = inplace_1d_smooth_extrema_detector(
        xp, u, dim1, buffer[..., :4], alpha_dim1, eps
    )
    modified2 = inplace_1d_smooth_extrema_detector(
        xp, u, dim2, buffer[..., :4], alpha_dim2, eps
    )

    modified = merge_slices(modified1, modified2)
    out[modified] = xp.minimum(alpha_dim1[modified], alpha_dim2[modified])

    return modified


def inplace_3d_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    buffer: ArrayLike,
    out: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the 3D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=7).
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    dim1 = "x"
    dim2 = "y"
    dim3 = "z"

    alpha_dim1 = buffer[..., 4:5]
    alpha_dim2 = buffer[..., 5:6]
    alpha_dim3 = buffer[..., 6:7]

    modified1 = inplace_1d_smooth_extrema_detector(
        xp, u, dim1, buffer[..., :4], alpha_dim1, eps
    )
    modified2 = inplace_1d_smooth_extrema_detector(
        xp, u, dim2, buffer[..., :4], alpha_dim2, eps
    )
    modified3 = inplace_1d_smooth_extrema_detector(
        xp, u, dim3, buffer[..., :4], alpha_dim3, eps
    )

    modified = merge_slices(modified1, modified2, modified3)
    out[modified] = xp.minimum(alpha_dim1[modified], alpha_dim2[modified])
    out[modified] = xp.minimum(alpha_dim3[modified], out[modified])

    return modified


def inplace_smooth_extrema_detector(
    xp: Any,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    buffer: ArrayLike,
    out: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the smooth extrema detector alpha inplace along specified dimensions.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple of dimensions along which to compute the smooth extrema
            detector. Has length 1, 2, or 3 with possible values "x", "y", "z".
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=4) for 1D,
            (nvars, nx, ny, nz, >=6) for 2D,
            or (nvars, nx, ny, nz, >=7) for 3D
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.

    """
    if len(active_dims) == 1:
        return inplace_1d_smooth_extrema_detector(
            xp, u, active_dims[0], buffer, out, eps
        )
    elif len(active_dims) == 2:
        return inplace_2d_smooth_extrema_detector(xp, u, active_dims, buffer, out, eps)
    elif len(active_dims) == 3:
        return inplace_3d_smooth_extrema_detector(xp, u, buffer, out, eps)
    raise ValueError("active_dims must have length 1, 2, or 3.")


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
