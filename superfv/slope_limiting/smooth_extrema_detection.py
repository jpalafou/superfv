from typing import Any, Literal, Tuple

from superfv.fv import DIM_TO_AXIS
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices, modify_slices
from superfv.tools.stability import avoid0


def inplace_central_difference(u: ArrayLike, axis: int, *, out: ArrayLike):
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
    *,
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
    dv[...] = avoid0(xp, 0.5 * dv, eps)

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
    *,
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
    *,
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
    dim1: Literal["x"] = "x"
    dim2: Literal["y"] = "y"
    dim3: Literal["z"] = "z"

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
    *,
    out: ArrayLike,
    buffer: ArrayLike,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=4) for 1D,
            (nvars, nx, ny, nz, >=6) for 2D,
            or (nvars, nx, ny, nz, >=7) for 3D
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
