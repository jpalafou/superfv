from types import ModuleType
from typing import Literal, Tuple, cast

from superfv.fv import DIM_TO_AXIS
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices, replace_slice
from superfv.tools.stability import avoid0


def central_difference(u: ArrayLike, axis: int, *, out: ArrayLike):
    """
    Compute 1D central difference, ignoring mesh size.

    Args:
        u: Array of data to differentiate.
        axis: Axis along which to compute the central difference.
        out: Array to which the output is assigned.
    """
    out[crop(axis, (1, -1))] = 0.5 * (
        u[crop(axis, (2, None))] - u[crop(axis, (None, -2))]
    )


def smooth_extrema_detector_1d(
    xp: ModuleType,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
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
    central_difference(u, axis, out=du)
    central_difference(du, axis, out=dv)
    dv[...] = avoid0(xp, 0.5 * dv, eps)

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (None, -2))] - du[crop(axis, (1, -1))]
    alpha_l = -xp.where(dv < 0, xp.maximum(vl, 0), xp.minimum(vl, 0)) / dv
    alpha_l[...] = xp.minimum(alpha_l, 1)

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, None))] - du[crop(axis, (1, -1))]
    alpha_r = xp.where(dv > 0, xp.maximum(vr, 0), xp.minimum(vr, 0)) / dv
    alpha_r[...] = xp.minimum(alpha_r, alpha_l)

    # take local minimum
    out[..., 0] = xp.minimum(alpha_l, alpha_r)

    # take min of neighbors and return
    lft_slc = crop(axis, (2, -4), ndim=5)
    cen_slc = crop(axis, (3, -3), ndim=5)
    rgt_slc = crop(axis, (4, -2), ndim=5)

    out[cen_slc] = xp.minimum(out[lft_slc], out[cen_slc])
    out[cen_slc] = xp.minimum(out[rgt_slc], out[cen_slc])

    modified = cast(Tuple[slice, ...], replace_slice(cen_slc, 4, slice(None, 1)))
    return modified


def smooth_extema_detector_2d(
    xp: ModuleType,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    dim1, dim2 = active_dims

    alpha_dim1 = buffer[..., 4:5]
    alpha_dim2 = buffer[..., 5:6]

    modified1 = smooth_extrema_detector_1d(
        xp, u, dim1, buffer[..., :4], out=alpha_dim1, eps=eps
    )
    modified2 = smooth_extrema_detector_1d(
        xp, u, dim2, buffer[..., :4], out=alpha_dim2, eps=eps
    )

    modified = merge_slices(modified1, modified2)
    out[modified] = xp.minimum(alpha_dim1[modified], alpha_dim2[modified])

    return modified


def smooth_extema_detector_3d(
    xp: ModuleType,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
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

    modified1 = smooth_extrema_detector_1d(
        xp, u, dim1, buffer[..., :4], out=alpha_dim1, eps=eps
    )
    modified2 = smooth_extrema_detector_1d(
        xp, u, dim2, buffer[..., :4], out=alpha_dim2, eps=eps
    )
    modified3 = smooth_extrema_detector_1d(
        xp, u, dim3, buffer[..., :4], out=alpha_dim3, eps=eps
    )

    modified = merge_slices(modified1, modified2, modified3)
    out[modified] = xp.minimum(alpha_dim1[modified], alpha_dim2[modified])
    out[modified] = xp.minimum(alpha_dim3[modified], out[modified])

    return modified


def smooth_extrema_detector(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the smooth extrema detector alpha along specified dimensions.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple of dimensions along which to compute the smooth extrema
            detector. Has length 1, 2, or 3 with possible values "x", "y", "z".
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=4) for 1D,
            (nvars, nx, ny, nz, >=6) for 2D,
            or (nvars, nx, ny, nz, >=7) for 3D
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.

    """
    if len(active_dims) == 1:
        return smooth_extrema_detector_1d(
            xp, u, active_dims[0], buffer, out=out, eps=eps
        )
    elif len(active_dims) == 2:
        return smooth_extema_detector_2d(xp, u, active_dims, buffer, out=out, eps=eps)
    elif len(active_dims) == 3:
        return smooth_extema_detector_3d(xp, u, buffer, out=out, eps=eps)
    raise ValueError("active_dims must have length 1, 2, or 3.")
