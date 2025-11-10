from types import ModuleType
from typing import Literal, Tuple, cast

from superfv.fv import DIM_TO_AXIS
from superfv.tools.buffer import check_buffer_slots
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
    *,
    out: ArrayLike,
    buffer: ArrayLike,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=7).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """

    # retrieve axis
    axis = DIM_TO_AXIS[dim]

    # allocate arrays
    check_buffer_slots(buffer, required=7)
    du = buffer[..., 0]
    dv = buffer[..., 1]
    vl = buffer[..., 2]
    vr = buffer[..., 3]
    alpha_l = buffer[..., 4]
    alpha_r = buffer[..., 5]
    alpha = buffer[..., 6:7]  # (..., 1)

    # compute derivatives
    central_difference(u, axis, out=du)
    central_difference(du, axis, out=dv)
    dv[...] = avoid0(xp, 0.5 * dv, eps)

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (None, -2))] - du[crop(axis, (1, -1))]
    alpha_l[...] = -xp.where(dv < 0, xp.maximum(vl, 0), xp.minimum(vl, 0)) / dv

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, None))] - du[crop(axis, (1, -1))]
    alpha_r[...] = xp.where(dv > 0, xp.maximum(vr, 0), xp.minimum(vr, 0)) / dv

    # combine left and right detectors
    alpha[..., 0] = xp.minimum(xp.minimum(alpha_l, alpha_r), 1.0)

    # take min of neighbors and return
    lft_slc = crop(axis, (2, -4), ndim=5)
    cen_slc = crop(axis, (3, -3), ndim=5)
    rgt_slc = crop(axis, (4, -2), ndim=5)

    out[cen_slc] = xp.minimum(alpha[lft_slc], alpha[cen_slc])
    out[cen_slc] = xp.minimum(alpha[rgt_slc], out[cen_slc])

    modified = cast(Tuple[slice, ...], replace_slice(cen_slc, 4, slice(None, 1)))
    return modified


def smooth_extrema_detector_2d(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    buffer: ArrayLike,
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
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=9).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    d1, d2 = active_dims

    check_buffer_slots(buffer, required=9)
    alph1 = buffer[..., :1]
    alph2 = buffer[..., 1:2]
    abuff = buffer[..., 2:]

    modified1 = smooth_extrema_detector_1d(xp, u, d1, out=alph1, buffer=abuff, eps=eps)
    modified2 = smooth_extrema_detector_1d(xp, u, d2, out=alph2, buffer=abuff, eps=eps)

    modified = merge_slices(modified1, modified2)
    out[modified] = xp.minimum(alph1[modified], alph2[modified])

    return modified


def smooth_extrema_detector_3d(
    xp: ModuleType,
    u: ArrayLike,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the 3D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=10).
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    d1: Literal["x"] = "x"
    d2: Literal["y"] = "y"
    d3: Literal["z"] = "z"

    check_buffer_slots(buffer, required=10)
    alph1 = buffer[..., :1]
    alph2 = buffer[..., 1:2]
    alph3 = buffer[..., 2:3]
    abuff = buffer[..., 3:]

    modified1 = smooth_extrema_detector_1d(xp, u, d1, out=alph1, buffer=abuff, eps=eps)
    modified2 = smooth_extrema_detector_1d(xp, u, d2, out=alph2, buffer=abuff, eps=eps)
    modified3 = smooth_extrema_detector_1d(xp, u, d3, out=alph3, buffer=abuff, eps=eps)

    modified = merge_slices(modified1, modified2, modified3)
    out[modified] = xp.minimum(alph1[modified], alph2[modified])
    out[modified] = xp.minimum(alph3[modified], out[modified])

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
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on the number (length) of active dimensions:
            - 1D: (nvars, nx, ny, nz, >=7)
            - 2D: (nvars, nx, ny, nz, >=9)
            - 3D: (nvars, nx, ny, nz, >=10)
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.

    """
    if len(active_dims) == 1:
        return smooth_extrema_detector_1d(
            xp, u, active_dims[0], out=out, buffer=buffer, eps=eps
        )
    elif len(active_dims) == 2:
        return smooth_extrema_detector_2d(
            xp, u, active_dims, out=out, buffer=buffer, eps=eps
        )
    elif len(active_dims) == 3:
        return smooth_extrema_detector_3d(xp, u, out=out, buffer=buffer, eps=eps)
    raise ValueError("active_dims must have length 1, 2, or 3.")
