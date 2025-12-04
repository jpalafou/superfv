from types import ModuleType
from typing import Literal, Tuple

from superfv.axes import DIM_TO_AXIS
from superfv.stencil import stencil_sweep
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices


def compute_eta(
    xp: ModuleType,
    w1: ArrayLike,
    wr: ArrayLike,
    axis: int,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
) -> Tuple[slice, ...]:
    """
    Compute the shock detector eta in 1D using the method of Berta et al. (2024).

    Args:
        xp: `np` namespace.
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        axis: Axis along which to compute the shock detector.
        out: Array to which eta is written. Has shape (1, nx, ny, nz).
        buffer: Array to which intermediate values are written. Has shape
            (nvars, nx, ny, nz, >=6).
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # allocate temporary arrays
    check_buffer_slots(buffer, required=6)
    delta1 = buffer[..., 0]
    delta2 = buffer[..., 1]
    delta3 = buffer[..., 2]
    delta4 = buffer[..., 3]
    eta_o = buffer[..., 4]
    eta_e = buffer[..., 5]

    # compute stencils
    stencil1 = 0.5 * xp.array([-1.0, 0.0, 1.0])
    stencil2 = xp.array([1.0, -2.0, 1.0])
    stencil3 = 0.5 * xp.array([-1.0, 2.0, 0.0, -2.0, 1.0])
    stencil4 = xp.array([1.0, -4.0, 6.0, -4.0, 1.0])
    inner = crop(axis, (2, -2), ndim=4)

    # compute deltas using stencil sweeps
    stencil_sweep(xp, w1, stencil1, axis, out=delta1)
    stencil_sweep(xp, w1, stencil2, axis, out=delta2)
    stencil_sweep(xp, w1, stencil3, axis, out=delta3)
    stencil_sweep(xp, w1, stencil4, axis, out=delta4)

    # compute eta values
    eta_o[...] = xp.abs(delta3) / (xp.abs(wr) + xp.abs(delta1) + xp.abs(delta3) + eps)
    eta_e[...] = xp.abs(delta4) / (xp.abs(wr) + xp.abs(delta2) + xp.abs(delta4) + eps)
    out[inner] = xp.maximum(eta_o[inner], eta_e[inner])

    return inner


def compute_shock_detector(
    xp: ModuleType,
    w1: ArrayLike,
    wr: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    eta_threshold: float,
    *,
    out: ArrayLike,
    eta: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the shock detector using the method of Berta et al. (2024), where a value
    of 1 indicates a smooth region and 0 indicates a shock.

    Args:
        xp: `np` namespace.
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        active_dims: Tuple of active dimensions along which to compute the shock
            detector.
        out: Array to which the shock detector is written. Has shape (1, nx, ny, nz).
            The minimum value over all active dimensions and variables is taken.
        eta: Array to which intermediate eta values are written. Has shape
            (nvars, nx, ny, nz).
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on the number (length) of active dimensions:
            - 1D: (nvars, nx, ny, nz, >=7)
            - 2D: (nvars, nx, ny, nz, >=9)
            - 3D: (nvars, nx, ny, nz, >=10)
        eps: Small value to avoid division by zero.
    """
    ndim = len(active_dims)

    # early escape for 1D case
    if ndim == 1:
        axis = DIM_TO_AXIS[active_dims[0]]

        eta_max = buffer[:1, ..., 0]
        eta_buff = buffer[..., 1:]

        inner = compute_eta(xp, w1, wr, axis, out=eta, buffer=eta_buff, eps=eps)

    elif ndim == 2:
        axis1 = DIM_TO_AXIS[active_dims[0]]
        axis2 = DIM_TO_AXIS[active_dims[1]]

        eta1 = buffer[..., 0]
        eta2 = buffer[..., 1]
        eta_max = buffer[:1, ..., 2]
        eta_buff = buffer[..., 3:]

        inner1 = compute_eta(xp, w1, wr, axis1, out=eta1, buffer=eta_buff, eps=eps)
        inner2 = compute_eta(xp, w1, wr, axis2, out=eta2, buffer=eta_buff, eps=eps)
        inner = merge_slices(inner1, inner2)
        eta[inner] = xp.maximum(eta1[inner], eta2[inner])

    elif ndim == 3:
        axis1 = DIM_TO_AXIS[active_dims[0]]
        axis2 = DIM_TO_AXIS[active_dims[1]]
        axis3 = DIM_TO_AXIS[active_dims[2]]

        eta1 = buffer[..., 0]
        eta2 = buffer[..., 1]
        eta3 = buffer[..., 2]
        eta_max = buffer[:1, ..., 3]
        eta_buff = buffer[..., 4:]

        inner1 = compute_eta(xp, w1, wr, axis1, out=eta1, buffer=eta_buff, eps=eps)
        inner2 = compute_eta(xp, w1, wr, axis2, out=eta2, buffer=eta_buff, eps=eps)
        inner3 = compute_eta(xp, w1, wr, axis3, out=eta3, buffer=eta_buff, eps=eps)
        inner = merge_slices(inner1, inner2, inner3)
        eta[inner] = xp.maximum(eta1[inner], eta2[inner])
        eta[inner] = xp.maximum(eta3[inner], eta[inner])

    else:
        raise ValueError("active_dims must have length 1, 2, or 3.")

    eta_max[...] = xp.max(eta, axis=0, keepdims=True)  # maximum over all variables
    out[inner] = xp.where(eta_max[inner] < eta_threshold, 1.0, 0.0)

    return inner
