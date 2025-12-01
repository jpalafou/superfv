from types import ModuleType
from typing import Literal, Tuple

from superfv.axes import DIM_TO_AXIS
from superfv.stencil import stencil_sweep
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop, merge_slices


def compute_shock_detector_in_1D(
    xp: ModuleType,
    wp: ArrayLike,
    eta_threshold: float,
    axis: int,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
) -> Tuple[slice, ...]:
    """
    Compute the shock detector in 1D using the method of Berta et al. (2024), where a
    value of 1 indicates a smooth region and 0 indicates a shock.

    Args:
        xp: `np` namespace.
        wp: Array of primitive variables. Has shape (nvars, nx, ny, nz).
        eta_threshold: Threshold for the shock detector.
        axis: Axis along which to compute the shock detector.
        out: Array to which the shock detector is written. Has shape (1, nx, ny, nz).
        buffer: Array to which intermediate values are written. Has shape
            (nvars, nx, ny, nz, >=8).
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # allocate temporary arrays
    check_buffer_slots(buffer, required=8)
    delta1 = buffer[..., 0]
    delta2 = buffer[..., 1]
    delta3 = buffer[..., 2]
    delta4 = buffer[..., 3]
    eta_o = buffer[..., 4]
    eta_e = buffer[..., 5]
    eta_multivar = buffer[..., 6]
    eta_c = buffer[:1, ..., 7]

    # compute stencils
    stencil1 = 0.5 * xp.array([-1.0, 0.0, 1.0])
    stencil2 = xp.array([1.0, -2.0, 1.0])
    stencil3 = 0.5 * xp.array([-1.0, 2.0, 0.0, -2.0, 1.0])
    stencil4 = xp.array([1.0, -4.0, 6.0, -4.0, 1.0])
    modified = crop(axis, (2, -2), ndim=4)

    # compute deltas using stencil sweeps
    stencil_sweep(xp, wp, stencil1, axis, out=delta1)
    stencil_sweep(xp, wp, stencil2, axis, out=delta2)
    stencil_sweep(xp, wp, stencil3, axis, out=delta3)
    stencil_sweep(xp, wp, stencil4, axis, out=delta4)

    # compute eta values
    eta_o[...] = xp.abs(delta3) / (xp.abs(wp) + xp.abs(delta1) + xp.abs(delta3) + eps)
    eta_e[...] = xp.abs(delta4) / (xp.abs(wp) + xp.abs(delta2) + xp.abs(delta4) + eps)
    eta_multivar[...] = xp.maximum(eta_o, eta_e)
    eta_c[...] = xp.max(eta_multivar, axis=0, keepdims=True)

    out[modified] = xp.where(eta_c[modified] < eta_threshold, 1.0, 0.0)

    return modified


def compute_shock_detector(
    xp: ModuleType,
    wp: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    eta_threshold: float,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
):
    """
    Compute the shock detector using the method of Berta et al. (2024), where a value
    of 1 indicates a smooth region and 0 indicates a shock.

    Args:
        xp: `np` namespace.
        wp: Array of primitive variables. Has shape (nvars, nx, ny, nz).
        active_dims: Tuple of active dimensions along which to compute the shock
            detector. The minimum is taken over all active dimensions.
        eta_threshold: Threshold for the shock detector.
        out: Array to which the shock detector is written. Has shape (1, nx, ny, nz).
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on the number (length) of active dimensions:
            - 1D: (nvars, nx, ny, nz, >=8)
            - 2D: (nvars, nx, ny, nz, >=10)
            - 3D: (nvars, nx, ny, nz, >=11)
        eps: Small value to avoid division by zero.
    """
    ndim = len(active_dims)

    # early escape for 1D case
    if ndim == 1:
        return compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[active_dims[0]],
            out=out,
            buffer=buffer,
            eps=eps,
        )
    elif ndim == 2:
        d1 = active_dims[0]
        d2 = active_dims[1]

        check_buffer_slots(buffer, required=10)
        eta1 = buffer[:1, ..., 0]
        eta2 = buffer[:1, ..., 1]
        eta_buff = buffer[..., 2:]

        modified1 = compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[d1],
            out=eta1,
            buffer=eta_buff,
            eps=eps,
        )
        modified2 = compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[d2],
            out=eta2,
            buffer=eta_buff,
            eps=eps,
        )

        modified = merge_slices(modified1, modified2)
        out[modified] = xp.minimum(eta1[modified], eta2[modified])
        return modified

    elif ndim == 3:
        d1 = active_dims[0]
        d2 = active_dims[1]
        d3 = active_dims[2]

        check_buffer_slots(buffer, required=11)
        eta1 = buffer[:1, ..., 0]
        eta2 = buffer[:1, ..., 1]
        eta3 = buffer[:1, ..., 2]
        eta_buff = buffer[..., 3:]

        modified1 = compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[d1],
            out=eta1,
            buffer=eta_buff,
            eps=eps,
        )
        modified2 = compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[d2],
            out=eta2,
            buffer=eta_buff,
            eps=eps,
        )
        modified3 = compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=DIM_TO_AXIS[d3],
            out=eta3,
            buffer=eta_buff,
            eps=eps,
        )

        modified = merge_slices(modified1, modified2, modified3)
        out[modified] = xp.minimum(eta1[modified], eta2[modified])
        out[modified] = xp.minimum(eta3[modified], out[modified])
        return modified

    raise ValueError("active_dims must have length 1, 2, or 3.")
