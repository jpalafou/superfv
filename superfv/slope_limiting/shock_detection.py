from types import ModuleType
from typing import Literal, Tuple

from superfv.stencil import stencil_sweep
from superfv.tools.device_management import ArrayLike


def compute_shock_detector_in_1D(
    xp: ModuleType,
    wp: ArrayLike,
    eta_threshold: float,
    axis: int,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
):
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
    """
    # allocate temporary arrays
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

    out[...] = xp.where(eta_c < eta_threshold, 1.0, 0.0)


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
        buffer: Array to which intermediate values are written. Has shape
            (nvars, nx, ny, nz, >=11).
        eps: Small value to avoid division by zero.
    """
    # allocate temporary arrays
    eta_x = buffer[:1, ..., 0]
    eta_y = buffer[:1, ..., 1]
    eta_z = buffer[:1, ..., 2]
    sub_buffer = buffer[..., 3:]

    # initialize eta arrays to be 1
    eta_x[...] = 1.0
    eta_y[...] = 1.0
    eta_z[...] = 1.0
    out[...] = 1.0

    # compute shock detector in each active dimension, taking the minimum of all
    # dimensions
    if "x" in active_dims:
        compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=1,
            out=eta_x,
            buffer=sub_buffer,
            eps=eps,
        )
        xp.minimum(out, eta_x, out=out)
    if "y" in active_dims:
        compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=2,
            out=eta_y,
            buffer=sub_buffer,
            eps=eps,
        )
        xp.minimum(out, eta_y, out=out)
    if "z" in active_dims:
        compute_shock_detector_in_1D(
            xp,
            wp,
            eta_threshold,
            axis=3,
            out=eta_z,
            buffer=sub_buffer,
            eps=eps,
        )
        xp.minimum(out, eta_z, out=out)
