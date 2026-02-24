from types import ModuleType
from typing import Literal, Tuple

from superfv.axes import DIM_TO_AXIS
from superfv.stencil import stencil_sweep
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
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
        out: Array to which eta is written. Has shape (nvars, nx, ny, nz).
        buffer: Array to which intermediate values are written. Has shape
            (nvars, nx, ny, nz, >=6).
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    if hasattr(xp, "cupy"):
        return compute_eta_kernel_helper(w1, wr, axis, out=out, eps=eps)

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
    of 0 indicates a smooth region and 1 indicates a shock.

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
    out[inner] = xp.where(eta_max[inner] < eta_threshold, 0, 1)

    return inner


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    compute_eta_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 wl2, float64 wl1, float64 wc, float64 wr1, float64 wr2"
            ", float64 wref, float64 eps"
        ),
        out_params="float64 eta",
        operation=(
            """
        double delta1 = 0.5 * (wr1 - wl1);
        double delta2 = wr1 - 2.0 * wc + wl1;
        double delta3 = 0.5 * (wr2 - 2.0 * wr1 + 2.0 * wl1 - wl2);
        double delta4 = wr2 - 4.0 * wr1 + 6.0 * wc - 4.0 * wl1 + wl2;
        double eta_o = fabs(delta3) / (fabs(wref) + fabs(delta1) + fabs(delta3) + eps);
        double eta_e = fabs(delta4) / (fabs(wref) + fabs(delta2) + fabs(delta4) + eps);
        eta = fmax(eta_o, eta_e);
        """
        ),
        name="compute_eta_kernel",
        no_return=True,
    )

    compute_shocks_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_shocks_kernel(
            const double *w,
            const double *wref,
            double *eta,
            int *has_shock,
            const double eta_threshold,
            const double eps,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // w shape          (nvars, nx, ny, nz)
            // wref shape       (nvars, nx, ny, nz)
            // eta shape        (nvars, nx, ny, nz, 3)
            // has_shock shape  (nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long nxyz = (long long)nx * ny * nz;
            bool usingx = nx > 1, usingy = ny > 1, usingz = nz > 1;

            for (long long ixyz = tid; ixyz < nxyz; ixyz += stride) {
                long long t = ixyz;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;

                // skip boundary cells
                if ((usingx && (ix < 2 || ix >= nx - 2))
                || (usingy && (iy < 2 || iy >= ny - 2))
                || (usingz && (iz < 2 || iz >= nz - 2))) {
                    continue;
                }

                // init has_shock to 0
                has_shock[ixyz] = 0;

                // loop over dimensions and variables
                for (int id = 0; id < 3; id++) {
                    switch (id) {
                        case 0: if (!usingx) continue; break;
                        case 1: if (!usingy) continue; break;
                        case 2: if (!usingz) continue; break;
                    }

                    for (int iv = 0; iv < nvars; iv++) {
                        long long ivxyz = iv * nxyz + ixyz;
                        long long i = ivxyz * 3 + id; // eta idx

                        long long j0, j1, j2, j3, j4;
                        switch (id) {
                            case 0:
                                j0 = (((long long)iv * nx + (ix - 2)) * ny + iy) * nz + iz;
                                j1 = (((long long)iv * nx + (ix - 1)) * ny + iy) * nz + iz;
                                j2 = ivxyz;
                                j3 = (((long long)iv * nx + (ix + 1)) * ny + iy) * nz + iz;
                                j4 = (((long long)iv * nx + (ix + 2)) * ny + iy) * nz + iz;
                                break;
                            case 1:
                                j0 = (((long long)iv * nx + ix) * ny + (iy - 2)) * nz + iz;
                                j1 = (((long long)iv * nx + ix) * ny + (iy - 1)) * nz + iz;
                                j2 = ivxyz;
                                j3 = (((long long)iv * nx + ix) * ny + (iy + 1)) * nz + iz;
                                j4 = (((long long)iv * nx + ix) * ny + (iy + 2)) * nz + iz;
                                break;
                            case 2:
                                j0 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 2);
                                j1 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 1);
                                j2 = ivxyz;
                                j3 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz + 1);
                                j4 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz + 2);
                                break;
                        }

                        // gather neighbors
                        double wl2 = w[j0];
                        double wl1 = w[j1];
                        double wc = w[j2];
                        double wr1 = w[j3];
                        double wr2 = w[j4];
                        double wrefval = wref[j2];

                        // compute eta
                        double delta1 = 0.5 * (wr1 - wl1);
                        double delta2 = wr1 - 2.0 * wc + wl1;
                        double delta3 = 0.5 * (wr2 - 2.0 * wr1 + 2.0 * wl1 - wl2);
                        double delta4 = wr2 - 4.0 * wr1 + 6.0 * wc - 4.0 * wl1 + wl2;
                        double eta_o = fabs(delta3)
                            / (fabs(wrefval) + fabs(delta1) + fabs(delta3) + eps);
                        double eta_e = fabs(delta4)
                            / (fabs(wrefval) + fabs(delta2) + fabs(delta4) + eps);
                        eta[i] = fmax(eta_o, eta_e);

                        // set to 1 if eta exceeds threshold
                        if (!has_shock[ixyz] && eta[i] > eta_threshold) has_shock[ixyz] = 1;
                    }
                }
            }
        }
        """,
        name="compute_shocks_kernel",
    )


def compute_eta_kernel_helper(
    w1: ArrayLike,
    wr: ArrayLike,
    axis: int,
    *,
    out: ArrayLike,
    eps: float = 1e-16,
) -> Tuple[slice, ...]:
    """
    Compute the shock detector eta in 1D using the method of Berta et al. (2024) with a
    custom CuPy kernel.

    Args:
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        axis: Axis along which to compute the shock detector.
        out: Array to which eta is written. Has shape (nvars, nx, ny, nz).
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    wl2 = w1[crop(axis, (None, -4), ndim=4)]
    wl1 = w1[crop(axis, (1, -3), ndim=4)]
    wc = w1[crop(axis, (2, -2), ndim=4)]
    wr1 = w1[crop(axis, (3, -1), ndim=4)]
    wr2 = w1[crop(axis, (4, None), ndim=4)]
    wref = wr[crop(axis, (2, -2), ndim=4)]

    inner_slice = crop(axis, (2, -2), ndim=4)
    eta_inner = out[inner_slice]

    compute_eta_kernel(wl2, wl1, wc, wr1, wr2, wref, eps, eta_inner)

    return inner_slice


def compute_shocks_kernel_helper(
    w1: ArrayLike,
    wr: ArrayLike,
    eta_threshold: float,
    eps: float,
    *,
    eta: ArrayLike,
    has_shock: ArrayLike,
) -> Tuple[slice, ...]:
    """
    Compute the shock detector using the method of Berta et al. (2024) with a custom
    CuPy kernel, where a value of 0 indicates a smooth region and 1 indicates a shock.

    Args:
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        eta_threshold: Threshold value for eta above which a shock is detected.
        eps: Small value to avoid division by zero.
        eta: Array to which intermediate eta values are written. Has shape
            (nvars, nx, ny, nz, 3), where the last axis corresponds to eta computed
            along the x, y, and z dimensions, respectively. Slices along the last axis
            corresponding to inactive dimensions are left uninitialized.
        has_shock: Array to which the shock detector is written. Has shape
            (1, nx, ny, nz). Must be int32 dtype.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    nvars, nx, ny, nz = w1.shape
    threads_per_block = 256
    blocks_per_grid = (nx * ny * nz + threads_per_block - 1) // threads_per_block

    # check arrays are contiguous
    if not w1.flags["C_CONTIGUOUS"]:
        raise ValueError("w1 must be C-contiguous.")
    if not wr.flags["C_CONTIGUOUS"]:
        raise ValueError("wr must be C-contiguous.")
    if not eta.flags["C_CONTIGUOUS"]:
        raise ValueError("eta must be C-contiguous.")
    if not has_shock.flags["C_CONTIGUOUS"]:
        raise ValueError("has_shock must be C-contiguous.")

    compute_shocks_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (w1, wr, eta, has_shock, eta_threshold, eps, nvars, nx, ny, nz),
    )

    inner_slice = merge_slices(
        crop(0, (None, None), ndim=4),
        crop(1, (2, -2), ndim=4) if nx > 1 else crop(1, (None, None), ndim=4),
        crop(2, (2, -2), ndim=4) if ny > 1 else crop(2, (None, None), ndim=4),
        crop(3, (2, -2), ndim=4) if nz > 1 else crop(3, (None, None), ndim=4),
    )
    return inner_slice
