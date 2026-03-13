from typing import List, Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.sweep import stencil_sweep
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices


def update_eta_1d(
    w1: np.ndarray,
    wr: np.ndarray,
    eta: np.ndarray,
    dim: Literal["x", "y", "z"],
    eps: float = 1e-15,
) -> Tuple[slice, ...]:
    """`
    Compute the shock detector parameter in 1D using the method of Berta et al. (2024),
    writing the result to `eta`, taking the maximum with existing values.

    Args:
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        eta: Array to which the shock detector parameter is written, taking the maximum
            with existing values. Has shape (nvars, nx, ny, nz).
        dim: Dimension along which to compute eta. Must be "x", "y", or "z".
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    na = np.newaxis

    # allocate temporary arrays
    delta1 = np.empty(w1.shape + (1,))
    delta2 = np.empty(w1.shape + (1,))
    delta3 = np.empty(w1.shape + (1,))
    delta4 = np.empty(w1.shape + (1,))
    eta_o = np.empty(w1.shape + (1,))
    eta_e = np.empty(w1.shape + (1,))

    # compute stencils
    stencil1 = 0.5 * np.array([[-1.0, 0.0, 1.0]])
    stencil2 = np.array([[1.0, -2.0, 1.0]])
    stencil3 = 0.5 * np.array([[-1.0, 2.0, 0.0, -2.0, 1.0]])
    stencil4 = np.array([[1.0, -4.0, 6.0, -4.0, 1.0]])
    inner = crop(DIM_TO_AXIS[dim], (2, -2), ndim=4)

    # compute deltas using stencil sweeps
    stencil_sweep(w1[..., na], stencil1, delta1, dim)
    stencil_sweep(w1[..., na], stencil2, delta2, dim)
    stencil_sweep(w1[..., na], stencil3, delta3, dim)
    stencil_sweep(w1[..., na], stencil4, delta4, dim)

    # compute eta values
    eta_o[...] = np.abs(delta3)
    eta_o /= np.abs(wr[..., na]) + np.abs(delta1) + np.abs(delta3) + eps
    eta_e[...] = np.abs(delta4)
    eta_e /= np.abs(wr[..., na]) + np.abs(delta2) + np.abs(delta4) + eps
    np.maximum(eta_o[..., 0][inner], eta[inner], out=eta[inner])
    np.maximum(eta_e[..., 0][inner], eta[inner], out=eta[inner])

    return inner


def detect_shocks(
    w1: ArrayLike,
    wr: ArrayLike,
    eta: ArrayLike,
    has_shock: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    eta_threshold: float,
    eps: float = 1e-15,
) -> Tuple[slice, ...]:
    """
    Detect shocks in the primitive variable array `w1`, writing the results to
    `eta` and `has_shock`.

    Args:
        w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
        wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
        eta: Array to which the maximum shock detector parameter across all
            `active_dims` and variables is written. Has shape (nvars, nx, ny, nz).
        has_shock: Array to which the shock detection mask is written. Has shape
            (1, nx, ny, nz).
        active_dims: Tuple of dimensions along which to compute the shock detector.
            Each dimension must be "x", "y", or "z".
        eta_threshold: Threshold value for eta above which a shock is detected.
        eps: Small value to avoid division by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    if CUPY_AVAILABLE and isinstance(w1, cp.ndarray):
        return detect_shocks_kernel_helper(w1, wr, eta, has_shock, eta_threshold, eps)

    eta[...] = 0.0
    valid_slices: List[Tuple[slice, ...]] = []
    for dim in active_dims:
        valid = update_eta_1d(w1, wr, eta, dim, eps)
        valid_slices.append(valid)

    valid = merge_slices(*valid_slices)
    has_shock[valid] = np.any(eta[valid] > eta_threshold, axis=0)
    return valid


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    detect_shocks_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void detect_shocks_kernel(
            const double* __restrict__ w,
            const double* __restrict__ wref,
            double* __restrict__ eta,
            int* __restrict__ has_shock,
            const double eta_threshold,
            const double eps,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // w shape          (nvars, nx, ny, nz)
            // wref shape       (nvars, nx, ny, nz)
            // eta shape        (nvars, nx, ny, nz)
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
                        long long i = iv * nxyz + ixyz;

                        // initialize eta
                        eta[i] = 0.0;

                        // compute neighbor indices
                        long long j0, j1, j2, j3, j4;
                        switch (id) {
                            case 0:
                                j0 = (((long long)iv * nx + (ix - 2)) * ny + iy) * nz + iz;
                                j1 = (((long long)iv * nx + (ix - 1)) * ny + iy) * nz + iz;
                                j2 = i;
                                j3 = (((long long)iv * nx + (ix + 1)) * ny + iy) * nz + iz;
                                j4 = (((long long)iv * nx + (ix + 2)) * ny + iy) * nz + iz;
                                break;
                            case 1:
                                j0 = (((long long)iv * nx + ix) * ny + (iy - 2)) * nz + iz;
                                j1 = (((long long)iv * nx + ix) * ny + (iy - 1)) * nz + iz;
                                j2 = i;
                                j3 = (((long long)iv * nx + ix) * ny + (iy + 1)) * nz + iz;
                                j4 = (((long long)iv * nx + ix) * ny + (iy + 2)) * nz + iz;
                                break;
                            case 2:
                                j0 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 2);
                                j1 = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 1);
                                j2 = i;
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

                        double eta_val = fmax(eta_o, eta_e);
                        if (eta_val > eta[i]) {
                            eta[i] = eta_val;
                        }

                        // set to 1 if eta exceeds threshold
                        if (!has_shock[ixyz] && eta[i] > eta_threshold) {
                            has_shock[ixyz] = 1;
                        }
                    }
                }
            }
        }
        """,
        name="detect_shocks_kernel",
    )

    def detect_shocks_kernel_helper(
        w1: cp.ndarray,
        wr: cp.ndarray,
        eta: cp.ndarray,
        has_shock: cp.ndarray,
        eta_threshold: float,
        eps: float,
    ) -> Tuple[slice, ...]:
        """
        Compute the shock detector using the method of Berta et al. (2024) with a
        custom CuPy kernel, where a value of 0 indicates a smooth region and 1
        indicates a shock.

        Args:
            w1: Array of lazy primitive variables. Has shape (nvars, nx, ny, nz).
            wr: Reference array of primitive variables. Has shape (nvars, nx, ny, nz).
            eta: Array to which intermediate eta values are written. Has shape
                (nvars, nx, ny, nz, 3), where the last axis corresponds to eta computed
                along the x, y, and z dimensions, respectively. Slices along the last
                axis corresponding to inactive dimensions are left uninitialized.
            has_shock: Array to which the shock detector is written. Has shape
                (1, nx, ny, nz). Must be int32 dtype.
            eta_threshold: Threshold value for eta above which a shock is detected.
            eps: Small value to avoid division by zero.

        Returns:
            Slice objects indicating the modified regions in the output array.
        """
        # check arrays are contiguous
        if not w1.flags["C_CONTIGUOUS"]:
            raise ValueError("w1 must be C-contiguous.")
        if not wr.flags["C_CONTIGUOUS"]:
            raise ValueError("wr must be C-contiguous.")
        if not eta.flags["C_CONTIGUOUS"]:
            raise ValueError("eta must be C-contiguous.")
        if not has_shock.flags["C_CONTIGUOUS"]:
            raise ValueError("has_shock must be C-contiguous.")
        if has_shock.dtype != cp.int32:
            raise ValueError("has_shock must be of dtype int32.")
        if eta.dtype != cp.float64:
            raise ValueError("eta must be of dtype float64.")
        if w1.dtype != cp.float64 or wr.dtype != cp.float64:
            raise ValueError("w1 and wr must be of dtype float64.")

        nvars, nx, ny, nz = w1.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nx * ny * nz + threads_per_block - 1) // threads_per_block

        detect_shocks_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (w1, wr, eta, has_shock, eta_threshold, eps, nvars, nx, ny, nz),
        )

        return (
            slice(None),
            slice(2, -2) if nx > 1 else slice(None),
            slice(2, -2) if ny > 1 else slice(None),
            slice(2, -2) if nz > 1 else slice(None),
        )
