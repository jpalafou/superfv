from types import ModuleType
from typing import Tuple, cast

import numpy as np

from superfv.tools.device_management import CUPY_AVAILABLE
from superfv.tools.slicing import crop, insert_slice


def stencil_sweep(
    xp: ModuleType, u: np.ndarray, weights: np.ndarray, axis: int, *, out: np.ndarray
) -> Tuple[slice, ...]:
    """
    Perform a stencil sweep along the specified axis.

    Args:
        xp: numpy or cupy module corresponding to the device on which the arrays are
            located.
        u: Input array to be swept, has shape (nvars, nx, ny, nz).
        weights: Stencil weights for the sweep, has shape (nstencils, stencil_size).
        axis: Axis along which to perform the sweep (1 for x, 2 for y, 3 for z).
        out: Output array to store the results of the sweep, has shape
            (nvars, nx, ny, nz, nstencils).

    Returns:
        Slice objects indicating the modified regions of `out` after the sweep.
    """
    if hasattr(xp, "cuda"):
        return sweep_kernel_helper(u, weights, out, axis)

    _, stencil_size = weights.shape

    slices = [
        crop(axis, (i, i - stencil_size + 1), ndim=4) for i in range(stencil_size)
    ]
    modified = slices[stencil_size // 2]

    out[modified] = 0.0
    for same_pos_weights, slc in zip(weights.T, slices):
        out[modified] += u[slc][..., np.newaxis] * same_pos_weights

    return cast(Tuple[slice, ...], insert_slice(modified, 4, slice(None)))


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    sweep_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void sweep_kernel(
            const double* __restrict__ u,
            const double* __restrict__ weights,
            double* __restrict__ uj,
            const int dim,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterpolations,
            const int nkernel
        ){
            // u    s   hape (nvars, nx, ny, nz)
            // weights  shape (ninterpolations, nkernel)
            // uj       shape (nvars, nx, ny, nz, ninterpolations)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;
            const int reach = (nkernel - 1) / 2;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t;

                // skip threads which reach out of bounds
                switch (dim) {
                    case 1: if (ix < reach || ix >= nx - reach) continue; break;
                    case 2: if (iy < reach || iy >= ny - reach) continue; break;
                    case 3: if (iz < reach || iz >= nz - reach) continue; break;
                    default: return; // invalid dimension
                }


                for (int qj = 0; qj < ninterpolations; qj++) {
                    long long j = ((((long long)iv * nx + ix) * ny + iy) * nz + iz)
                        * ninterpolations + qj;

                    double result = 0.0;

                    for (int ik = 0; ik < nkernel; ik++) {
                        // compute neighbor index
                        int off = ik - reach;
                        int kv = iv, kx = ix, ky = iy, kz = iz;
                        switch (dim) {
                            case 1: kx += off; break;
                            case 2: ky += off; break;
                            case 3: kz += off; break;
                        }
                        long long k = (((long long)kv * nx + kx) * ny + ky) * nz + kz;

                        // compute weight index
                        long long wj = (long long) qj * nkernel + ik;

                        result += weights[wj] * u[k];
                    }

                    // update output array
                    uj[j] = result;
                }
            }
        }
        """,
        name="sweep_kernel",
    )

    def sweep_kernel_helper(
        u: cp.ndarray, weights: cp.ndarray, uj: cp.ndarray, axis: int
    ) -> Tuple[slice, ...]:
        if not u.flags.c_contiguous or not u.ndim == 4:
            raise ValueError("Array `u` must be a C-contiguous, 4-dimensional array.")
        if not weights.flags.c_contiguous or not weights.ndim == 2:
            raise ValueError(
                "Array `weights` must be a C-contiguous, 2-dimensional array."
            )
        if not uj.flags.c_contiguous or not uj.ndim == 5:
            raise ValueError("Array `uj` must be a C-contiguous, 5-dimensional array.")
        if not uj.shape[:4] == u.shape or not uj.shape[4] == weights.shape[0]:
            raise ValueError(
                "Array `uj` must have shape (nvars, nx, ny, nz, ninterpolations)"
            )
        if (
            not u.dtype == cp.float64
            or not weights.dtype == cp.float64
            or not uj.dtype == cp.float64
        ):
            raise ValueError("All input arrays must have dtype float64.")

        nvars, nx, ny, nz = u.shape
        ninterpolations, nkernel = weights.shape
        reach = (nkernel - 1) // 2

        threads_per_block = 256
        blocks_per_grid = min(
            (nvars * nx * ny * nz + threads_per_block - 1) // threads_per_block,
            65535,
        )
        sweep_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (u, weights, uj, axis, nvars, nx, ny, nz, ninterpolations, nkernel),
        )

        return cast(
            Tuple[slice, ...],
            insert_slice(crop(axis, (reach, -reach), ndim=4), 4, slice(None)),
        )
