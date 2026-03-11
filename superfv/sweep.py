from itertools import product
from typing import Literal, Tuple

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, replace_slice


def stencil_sweep(
    u: ArrayLike,
    weights: ArrayLike,
    out: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> Tuple[slice, ...]:
    """
    Perform a sweep of stencils contained in `weights` along the specified dimension
    `dim` on the input array `u`, storing the results in `out`.

    Args:
        u: Input array with shape (nvars, nx, ny, nz, ninterps).
        weights: Input array with shape (nouterps, stencil_size).
        out: Array to which the result is written, has shape
            (nvars, nx, ny, nz, ninterps * nouterps).
        dim: Dimension to sweep along ("x", "y", "z").

    Returns:
        Slice objects indicating the modified regions of `out` after the sweep.
    """
    axis = DIM_TO_AXIS[dim]

    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        return sweep_kernel_helper(u, weights, out, axis)

    _, _, _, _, ninterps = u.shape
    nouterps, stencil_size = weights.shape

    slices = [
        crop(axis, (i, i - stencil_size + 1), ndim=5) for i in range(stencil_size)
    ]
    modified = slices[stencil_size // 2]

    out[modified] = 0.0
    for i, (same_pos_weights, slc) in product(range(ninterps), zip(weights.T, slices)):
        in_slc = replace_slice(slc, 4, slice(i, i + 1))
        out_slc = replace_slice(modified, 4, slice(i * nouterps, (i + 1) * nouterps))
        out[out_slc] += u[in_slc] * same_pos_weights

    return modified


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
            const int ninterps,
            const int nouterps,
            const int nkernel
        ){
            // u        shape (nvars, nx, ny, nz, ninterps)
            // weights  shape (nouterps, nkernel)
            // uj       shape (nvars, nx, ny, nz, ninterps * nouterps)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz * ninterps;
            const int reach = (nkernel - 1) / 2;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int ii = t % ninterps; t /= ninterps;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t % nvars;

                // skip threads which reach out of bounds
                switch (dim) {
                    case 1: if (ix < reach || ix >= nx - reach) continue; break;
                    case 2: if (iy < reach || iy >= ny - reach) continue; break;
                    case 3: if (iz < reach || iz >= nz - reach) continue; break;
                    default: return; // invalid dimension
                }


                for (int qj = 0; qj < nouterps; qj++) {
                    long long j = (((((long long)iv * nx + ix) * ny + iy) * nz + iz)
                        * ninterps + ii) * nouterps + qj;

                    double result = 0.0;

                    for (int ik = 0; ik < nkernel; ik++) {
                        // compute neighbor index
                        int off = ik - reach;
                        int kv = iv, kx = ix, ky = iy, kz = iz, ki = ii;
                        switch (dim) {
                            case 1: kx += off; break;
                            case 2: ky += off; break;
                            case 3: kz += off; break;
                        }
                        long long k = ((((long long)kv * nx + kx) * ny + ky) * nz + kz)
                            * ninterps + ki;

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
        if not u.flags.c_contiguous or u.ndim != 5:
            raise ValueError("Array `u` must be a C-contiguous, 5-dimensional array.")
        if not weights.flags.c_contiguous or weights.ndim != 2:
            raise ValueError(
                "Array `weights` must be a C-contiguous, 2-dimensional array."
            )

        nvars, nx, ny, nz, ninterps = u.shape
        nouterps, nkernel = weights.shape

        if not uj.flags.c_contiguous or uj.ndim != 5:
            raise ValueError("Array `uj` must be a C-contiguous, 5-dimensional array.")
        if uj.shape[:4] != u.shape[:4] or uj.shape[4] != ninterps * nouterps:
            raise ValueError(
                "Array `uj` must have shape (nvars, nx, ny, nz, ninterps * nouterps)"
            )
        if (
            u.dtype != cp.float64
            or weights.dtype != cp.float64
            or uj.dtype != cp.float64
        ):
            raise ValueError("All input arrays must have dtype float64.")

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        ntotal = nvars * nx * ny * nz * ninterps
        blocks_per_grid = (ntotal + threads_per_block - 1) // threads_per_block

        sweep_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (u, weights, uj, axis, nvars, nx, ny, nz, ninterps, nouterps, nkernel),
        )

        reach = (nkernel - 1) // 2
        return crop(axis, (reach, -reach), ndim=5)
