from types import ModuleType

from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike


def perform_quadrature(
    xp: ModuleType, uj: ArrayLike, weights: ArrayLike, out: ArrayLike
):
    """
    Multiply the nodes of `u` by the corresponding quadrature weights and take the sum.

    Args:
        xp: numpy or cupy module corresponding to the device on which the arrays are
            located.
        uj: Input array with shape (nvars, nx, ny, nz, ninterps).
        weights: Quadrature weights with shape (ninterps,).
        out: Output array to store the results of the quadrature, has shape
            (nvars, nx, ny, nz).
    """
    xp.sum(uj * weights.reshape(1, 1, 1, 1, -1), axis=4, out=out)


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    quadrature_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void quadrature_kernel(
            const double* __restrict__ uj,
            const double* __restrict__ weights,
            double* __restrict__ out,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterps,
        ){
            // uj       shape (nvars, nx, ny, nz, ninterps)
            // weights  shape (ninterps)
            // out      shape (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            for (long long i = tid; i < ntotal; i += stride) {
                double result = 0.0;
                for (int qj = 0; qj < ninterps; j++) {
                    long long j = ntotal * ninterps + qj;
                    result += weights[qj] * uj[j];
                }
                out[i] = result;
            }
        }
        """,
        name="quadrature_kernel",
    )

    def quadarture_kernel_helper(uj: cp.ndarray, weights: cp.ndarray, out: cp.ndarray):
        if not uj.flags.c_contiguous or uj.ndim != 5:
            raise ValueError("Array `uj` must be a C-contiguous, 5-dimensional array.")

        nvars, nx, ny, nz, ninterps = uj.shape

        if not weights.flags.c_contiguous or weights.shape == (ninterps,):
            raise ValueError(
                "Array `weights` must be a C-contiguous array of shape (ninterps,)."
            )
        if not out.flags.c_contiguous or out.shape == uj.shape[:4]:
            raise ValueError(
                "Array `out` must be a C-contiguous array of shape "
                "(nvars, nx, ny, nz)."
            )
        if (
            uj.dtype != cp.float64
            or weights.dtype != cp.float64
            or out.dtype != cp.float64
        ):
            raise ValueError("All input arrays must have dtype float64.")

        threads_per_block = 256
        ntotal = nvars * nx * ny * nz
        blocks_per_grid = min(
            (ntotal + threads_per_block - 1) // threads_per_block,
            65535,
        )
        quadrature_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (uj, weights, out, nvars, nx, ny, nz, ninterps),
        )
