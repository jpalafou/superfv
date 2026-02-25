from typing import Literal, Optional, Tuple

from superfv.axes import DIM_TO_AXIS
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    interpolate_central_quantity_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void interpolate_central_quantity_kernel(
            const double* __restrict__ u,
            double* __restrict__ uj,
            const int mode,
            const int p,
            const int dim,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u    shape (nvars, nx, ny, nz)
            // uj   shape (nvars, nx, ny, nz)
            // mode 0 for cell-center interpolation, 1 for finite-volume integration
            // p    polynomial degree {0, ..., 7}
            // dim  dimension to interpolate along {0, 1, 2} for (x, y, z, respectively)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            // assign conservative interpolation or trasnverse integral weights
            double w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0, w4 = 0.0, w5 = 0.0, w6 = 0.0;
            int size, reach;
            if (mode == 0) { // interpolate
                if (p <= 1) {
                    w0 = 1.0;
                    size = 1;
                } else if (p <= 3) {
                    w0 = -1.0 / 24.0;
                    w1 = 13.0 / 12.0;
                    w2 = w0;
                    size = 3;
                } else if (p <= 5) {
                    w0 = 3.0 / 640.0;
                    w1 = -29.0 / 480.0;
                    w2 = 1067.0 / 960.0;
                    w3 = w1, w4 = w0;
                    size = 5;
                } else if (p <= 7) {
                    w0 = -5.0 / 7168.0;
                    w1 = 159.0 / 17920;
                    w2 = -7621.0 / 107520.0;
                    w3 = 30251.0 / 26880.0;
                    w4 = w2, w5 = w1, w6 = w0;
                    size = 7;
                } else {
                    // higher-order interpolation not implemented
                    return;
                }
            } else if (mode == 1) { // integrate
                if (p <= 1) {
                    w0 = 1.0;
                    size = 1;
                } else if (p <= 3) {
                    w0 = 1.0 / 24.0;
                    w1 = 11.0 / 12.0;
                    w2 = w0;
                    size = 3;
                } else if (p <= 5) {
                    w0 = -17.0 / 5760.0;
                    w1 = 77.0  / 1440.0;
                    w2 = 863.0 / 960.0;
                    w3 = w1, w4 = w0;
                    size = 5;
                } else if (p <= 7) {
                    w0 = 367.0 / 967680.0;
                    w1 = -281.0 / 53760.0;
                    w2 = 6361.0 / 107520.0;
                    w3 = 215641.0 / 241920.0;
                    w4 = w2, w5 = w1, w6 = w0;
                    size = 7;
                }
            } else {
                return; // invalid mode
            }
            reach = (size - 1) / 2;

            long long ntotal = (long long)nvars*(long long)nx*(long long)ny*(long long)nz;
            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = t % nz; t /= nz;
                int iy = t % ny; t /= ny;
                int ix = t % nx; t /= nx;
                int iv = t;

                // skip threads which reach out of bounds
                switch (dim) {
                    case 0: if (ix < reach || ix >= nx - reach) continue; break;
                    case 1: if (iy < reach || iy >= ny - reach) continue; break;
                    case 2: if (iz < reach || iz >= nz - reach) continue; break;
                    default: return; // invalid dimension
                }

                double result = 0.0;
                for (int q = 0; q < size; q++) {
                    // get node index j
                    int off = q - reach;
                    int jv = iv;
                    int jx = ix, jy = iy, jz = iz;
                    switch (dim) {
                        case 0: jx += off; break;
                        case 1: jy += off; break;
                        case 2: jz += off; break;
                    }
                    long long j = ((((long long)jv * nx + jx) * ny + jy) * nz + jz);

                    // get stencil weights
                    double w;
                    switch (q) {
                        case 0: w = w0; break;
                        case 1: w = w1; break;
                        case 2: w = w2; break;
                        case 3: w = w3; break;
                        case 4: w = w4; break;
                        case 5: w = w5; break;
                        case 6: w = w6; break;
                    }

                    // update result
                    result += w * u[j];
                }
                // update output array
                uj[i] = result;
            }
        }
        """,
        "interpolate_central_quantity_kernel",
    )

    gauss_legendre_quadrature_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void gauss_legendre_quadrature_kernel(
            const double* wj,
            double* out,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int nquadrature
        ){
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            double w0=0, w1=0, w2=0, w3=0, w4=0;
            switch (nquadrature) {
                case 1:
                    w0 = 1.0;
                    break;
                case 2:
                    w0 = 0.5;
                    w1 = w0;
                    break;
                case 3:
                    w0 = 5.0 / 18.0;
                    w1 = 4.0 / 9.0;
                    w2 = w0;
                    break;
                case 4:
                    w0 = (18.0 - sqrt(30.0)) / 72.0;
                    w1 = (18.0 + sqrt(30.0)) / 72.0;
                    w2 = w1;
                    w3 = w0;
                    break;
                case 5:
                    w0 = (322.0 - 13.0 * sqrt(70.0)) / 1800.0;
                    w1 = (322.0 + 13.0 * sqrt(70.0)) / 1800.0;
                    w2 = 64.0 / 225;
                    w3 = w1;
                    w4 = w0;
                    break;
                default:
                    return;
                }

            for (long long i = tid; i < ntotal; i += stride) {
                const double* row = wj + ((size_t)i * nquadrature);
                double result = 0.0;
                for (int j = 0; j < nquadrature; j++) {
                    double wq = row[j];
                    switch (j) {
                        case 0: result += w0 * wq; break;
                        case 1: result += w1 * wq; break;
                        case 2: result += w2 * wq; break;
                        case 3: result += w3 * wq; break;
                        case 4: result += w4 * wq; break;
                    }
                }
                out[i] = result;
            }
    }
        """,
        "gauss_legendre_quadrature_kernel",
    )

    def interpolate_central_quantity_kernel_helper(
        u: ArrayLike,
        uj: ArrayLike,
        mode: Literal[0, 1],
        p: int,
        dim: Literal["x", "y", "z"],
    ) -> Tuple[slice, ...]:
        nvars, nx, ny, nz = u.shape
        dim_int = {"x": 0, "y": 1, "z": 2}[dim]
        reach = p // 2

        if mode not in {0, 1}:
            raise ValueError("Mode must be 0 for interpolation or 1 for integration")
        if p not in {0, 1, 2, 3, 4, 5, 6, 7}:
            raise ValueError(
                "Polynomial degree p must be an integer in the range [0, 7]"
            )
        if not u.flags.c_contiguous:
            raise ValueError("Input array u must be C-contiguous")
        if not uj.flags.c_contiguous:
            raise ValueError("Output array uj must be C-contiguous")
        if u.dtype != cp.float64:
            raise ValueError("Input array u must be of type float64")
        if uj.dtype != cp.float64:
            raise ValueError("Output array uj must be of type float64")

        n = u.size
        threads = 256
        blocks = (n + threads - 1) // threads

        interpolate_central_quantity_kernel(
            (blocks,), (threads,), (u, uj, mode, p, dim_int, nvars, nx, ny, nz)
        )
        return crop(DIM_TO_AXIS[dim], (reach, -reach), ndim=4)


def interpolate_central_quantity(
    u: ArrayLike,
    uj: ArrayLike,
    mode: Literal[0, 1],
    p: int,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    uu: Optional[ArrayLike] = None,
    uuu: Optional[ArrayLike] = None,
) -> Tuple[slice, ...]:
    """
    Perform a central sweep interpolation or integration along the specified
    active dimensions.

    Args:
        u: Input array of shape (nvars, nx, ny, nz) containing the original data.
        uj: Output array of shape (nvars, nx, ny, nz) to store the results of the
            interpolation/integration.
        mode: 0 for cell-center interpolation, 1 for finite-volume integration.
        p: Polynomial degree for the interpolation/integration (0 to 7).
        active_dims: Tuple of active dimensions to perform the sweep along, each being
            "x", "y", or "z". The length of this tuple determines the number of sweeps.
        uu: Optional intermediate array for 2D interpolation with shape
            (nvars, nx, ny, nz).
        uuu: Optional intermediate array for 3D interpolation with shape
            (nvars, nx, ny, nz).

    Returns:
        A tuple of slices corresponding to the valid region of the output array after
        performing the central sweep along the specified dimensions.
    """
    ndim = len(active_dims)

    if ndim == 1:
        return interpolate_central_quantity_kernel_helper(
            u, uj, mode, p, active_dims[0]
        )

    if ndim == 2:
        if uu is None:
            raise ValueError(
                "Intermediate array uu must be provided for 2D interpolation"
            )
        slc1 = interpolate_central_quantity_kernel_helper(
            u, uu, mode, p, active_dims[0]
        )
        slc2 = interpolate_central_quantity_kernel_helper(
            uu, uj, mode, p, active_dims[1]
        )
        return merge_slices(slc1, slc2)

    if ndim == 3:
        if uu is None or uuu is None:
            raise ValueError(
                "Intermediate arrays uu and uuu must be provided for 3D interpolation"
            )
        slc1 = interpolate_central_quantity_kernel_helper(
            u, uu, mode, p, active_dims[0]
        )
        slc2 = interpolate_central_quantity_kernel_helper(
            uu, uuu, mode, p, active_dims[1]
        )
        slc3 = interpolate_central_quantity_kernel_helper(
            uuu, uj, mode, p, active_dims[2]
        )
        return merge_slices(slc1, slc2, slc3)

    raise ValueError("active_dims must have length 1, 2, or 3")


def gauss_legendre_quadrature_kernel_helper(u: ArrayLike, p: int, out: ArrayLike):
    nquadrature = -(-(p + 1) // 2)

    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous for CUDA kernel.")
    if not u[..., :nquadrature].flags.c_contiguous:
        raise ValueError("Input array must be C-contiguous for CUDA kernel.")

    nvars, nx, ny, nz, _ = u.shape
    threads_per_block = 256
    blocks_per_grid = (
        nvars * nx * ny * nz + threads_per_block - 1
    ) // threads_per_block
    gauss_legendre_quadrature_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (u[..., :nquadrature], out, nvars, nx, ny, nz, nquadrature),
    )
