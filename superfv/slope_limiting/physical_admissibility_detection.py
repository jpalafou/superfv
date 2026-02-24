from types import ModuleType

from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike


def detect_PAD_violations(
    xp: ModuleType,
    wj: ArrayLike,
    physical_bounds: ArrayLike,
    tol: float = 1e-15,
    *,
    violated_vars: ArrayLike,
    violated_cells: ArrayLike,
):
    """
    Detects physical admissibility violations in the input array `wj`.

    Args:
        xp: The array module to use (e.g., numpy or cupy).
        wj: The input array of shape (nvars, nx, ny, nz).
        physical_bounds: An array of shape (nvars, 2) containing the physical bounds
            for each variable.
        violated_vars: Output array for variable-wise violations. Has shape
            (nvars, nx, ny, nz). 1 indicates a violation, 0 indicates no violation.
        violated_cells: Output array for cell-wise violations. Has shape
            (1, nx, ny, nz). 1 indicates a violation, 0 indicates no violation.
        tol: A small tolerance value to determine if a violation has occurred. Default
            is 1e-15.
    """
    if hasattr(xp, "cuda"):
        PAD_kernel_helper(
            wj,
            physical_bounds,
            violated_vars=violated_vars,
            violated_cells=violated_cells,
            tol=tol,
        )
        return

    if wj.ndim != 4:
        raise ValueError("wj must be 4D with shape (nvars, nx, ny, nz).")

    lower = physical_bounds[:, 0].reshape(-1, 1, 1, 1)
    upper = physical_bounds[:, 1].reshape(-1, 1, 1, 1)

    violated_vars[...] = wj < lower + tol
    xp.logical_or(violated_vars, wj > upper - tol, out=violated_vars)
    violated_cells[...] = xp.any(violated_vars > 0, axis=0)


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    PAD_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void PAD_kernel(
            const double* __restrict__ wj,
            const double* __restrict__ physical_bounds,
            int* __restrict__ violated_vars,
            int* __restrict__ violated_cells,
            const double tol,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // wj               : (nvars, nx, ny, nz)
            // physical_bounds  : (nvars, 2)
            // violated_cells   : (nx, ny, nz)
            // violated_vars    : (nvars, nx, ny, nz)

            const long long tid    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long nxyz = (long long)nx * (long long)ny * (long long)nz;

            for (long long ixyz = tid; ixyz < nxyz; ixyz += stride) {
                long long t = ixyz;
                const int iz = t % nz; t /= nz;
                const int iy = t % ny; t /= ny;
                const int ix = t % nx; t /= nx;

                bool violated = false;

                for (int iv = 0; iv < nvars; iv++) {
                    const long long i = (((long long)iv * nx + ix) * ny + iy) * nz + iz;
                    const double val = wj[i];

                    const double lower_bound = physical_bounds[iv * 2];
                    const double upper_bound = physical_bounds[iv * 2 + 1];

                    const int var_violated = (val < lower_bound + tol
                        || val > upper_bound - tol) ? 1 : 0;
                    violated_vars[i] = var_violated;

                    if (var_violated) {
                        violated = true;
                    }
                }

                violated_cells[ixyz] = violated ? 1 : 0;
            }
        }
        """,
        "PAD_kernel",
    )


def PAD_kernel_helper(
    wj: ArrayLike,
    physical_bounds: ArrayLike,
    *,
    violated_vars: ArrayLike,
    violated_cells: ArrayLike,
    tol: float = 1e-15,
):
    """
    Helper function to launch the PAD kernel on the GPU.

    Args:
        wj: The input array of shape (nvars, nx, ny, nz)
        physical_bounds: An array of shape (nvars, 2) containing the physical bounds
            for each variable.
        violated_vars: An array to which the variable-wise violation mask is written. Has shape
            (nvars, nx, ny, nz). Must be int32 dtype.
        violated_cells: An array to which the cell-wise violation mask is written. Has shape
            (1, nx, ny, nz). Must be int32 dtype.
        tol: A small tolerance value to determine if a violation has occurred.
    """
    if wj.ndim != 4:
        raise ValueError("wj must be 4D with shape (nvars, nx, ny, nz).")

    nvars, nx, ny, nz = wj.shape

    if not wj.flags.c_contiguous:
        raise ValueError("wj must be C-contiguous.")
    if not physical_bounds.flags.c_contiguous:
        raise ValueError("physical_bounds must be C-contiguous.")
    if not violated_vars.flags.c_contiguous:
        raise ValueError("violated_vars must be C-contiguous.")
    if not violated_cells.flags.c_contiguous:
        raise ValueError("violated_cells must be C-contiguous.")
    if violated_vars.dtype != cp.int32:
        raise ValueError("violated_vars must be of dtype int32.")
    if violated_cells.dtype != cp.int32:
        raise ValueError("violated_cells must be of dtype int32.")

    threads_per_block = 256
    blocks_per_grid = (nx * ny * nz + threads_per_block - 1) // threads_per_block

    PAD_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            wj,
            physical_bounds,
            violated_vars,
            violated_cells,
            tol,
            nvars,
            nx,
            ny,
            nz,
        ),
    )
