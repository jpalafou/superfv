from types import ModuleType

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike


def detect_PAD_violations(
    xp: ModuleType,
    wj: ArrayLike,
    physical_bounds: ArrayLike,
    tol: float = 1e-15,
    *,
    violation_amounts: ArrayLike,
    cell_violated: ArrayLike,
):
    """
    Detects physical admissibility violations in the input array `wj`.

    Args:
        xp: The array module to use (e.g., numpy or cupy).
        wj: The input array of shape (nvars, nx, ny, nz).
        physical_bounds: An array of shape (nvars, 2) containing the physical bounds
            for each variable.
        violation_amounts: Output array for the amount of violation for each variable.
            Has shape (nvars, nx, ny, nz).
        cell_violated: Output array for cell-wise violations. Has shape
            (1, nx, ny, nz). 1 indicates a violation, 0 indicates no violation.
        tol: A small tolerance value to determine if a violation has occurred. Default
            is 1e-15.
    """
    if hasattr(xp, "cuda"):
        PAD_kernel_helper(
            wj,
            physical_bounds,
            violation_amounts,
            cell_violated,
            tol=tol,
        )
        return

    if wj.ndim != 4:
        raise ValueError("wj must be 4D with shape (nvars, nx, ny, nz).")

    lower = physical_bounds[:, 0].reshape(-1, 1, 1, 1) - tol
    upper = physical_bounds[:, 1].reshape(-1, 1, 1, 1) + tol

    violation_amounts[...] = 0.0
    xp.minimum(wj - lower, violation_amounts, out=violation_amounts)
    xp.minimum(upper - wj, violation_amounts, out=violation_amounts)
    xp.any(violation_amounts, axis=0, keepdims=True, out=cell_violated)


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    PAD_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void PAD_kernel(
            const double* __restrict__ wj,
            const double* __restrict__ physical_bounds,
            double* __restrict__ violation_amounts,
            int* __restrict__ cell_violated,
            const double tol,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // wj                   : (nvars, nx, ny, nz)
            // physical_bounds      : (nvars, 2)
            // violation_amounts    : (nvars, nx, ny, nz)
            // cell_violated        : (1, nx, ny, nz)

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

                    const double lower_bound = physical_bounds[iv * 2] - tol;
                    const double upper_bound = physical_bounds[iv * 2 + 1] + tol;

                    double violation_amount = 0.0;
                    violation_amount = fmin(val - lower_bound, violation_amount);
                    violation_amount = fmin(upper_bound - val, violation_amount);

                    if (violation_amount < 0.0) {
                        violated = true;
                    }

                    violation_amounts[i] = violation_amount;
                }
                cell_violated[ixyz] = violated ? 1 : 0;
            }
        }
        """,
        "PAD_kernel",
    )

    def PAD_kernel_helper(
        wj: cp.ndarray,
        physical_bounds: cp.ndarray,
        violation_amounts: cp.ndarray,
        cell_violated: cp.ndarray,
        tol: float = 1e-15,
    ):
        """
        Helper function to launch the PAD kernel on the GPU.

        Args:
            wj: The input array of shape (nvars, nx, ny, nz)
            physical_bounds: An array of shape (nvars, 2) containing the physical
                bounds for each variable.
            violation_amounts: An array to which the violation amounts are written.
                Has shape (nvars, nx, ny, nz). Must be float64 dtype.
            cell_violated: An array to which the cell-wise violation mask is written.
                Has shape (1, nx, ny, nz). Must be int32 dtype.
            tol: A small tolerance value to determine if a violation has occurred.
        """
        if not wj.flags.c_contiguous:
            raise ValueError("wj must be C-contiguous.")
        if not physical_bounds.flags.c_contiguous:
            raise ValueError("physical_bounds must be C-contiguous.")
        if not violation_amounts.flags.c_contiguous:
            raise ValueError("violation_amounts must be C-contiguous.")
        if not cell_violated.flags.c_contiguous:
            raise ValueError("cell_violated must be C-contiguous.")
        if wj.ndim != 4 or violation_amounts.ndim != 4 or cell_violated.ndim != 4:
            raise ValueError(
                "wj, violation_amounts, and cell_violated must be 4D arrays."
            )
        if (
            physical_bounds.ndim != 2
            or physical_bounds.shape[0] != wj.shape[0]
            or physical_bounds.shape[1] != 2
        ):
            raise ValueError(
                "physical_bounds must be a 2D array with shape (nvars, 2)."
            )
        if not wj.dtype == cp.float64:
            raise ValueError("wj must be of dtype float64.")
        if not physical_bounds.dtype == cp.float64:
            raise ValueError("physical_bounds must be of dtype float64.")
        if violation_amounts.dtype != cp.float64:
            raise ValueError("violation_amounts must be of dtype float64.")
        if cell_violated.dtype != cp.int32:
            raise ValueError("cell_violated must be of dtype int32.")

        nvars, nx, ny, nz = wj.shape

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nx * ny * nz + threads_per_block - 1) // threads_per_block

        PAD_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                wj,
                physical_bounds,
                violation_amounts,
                cell_violated,
                tol,
                nvars,
                nx,
                ny,
                nz,
            ),
        )
