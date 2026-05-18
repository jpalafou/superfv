from __future__ import annotations

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike


def compute_theta(
    w: ArrayLike,
    wj: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    Mj: ArrayLike,
    mj: ArrayLike,
    theta: ArrayLike,
    tol: float = 1e-15,
):
    """
    Compute Zhang and Shu's a priori slope limiting parameter and write it to `theta`
    while alpha updating the nodal maxima and minima, `Mj` and `mj`, respectively.
    Renders a single ghost cell layer along each active dimension of the output arrays
    invalid.

    Args:
        w: Array of cell-average values with shape (nvars, nx, ny, nz).
        wj: Array of high-order interpolation values with shape
            (nvars, nx, ny, nz, ninterps).
        M: Array of local maxima with shape (nvars, nx, ny, nz).
        m: Array of local minima with shape (nvars, nx, ny, nz).
        Mj: Array to which nodal maxima are written. Has shape (nvars, nx, ny, nz).
        mj: Array to which nodal minima are written. Has shape (nvars, nx, ny, nz).
        theta: Array to which the Zhang-Shu limiter is written. Has shape
            (nvars, nx, ny, nz).
        tol: Tolerance to avoidi 0 in the denominator of the theta calculation.
    """
    if CUPY_AVAILABLE and isinstance(w, cp.ndarray):
        compute_theta_kernel_helper(w, wj, M, m, Mj, mj, theta, tol)
        return

    # compute nodal maximum principle
    np.max(wj, axis=4, out=Mj)
    np.min(wj, axis=4, out=mj)

    # compute theta
    theta[...] = np.minimum(
        np.minimum(
            np.divide(np.abs(M - w), np.abs(Mj - w) + tol),
            np.divide(np.abs(m - w), np.abs(mj - w) + tol),
        ),
        1.0,
    )


def zhang_shu_operator(wj: ArrayLike, w: ArrayLike, theta: ArrayLike):
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        wj: Array of high-order interpolation values that is revised.
        w: Array of first-order interpolation values.
        theta: Array of limiting coefficients.
    """
    wj[...] = theta * (wj - w) + w


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    compute_theta_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_theta_kernel(
            const double* __restrict__ w,
            const double* __restrict__ wj,
            const double* __restrict__ M,
            const double* __restrict__ m,
            double* __restrict__ Mj,
            double* __restrict__ mj,
            double* __restrict__ theta,
            const double eps,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterps
        ) {
            // w        has shape (nvars, nx, ny, nz)
            // wj       has shape (nvars, nx, ny, nz, ninterps)
            // M, m     have shape (nvars, nx, ny, nz)
            // Mj, mj   have shape (nvars, nx, ny, nz)
            // theta    has shape (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            long long n = (long long)nvars * nx * ny * nz;

            for (long long i = tid; i < n; i += stride) {

                const double* row = wj + i * (long long)ninterps;

                double mj_val = row[0];
                double Mj_val = row[0];

                for (int ii = 1; ii < ninterps; ++ii) {
                    const double vj = row[ii];
                    mj_val = (vj < mj_val) ? vj : mj_val;
                    Mj_val = (vj > Mj_val) ? vj : Mj_val;
                }

                mj[i] = mj_val;
                Mj[i] = Mj_val;
                double theta_M = fabs(M[i] - w[i]) / (fabs(Mj_val - w[i]) + eps);
                double theta_m = fabs(m[i] - w[i]) / (fabs(mj_val - w[i]) + eps);
                theta[i] = fmin(1.0, fmin(theta_M, theta_m));
            }
        }
        """,
        "compute_theta_kernel",
    )

    def compute_theta_kernel_helper(
        w: cp.ndarray,
        wj: cp.ndarray,
        M: cp.ndarray,
        m: cp.ndarray,
        Mj: cp.ndarray,
        mj: cp.ndarray,
        theta: cp.ndarray,
        eps: float,
    ):
        if not w.flags.c_contiguous or w.ndim != 4:
            raise ValueError("Array `w` must be a C-contiguous, 4-dimensional array.")
        if not wj.flags.c_contiguous or wj.ndim != 5 or wj.shape[:4] != w.shape:
            raise ValueError(
                "Array `wj` must be a C-contiguous, 5-dimensional array of shape "
                "(nvars, nx, ny, nz)."
            )
        if not M.flags.c_contiguous or M.shape != w.shape:
            raise ValueError("Array `M` must be a C-contiguous array of the same shape as `w`.")
        if not m.flags.c_contiguous or m.shape != w.shape:
            raise ValueError("Array `m` must be a C-contiguous array of the same shape as `w`.")
        if not Mj.flags.c_contiguous or Mj.shape != w.shape:
            raise ValueError("Array `Mj` must be a C-contiguous array of the same shape as `w`.")
        if not mj.flags.c_contiguous or mj.shape != w.shape:
            raise ValueError("Array `mj` must be a C-contiguous array of the same shape as `w`.")
        if not theta.flags.c_contiguous or theta.shape != w.shape:
            raise ValueError("Array `theta` must be a C-contiguous array of the same shape as `w`.")
        if (
            w.dtype != cp.float64
            or wj.dtype != cp.float64
            or M.dtype != cp.float64
            or m.dtype != cp.float64
            or Mj.dtype != cp.float64
            or mj.dtype != cp.float64
            or theta.dtype != cp.float64
        ):
            raise ValueError("All input arrays must have dtype float64.")

        nvars, nx, ny, nz = w.shape
        _, _, _, _, ninterps = wj.shape

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nvars * nx + threads_per_block - 1) // threads_per_block

        compute_theta_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (w, wj, M, m, Mj, mj, theta, eps, nvars, nx, ny, nz, ninterps),
        )
