from typing import List, Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices
from superfv.tools.stability import avoid0


def central_difference(u: np.ndarray, out: np.ndarray, axis: int):
    """
    Compute the 1D central difference of `u` along `axis`, writing the result to `out`.
    """
    out[crop(axis, (1, -1))] = 0.5 * (
        u[crop(axis, (2, None))] - u[crop(axis, (None, -2))]
    )


def update_alpha_1d(
    u: np.ndarray,
    alpha: np.ndarray,
    dim: Literal["x", "y", "z"],
    check_uniformity: bool,
    uniformity_tol: float = 1e-3,
    eps: float = 1e-15,
) -> Tuple[slice, ...]:
    """
    Compute the 1D smooth extrema detector for array `u` along dimension `dim`,
    writing the result to `alpha`, taking the minimum with the existing values in
    `alpha`.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        alpha: Array to which the computed smooth extrema detector is written, taking
            the minimum with existing values. Has shape (nvars, nx, ny, nz).
        dim: Dimension along which to compute the smooth extrema detector. One of
            "x", "y", or "z".
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        uniformity_tol: Tolerance used to detect uniform regions.
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # retrieve axis
    axis = DIM_TO_AXIS[dim]

    # valid inner region
    inner = crop(axis, (3, -3))

    # allocate arrays
    du = np.empty_like(u)
    dv = np.empty_like(u)
    vl = np.empty_like(u)
    vr = np.empty_like(u)
    alpha_l = np.empty_like(u)
    alpha_r = np.empty_like(u)
    alpha_neighbors = np.empty_like(u)
    dmp_m = np.empty_like(u)
    dmp_M = np.empty_like(u)
    uniform = np.empty_like(u)

    # compute derivatives
    central_difference(u, du, axis)
    central_difference(du, dv, axis)
    dv[...] = avoid0(0.5 * dv, eps)

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (None, -2))] - du[crop(axis, (1, -1))]
    alpha_l[...] = -np.where(dv < 0, np.maximum(vl, 0), np.minimum(vl, 0)) / dv

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, None))] - du[crop(axis, (1, -1))]
    alpha_r[...] = np.where(dv > 0, np.maximum(vr, 0), np.minimum(vr, 0)) / dv

    # combine left and right detectors
    np.minimum(alpha_l, alpha_r, out=alpha_neighbors)
    np.minimum(alpha_neighbors, 1.0, out=alpha_neighbors)

    # relax alpha in uniform regions
    if check_uniformity:
        lft_slc = crop(axis, (2, None))
        ctr_slc = crop(axis, (1, -1))
        rgt_slc = crop(axis, (None, -2))

        dmp_m[ctr_slc] = np.minimum(np.minimum(u[lft_slc], u[ctr_slc]), u[rgt_slc])
        dmp_M[ctr_slc] = np.maximum(np.maximum(u[lft_slc], u[ctr_slc]), u[rgt_slc])

        uniform[...] = np.abs(dmp_M - dmp_m) <= uniformity_tol * np.abs(u)

        np.maximum(alpha_neighbors, uniform, out=alpha_neighbors)

    # take min of neighbors and return
    left = crop(axis, (2, -4), ndim=4)
    inner = crop(axis, (3, -3), ndim=4)
    right = crop(axis, (4, -2), ndim=4)

    np.minimum(alpha_neighbors[left], alpha[inner], out=alpha[inner])
    np.minimum(alpha_neighbors[inner], alpha[inner], out=alpha[inner])
    np.minimum(alpha_neighbors[right], alpha[inner], out=alpha[inner])

    return inner


def compute_alpha(
    u: ArrayLike,
    alpha: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    check_uniformity: bool,
    uniformity_tol: float = 1e-3,
    eps: float = 1e-15,
) -> Tuple[slice, ...]:
    """
    Compute the smooth extrema detector for array `u` along all `active_dims`,
    writing the result to `alpha`.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        alpha: Array to which the minimum smooth extrema detector across all
            `active_dims` is written. Has shape (nvars, nx, ny, nz).
        active_dims: Tuple of dimensions along which to compute the smooth extrema
            detector. Each dimension is one of "x", "y", or "z".
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        uniformity_tol: Tolerance used to detect uniform regions.
        eps: Small tolerance used to avoid dividing by zero.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        return compute_alpha_kernel_helper(
            u, alpha, check_uniformity, uniformity_tol, eps
        )

    alpha[...] = 1.0
    valid_slices: List[Tuple[slice, ...]] = []
    for dim in active_dims:
        valid = update_alpha_1d(u, alpha, dim, check_uniformity, uniformity_tol, eps)
        valid_slices.append(valid)
    return merge_slices(*valid_slices)


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    compute_alpha_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_alpha_kernel(
            const double* __restrict__ u,
            double* __restrict__ alpha,
            const bool check_uniformity,
            const double uniformity_tol,
            const double eps,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u        (nvars, nx, ny, nz)
            // alpha    (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;
            bool usingx = nx > 1, usingy = ny > 1, usingz = nz > 1;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;
                int iv = (int)t;

                // skip boundary cells (ghost zones remain unmodified)
                if ((usingx && (ix < 3 || ix >= nx - 3))
                || (usingy && (iy < 3 || iy >= ny - 3))
                || (usingz && (iz < 3 || iz >= nz - 3))) {
                    continue;
                }

                double alpha_i = 1.0;

                for (int id = 0; id < 3; id++) {
                    // skip inactive dimensions
                    switch (id) {
                        case 0: if (!usingx) continue; break;
                        case 1: if (!usingy) continue; break;
                        case 2: if (!usingz) continue; break;
                    }

                    double alpha_dim_i = 1.0;

                    for (int off = -1; off <= 1; off++) {
                        int jv = iv, jx = ix, jy = iy, jz = iz;
                        long long j0, j1, j2, j3, j4;
                        switch (id) {
                            case 0:
                                jx += off;
                                j0 = (((long long)jv * nx + (jx - 2)) * ny + jy) * nz
                                    + jz;
                                j1 = (((long long)jv * nx + (jx - 1)) * ny + jy) * nz
                                    + jz;
                                j2 = (((long long)jv * nx + jx) * ny + jy) * nz + jz;
                                j3 = (((long long)jv * nx + (jx + 1)) * ny + jy) * nz
                                    + jz;
                                j4 = (((long long)jv * nx + (jx + 2)) * ny + jy) * nz
                                    + jz;
                                break;
                            case 1:
                                jy += off;
                                j0 = (((long long)jv * nx + ix) * ny + (jy - 2)) * nz
                                    + jz;
                                j1 = (((long long)jv * nx + ix) * ny + (jy - 1)) * nz
                                    + jz;
                                j2 = (((long long)jv * nx + ix) * ny + jy) * nz + jz;
                                j3 = (((long long)jv * nx + ix) * ny + (jy + 1)) * nz
                                    + jz;
                                j4 = (((long long)jv * nx + ix) * ny + (jy + 2)) * nz
                                    + jz;
                                break;
                            case 2:
                                jz += off;
                                j0 = (((long long)jv * nx + ix) * ny + iy) * nz
                                    + (jz - 2);
                                j1 = (((long long)jv * nx + ix) * ny + iy) * nz
                                    + (jz - 1);
                                j2 = (((long long)jv * nx + ix) * ny + iy) * nz + jz;
                                j3 = (((long long)jv * nx + ix) * ny + iy) * nz
                                    + (jz + 1);
                                j4 = (((long long)jv * nx + ix) * ny + iy) * nz
                                    + (jz + 2);
                        }

                        // compute alpha for this dimension using a 5-point stencil
                        double dul = 0.5 * (u[j2] - u[j0]);
                        double duc = 0.5 * (u[j3] - u[j1]);
                        double dur = 0.5 * (u[j4] - u[j2]);

                        double dv = 0.25 * (dur - dul);
                        double dv_safe = (fabs(dv) > eps ? dv : (dv > 0 ? eps : -eps));

                        double dvl = dul - duc;
                        double alphal = -((dv < 0) ? fmax(dvl, 0.0) : fmin(dvl, 0.0)) / dv_safe;

                        double dvr = dur - duc;
                        double alphar = ((dv > 0) ? fmax(dvr, 0.0) : fmin(dvr, 0.0)) / dv_safe;

                        double alpha_dim_j = fmin(fmin(alphal, alphar), 1.0);

                        // relax alpha in uniform regions
                        if (check_uniformity) {
                            double m = fmin(fmin(u[j1], u[j2]), u[j3]);
                            double M = fmax(fmax(u[j1], u[j2]), u[j3]);
                            bool uniform = fabs(M - m) <= uniformity_tol * fabs(u[j2]);

                            if (uniform) {
                                alpha_dim_j = 1.0;
                            }
                        }

                        // update dimension-wise alpha using neighbor min
                        if (alpha_dim_j < alpha_dim_i) {
                            alpha_dim_i = alpha_dim_j;
                        }
                    }
                    // update overall alpha using dimension-wise min
                    if (alpha_dim_i < alpha_i) {
                        alpha_i = alpha_dim_i;
                    }
                }
                alpha[i] = alpha_i;
            }
        }
        """,
        name="compute_alpha_kernel",
    )

    def compute_alpha_kernel_helper(
        u: cp.ndarray,
        alpha: cp.ndarray,
        check_uniformity: bool,
        uniformity_tol: float,
        eps: float,
    ) -> Tuple[slice, ...]:
        if not u.flags.c_contiguous or not alpha.flags.c_contiguous:
            raise ValueError("u and alpha must be C-contiguous for the kernel.")
        if u.dtype != cp.float64 or alpha.dtype != cp.float64:
            raise ValueError("u and alpha must be of type float64 for the kernel.")
        if u.ndim != 4 or alpha.ndim != 4:
            raise ValueError(
                "u and alpha must be 4D arrays with shape (nvars, nx, ny, nz)."
            )
        if u.shape != alpha.shape:
            raise ValueError("u and alpha must have the same shape.")

        nvars, nx, ny, nz = u.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (
            nvars * nx * ny * nz + threads_per_block - 1
        ) // threads_per_block

        compute_alpha_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                u,
                alpha,
                check_uniformity,
                uniformity_tol,
                eps,
                nvars,
                nx,
                ny,
                nz,
            ),
        )

        return (
            slice(None),
            slice(3, -3) if nx > 1 else slice(None),
            slice(3, -3) if ny > 1 else slice(None),
            slice(3, -3) if nz > 1 else slice(None),
        )
