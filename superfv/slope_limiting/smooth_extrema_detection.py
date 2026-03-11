from typing import Literal, Tuple, cast

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE
from superfv.tools.slicing import crop, merge_slices, replace_slice
from superfv.tools.stability import avoid0


def central_difference(u: np.ndarray, axis: int, *, out: np.ndarray):
    """
    Compute 1D central difference, ignoring mesh size.

    Args:
        u: Array of data to differentiate.
        axis: Axis along which to compute the central difference.
        out: Array to which the output is assigned.
    """
    out[crop(axis, (1, -1))] = 0.5 * (
        u[crop(axis, (2, None))] - u[crop(axis, (None, -2))]
    )


def smooth_extrema_detector_1d(
    u: np.ndarray,
    dim: Literal["x", "y", "z"],
    check_uniformity: bool,
    *,
    out: np.ndarray,
    buffer: np.ndarray,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 1D smooth extrema detector alpha.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        dim: Dimension along which to compute the smooth extrema detector: "x", "y",
            "z".
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=10).
        eps: Small tolerance used to avoid dividing by zero.
        uniformity_tol: Tolerance used to detect uniform regions.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    # retrieve axis
    axis = DIM_TO_AXIS[dim]

    # allocate arrays
    check_buffer_slots(buffer, required=10)
    du = buffer[..., 0]
    dv = buffer[..., 1]
    vl = buffer[..., 2]
    vr = buffer[..., 3]
    alpha_l = buffer[..., 4]
    alpha_r = buffer[..., 5]
    alpha = buffer[..., 6:7]  # (..., 1)
    dmp_m = buffer[..., 7]
    dmp_M = buffer[..., 8]
    uniform = buffer[..., 9]

    # compute derivatives
    central_difference(u, axis, out=du)
    central_difference(du, axis, out=dv)
    dv[...] = avoid0(0.5 * dv, eps)

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (None, -2))] - du[crop(axis, (1, -1))]
    alpha_l[...] = -np.where(dv < 0, np.maximum(vl, 0), np.minimum(vl, 0)) / dv

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, None))] - du[crop(axis, (1, -1))]
    alpha_r[...] = np.where(dv > 0, np.maximum(vr, 0), np.minimum(vr, 0)) / dv

    # combine left and right detectors
    alpha[..., 0] = np.minimum(np.minimum(alpha_l, alpha_r), 1.0)

    # relax alpha in uniform regions
    if check_uniformity:
        lft_slc = crop(axis, (2, None))
        ctr_slc = crop(axis, (1, -1))
        rgt_slc = crop(axis, (None, -2))

        dmp_m[ctr_slc] = np.minimum(np.minimum(u[lft_slc], u[ctr_slc]), u[rgt_slc])
        dmp_M[ctr_slc] = np.maximum(np.maximum(u[lft_slc], u[ctr_slc]), u[rgt_slc])

        uniform[...] = np.abs(dmp_M - dmp_m) <= uniformity_tol * np.abs(u)

        alpha[..., 0] = np.where(uniform == 1, 1.0, alpha[..., 0])

    # take min of neighbors and return
    lft_slc = crop(axis, (2, -4), ndim=5)
    cen_slc = crop(axis, (3, -3), ndim=5)
    rgt_slc = crop(axis, (4, -2), ndim=5)

    out[cen_slc] = np.minimum(alpha[lft_slc], alpha[cen_slc])
    out[cen_slc] = np.minimum(alpha[rgt_slc], out[cen_slc])

    modified = cast(Tuple[slice, ...], replace_slice(cen_slc, 4, slice(None, 1)))
    return modified


def smooth_extrema_detector_2d(
    u: np.ndarray,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    check_uniformity: bool,
    *,
    out: np.ndarray,
    buffer: np.ndarray,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 2D smooth extrema detector alpha.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple of two dimensions along which to compute the smooth extrema
            detector: ("x", "y"), ("x", "z"), or ("y", "z").\
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=12).
        eps: Small tolerance used to avoid dividing by zero.
        uniformity_tol: Tolerance used to detect uniform regions.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    d1, d2 = active_dims

    check_buffer_slots(buffer, required=12)
    alph1 = buffer[..., :1]
    alph2 = buffer[..., 1:2]
    abuff = buffer[..., 2:]

    modified1 = smooth_extrema_detector_1d(
        u,
        d1,
        check_uniformity,
        out=alph1,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified2 = smooth_extrema_detector_1d(
        u,
        d2,
        check_uniformity,
        out=alph2,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )

    modified = merge_slices(modified1, modified2)
    out[modified] = np.minimum(alph1[modified], alph2[modified])

    return modified


def smooth_extrema_detector_3d(
    u: np.ndarray,
    check_uniformity: bool,
    *,
    out: np.ndarray,
    buffer: np.ndarray,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 3D smooth extrema detector alpha.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has shape
            (nvars, nx, ny, nz, >=13).
        eps: Small tolerance used to avoid dividing by zero.
        uniformity_tol: Tolerance used to detect uniform regions.
    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    d1: Literal["x"] = "x"
    d2: Literal["y"] = "y"
    d3: Literal["z"] = "z"

    check_buffer_slots(buffer, required=13)
    alph1 = buffer[..., :1]
    alph2 = buffer[..., 1:2]
    alph3 = buffer[..., 2:3]
    abuff = buffer[..., 3:]

    modified1 = smooth_extrema_detector_1d(
        u,
        d1,
        check_uniformity,
        out=alph1,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified2 = smooth_extrema_detector_1d(
        u,
        d2,
        check_uniformity,
        out=alph2,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified3 = smooth_extrema_detector_1d(
        u,
        d3,
        check_uniformity,
        out=alph3,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )

    modified = merge_slices(modified1, modified2, modified3)
    out[modified] = np.minimum(alph1[modified], alph2[modified])
    out[modified] = np.minimum(alph3[modified], out[modified])

    return modified


def smooth_extrema_detector(
    u: np.ndarray,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    check_uniformity: bool,
    *,
    out: np.ndarray,
    buffer: np.ndarray,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the smooth extrema detector alpha along specified dimensions.

    Args:
        u: Array of data used to compute the smooth extrema detector. Has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple of dimensions along which to compute the smooth extrema
            detector. Has length 1, 2, or 3 with possible values "x", "y", "z".
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions. Uniform
            regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        out: Array to which alpha is assigned. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on the number (length) of active dimensions:
            - 1D: (nvars, nx, ny, nz, >=7)
            - 2D: (nvars, nx, ny, nz, >=9)
            - 3D: (nvars, nx, ny, nz, >=10)
        eps: Small tolerance used to avoid dividing by zero.
        uniformity_tol: Tolerance used to detect uniform regions.

    Returns:
        Slice objects indicating the modified regions in the output array.

    """
    if len(active_dims) == 1:
        return smooth_extrema_detector_1d(
            u,
            active_dims[0],
            check_uniformity,
            out=out,
            buffer=buffer,
            eps=eps,
            uniformity_tol=uniformity_tol,
        )
    elif len(active_dims) == 2:
        return smooth_extrema_detector_2d(
            u,
            active_dims,
            check_uniformity,
            out=out,
            buffer=buffer,
            eps=eps,
            uniformity_tol=uniformity_tol,
        )
    elif len(active_dims) == 3:
        return smooth_extrema_detector_3d(
            u,
            check_uniformity,
            out=out,
            buffer=buffer,
            eps=eps,
            uniformity_tol=uniformity_tol,
        )
    raise ValueError("active_dims must have length 1, 2, or 3.")


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    compute_alpha_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_alpha_kernel(
            const double* __restrict__ u,
            double* __restrict__ alpha,
            const double eps,
            const bool check_uniformity,
            const double uniformity_tol,
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
        eps: float,
        check_uniformity: bool,
        uniformity_tol: float,
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
                eps,
                check_uniformity,
                uniformity_tol,
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
            slice(None, 1),
        )
