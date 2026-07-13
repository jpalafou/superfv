from enum import Enum
from functools import lru_cache
from typing import Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.slope_limiting import gather_neighbor_slices
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices


class MUSCL_SlopeLimiter(Enum):
    MINMOD = 0
    MONCEN = 1
    PP2D = 2
    NONE = 3


def compute_1d_limited_slopes(
    u: np.ndarray,
    alpha: np.ndarray,
    out: np.ndarray,
    face_dim: Literal["x", "y", "z"],
    limiter: MUSCL_SlopeLimiter,
    use_SED: bool = False,
) -> Tuple[slice, ...]:
    """
    Compute limited slopes from `u` using a 1D MUSCL limiter and write them to the `out` array
    with optional relaxation from the smooth extrema detector `alpha`. Renders a single ghost
    cell layer along the `face_dim` dimension of the output array invalid.

    Args:
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        alpha: Array to store the smooth extrema detector values if use_SED is True in
            `config`. Has shape (nvars, nx, ny, nz).
        out: Output array to which the limited slopes are written. Has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the limited slopes are computed.
        limiter: The slope limiter to use.
        use_SED: Whether to apply smooth extrema detection (use_SED). If True, the limited
            slopes are relaxed to the unlimited centered difference slopes in smooth
            extrema regions where alpha >= 1.

    Returns:
        Slice objects indicating the valid region of the output array.
    """
    # define slices for left, center, and right nodes
    left = crop(DIM_TO_AXIS[face_dim], (None, -2), ndim=4)
    inner = crop(DIM_TO_AXIS[face_dim], (1, -1), ndim=4)
    right = crop(DIM_TO_AXIS[face_dim], (2, None), ndim=4)

    # write slopes to `out` array
    match limiter:
        case MUSCL_SlopeLimiter.MINMOD:
            dlft = u[inner] - u[left]
            drgt = u[right] - u[inner]
            dcen = 0.5 * (dlft + drgt)
            dsgn = np.sign(dlft)
            dslp = dsgn * np.minimum(np.abs(dlft), np.abs(drgt))
            out[inner] = np.where(dlft * drgt <= 0, 0, dslp)
            if use_SED:
                if alpha is None:
                    raise ValueError("alpha array must be provided when use_SED is True.")
                out[inner] = np.where(alpha[inner] < 1, out[inner], dcen)
        case MUSCL_SlopeLimiter.MONCEN:
            dlft = u[inner] - u[left]
            drgt = u[right] - u[inner]
            dcen = 0.5 * (dlft + drgt)
            dsgn = np.sign(dcen)
            dslp = dsgn * np.minimum(np.minimum(np.abs(2 * dlft), 2 * np.abs(drgt)), np.abs(dcen))
            out[inner] = np.where(dlft * drgt <= 0, 0, dslp)
            if use_SED:
                if alpha is None:
                    raise ValueError("alpha array must be provided when use_SED is True.")
                out[inner] = np.where(alpha[inner] < 1, out[inner], dcen)
        case MUSCL_SlopeLimiter.PP2D:
            raise ValueError("Oops, use the `compute_PP2D_slopes` function instead.")
        case MUSCL_SlopeLimiter.NONE:
            out[inner] = 0.5 * (u[right] - u[left])
        case _:
            raise ValueError(f"Unknown limiter: {limiter}.")

    return inner


def compute_PP2D_slopes(
    u: np.ndarray,
    alpha: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    eps: float = 1e-20,
    use_SED: bool = False,
) -> Tuple[slice, ...]:
    """
    Compute PP2D limited slopes from `u` and write them to the `Sx` and `Sy`
    arrays with optional relaxation from the smooth extrema detector `alpha`.
    Renders a single ghost cell layer along each active dimension of the
    output arrays invalid.

    Args:
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        alpha: Array to store the smooth extrema detector values if use_SED is True in
            `config`. Has shape (nvars, nx, ny, nz).
        Sx: Output array to which the slopes in the direction of the first active dims
            are written. Has shape (nvars, nx, ny, nz).
        Sy: Output array to which the slopes in the direction of the second active dims
            are written. Has shape (nvars, nx, ny, nz).
        active_dims: Tuple containing two active dims ("x", "y", or "z").
        eps: Tolerance value.
        use_SED: Whether to apply smooth extrema detection (use_SED). If True, the limited
            slopes are relaxed to the unlimited centered difference slopes in smooth
            extrema regions where alpha >= 1.

    Returns:
        Slice objects indicating the valid region of the output arrays `Sx` and `Sy`.
    """
    axis1 = DIM_TO_AXIS[active_dims[0]]
    axis2 = DIM_TO_AXIS[active_dims[1]]

    # allocate arrays
    V_min_neighbors = np.empty((8,) + u.shape)
    V_max_neighbors = np.empty((8,) + u.shape)

    # assign slices
    inner1 = crop(axis1, (1, -1), ndim=4)
    inner2 = crop(axis2, (1, -1), ndim=4)
    inner = merge_slices(inner1, inner2)

    left1 = merge_slices(crop(axis1, (None, -2), ndim=4), inner2)
    right1 = merge_slices(crop(axis1, (2, None), ndim=4), inner2)
    left2 = merge_slices(inner1, crop(axis2, (None, -2), ndim=4))
    right2 = merge_slices(inner1, crop(axis2, (2, None), ndim=4))

    # compute second-order slopes
    Sx[inner] = 0.5 * (u[right1] - u[left1])
    Sy[inner] = 0.5 * (u[right2] - u[left2])

    # compute PPD2 limiter
    neighbor_slices = gather_neighbor_slices(active_dims, include_corners=True)
    c_slc = neighbor_slices[0]

    V_min_neighbors[insert_slice(c_slc, 0, slice(None))] = np.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_min = np.minimum(np.min(V_min_neighbors, axis=0), -eps)

    V_max_neighbors[insert_slice(c_slc, 0, slice(None))] = np.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_max = np.maximum(np.max(V_max_neighbors, axis=0), eps)

    V = 2 * np.minimum(np.abs(V_min), np.abs(V_max)) / (np.abs(Sx) + np.abs(Sy))
    theta = np.minimum(V, 1)

    # apply use_SED if requested
    if use_SED:
        if alpha is None:
            raise ValueError("alpha array must be provided when use_SED is True.")
        theta[...] = np.where(alpha < 1, theta, 1.0)

    Sx[inner] = theta[inner] * Sx[inner]
    Sy[inner] = theta[inner] * Sy[inner]

    return inner


def compute_MUSCL_slopes(
    u: ArrayLike,
    alpha: ArrayLike,
    slopes: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    limiter: MUSCL_SlopeLimiter,
    use_SED: bool = False,
    eps: float = 1e-20,
):
    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        MUSCL_kernel_helper(
            u,
            alpha,
            slopes,
            output="slopes",
            limiter=limiter,
            dim=dim,
            active_dims=active_dims,
            eps=eps,
            SED=use_SED,
        )
        return
    if limiter == MUSCL_SlopeLimiter.PP2D:
        if len(active_dims) != 2:
            raise ValueError("PP2D slope limiter requires exactly two active dimensions.")
        Sx = np.empty_like(u)
        Sy = np.empty_like(u)
        compute_PP2D_slopes(u, alpha, Sx, Sy, active_dims, eps=eps, use_SED=use_SED)
        if dim == active_dims[0]:
            slopes[...] = Sx
        elif dim == active_dims[1]:
            slopes[...] = Sy
        else:
            raise ValueError(f"Invalid dim {dim} for active_dims {active_dims}.")
    else:
        compute_1d_limited_slopes(u, alpha, slopes, dim, limiter, use_SED=use_SED)


def reconstruct_MUSCL_faces(
    u: ArrayLike,
    alpha: ArrayLike,
    faces: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    limiter: MUSCL_SlopeLimiter,
    use_SED: bool = False,
    eps: float = 1e-20,
):
    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        MUSCL_kernel_helper(
            u,
            alpha,
            faces,
            output="faces",
            limiter=limiter,
            dim=dim,
            active_dims=active_dims,
            eps=eps,
            SED=use_SED,
        )
        return
    slopes = np.empty_like(u)
    compute_MUSCL_slopes(u, alpha, slopes, dim, active_dims, limiter, use_SED=use_SED, eps=eps)
    faces[..., 0] = u - 0.5 * slopes  # left face
    faces[..., 1] = u + 0.5 * slopes  # right face


# - - - - - DEFINE CUPY KERNELS FOR GPU COMPUTATION - - - - -

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    unlimited_device_function = """
        __device__ __forceinline__ double limited_slope(double ul, double uc, double ur) {
            return 0.5 * (ur - ul);
        }
        """

    minmod_device_function = """
        __device__ __forceinline__ double limited_slope(double ul, double uc, double ur) {
            const double dl = uc - ul;
            const double dr = ur - uc;

            if (dl * dr <= 0.0) {
                return 0.0;
            }

            const double sgn = dl > 0.0 ? 1.0 : -1.0;
            return sgn * fmin(fabs(dl), fabs(dr));
        }
        """

    moncen_device_function = """
        __device__ __forceinline__ double limited_slope(double ul, double uc, double ur) {
            const double dl = uc - ul;
            const double dr = ur - uc;

            if (dl * dr <= 0.0) {
                return 0.0;
            }

            const double dc = 0.5 * (dl + dr);
            const double sgn = dc > 0.0 ? 1.0 : -1.0;
            return sgn * fmin(fmin(fabs(2.0 * dl), fabs(2.0 * dr)), fabs(dc));
        }
        """

    PP2D_device_function = """
        __device__ __forceinline__ double PP2D_limiter(
            double u00,
            double u01,
            double u02,
            double u10,
            double u11,
            double u12,
            double u20,
            double u21,
            double u22,
            double eps
        ) {
            const double sx = 0.5 * (u12 - u10);
            const double sy = 0.5 * (u21 - u01);

            const double uc = u11;
            double du0 = u00 - uc;
            double du1 = u01 - uc;
            double du2 = u02 - uc;
            double du3 = u10 - uc;
            double du5 = u12 - uc;
            double du6 = u20 - uc;
            double du7 = u21 - uc;
            double du8 = u22 - uc;

            double vmin = -eps;
            vmin = fmin(vmin, du0);
            vmin = fmin(vmin, du1);
            vmin = fmin(vmin, du2);
            vmin = fmin(vmin, du3);
            vmin = fmin(vmin, du5);
            vmin = fmin(vmin, du6);
            vmin = fmin(vmin, du7);
            vmin = fmin(vmin, du8);

            double vmax = eps;
            vmax = fmax(vmax, du0);
            vmax = fmax(vmax, du1);
            vmax = fmax(vmax, du2);
            vmax = fmax(vmax, du3);
            vmax = fmax(vmax, du5);
            vmax = fmax(vmax, du6);
            vmax = fmax(vmax, du7);
            vmax = fmax(vmax, du8);

            double v = 2.0 * fmin(fabs(vmin), fabs(vmax)) / (fabs(sx) + fabs(sy));
            return fmin(v, 1.0);
        }
        """

    MUSCL_kernel_body = """
        extern "C" __global__
        void MUSCL_kernel(
            const double* __restrict__ u,
            const double* __restrict__ alpha,
            double* __restrict__ OUTPUT_ARG,
            const int dim,
            const bool xactive,
            const bool yactive,
            const bool zactive,
            const bool use_PP2D,
            const double eps,
            const bool SED,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u        (nvars, nx, ny, nz)
            // alpha    (nvars, nx, ny, nz)
            // faces    (nvars, nx, ny, nz)
            OUTPUT_COMMENT

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;
                int iv = (int)t;

                // skip boundary cells only along active slope dimensions
                if (use_PP2D) {
                    // PP2D requires two active dimensions, so we skip boundary cells along both of them
                    if (nx > 1 && (ix < 1 || ix >= nx - 1)) continue;
                    if (ny > 1 && (iy < 1 || iy >= ny - 1)) continue;
                    if (nz > 1 && (iz < 1 || iz >= nz - 1)) continue;
                } else {
                    // 1D slope limiters only require one active dimension, so we skip boundary cells along that dimension
                    switch (dim) {
                        case 1: if (ix < 1 || ix >= nx - 1) continue; break;
                        case 2: if (iy < 1 || iy >= ny - 1) continue; break;
                        case 3: if (iz < 1 || iz >= nz - 1) continue; break;
                        default: break;
                    }
                }

                // assign some useful variables
                long long il, ir;
                switch (dim) {
                    case 1: // x-slope
                        il = (((long long)iv * nx + (ix - 1)) * ny + iy) * nz + iz;
                        ir = (((long long)iv * nx + (ix + 1)) * ny + iy) * nz + iz;
                        break;
                    case 2: // y-slope
                        il = (((long long)iv * nx + ix) * ny + (iy - 1)) * nz + iz;
                        ir = (((long long)iv * nx + ix) * ny + (iy + 1)) * nz + iz;
                        break;
                    case 3: // z-slope
                        il = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 1);
                        ir = (((long long)iv * nx + ix) * ny + iy) * nz + (iz + 1);
                        break;
                }
                double myslope;
                double mysecondorderslope = 0.5 * (u[ir] - u[il]);

                // compute limited slope
                if (SED && alpha[i] >= 1.0) {
                    myslope = mysecondorderslope;
                } else if (use_PP2D) {
                    // assume 2D layout
                    long long stride1, stride2;
                    if (xactive && yactive && !zactive) {
                        stride1 = (long long)ny * nz;
                        stride2 = nz;
                    } else if (xactive && !yactive && zactive) {
                        stride1 = (long long)ny * nz;
                        stride2 = 1;
                    } else if (!xactive && yactive && zactive) {
                        stride1 = nz;
                        stride2 = 1;
                    } else {
                        return; // invalid configuration
                    }

                    long long i00 = i - stride1 - stride2;
                    long long i01 = i           - stride2;
                    long long i02 = i + stride1 - stride2;
                    long long i10 = i - stride1;
                    long long i11 = i;
                    long long i12 = i + stride1;
                    long long i20 = i - stride1 + stride2;
                    long long i21 = i           + stride2;
                    long long i22 = i + stride1 + stride2;

                    double theta = PP2D_limiter(
                        u[i00], u[i01], u[i02],
                        u[i10], u[i11], u[i12],
                        u[i20], u[i21], u[i22],
                        eps
                    );
                    myslope = theta * mysecondorderslope;
                } else {
                    myslope = limited_slope(u[il], u[i], u[ir]);
                }

                // assign either slopes or faces
                OUTPUT_ASSIGNMENT1
                OUTPUT_ASSIGNMENT2
            }
        }
        """

    @lru_cache(maxsize=8)
    def MUSCL_kernel_builder(limiter: MUSCL_SlopeLimiter, x: Literal["faces", "slopes"]):
        name = ""
        body = ""
        match limiter:
            case MUSCL_SlopeLimiter.NONE:
                body += unlimited_device_function
            case MUSCL_SlopeLimiter.MINMOD:
                body += minmod_device_function
            case MUSCL_SlopeLimiter.MONCEN:
                body += moncen_device_function
            case MUSCL_SlopeLimiter.PP2D:
                body += unlimited_device_function  # keeps limited_slope defined
            case _:
                raise ValueError(f"Unknown MUSCL slope limiter: {limiter}")
        body += PP2D_device_function

        spec = MUSCL_kernel_body
        if x == "slopes":
            name = f"MUSCL_slopes_kernel_{limiter.name}"
            spec = spec.replace("MUSCL_kernel", name)
            spec = spec.replace("OUTPUT_ARG", "slopes")
            spec = spec.replace("OUTPUT_COMMENT", "// slopes   (nvars, nx, ny, nz)")
            spec = spec.replace("OUTPUT_ASSIGNMENT1", "slopes[i] = myslope;")
            spec = spec.replace("OUTPUT_ASSIGNMENT2", "")
        elif x == "faces":
            name = f"MUSCL_faces_kernel_{limiter.name}"
            spec = spec.replace("MUSCL_kernel", name)
            spec = spec.replace("OUTPUT_ARG", "faces")
            spec = spec.replace("OUTPUT_COMMENT", "// faces    (nvars, nx, ny, nz, 2)")
            spec = spec.replace(
                "OUTPUT_ASSIGNMENT1", "faces[2 * i + 0] = u[i] - 0.5 * myslope; // left face"
            )
            spec = spec.replace(
                "OUTPUT_ASSIGNMENT2", "faces[2 * i + 1] = u[i] + 0.5 * myslope; // right face"
            )
        else:
            raise ValueError(f"Unknown output type: {x}")

        body += spec
        return cp.RawKernel(body, name=name)

    def MUSCL_kernel_helper(
        u: cp.ndarray,
        alpha: cp.ndarray,
        slopes_or_faces: cp.ndarray,
        output: Literal["slopes", "faces"],
        limiter: MUSCL_SlopeLimiter,
        dim: Literal["x", "y", "z"],
        active_dims: Tuple[Literal["x", "y", "z"], ...],
        eps: float,
        SED: bool,
    ):
        if u.ndim != 4:
            raise ValueError("u must be a 4D array with shape " "(nvars, nx, ny, nz)")
        if output not in ("slopes", "faces"):
            raise ValueError("output must be either 'slopes' or 'faces'")
        if limiter == MUSCL_SlopeLimiter.PP2D and len(active_dims) != 2:
            raise ValueError("PP2D slope limiter requires exactly two active dimensions.")
        if output == "slopes" and slopes_or_faces.shape != u.shape:
            raise ValueError("slopes must have the same shape as u")
        if output == "faces" and slopes_or_faces.shape != u.shape + (2,):
            raise ValueError("faces must be a 5D array with shape (nvars, nx, ny, nz, 2)")
        if not u.flags.c_contiguous or not slopes_or_faces.flags.c_contiguous:
            raise ValueError("u and slopes_or_faces must be C-contiguous arrays")
        if u.dtype != cp.float64 or slopes_or_faces.dtype != cp.float64:
            raise ValueError("u and slopes_or_faces must be of dtype float64")
        if SED:
            if alpha is None:
                raise ValueError("alpha array must be provided when SED is True")
            if alpha.shape != u.shape:
                raise ValueError(
                    "alpha must be a 4D array with the same shape as u when SED is " "True"
                )
            if not alpha.flags.c_contiguous:
                raise ValueError("alpha must be a C-contiguous array when SED is True")
            if alpha.dtype != cp.float64:
                raise ValueError("alpha must be of dtype float64 when SED is True")

        kernel = MUSCL_kernel_builder(limiter, output)

        nvars, nx, ny, nz = u.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nvars * nx * ny * nz + threads_per_block - 1) // threads_per_block

        kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                u,
                alpha,
                slopes_or_faces,
                DIM_TO_AXIS[dim],
                "x" in active_dims,
                "y" in active_dims,
                "z" in active_dims,
                limiter == MUSCL_SlopeLimiter.PP2D,
                eps,
                SED,
                nvars,
                nx,
                ny,
                nz,
            ),
        )
