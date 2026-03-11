from dataclasses import dataclass
from typing import Literal, Optional, Tuple, cast

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.interpolation_schemes import InterpolationScheme, LimiterConfig
from superfv.slope_limiting import gather_neighbor_slices
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE
from superfv.tools.slicing import crop, insert_slice, merge_slices, replace_slice


@dataclass(frozen=True, slots=True)
class musclConfig(LimiterConfig):
    """
    Configuration for the MUSCL slope limiter.

    Attributes:
        shock_detection: Whether to enable shock detection.
        smooth_extrema_detection: Whether to enable smooth extrema detection.
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions if smooth
            extrema detection is enabled. Uniform regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        PAD_atol: Absolute tolerance for physical admissibility detection if
            physical_admissibility_detection is True.
        uniformity_tol: Tolerance for uniformity check when check_uniformity is True.
        limiter: Optional slope limiter specification of "minmod", "moncen", "PP2D".
    """

    limiter: Optional[Literal["minmod", "moncen", "PP2D"]] = None

    def key(self) -> str:
        return f"muscl-{self.limiter}"

    def to_dict(self) -> dict:
        out = LimiterConfig.to_dict(self)
        out.update(dict(limiter=self.limiter))
        return out


@dataclass(frozen=True, slots=True)
class musclInterpolationScheme(InterpolationScheme):
    """
    Configuration for MUSCL interpolation schemes.

    Attributes:
        p: The polynomial degree. Must be 1.
        flux_recipe: The flux recipe to use. For MUSCL schemes, this simplifies to:
            - 1: compute conservative slopes -> limit conservative slopes -> compute
                primitive nodes -> compute fluxes
            - 2: compute primitive cell averages -> compute primitive slopes -> limit
                primitive slopes -> compute fluxes
        limiter_config: The MUSCL limiter configuration to use.
    """

    p: int = 1
    flux_recipe: Literal[1, 2] = 2
    limiter_config: musclConfig = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=False,
        check_uniformity=False,
        physical_admissibility_detection=False,
    )

    def __post_init__(self):
        InterpolationScheme.__post_init__(self)
        if self.p != 1:
            raise ValueError("musclInterpolationScheme must have p=1")
        if not isinstance(self.limiter_config, musclConfig):
            raise ValueError("musclInterpolationScheme requires a musclConfig")

    def key(self) -> str:
        return self.limiter_config.key()

    def to_dict(self) -> dict:
        return dict(
            p=self.p,
            flux_recipe=self.flux_recipe,
            limiter_config=(
                None if self.limiter_config is None else self.limiter_config.to_dict()
            ),
        )


def compute_limited_slopes(
    u: np.ndarray,
    face_dim: Literal["x", "y", "z"],
    *,
    out: np.ndarray,
    buffer: np.ndarray,
    config: musclConfig,
    alpha: Optional[np.ndarray] = None,
) -> Tuple[slice, ...]:
    """
    Compute limited slopes for face-centered nodes from an array of finite
    volume averages.

    Args:
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        face_dim: Dimension along which the limited slopes are computed.
        out: Output array to store the limited slopes. Has shape
            (nvars, nx, ny, nz, nout). The result is stored in out[..., 0].
        buffer: Array to which temporary values are assigned with shape
            (nvars, nx, ny, nz, >=5).
        config: The MUSCL limiter configuration to use.
        alpha: Array to store the smooth extrema detector values if SED is True. Has
            shape (nvars, nx, ny, nz, 1).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    limiter = config.limiter
    SED = config.smooth_extrema_detection

    # define slices for left, center, and right nodes
    slc_l = crop(DIM_TO_AXIS[face_dim], (None, -2), ndim=4)
    slc_c = crop(DIM_TO_AXIS[face_dim], (1, -1), ndim=4)
    slc_r = crop(DIM_TO_AXIS[face_dim], (2, None), ndim=4)
    inner = insert_slice(slc_c, 4, 0)

    # allocate arrays
    check_buffer_slots(buffer, required=5)
    dlft = buffer[replace_slice(inner, 4, 0)]
    drgt = buffer[replace_slice(inner, 4, 1)]
    dcen = buffer[replace_slice(inner, 4, 2)]
    dsgn = buffer[replace_slice(inner, 4, 3)]
    dslp = buffer[replace_slice(inner, 4, 4)]

    # write slopes to `out` array
    match limiter:
        case "minmod":
            dlft[...] = u[slc_c] - u[slc_l]
            drgt[...] = u[slc_r] - u[slc_c]
            dcen[...] = 0.5 * (dlft + drgt)
            dsgn[...] = np.sign(dlft)
            dslp[...] = dsgn * np.minimum(np.abs(dlft), np.abs(drgt))
            out[inner] = np.where(dlft * drgt <= 0, 0, dslp)
            if SED:
                if alpha is None:
                    raise ValueError("alpha array must be provided when SED is True.")
                out[inner] = np.where(alpha[inner] < 1, out[inner], dcen)
        case "moncen":
            dlft[...] = u[slc_c] - u[slc_l]
            drgt[...] = u[slc_r] - u[slc_c]
            dcen[...] = 0.5 * (dlft + drgt)
            dsgn[...] = np.sign(dcen)
            dslp[...] = dsgn * np.minimum(
                np.minimum(np.abs(2 * dlft), 2 * np.abs(drgt)), np.abs(dcen)
            )
            out[inner] = np.where(dlft * drgt <= 0, 0, dslp)
            if SED:
                if alpha is None:
                    raise ValueError("alpha array must be provided when SED is True.")
                out[inner] = np.where(alpha[inner] < 1, out[inner], dcen)
        case "PP2D":
            raise ValueError("Oops, use the `compute_PP2D_slopes` function instead.")
        case None:
            out[inner] = 0.5 * (u[slc_r] - u[slc_l])
        case _:
            raise ValueError(f"Unknown limiter: {limiter}.")

    modified = cast(Tuple[slice, ...], replace_slice(inner, 4, slice(None, 1)))
    return modified


def compute_PP2D_slopes(
    u: np.ndarray,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    Sx: np.ndarray,
    Sy: np.ndarray,
    buffer: np.ndarray,
    eps: float = 1e-20,
    config: musclConfig,
    alpha: Optional[np.ndarray] = None,
) -> Tuple[slice, ...]:
    """
    Compute PP2D limited slopes and write them to the 'Sx' and 'Sy' arrays.

    Args:
        u: Array of finite volume averages to compute slopes from, has shape
            (nvars, nx, ny, nz).
        active_dims: Tuple indicating the active dimensions for interpolation. Can be
            some combination of 'x', 'y', and 'z'. For example, ('x', 'y') for the
            interpolation of nodes along the face of a cell on a two-dimensional grid.
        Sx: Output array to store the limited slopes in the first active dimension. Has
            shape (nvars, nx, ny, nz, 1).
        Sy: Output array to store the limited slopes in the second active dimension.
            Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which temporary values are assigned with shape
            (nvars, nx, ny, nz, >=4).
        eps: Small number to avoid division by zero.
        config: The MUSCL limiter configuration to use.
        alpha: Array to store the smooth extrema detector values if SED is True. Has
            shape (nvars, nx, ny, nz, 1).

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    SED = config.smooth_extrema_detection

    if len(active_dims) != 2:
        raise ValueError("PP2D slope limiter requires exactly two active dimensions.")

    axis1 = DIM_TO_AXIS[active_dims[0]]
    axis2 = DIM_TO_AXIS[active_dims[1]]

    # allocate arrays
    check_buffer_slots(buffer, required=4)
    V_min = buffer[..., 0]
    V_max = buffer[..., 1]
    V = buffer[..., 2]
    theta = buffer[..., 3:4]

    V_min_neighbors = np.empty((8,) + u.shape)
    V_max_neighbors = np.empty((8,) + u.shape)

    # assign slices
    slc1_c = crop(axis1, (1, -1), ndim=4)
    slc2_c = crop(axis2, (1, -1), ndim=4)

    slc_c = insert_slice(merge_slices(slc1_c, slc2_c), 4, 0)

    slc1_l = merge_slices(crop(axis1, (None, -2), ndim=4), slc2_c)
    slc1_r = merge_slices(crop(axis1, (2, None), ndim=4), slc2_c)
    slc2_l = merge_slices(slc1_c, crop(axis2, (None, -2), ndim=4))
    slc2_r = merge_slices(slc1_c, crop(axis2, (2, None), ndim=4))

    # compute second-order slopes
    Sx[slc_c] = 0.5 * (u[slc1_r] - u[slc1_l])
    Sy[slc_c] = 0.5 * (u[slc2_r] - u[slc2_l])

    # compute PPD2 limiter
    neighbor_slices = gather_neighbor_slices(active_dims, include_corners=True)
    c_slc = neighbor_slices[0]

    V_min_neighbors[insert_slice(c_slc, 0, slice(None))] = np.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_min[...] = np.minimum(np.min(V_min_neighbors, axis=0), -eps)

    V_max_neighbors[insert_slice(c_slc, 0, slice(None))] = np.array(
        [u[slc] - u[c_slc] for slc in neighbor_slices[1:]]
    )
    V_max[...] = np.maximum(np.max(V_max_neighbors, axis=0), eps)

    V[...] = (
        2
        * np.minimum(np.abs(V_min), np.abs(V_max))
        / (np.abs(Sx[..., 0]) + np.abs(Sy[..., 0]))
    )
    theta[..., 0] = np.minimum(V, 1)

    # apply SED if requested
    if SED:
        if alpha is None:
            raise ValueError("alpha array must be provided when SED is True.")
        theta[...] = np.where(alpha < 1, theta, 1.0)

    modified = cast(Tuple[slice, ...], replace_slice(slc_c, 4, slice(None, 1)))

    Sx[modified] = theta[modified] * Sx[modified]
    Sy[modified] = theta[modified] * Sy[modified]

    return modified


# - - - - - DEFINE CUPY KERNELS FOR GPU COMPUTATION - - - - -

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    MUSCL_slopes_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void MUSCL_slopes_kernel(
            const double* __restrict__ u,
            const double* __restrict__ alpha,
            double* __restrict__ slopes,
            const int dim,
            const int slope_type,
            const bool SED,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u        (nvars, nx, ny, nz)
            // alpha    (nvars, nx, ny, nz)
            // slopes   (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ntotal = (long long)nvars * nx * ny * nz;

            for (long long i = tid; i < ntotal; i += stride) {
                long long t = i;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;
                int iv = (int)t;

                // get neighbor indices
                long long jl, jr;

                switch (dim) {
                    case 1: // x-slope
                        if (ix < 1 || ix >= nx - 1) {continue;}
                        jl = (((long long)iv * nx + (ix - 1)) * ny + iy) * nz + iz;
                        jr = (((long long)iv * nx + (ix + 1)) * ny + iy) * nz + iz;
                        break;
                    case 2: // y-slope
                        if (iy < 1 || iy >= ny - 1) {continue;}
                        jl = (((long long)iv * nx + ix) * ny + (iy - 1)) * nz + iz;
                        jr = (((long long)iv * nx + ix) * ny + (iy + 1)) * nz + iz;
                        break;
                    case 3: // z-slope
                        if (iz < 1 || iz >= nz - 1) {continue;}
                        jl = (((long long)iv * nx + ix) * ny + iy) * nz + (iz - 1);
                        jr = (((long long)iv * nx + ix) * ny + iy) * nz + (iz + 1);
                        break;
                    default:
                        slopes[i] = 0.0 / 0.0;
                        continue;
                }

                // gather neighbor values
                double ul = u[jl];
                double uc = u[i];
                double ur = u[jr];

                // compute slopes
                double dl = uc - ul;
                double dr = ur - uc;
                double dc = 0.5 * (dl + dr);
                double slope;

                // SED relaxation
                if (SED && alpha[i] >= 1.0) {
                    slopes[i] = dc;
                    continue;
                }

                // apply limiter
                switch (slope_type) {
                    case 0: // no limiter
                        slope = dc;
                        break;
                    case 1: // minmod
                        if (dl * dr <= 0) {
                            slope = 0.0;
                        } else {
                            double sgn = (dl > 0) ? 1.0 : -1.0;
                            slope = sgn * fmin(fabs(dl), fabs(dr));
                        }
                        break;
                    case 2: // moncen
                        if (dl * dr <= 0) {
                            slope = 0.0;
                        } else {
                            double sgn = (dc > 0) ? 1.0 : -1.0;
                            slope = sgn * fmin(
                                fmin(fabs(2.0 * dl), fabs(2.0 * dr)),
                                fabs(dc)
                            );
                        }
                        break;
                    default: // undefined limiter, sabotage with NaN
                        slope = 0.0 / 0.0;
                        break;
                }
                slopes[i] = slope;
            }
        }
        """,
        name="MUSCL_slopes_kernel",
    )

    def MUSCL_slopes_kernel_helper(
        u: cp.ndarray,
        alpha: cp.ndarray,
        slopes: cp.ndarray,
        dim: Literal["x", "y", "z"],
        slope_type: Literal[None, "minmod", "moncen", "PP2D"],
        SED: bool,
    ) -> Tuple[slice, ...]:
        if slope_type == "PP2D":
            raise ValueError(
                "PP2D slopes should be computed with the compute_PP2D_slopes "
                "function, not the MUSCL_slopes_kernel."
            )
        if u.ndim != 4 or slopes.ndim != 4:
            raise ValueError(
                "u and slopes must be 4D arrays with shape (nvars, nx, ny, nz)"
            )
        if u.shape != slopes.shape:
            raise ValueError("u and slopes must have the same shape")
        if not u.flags.c_contiguous or not slopes.flags.c_contiguous:
            raise ValueError("u and slopes must be C-contiguous arrays")
        if u.dtype != cp.float64 or slopes.dtype != cp.float64:
            raise ValueError("u and slopes must be of dtype float64")
        if SED:
            if alpha is None:
                raise ValueError("alpha array must be provided when SED is True")
            if alpha.ndim != 4 or alpha.shape != u.shape:
                raise ValueError(
                    "alpha must be a 4D array with the same shape as u when SED is "
                    "True"
                )
            if not alpha.flags.c_contiguous:
                raise ValueError("alpha must be a C-contiguous array when SED is True")
            if alpha.dtype != cp.float64:
                raise ValueError("alpha must be of dtype float64 when SED is True")

        nvars, nx, ny, nz = u.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (
            nvars * nx * ny * nz + threads_per_block - 1
        ) // threads_per_block

        MUSCL_slopes_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                u,
                alpha,
                slopes,
                DIM_TO_AXIS[dim],
                {"minmod": 1, "moncen": 2, None: 0}[slope_type],
                SED,
                nvars,
                nx,
                ny,
                nz,
            ),
        )

        return (
            slice(None),
            slice(1, -1) if dim == "x" else slice(None),
            slice(1, -1) if dim == "y" else slice(None),
            slice(1, -1) if dim == "z" else slice(None),
        )

    PP2D_slopes_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void PP2D_slopes_kernel(
            const double* __restrict__ u,
            const double* __restrict__ alpha,
            double* __restrict__ xslopes,
            double* __restrict__ yslopes,
            const int xdim,
            const int ydim,
            const double eps,
            const bool SED,
            const int nvars,
            const int nx,
            const int ny,
            const int nz
        ){
            // u        (nvars, nx, ny, nz)
            // alpha    (nvars, nx, ny, nz)
            // xslopes  (nvars, nx, ny, nz)
            // yslopes  (nvars, nx, ny, nz)

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
                switch (xdim) {
                    case 1: if (ix < 1 || ix >= nx - 1) continue; break;
                    case 2: if (iy < 1 || iy >= ny - 1) continue; break;
                    case 3: if (iz < 1 || iz >= nz - 1) continue; break;
                    default: break;
                }
                switch (ydim) {
                    case 1: if (ix < 1 || ix >= nx - 1) continue; break;
                    case 2: if (iy < 1 || iy >= ny - 1) continue; break;
                    case 3: if (iz < 1 || iz >= nz - 1) continue; break;
                    default: break;
                }

                // get neighbor indices
                // j0, j1, j2
                // j3, j4, j5
                // j6, j7, j8

                int j0v = iv, j0x = ix, j0y = iy, j0z = iz;
                int j1v = iv, j1x = ix, j1y = iy, j1z = iz;
                int j2v = iv, j2x = ix, j2y = iy, j2z = iz;
                int j3v = iv, j3x = ix, j3y = iy, j3z = iz;
                int j5v = iv, j5x = ix, j5y = iy, j5z = iz;
                int j6v = iv, j6x = ix, j6y = iy, j6z = iz;
                int j7v = iv, j7x = ix, j7y = iy, j7z = iz;
                int j8v = iv, j8x = ix, j8y = iy, j8z = iz;

                switch (xdim) {
                    case 1:
                        j0x -= 1; j3x -= 1; j6x -= 1;
                        j2x += 1; j5x += 1; j8x += 1;
                        break;
                    case 2:
                        j0y -= 1; j3y -= 1; j6y -= 1;
                        j2y += 1; j5y += 1; j8y += 1;
                        break;
                    case 3:
                        j0z -= 1; j3z -= 1; j6z -= 1;
                        j2z += 1; j5z += 1; j8z += 1;
                        break;
                }

                switch (ydim) {
                    case 1:
                        j0x -= 1; j1x -= 1; j2x -= 1;
                        j6x += 1; j7x += 1; j8x += 1;
                        break;
                    case 2:
                        j0y -= 1; j1y -= 1; j2y -= 1;
                        j6y += 1; j7y += 1; j8y += 1;
                        break;
                    case 3:
                        j0z -= 1; j1z -= 1; j2z -= 1;
                        j6z += 1; j7z += 1; j8z += 1;
                        break;
                }

                long long j0 = (((long long)j0v * nx + j0x) * ny + j0y) * nz + j0z;
                long long j1 = (((long long)j1v * nx + j1x) * ny + j1y) * nz + j1z;
                long long j2 = (((long long)j2v * nx + j2x) * ny + j2y) * nz + j2z;
                long long j3 = (((long long)j3v * nx + j3x) * ny + j3y) * nz + j3z;
                long long j5 = (((long long)j5v * nx + j5x) * ny + j5y) * nz + j5z;
                long long j6 = (((long long)j6v * nx + j6x) * ny + j6y) * nz + j6z;
                long long j7 = (((long long)j7v * nx + j7x) * ny + j7y) * nz + j7z;
                long long j8 = (((long long)j8v * nx + j8x) * ny + j8y) * nz + j8z;

                // compute slopes
                double sx = 0.5 * (u[j5] - u[j3]);
                double sy = 0.5 * (u[j7] - u[j1]);

                double uc = u[i];
                double du0 = u[j0] - uc;
                double du1 = u[j1] - uc;
                double du2 = u[j2] - uc;
                double du3 = u[j3] - uc;
                double du5 = u[j5] - uc;
                double du6 = u[j6] - uc;
                double du7 = u[j7] - uc;
                double du8 = u[j8] - uc;

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
                double theta = fmin(v, 1.0);

                // SED relaxation
                if (SED && alpha[i] >= 1.0) {
                    theta = 1.0;
                }

                // limit slopes
                sx *= theta;
                sy *= theta;

                xslopes[i] = sx;
                yslopes[i] = sy;
            }
        }
        """,
        name="PP2D_slopes_kernel",
    )

    def PP2D_slopes_kernel_helper(
        u: cp.ndarray,
        alpha: cp.ndarray,
        xslopes: cp.ndarray,
        yslopes: cp.ndarray,
        xdim: Literal["x", "y", "z"],
        ydim: Literal["x", "y", "z"],
        eps: float,
        SED: bool,
    ) -> Tuple[slice, ...]:
        if u.ndim != 4 or xslopes.ndim != 4 or yslopes.ndim != 4:
            raise ValueError(
                "u, xslopes, and yslopes must be 4D arrays with shape "
                "(nvars, nx, ny, nz)"
            )
        if u.shape != xslopes.shape or u.shape != yslopes.shape:
            raise ValueError("u, xslopes, and yslopes must have the same shape")
        if (
            not u.flags.c_contiguous
            or not xslopes.flags.c_contiguous
            or not yslopes.flags.c_contiguous
        ):
            raise ValueError("u, xslopes, and yslopes must be C-contiguous arrays")
        if (
            u.dtype != cp.float64
            or xslopes.dtype != cp.float64
            or yslopes.dtype != cp.float64
        ):
            raise ValueError("u, xslopes, and yslopes must be of dtype float64")
        if SED:
            if alpha is None:
                raise ValueError("alpha array must be provided when SED is True")
            if alpha.ndim != 4 or alpha.shape != u.shape:
                raise ValueError(
                    "alpha must be a 4D array with the same shape as u when SED is "
                    "True"
                )
            if not alpha.flags.c_contiguous:
                raise ValueError("alpha must be a C-contiguous array when SED is True")
            if alpha.dtype != cp.float64:
                raise ValueError("alpha must be of dtype float64 when SED is True")

        nvars, nx, ny, nz = u.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (
            nvars * nx * ny * nz + threads_per_block - 1
        ) // threads_per_block

        PP2D_slopes_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                u,
                alpha,
                xslopes,
                yslopes,
                DIM_TO_AXIS[xdim],
                DIM_TO_AXIS[ydim],
                eps,
                SED,
                nvars,
                nx,
                ny,
                nz,
            ),
        )

        return (
            slice(None),
            slice(1, -1) if "x" in (xdim, ydim) else slice(None),
            slice(1, -1) if "y" in (xdim, ydim) else slice(None),
            slice(1, -1) if "z" in (xdim, ydim) else slice(None),
        )
