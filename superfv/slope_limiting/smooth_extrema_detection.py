from types import ModuleType
from typing import Literal, Tuple, cast

from superfv.axes import DIM_TO_AXIS
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices, replace_slice
from superfv.tools.stability import avoid0


def central_difference(u: ArrayLike, axis: int, *, out: ArrayLike):
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
    xp: ModuleType,
    u: ArrayLike,
    dim: Literal["x", "y", "z"],
    check_uniformity: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 1D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
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
    if hasattr(xp, "cuda"):
        return smooth_extrema_detector_kernel_helper(
            xp,
            u,
            dim,
            check_uniformity,
            out=out,
            buffer=buffer,
            eps=eps,
            uniformity_tol=uniformity_tol,
        )

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
    dv[...] = avoid0(xp, 0.5 * dv, eps)

    # left detector
    vl[crop(axis, (1, -1))] = du[crop(axis, (None, -2))] - du[crop(axis, (1, -1))]
    alpha_l[...] = -xp.where(dv < 0, xp.maximum(vl, 0), xp.minimum(vl, 0)) / dv

    # right detector
    vr[crop(axis, (1, -1))] = du[crop(axis, (2, None))] - du[crop(axis, (1, -1))]
    alpha_r[...] = xp.where(dv > 0, xp.maximum(vr, 0), xp.minimum(vr, 0)) / dv

    # combine left and right detectors
    alpha[..., 0] = xp.minimum(xp.minimum(alpha_l, alpha_r), 1.0)

    # relax alpha in uniform regions
    if check_uniformity:
        lft_slc = crop(axis, (2, None))
        ctr_slc = crop(axis, (1, -1))
        rgt_slc = crop(axis, (None, -2))

        dmp_m[ctr_slc] = xp.minimum(xp.minimum(u[lft_slc], u[ctr_slc]), u[rgt_slc])
        dmp_M[ctr_slc] = xp.maximum(xp.maximum(u[lft_slc], u[ctr_slc]), u[rgt_slc])

        uniform[...] = xp.abs(dmp_M - dmp_m) <= uniformity_tol * xp.abs(u)

        alpha[..., 0] = xp.where(uniform == 1, 1.0, alpha[..., 0])

    # take min of neighbors and return
    lft_slc = crop(axis, (2, -4), ndim=5)
    cen_slc = crop(axis, (3, -3), ndim=5)
    rgt_slc = crop(axis, (4, -2), ndim=5)

    out[cen_slc] = xp.minimum(alpha[lft_slc], alpha[cen_slc])
    out[cen_slc] = xp.minimum(alpha[rgt_slc], out[cen_slc])

    modified = cast(Tuple[slice, ...], replace_slice(cen_slc, 4, slice(None, 1)))
    return modified


def smooth_extrema_detector_2d(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    check_uniformity: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 2D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
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
        xp,
        u,
        d1,
        check_uniformity,
        out=alph1,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified2 = smooth_extrema_detector_1d(
        xp,
        u,
        d2,
        check_uniformity,
        out=alph2,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )

    modified = merge_slices(modified1, modified2)
    out[modified] = xp.minimum(alph1[modified], alph2[modified])

    return modified


def smooth_extrema_detector_3d(
    xp: ModuleType,
    u: ArrayLike,
    check_uniformity: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the 3D smooth extrema detector alpha.

    Args:
        xp: `np` namespace.
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
        xp,
        u,
        d1,
        check_uniformity,
        out=alph1,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified2 = smooth_extrema_detector_1d(
        xp,
        u,
        d2,
        check_uniformity,
        out=alph2,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )
    modified3 = smooth_extrema_detector_1d(
        xp,
        u,
        d3,
        check_uniformity,
        out=alph3,
        buffer=abuff,
        eps=eps,
        uniformity_tol=uniformity_tol,
    )

    modified = merge_slices(modified1, modified2, modified3)
    out[modified] = xp.minimum(alph1[modified], alph2[modified])
    out[modified] = xp.minimum(alph3[modified], out[modified])

    return modified


def smooth_extrema_detector(
    xp: ModuleType,
    u: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    check_uniformity: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
) -> Tuple[slice, ...]:
    """
    Compute the smooth extrema detector alpha along specified dimensions.

    Args:
        xp: `np` namespace.
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
            xp,
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
            xp,
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
            xp,
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

    smooth_extrema_detector_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul2, float64 ul1, float64 uc, float64 ur1, float64 ur2"
            ", float64 eps"
        ),
        out_params="float64 alpha",
        operation=(
            """
            double dul = 0.5 * (uc - ul2);
            double duc = 0.5 * (ur1 - ul1);
            double dur = 0.5 * (ur2 - uc);

            double dv = 0.25 * (dur - dul);
            double dv_safe = (
                fabs(dv) > eps
                    ? dv
                    : (dv > 0 ? eps : -eps)
            );

            double dvl = dul - duc;
            double alphal = -((dv < 0) ? fmax(dvl, 0.0) : fmin(dvl, 0.0)) / dv_safe;

            double dvr = dur - duc;
            double alphar = ((dv > 0) ? fmax(dvr, 0.0) : fmin(dvr, 0.0)) / dv_safe;

            alpha = fmin(fmin(alphal, alphar), 1.0);
            """
        ),
        name="smooth_extrema_detector_kernel",
        no_return=True,
    )

    smooth_extrema_detector_neighbor_min_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 ul3, float64 ul2, float64 ul1, float64 uc, float64 ur1, "
            "float64 ur2, float64 ur3, float64 eps, "
            "bool check_uniformity, float64 uniformity_tol"
        ),
        out_params="float64 alpha",
        operation=(
            """
            double dul2 = 0.5 * (ul1 - ul3);
            double dul1 = 0.5 * (uc - ul2);
            double duc = 0.5 * (ur1 - ul1);
            double dur1 = 0.5 * (ur2 - uc);
            double dur2 = 0.5 * (ur3 - ur1);

            double dvl = 0.25 * (duc - dul2);
            double dvc = 0.25 * (dur1 - dul1);
            double dvr = 0.25 * (dur2 - duc);

            double dvl_safe = (
                fabs(dvl) > eps
                    ? dvl
                    : (dvl > 0 ? eps : -eps)
            );
            double dvc_safe = (
                fabs(dvc) > eps
                    ? dvc
                    : (dvc > 0 ? eps : -eps)
            );
            double dvr_safe = (
                fabs(dvr) > eps
                    ? dvr
                    : (dvr > 0 ? eps : -eps)
            );

            // left neighbor alphas
            double dvl_l = dul2 - dul1;
            double alphal_l = -(
                (dvl < 0 ? fmax(dvl_l, 0.0) : fmin(dvl_l, 0.0)) / dvl_safe
            );

            double dvl_c = dul1 - duc;
            double alphal_c = -(
                (dvc < 0 ? fmax(dvl_c, 0.0) : fmin(dvl_c, 0.0)) / dvc_safe
            );

            double dvl_r = duc - dur1;
            double alphal_r = -(
                (dvr < 0 ? fmax(dvl_r, 0.0) : fmin(dvl_r, 0.0)) / dvr_safe
            );

            // right neighbor alphas
            double dvr_l = duc - dul1;
            double alphar_l = (
                (dvl > 0 ? fmax(dvr_l, 0.0) : fmin(dvr_l, 0.0)) / dvl_safe
            );

            double dvr_c = dur1 - duc;
            double alphar_c = (
                (dvc > 0 ? fmax(dvr_c, 0.0) : fmin(dvr_c, 0.0)) / dvc_safe
            );

            double dvr_r = dur2 - dur1;
            double alphar_r = (
                (dvr > 0 ? fmax(dvr_r, 0.0) : fmin(dvr_r, 0.0)) / dvr_safe
            );

            double alphal = fmin(alphal_l, alphar_l);
            double alphac = fmin(alphal_c, alphar_c);
            double alphar = fmin(alphal_r, alphar_r);

            if (check_uniformity) {
                double dmp_m_l = fmin(fmin(ul2, ul1), uc);
                double dmp_M_l = fmax(fmax(ul2, ul1), uc);

                double dmp_m_c = fmin(fmin(ul1, uc), ur1);
                double dmp_M_c = fmax(fmax(ul1, uc), ur1);

                double dmp_m_r = fmin(fmin(uc, ur1), ur2);
                double dmp_M_r = fmax(fmax(uc, ur1), ur2);

                bool uniform_l = fabs(dmp_M_l - dmp_m_l) <= uniformity_tol * fabs(uc);
                bool uniform_c = fabs(dmp_M_c - dmp_m_c) <= uniformity_tol * fabs(uc);
                bool uniform_r = fabs(dmp_M_r - dmp_m_r) <= uniformity_tol * fabs(uc);

                if (uniform_l) {
                    alphal = 1.0;
                }
                if (uniform_c) {
                    alphac = 1.0;
                }
                if (uniform_r) {
                    alphar = 1.0;
                }
            }

            alpha = fmin(fmin(alphal, alphac), alphar);
            """
        ),
        name="smooth_extrema_detector_neighbor_min_kernel",
        no_return=True,
    )


def smooth_extrema_detector_kernel_helper(
    xp: ModuleType,
    u: ArrayLike,
    dim: Literal["x", "y", "z"],
    check_uniformity: bool,
    *,
    out: ArrayLike,
    buffer: ArrayLike,
    eps: float = 1e-16,
    uniformity_tol: float = 1e-3,
):
    """
    Compute the 1D smooth extrema detector alpha using a CuPy kernel.

    Args:
        xp: `np` namespace.
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
            (nvars, nx, ny, nz, >=7).
        eps: Small tolerance used to avoid dividing by zero.
        uniformity_tol: Tolerance used to detect uniform regions.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    axis = DIM_TO_AXIS[dim]

    ul3 = u[crop(axis, (None, -6), ndim=4)]
    ul2 = u[crop(axis, (1, -5), ndim=4)]
    ul1 = u[crop(axis, (2, -4), ndim=4)]
    uc = u[crop(axis, (3, -3), ndim=4)]
    ur1 = u[crop(axis, (4, -2), ndim=4)]
    ur2 = u[crop(axis, (5, -1), ndim=4)]
    ur3 = u[crop(axis, (6, None), ndim=4)]

    inner_slice = insert_slice(crop(axis, (3, -3), ndim=4), 4, 0)
    alpha_inner = out[inner_slice]

    smooth_extrema_detector_neighbor_min_kernel(
        ul3,
        ul2,
        ul1,
        uc,
        ur1,
        ur2,
        ur3,
        eps,
        check_uniformity,
        uniformity_tol,
        alpha_inner,
    )

    modified = cast(Tuple[slice, ...], replace_slice(inner_slice, 4, slice(None, 1)))
    return modified
