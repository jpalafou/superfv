from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

import numpy as np

from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import (
    inplace_smooth_extrema_detector,
)
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import crop_to_center, intersection_shape, modify_slices

from .smooth_extrema_detection import compute_smooth_extrema_detector

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver


def compute_theta(
    xp: ModuleType,
    u: ArrayLike,
    center_nodes: ArrayLike,
    x_nodes: Optional[ArrayLike],
    y_nodes: Optional[ArrayLike],
    z_nodes: Optional[ArrayLike],
    buffer: ArrayLike,
    out: ArrayLike,
    include_corners: bool = False,
    SED: bool = False,
    tol: float = 1e-16,
):
    """
    Compute Zhang and Shu's a priori slope limiting parameter theta based on arrays of
    finite-volume nodes and averages.

    Args:
        xp: `np` namespace.
        u: Array of finite-volume average. Has shape (nvars, nx, ny, nz).
        center_nodes: Array of central node values. Has shape (nvars, nx, ny, nz, 1).
        x_nodes, y_nodes, z_nodes: Optional array of x,y,z-face node values. Has shape
            (nvars, nx, ny, nz, 2*n_nodes). If None, the x,y,z face is not considered.
        buffer: Array to which intermediate values are written.
        out: Array to which theta is written. Has shape (nvars, nx, ny, nz, 1).
        include_corners: Whether to include corners when computing the discrete maximum
            principle. Defaults to False.
        tol: Small tolerance value for division.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """

    # allocate arrays
    dmp = buffer[..., :2]
    node_mp = buffer[..., 2:4]

    # compute discrete maximum principle
    active_dims = tuple(
        dim
        for dim, arr in zip(["x", "y", "z"], [x_nodes, y_nodes, z_nodes])
        if arr is not None
    )
    dmp_modified = compute_dmp(xp, u, active_dims, include_corners, dmp)

    # compute nodal maximum principle
    node_mp[..., 0] = center_nodes[..., 0]
    node_mp[..., 1] = center_nodes[..., 0]
    for nodes in [x_nodes, y_nodes, z_nodes]:
        if nodes is None:
            continue
        xp.minimum(node_mp[..., 0], xp.min(nodes, axis=4), out=node_mp[..., 0])
        xp.maximum(node_mp[..., 1], xp.max(nodes, axis=4), out=node_mp[..., 1])

    # compute theta
    m = dmp[..., 0]
    M = dmp[..., 1]
    mj = node_mp[..., 0]
    Mj = node_mp[..., 1]
    theta = xp.minimum(
        xp.minimum(
            xp.divide(xp.abs(M - u), xp.abs(Mj - u) + tol),
            xp.divide(xp.abs(m - u), xp.abs(mj - u) + tol),
        ),
        1.0,
    )

    # assign theta
    out_modified = modify_slices(dmp_modified, axis=4, new_slice=0)
    out[out_modified] = theta[out_modified[:-1]]

    # relax theta using a smooth extrema detector
    if SED:
        alpha = buffer[..., :1]
        modified = inplace_smooth_extrema_detector(
            xp, u, active_dims, buffer[..., 1:], alpha
        )
        out[modified] = xp.where(alpha[modified] == 1, 1, out[modified])
    else:
        modified = modify_slices(out_modified, axis=4, new_slice=slice(None, 1))

    return modified


def zhang_shu_operator(u_ho: ArrayLike, u_fo: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        u_ho: Array of high-order interpolation values.
        u_fo: Array of first-order interpolation values.
        theta: Array of limiting coefficients.

    Returns:
        ArrayLike: Array of limited values.
    """
    return theta * (u_ho - u_fo) + u_fo


def gather_nodes(
    fv_solver: FiniteVolumeSolver,
    averages: ArrayLike,
    interpolation_scheme: Literal["transverse", "gauss-legendre"],
    p: int,
) -> Tuple[
    Optional[ArrayLike],
    Optional[ArrayLike],
    Optional[ArrayLike],
    Optional[ArrayLike],
    Optional[ArrayLike],
    Optional[ArrayLike],
    ArrayLike,
]:
    """
    Return nodal interpolations for all cell faces.

    Args:
        fv_solver: FiniteVolumeSovler object.
        averages: Array of finite volume averages, has shape (nvars, nx, ny, nz).
        interpolation_scheme: Interpolation mode for computing the face nodes. Possible
            values:
            - "transverse": Transverse interpolation.
            - "gauss-legendre": Gauss-Legendre interpolation.
        p: Polynomial interpolation degree.

    Returns:
        Seven array of interpolation nodes, one for each face and the centroid.
        If the face along the respective dimension is not used, the corresponding
        array is None. The arrays have shape (nvars, nx, ny, nz, ninterpolations) and
        are ordered as follows:
        - Left x face node array or None.
        - Right x face node array or None.
        - Left y face node array or None.
        - Right y face node array or None.
        - Left z face node array or None.
        - Right z face node array or None.
        - Centroid node array with ninterpolations=1.
    """
    hoxl, hoxr = (
        fv_solver.interpolate_face_nodes(
            averages,
            dim="x",
            interpolation_scheme=interpolation_scheme,
            p=p,
            slope_limiter=None,
        )
        if fv_solver.using_xdim
        else (None, None)
    )
    hoyl, hoyr = (
        fv_solver.interpolate_face_nodes(
            averages,
            dim="y",
            interpolation_scheme=interpolation_scheme,
            p=p,
            slope_limiter=None,
        )
        if fv_solver.using_ydim
        else (None, None)
    )
    hozl, hozr = (
        fv_solver.interpolate_face_nodes(
            averages,
            dim="z",
            interpolation_scheme=interpolation_scheme,
            p=p,
            slope_limiter=None,
        )
        if fv_solver.using_zdim
        else (None, None)
    )
    hocc = fv_solver.interpolate_cell_centers(
        averages,
        p=p,
        sweep_order="xyz",
    )[..., np.newaxis]
    return (hoxl, hoxr, hoyl, hoyr, hozl, hozr, hocc)


def zhang_shu_limiter(
    fv_solver: FiniteVolumeSolver,
    averages: ArrayLike,
    dim: Literal["x", "y", "z"],
    interpolation_scheme: Literal["transverse", "gauss-legendre"],
    p: int,
    tol: float = 1e-16,
    SED: bool = False,
    convert_to_primitives: bool = False,
    primitive_fallback: Optional[ArrayLike] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Caches the limited face values for the advection equation using a variable-wise
    Zhang-Shu limiter for all variables.

    Args:
        fv_solver: FiniteVolumeSolver object.
        averages: Array of conservative cell averages. Has shape (nvars, nx, ny, nz).
        dim: Dimension of the face to limit. Can be "x", "y", or "z".
        interpolation_scheme: Interpolation mode for computing the face nodes. Possible
            values:
            - "transverse": Transverse interpolation.
            - "gauss-legendre": Gauss-Legendre interpolation.
        p: Polynomial interpolation degree. Must be >= 1.
        tol: Tolerance for dividing by zero. Default is 1e-16.
        SED: Whether to use the Smooth Extrema Detector when computing the limiter
            theta.
        convert_to_primitives: Whether to convert the high-order nodes to primitive
            variables before limiting. If True, `primitive_fallback` must be provided.
        primitive_fallback: Fallback values used by the limiter if
            `convert_to_primitives` is True. If None, the averages are used.

    Returns:
        Left and right limited face node values as a tuple of two arrays. Each array
        has shape (nvars, <nx, <ny, <nz, ninterpolations).
    """
    xp = fv_solver.xp
    limiting_slice = slice(None)  # All variables for now

    def zs_str(p, dim, pos):
        return f"zs_p{p}{dim}{pos}_face_nodes"

    # early escape if the limited nodes have already been computed
    l_str, r_str = zs_str(p, dim, "l"), zs_str(p, dim, "r")
    if l_str in fv_solver.ZS_cache:
        return fv_solver.ZS_cache[l_str], fv_solver.ZS_cache[r_str]

    # clear ZS cache and compute the zhang-shu limited nodes
    if p < 1:
        raise ValueError("p must be >= 1.")

    # get interpolations
    hoxl, hoxr, hoyl, hoyr, hozl, hozr, hocc = gather_nodes(
        fv_solver, averages, interpolation_scheme, p
    )
    if convert_to_primitives:
        if fv_solver.using_xdim:
            hoxl = fv_solver.primitives_from_conservatives(cast(ArrayLike, hoxl))
            hoxr = fv_solver.primitives_from_conservatives(cast(ArrayLike, hoxr))
        if fv_solver.using_ydim:
            hoyl = fv_solver.primitives_from_conservatives(cast(ArrayLike, hoyl))
            hoyr = fv_solver.primitives_from_conservatives(cast(ArrayLike, hoyr))
        if fv_solver.using_zdim:
            hozl = fv_solver.primitives_from_conservatives(cast(ArrayLike, hozl))
            hozr = fv_solver.primitives_from_conservatives(cast(ArrayLike, hozr))
        hocc = fv_solver.primitives_from_conservatives(cast(ArrayLike, hocc))
        if primitive_fallback is None:
            raise ValueError(
                "Fallback must be provided if convert_to_primitives is True."
            )
        foc = primitive_fallback
    else:
        foc = averages

    # get dmp
    m, M = compute_dmp(
        xp, averages[limiting_slice], dims=fv_solver.dims, include_corners=False
    )

    # get common shape
    shapes = [
        (1,) + cast(ArrayLike, arr).shape[1:4] + (1,)
        for arr in [hoxl, hoyl, hozl, hocc, m]
        if arr is not None
    ]
    target_shape = intersection_shape(*shapes)

    # helper function to crop arrays
    def crop(arr: ArrayLike, add_axis: bool = False) -> ArrayLike:
        return crop_to_center(
            arr[..., np.newaxis] if add_axis else arr,
            target_shape,
            ignore_axes=(0, 4),
        )

    # crop some arrays
    hocc = hocc[limiting_slice]
    foc, m, M = map(lambda arr: crop(arr, add_axis=True), [foc, m, M])

    # compute nodal min, max
    mj, Mj = hocc.copy(), hocc.copy()

    def update_min_and_max(arr):
        """
        Update min and max nodes.
        """
        np.minimum(mj, np.min(arr, axis=4, keepdims=True), out=mj)
        np.maximum(Mj, np.max(arr, axis=4, keepdims=True), out=Mj)

    for using_dim, left, right in [
        (fv_solver.using_xdim, hoxl, hoxr),
        (fv_solver.using_ydim, hoyl, hoyr),
        (fv_solver.using_zdim, hozl, hozr),
    ]:
        if using_dim:
            left, right = map(crop, [cast(ArrayLike, left), cast(ArrayLike, right)])
            update_min_and_max(left[limiting_slice])
            update_min_and_max(right[limiting_slice])

    # compute theta
    foc = foc[limiting_slice]
    theta = np.minimum(
        np.minimum(
            np.divide(np.abs(M - foc), np.abs(Mj - foc) + tol),
            np.divide(np.abs(m - foc), np.abs(mj - foc) + tol),
        ),
        1,
    )

    # compute smooth extrema detector
    if SED:
        alpha = compute_smooth_extrema_detector(
            xp,
            fv_solver.apply_bc(
                crop_to_center(
                    (
                        cast(ArrayLike, primitive_fallback)
                        if convert_to_primitives
                        else averages
                    ),
                    fv_solver.arrays["u"].shape,
                ),
                3,
            )[limiting_slice],
            axes=fv_solver.axes,
        )
        alpha = fv_solver.bc_for_smooth_extrema_detection(
            alpha,
            (
                (theta.shape[1] - fv_solver.mesh.nx) // 2,
                (theta.shape[2] - fv_solver.mesh.ny) // 2,
                (theta.shape[3] - fv_solver.mesh.nz) // 2,
            ),
        )
        alpha = crop(alpha, add_axis=True)
        theta[...] = np.where(alpha == 1, 1, theta)

    # limit and escape
    if fv_solver.using_xdim:
        fv_solver.ZS_cache.add(
            zs_str(p, "x", "l"), zhang_shu_operator(cast(ArrayLike, hoxl), foc, theta)
        )
        fv_solver.ZS_cache.add(
            zs_str(p, "x", "r"), zhang_shu_operator(cast(ArrayLike, hoxr), foc, theta)
        )
    if fv_solver.using_ydim:
        fv_solver.ZS_cache.add(
            zs_str(p, "y", "l"), zhang_shu_operator(cast(ArrayLike, hoyl), foc, theta)
        )
        fv_solver.ZS_cache.add(
            zs_str(p, "y", "r"), zhang_shu_operator(cast(ArrayLike, hoyr), foc, theta)
        )
    if fv_solver.using_zdim:
        fv_solver.ZS_cache.add(
            zs_str(p, "z", "l"), zhang_shu_operator(cast(ArrayLike, hozl), foc, theta)
        )
        fv_solver.ZS_cache.add(
            zs_str(p, "z", "r"), zhang_shu_operator(cast(ArrayLike, hozr), foc, theta)
        )
    return fv_solver.ZS_cache[l_str], fv_solver.ZS_cache[r_str]
