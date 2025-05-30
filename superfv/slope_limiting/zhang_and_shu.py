from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

import numpy as np

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import ArrayLike, crop_to_center, intersection_shape

from .smooth_extrema_detection import compute_smooth_extrema_detector

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver


def zhang_shu_operator(u_ho: ArrayLike, u_fo: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        u_ho (ArrayLike): Array of high-order interpolation values.
        u_fo (ArrayLike): Array of first-order interpolation values.
        theta (ArrayLike): Array of limiting coefficients.

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
        fv_solver (FiniteVolumeSolver): Finite volume solver object.
        averages (ArrayLike): Array of finite volume averages, has shape
            (nvars, nx, ny, nz).
        interpolation_scheme (Literal["transverse", "gauss-legendre"]): Interpolation
            mode.
        p (int): Polynomial interpolation degree.

    Returns:
        Tuple[Optional[ArrayLike], ...]: Arrays of interpolation nodes if the face
            along the respective dimension is used, otherwise None. If not None, the
            arrays have shape (nvars, nx, ny, nz, ninterpolations) and are ordered as
            follows:
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
        interpolation_scheme=interpolation_scheme,
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
        fv_solver (FiniteVolumeSolver): Finite volume solver object.
        averages (ArrayLike): Array of conservative cell averages. Has shape
            (nvars, nx, ny, nz).
        dim (Literal["x", "y", "z"]): Dimension of the face to limit.
        interpolation_scheme (Literal["transverse", "gauss-legendre"]): Interpolation
            mode.
        p (int): Polynomial order. Must be >= 1.
        tol (float): Tolerance for dividing by zero.
        SED (bool): Whether to use the SED method.
        convert_to_primitives (bool): Whether to convert the high-order nodes to
            primitive variables before limiting. If True, `primitive_fallback` must be
            provided.
        primitive_fallback (Optional[ArrayLike]): Fallback values used by the limiter
            if `convert_to_primitives` is True.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Limited face values (left, right). Each has shape
            (nvars, <nx, <ny, <nz, ninterpolations).
    """
    xp = fv_solver.xp
    _slc = fv_solver.array_slicer
    __limiting_slc__ = slice(None)  # All variables for now

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
        xp, averages[__limiting_slc__], dims=fv_solver.dims, include_corners=False
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
            axes=(1, 2, 3),
        )

    # crop some arrays
    hocc = hocc[__limiting_slc__]
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
            update_min_and_max(left[__limiting_slc__])
            update_min_and_max(right[__limiting_slc__])

    # compute theta
    _foc = foc[__limiting_slc__]
    theta = np.minimum(
        np.minimum(
            np.divide(np.abs(M - _foc), np.abs(Mj - _foc) + tol),
            np.divide(np.abs(m - _foc), np.abs(mj - _foc) + tol),
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
            )[__limiting_slc__],
            axes=fv_solver.axes,
        )
        alpha = fv_solver.bc_for_smooth_extrema_detection(
            alpha,
            (
                (theta.shape[1] - fv_solver.nx) // 2,
                (theta.shape[2] - fv_solver.ny) // 2,
                (theta.shape[3] - fv_solver.nz) // 2,
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
