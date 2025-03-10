from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import numpy as np

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import ArrayLike, crop_to_center, intersection_shape

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


def get_all_high_order_face_nodes(
    fv_solver: FiniteVolumeSolver,
    averages: ArrayLike,
    interpolation_scheme: Literal["transverse", "gauss-legendre"],
    p: int,
) -> List[List[Optional[ArrayLike]]]:
    """
    Get high-order nodes for all faces and cache them in the interpolation cache.

    Args:
        fv_solver (FiniteVolumeSolver): Finite volume solver object.
        averages (ArrayLike): Array of cell averages. Has shape (nvars, nx, ny, nz).
        interpolation_scheme (Literal["transverse", "gauss-legendre"]): Interpolation
            mode.
        p (int): Polynomial order. Must be >= 1.

    Returns:
        List[List[Optional[ArrayLike]]]: List of face nodes ([[left, right], ...]).
            Each face node array has shape (nvars, <nx, <ny, <nz, ninterpolations).
    """
    faces: List[List[Optional[ArrayLike]]] = []
    for dim in ["x", "y", "z"]:
        if not fv_solver.using[dim]:
            faces.append([None, None])
            continue
        faces.append(
            [
                *fv_solver.interpolate_face_nodes(
                    averages,
                    dim=dim,
                    interpolation_scheme=interpolation_scheme,
                    p=p,
                    slope_limiter=None,
                )
            ]
        )
    return faces


def zhang_shu_advection(
    fv_solver: FiniteVolumeSolver,
    averages: ArrayLike,
    dim: Literal["x", "y", "z"],
    interpolation_scheme: Literal["transverse", "gauss-legendre"],
    p: int,
    tol: float = 1e-16,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Zhang and Shu slope limiter for the scalar advection equation.

    Args:
        fv_solver (FiniteVolumeSolver): Finite volume solver object.
        averages (ArrayLike): Array of cell averages. Has shape (nvars, nx, ny, nz).
        dim (Literal["x", "y", "z"]): Dimension of the face to limit.
        interpolation_scheme (Literal["transverse", "gauss-legendre"]): Interpolation
            mode.
        p (int): Polynomial order. Must be >= 1.
        tol (float): Tolerance for dividing by zero.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Limited face values (left, right). Each has shape
            (nvars, <nx, <ny, <nz).
    """
    xp = fv_solver.xp

    def zs_str(p, dim, pos):
        return f"zs_p{p}{dim}{pos}_face_nodes"

    # early escape if the limited nodes have already been computed
    l_str, r_str = zs_str(p, dim, "l"), zs_str(p, dim, "r")
    if l_str in fv_solver.ZS_cache:
        return fv_solver.ZS_cache[l_str], fv_solver.ZS_cache[r_str]

    # clear ZS cache and compute the zhang-shu limited nodes
    _slc = fv_solver.array_slicer
    if p < 1:
        raise ValueError("p must be >= 1.")

    # get interpolations
    (hoxl, hoxr), (hoyl, hoyr), (hozl, hozr) = get_all_high_order_face_nodes(
        fv_solver, averages, interpolation_scheme, p
    )
    hoc = fv_solver.interpolate_cell_centers(
        averages,
        interpolation_scheme=interpolation_scheme,
        p=p,
        sweep_order="xyz",
    )
    foc = averages

    # get dmp
    m, M = compute_dmp(xp, averages, dims=fv_solver.dims, include_corners=False)

    # get common shape
    shapes = [
        cast(ArrayLike, arr).shape[:4] + (1,)
        for arr in [hoxl, hoyl, hozl, hoc, m]
        if arr is not None
    ]
    target_shape = intersection_shape(*shapes)

    # helper function to crop arrays
    def crop(arr, add_axis=False):
        """
        Crop array to match target shape, optionally adding a new axis.
        """
        if add_axis:
            arr = arr[..., np.newaxis]
        return crop_to_center(arr, target_shape, axes=(1, 2, 3))

    # crop the cell scalar quantities, adding an axis, and update min and max nodes
    foc, hoc, m, M = map(lambda arr: crop(arr, add_axis=True), [foc, hoc, m, M])
    mj, Mj = hoc.copy(), hoc.copy()

    def update_min_and_max(arr):
        """
        Update min and max nodes.
        """
        np.minimum(mj, np.min(arr, axis=4, keepdims=True), out=mj)
        np.maximum(Mj, np.max(arr, axis=4, keepdims=True), out=Mj)

    # loop through dimensions, cropping without adding an axis
    for using_dim, left, right in [
        (fv_solver.using_xdim, hoxl, hoxr),
        (fv_solver.using_ydim, hoyl, hoyr),
        (fv_solver.using_zdim, hozl, hozr),
    ]:
        if using_dim:
            left, right = map(crop, [left, right])  # No new axis here
            update_min_and_max(left)
            update_min_and_max(right)

    # compute theta
    theta = np.ones_like(foc)
    theta[_slc("rho")] = np.minimum(
        np.minimum(
            np.divide(np.abs(M - foc), np.abs(Mj - foc) + tol),
            np.divide(np.abs(m - foc), np.abs(mj - foc) + tol),
        ),
        1,
    )[_slc("rho")]

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
