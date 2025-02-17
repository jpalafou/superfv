from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import ArrayLike, crop_to_center

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
        dim (Literal["x", "y", "z"]): Dimension to limit.
        interpolation_scheme (Literal["transverse", "gauss-legendre"]): Interpolation
            mode.
        p (int): Polynomial order. Must be >= 1.
        tol (float): Tolerance for dividing by zero.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Limited face values (left, right). Each has shape
            (nvars, <nx, <ny, <nz).
    """
    _slc = fv_solver.array_slicer
    if p < 1:
        raise ValueError("p must be >= 1.")

    # get interpolations
    fo_left_face, fo_right_face = averages[..., np.newaxis], averages[..., np.newaxis]
    ho_left_face, ho_right_face = fv_solver.interpolate_face_nodes(
        averages,
        dim=dim,
        interpolation_scheme=interpolation_scheme,
        p=p,
        slope_limiter=None,
    )
    ho_center = fv_solver.interpolate_cell_centers(
        averages,
        interpolation_scheme=interpolation_scheme,
        p=p,
        sweep_order={"x": "yzx", "y": "zxy", "z": "xyz"}[dim],
    )[..., np.newaxis]

    # get dmp
    m, M = compute_dmp(averages, dims=fv_solver.dims, include_corners=False)

    # crop to match dmp shape and select only rho
    target_shape = ho_left_face.shape
    fo_left_face = crop_to_center(fo_left_face, target_shape, axes=(1, 2, 3))
    fo_right_face = crop_to_center(fo_right_face, target_shape, axes=(1, 2, 3))
    ho_left_face = ho_left_face
    ho_right_face = ho_right_face
    ho_center = ho_center
    m = crop_to_center(m[..., np.newaxis], target_shape, axes=(1, 2, 3))
    M = crop_to_center(M[..., np.newaxis], target_shape, axes=(1, 2, 3))

    # limit
    limited_faces = []
    for ho_face, fo_face in zip(
        [ho_left_face, ho_right_face], [fo_left_face, fo_right_face]
    ):
        nodes = np.concatenate((ho_face, ho_center), axis=4)
        mj = np.min(nodes, axis=4, keepdims=True)
        Mj = np.max(nodes, axis=4, keepdims=True)
        theta = np.minimum(
            np.minimum(
                np.divide(np.abs(M - fo_face), np.abs(Mj - fo_face) + tol),
                np.divide(np.abs(m - fo_face), np.abs(mj - fo_face) + tol),
            ),
            1,
        )
        limited_face = ho_face.copy()
        limited_face[_slc("rho")] = zhang_shu_operator(
            ho_face[_slc("rho")],
            fo_face[_slc("rho")],
            theta[_slc("rho")],
        )
        limited_faces.append(limited_face)

    return limited_faces[0], limited_faces[1]
