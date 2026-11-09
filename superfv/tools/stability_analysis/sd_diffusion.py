# mypy: ignore-errors


from functools import lru_cache

import numpy as np

from superfv.tools.stability_analysis.sd_advection import (
    compute_solution_and_flux_point_coordinates,
    fp_to_sp_prime_matrix,
    sp_to_fp_matrix,
)

na = np.newaxis


@lru_cache(maxsize=None)
def derive_sd_diffusion_Omega_h_squared_over_nu_matrix(k, p):
    """
    k: wave number
    p: polynomial degree
    returns: the matrix representation of the SD diffusion scheme's Omega * h^2 / nu,
        where h is the subcell spacing
    """
    sp_to_fp = sp_to_fp_matrix(p).astype(np.complex128)
    fp_to_sp_prime = fp_to_sp_prime_matrix(p).astype(np.complex128)
    phase = np.exp(1j * k * (p + 1))

    # Choose the left state before differentiating the solution.
    sp_to_fp_left = sp_to_fp.copy()
    sp_to_fp_left[0] = sp_to_fp[-1] / phase

    # Choose the right state before differentiating the gradient.
    sp_to_fp_right = sp_to_fp.copy()
    sp_to_fp_right[-1] = sp_to_fp[0] * phase

    return fp_to_sp_prime @ sp_to_fp_right @ fp_to_sp_prime @ sp_to_fp_left


@lru_cache(maxsize=None)
def derive_MUSCL_diffusion_Omega_h_squared_over_nu_matrix(k, p):
    """
    k: wave number
    p: SD polynomial degree defining the nonuniform finite volume subcells
    returns: the matrix representation of the unlimited MUSCL diffusion scheme's
        Omega * h^2 / nu, where h is the average subcell spacing
    """
    _, flux_points = compute_solution_and_flux_point_coordinates(p)
    cell_widths = flux_points[1:] - flux_points[:-1]
    cell_centers = 0.5 * (flux_points[:-1] + flux_points[1:])
    phase = np.exp(1j * k * (p + 1))

    identity = np.eye(p + 1, dtype=np.complex128)
    cell_to_left = np.roll(identity, 1, axis=0)
    cell_to_left[0] /= phase
    cell_to_right = np.roll(identity, -1, axis=0)
    cell_to_right[-1] *= phase

    left_spacing = cell_centers - np.roll(cell_centers, 1)
    left_spacing[0] += p + 1
    right_spacing = np.roll(cell_centers, -1) - cell_centers
    right_spacing[-1] += p + 1

    central_difference = (cell_to_right - cell_to_left) / (left_spacing + right_spacing)[:, na]

    # Choose the left solution state before differentiating it at cell centers.
    cell_to_left_face = identity + 0.5 * cell_widths[:, na] * central_difference
    cell_to_gradient = (identity - cell_to_left) @ cell_to_left_face / cell_widths[:, na]

    # Choose the right gradient state at each diffusive face.
    gradient_to_right_face = cell_to_right @ (
        identity - 0.5 * cell_widths[:, na] * central_difference
    )

    return (
        (identity - cell_to_left) @ gradient_to_right_face @ cell_to_gradient / cell_widths[:, na]
    )


def compute_sd_diffusion_Omega_h_squared_over_nu_eigenvalues(k_values, p):
    """
    k_values: array of wave numbers
    p: polynomial degree
    returns: the eigenvalues of the matrix representation of the SD diffusion scheme's
        Omega * h^2 / nu
    """
    eigenvalues = np.empty((len(k_values), p + 1), dtype=np.complex128)
    for i, k in enumerate(k_values):
        mat = derive_sd_diffusion_Omega_h_squared_over_nu_matrix(k, p)
        eigenvalues[i, :] = np.linalg.eigvals(mat)
    return eigenvalues
