# mypy: ignore-errors


from functools import lru_cache

import numpy as np

from superfv.tools.stability_analysis.sd_advection import (
    fp_to_sp_prime_matrix,
    sp_to_fp_matrix,
)


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
