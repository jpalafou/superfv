# mypy: ignore-errors


from functools import lru_cache

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss

from .fv_polynomials import unweighted_lagrange_interpolation_polynomials
from .variables import h, x

h_ave = h
u0 = sp.symbols("u0", real=True)  # reference solution value inside element m=0


def chebyshev_zeros(n):
    if n == 0:
        return np.array([])
    return np.array(
        [float(sp.functions.special.polynomials.chebyshevt_root(n, i)) for i in range(n)]
    )


def guass_legendre_points(n):
    if n == 0:
        return np.array([])
    xi, _ = leggauss(n)
    return xi


# @lru_cache(maxsize=None)
def compute_solution_and_flux_point_coordinates(p):
    """
    p: polynomial degree
    returns:
        sps: solution point coordinates on [-(p+1)/2, (p+1)/2]
        fps: flux point coordinates on [-(p+1)/2, (p+1)/2]
    """
    sps = np.sort(chebyshev_zeros(p + 1))
    fps = np.array([-1] + list(np.sort(guass_legendre_points(p))) + [1])
    rescale = (p + 1) / 2  # let h=1
    return sps * rescale, fps * rescale


@lru_cache(maxsize=None)
def sp_to_fp_matrix(p):
    """
    p: polynomial degree
    returns: SD matrix that transforms the solution point values to the flux point values for a
    given polynomial order p. Solution points and fluxes points are on [-(p+1)/2, (p+1)/2].
    """
    sps, fps = compute_solution_and_flux_point_coordinates(p)

    lagrange_polynomials = unweighted_lagrange_interpolation_polynomials(sps)

    mat = np.empty((p + 2, p + 1))
    for i, _x in enumerate(fps):
        for j, Lj in enumerate(lagrange_polynomials):
            mat[i, j] = float(Lj.subs(x, _x))

    return mat


@lru_cache(maxsize=None)
def fp_to_sp_prime_matrix(p):
    """
    p: polynomial degree
    returns: SD matrix that transforms the flux point values to the solution point derivatives for
    a given polynomial order p. Solution points and fluxes points are on [-(p+1)/2, (p+1)/2].
    """
    sps, fps = compute_solution_and_flux_point_coordinates(p)

    lagrange_polynomials = unweighted_lagrange_interpolation_polynomials(fps)

    mat = np.empty((p + 1, p + 2))
    for i, _x in enumerate(sps):
        for j, Lj in enumerate(lagrange_polynomials):
            mat[i, j] = float(sp.diff(Lj, x).subs(x, _x))

    return mat


def harmonic_sp_dof(k, p, n, m=0):
    """
    k: wave number
    p: polynomial degree
    n: index of the solution point inside the element
    m: index of the element
    returns: the harmonic mode representation of the solution point value at index n in element m,
        divided by the reference solution value u0
    """
    sps, _ = compute_solution_and_flux_point_coordinates(p)
    try:
        d = sps[n]
    except IndexError:
        raise IndexError(f"Invalid solution point index n={n} for polynomial degree p={p}.")
    elem_length = p + 1  # let h=1
    x_mn = elem_length * m + d
    return np.exp(1j * k * x_mn)


def solution_vector(k, p, m=0):
    """
    k: wave number
    p: polynomial degree
    m: index of the element
    returns: the vector of harmonic mode representations of the solution point values in element m,
        divided by the reference solution value u0
    """
    dofs = [harmonic_sp_dof(k, p, n, m) for n in range(p + 1)]
    return np.array(dofs)


@lru_cache(maxsize=None)
def derive_sd_upwind_Omega_h_over_a_matrix(k, p):
    """
    k: wave number
    p: polynomial degree
    returns: the matrix representation of the upwind SD scheme's Omega * h / a
    """
    sp_to_fp = sp_to_fp_matrix(p).astype(np.complex128)
    fp_to_sp_prime = fp_to_sp_prime_matrix(p).astype(np.complex128)
    phase = np.exp(1j * k * (p + 1))

    # Choose the left state before differentiating the flux.
    sp_to_fp_upwind = sp_to_fp.copy()
    sp_to_fp_upwind[0] = sp_to_fp[-1] / phase

    return -fp_to_sp_prime @ sp_to_fp_upwind


def compute_sd_upwind_Omega_h_over_a_eigenvalues(k_values, p):
    """
    k_values: array of wave numbers
    p: polynomial degree
    returns: the eigenvalues of the matrix representation of the upwind SD scheme's Omega * h / a
    """
    eigenvalues = np.empty((len(k_values), p + 1), dtype=np.complex128)
    for i, k in enumerate(k_values):
        mat = derive_sd_upwind_Omega_h_over_a_matrix(k, p)
        eigenvalues[i, :] = np.linalg.eigvals(mat)
    return eigenvalues


def plot_sd_advection_stability_trace(ax, p, k_values, **kwargs):
    eigenvalues = compute_sd_upwind_Omega_h_over_a_eigenvalues(k_values, p)

    ax.plot(
        eigenvalues.real.flat,
        eigenvalues.imag.flat,
        linestyle="none",
        marker=".",
        mfc="none",
        **kwargs,
    )
