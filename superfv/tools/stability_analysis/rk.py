# mypy: ignore-errors

import numpy as np
import sympy as sp

from superfv.tools.stability_analysis.fv_diffusion import (
    derive_fv_diffusion_Omega_h_squared_over_nu_trace_func,
)
from superfv.tools.stability_analysis.sd_advection import (
    derive_MUSCL_upwind_Omega_h_over_a_matrix,
    derive_sd_upwind_Omega_h_over_a_matrix,
)

from .fv_advection import derive_fv_upwind_Omega_h_over_a_trace_func
from .sd_diffusion import (
    derive_MUSCL_diffusion_Omega_h_squared_over_nu_matrix,
    derive_sd_diffusion_Omega_h_squared_over_nu_matrix,
)
from .variables import Omega, dt, u0, z

Omega_dt = sp.symbols("Omega_dt")


def module_variable(func):
    """
    decorator to make the output of a function with no arguments a module-level variable
    """
    return func()


def get_amplification_factor(update):
    """
    given an RK update for u1 in terms of u0, return the amplification factor |u1 / u0|
    """
    return sp.Abs(update / u0)


def get_amplification_function(update):
    """
    given an RK update for u1 in terms of u0, return a function that returns the amplification
    factor |u1 / u0| as a function of Omega_dt
    """
    in_terms_of_z = update.subs(Omega * dt, z)
    return sp.lambdify(z, get_amplification_factor(in_terms_of_z), modules="numpy")


@module_variable
def forward_euler_update():
    return sp.simplify(u0 + dt * Omega * u0)


@module_variable
def forward_euler_amplification():
    return get_amplification_factor(forward_euler_update)


@module_variable
def forward_euler_amplification_function():
    return get_amplification_function(forward_euler_update)


@module_variable
def ssprk2_update():
    k0 = Omega * u0
    u1 = u0 + dt * k0
    k1 = Omega * u1
    u2 = sp.Rational(1, 2) * (u0 + u1 + dt * k1)
    return sp.simplify(u2)


@module_variable
def ssprk2_amplification():
    return get_amplification_factor(ssprk2_update)


@module_variable
def ssprk2_amplification_function():
    return get_amplification_function(ssprk2_update)


@module_variable
def ssprk3_update():
    k0 = Omega * u0
    u1 = u0 + dt * k0
    k1 = Omega * u1
    u2 = u0 + sp.Rational(1, 4) * dt * (k0 + k1)
    k2 = Omega * u2
    u3 = u0 + sp.Rational(1, 6) * dt * (k0 + k1 + 4 * k2)
    return sp.simplify(u3)


@module_variable
def ssprk3_amplification():
    return get_amplification_factor(ssprk3_update)


@module_variable
def ssprk3_amplification_function():
    return get_amplification_function(ssprk3_update)


@module_variable
def rk4_update():
    k0 = Omega * u0
    k1 = Omega * (u0 + sp.Rational(1, 2) * dt * k0)
    k2 = Omega * (u0 + sp.Rational(1, 2) * dt * k1)
    k3 = Omega * (u0 + dt * k2)
    u1 = u0 + sp.Rational(1, 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
    return sp.simplify(u1)


@module_variable
def rk4_amplification():
    return get_amplification_factor(rk4_update)


@module_variable
def rk4_amplification_function():
    return get_amplification_function(rk4_update)


def compute_rk_amplification(
    p, k_values, scheme="fv", rk="ssprk3", operator="advection", *, coeff_C=None, coeff_D=None
):
    """
    p: polynomial degree of spatial discretization
    k_values: array of nondimensional wave numbers k * h, where h is the subcell spacing
    scheme: "fv", "sd", or "sd_fallback"
    rk: Runge-Kutta method, one of "forward_euler", "ssprk2", "ssprk3", "rk4"
    operator: "advection", "diffusion", or "both"
    coeff_C: advection coefficient C = dt * a / h
    coeff_D: diffusion coefficient D = dt * nu / h^2

    returns: the maximum amplification factor over all wave numbers for the given spatial
        discretization, operator coefficients, and Runge-Kutta method
    """
    if operator not in ("advection", "diffusion", "both"):
        raise ValueError(f"Unknown operator: {operator}")

    omega_dt_values = 0.0
    if scheme == "fv":
        if operator in ("advection", "both"):
            if coeff_C is None:
                raise ValueError("coeff_C must be provided for advection operator")
            afunc = derive_fv_upwind_Omega_h_over_a_trace_func(p)
            omega_h_over_a_values = afunc(k_values)
            omega_dt_values += omega_h_over_a_values * coeff_C
        if operator in ("diffusion", "both"):
            if coeff_D is None:
                raise ValueError("coeff_D must be provided for diffusion operator")
            dfunc = derive_fv_diffusion_Omega_h_squared_over_nu_trace_func(p)
            omega_h_squared_over_nu_values = dfunc(k_values)
            omega_dt_values += omega_h_squared_over_nu_values * coeff_D
    elif scheme in ("sd", "sd_fallback"):
        if scheme == "sd":
            derive_advection_matrix = derive_sd_upwind_Omega_h_over_a_matrix
            derive_diffusion_matrix = derive_sd_diffusion_Omega_h_squared_over_nu_matrix
        else:
            derive_advection_matrix = derive_MUSCL_upwind_Omega_h_over_a_matrix
            derive_diffusion_matrix = derive_MUSCL_diffusion_Omega_h_squared_over_nu_matrix

        mat = np.zeros((len(k_values), p + 1, p + 1), dtype=np.complex128)
        eigenvalues = np.empty((len(k_values), p + 1), dtype=np.complex128)
        if operator in ("advection", "both"):
            if coeff_C is None:
                raise ValueError("coeff_C must be provided for advection operator")
            for i, kval in enumerate(k_values):
                amat = derive_advection_matrix(kval, p)
                mat[i] += coeff_C * amat
        if operator in ("diffusion", "both"):
            if coeff_D is None:
                raise ValueError("coeff_D must be provided for diffusion operator")
            for i, kval in enumerate(k_values):
                dmat = derive_diffusion_matrix(kval, p)
                mat[i] += coeff_D * dmat
        for i in range(len(k_values)):
            eigenvalues[i] = np.linalg.eigvals(mat[i])
        omega_dt_values = eigenvalues.flatten()
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    match rk:
        case "forward_euler":
            amplification_function = forward_euler_amplification_function
        case "ssprk2":
            amplification_function = ssprk2_amplification_function
        case "ssprk3":
            amplification_function = ssprk3_amplification_function
        case "rk4":
            amplification_function = rk4_amplification_function
        case _:
            raise ValueError(f"Unknown Runge-Kutta method: {rk}")

    amplification_values = amplification_function(omega_dt_values)
    return np.max(np.abs(amplification_values))


def plot_stability_region(ax, amplification, z_real_1D, z_imag_1D, fill=False, **kwargs):
    """
    ax: matplotlib axis
    amplification: sympy expression for the amplification factor |u1 / u0| as a function of z
    z_real_1D: 1D array of real parts of z values
    z_imag_1D: 1D array of imaginary parts of z values
    fill: if True, use contourf; if False, use contour
    kwargs: additional keyword arguments to pass to contour or contourf
    """
    z_real, z_imag = np.meshgrid(z_real_1D, z_imag_1D)
    z_complex = z_real + 1j * z_imag

    amplification_simplified = amplification.subs(Omega * dt, z)
    evaluate_amplification = sp.lambdify(z, amplification_simplified, modules="numpy")
    amplification_values = evaluate_amplification(z_complex)

    if fill:
        ax.contourf(z_real, z_imag, amplification_values, **kwargs)
    else:
        ax.contour(z_real, z_imag, amplification_values, **kwargs)
