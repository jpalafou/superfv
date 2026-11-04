# mypy: ignore-errors

import numpy as np
import sympy as sp

from superfv.tools.stability_analysis.fv_diffusion import (
    derive_fv_diffusion_Omega_h_squared_over_nu_trace_func,
)
from superfv.tools.stability_analysis.sd_advection import (
    compute_sd_upwind_Omega_h_over_a_eigenvalues,
)

from .fv_advection import derive_fv_upwind_Omega_h_over_a_trace_func
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


def compute_fv_rk_amplification(
    p, k_values, *, cfl_value=None, alpha_value=None, rk="ssprk3", operator="advection"
):
    if operator == "advection":
        if cfl_value is None:
            raise ValueError("Specify cfl_value for advection operator with FV schemes.")
        omega_h_over_a_func = derive_fv_upwind_Omega_h_over_a_trace_func(p)
        omega_dt_values = omega_h_over_a_func(k_values) * cfl_value
    elif operator == "diffusion":
        if alpha_value is None:
            raise ValueError("Specify alpha_value for diffusion operator with FV schemes.")
        omega_h_squared_over_nu_func = derive_fv_diffusion_Omega_h_squared_over_nu_trace_func(p)
        omega_h_squared_over_nu_values = omega_h_squared_over_nu_func(k_values)
        # dt = alpha * h^2 / nu -> alpha = dt * nu / h^2
        omega_dt_values = omega_h_squared_over_nu_values * alpha_value
    else:
        raise ValueError(f"Unknown operator: {operator}")

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


def compute_sd_rk_amplification(p, k_values, *, cfl_value=None, rk="ssprk3", operator="advection"):
    if operator == "advection":
        if cfl_value is None:
            raise ValueError("cfl_value must be provided for advection operator with SD schemes.")
        omega_h_over_a_eigen = compute_sd_upwind_Omega_h_over_a_eigenvalues(k_values, p)
        omega_dt_values = omega_h_over_a_eigen * cfl_value
    elif operator == "diffusion":
        raise NotImplementedError("Diffusion operator is not implemented for SD schemes.")
    else:
        raise ValueError(f"Unknown operator: {operator}")

    match rk:
        case "forward_euler":
            amplification = forward_euler_amplification
        case "ssprk2":
            amplification = ssprk2_amplification
        case "ssprk3":
            amplification = ssprk3_amplification
        case "rk4":
            amplification = rk4_amplification
        case _:
            raise ValueError(f"Unknown Runge-Kutta method: {rk}")

    amplification = amplification.subs(Omega * dt, Omega_dt)
    amplification_func = sp.lambdify(Omega_dt, amplification, modules="numpy")
    amplification_values = amplification_func(omega_dt_values.flatten())
    return np.max(np.abs(amplification_values))


def plot_stability_region(ax, amplification, z_real_1D, z_imag_1D, fill=False, **kwargs):
    z_real, z_imag = np.meshgrid(z_real_1D, z_imag_1D)
    z_complex = z_real + 1j * z_imag

    amplification_simplified = amplification.subs(Omega * dt, z)
    evaluate_amplification = sp.lambdify(z, amplification_simplified, modules="numpy")
    amplification_values = evaluate_amplification(z_complex)

    if fill:
        ax.contourf(z_real, z_imag, amplification_values, **kwargs)
    else:
        ax.contour(z_real, z_imag, amplification_values, **kwargs)
