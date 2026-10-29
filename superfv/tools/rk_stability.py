# mypy: ignore-errors

import numpy as np
import sympy as sp

from superfv.tools.fv_advection_stability import a, derive_fv_upwind_Omega_h_over_a
from superfv.tools.polynomials import h

dt, cfl = sp.symbols("dt CFL", real=True)
u0, Omega, z = sp.symbols("u^0 Omega z", complex=True)


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


@module_variable
def forward_euler_update():
    return sp.simplify(u0 + dt * Omega * u0)


@module_variable
def forward_euler_amplification():
    return get_amplification_factor(forward_euler_update)


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


def compute_fv_upwinding_rk_amplification(p, rk="ssprk3", sub_in_cfl=True):
    omega_h_over_a = derive_fv_upwind_Omega_h_over_a(p)
    omega = omega_h_over_a * a / h

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

    out = amplification.subs(Omega, omega)
    if sub_in_cfl:
        out = out.subs(dt * a / h, cfl)
    return out


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
