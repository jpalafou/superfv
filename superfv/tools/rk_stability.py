# mypy: ignore-errors

import numpy as np
import sympy as sp

from superfv.tools.fv_stability import a
from superfv.tools.polynomials import h

dt, cfl = sp.symbols("dt CFL", real=True)
u0, Omega = sp.symbols("u^0 Omega", real=False)


def replace_dt_with_cfl(x):
    return x.subs(dt, cfl * h / a)


def replace_dt_with_cfl_wrapper(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return replace_dt_with_cfl(result)

    return wrapper


@replace_dt_with_cfl_wrapper
def derive_forward_euler_amplification():
    u1 = u0 + dt * Omega * u0
    return sp.simplify(sp.Abs(u1 / u0))


@replace_dt_with_cfl_wrapper
def derive_ssprk2_amplification():
    k0 = Omega * u0
    u1 = u0 + dt * k0
    k1 = Omega * u1
    u2 = sp.Rational(1, 2) * (u0 + u1 + dt * k1)
    return sp.simplify(sp.Abs(u2 / u0))


@replace_dt_with_cfl_wrapper
def derive_ssprk3_amplification():
    k0 = Omega * u0
    u1 = u0 + dt * k0
    k1 = Omega * u1
    u2 = u0 + sp.Rational(1, 4) * dt * (k0 + k1)
    k2 = Omega * u2
    u3 = u0 + sp.Rational(1, 6) * dt * (k0 + k1 + 4 * k2)
    return sp.simplify(sp.Abs(u3 / u0))


@replace_dt_with_cfl_wrapper
def derive_rk4_amplification():
    k0 = Omega * u0
    k1 = Omega * (u0 + sp.Rational(1, 2) * dt * k0)
    k2 = Omega * (u0 + sp.Rational(1, 2) * dt * k1)
    k3 = Omega * (u0 + dt * k2)
    u1 = u0 + sp.Rational(1, 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
    return sp.simplify(sp.Abs(u1 / u0))


def plot_stability_region(ax, amplification, z_real_1D, z_imag_1D, real_cfl, fill=False, **kwargs):
    z_real, z_imag = np.meshgrid(z_real_1D, z_imag_1D)
    z = z_real + 1j * z_imag

    zsym = sp.symbols("z", complex=True)
    amplification_simplified = sp.simplify(
        amplification.subs(Omega * h / a, zsym).subs(cfl, real_cfl)
    )
    evaluate_amplification = sp.lambdify(zsym, amplification_simplified, modules="numpy")
    amplification_values = evaluate_amplification(z)

    if fill:
        ax.contourf(z_real, z_imag, amplification_values, **kwargs)
    else:
        ax.contour(z_real, z_imag, amplification_values, **kwargs)
