# mypy: ignore-errors

from functools import lru_cache

import sympy as sp

from .fv_advection import inject_harmonic_modes
from .fv_polynomials import derive_conservative_interpolation_polynomial
from .variables import h, k, nu, theta, x, x_i


def derive_interpolation_of_derivative_at_face(p, face):
    """
    Return the
    """
    n = (p + 2) // 2
    poly = derive_conservative_interpolation_polynomial(n, n, derivative_order=2)
    if face == "left":
        return sp.simplify(poly.subs(x, -h / 2) / h)
    if face == "right":
        return sp.simplify(poly.subs(x, h / 2) / h)
    raise ValueError("Invalid face specified. Use 'left' or 'right'.")


def derive_fv_diffusion_rhs(p):
    left_dudx = derive_interpolation_of_derivative_at_face(p, "left")
    right_dudx = derive_interpolation_of_derivative_at_face(p, "right")

    return sp.expand((nu / h) * (right_dudx - left_dudx))


def derive_fv_diffusion_harmonic_rhs(p):
    rhs = derive_fv_diffusion_rhs(p)
    return sp.simplify(inject_harmonic_modes(rhs))


def derive_fv_diffusion_Omega(p):
    rhs = derive_fv_diffusion_harmonic_rhs(p)
    z = sp.exp(k * sp.I * x_i)
    Omega = sp.simplify(rhs / z)
    Omega = sp.simplify(sp.expand_complex(Omega))
    Omega = Omega.subs(k * h, theta)

    # Check that theta appears only linearly inside trigonometric functions
    trig_terms = Omega.atoms(sp.sin, sp.cos)
    assert not Omega.xreplace({term: sp.Dummy() for term in trig_terms}).has(theta)
    assert all(sp.diff(term.args[0], theta, 2) == 0 for term in trig_terms)

    return Omega


def derive_fv_diffusion_Omega_h_squared_over_nu(p):
    return derive_fv_diffusion_Omega(p) * h**2 / nu


@lru_cache(maxsize=None)
def derive_fv_diffusion_Omega_h_squared_over_nu_trace_func(p):
    """
    p: polynomial degree
    returns: a function that takes an array of theta values and returns the harmonic mode
        representation of the diffusion finite volume scheme's Omega evaluated at those
        theta values
    """
    Omega_h_squared_over_nu = derive_fv_diffusion_Omega_h_squared_over_nu(p)
    out = sp.lambdify(theta, Omega_h_squared_over_nu, modules="numpy")
    return out
