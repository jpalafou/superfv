# mypy: ignore-errors

import sympy as sp

from superfv.tools.fv_advection_stability import inject_harmonic_modes
from superfv.tools.polynomials import derive_conservative_interpolation_polynomial, h, x

nu = sp.symbols("nu", real=True)


def derive_interpolation_of_derivative_at_face(p, face):
    """
    Return the
    """
    n = (p + 1) // 2
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
    # .rewrite(sp.sin)
    return sp.simplify(inject_harmonic_modes(rhs))
