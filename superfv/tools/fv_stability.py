# mypy: ignore-errors

import numpy as np
import sympy as sp

from superfv.tools.polynomials import (
    compute_stencil,
    get_coeff_from_U_i,
    get_index_from_U_i,
    h,
    shift_stencil_indices,
)

a, x_i = sp.symbols("a x_i", real=True)
k = sp.symbols("k", real=False)


def inject_harmonic_modes(stencil):
    """
    stencil: expression representing the stencil in terms of U_i
    returns: the harmonic mode representation of the stencil,
        where U_i are replaced by e^{x_i + k h}
    """

    terms = stencil.as_ordered_terms()

    def inject_term(term):
        index = get_index_from_U_i(term)
        return sp.exp(k * 1j * (x_i + index * h)) * get_coeff_from_U_i(term)

    return sum(inject_term(term) for term in terms)


def derive_fv_upwind_rhs(p):
    """
    p: the polynomial representing the finite volume scheme
    returns: the right-hand side of the upwind finite volume scheme
    """
    stencil_r = compute_stencil(p, h / 2)
    stencil_l = shift_stencil_indices(stencil_r, -1)

    upwind_rhs = sp.expand(-(a / h) * (stencil_r - stencil_l))
    return upwind_rhs


def derive_fv_upwind_omega_h_over_a(p):
    """
    p: the polynomial representing the finite volume scheme
    returns: the harmonic mode representation of the upwind finite volume scheme's omega * h / a
    """
    upwind_rhs = derive_fv_upwind_rhs(p)
    upwind_harmonic_rhs = inject_harmonic_modes(upwind_rhs)

    z = sp.exp(k * 1j * x_i)
    omega = sp.simplify(upwind_harmonic_rhs / z)
    omega_h_over_a = sp.simplify(omega * h / a)
    omega_h_over_a = omega_h_over_a.rewrite(sp.sin)
    return omega_h_over_a


def compute_fv_upwind_omega_h_over_a_harmonic_trace(p, n=100):
    """
    p: the polynomial representing the finite volume scheme
    kh: the values of k*h for which to compute the harmonic trace as an array of shape (n,)
    returns: the harmonic mode representation of the upwind finite volume scheme's omega * h / a
        evaluated over a trigonometric periodic grid as an array of shape (n,)
    """
    omega_h_over_a = derive_fv_upwind_omega_h_over_a(p)

    def get_complex_numeric(kh_expression, kh_value):
        x = kh_expression.subs(h, 1).subs(k, kh_value)
        return float(sp.re(x)) + 1j * float(sp.im(x))

    kh = np.linspace(-np.pi, np.pi, n)
    omega_h_over_a_values = np.array([get_complex_numeric(omega_h_over_a, x) for x in kh])
    return omega_h_over_a_values


def plot_fv_upwind_omega_h_over_a_harmonic_trace(ax, p, n=100, **kwargs):
    """
    plots the harmonic mode representation of the upwind finite volume scheme's omega * h / a
    with Re(omega * h / a) on the x-axis and Im(omega * h / a) on the y-axis

    ax: the matplotlib axes object on which to plot
    p: the polynomial representing the finite volume scheme
    n: the number of points to use for the harmonic trace
    **kwargs: additional keyword arguments for the plot function
    """
    trace = compute_fv_upwind_omega_h_over_a_harmonic_trace(p, n)
    ax.plot(trace.real, trace.imag, **kwargs)
