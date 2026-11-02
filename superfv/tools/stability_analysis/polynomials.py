# mypy: ignore-errors

import sympy as sp

x, h = sp.symbols("x h", real=True)


def lagrange_interpolation_polynomial(x_nodes, f_nodes):
    """
    x_nodes: list of x-coordinates of the nodes
    f_nodes: list of function values at the nodes
    returns: the Lagrange interpolation polynomial in terms of x
    """
    if len(x_nodes) != len(f_nodes):
        raise ValueError("x_nodes and f_nodes must have the same length.")
    n = len(x_nodes)
    poly = 0
    for j in range(n):
        L_j = 1
        for k in range(n):
            if j == k:
                continue
            L_j *= (x - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
        poly += f_nodes[j] * L_j
    return poly


def cumulate_masses(averages):
    """
    averages: list of cell averages U_i
    returns: list of cumulative masses starting from 0
    """
    cum_masses = [0]
    for avg in averages:
        cum_masses.append(cum_masses[-1] + avg)
    return cum_masses


def derive_conservative_interpolation_polynomial(l, r, derivative_order=1):
    """
    l: number of finite volume cells to the left of the central cell
    r: number of finite volume cells to the right of the central cell
    derivative_order: order of the derivative to compute
    returns: the derivative of the conservative interpolation polynomial of the specified order
        in terms of x, h, and the cell averages U_i
    """
    interfaces = [-l * h - h / 2] + [i * h + h / 2 for i in range(-l, r + 1)]

    averages = sp.symbols(" ".join(f"U_{i}" for i in range(-l, r + 1)), real=True)
    if not isinstance(averages, tuple):
        averages = (averages,)

    cumulative_masses = cumulate_masses(averages)
    cumulate_mass_poly = lagrange_interpolation_polynomial(interfaces, cumulative_masses)

    u = (h**derivative_order) * sp.diff(cumulate_mass_poly, x, derivative_order)
    return u


def compute_cell_average(poly):
    """
    poly: polynomial to integrate over the central cell
    returns: expression for the cell average of the polynomial over the central cell
    """
    return sp.simplify(sp.integrate(poly, (x, -h / 2, h / 2))) / h


def degree_p_conservative_interpolation_polynomial(p):
    """
    p: degree of the conservative interpolation polynomial
    returns: the conservative interpolation polynomial of degree p
        in terms of x, h, and the cell averages U_i
    """
    if (p + 1) % 2 == 0:
        poly1 = derive_conservative_interpolation_polynomial(p // 2, (p + 1) // 2)
        poly2 = derive_conservative_interpolation_polynomial((p + 1) // 2, p // 2)
        poly = (poly1 + poly2) / 2
    else:
        poly = derive_conservative_interpolation_polynomial((p + 1) // 2, (p + 1) // 2)
    assert compute_cell_average(poly) == sp.symbols("U_0", real=True)
    return poly


def compute_stencil(p, x_eval):
    """
    p: degree of the conservative interpolation polynomial
    x_eval: point at which to evaluate the stencil in terms of h
    returns: the stencil evaluated at the specified point in terms of U_i.
        ideally the h terms cancel out.
    """
    poly = degree_p_conservative_interpolation_polynomial(p)
    return sp.simplify(poly.subs(x, x_eval))


def get_index_from_U_i(U_i):
    """
    U_i: symbol representing a cell average
    returns: the index of the cell average
    """
    numbers = ""
    for i, x in enumerate(str(U_i).split("U_")[-1]):
        if x.isdigit() or (x == "-" and i == 0):
            numbers += x
        else:
            break
    return int(numbers)


def get_coeff_from_U_i(U_i):
    """
    U_i: symbol representing a cell average
    returns: the coefficient of the cell average in the term
    """
    index = get_index_from_U_i(U_i)
    term = sp.Symbol(f"U_{index}", real=True)
    return U_i / term


def shift_stencil_indices(stencil, shift):
    """
    stencil: expression representing the stencil in terms of U_i
    shift: number of positions to shift the stencil
    returns: the shifted stencil in terms of U_{i+shift}
    """
    terms = stencil.as_ordered_terms()

    def shift_term(term, shift):
        index = get_index_from_U_i(term)
        new_index = index + shift
        return sp.Symbol(f"U_{new_index}", real=True) * get_coeff_from_U_i(term)

    return sum(shift_term(term, shift) for term in terms)
