# mypy: ignore-errors

import sympy as sp

# real variables
a = sp.symbols("a", real=True)  # advection speed
cfl = sp.symbols("CFL", real=True)  # Courant-Friedrichs-Lewy factor
dt = sp.symbols("dt", real=True)  # time step size
h = sp.symbols("h", real=True)  # cell width
k = sp.symbols("k", real=True)  # wave number
nu = sp.symbols("nu", real=True)  # diffusion coefficient
theta = sp.symbols("theta", real=True)  # theta = k * h
x = sp.symbols("x", real=True)  # spatial coordinate
x_i = sp.symbols("x_i", real=True)  # cell center coordinate

# complex variables
Omega = sp.symbols("Omega", complex=True)  # amplification factor
u0 = sp.symbols("u^0", complex=True)  # initial solution value
z = sp.symbols("z", complex=True)  # complex variable for stability region plotting
