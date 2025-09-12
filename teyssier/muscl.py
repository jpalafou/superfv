"""
1D Euler solver written by Romain Teyssier
"""

import numpy as np

import teyssier.riemann as riemann

gamma = 1.4


def prim_to_cons(w):
    u = 0.0 * w
    # density
    u[0] = w[0]
    # momentum
    u[1] = w[0] * w[1]
    # total energy
    u[2] = 0.5 * w[0] * w[1] ** 2 + w[2] / (gamma - 1)
    return u


def prim_to_flux(w):
    f = 0.0 * w
    # mass flux
    f[0] = w[0] * w[1]
    # momentum flux
    f[1] = w[0] * w[1] ** 2 + w[2]
    # total energy flux
    f[2] = (0.5 * w[0] * w[1] ** 2 + gamma * w[2] / (gamma - 1)) * w[1]
    return f


def cons_to_prim(u):
    w = 0.0 * u
    # density
    w[0] = u[0]
    # velocity
    w[1] = u[1] / u[0]
    # pressure
    w[2] = (gamma - 1) * (u[2] - 0.5 * w[0] * w[1] ** 2)
    return w


def set_ic(x, ic_type="sod test"):
    n = x.size
    d = np.zeros(n)
    v = np.zeros(n)
    p = np.zeros(n)
    if ic_type == "sod test":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = 0
                p[i] = 1
            else:
                d[i] = 0.125
                v[i] = 0
                p[i] = 0.1
    elif ic_type == "toro test1":
        for i in range(0, n):
            if x[i] < 0.3:
                d[i] = 1
                v[i] = 0.75
                p[i] = 1
            else:
                d[i] = 0.125
                v[i] = 0
                p[i] = 0.1
    elif ic_type == "toro test2":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = -2
                p[i] = 0.4
            else:
                d[i] = 1
                v[i] = 2
                p[i] = 0.4
    elif ic_type == "toro test3":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = 0
                p[i] = 1000
            else:
                d[i] = 1
                v[i] = 0
                p[i] = 0.01
    elif ic_type == "shu osher":
        for i in range(0, n):
            if x[i] < 0.125:
                d[i] = 3.857143
                v[i] = 2.629369
                p[i] = 10.33333
            else:
                d[i] = 1 + 0.2 * np.sin(2 * np.pi * 8 * x[i])
                v[i] = 0
                p[i] = 1
    elif ic_type == "sedov":
        d[...] = 1.0
        p[...] = 1e-5
        p[0] = 0.5 * (gamma - 1) * 1.0 / (1 / n)
    elif ic_type == "sinus":
        rho_max = 2.0
        rho_min = 1.0
        p0 = 1.0

        d[...] = (rho_max - rho_min) * (0.5 * np.sin(2 * np.pi * x) + 0.5) + rho_min
        v[...] = 1.0
        p[...] = p0
    else:
        print("Unknown IC type=", ic_type)
    # convert to conservative variables
    w = np.reshape([d, v, p], (3, n))
    u = prim_to_cons(w)
    return u


def set_bc(u, type="periodic"):
    if type == "periodic":
        u[:, 0] = u[:, -4]
        u[:, 1] = u[:, -3]
        u[:, -1] = u[:, 3]
        u[:, -2] = u[:, 2]
    elif type == "free":
        u[:, 0] = u[:, 2]
        u[:, 1] = u[:, 2]
        u[:, -1] = u[:, -3]
        u[:, -2] = u[:, -3]
    elif type == "wall":
        u[:, 0] = u[:, 3]
        u[:, 1] = u[:, 2]
        u[:, -1] = u[:, -4]
        u[:, -2] = u[:, -3]
        u[1, 0] = -u[1, 3]
        u[1, 1] = -u[1, 2]
        u[1, -1] = -u[1, -4]
        u[1, -2] = -u[1, -3]
    elif type == "sedov":
        # reflective left side
        u[:, 0] = u[:, 3]
        u[:, 1] = u[:, 2]
        u[1, :2] *= -1
        # outflow right side
        u[:, -1] = u[:, -3]
        u[:, -2] = u[:, -3]
    else:
        print("Unknown BC type")


def compute_slope(u, alpha, type="minmod"):

    du = np.zeros([u.size - 2])

    if type == "minmod":
        dlft = u[:, 1:-1] - u[:, :-2]
        drgt = u[:, 2:] - u[:, 1:-1]
        dsgn = np.sign(dlft)
        dslp = dsgn * np.minimum(abs(dlft), abs(drgt))
        du = np.where(dlft * drgt <= 0, 0, dslp)

    elif type == "moncen":
        dlft = u[:, 1:-1] - u[:, :-2]
        drgt = u[:, 2:] - u[:, 1:-1]
        dcen = 0.5 * (dlft + drgt)
        dsgn = np.sign(dcen)
        dslp = dsgn * np.minimum(np.minimum(abs(2 * dlft), abs(2 * drgt)), abs(dcen))
        du = np.where(dlft * drgt <= 0, 0, dslp)

    elif type == "moncen2":
        alft = alpha[:, :-2]
        argt = alpha[:, 2:]
        acen = alpha[:, 1:-1]
        aslp = np.minimum(np.minimum(alft, argt), acen)
        dlft = u[:, 1:-1] - u[:, :-2]
        drgt = u[:, 2:] - u[:, 1:-1]
        dcen = 0.5 * (dlft + drgt)
        dsgn = np.sign(dcen)
        dslp = dsgn * np.minimum(np.minimum(abs(2 * dlft), abs(2 * drgt)), abs(dcen))
        du1 = np.where(dlft * drgt <= 0, 0, dslp)
        du = np.where(aslp < 1, du1, dcen)

    elif type == "nolim":
        du = 0.5 * (u[:, 2:] - u[:, :-2])
    else:  # first order godunov scheme
        du = 0.0 * (u[:, 2:] - u[:, :-2])

    return du


def smooth_extrema(u, bc_type):

    # compute central first derivative
    du = 0.5 * (u[:, 2:] - u[:, :-2])
    uprime = np.pad(du, [(0, 0), (1, 1)])
    set_bc(uprime, bc_type)

    # compute left, right and central second derivative
    dlft = uprime[:, 1:-1] - uprime[:, :-2]
    drgt = uprime[:, 2:] - uprime[:, 1:-1]
    dmid = 0.5 * (dlft + drgt)

    # detect discontinuity on the left (alpha_left<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * dlft, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * dlft, 0) / dmid)
    alfl = np.where(dmid > 0, alfp, alfm)
    alfl = np.where(dmid == 0, 1, alfl)

    # detect discontinuity on the right (alpha_right<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * drgt, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * drgt, 0) / dmid)
    alfr = np.where(dmid > 0, alfp, alfm)
    alfr = np.where(dmid == 0, 1, alfr)

    # finalize smooth extrema marker (alpha=1)
    alf = np.minimum(alfl, alfr)
    alpha = np.pad(alf, [(0, 0), (1, 1)])
    set_bc(alpha, bc_type)

    return alpha


def muscl(
    tend=1,
    n=100,
    ic_type="sod test",
    bc_type="periodic",
    riemann_solver="llf",
    slp_type="moncen",
):

    # set run parameters
    h = 1 / n
    nitermax = 100000
    print("cell=", n, " itermax=", nitermax)

    # set grid geometry
    xf = np.linspace(0, 1, n + 1)
    x = 0.5 * (xf[1:] + xf[:-1])

    # allocate permanent storage
    u = np.zeros([nitermax + 1, 3, n])

    # set initial conditions
    u[0] = set_ic(x, ic_type=ic_type)

    # allocate temporary workspace
    uold = np.zeros([3, n + 4])

    # init time and iteration counter
    t = 0
    niter = 1

    # main time loop
    while t < tend and niter <= nitermax:

        uold[:, 2:-2] = u[niter - 1]  # copy old solution
        set_bc(uold, bc_type)  # set boundary conditions

        wold = cons_to_prim(uold)  # convert to primitive variables
        cold = np.sqrt(gamma * wold[2] / wold[0])  # compute sound speed
        dt = 0.8 * h / max(abs(wold[1]) + cold)  # compute time step

        alpha = smooth_extrema(wold, bc_type)  # compute smooth extrema detector
        dw = compute_slope(wold, alpha, slp_type)  # compute TVD slopes

        wcen = wold[:, 1:-1]
        ds = 0.0 * dw

        ds[0] = (
            -wcen[1] * dw[0] - wcen[0] * dw[1]
        )  # compute predictor source term for density
        ds[1] = (
            -wcen[1] * dw[1] - 1 / wcen[0] * dw[2]
        )  # compute predictor source term for velocity
        ds[2] = (
            -wcen[1] * dw[2] - gamma * wcen[2] * dw[1]
        )  # compute predictor source term for pressure

        wplus = wcen - dw / 2 + ds * dt / 2 / h  # extrapolate to left interface
        wminus = wcen + dw / 2 + ds * dt / 2 / h  # extrapolate to right interface

        wleft = wminus[:, :-1]  # left state for riemann problem
        wright = wplus[:, 1:]  # right state for riemann problem

        if riemann_solver == "llf":
            flux = riemann.llf(wleft, wright)

        if riemann_solver == "hll":
            flux = riemann.hll(wleft, wright)

        if riemann_solver == "hllc":
            flux = riemann.hllc(wleft, wright)

        if riemann_solver == "exact":
            flux = riemann.exact(wleft, wright)

        uold[:, 2:-2] = uold[:, 2:-2] - dt / h * (
            flux[:, 1:] - flux[:, :-1]
        )  # update solution
        u[niter] = uold[:, 2:-2]  # store new solution
        t = t + dt  # update time
        #        print(niter,t,dt)
        niter = niter + 1  # update iteration counter

    print("Done ", niter - 1, t)
    return u[0:niter]
