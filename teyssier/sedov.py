import numpy as np


def sedovana(gamma: float = 1.4, dim: int = 1, n1: int = 1000, n2: int = 1000):
    """
    purpose:
        This procedure computes the analytical solution for the Sedov
        blast wave in 1, 2 or 3 dimensions (planar, cylindrical and
        spherical case).

        This function returns 4 arrays of shape (n1 + n2 + 2,). The
        output variables are the following dimensionless quantities:
        r (position from the point like explosion, d (density), u
        (velocity) and p (pressure). To recover the true value, you
        have to rescale these dimensionless values to the true values,
        defining first the total energy E_0, the initial mass density
        rho_0 and the time t you consider and finally computing the
        true values using  the following scaling laws:

        r = r * (E_0/rho_0)^(1./(dim+2.)) * t^(2./(dim+2.))
        d = d * rho_0
        u = u * (E_0/rho_0)^(1./(dim+2.)) * t^(-dim/(dim+2.))
        p = p * (E_0/rho_0)^(2./(dim+2.)) * t^(-2.*dim/(dim+2.)) * rho_0

    args:
        gamma (float) : adiabatic exponent of the fluid (default: 1.4)
        dim (int) : dimensionality of the problem (default: 1)
        n1 (int) : number of points in the first region (default: 1000)
        n2 (int) : number of points in the second region (default: 1000)

    returns:
        r (array) : position from the point like explosion, shape (n1+n2+2,)
        d (array) : density, shape (n1+n2+2,)
        u (array) : velocity, shape (n1+n2+2,)
        p (array) : pressure, shape (n1+n2+2,)

    example:
        Compute the analytical solution of the planar Sedov blast
        wave for a gamma=1.6667 fluid

            r, d, u, p = sedovana(gamma=1.6667, dim=1)

    MODIFICATION HISTORY:
        Written by:     Romain Teyssier, 01/01/2000.
                        e-mail: Romain.Teyssier@cea.fr
        Fevrier, 2001:  Comments and header added by Romain Teyssier.
        Juillet, 2024:  Translated to Python by Jonathan Palafoutas.
    """
    g = gamma
    n = dim

    vmax = 4 / (n + 2) / (g + 1)
    vmin = 2 / (n + 2) / g
    v = vmin + np.power(10, -10 * (1 - (np.arange(n1, dtype=np.float64) + 1) / n1)) * (
        vmax - vmin
    )
    a2 = (1 - g) / (2 * (g - 1) + n)
    a1 = (n + 2) * g / (2 + n * (g - 1)) * (2 * n * (2 - g) / g / (n + 2) ** 2 - a2)
    a3 = n / (2 * (g - 1) + n)
    a4 = a1 * (n + 2) / (2 - g)
    a5 = 2 / (g - 2)

    r1 = (
        ((n + 2) * (g + 1) / 4 * v) ** (-2 / (2 + n))
        * ((g + 1) / (g - 1) * ((n + 2) * g / 2 * v - 1)) ** (-a2)
        * (
            (n + 2)
            * (g + 1)
            / ((n + 2) * (g + 1) - 2 * (2 + n * (g - 1)))
            * (1 - (2 + n * (g - 1)) / 2 * v)
        )
        ** (-a1)
    )

    u1 = (n + 2) * (g + 1) / 4 * v * r1

    d1 = (
        ((g + 1) / (g - 1) * ((n + 2) * g / 2 * v - 1)) ** a3
        * ((g + 1) / (g - 1) * (1 - (n + 2) / 2 * v)) ** a5
        * (
            (n + 2)
            * (g + 1)
            / ((n + 2) * (g + 1) - 2 * (2 + n * (g - 1)))
            * (1 - (2 + n * (g - 1)) / 2 * v)
        )
        ** a4
    )
    p1 = (
        ((n + 2) * (g + 1) / 4 * v) ** (2 * n / (2 + n))
        * ((g + 1) / (g - 1) * (1 - (n + 2) / 2 * v)) ** (a5 + 1)
        * (
            (n + 2)
            * (g + 1)
            / ((n + 2) * (g + 1) - 2 * (2 + n * (g - 1)))
            * (1 - (2 + n * (g - 1)) / 2 * v)
        )
        ** (a4 - 2 * a1)
    )

    r2 = r1[0] * (np.arange(n2, dtype=np.float32) + 0.5) / n2
    u2 = u1[0] * r2 / r1[0]
    d2 = d1[0] * (r2 / r1[0]) ** (n / (g - 1))
    p2 = p1[0] * np.ones_like(r2)

    length = r1.size + r2.size + 2
    region1 = slice(0, r2.size)
    region2 = slice(r2.size, r1.size + r2.size)
    region3 = r1.size + r2.size
    region4 = r1.size + r2.size + 1

    r = np.empty((length,))
    r[region1] = r2
    r[region2] = r1
    r[region3] = r1.max()
    r[region4] = r1.max() + 1000
    d = np.empty((length,))
    d[region1] = d2
    d[region2] = d1
    d[region3] = 1 / ((g + 1) / (g - 1))
    d[region4] = 1 / ((g + 1) / (g - 1))
    u = np.empty((length,))
    u[region1] = u2
    u[region2] = u1
    u[region3] = 0
    u[region4] = 0
    p = np.empty((length,))
    p[region1] = p2
    p[region2] = p1
    p[region3] = 0
    p[region4] = 0

    d = d * (g + 1) / (g - 1)
    u = u * 4 / (n + 2) / (g + 1)
    p = p * 8 / (n + 2) ** 2 / (g + 1)

    nn = r.size
    vol = np.empty((nn,))
    for i in range(1, nn):
        vol[i] = r[i] ** n - r[i - 1] ** n
    vol[0] = r[0] ** n
    if n == 1:
        const = 2
    elif n == 2:
        const = np.pi
    elif n == 3:
        const = 4 * np.pi / 3
    else:
        raise ValueError("Invalid dimension")

    vol = vol * const
    int1 = (d * u * u / 2) * vol
    int2 = p / (g - 1) * vol
    sum1 = np.sum(int1)
    sum2 = np.sum(int2)
    sum1_sum2 = sum1 + sum2
    chi0 = sum1_sum2 ** (-1 / (2 + n))
    print(f"{chi0=}")
    r = r * chi0
    u = u * chi0
    p = p * chi0**2

    return r, d, u, p
