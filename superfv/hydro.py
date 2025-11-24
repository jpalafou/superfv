from types import ModuleType
from typing import Literal, Tuple

import numpy as np

from .mesh import UniformFVMesh
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import VariableIndexMap


def sound_speed(
    xp: ModuleType, idx: VariableIndexMap, w: ArrayLike, gamma: float
) -> ArrayLike:
    """
    Compute the sound speed from primitive variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        cs: Sound speed array. Has shape (1, nx, ny, nz, ...).
    """
    rho = w[idx("rho", keepdims=True)]
    P = w[idx("P", keepdims=True)]
    cs2 = gamma * P / rho
    cs = xp.sqrt(xp.maximum(cs2, 0.0))
    return cs


def prim_to_cons(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    gamma: float,
) -> ArrayLike:
    """
    Convert primitive variables to conservative variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        u: Array with conservative variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = w[idx("rho")]
    vx = w[idx("vx")]
    vy = w[idx("vy")]
    vz = w[idx("vz")]
    P = w[idx("P")]

    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    u = xp.empty_like(w)

    u[idx("rho")] = rho
    u[idx("mx")] = rho * vx
    u[idx("my")] = rho * vy
    u[idx("mz")] = rho * vz
    u[idx("E")] = K + P / (gamma - 1)

    if "passives" in idx:
        u[idx("passives")] = rho * w[idx("passives")]

    return u


def cons_to_prim(
    xp: ModuleType,
    idx: VariableIndexMap,
    u: ArrayLike,
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
) -> ArrayLike:
    """
    Convert conservative variables to primitive variables.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        u: Array of conservative variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.
        isothermal: Whether the equation of state is isothermal.
        iso_cs: Isothermal sound speed (used if isothermal is True).

    Returns:
        w: Array with primitive variables. Has shape (nvars, nx, ny, nz, ...).
    """
    rho = u[idx("rho")]
    mx = u[idx("mx")]
    my = u[idx("my")]
    mz = u[idx("mz")]
    E = u[idx("E")]

    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    K = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    w = xp.empty_like(u)

    w[idx("rho")] = rho
    w[idx("vx")] = vx
    w[idx("vy")] = vy
    w[idx("vz")] = vz
    w[idx("P")] = rho * iso_cs**2 if isothermal else (gamma - 1) * (E - K)

    if "passives" in idx:
        w[idx("passives")] = u[idx("passives")] / rho

    return w


def fluxes(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
) -> ArrayLike:
    """
    Compute the fluxes for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        gamma: Adiabatic index.

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    d1 = dim
    d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    rho = w[idx("rho")]
    v1 = w[idx("v" + d1)]
    v2 = w[idx("v" + d2)]
    v3 = w[idx("v" + d3)]
    P = w[idx("P")]

    K = 0.5 * rho * (v1**2 + v2**2 + v3**2)

    F = xp.empty_like(w)

    F[idx("rho")] = rho * v1
    F[idx("m" + d1)] = rho * v1**2 + P
    F[idx("m" + d2)] = rho * v1 * v2
    F[idx("m" + d3)] = rho * v1 * v3
    F[idx("E")] = (K + P / (gamma - 1) + P) * v1

    if "passives" in idx:
        F[idx("passives")] = rho * v1 * w[idx("passives")]

    return F


def turbulent_power_specta(
    xp: ModuleType,
    idx: VariableIndexMap,
    w: ArrayLike,
    mesh: UniformFVMesh,
    nbins: int,
    binmode: Literal["linear", "log"] = "linear",
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the turbulent kinetic energy power spectrum from the velocity field.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        mesh: UniformFVMesh object defining the computational mesh.
        nbins: Number of bins for the power spectrum.
        binmode: Bin spacing mode, either "linear" or "log".

    Returns:
        k_centers: Array of shape (nbins,) with the center of each wavenumber bin.
        E_k: Array of shape (nbins,) with the energy in each wavenumber bin.
    """
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.hx, mesh.hy, mesh.hz
    active_dims = mesh.active_dims

    N = nx * ny * nz
    active_axes = tuple({"x": 0, "y": 1, "z": 2}[d] for d in active_dims)

    vx = w[idx("vx")] - w[idx("vx")].mean()
    vy = w[idx("vy")] - w[idx("vy")].mean()
    vz = w[idx("vz")] - w[idx("vz")].mean()

    Vx = xp.fft.fftn(vx, axes=active_axes, norm="ortho")
    Vy = xp.fft.fftn(vy, axes=active_axes, norm="ortho")
    Vz = xp.fft.fftn(vz, axes=active_axes, norm="ortho")

    # Per-mode kinetic energy (per *sum* convention). We'll rescale for "mean" below.
    P = 0.5 * (xp.abs(Vx) ** 2 + xp.abs(Vy) ** 2 + xp.abs(Vz) ** 2)
    P = P / N  # rescale to get mean(|v|^2) = sum P

    # Angular wavenumbers (rad/length)
    kx = (
        2.0 * np.pi * xp.fft.fftfreq(nx, d=dx)
        if "x" in active_dims
        else xp.array([0.0])
    )
    ky = (
        2.0 * np.pi * xp.fft.fftfreq(ny, d=dy)
        if "y" in active_dims
        else xp.array([0.0])
    )
    kz = (
        2.0 * np.pi * xp.fft.fftfreq(nz, d=dz)
        if "z" in active_dims
        else xp.array([0.0])
    )
    KX, KY, KZ = xp.meshgrid(kx, ky, kz, indexing="ij")
    Kmag = xp.sqrt(KX**2 + KY**2 + KZ**2)

    k = Kmag.ravel()
    p = P.ravel()

    mask = k > 0
    k = k[mask]
    p = p[mask]

    if binmode == "linear":
        edges = xp.linspace(k.min(), k.max(), nbins + 1)
        k_centers = 0.5 * (edges[:-1] + edges[1:])
    elif binmode == "log":
        edges = xp.logspace(xp.log10(k.min()), xp.log10(k.max()), nbins + 1)
        k_centers = xp.sqrt(edges[:-1] * edges[1:])
    else:
        raise ValueError("binmode must be 'linear' or 'log'")

    bin_idx = xp.digitize(k, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[valid]
    p = p[valid]

    E_shell = xp.bincount(bin_idx, weights=p, minlength=nbins)
    widths = edges[1:] - edges[:-1]
    E_k = E_shell / widths

    return k_centers, E_k


if CUPY_AVAILABLE:
    import cupy as cp

    sound_speed_cp = cp.ElementwiseKernel(
        in_params="float64 rho, float64 P, float64 gamma",
        out_params="float64 cs",
        operation="""
            double val = gamma * P / rho;
            cs = sqrt(fmax(val, 0.0));
        """,
        name="sound_speed_ew",
    )
