from typing import Literal, Tuple

import numpy as np

from superfv.mesh import UniformFiniteVolumeMesh

from .variable_index_map import VariableIndexMap


def turbulent_power_specta(
    idx: VariableIndexMap,
    w: np.ndarray,
    mesh: UniformFiniteVolumeMesh,
    nbins: int,
    binmode: Literal["linear", "log"] = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the turbulent kinetic energy power spectrum from the velocity field.

    Args:
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        mesh: UniformFiniteVolumeMesh object defining the computational mesh.
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

    Vx = np.fft.fftn(vx, axes=active_axes, norm="ortho")
    Vy = np.fft.fftn(vy, axes=active_axes, norm="ortho")
    Vz = np.fft.fftn(vz, axes=active_axes, norm="ortho")

    # Per-mode kinetic energy (per *sum* convention). We'll rescale for "mean" below.
    P = 0.5 * (np.abs(Vx) ** 2 + np.abs(Vy) ** 2 + np.abs(Vz) ** 2)
    P = P / N  # rescale to get mean(|v|^2) = sum P

    # Angular wavenumbers (rad/length)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx) if "x" in active_dims else np.array([0.0])
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy) if "y" in active_dims else np.array([0.0])
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz) if "z" in active_dims else np.array([0.0])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k = Kmag.ravel()
    p = P.ravel()

    mask = k > 0
    k = k[mask]
    p = p[mask]

    if binmode == "linear":
        edges = np.linspace(k.min(), k.max(), nbins + 1)
        k_centers = 0.5 * (edges[:-1] + edges[1:])
    elif binmode == "log":
        edges = np.logspace(np.log10(k.min()), np.log10(k.max()), nbins + 1)
        k_centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        raise ValueError("binmode must be 'linear' or 'log'")

    bin_idx = np.digitize(k, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    bin_idx = bin_idx[valid]
    p = p[valid]

    E_shell = np.bincount(bin_idx, weights=p, minlength=nbins)
    widths = edges[1:] - edges[:-1]
    E_k = E_shell / widths

    return k_centers, E_k
