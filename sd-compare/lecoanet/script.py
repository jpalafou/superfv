import h5py
import numpy as np


def project_dedalus_to_uniform_cell_averages(
    filename: str,
    nx: int,
    ny: int,
) -> np.ndarray:
    with h5py.File(filename, "r") as f:
        x = np.asarray(f["scales"]["x"]).squeeze()
        y = np.asarray(f["scales"]["z"]).squeeze()
        c = np.asarray(f["tasks"]["c"])

    c = c[0]
    if c.shape != (len(x), len(y)):
        raise ValueError(f"Expected c.shape == {(len(x), len(y))}, got {c.shape}.")

    return _grid_values_to_uniform_cell_averages(c, nx, ny)


def _grid_values_to_uniform_cell_averages(
    values: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    nxf, nyf = values.shape
    coeff = np.fft.fft2(values) / (nxf * nyf)

    kx = np.fft.fftfreq(nxf) * nxf
    ky = np.fft.fftfreq(nyf) * nyf

    # Average each Fourier mode over a target FV cell and shift to cell centers.
    coeff *= (
        np.sinc(kx / nx)[:, None]
        * np.sinc(ky / ny)[None, :]
        * np.exp(1j * np.pi * kx / nx)[:, None]
        * np.exp(1j * np.pi * ky / ny)[None, :]
    )

    coarse_coeff = np.zeros((nx, ny), dtype=np.complex128)
    np.add.at(
        coarse_coeff,
        (
            np.mod(kx.astype(int), nx)[:, None],
            np.mod(ky.astype(int), ny)[None, :],
        ),
        coeff,
    )

    return np.fft.ifft2(coarse_coeff * (nx * ny)).real
