# Dedalus To FV Cell Averages

The Zenodo Dedalus reference files store grid values, not raw Fourier
coefficients. For the dye field,

```python
c = np.asarray(f["tasks"]["c"])[0]
```

is a periodic grid representation of the Dedalus spectral solution. To compare
against finite-volume data, we project this periodic interpolant to uniform cell
averages.

Let the target FV grid have `nx` by `ny` cells on the full periodic domain. We
first compute Fourier coefficients from the Dedalus grid values:

```text
c_hat[kx, ky] = fft2(c)[kx, ky] / (nxf * nyf)
```

For one Fourier mode `exp(2 pi i k x)`, the average over a target cell of width
`dx = 1 / nx` is the point value at the cell center multiplied by

```text
sinc(k / nx)
```

where NumPy's `sinc(a)` means `sin(pi a) / (pi a)`. In 2D, use
`sinc(kx / nx) * sinc(ky / ny)`.

The factor

```text
exp(i pi kx / nx) * exp(i pi ky / ny)
```

shifts the Fourier mode from the origin to FV cell centers
`((i + 1/2) / nx, (j + 1/2) / ny)`.

After applying this cell-average filter and phase shift, modes are folded onto
the target Fourier grid modulo `(nx, ny)`. The inverse FFT then gives the
uniform FV cell averages.

This differs slightly from Lecoanet et al.'s Appendix A procedure, which uses
spectral interpolation to a common grid while treating Athena data as
cell-centered. Here we add the exact cell-average filter because SuperFV and
the regularized SPD output are finite-volume quantities.
