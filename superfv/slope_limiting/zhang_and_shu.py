from __future__ import annotations

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.variable_index_map import VariableIndexMap


def compute_theta(
    w: ArrayLike,
    wj: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    Mj: ArrayLike,
    mj: ArrayLike,
    theta: ArrayLike,
    tol: float = 1e-15,
):
    """
    Compute Zhang and Shu's a priori slope limiting parameter and write it to `theta`
    while alpha updating the nodal maxima and minima, `Mj` and `mj`, respectively.
    Renders a single ghost cell layer along each active dimension of the output arrays
    invalid.

    Args:
        w: Array of cell-average values with shape (nvars, nx, ny, nz).
        wj: Array of high-order interpolation values with shape
            (nvars, nx, ny, nz, ninterps).
        M: Array of local maxima with shape (nvars, nx, ny, nz).
        m: Array of local minima with shape (nvars, nx, ny, nz).
        Mj: Array to which nodal maxima are written. Has shape (nvars, nx, ny, nz).
        mj: Array to which nodal minima are written. Has shape (nvars, nx, ny, nz).
        theta: Array to which the Zhang-Shu limiter is written. Has shape
            (nvars, nx, ny, nz).
        tol: Tolerance to avoidi 0 in the denominator of the theta calculation.
    """
    if CUPY_AVAILABLE and isinstance(w, cp.ndarray):
        compute_theta_kernel_helper(w, wj, M, m, Mj, mj, theta, tol)
        return

    # compute nodal maximum principle
    np.max(wj, axis=4, out=Mj)
    np.min(wj, axis=4, out=mj)

    # compute theta
    theta[...] = np.minimum(
        np.minimum(
            np.divide(np.abs(M - w), np.abs(Mj - w) + tol),
            np.divide(np.abs(m - w), np.abs(mj - w) + tol),
        ),
        1.0,
    )


def compute_pp_theta(
    u: ArrayLike,
    uj: ArrayLike,
    theta: ArrayLike,
    idx: VariableIndexMap,
    rho_min: float,
    P_min: float,
    gamma: float,
    eps: float = 1e-15,
):
    """
    Positivity-preserving (density and pressure) version of Zhang and Shu's a priori limiter.

    Args:
        u: Array of cell-average values with shape (nvars, nx, ny, nz).
        uj: Array of high-order interpolation values with shape
            (nvars, nx, ny, nz, ninterps).
        theta: Array to which the Zhang-Shu limiter is written. Has shape
            (nvars, nx, ny, nz).
        idx: VariableIndexMap object that maps variable names to their indices in the arrays.
        rho_min: Minimum density value to enforce.
        P_min: Minimum pressure value to enforce.
        gamma: Ratio of specific heats.
        eps: Tolerance for divisions and root admissibility checks.
    """
    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        compute_pp_theta_kernel_helper(u, uj, theta, idx, rho_min, P_min, gamma, eps)
        return

    na = np.newaxis
    irho = idx("rho")
    imx = idx("mx")
    imy = idx("my")
    imz = idx("mz")
    iE = idx("E")

    rho_jmin = np.min(uj[irho], axis=3)
    theta1 = np.where(
        rho_jmin < rho_min,
        np.minimum((u[irho] - rho_min) / (u[irho] - rho_jmin + eps), 1.0),
        1.0,
    )

    # Coefficients use the same sign convention as notebooks/ZhangShuPP.ipynb.
    rho_bar = u[irho, ..., na]
    mx_bar = u[imx, ..., na]
    my_bar = u[imy, ..., na]
    mz_bar = u[imz, ..., na]
    E_bar = u[iE, ..., na]

    rho_hat = theta1[..., na] * (uj[irho] - rho_bar) + rho_bar
    drho = rho_hat - rho_bar
    dmx = uj[imx] - mx_bar
    dmy = uj[imy] - my_bar
    dmz = uj[imz] - mz_bar
    dE = uj[iE] - E_bar

    P_qhat = (gamma - 1) * (uj[iE] - 0.5 * (uj[imx] ** 2 + uj[imy] ** 2 + uj[imz] ** 2) / rho_hat)

    momentum_diff_sq = dmx**2 + dmy**2 + dmz**2
    momentum_bar_sq = mx_bar**2 + my_bar**2 + mz_bar**2
    momentum_bar_dot_diff = mx_bar * dmx + my_bar * dmy + mz_bar * dmz

    coeff_A = (gamma - 1) * (momentum_diff_sq - 2 * dE * drho)
    coeff_B = (
        -2 * (gamma - 1) * (E_bar * drho + rho_bar * dE - momentum_bar_dot_diff) + 2 * P_min * drho
    )
    coeff_C = (gamma - 1) * (momentum_bar_sq - 2 * E_bar * rho_bar) + 2 * P_min * rho_bar

    discriminant = np.maximum(coeff_B**2 - 4 * coeff_A * coeff_C, 0.0)
    sqrt_discriminant = np.sqrt(discriminant)

    is_quadratic = np.abs(coeff_A) > eps
    is_linear = ~is_quadratic & (np.abs(coeff_B) > eps)

    t_lower = np.full_like(coeff_A, np.inf)
    t_upper = np.full_like(coeff_A, np.inf)
    t_linear = np.full_like(coeff_A, np.inf)
    np.divide(
        -coeff_B - sqrt_discriminant,
        2 * coeff_A,
        out=t_lower,
        where=is_quadratic,
    )
    np.divide(
        -coeff_B + sqrt_discriminant,
        2 * coeff_A,
        out=t_upper,
        where=is_quadratic,
    )
    np.divide(-coeff_C, coeff_B, out=t_linear, where=is_linear)

    t_roots = np.stack([t_lower, t_upper, t_linear])
    root_is_admissible = (t_roots >= -eps) & (t_roots <= 1.0 + eps)
    t_roots = np.where(root_is_admissible, np.clip(t_roots, 0.0, 1.0), np.inf)
    t = np.min(t_roots, axis=0)
    t = np.where(np.isfinite(t), t, 0.0)
    t = np.where(P_qhat < P_min, t, 1.0)

    theta2 = np.min(t, axis=-1)
    uj[irho] = rho_hat
    theta[...] = theta2[na, ...]


def zhang_shu_operator(wj: ArrayLike, w: ArrayLike, theta: ArrayLike):
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        wj: Array of high-order interpolation values that is revised.
        w: Array of first-order interpolation values.
        theta: Array of limiting coefficients.
    """
    wj[...] = theta * (wj - w) + w


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    compute_theta_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_theta_kernel(
            const double* __restrict__ w,
            const double* __restrict__ wj,
            const double* __restrict__ M,
            const double* __restrict__ m,
            double* __restrict__ Mj,
            double* __restrict__ mj,
            double* __restrict__ theta,
            const double eps,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterps
        ) {
            // w        has shape (nvars, nx, ny, nz)
            // wj       has shape (nvars, nx, ny, nz, ninterps)
            // M, m     have shape (nvars, nx, ny, nz)
            // Mj, mj   have shape (nvars, nx, ny, nz)
            // theta    has shape (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            long long n = (long long)nvars * nx * ny * nz;

            for (long long i = tid; i < n; i += stride) {

                const double* row = wj + i * (long long)ninterps;

                double mj_val = row[0];
                double Mj_val = row[0];

                for (int ii = 1; ii < ninterps; ++ii) {
                    const double vj = row[ii];
                    mj_val = (vj < mj_val) ? vj : mj_val;
                    Mj_val = (vj > Mj_val) ? vj : Mj_val;
                }

                mj[i] = mj_val;
                Mj[i] = Mj_val;
                double theta_M = fabs(M[i] - w[i]) / (fabs(Mj_val - w[i]) + eps);
                double theta_m = fabs(m[i] - w[i]) / (fabs(mj_val - w[i]) + eps);
                theta[i] = fmin(1.0, fmin(theta_M, theta_m));
            }
        }
        """,
        "compute_theta_kernel",
    )

    compute_pp_theta_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_pp_theta_kernel(
            const double* __restrict__ u,
            double* __restrict__ uj,
            double* __restrict__ theta,
            const double rho_min,
            const double P_min,
            const double gamma,
            const double eps,
            const int irho,
            const int imx,
            const int imy,
            const int imz,
            const int iE,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterps
        ) {
            // u        has shape (nvars, nx, ny, nz)
            // uj       has shape (nvars, nx, ny, nz, ninterps), and density is updated in-place
            // theta    has shape (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long ncells = (long long)nx * ny * nz;
            const double root_inf = 1.0e300;

            for (long long cell = tid; cell < ncells; cell += stride) {
                const long long rho_cell = (long long)irho * ncells + cell;
                const long long mx_cell = (long long)imx * ncells + cell;
                const long long my_cell = (long long)imy * ncells + cell;
                const long long mz_cell = (long long)imz * ncells + cell;
                const long long E_cell = (long long)iE * ncells + cell;

                const long long rho_node = rho_cell * (long long)ninterps;
                const long long mx_node = mx_cell * (long long)ninterps;
                const long long my_node = my_cell * (long long)ninterps;
                const long long mz_node = mz_cell * (long long)ninterps;
                const long long E_node = E_cell * (long long)ninterps;

                double rho_jmin = uj[rho_node];
                for (int alpha = 1; alpha < ninterps; ++alpha) {
                    const double rho_j = uj[rho_node + alpha];
                    rho_jmin = (rho_j < rho_jmin) ? rho_j : rho_jmin;
                }

                const double rho_bar = u[rho_cell];
                const double theta1 = (rho_jmin < rho_min)
                    ? fmin((rho_bar - rho_min) / (rho_bar - rho_jmin + eps), 1.0)
                    : 1.0;
                const double mx_bar = u[mx_cell];
                const double my_bar = u[my_cell];
                const double mz_bar = u[mz_cell];
                const double E_bar = u[E_cell];
                const double momentum_bar_sq = mx_bar * mx_bar + my_bar * my_bar + mz_bar * mz_bar;

                double theta2 = 1.0;
                for (int alpha = 0; alpha < ninterps; ++alpha) {
                    const double rho_hat = theta1 * (uj[rho_node + alpha] - rho_bar) + rho_bar;
                    const double mx_hat = uj[mx_node + alpha];
                    const double my_hat = uj[my_node + alpha];
                    const double mz_hat = uj[mz_node + alpha];
                    const double E_hat = uj[E_node + alpha];

                    uj[rho_node + alpha] = rho_hat;

                    const double P_hat = (gamma - 1.0) * (
                        E_hat - 0.5 * (
                            mx_hat * mx_hat + my_hat * my_hat + mz_hat * mz_hat
                        ) / rho_hat
                    );

                    double t = 1.0;
                    if (P_hat < P_min) {
                        const double drho = rho_hat - rho_bar;
                        const double dmx = mx_hat - mx_bar;
                        const double dmy = my_hat - my_bar;
                        const double dmz = mz_hat - mz_bar;
                        const double dE = E_hat - E_bar;

                        const double momentum_diff_sq = dmx * dmx + dmy * dmy + dmz * dmz;
                        const double momentum_bar_dot_diff =
                            mx_bar * dmx + my_bar * dmy + mz_bar * dmz;

                        const double coeff_A =
                            (gamma - 1.0) * (momentum_diff_sq - 2.0 * dE * drho);
                        const double coeff_B =
                            -2.0 * (gamma - 1.0) * (
                                E_bar * drho + rho_bar * dE - momentum_bar_dot_diff
                            ) + 2.0 * P_min * drho;
                        const double coeff_C =
                            (gamma - 1.0) * (momentum_bar_sq - 2.0 * E_bar * rho_bar)
                            + 2.0 * P_min * rho_bar;

                        const double discriminant =
                            fmax(coeff_B * coeff_B - 4.0 * coeff_A * coeff_C, 0.0);
                        const double sqrt_discriminant = sqrt(discriminant);

                        double t_best = root_inf;
                        if (fabs(coeff_A) > eps) {
                            const double denom = 2.0 * coeff_A;
                            double t_lower = (-coeff_B - sqrt_discriminant) / denom;
                            double t_upper = (-coeff_B + sqrt_discriminant) / denom;

                            if (t_lower >= -eps && t_lower <= 1.0 + eps) {
                                t_lower = fmin(fmax(t_lower, 0.0), 1.0);
                                t_best = fmin(t_best, t_lower);
                            }
                            if (t_upper >= -eps && t_upper <= 1.0 + eps) {
                                t_upper = fmin(fmax(t_upper, 0.0), 1.0);
                                t_best = fmin(t_best, t_upper);
                            }
                        } else if (fabs(coeff_B) > eps) {
                            double t_linear = -coeff_C / coeff_B;
                            if (t_linear >= -eps && t_linear <= 1.0 + eps) {
                                t_linear = fmin(fmax(t_linear, 0.0), 1.0);
                                t_best = t_linear;
                            }
                        }

                        t = (t_best < root_inf) ? t_best : 0.0;
                    }

                    theta2 = fmin(theta2, t);
                }

                for (int var = 0; var < nvars; ++var) {
                    theta[(long long)var * ncells + cell] = theta2;
                }
            }
        }
        """,
        "compute_pp_theta_kernel",
    )

    def compute_theta_kernel_helper(
        w: cp.ndarray,
        wj: cp.ndarray,
        M: cp.ndarray,
        m: cp.ndarray,
        Mj: cp.ndarray,
        mj: cp.ndarray,
        theta: cp.ndarray,
        eps: float,
    ):
        if not w.flags.c_contiguous or w.ndim != 4:
            raise ValueError("Array `w` must be a C-contiguous, 4-dimensional array.")
        if not wj.flags.c_contiguous or wj.ndim != 5 or wj.shape[:4] != w.shape:
            raise ValueError(
                "Array `wj` must be a C-contiguous, 5-dimensional array of shape "
                "(nvars, nx, ny, nz)."
            )
        if not M.flags.c_contiguous or M.shape != w.shape:
            raise ValueError("Array `M` must be a C-contiguous array of the same shape as `w`.")
        if not m.flags.c_contiguous or m.shape != w.shape:
            raise ValueError("Array `m` must be a C-contiguous array of the same shape as `w`.")
        if not Mj.flags.c_contiguous or Mj.shape != w.shape:
            raise ValueError("Array `Mj` must be a C-contiguous array of the same shape as `w`.")
        if not mj.flags.c_contiguous or mj.shape != w.shape:
            raise ValueError("Array `mj` must be a C-contiguous array of the same shape as `w`.")
        if not theta.flags.c_contiguous or theta.shape != w.shape:
            raise ValueError("Array `theta` must be a C-contiguous array of the same shape as `w`.")
        if (
            w.dtype != cp.float64
            or wj.dtype != cp.float64
            or M.dtype != cp.float64
            or m.dtype != cp.float64
            or Mj.dtype != cp.float64
            or mj.dtype != cp.float64
            or theta.dtype != cp.float64
        ):
            raise ValueError("All input arrays must have dtype float64.")

        nvars, nx, ny, nz = w.shape
        _, _, _, _, ninterps = wj.shape

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nvars * nx + threads_per_block - 1) // threads_per_block

        compute_theta_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (w, wj, M, m, Mj, mj, theta, eps, nvars, nx, ny, nz, ninterps),
        )

    def compute_pp_theta_kernel_helper(
        u: cp.ndarray,
        uj: cp.ndarray,
        theta: cp.ndarray,
        idx: VariableIndexMap,
        rho_min: float,
        P_min: float,
        gamma: float,
        eps: float,
    ):
        if not u.flags.c_contiguous or u.ndim != 4:
            raise ValueError("Array `u` must be a C-contiguous, 4-dimensional array.")
        if not uj.flags.c_contiguous or uj.ndim != 5 or uj.shape[:4] != u.shape:
            raise ValueError(
                "Array `uj` must be a C-contiguous, 5-dimensional array of shape "
                "(nvars, nx, ny, nz, ninterps)."
            )
        if not theta.flags.c_contiguous or theta.shape != u.shape:
            raise ValueError("Array `theta` must be a C-contiguous array of the same shape as `u`.")
        if u.dtype != cp.float64 or uj.dtype != cp.float64 or theta.dtype != cp.float64:
            raise ValueError("All input arrays must have dtype float64.")

        nvars, nx, ny, nz = u.shape
        _, _, _, _, ninterps = uj.shape
        irho = idx("rho")
        imx = idx("mx")
        imy = idx("my")
        imz = idx("mz")
        iE = idx("E")
        if max(irho, imx, imy, imz, iE) >= nvars:
            raise ValueError("Index map contains a conservative variable outside `u`.")

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nx * ny * nz + threads_per_block - 1) // threads_per_block

        compute_pp_theta_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                u,
                uj,
                theta,
                rho_min,
                P_min,
                gamma,
                eps,
                irho,
                imx,
                imy,
                imz,
                iE,
                nvars,
                nx,
                ny,
                nz,
                ninterps,
            ),
        )
