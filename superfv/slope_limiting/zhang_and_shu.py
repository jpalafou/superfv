from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.interpolation_schemes import LimiterConfig
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike

if TYPE_CHECKING:
    from superfv.finite_volume_solver import FiniteVolumeSolver


@dataclass(frozen=True, slots=True)
class ZhangShuConfig(LimiterConfig):
    """
    Configuration for the Zhang-Shu slope limiter,

    θ = min(|M-u|/|Mj-u|, |m-u|/|mj-u|, 1),

    where M, m are the local maxima and minima, respectively, and Mj and mj are the
    maxima and minima of the nodes, respectively.

    Attributes:
        shock_detection: Whether to enable shock detection.
        smooth_extrema_detection: Whether to enable smooth extrema detection.
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        include_corners: Whether to include corner cells when computing the discrete
            maximum principle.
        adaptive_dt: Whether to use adaptive time stepping. If True,
            physical_admissibility_detection must also be True.
        adaptive_dt_tol: Tolerance for physical admissibility detection when deciding
            whether to refine the timestep size if adaptive_dt is True.
        theta_denom_tol: Tolerance for the denominator of the theta calculation.
    """

    include_corners: bool = False
    adaptive_dt: bool = False
    adaptive_dt_tol: float = 1e-15
    theta_denom_tol: float = 1e-15

    def __post_init__(self):
        LimiterConfig.__post_init__(self)
        if self.adaptive_dt and not self.physical_admissibility_detection:
            raise ValueError(
                "adaptive_dt can only be True if physical_admissibility_detection"
                " is True."
            )

    def key(self) -> str:
        return "zhang-shu"

    def to_dict(self) -> dict:
        out = LimiterConfig.to_dict(self)
        out.update(
            dict(
                include_corners=self.include_corners,
                adaptive_dt=self.adaptive_dt,
                theta_denom_tol=self.theta_denom_tol,
            )
        )
        return out


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
    Compute Zhang and Shu's a priori slope limiting parameter and write it to `theta`.

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


def zhang_shu_operator(wj: ArrayLike, w: ArrayLike, theta: ArrayLike):
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        wj: Array of high-order interpolation values that is revised.
        w: Array of first-order interpolation values.
        theta: Array of limiting coefficients.
    """
    wj[...] = theta * (wj - w) + w


def init_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Initialize Zhang-Shu limiter statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_1-theta_mean"] = []
    step_log["nfine_1-theta_max"] = []

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_1-theta_{var}"] = []


def clear_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Clear Zhang-Shu limiter statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_1-theta_mean"].clear()
    step_log["nfine_1-theta_max"].clear()

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_1-theta_{var}"].clear()


def append_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Append Zhang-Shu limiter statistics from the finite volume solver's arrays to the
    step log. Specifically, log the sum of cells with 1 - theta > 0 using various
    criteria:
        nfine_1-theta_mean: Sum over all cells of the mean over all variables of
            1-theta.
        nfine_1-theta_max: Sum over all cells of the max over all variables of
            1-theta.
        nfine_1-theta_{var}: Sum over all cells of 1-theta for variable `var`.

    Args:
        fv_solver: The finite volume solver instance.
    """
    xp = fv_solver.xp
    arrays = fv_solver.arrays
    interior = fv_solver.interior
    idx = fv_solver.variable_index_map
    nvars = fv_solver.nvars
    step_log = fv_solver.step_log  # gets mutated

    # allocate arrays
    theta = arrays["_theta_"][interior][..., 0]

    if nvars < 4:
        raise ValueError(
            "Zhang-Shu limiter logging requires at least 4 variable slots."
        )

    one_minus_theta = xp.empty_like(theta)
    mean_one_minus_theta = xp.empty_like(theta[0, ...])
    max_one_minus_theta = xp.empty_like(theta[0, ...])

    # track scalar quantities
    one_minus_theta[...] = 1 - theta

    xp.mean(one_minus_theta, axis=0, out=mean_one_minus_theta)
    xp.max(one_minus_theta, axis=0, out=max_one_minus_theta)

    n_mean = xp.sum(mean_one_minus_theta).item()
    n_max = xp.sum(max_one_minus_theta).item()

    step_log["nfine_1-theta_mean"].append(n_mean)
    step_log["nfine_1-theta_max"].append(n_max)

    for var in idx.var_idx_map.keys():
        n = xp.sum(one_minus_theta[idx(var), ...]).item()
        step_log[f"nfine_1-theta_{var}"].append(n)


def log_zhang_shu_scalar_statistics(
    fv_solver: FiniteVolumeSolver, data: Dict[str, Any]
):
    """
    Log Zhang-Shu limiter statistics from the finite volume solver's step log into the
    provided data dictionary.

    Args:
        fv_solver: The finite volume solver instance.
        data: The dictionary to which statistics are logged.
    """
    step_log = fv_solver.step_log
    idx = fv_solver.variable_index_map

    def zero_max(lst: list[float]) -> float:
        return max(lst) if lst else 0.0

    new_data = {
        "nfine_1-theta_mean": step_log["nfine_1-theta_mean"].copy(),
        "nfine_1-theta_max": step_log["nfine_1-theta_max"].copy(),
        "n_1-theta_mean": zero_max(step_log["nfine_1-theta_mean"]),
        "n_1-theta_max": zero_max(step_log["nfine_1-theta_max"]),
    }

    for var in idx.var_idx_map.keys():
        new_data[f"nfine_1-theta_{var}"] = step_log[f"nfine_1-theta_{var}"].copy()
        new_data[f"n_1-theta_{var}"] = zero_max(step_log[f"nfine_1-theta_{var}"])

    data.update(new_data)


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
            raise ValueError(
                "Array `M` must be a C-contiguous array of the same shape as `w`."
            )
        if not m.flags.c_contiguous or m.shape != w.shape:
            raise ValueError(
                "Array `m` must be a C-contiguous array of the same shape as `w`."
            )
        if not Mj.flags.c_contiguous or Mj.shape != w.shape:
            raise ValueError(
                "Array `Mj` must be a C-contiguous array of the same shape as `w`."
            )
        if not mj.flags.c_contiguous or mj.shape != w.shape:
            raise ValueError(
                "Array `mj` must be a C-contiguous array of the same shape as `w`."
            )
        if not theta.flags.c_contiguous or theta.shape != w.shape:
            raise ValueError(
                "Array `theta` must be a C-contiguous array of the same shape as `w`."
            )
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
