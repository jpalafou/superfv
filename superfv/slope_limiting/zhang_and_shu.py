from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, cast

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE
from superfv.tools.slicing import insert_slice

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
        check_uniformity: Whether to relax alpha to 1.0 in uniform regions if smooth
            extrema detection is enabled. Uniform regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
        physical_admissibility_detection: Whether to enable physical admissibility
            detection (PAD).
        eta_max: Eta threshold for shock detection if shock_detection is True.
        PAD_bounds: Array with shape (nvars, 2) specifying the lower and upper bounds,
            respectively, for each variable when physical_admissibility_detection is
            True. Must be provided if physical_admissibility_detection is True.
        PAD_atol: Absolute tolerance for physical admissibility detection if
            physical_admissibility_detection is True.
        uniformity_tol: Tolerance for uniformity check when check_uniformity is True.
        include_corners: Whether to include corner cells when computing the discrete
            maximum principle.
        adaptive_dt: Whether to use adaptive time stepping. If True,
            physical_admissibility_detection must also be True.
        theta_denom_tol: Tolerance for the denominator of the theta calculation.
    """

    include_corners: bool = False
    adaptive_dt: bool = False
    theta_denom_tol: float = 1e-16

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
    u: np.ndarray,
    center_nodes: np.ndarray,
    x_nodes: Optional[np.ndarray],
    y_nodes: Optional[np.ndarray],
    z_nodes: Optional[np.ndarray],
    *,
    out: np.ndarray,
    M: np.ndarray,
    m: np.ndarray,
    Mj: np.ndarray,
    mj: np.ndarray,
    buffer: np.ndarray,
    config: ZhangShuConfig,
) -> Tuple[slice, ...]:
    """
    Compute Zhang and Shu's a priori slope limiting parameter theta based on arrays of
    finite-volume nodes and averages.

    Args:
        u: Array of finite-volume average. Has shape (nvars, nx, ny, nz).
        center_nodes: Array of central node values. Has shape (nvars, nx, ny, nz, 1).
        x_nodes, y_nodes, z_nodes: Optional array of x,y,z-face node values. Has shape
            (nvars, nx, ny, nz, 2*n_nodes). If None, the x,y,z face is not considered.
        out: Array to which theta is written. Has shape (nvars, nx, ny, nz, >=1).
        M: Array to which the discrete maximum principle is written. Has shape
            (nvars, nx, ny, nz).
        m: Array to which the discrete minimum principle is written. Has shape
            (nvars, nx, ny, nz).
        Mj: Array to which the nodal maximum principle is written. Has shape
            (nvars, nx, ny, nz).
        mj: Array to which the nodal minimum principle is written. Has shape
            (nvars, nx, ny, nz).
        buffer: Array to which temporary values are assigned. Has different shape
            requirements depending on whether SED is used and the number (length) of
            active dimensions:
            - without SED: (nvars, nx, ny, nz, >=3)
            - with SED, 1D: (nvars, nx, ny, nz, >=10)
            - with SED, 2D: (nvars, nx, ny, nz, >=12)
            - with SED, 3D: (nvars, nx, ny, nz, >=13)
        config: Configuration for the Zhang-Shu limiter.

    Returns:
        Slice objects indicating the valid regions in the output array.
    """
    include_corners = config.include_corners
    tol = config.theta_denom_tol

    # allocate arrays
    check_buffer_slots(buffer, required=1)
    theta = buffer[..., :1]

    # compute discrete maximum principle
    active_dims = tuple(
        cast(Literal["x", "y", "z"], dim)
        for dim, arr in zip(["x", "y", "z"], [x_nodes, y_nodes, z_nodes])
        if arr is not None
    )
    dmp_valid = compute_dmp(u, active_dims, include_corners, M=M, m=m)

    # compute nodal maximum principle
    Mj[...] = center_nodes[..., 0]
    mj[...] = center_nodes[..., 0]
    for nodes in [x_nodes, y_nodes, z_nodes]:
        if nodes is None:
            continue
        np.minimum(mj, np.min(nodes, axis=4), out=mj)
        np.maximum(Mj, np.max(nodes, axis=4), out=Mj)

    # compute theta
    theta[..., 0] = np.minimum(
        np.minimum(
            np.divide(np.abs(M - u), np.abs(Mj - u) + tol),
            np.divide(np.abs(m - u), np.abs(mj - u) + tol),
        ),
        1.0,
    )

    valid = cast(Tuple[slice, ...], insert_slice(dmp_valid, 4, slice(0, 1)))
    out[valid] = theta[valid]

    return valid


def zhang_shu_operator(wj: np.ndarray, w: np.ndarray, theta: np.ndarray):
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
    check_buffer_slots(arrays["_buffer_"], required=3)
    theta = arrays["_theta_"][interior][..., 0]
    buffer = arrays["_buffer_"]

    if nvars < 4:
        raise ValueError(
            "Zhang-Shu limiter logging requires at least 4 variable slots."
        )

    one_minus_theta = buffer[interior][..., 0]
    mean_one_minus_theta = buffer[interior][0, ..., 2]
    max_one_minus_theta = buffer[interior][1, ..., 2]

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
