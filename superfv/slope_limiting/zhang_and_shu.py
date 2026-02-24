from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, cast

from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.tools.buffer import check_buffer_slots
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
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


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    theta_kernel = cp.ElementwiseKernel(
        in_params=(
            "float64 u, float64 M, float64 m, float64 Mj, float64 mj, float64 tol"
        ),
        out_params="float64 theta",
        operation="""
        double theta_M = fabs(M - u) / (fabs(Mj - u) + tol);
        double theta_m = fabs(m - u) / (fabs(mj - u) + tol);
        theta = fmin(fmin(theta_M, theta_m), 1.0);
        """,
        name="theta_kernel",
        no_return=True,
    )

    compute_theta_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void compute_theta_kernel(
            const double* w,
            const double* wj,
            const double* M,
            const double* m,
            double* Mj,
            double* mj,
            double* out,
            const int nvars,
            const int nx,
            const int ninterps,
            const double eps
        ) {
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            int n = nvars * nx;

            for (int i = tid; i < n; i += stride) {
                const double* row = wj + ((size_t)i) * (size_t)ninterps;

                mj[i] = row[0];
                Mj[i] = row[0];
                for (int j = 1; j < ninterps; ++j) {
                    double vj = row[j];
                    mj[i] = (vj < mj[i]) ? vj : mj[i];
                    Mj[i] = (vj > Mj[i]) ? vj : Mj[i];
                }

                out[i] = 1.0;
                out[i] = fmin(fabs(M[i] - w[i]) / (fabs(Mj[i] - w[i]) + eps), out[i]);
                out[i] = fmin(fabs(m[i] - w[i]) / (fabs(mj[i] - w[i]) + eps), out[i]);
            }
        }
        """,
        "compute_theta_kernel",
    )


def compute_theta_kernel_helper(
    w: ArrayLike,
    wj: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    Mj: ArrayLike,
    mj: ArrayLike,
    out: ArrayLike,
    eps: float,
):
    nvars, nx = w.shape
    ninterps = wj.shape[2]
    threads_per_block = 256
    blocks_per_grid = (nvars * nx + threads_per_block - 1) // threads_per_block
    compute_theta_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (w, wj, M, m, Mj, mj, out, nvars, nx, ninterps, eps),
    )


def compute_theta(
    xp: ModuleType,
    u: ArrayLike,
    center_nodes: ArrayLike,
    x_nodes: Optional[ArrayLike],
    y_nodes: Optional[ArrayLike],
    z_nodes: Optional[ArrayLike],
    *,
    out: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    Mj: ArrayLike,
    mj: ArrayLike,
    buffer: ArrayLike,
    config: ZhangShuConfig,
) -> Tuple[slice, ...]:
    """
    Compute Zhang and Shu's a priori slope limiting parameter theta based on arrays of
    finite-volume nodes and averages.

    Args:
        xp: `np` namespace.
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
    dmp_valid = compute_dmp(xp, u, active_dims, include_corners, M=M, m=m)

    # compute nodal maximum principle
    Mj[...] = center_nodes[..., 0]
    mj[...] = center_nodes[..., 0]
    for nodes in [x_nodes, y_nodes, z_nodes]:
        if nodes is None:
            continue
        xp.minimum(mj, xp.min(nodes, axis=4), out=mj)
        xp.maximum(Mj, xp.max(nodes, axis=4), out=Mj)

    # compute theta
    if hasattr(xp, "cuda"):
        theta_kernel(u, M, m, Mj, mj, tol, theta[..., 0])
    else:
        theta[..., 0] = xp.minimum(
            xp.minimum(
                xp.divide(xp.abs(M - u), xp.abs(Mj - u) + tol),
                xp.divide(xp.abs(m - u), xp.abs(mj - u) + tol),
            ),
            1.0,
        )

    valid = cast(Tuple[slice, ...], insert_slice(dmp_valid, 4, slice(0, 1)))
    out[valid] = theta[valid]

    return valid


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
