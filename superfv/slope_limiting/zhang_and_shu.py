from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, cast

from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import replace_slice

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
        SED: Whether to use the smooth extrema detector to relax the limiter.
        adaptive_dt: Whether to use adaptive time stepping.
        max_dt_revisions: The maximum number of revisions for the time step.
        include_corners: Whether to include corner cells when computing the discrete
            maximum principle.
        PAD_bounds: An array of shape (nvars, 2) specifying the physical bounds
            (min, max) for each variable.
        PAD_atol: Absolute tolerance for PAD violations.
        tol: Tolerance for the denominator of the
    """

    SED: bool
    adaptive_dt: bool
    max_dt_revisions: int
    include_corners: bool = False
    PAD_bounds: Optional[ArrayLike] = None
    PAD_atol: float = 0.0
    tol: float = 1e-16

    def __post_init__(self):
        if self.adaptive_dt and self.PAD_bounds is None:
            raise ValueError(
                "Adaptive time stepping requires PAD_bounds to be set. "
                "Set adaptive_dt=False if you do not want to use PAD."
            )

    def key(self) -> str:
        return "zhang-shu"

    def to_dict(self) -> dict:
        return dict(
            SED=self.SED,
            adaptive_dt=self.adaptive_dt,
            max_dt_revisions=self.max_dt_revisions,
            include_corners=self.include_corners,
            PAD_bounds=(
                None
                if self.PAD_bounds is None
                else self.PAD_bounds[:, 0, 0, 0, :].tolist()
            ),
            PAD_atol=self.PAD_atol,
            tol=self.tol,
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
    dmp: ArrayLike,
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
        dmp: Array to which the discrete maximum principle is written. Has shape
            (nvars, nx, ny, nz, >=2).
        buffer: Array to which intermediate values are written. Has shape
            (nvars, nx, ny, nz, >=3).
        config: Configuration for the Zhang-Shu limiter.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    include_corners = config.include_corners
    SED = config.SED
    tol = config.tol

    # allocate arrays
    node_mp = buffer[..., :2]
    alpha = buffer[..., 2:3]

    # compute discrete maximum principle
    active_dims = tuple(
        cast(Literal["x", "y", "z"], dim)
        for dim, arr in zip(["x", "y", "z"], [x_nodes, y_nodes, z_nodes])
        if arr is not None
    )
    dmp_modified = compute_dmp(
        xp, u, active_dims, out=dmp, include_corners=include_corners
    )

    # compute nodal maximum principle
    node_mp[..., 0] = center_nodes[..., 0]
    node_mp[..., 1] = center_nodes[..., 0]
    for nodes in [x_nodes, y_nodes, z_nodes]:
        if nodes is None:
            continue
        xp.minimum(node_mp[..., 0], xp.min(nodes, axis=4), out=node_mp[..., 0])
        xp.maximum(node_mp[..., 1], xp.max(nodes, axis=4), out=node_mp[..., 1])

    # compute theta
    m = dmp[..., 0]
    M = dmp[..., 1]
    mj = node_mp[..., 0]
    Mj = node_mp[..., 1]
    theta = xp.minimum(
        xp.minimum(
            xp.divide(xp.abs(M - u), xp.abs(Mj - u) + tol),
            xp.divide(xp.abs(m - u), xp.abs(mj - u) + tol),
        ),
        1.0,
    )

    # assign theta
    inner = replace_slice(dmp_modified, 4, 0)
    out[inner] = theta[inner[:-1]]

    # relax theta using a smooth extrema detector
    if SED:
        modified = smooth_extrema_detector(
            xp,
            u,
            active_dims,
            out=alpha,
            buffer=buffer[..., 1:],
        )
        out[modified] = xp.where(alpha[modified] < 1, out[modified], 1)
    else:
        modified = cast(Tuple[slice, ...], replace_slice(inner, 4, slice(0, 1)))

    return modified


def zhang_shu_operator(u_ho: ArrayLike, u_fo: ArrayLike, theta: ArrayLike) -> ArrayLike:
    """
    Zhang and Shu operator for limiting the high-order solution.

    Args:
        u_ho: Array of high-order interpolation values.
        u_fo: Array of first-order interpolation values.
        theta: Array of limiting coefficients.

    Returns:
        ArrayLike: Array of limited values.
    """
    return theta * (u_ho - u_fo) + u_fo


def init_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Initialize Zhang-Shu limiter statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_1-theta_real_mean"] = []
    step_log["nfine_1-theta_real_max"] = []
    step_log["nfine_1-theta_vis_mean"] = []
    step_log["nfine_1-theta_vis_max"] = []

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_1-theta_real_{var}"] = []
        step_log[f"nfine_1-theta_vis_{var}"] = []


def clear_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Clear Zhang-Shu limiter statistics in the finite volume solver's step log.

    Args:
        fv_solver: The finite volume solver instance.
    """
    idx = fv_solver.variable_index_map
    step_log = fv_solver.step_log  # gets mutated

    step_log["nfine_1-theta_real_mean"].clear()
    step_log["nfine_1-theta_real_max"].clear()
    step_log["nfine_1-theta_vis_mean"].clear()
    step_log["nfine_1-theta_vis_max"].clear()

    for var in idx.var_idx_map.keys():
        step_log[f"nfine_1-theta_real_{var}"].clear()
        step_log[f"nfine_1-theta_vis_{var}"].clear()


def append_zhang_shu_scalar_statistics(fv_solver: FiniteVolumeSolver):
    """
    Append Zhang-Shu limiter statistics from the finite volume solver's arrays to the
    step log. Specifically, log the sum of cells with 1 - theta > 0 using various
    criteria:
        nfine_1-theta_real_mean: Real values, mean over all variables.
        nfine_1-theta_real_max: Real values, max over all variables.
        nfine_1-theta_vis_mean: Visualized values, mean over all variables.
        nfine_1-theta_vis_max: Visualized values, max over all variables.
        nfine_1-theta_{var}_real: Real values for variable `var`.
        nfine_1-theta_{var}_vis: Visualized values for variable `var`.

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
    theta = arrays["theta"][interior][..., 0]
    theta_vis = arrays["theta_vis"]
    buffer = arrays["buffer"]

    if nvars < 4:
        raise ValueError(
            "Zhang-Shu limiter logging requires at least 4 variable slots."
        )

    one_minus_theta_real = buffer[interior][..., 0]
    one_minus_theta_vis = buffer[interior][..., 1]
    mean_one_minus_theta_real = buffer[interior][0, ..., 2]
    max_one_minus_theta_real = buffer[interior][1, ..., 2]
    mean_one_minus_theta_vis = buffer[interior][2, ..., 2]
    max_one_minus_theta_vis = buffer[interior][3, ..., 2]

    # track scalar quantities
    one_minus_theta_real[...] = 1 - theta
    one_minus_theta_vis[...] = 1 - theta_vis

    xp.mean(one_minus_theta_real, axis=0, out=mean_one_minus_theta_real)
    xp.max(one_minus_theta_real, axis=0, out=max_one_minus_theta_real)

    xp.mean(one_minus_theta_vis, axis=0, out=mean_one_minus_theta_vis)
    xp.max(one_minus_theta_vis, axis=0, out=max_one_minus_theta_vis)

    n_real_mean = xp.sum(mean_one_minus_theta_real).item()
    n_real_max = xp.sum(max_one_minus_theta_real).item()
    n_vis_mean = xp.sum(mean_one_minus_theta_vis).item()
    n_vis_max = xp.sum(max_one_minus_theta_vis).item()

    step_log["nfine_1-theta_real_mean"].append(n_real_mean)
    step_log["nfine_1-theta_real_max"].append(n_real_max)
    step_log["nfine_1-theta_vis_mean"].append(n_vis_mean)
    step_log["nfine_1-theta_vis_max"].append(n_vis_max)

    for var in idx.var_idx_map.keys():
        n_real = xp.sum(one_minus_theta_real[idx(var), ...]).item()
        n_vis = xp.sum(one_minus_theta_vis[idx(var), ...]).item()

        step_log[f"nfine_1-theta_real_{var}"].append(n_real)
        step_log[f"nfine_1-theta_vis_{var}"].append(n_vis)


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
        "nfine_1-theta_real_mean": step_log["nfine_1-theta_real_mean"].copy(),
        "nfine_1-theta_real_max": step_log["nfine_1-theta_real_max"].copy(),
        "nfine_1-theta_vis_mean": step_log["nfine_1-theta_vis_mean"].copy(),
        "nfine_1-theta_vis_max": step_log["nfine_1-theta_vis_max"].copy(),
        "n_1-theta_real_mean": zero_max(step_log["nfine_1-theta_real_mean"]),
        "n_1-theta_real_max": zero_max(step_log["nfine_1-theta_real_max"]),
        "n_1-theta_vis_mean": zero_max(step_log["nfine_1-theta_vis_mean"]),
        "n_1-theta_vis_max": zero_max(step_log["nfine_1-theta_vis_max"]),
    }

    for var in idx.var_idx_map.keys():
        keys = ["1-theta_real", "1-theta_vis"]
        for key in keys:
            new_data[f"nfine_{key}_{var}"] = step_log[f"nfine_{key}_{var}"].copy()
            new_data[f"n_{key}_{var}"] = zero_max(step_log[f"nfine_{key}_{var}"])

    data.update(new_data)
