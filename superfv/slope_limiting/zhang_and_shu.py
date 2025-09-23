from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Literal, Optional, cast

from superfv.interpolation_schemes import LimiterConfig
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import modify_slices


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
    buffer: ArrayLike,
    config: ZhangShuConfig,
):
    """
    Compute Zhang and Shu's a priori slope limiting parameter theta based on arrays of
    finite-volume nodes and averages.

    Args:
        xp: `np` namespace.
        u: Array of finite-volume average. Has shape (nvars, nx, ny, nz).
        center_nodes: Array of central node values. Has shape (nvars, nx, ny, nz, 1).
        x_nodes, y_nodes, z_nodes: Optional array of x,y,z-face node values. Has shape
            (nvars, nx, ny, nz, 2*n_nodes). If None, the x,y,z face is not considered.
        out: Array to which theta is written. Has shape (nvars, nx, ny, nz, 1).
        buffer: Array to which intermediate values are written.
        config: Configuration for the Zhang-Shu limiter.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    include_corners = config.include_corners
    SED = config.SED
    tol = config.tol

    # allocate arrays
    dmp = buffer[..., :2]
    node_mp = buffer[..., 2:4]

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
    out_modified = modify_slices(dmp_modified, axis=4, new_slice=0)
    out[out_modified] = theta[out_modified[:-1]]

    # relax theta using a smooth extrema detector
    if SED:
        alpha = buffer[..., :1]
        modified = smooth_extrema_detector(
            xp,
            u,
            active_dims,
            out=alpha,
            buffer=buffer[..., 1:],
        )
        out[modified] = xp.where(alpha[modified] < 1, out[modified], 1)
    else:
        modified = modify_slices(out_modified, axis=4, new_slice=slice(None, 1))

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
