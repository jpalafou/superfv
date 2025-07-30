from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Optional, cast

from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.smooth_extrema_detection import (
    inplace_smooth_extrema_detector,
)
from superfv.tools.device_management import ArrayLike
from superfv.tools.slicing import modify_slices

if TYPE_CHECKING:
    pass


@dataclass
class ZhangShuConfig:
    """
    Configuration for Zhang and Shu slope limiting.

    Attributes:
        include_corners: Whether to include corners when computing the discrete maximum
            principle.
        tol: Small tolerance value for division in the expression for theta:
            min(abs(M - u) / (abs(Mj - u) + tol), abs(m - u) / (abs(mj - u) + tol))
        SED: Whether to use the smooth extrema detector to relax theta.
        adaptive_dt: Whether to adapt the time step based on a PAD condition.
        max_dt_revisions: Maximum number of revisions for the adaptive time step.
        PAD_bounds: Optional bounds for the PAD condition. Has shape
            (nvars, 1, 1, 1, 2), where the minimum and maximum values are stored
            in the first and second elements of the last axis, respectively, for each
            variable. If None, the PAD condition is not used.
        PAD_atol: Absolute tolerance for the PAD condition.
    """

    include_corners: bool = False
    tol: float = 1e-16
    SED: bool = False
    adaptive_dt: bool = False
    max_dt_revisions: int = 0
    PAD_bounds: Optional[ArrayLike] = None
    PAD_atol: float = 0.0

    def __post_init__(self):
        if self.adaptive_dt and self.PAD_bounds is None:
            raise ValueError(
                "Adaptive time stepping requires PAD_bounds to be set. "
                "Set adaptive_dt=False if you do not want to use PAD."
            )


def compute_theta(
    xp: ModuleType,
    u: ArrayLike,
    center_nodes: ArrayLike,
    x_nodes: Optional[ArrayLike],
    y_nodes: Optional[ArrayLike],
    z_nodes: Optional[ArrayLike],
    buffer: ArrayLike,
    out: ArrayLike,
    include_corners: bool = False,
    SED: bool = False,
    tol: float = 1e-16,
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
        buffer: Array to which intermediate values are written.
        out: Array to which theta is written. Has shape (nvars, nx, ny, nz, 1).
        include_corners: Whether to include corners when computing the discrete maximum
            principle. Defaults to False.
        tol: Small tolerance value for division.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """

    # allocate arrays
    dmp = buffer[..., :2]
    node_mp = buffer[..., 2:4]

    # compute discrete maximum principle
    active_dims = tuple(
        cast(Literal["x", "y", "z"], dim)
        for dim, arr in zip(["x", "y", "z"], [x_nodes, y_nodes, z_nodes])
        if arr is not None
    )
    dmp_modified = compute_dmp(xp, u, active_dims, include_corners, dmp)

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
        modified = inplace_smooth_extrema_detector(
            xp, u, active_dims, buffer[..., 1:], alpha
        )
        out[modified] = xp.where(alpha[modified] == 1, 1, out[modified])
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
