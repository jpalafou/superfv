from typing import Dict, Literal, Optional, Tuple

import numpy as np

from superfv.tools.step_history import MultiTimer

from .axes import DIM_TO_AXIS
from .configs import BoundaryConditionParameters, FV_SchemeParameters
from .finite_volume_driver import (
    apply_fv_bc,
    apply_zhang_shu_limiter,
    get_interior_view,
    get_n_nodes_per_face,
    get_riemann_solver_slices,
    integrate_gauss_legendre_face_nodes,
    integrate_transverse_nodes,
    interpolate_cell_centers,
    interpolate_face_nodes,
)
from .mesh import UniformFiniteVolumeMesh
from .riemann_solvers import solve_riemann_problem
from .slope_limiting.muscl import compute_MUSCL_slopes, reconstruct_MUSCL_faces
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .stencils import conservative_interpolation, transverse_integration
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def update_advection_workspace(
    u: ArrayLike,
    _u_: ArrayLike,
    t: float,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    bc_params: BoundaryConditionParameters,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the `_u_` array with ghost-cell-padded finite volume averages.

    u: shape (nvars, nx, ny, nz) - Input array of cell averages without ghost cells.
        Is not modified.
    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Array to which u is written with boundary
        conditions applied to ghost cells along active dimensions.
    t: Current simulation time, used for time-dependent boundary conditions.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    mesh: Mesh object containing information about the mesh and its dimensions.
    bc_params: Parameters for the boundary conditions.
    timer: Optional[MultiTimer] = None - Timer object for performance measurement.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    active_dims = mesh.active_dims
    interior = get_interior_view(active_dims, mesh.nghost)

    # Update interior FV averages and apply BC
    timer is not None and timer.start("boundary_conditions", using_cupy)  # TIMER START
    _u_[interior] = u
    apply_fv_bc(_u_, idx, mesh, t, bc_params, None)
    timer is not None and timer.stop("boundary_conditions", using_cupy)  # TIMER STOP


def update_weno_fluxes(
    _u_: ArrayLike,
    _F_: ArrayLike,
    _G_: ArrayLike,
    _H_: ArrayLike,
    _theta_: ArrayLike,
    _ucc_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    timer: Optional[MultiTimer] = None,
):
    """Reconstruct advection face nodes, optionally limit them, and integrate upwind fluxes."""
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    xp = cp if using_cupy else np
    na = xp.newaxis
    fv = fv_params
    use_GL = fv.flux_quadrature == "gauss_legendre"
    n_nodes = get_n_nodes_per_face(len(active_dims), fv)
    node_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}

    for dim in active_dims:
        _nodes_ = xp.empty(_u_.shape + (2 * n_nodes,))
        interpolate_face_nodes(_u_, _nodes_, dim, active_dims, fv.p, use_GL, timer)
        node_dict[dim] = _nodes_

    if fv.zhang_shu_params.use_ZS:
        if fv.p > 1:
            interpolate_cell_centers(_u_, _ucc_, active_dims, fv.p, timer)
        apply_zhang_shu_limiter(
            _u_,
            _ucc_,
            node_dict["x"] if "x" in active_dims else np.array([]),
            node_dict["y"] if "y" in active_dims else np.array([]),
            node_dict["z"] if "z" in active_dims else np.array([]),
            _theta_,
            _alpha_,
            idx,
            True,
            active_dims,
            fv.p,
            fv.zhang_shu_params,
            timer,
        )

    for dim in active_dims:
        minus, plus = get_riemann_solver_slices(DIM_TO_AXIS[dim], n_nodes, mesh.nghost)
        _nodes_ = node_dict[dim]
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim]
        _fnodes_ = _F_out_[..., na] if len(active_dims) == 1 else xp.empty_like(_nodes_[minus])
        solve_riemann_problem(
            _nodes_[minus],
            _nodes_[plus],
            _fnodes_,
            "upwind",
            dim,
            idx,
            np.nan,
            using_cupy=using_cupy,
            timer=timer,
        )

        if len(active_dims) == 1:
            break

        timer is not None and timer.start("stencil_sweep", using_cupy)  # TIMER START
        if use_GL:
            integrate_gauss_legendre_face_nodes(_fnodes_, _F_out_, dim, active_dims, fv.p)
        else:
            integrate_transverse_nodes(_fnodes_, _F_out_, dim, active_dims, fv.p)
        timer is not None and timer.stop("stencil_sweep", using_cupy)  # TIMER STOP


def compute_flux_jvp(
    u: ArrayLike,
    vec: ArrayLike,
    jvp: ArrayLike,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
):
    """Apply the advection flux Jacobian, with zero velocity and dummy energy fluxes, to vec."""
    iv = idx("v" + dim)
    jvp[...] = u[iv] * vec + u * vec[iv]
    jvp[idx("v")] = 0.0
    jvp[idx("E")] = 0.0


def update_MUSCL_fluxes(
    _u_: ArrayLike,
    _F_: ArrayLike,
    _G_: ArrayLike,
    _H_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    hancock_dt: float = 0.0,
    timer: Optional[MultiTimer] = None,
):
    """Update advection fluxes with MUSCL, including the Hancock predictor when dt > 0."""
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    xp = cp if using_cupy else np
    na = xp.newaxis
    fv = fv_params
    if not fv.muscl_params.use_MUSCL:
        raise ValueError("update_MUSCL_fluxes requires a MUSCL scheme.")

    if fv.muscl_params.SED_params.use_SED:
        compute_alpha(_u_, _alpha_, active_dims, fv.muscl_params.SED_params.clip_zero_tol)

    slope_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}
    if hancock_dt > 0.0:
        for dim in active_dims:
            _slopes_ = xp.empty_like(_u_)
            compute_MUSCL_slopes(
                _u_,
                _alpha_,
                _slopes_,
                dim,
                active_dims,
                fv.muscl_params.MUSCL_limiter,
                fv.muscl_params.SED_params.use_SED,
            )
            slope_dict[dim] = _slopes_

        _predictor_q_ = _u_.copy()
        _jvp_ = xp.empty_like(_u_)
        for dim in active_dims:
            compute_flux_jvp(_u_, slope_dict[dim], _jvp_, idx, dim)
            _predictor_q_ -= 0.5 * _jvp_ * hancock_dt / getattr(mesh, "h" + dim)

    for dim in active_dims:
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim][..., na]
        if hancock_dt > 0.0:
            _slopes_ = slope_dict[dim]
            _left_face_ = _predictor_q_ - 0.5 * _slopes_
            _right_face_ = _predictor_q_ + 0.5 * _slopes_
            _faces_ = xp.concatenate([_left_face_[..., na], _right_face_[..., na]], axis=4)
        else:
            _faces_ = xp.empty((*_u_.shape, 2))
            reconstruct_MUSCL_faces(
                _u_,
                _alpha_,
                _faces_,
                dim,
                active_dims,
                fv.muscl_params.MUSCL_limiter,
                fv.muscl_params.SED_params.use_SED,
            )

        minus, plus = get_riemann_solver_slices(DIM_TO_AXIS[dim], 1, mesh.nghost)
        solve_riemann_problem(
            _faces_[minus],
            _faces_[plus],
            _F_out_,
            "upwind",
            dim,
            idx,
            np.nan,
            using_cupy=using_cupy,
            timer=timer,
        )


def update_advection_fluxes(
    _u_: ArrayLike,
    _F_: ArrayLike,
    _G_: ArrayLike,
    _H_: ArrayLike,
    _theta_: ArrayLike,
    _qcc_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    hancock_dt: float = 0.0,
    timer: Optional[MultiTimer] = None,
):
    """
    Update advection fluxes using WENO/Zhang-Shu or MUSCL reconstruction.

    _u_ contains ghost-padded density, prescribed velocities, and optional passive fields.
    Flux arrays have the same padded transverse dimensions as _u_ and n + 1 faces in their
    normal direction. _theta_ and _qcc_ are cell-shaped scratch arrays needed only for ZS;
    _alpha_ is cell-shaped scratch for ZS or MUSCL/SED. Inactive flux arrays may be empty.
    Passive fields use the same conservative advection equation as density, without any
    density weighting. Velocity fluxes are zero. hancock_dt is used only for MUSCL.
    """
    if fv_params.muscl_params.use_MUSCL:
        update_MUSCL_fluxes(
            _u_,
            _F_,
            _G_,
            _H_,
            _alpha_,
            idx,
            active_dims,
            mesh,
            fv_params,
            hancock_dt,
            timer,
        )
    else:
        update_weno_fluxes(
            _u_,
            _F_,
            _G_,
            _H_,
            _theta_,
            _qcc_,
            _alpha_,
            idx,
            active_dims,
            mesh,
            fv_params,
            timer,
        )


def compute_advection_dt(
    u: ArrayLike,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    CFL: float,
) -> float:
    """Compute CFL / max(sum(|v_d| / h_d)); stationary flow has no advective restriction."""
    xp = cp if CUPY_AVAILABLE and isinstance(u, cp.ndarray) else np
    sum_of_s_over_h = xp.zeros_like(u[idx("rho")])
    for dim in mesh.active_dims:
        sum_of_s_over_h += xp.abs(u[idx("v" + dim)]) / getattr(mesh, "h" + dim)
    max_speed = xp.max(sum_of_s_over_h).item()
    return CFL / max_speed if max_speed != 0.0 else np.inf


def compute_advection_nghost(fv_scheme: FV_SchemeParameters, ndim: int) -> int:
    """Return the ghost depth needed for advection reconstruction, limiting, and fluxes."""
    if fv_scheme.muscl_params.use_MUSCL:
        return (3 if fv_scheme.muscl_params.SED_params.use_SED else 1) + 1

    left_right_reach = conservative_interpolation.left_right(fv_scheme.p).shape[1] // 2
    transverse_reach = transverse_integration(fv_scheme.p).shape[1] // 2
    nghost = left_right_reach
    if fv_scheme.zhang_shu_params.use_ZS:
        nghost += 3 if fv_scheme.zhang_shu_params.SED_params.use_SED else 1
    nghost += max(1, transverse_reach) if fv_scheme.flux_quadrature == "transverse" else 1

    mood_cost = 0
    if fv_scheme.mood_params.use_MOOD:
        if fv_scheme.mood_params.NAD_params.use_NAD:
            mood_cost = 3 if fv_scheme.mood_params.NAD_params.SED_params.use_SED else 1
        if fv_scheme.mood_params.blend_troubles:
            mood_cost = 3
    return max(nghost, mood_cost)
