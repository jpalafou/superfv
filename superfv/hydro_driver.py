from typing import Dict, Literal, Optional, Tuple

import numpy as np

from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.step_history import MultiTimer

from .axes import DIM_TO_AXIS, XYZ_TUPLE
from .configs import BoundaryConditionParameters, FV_SchemeParameters, HydroParameters
from .finite_volume_driver import (
    Stencil,
    _stencil_cache,
    apply_fv_bc,
    apply_zhang_shu_limiter,
    get_interior_view,
    get_riemann_solver_slices,
    integrate_cell_averages,
    integrate_gauss_legendre_face_nodes,
    integrate_transverse_nodes,
    interpolate_cell_centers,
    interpolate_face_nodes,
    interpolate_interface_nodes,
)
from .hydro import cons_to_prim, prim_to_cs
from .mesh import UniformFiniteVolumeMesh
from .riemann_solvers import solve_riemann_problem
from .slope_limiting.muscl import compute_MUSCL_slopes, reconstruct_MUSCL_faces
from .slope_limiting.shock_detection import detect_shocks
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .stencils import (
    conservative_interpolation,
    finite_difference,
    transverse_integration,
)
from .sweep import stencil_sweep
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def _fv_detect_shocks(
    _q_: ArrayLike,
    _cs_: ArrayLike,
    _has_shock_: ArrayLike,
    primitives: bool,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    fv_params: FV_SchemeParameters,
):
    """
    Detect shocks in `_q_` and write the result to `_has_shock_`.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_q_, cp.ndarray) else np
    na = xp.newaxis

    if not fv_params.shock_detection_params.use_shock_detection:
        raise ValueError("Shock detection is not enabled in the provided FV scheme.")

    # Allocate temporary arrays
    _q_ref_ = _q_.copy()
    _eta_ = xp.empty_like(_q_)

    # Update `_q_ref_` with cs if detecting shocks from primitives or rho * cs otherwise
    if primitives:
        _q_ref_[idx("v")] = _cs_[na, ...]
    else:
        _q_ref_[idx("m")] = _q_[idx("rho", keepdims=True)] * _cs_[na, ...]

    detect_shocks(
        _q_, _q_ref_, _eta_, _has_shock_, active_dims, fv_params.shock_detection_params.eta_max
    )


def fv_cons_to_prim(
    u: ArrayLike,
    w: ArrayLike,
    idx: VariableIndexMap,
    hydro_params: HydroParameters,
    using_cupy: bool,
    timer: Optional[MultiTimer] = None,
):
    timer is not None and timer.start("primitive_conservative", using_cupy)  # TIMER START
    cons_to_prim(u, w, idx, hydro_params.gamma, hydro_params.isothermal, hydro_params.iso_cs)
    timer is not None and timer.stop("primitive_conservative", using_cupy)  # TIMER STOP


def update_hydro_workspace(
    u: ArrayLike,
    _u_: ArrayLike,
    _w_: ArrayLike,
    _w1_: ArrayLike,
    _has_shock_: ArrayLike,
    t: float,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    bc_params: BoundaryConditionParameters,
    hydro_params: HydroParameters,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the `_u_` and `_w_` arrays with ghost-cell-padded conservative and primitive finite
    volume averages, respectively.

    u: shape (nvars, nx, ny, nz) - Input array of conservative cell averages without ghost cells.
        Is not modified.
    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Array to which u is written with boundary
        conditions applied to ghost cells along active dimensions.
    _w_: _u_.shape - Array to which primitive cell averages are written with ghost cells according
        to the strategy specified by `fv_params.lazy_primitive_mode`.
    _w1_: _u_.shape - Array to which primitive cell averages are written with the second-order
        approximation. If `fv_params.p < 2` or `fv_params.lazy_primitive_mode == "full"`, this
        array is not written to and can be empty.
    _has_shock_: _u_[:1, ...].shape - Array to which shock detection results are written if shock
        detection and adaptive primitive mode are enabled. Otherwise, can be an empty array.
    t: Current simulation time, used for time-dependent boundary conditions.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    mesh: Mesh object containing information about the mesh and its dimensions.
    fv_params: Parameters for the finite volume scheme.
    bc_params: Parameters for the boundary conditions.
    hydro_params: Hydrodynamic parameters.
    timer: Optional[MultiTimer] = None - Timer object for performance measurement.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    xp = cp if using_cupy else np
    hp = hydro_params
    fv = fv_params
    active_dims = mesh.active_dims
    interior = get_interior_view(active_dims, mesh.nghost)

    # 0) Update interior conservative FV averages and apply BC
    timer is not None and timer.start("boundary_conditions", using_cupy)  # TIMER START
    _u_[interior] = u
    apply_fv_bc(_u_, idx, mesh, t, bc_params, hydro_params)
    timer is not None and timer.stop("boundary_conditions", using_cupy)  # TIMER STOP

    # Early escape for low-order scheme. The rest of the arrays are useless.
    if fv.p < 2:
        fv_cons_to_prim(_u_, _w_, idx, hp, using_cupy, timer)
        return

    # Allocate some temp arrays
    _qcc_ = xp.empty_like(_u_)

    # 1) Compute primitive cell-cenetered values
    interpolate_cell_centers(_u_, _qcc_, active_dims, fv.p, timer)
    fv_cons_to_prim(_qcc_, _qcc_, idx, hp, using_cupy, timer)

    # 2) Compute primitive finite-volume averages
    if fv.lazy_primitive_mode == "full":
        fv_cons_to_prim(_u_, _w_, idx, hp, using_cupy, timer)
        return

    fv_cons_to_prim(_u_, _w1_, idx, hp, using_cupy, timer)

    if fv.lazy_primitive_mode == "none":
        integrate_cell_averages(_qcc_, _w_, active_dims, fv.p, timer)

        # Ensure density is always transformed in the lazy way
        _w_[idx("rho"), ...] = _u_[idx("rho"), ...]
        return

    if fv.lazy_primitive_mode == "adaptive":
        integrate_cell_averages(_qcc_, _w_, active_dims, fv.p, timer)

        # Allocate a temp array for sound speed
        _cs_ = xp.empty_like(_u_[idx("rho")])  # TEMP ARRAY

        # Detect shocks and flag them in _has_shock_
        prim_to_cs(_w1_, _cs_, idx, hp.gamma, hp.isothermal, hp.iso_cs)
        _fv_detect_shocks(
            _u_ if fv.flux_recipe == "cons_lim_prim" else _w_,
            _cs_,
            _has_shock_,
            False if fv.flux_recipe == "cons_lim_prim" else True,
            idx,
            active_dims,
            fv,
        )

        # Flag PAD violations as shocks
        if fv.shock_detection_params.PAD_params.use_PAD:
            for v, (lb, ub) in fv.shock_detection_params.PAD_params.bounds.items():
                if lb is not None:
                    xp.maximum(_has_shock_, _w_[idx(v)] < lb, out=_has_shock_)
                if ub is not None:
                    xp.maximum(_has_shock_, _w_[idx(v)] > ub, out=_has_shock_)

        _w_[...] = xp.where(_has_shock_, _w1_, _w_)

        # Ensure density is always transformed in the lazy way
        _w_[idx("rho"), ...] = _u_[idx("rho"), ...]
        return


def reconstruct_centroid(
    _u_: ArrayLike,
    _w_: ArrayLike,
    _qcc_: ArrayLike,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    fv_params: FV_SchemeParameters,
    hydro_params: HydroParameters,
    timer: Optional[MultiTimer] = None,
):
    """
    Reconstruct the cell-centered primitive or conservative variables from the primitive or
    conservative cell averages, depending on the specified `flux_recipe`.

    Args:
        _u_: Conservative cell averages with ghost cells with shape (nvars, _nx_, _ny_, _nz_).
            Is not modified.
        _w_: Primitive cell averages with ghost cells with same shape as `_u_`. Is not modified.
        _qcc_: Array to which the reconstructed primitive cell-centered values are written. Has
            the same shape as `_u_` and `_w_`.
        idx: Index map for accessing variables in the arrays.
        active_dims: Active dimensions of the mesh.
        fv_params: Finite volume scheme parameters.
        hydro_params: Hydrodynamic parameters.
        timer: Optional timer for performance measurement.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)

    match fv_params.flux_recipe:
        case "cons_lim_prim":
            interpolate_cell_centers(_u_, _qcc_, active_dims, fv_params.p, timer)
        case "cons_prim_lim":
            interpolate_cell_centers(_u_, _qcc_, active_dims, fv_params.p, timer)
            fv_cons_to_prim(_qcc_, _qcc_, idx, hydro_params, using_cupy, timer)
        case "prim_prim_lim":
            interpolate_cell_centers(_w_, _qcc_, active_dims, fv_params.p, timer)
        case _:
            raise ValueError(f"Unknown flux_recipe: {fv_params.flux_recipe}")


def _get_n_nodes_per_face(ndim: int, fv_params: FV_SchemeParameters) -> int:
    if fv_params.flux_quadrature == "gauss_legendre":
        n_gauss_legendre = conservative_interpolation.n_gauss_legendre_nodes(fv_params.p)
        return n_gauss_legendre ** (ndim - 1)
    else:
        return 1


def enforce_positive_nodes(
    wj: ArrayLike, w1: ArrayLike, idx: VariableIndexMap, hp: HydroParameters
):
    if CUPY_AVAILABLE and isinstance(wj, cp.ndarray):
        enforce_positive_nodes_kernel_helper(wj, w1, idx, hp.rho_min, hp.P_min)
        return
    na = np.newaxis

    density = wj[idx("rho", keepdims=True)]
    pressure = wj[idx("P", keepdims=True)]

    nonpositive_nodes = np.logical_or(density < hp.rho_min, pressure < hp.P_min)
    wj[...] = np.where(nonpositive_nodes, w1[..., na], wj)


def add_viscuous_fluxes(
    _w_: ArrayLike,
    _f_: ArrayLike,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    nghost: int,
    hx: float,
    hy: float,
    hz: float,
    p: int,
    use_GL: bool,
    nu: float,
    Chi: float,
    nu_dye: float,
    timer: Optional[MultiTimer] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_w_, cp.ndarray)
    xp = cp if cupy else np
    transverse_first_derivative_stencil = _stencil_cache(
        Stencil.FINITE_DIFFERENCE_FIRST_DERIVATIVE, p, cupy
    )

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    # f has shape (nvars, nx + 1, ny + 2 * ng, nz + 2 * ng, 1)
    # w has shape (nvars, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng, 1)
    in1 = tuple(slice(nghost - 1, -nghost) if d == dim else slice(None) for d in XYZ_TUPLE)

    # interpolate continuous interface nodes using an even stencil
    _wf_ = xp.empty((*_w_.shape, 1))  # TEMP ARRAY
    interpolate_interface_nodes(_w_, _wf_, dim, active_dims, p, use_GL, timer=timer)

    rho = _wf_[idx("rho"), *in1]

    if nu > 0.0:
        vx = _wf_[idx("vx"), *in1]
        vy = _wf_[idx("vy"), *in1]
        vz = _wf_[idx("vz"), *in1]
        vxyzf = _wf_[idx("v"), ...]
        vxyz = _w_[idx("v"), ...]

        # assign dvdx9 as [dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz]
        dvdx9 = xp.zeros((9, *_w_.shape[1:4], 1))  # TEMP ARRAY
        for i, d in enumerate(XYZ_TUPLE):
            slc = slice(i * 3, (i + 1) * 3)
            h = {"x": hx, "y": hy, "z": hz}[d]
            if d not in active_dims:
                continue
            timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START
            if d == dim:
                interpolate_interface_nodes(
                    vxyz, dvdx9[slc, ...], d, active_dims, p, use_GL, first_derivative=True, h=h
                )
            else:
                stencil_sweep(vxyzf, transverse_first_derivative_stencil / h, dvdx9[slc, ...], d)
            timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP

        minus_nu_rho = -nu * rho  # TEMP ARRAY
        if dim == "x":
            Pi11 = 4 * dvdx9[0, *in1] - 2 * dvdx9[4, *in1] - 2 * dvdx9[8, *in1]  # TEMP ARRAY
            Pi11 *= minus_nu_rho / 3
            Pi12 = minus_nu_rho * (dvdx9[1, *in1] + dvdx9[3, *in1])  # TEMP ARRAY
            Pi13 = minus_nu_rho * (dvdx9[2, *in1] + dvdx9[6, *in1])  # TEMP ARRAY

            _f_[idx("mx"), ...] += Pi11
            _f_[idx("my"), ...] += Pi12
            _f_[idx("mz"), ...] += Pi13
            _f_[idx("E"), ...] += vx * Pi11 + vy * Pi12 + vz * Pi13
        elif dim == "y":
            Pi22 = -2 * dvdx9[0, *in1] + 4 * dvdx9[4, *in1] - 2 * dvdx9[8, *in1]  # TEMP ARRAY
            Pi22 *= minus_nu_rho / 3
            Pi12 = minus_nu_rho * (dvdx9[1, *in1] + dvdx9[3, *in1])  # TEMP ARRAY
            Pi23 = minus_nu_rho * (dvdx9[5, *in1] + dvdx9[7, *in1])  # TEMP ARRAY

            _f_[idx("mx"), ...] += Pi12
            _f_[idx("my"), ...] += Pi22
            _f_[idx("mz"), ...] += Pi23
            _f_[idx("E"), ...] += vx * Pi12 + vy * Pi22 + vz * Pi23
        elif dim == "z":
            Pi33 = -2 * dvdx9[0, *in1] - 2 * dvdx9[4, *in1] + 4 * dvdx9[8, *in1]  # TEMP ARRAY
            Pi33 *= minus_nu_rho / 3
            Pi13 = minus_nu_rho * (dvdx9[2, *in1] + dvdx9[6, *in1])  # TEMP ARRAY
            Pi23 = minus_nu_rho * (dvdx9[5, *in1] + dvdx9[7, *in1])  # TEMP ARRAY

            _f_[idx("mx"), ...] += Pi13
            _f_[idx("my"), ...] += Pi23
            _f_[idx("mz"), ...] += Pi33
            _f_[idx("E"), ...] += vx * Pi13 + vy * Pi23 + vz * Pi33
        else:
            raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    # thermal fluxes
    if Chi > 0.0:
        dTdx = xp.empty((*_w_[idx("rho", keepdims=True)].shape, 1))  # TEMP ARRAY
        h = {"x": hx, "y": hy, "z": hz}[dim]
        interpolate_interface_nodes(
            _w_[idx("P", keepdims=True), ...] / _w_[idx("rho", keepdims=True), ...],
            dTdx,
            dim,
            active_dims,
            p,
            use_GL,
            first_derivative=True,
            h=h,
            timer=timer,
        )
        _f_[idx("E"), ...] += -Chi * rho * dTdx[0, *in1]

    # dye concentration flux
    if nu_dye > 0.0:
        dcdx = xp.empty((*_w_[idx("rho", keepdims=True)].shape, 1))  # TEMP ARRAY
        h = {"x": hx, "y": hy, "z": hz}[dim]
        interpolate_interface_nodes(
            _w_[idx("dye", keepdims=True), ...],
            dcdx,
            dim,
            active_dims,
            p,
            use_GL,
            first_derivative=True,
            h=h,
            timer=timer,
        )
        _f_[idx("dye"), ...] += -rho * nu_dye * dcdx[0, *in1]


def update_weno_fluxes(
    _u_: ArrayLike,
    _w_: ArrayLike,
    _w1_: ArrayLike,
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
    hydro_params: HydroParameters,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the finite volume fluxes with a WENO reconstruction and the Zhang-Shu limiter.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Input array of conservative cell averages
        with ghost cells along active dimensions. Is not modified.
    _w_: _u_.shape - Input array of primitive cell averages with ghost cells. Should represent the
        same physical state as _u_. Is not modified.
    _w1_: _u_.shape - Input array of lazy primitive cell averages with ghost cells. Should
        represent the same physical state as _u_. Only used if `fv_params.p > 1` and
        `fv_params.lazy_primitive_mode != "full"`. Is not modified.
    _F_: shape (nvars, nx + 1, mesh._ny_, mesh._nz_) - Array to which the x-fluxes are written if
        "x" is in active_dims.
    _G_: shape (nvars, mesh._nx_, ny + 1, mesh._nz_) - Array to which the y-fluxes are written if
        "y" is in active_dims.
    _H_: shape (nvars, mesh._nx_, mesh._ny_, nz + 1) - Array to which the z-fluxes are written if
        "z" is in active_dims.
    _theta_: _u_.shape - Array to which the Zhang-Shu limiter values are written if enabled in
        the FV scheme. Otherwise, can be an empty array.
    _qcc_: _u_.shape - Scratch array for cell-centered values required if the Zhang-Shu limiter is
        enabled in the FV scheme. Otherwise, can be an empty array.
    _alpha_: _u_.shape - Array to which the smooth extrema detection values are written if enabled
        in the FV scheme. Must match _u_.shape if the Zhang-Shu limiter is enabled. Otherwise, can
        be an empty array.
    idx: Variable index map for the conserved and primitive variables.
    active_dims: Tuple of active spatial dimensions.
    mesh: Mesh object containing information about the mesh and its dimensions.
    fv_params: Parameters for the finite volume scheme.
    hydro_params: Hydrodynamic parameters.
    timer: MultiTimer object for timing various routines. If None, timing is not performed.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    xp = cp if using_cupy else np
    na = xp.newaxis
    fv = fv_params
    hp = hydro_params
    nghost = mesh.nghost
    use_GL = fv_params.flux_quadrature == "gauss_legendre"

    # 1) Interpolate nodes at each face
    _q_ = _w_ if fv.flux_recipe == "prim_prim_lim" else _u_

    node_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}
    n_nodes = _get_n_nodes_per_face(len(active_dims), fv)

    for dim in active_dims:
        _nodes_ = xp.empty(_w_.shape + (2 * n_nodes,))  # TEMP ARRAY
        interpolate_face_nodes(_q_, _nodes_, dim, active_dims, fv.p, use_GL, timer)

        if fv.flux_recipe == "cons_prim_lim":
            fv_cons_to_prim(_nodes_, _nodes_, idx, hp, using_cupy, timer)

        node_dict[dim] = _nodes_

    # Optionally apply slope-limiting to the nodes
    if fv.zhang_shu_params.use_ZS:
        # reconstruct cell center
        if fv.p > 1:
            reconstruct_centroid(_u_, _w_, _qcc_, idx, active_dims, fv, hp, timer)

        # a priori slope limiting
        apply_zhang_shu_limiter(
            _u_ if fv.flux_recipe == "cons_lim_prim" else _w_,
            _qcc_,
            node_dict["x"] if "x" in active_dims else np.array([]),
            node_dict["y"] if "y" in active_dims else np.array([]),
            node_dict["z"] if "z" in active_dims else np.array([]),
            _theta_,
            _alpha_,
            idx,
            False if fv.flux_recipe == "cons_lim_prim" else True,
            active_dims,
            fv.p,
            fv.zhang_shu_params,
            timer,
        )

    # 2) Solve Riemann problem at each face node and integrate to get fluxes
    for dim in active_dims:
        axis = DIM_TO_AXIS[dim]
        minus, plus = get_riemann_solver_slices(axis, n_nodes, nghost)

        _nodes_ = node_dict[dim]
        _left_nodes_ = _nodes_[minus]
        _right_nodes_ = _nodes_[plus]
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim]

        if fv.flux_recipe == "cons_lim_prim":
            fv_cons_to_prim(_nodes_, _nodes_, idx, hp, using_cupy, timer)
        if fv.positivity_guard:
            _wfallback_ = _w1_ if fv.p > 1 and fv.lazy_primitive_mode != "full" else _w_
            enforce_positive_nodes(_nodes_, _wfallback_, idx, hp)

        if len(active_dims) == 1:
            _fnodes_ = _F_out_[..., na]
        else:
            _fnodes_ = xp.empty_like(_left_nodes_)  # TEMP ARRAY

        solve_riemann_problem(
            _left_nodes_,
            _right_nodes_,
            _fnodes_,
            fv_params.riemann_solver,
            dim,
            idx,
            hp.gamma,
            hp.isothermal,
            hp.iso_cs,
            using_cupy,
            timer,
        )

        # Compute viscuous fluxes
        if hydro_params.diffusion:
            add_viscuous_fluxes(
                _w_,
                _fnodes_,
                idx,
                dim,
                active_dims,
                mesh.nghost,
                mesh.hx,
                mesh.hy,
                mesh.hz,
                fv.p,
                use_GL,
                hydro_params.nu,
                hydro_params.Chi,
                hydro_params.nu_dye,
                timer,
            )

        if len(active_dims) == 1:
            break

        timer is not None and timer.start("stencil_sweep", using_cupy)  # TIMER START
        if use_GL:
            integrate_gauss_legendre_face_nodes(_fnodes_, _F_out_, dim, active_dims, fv_params.p)
        else:
            integrate_transverse_nodes(_fnodes_, _F_out_, dim, active_dims, fv_params.p)
        timer is not None and timer.stop("stencil_sweep", using_cupy)  # TIMER STOP


def compute_flux_jvp(
    w: ArrayLike,
    vec: ArrayLike,
    jvp: ArrayLike,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
    primitives: bool = True,
):
    if not primitives:
        raise NotImplementedError("JVP for conservative variable fluxes not implemented yet.")

    iv = idx("v" + dim)

    jvp[idx("rho")] = w[iv] * vec[idx("rho")] + w[idx("rho")] * vec[iv]
    jvp[idx("vx")] = w[iv] * vec[idx("vx")]
    jvp[idx("vy")] = w[iv] * vec[idx("vy")]
    jvp[idx("vz")] = w[iv] * vec[idx("vz")]
    jvp[iv] += (1 / w[idx("rho")]) * vec[idx("P")]
    jvp[idx("P")] = (
        iso_cs**2 * jvp[idx("rho")]
        if isothermal
        else gamma * w[idx("P")] * vec[iv] + w[iv] * vec[idx("P")]
    )
    if "passives" in idx.group_var_map:
        jvp[idx("passives")] = w[iv] * vec[idx("passives")] + w[idx("passives")] * vec[iv]

    return


def update_MUSCL_fluxes(
    _u_: ArrayLike,
    _w_: ArrayLike,
    _F_: ArrayLike,
    _G_: ArrayLike,
    _H_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    hydro_params: HydroParameters,
    hancock_dt: float = 0.0,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the finite volume fluxes with a MUSCL-Hancock scheme.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Input array of conservative cell averages
        with ghost cells along active dimensions. Is not modified.
    _w_: _u_.shape - Input array of primitive cell averages with ghost cells. Should represent the
        same physical state as _u_. Is not modified.
    _F_: shape (nvars, nx + 1, mesh._ny_, mesh._nz_) - Array to which the x-fluxes are written if
        "x" is in active_dims.
    _G_: shape (nvars, mesh._nx_, ny + 1, mesh._nz_) - Array to which the y-fluxes are written if
        "y" is in active_dims.
    _H_: shape (nvars, mesh._nx_, mesh._ny_, nz + 1) - Array to which the z-fluxes are written if
        "z" is in active_dims.
    _alpha_: _u_.shape - Array to which the smooth extrema detection values are written if enabled
        in the FV scheme. Otherwise, can be an empty array.
    idx: Variable index map for the conserved and primitive variables.
    active_dims: Tuple of active spatial dimensions.
    mesh: Mesh object containing information about the mesh and its dimensions.
    fv_params: Parameters for the finite volume scheme.
    hydro_params: Hydrodynamic parameters.
    hancock_dt: Time step to use for MUSCL-Hancock predictor step. If nonpositive, the fluxes are
        computed with the pure MUSCL scheme in `update_MUSCL_fluxes`.
    timer: MultiTimer object for timing various routines. If None, timing is not performed.
    """
    using_cupy = CUPY_AVAILABLE and isinstance(_u_, cp.ndarray)
    xp = cp if using_cupy else np
    na = xp.newaxis
    fv = fv_params
    hp = hydro_params
    nghost = mesh.nghost

    if not fv.muscl_params.use_MUSCL:
        raise ValueError("update_fluxes_with_muscl_scheme should only be called for MUSCL schemes.")

    # 1) Compute slopes from either conservatives or primitives
    _q_ = _u_ if fv.flux_recipe == "cons_lim_prim" else _w_

    if fv.muscl_params.SED_params.use_SED:
        compute_alpha(_q_, _alpha_, active_dims, fv.muscl_params.SED_params.clip_zero_tol)

    # Optional predictor step for MUSCL-Hancock scheme
    _predictor_q_ = np.array([])
    slope_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}

    if hancock_dt > 0.0:
        for dim in active_dims:
            _slopes_ = xp.empty_like(_q_)  # TEMP ARRAY
            compute_MUSCL_slopes(
                _q_,
                _alpha_,
                _slopes_,
                dim,
                active_dims,
                fv.muscl_params.MUSCL_limiter,
                fv.muscl_params.SED_params.use_SED,
            )
            slope_dict[dim] = _slopes_

        # 2) Compute predictor step
        _predictor_q_ = _q_.copy()  # TEMP ARRAY
        _jvp_ = xp.empty_like(_q_)  # TEMP ARRAY

        for dim in active_dims:
            h = getattr(mesh, "h" + dim)

            compute_flux_jvp(
                _q_,
                slope_dict[dim],
                _jvp_,
                idx,
                dim,
                gamma=hydro_params.gamma,
                isothermal=hydro_params.isothermal,
                iso_cs=hydro_params.iso_cs,
                primitives=fv.flux_recipe != "cons_lim_prim",
            )
            _predictor_q_ -= 0.5 * _jvp_ * hancock_dt / h

    # 3) # Corrector step
    for dim in active_dims:
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim][..., na]

        # Reconstruct left and right faces
        if hancock_dt > 0.0:
            _slopes_ = slope_dict[dim]
            _left_face_ = _predictor_q_ - 0.5 * _slopes_  # TEMP ARRAY
            _right_face_ = _predictor_q_ + 0.5 * _slopes_  # TEMP ARRAY
            _faces_ = xp.concatenate(
                [_left_face_[..., na], _right_face_[..., na]], axis=4
            )  # TEMP ARRAY
        else:
            _faces_ = xp.empty((*_w_.shape[:4], 2))  # TEMP ARRAY
            reconstruct_MUSCL_faces(
                _q_,
                _alpha_,
                _faces_,
                dim,
                active_dims,
                fv.muscl_params.MUSCL_limiter,
                fv.muscl_params.SED_params.use_SED,
                fv.muscl_params.SED_params.clip_zero_tol,
            )

        # Ensure faces are positive and primitive
        if fv.flux_recipe == "cons_lim_prim":
            fv_cons_to_prim(_faces_, _faces_, idx, hp, using_cupy, timer)
        if fv.positivity_guard:
            enforce_positive_nodes(_faces_, _w_, idx, hp)

        # Solve Riemann problem at faces
        minus, plus = get_riemann_solver_slices(DIM_TO_AXIS[dim], 1, nghost)
        solve_riemann_problem(
            _faces_[minus],
            _faces_[plus],
            _F_out_,
            fv_params.riemann_solver,
            dim,
            idx,
            hp.gamma,
            hp.isothermal,
            hp.iso_cs,
            using_cupy,
            timer,
        )

        # Compute viscuous fluxes
        if hydro_params.diffusion:
            add_viscuous_fluxes(
                _w_,
                _F_out_,
                idx,
                dim,
                active_dims,
                mesh.nghost,
                mesh.hx,
                mesh.hy,
                mesh.hz,
                fv.p,
                False,
                hydro_params.nu,
                hydro_params.Chi,
                hydro_params.nu_dye,
                timer,
            )


def update_hydro_fluxes(
    _u_: ArrayLike,
    _w_: ArrayLike,
    _w1_: ArrayLike,
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
    hydro_params: HydroParameters,
    hancock_dt: float = 0.0,
    timer: Optional[MultiTimer] = None,
):
    """
    Update the finite volume fluxes with a WENO or MUSCL scheme.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Input array of conservative cell averages
        with ghost cells along active dimensions. Is not modified.
    _w_: _u_.shape - Input array of primitive cell averages with ghost cells. Should represent the
        same physical state as _u_. Is not modified.
    _w1_: _u_.shape - Input array of lazy primitive cell averages with ghost cells. Should
        represent the same physical state as _u_. Only used if `fv_params.p > 1` and
        `fv_params.lazy_primitive_mode != "full"`. Is not modified.
    _F_: shape (nvars, nx + 1, mesh._ny_, mesh._nz_) - Array to which the x-fluxes are written if
        "x" is in active_dims.
    _G_: shape (nvars, mesh._nx_, ny + 1, mesh._nz_) - Array to which the y-fluxes are written if
        "y" is in active_dims.
    _H_: shape (nvars, mesh._nx_, mesh._ny_, nz + 1) - Array to which the z-fluxes are written if
        "z" is in active_dims.
    _theta_: _u_.shape - Array to which the Zhang-Shu limiter values are written if enabled in
        the FV scheme. Otherwise, can be an empty array.
    _qcc_: _u_.shape - Scratch array for cell-centered values required if the Zhang-Shu limiter is
        enabled in the FV scheme. Otherwise, can be an empty array.
    _alpha_: _u_.shape - Array to which the smooth extrema detection values are written if enabled
        in the FV scheme. Must match _u_.shape if the Zhang-Shu limiter is enabled. Otherwise, can
        be an empty array.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    active_dims: Tuple of active spatial dimensions.
    mesh: Mesh object containing information about the mesh and its dimensions.
    fv_params: Parameters for the finite volume scheme.
    hydro_params: Hydrodynamic parameters.
    hancock_dt: Time step to use for MUSCL-Hancock predictor step. If 0, the predictor step is
        skipped. Is ignored if the FV scheme is not a MUSCL scheme.
    timer: MultiTimer object for timing various routines. If None, timing is not performed.
    """
    if fv_params.muscl_params.use_MUSCL:
        update_MUSCL_fluxes(
            _u_,
            _w_,
            _F_,
            _G_,
            _H_,
            _alpha_,
            idx,
            active_dims,
            mesh,
            fv_params,
            hydro_params,
            hancock_dt,
            timer,
        )
    else:
        update_weno_fluxes(
            _u_,
            _w_,
            _w1_,
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
            hydro_params,
            timer,
        )


def compute_hydro_dt(
    u: ArrayLike,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    hydro_params: HydroParameters,
) -> float:
    """
    Compute the finite volume time step.

    u: shape (nvars, mesh.nx, mesh.ny, mesh.nz) - Input array of conservative cell averages
        without ghost cells. Is not modified.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    mesh: Mesh object containing information about the mesh and its dimensions.
    hydro_params: Hydrodynamic parameters.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(u, cp.ndarray) else np
    hp = hydro_params

    w = xp.empty_like(u)  # TEMP ARRAY
    cs = xp.empty_like(u[idx("rho")])  # TEMP ARRAY
    sum_of_s_over_h = xp.zeros_like(cs)  # TEMP ARRAY

    cons_to_prim(u, w, idx, hp.gamma, hp.isothermal, hp.iso_cs)
    prim_to_cs(w, cs, idx, hp.gamma, hp.isothermal, hp.iso_cs)

    for dim, h in zip(["x", "y", "z"], [mesh.hx, mesh.hy, mesh.hz]):
        if dim not in mesh.active_dims:
            continue
        s = xp.abs(w[idx("v" + dim)]) + cs
        sum_of_s_over_h += s / h

    max_speed = xp.max(sum_of_s_over_h).item()
    dt = hp.CFL / max_speed

    if hp.diffusion:
        inv_h2 = sum(1.0 / getattr(mesh, f"h{dim}") ** 2 for dim in mesh.active_dims)
        dt_nu = hp.CFL / (2.0 * hp.nu * inv_h2)
        dt = min(dt, dt_nu)

    return dt


def compute_hydro_nghost(fv_scheme: FV_SchemeParameters, ndim: int, viscosity: bool = False) -> int:
    """
    Returns the number of ghost cells required in the padded arrays of the HydroSolver based on
    the provided `fv_scheme`.
    """
    left_right_reach = conservative_interpolation.left_right(fv_scheme.p).shape[1] // 2
    cell_center_reach = conservative_interpolation.cell_center(fv_scheme.p).shape[1] // 2
    interface_reach = conservative_interpolation.interface(fv_scheme.p).shape[1] // 2
    in_deriv_reach = (
        conservative_interpolation.interface_first_derivative(fv_scheme.p).shape[1] // 2
    )
    first_derivative_reach = finite_difference.first_derivative(fv_scheme.p).shape[1] // 2
    transverse_reach = transverse_integration(fv_scheme.p).shape[1] // 2

    # MUSCL scheme early escape
    if fv_scheme.muscl_params.use_MUSCL:
        face_node_cost = 3 if fv_scheme.muscl_params.SED_params.use_SED else 1
        riemann_cost = 1
        return face_node_cost + riemann_cost

    # Cost of update_hydro_workspace
    match fv_scheme.lazy_primitive_mode:
        case "none":
            w_workspace_setup_cost = cell_center_reach + transverse_reach
        case "adaptive":
            w_workspace_setup_cost = cell_center_reach + transverse_reach + 2
        case "full":
            w_workspace_setup_cost = 0

    # Cost of interpolating primitive node faces
    face_node_cost = left_right_reach
    if fv_scheme.flux_recipe == "prim_prim_lim":
        face_node_cost += w_workspace_setup_cost
    nghost = face_node_cost

    # Zhang-Shu limiter cost
    ZS_cost = 0
    if fv_scheme.zhang_shu_params.use_ZS:
        ZS_cost = 3 if fv_scheme.zhang_shu_params.SED_params.use_SED else 1
        if fv_scheme.flux_recipe == "cons_prim_lim":
            ZS_cost = max(ZS_cost, w_workspace_setup_cost)
    nghost += ZS_cost

    # Flux integral cost
    riemann_cost = 1
    if fv_scheme.flux_quadrature == "transverse":
        transverse_cost = transverse_reach
    else:
        transverse_cost = 0
    flux_integral_cost = max(riemann_cost, transverse_cost)
    nghost += flux_integral_cost

    # Viscosity cost
    if viscosity:
        interface_cost = max(cell_center_reach, interface_reach)
        in_deriv_cost = max(cell_center_reach, in_deriv_reach)

        viscosity_cost = w_workspace_setup_cost
        viscosity_cost += max(interface_cost, in_deriv_cost)
        viscosity_cost += first_derivative_reach  # Riemann solve
        viscosity_cost += flux_integral_cost
    else:
        viscosity_cost = 0
    nghost = max(nghost, viscosity_cost)

    # MOOD cost
    mood_cost = 0
    if fv_scheme.mood_params.use_MOOD:
        if fv_scheme.mood_params.NAD_params.use_NAD:
            mood_cost = 1
            if fv_scheme.mood_params.NAD_params.SED_params.use_SED:
                mood_cost = 3
        if fv_scheme.mood_params.blend_troubles:
            mood_cost = 3
    nghost = max(nghost, mood_cost)

    return nghost


if CUPY_AVAILABLE:
    enforce_positive_nodes_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void enforce_positive_nodes_kernel(
            double* __restrict__ wj,
            const double* __restrict__ w1,
            const double rho_min,
            const double P_min,
            const int rho_idx,
            const int P_idx,
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int ninterps
        ){
            // wj   (nvars, nx, ny, nz, ninterps)
            // w1   (nvars, nx, ny, nz)

            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long nxyzterps = (long long) nx * ny * nz * ninterps;

            for (long long ixyzterp = tid; ixyzterp < nxyzterps; ixyzterp += stride) {
                long long t = ixyzterp;
                int j = (int)(t % ninterps); t /= ninterps;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;

                long long irho = ((((long long)rho_idx * nx + ix) * ny + iy) * nz + iz)
                    * ninterps + j;
                long long iP = ((((long long)P_idx * nx + ix) * ny + iy) * nz + iz)
                    * ninterps + j;

                if (wj[irho] < rho_min || wj[iP] < P_min) {
                    for (int iv = 0; iv < nvars; iv++) {
                        long long i = ((((long long)iv * nx + ix) * ny + iy) * nz + iz);
                        wj[i * ninterps + j] = w1[i];
                    }
                }
            }
        }
        """,
        name="enforce_positive_nodes_kernel",
    )

    def enforce_positive_nodes_kernel_helper(
        wj: ArrayLike, w1: ArrayLike, idx: VariableIndexMap, rho_min: float, P_min: float
    ):
        if not wj.flags.c_contiguous or not w1.flags.c_contiguous:
            raise ValueError("Input arrays must be C-contiguous.")
        if wj.dtype != cp.float64 or w1.dtype != cp.float64:
            raise ValueError("Input arrays must be of type float64.")
        if wj.ndim != 5 or w1.ndim != 4 or wj.shape[:4] != w1.shape:
            raise ValueError("Input arrays must have the correct number of dimensions.")

        nvars, nx, ny, nz, ninterps = wj.shape
        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (nx * ny * nz * ninterps + threads_per_block - 1) // threads_per_block

        enforce_positive_nodes_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (
                wj,
                w1,
                rho_min,
                P_min,
                idx("rho"),
                idx("P"),
                nvars,
                nx,
                ny,
                nz,
                ninterps,
            ),
        )
