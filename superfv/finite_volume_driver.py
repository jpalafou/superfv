from enum import Enum
from functools import lru_cache
from typing import Dict, Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS, XYZ_TUPLE
from superfv.boundary_conditions import apply_bc
from superfv.slope_limiting.muscl import (
    MUSCL_SlopeLimiter,
    compute_MUSCL_slopes,
    compute_PP2D_slopes,
)
from superfv.tools.slicing import crop, merge_slices

from .configs import (
    BoundaryConditionParameters,
    FluxQuadrature,
    FluxRecipe,
    FV_SchemeParameters,
    HydroParameters,
    LazyPrimitiveMode,
    ZhangShuParameters,
)
from .hydro import cons_to_prim, prim_to_cs
from .mesh import UniformFiniteVolumeMesh
from .quadrature import perform_quadrature
from .riemann_solvers import solve_riemann_problem
from .slope_limiting import compute_dmp
from .slope_limiting.shock_detection import detect_shocks
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .slope_limiting.zhang_and_shu import compute_theta
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


class FV_Stencil(Enum):
    CONSERVATIVE_INTERPOLATION_CENTER = 0
    CONSERVATIVE_INTERPOLATION_LEFT_RIGHT = 1
    CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE = 2
    CONSERVATIVE_INTERPOLATION_INTERFACE = 3
    CONSERVATIVE_INTERPOLATION_INTERFACE_FIRST_DERIVATIVE = 4
    TRANSVERSE_INTEGRATION = 5
    FINITE_DIFFERENCE_FIRST_DERIVATIVE = 6


@lru_cache
def _stencil_cache(interpolation: FV_Stencil, p: int, cupy: bool = False):
    match interpolation:
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER:
            stencil = conservative_interpolation.cell_center(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_LEFT_RIGHT:
            stencil = conservative_interpolation.left_right(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE:
            stencil = conservative_interpolation.gauss_legendre_nodes(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_INTERFACE:
            stencil = conservative_interpolation.interface(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_INTERFACE_FIRST_DERIVATIVE:
            stencil = conservative_interpolation.interface_first_derivative(p)
        case FV_Stencil.TRANSVERSE_INTEGRATION:
            stencil = transverse_integration(p)
        case FV_Stencil.FINITE_DIFFERENCE_FIRST_DERIVATIVE:
            stencil = finite_difference.first_derivative(p)

    if CUPY_AVAILABLE and cupy:
        return cp.asarray(stencil)
    else:
        return stencil


@lru_cache
def _gauss_legendre_weights_cache(p: int, ndim: int, cupy: bool = False):
    weights = conservative_interpolation.gauss_legendre_weights(p, ndim - 1)

    if CUPY_AVAILABLE and cupy:
        return cp.asarray(weights)
    else:
        return weights


def interpolate_cell_centers(
    _q_: ArrayLike,
    _qcc_: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
):
    """
    Interpolate finite volume cell centers from `_q_` and write them to `_qcc_`.
    """
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    weights = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER, p, cupy)
    na = cp.newaxis if cupy else np.newaxis
    ndim = len(active_dims)

    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape != _qcc_.shape:
        raise ValueError("_q_ and _qcc_ must have the same shape.")

    if p < 2:
        _qcc_[...] = _q_
        return

    if ndim == 1:
        stencil_sweep(_q_[..., na], weights, _qcc_[..., na], active_dims[0])
        return
    elif ndim == 2:
        _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

        stencil_sweep(_q_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qcc_[..., na], active_dims[1])
        return
    elif ndim == 3:
        _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])
        _qtemp2_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

        stencil_sweep(_q_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _qcc_[..., na], active_dims[2])
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def integrate_cell_averages(
    _qcc_: ArrayLike,
    _q_: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
):
    """
    Interpolate finite volume cell averages from `_qcc_` and write them to `_q_`.
    """
    cupy = CUPY_AVAILABLE and isinstance(_qcc_, cp.ndarray)
    weights = _stencil_cache(FV_Stencil.TRANSVERSE_INTEGRATION, p, cupy)
    na = cp.newaxis if cupy else np.newaxis
    ndim = len(active_dims)

    if _qcc_.ndim != 4:
        raise ValueError("_qcc_ must be 4D.")
    if _qcc_.shape != _q_.shape:
        raise ValueError("_qcc_ and _q_ must have the same shape.")

    if p < 2:
        _q_[...] = _qcc_
        return

    if ndim == 1:
        stencil_sweep(_qcc_[..., na], weights, _q_[..., na], active_dims[0])
        return
    elif ndim == 2:
        _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

        stencil_sweep(_qcc_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _q_[..., na], active_dims[1])
        return
    elif ndim == 3:
        _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])
        _qtemp2_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

        stencil_sweep(_qcc_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _q_[..., na], active_dims[2])
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def interpolate_face_nodes(
    _q_: ArrayLike,
    _qj_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    gauss_legendre: bool = False,
):
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    lr_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_LEFT_RIGHT, p, cupy)
    na = cp.newaxis if cupy else np.newaxis
    ndim = len(active_dims)

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape[:4] != _qj_.shape[:4]:
        raise ValueError("The first 4 dimensions of _q_ and _qj_ must match.")

    if p < 1:
        if _qj_.shape[4] != 2:
            raise ValueError("The 5th dimension of _qj_ must be 2 for 1D interpolation.")
        _qj_[...] = _q_[..., na]
        return

    base_shape = _q_.shape

    if ndim == 1:
        if gauss_legendre:
            raise ValueError("No Gauss-Legendre face nodes in 1D.")
        if _qj_.shape[4] != 2:
            raise ValueError("The 5th dimension of _qj_ must be 2 for 1D interpolation.")
        stencil_sweep(_q_[..., na], lr_stencil, _qj_, dim)
        return

    if gauss_legendre:
        gl_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE, p, cupy)
        ngl = gl_stencil.shape[0]

        if ndim == 2:
            if _qj_.shape[4] != 2 * ngl:
                raise ValueError(
                    "The 5th dimension of _qj_ must be 2 two times the number of GL nodes."
                )

            _qtemp1_ = cp.empty((*base_shape, 2)) if cupy else np.empty((*base_shape, 2))

            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_[..., na], lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qj_, trans_dim)
            return
        elif ndim == 3:
            if _qj_.shape[4] != 2 * ngl**2:
                raise ValueError(
                    "The 5th dimension of _qj_ must be 2 two times the number of GL nodes squared."
                )

            _qtemp1_ = cp.empty((*base_shape, 2)) if cupy else np.empty((*base_shape, 2))
            _qtemp2_ = (
                cp.empty((*base_shape, 2 * ngl)) if cupy else np.empty((*base_shape, 2 * ngl))
            )

            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_[..., na], lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, gl_stencil, _qj_, trans_dim2)
            return
    else:
        cc_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER, p, cupy)
        if _qj_.shape[4] != 2:
            raise ValueError(
                "The 5th dimension of _qj_ must be 2 for transverse integration in 2D or 3D."
            )

        if ndim == 2:
            _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_[..., na], cc_stencil, _qtemp1_, trans_dim)
            stencil_sweep(_qtemp1_, lr_stencil, _qj_, dim)
            return
        elif ndim == 3:
            _qtemp1_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])
            _qtemp2_ = cp.empty_like(_q_[..., na]) if cupy else np.empty_like(_q_[..., na])

            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_[..., na], cc_stencil, _qtemp1_, trans_dim2)
            stencil_sweep(_qtemp1_, cc_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, lr_stencil, _qj_, dim)
            return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def interpolate_interface_nodes(
    _q_: ArrayLike,
    _qf_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    gauss_legendre: bool = False,
):
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    xp = cp if cupy else np
    interface_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_INTERFACE, p, cupy)
    center_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER, p, cupy)
    na = xp.newaxis
    ndim = len(active_dims)

    if gauss_legendre:
        raise NotImplementedError("Gauss-Legendre interface nodes are not implemented yet.")
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape[:4] != _qf_.shape[:4]:
        raise ValueError("The first 4 dimensions of _q_ and _qf_ must match.")
    if _qf_.shape[4] != 1:
        raise ValueError("The 5th dimension of _qf_ must be 1 for interface nodes.")

    if ndim == 1:
        stencil_sweep(_q_[..., na], interface_stencil, _qf_, dim)
        return

    if ndim == 2:
        trans_dim = [d for d in active_dims if d != dim][0]
        _qtemp_ = xp.empty((*_q_.shape, 1))
        stencil_sweep(_q_[..., na], interface_stencil, _qtemp_, dim)
        stencil_sweep(_qtemp_, center_stencil, _qf_, trans_dim)
        return

    if ndim == 3:
        trans_dims = [d for d in active_dims if d != dim]
        trans_dim1 = trans_dims[0]
        trans_dim2 = trans_dims[1]
        _qtemp1_ = xp.empty((*_q_.shape, 1))
        _qtemp2_ = xp.empty((*_q_.shape, 1))
        stencil_sweep(_q_[..., na], interface_stencil, _qtemp1_, dim)
        stencil_sweep(_qtemp1_, center_stencil, _qtemp2_, trans_dim1)
        stencil_sweep(_qtemp2_, center_stencil, _qf_, trans_dim2)
        return


def integrate_gauss_legendre_face_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
):
    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    ndim = len(active_dims)
    weights = _gauss_legendre_weights_cache(p, ndim, cupy)

    if ndim == 1:
        raise ValueError("No Gauss-Legendre face nodes in 1D.")
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _qj_.ndim != 5:
        raise ValueError("_qj_ must be 5D.")
    if _qF_.ndim != 4:
        raise ValueError("_qF_ must be 4D.")
    if _qj_.shape[:4] != _qF_.shape[:4]:
        raise ValueError("The first 4 dimensions of _qj_ and _qF_ must match.")

    perform_quadrature(_qj_, weights, _qF_)


def integrate_transverse_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
):
    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    stencil = _stencil_cache(FV_Stencil.TRANSVERSE_INTEGRATION, p, cupy)
    na = cp.newaxis if cupy else np.newaxis
    ndim = len(active_dims)

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _qj_.ndim != 5 and _qj_.shape[4] == 1:
        raise ValueError("_qj_ must be 5D with the 5th dimension equal to 1.")
    if _qF_.shape != _qj_.shape[:4]:
        raise ValueError("The shape of _qF_ must match the first 4 dimensions of _qj_.")

    if p < 2:
        _qF_[...] = _qj_[..., 0]
        return

    if ndim == 1:
        raise ValueError("Cannot integrate transverse face nodes in 1D.")
    if ndim == 2:
        trans_dim = [d for d in active_dims if d != dim][0]
        stencil_sweep(_qj_, stencil, _qF_[..., na], trans_dim)
        return
    elif ndim == 3:
        _qtemp_ = cp.empty_like(_qj_) if cupy else np.empty_like(_qj_)

        trans_dims = [d for d in active_dims if d != dim]
        trans_dim1 = trans_dims[0]
        trans_dim2 = trans_dims[1]
        stencil_sweep(_qj_, stencil, _qtemp_, trans_dim1)
        stencil_sweep(_qtemp_, stencil, _qF_[..., na], trans_dim2)
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def apply_fv_bc(
    _u_: ArrayLike,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    t: float,
    bc_params: BoundaryConditionParameters,
):
    """
    Update the ghost-cell-padded array of conservative cell averages `_u_` in-place with the
    specified boundary conditions.
    """
    apply_bc(
        _u_,
        mesh.nghost,
        bc_params.bcx,
        bc_params.bcy,
        bc_params.bcz,
        bc_params.bcx_callable_lower,
        bc_params.bcx_callable_upper,
        bc_params.bcy_callable_lower,
        bc_params.bcy_callable_upper,
        bc_params.bcz_callable_lower,
        bc_params.bcz_callable_upper,
        idx,
        mesh,
        t,
        bc_params.sampling_p,
    )


@lru_cache(maxsize=None)
def _get_interior_view(
    active_dims: Tuple[Literal["x", "y", "z"], ...], nghost: int, ndim: int = 4
) -> Tuple[slice, ...]:
    return merge_slices(
        *[crop(DIM_TO_AXIS[dim], (nghost, -nghost), ndim=ndim) for dim in active_dims]
    )


def get_interior_view(
    active_dims: Tuple[Literal["x", "y", "z"], ...], nghost: int, ndim: int = 4
) -> Tuple[slice, ...]:
    """
    Get a view of the interior, excluding ghost cells, for an array of shape
    (nvars, _nx_, _ny_, _nz_) or (nvars, _nx_, _ny_, _nz_, ninterps).
    """
    return _get_interior_view(active_dims, nghost, ndim)


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
    if not fv_params.shock_detection_params.use_shock_detection:
        raise ValueError("Shock detection is not enabled in the provided FV scheme.")

    # Allocate temporary arrays
    _q_ref_ = _q_.copy()
    if CUPY_AVAILABLE and isinstance(_q_, cp.ndarray):
        _eta_ = cp.empty_like(_q_)
    else:
        _eta_ = np.empty_like(_q_)

    # Update `_q_ref_` with cs if detecting shocks from primitives or rho * cs otherwise
    for dim in active_dims:
        _q_ref_[idx("v" + dim)] = _cs_ if primitives else _q_[idx("rho")] * _cs_

    detect_shocks(
        _q_, _q_ref_, _eta_, _has_shock_, active_dims, fv_params.shock_detection_params.eta_max
    )


def update_fv_workspace(
    u: ArrayLike,
    _u_: ArrayLike,
    _w_: ArrayLike,
    _has_shock_: ArrayLike,
    t: float,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    fv_params: FV_SchemeParameters,
    bc_params: BoundaryConditionParameters,
    hydro_params: HydroParameters,
):
    """
    Update the `_u_` and `_w_` arrays with ghost-cell-padded conservative and primitive finite
    volume averages, respectively.

    u: shape (nvars, nx, ny, nz) - Input array of conservative cell averages without ghost cells.
        Is not modified.
    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Array to which u is written with boundary
        conditions applied to ghost cells along active dimensions.
    _w_: _u_.shape - Array to which primitive cell averages are written with ghost cells.
    _has_shock_: _u_[:1, ...].shape - Array to which shock detection results are written if shock
        detection and adaptive primitive mode are enabled. Otherwise, can be an empty array.
    t: Current simulation time, used for time-dependent boundary conditions.
    idx: VariableIndexMap for indexing into the variable dimension of the arrays.
    mesh: Mesh object containing information about the mesh and its dimensions.
    fv_params: Parameters for the finite volume scheme.
    bc_params: Parameters for the boundary conditions.
    hydro_params: Hydrodynamic parameters.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_u_, cp.ndarray) else np
    hp = hydro_params
    fv = fv_params
    active_dims = mesh.active_dims
    interior = get_interior_view(active_dims, mesh.nghost)

    # 0) Update interior conservative FV averages and apply BC
    _u_[interior] = u
    apply_fv_bc(_u_, idx, mesh, t, bc_params)

    # Early escape for low-order scheme. The rest of the arrays are useless.
    if fv.p < 2:
        cons_to_prim(_u_, _w_, idx, hp.gamma, hp.isothermal, hp.iso_cs)
        return

    # Allocate some temp arrays
    _qcc_ = xp.empty_like(_u_)

    # 1) Compute primitive cell-cenetered values
    interpolate_cell_centers(_u_, _qcc_, active_dims, fv.p)
    cons_to_prim(_qcc_, _qcc_, idx, hp.gamma, hp.isothermal, hp.iso_cs)

    # 2) Compute primitive finite-volume averages
    if fv.lazy_primitive_mode == LazyPrimitiveMode.FULL:
        cons_to_prim(_u_, _w_, idx, hp.gamma, hp.isothermal, hp.iso_cs)
        return

    if fv.lazy_primitive_mode == LazyPrimitiveMode.NONE:
        integrate_cell_averages(_qcc_, _w_, active_dims, fv.p)

        # Ensure density is always transformed in the lazy way
        _w_[idx("rho"), ...] = _u_[idx("rho"), ...]
        return

    if fv.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE:
        integrate_cell_averages(_qcc_, _w_, active_dims, fv.p)

        # Allocate some more temp arrays
        _w1_ = xp.empty_like(_u_)
        _cs_ = xp.empty_like(_u_[idx("rho")])

        # Get lazy primitives
        cons_to_prim(_u_, _w1_, idx, hp.gamma, hp.isothermal, hp.iso_cs)

        # Detect shocks and flag them in _has_shock_
        prim_to_cs(_w1_, _cs_, idx, hp.gamma, hp.isothermal, hp.iso_cs)
        _fv_detect_shocks(
            _u_ if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _w_,
            _cs_,
            _has_shock_,
            False if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM else True,
            idx,
            active_dims,
            fv,
        )

        # Flag PAD violations as shocks
        if fv.mood_params.PAD_params.use_PAD:
            raise NotImplementedError(
                "Adaptive primitive mode with MOOD PAD is not implemented yet."
            )
        if fv.zhang_shu_params.PAD_params.use_PAD:
            for v, (lb, ub) in fv.zhang_shu_params.PAD_params.bounds.items():
                if lb is not None:
                    xp.maximum(_has_shock_, _w_[idx(v)] < lb, out=_has_shock_)
                if ub is not None:
                    xp.maximum(_has_shock_, _w_[idx(v)] > ub, out=_has_shock_)

        _w_[...] = xp.where(_has_shock_, _w_, _w1_)

        # Ensure density is always transformed in the lazy way
        _w_[idx("rho"), ...] = _u_[idx("rho"), ...]
        return


def _apply_zhang_shu_limiter_to_node_array(nodes: ArrayLike, fallback: ArrayLike, theta: ArrayLike):
    xp = cp if CUPY_AVAILABLE and isinstance(nodes, cp.ndarray) else np
    na = xp.newaxis

    xp.subtract(nodes, fallback[..., na], out=nodes)  # wj - w
    xp.multiply(nodes, theta[..., na], out=nodes)  # theta * (wj - w)
    xp.add(nodes, fallback[..., na], out=nodes)  # theta * (wj - w) + w


def apply_zhang_shu_limiter(
    _q_: ArrayLike,
    _x_nodes_: ArrayLike,
    _y_nodes_: ArrayLike,
    _z_nodes_: ArrayLike,
    _qcc_: ArrayLike,
    _theta_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    primitives: bool,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    params: ZhangShuParameters,
):
    xp = cp if CUPY_AVAILABLE and isinstance(_q_, cp.ndarray) else np
    na = xp.newaxis

    # Validate input
    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape != _qcc_.shape:
        raise ValueError("_q_ and _qcc_ must have the same shape.")
    if _theta_.shape != _q_.shape:
        raise ValueError("_theta_ must have the same shape as _q_.")
    if _alpha_.shape != _q_.shape:
        raise ValueError("_alpha_ must have the same shape as _q_.")

    # Allocate temporary arrays
    _M_ = xp.empty_like(_q_)
    _m_ = xp.empty_like(_q_)
    _Mj_ = xp.empty_like(_q_)
    _mj_ = xp.empty_like(_q_)

    # Gather node arrays into dict
    node_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {
        dim: nodes
        for dim, nodes in zip(("x", "y", "z"), (_x_nodes_, _y_nodes_, _z_nodes_))
        if dim in active_dims
    }

    # 1) Gather all node arrays to get nodal minima and maxima
    if p > 1:
        interpolate_cell_centers(_q_, _qcc_, active_dims, p)
        _qj_ = xp.concatenate([_qcc_[..., na]] + [node_dict[dim] for dim in active_dims], axis=4)
    else:
        _qj_ = xp.concatenate([node_dict[dim] for dim in active_dims], axis=4)

    # 2) Update _M_ and _m_ with discrete maximum principle
    compute_dmp(_q_, _M_, _m_, active_dims, params.include_corners)

    # 3) Update _Mj_ and _mj_ with nodal maxima and minima and _theta_ with the Zhang-Shu limiter
    compute_theta(
        _q_, _qj_, _M_, _m_, _Mj_, _mj_, _theta_, params.theta_denom_tol
    )  # update _Mj_, _mj_, and _theta_

    # 4) Relax _theta_ with smooth extrema detection and omit variables from limiting
    if params.SED_params.use_SED:
        if _alpha_ is None:
            raise ValueError("_alpha_ must be provided if detect_smooth_extrema is True")
        compute_alpha(_q_, _alpha_, active_dims, params.SED_params.clip_zero_tol)

        if params.PAD_params.use_PAD and primitives:
            _physical_ = xp.ones_like(_alpha_[0], dtype=bool)  # TEMP ARRAY
            for v, (lb, ub) in params.PAD_params.bounds.items():
                if ub is not None:
                    _physical_ &= _Mj_[idx(v)] <= ub
                if lb is not None:
                    _physical_ &= _mj_[idx(v)] >= lb
            _alpha_ *= _physical_
        xp.maximum(_theta_, _alpha_ >= 1, out=_theta_)

    if "omit_ZS" in idx.group_var_map:
        _theta_[idx("omit_ZS")] = 1.0

    # Apply limiter to node arrays
    if "x" in active_dims:
        _apply_zhang_shu_limiter_to_node_array(_x_nodes_, _q_, _theta_)
    if "y" in active_dims:
        _apply_zhang_shu_limiter_to_node_array(_y_nodes_, _q_, _theta_)
    if "z" in active_dims:
        _apply_zhang_shu_limiter_to_node_array(_z_nodes_, _q_, _theta_)


def _get_n_nodes_per_face(ndim: int, fv_params: FV_SchemeParameters) -> int:
    if fv_params.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
        n_gauss_legendre = conservative_interpolation.n_gauss_legendre_nodes(fv_params.p)
        return n_gauss_legendre ** (ndim - 1)
    else:
        return 1


def _positivity_guard(wj: ArrayLike, w1: ArrayLike, idx: VariableIndexMap, hp: HydroParameters):
    xp = cp if CUPY_AVAILABLE and isinstance(wj, cp.ndarray) else np
    na = xp.newaxis

    density = wj[idx("rho")]
    pressure = wj[idx("P")]

    density[...] = xp.where(density < hp.rho_min, w1[idx("rho"), ..., na], density)
    pressure[...] = xp.where(pressure < hp.P_min, w1[idx("P"), ..., na], pressure)


@lru_cache(maxsize=None)
def _get_riemann_solver_slices(
    axis: int, n_nodes_per_face: int, n_ghost: int
) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    # Get slice left of interface
    minus = crop(4, (n_nodes_per_face, 2 * n_nodes_per_face), ndim=5)  # Cell right face state
    minus = merge_slices(minus, crop(axis, (n_ghost - 1, -n_ghost), ndim=5))

    # Get slice right of interface
    plus = crop(4, (None, n_nodes_per_face), ndim=5)  # Cell left face state
    plus = merge_slices(plus, crop(axis, (n_ghost, -n_ghost + 1), ndim=5))

    return minus, plus


def compute_nu(w: ArrayLike, idx: VariableIndexMap, Re: float, boxlen: float):
    vmag = np.sqrt(np.sum(np.square(w[idx("v")]), axis=1))  # TEMP ARRAY
    nu = vmag * boxlen / Re  # TEMP ARRAY
    return nu


def add_viscuous_fluxes(
    _w_: ArrayLike,
    _f_: ArrayLike,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    use_GL: bool,
    Re: float,
    Chi: float,
    boxlen: float,
):
    cupy = CUPY_AVAILABLE and isinstance(_w_, cp.ndarray)
    xp = cp if cupy else np
    normal_first_derivative_stencil = _stencil_cache(
        FV_Stencil.FINITE_DIFFERENCE_FIRST_DERIVATIVE, p, cupy
    )
    transverse_first_derivative_stencil = _stencil_cache(
        FV_Stencil.FINITE_DIFFERENCE_FIRST_DERIVATIVE, p, cupy
    )

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    if "passives" in idx.group_var_map:
        raise NotImplementedError("Viscous fluxes for passive scalars are not implemented yet.")

    rho = _w_[idx("rho")]
    vx = _w_[idx("vx")]
    vy = _w_[idx("vy")]
    vz = _w_[idx("vz")]

    _wf_ = xp.empty_like(_w_)  # TEMP ARRAY
    interpolate_interface_nodes(_w_, _wf_, dim, active_dims, p, use_GL)

    # assign dvdx9 as [dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz]
    dvdx9 = xp.zeros((9, *_wf_.shape[1:]))  # TEMP ARRAY
    for i, d in enumerate(XYZ_TUPLE):
        slc = slice(i * 3, (i + 1) * 3)
        if d == dim:
            stencil_sweep(_w_[idx("v")], normal_first_derivative_stencil, dvdx9[slc, ...], d)
        elif d in active_dims:
            stencil_sweep(_wf_[idx("v")], transverse_first_derivative_stencil, dvdx9[slc, ...], d)

    minus_nu_rho = -compute_nu(_wf_, idx, Re, boxlen) * rho  # TEMP ARRAY
    if dim == "x":
        Pi11 = 4 * dvdx9[0, ...] - 2 * dvdx9[4, ...] - 2 * dvdx9[8, ...]  # TEMP ARRAY
        Pi11 *= minus_nu_rho / 3
        Pi12 = minus_nu_rho * (dvdx9[1, ...] + dvdx9[3, ...])  # TEMP ARRAY
        Pi13 = minus_nu_rho * (dvdx9[2, ...] + dvdx9[6, ...])  # TEMP ARRAY

        _f_[idx("mx")] += Pi11
        _f_[idx("my")] += Pi12
        _f_[idx("mz")] += Pi13
        _f_[idx("E")] += vx * Pi11 + vy * Pi12 + vz * Pi13
    elif dim == "y":
        Pi22 = -2 * dvdx9[0, ...] + 4 * dvdx9[4, ...] - 2 * dvdx9[8, ...]  # TEMP ARRAY
        Pi22 *= minus_nu_rho / 3
        Pi12 = minus_nu_rho * (dvdx9[1, ...] + dvdx9[3, ...])  # TEMP ARRAY
        Pi23 = minus_nu_rho * (dvdx9[5, ...] + dvdx9[7, ...])  # TEMP ARRAY

        _f_[idx("mx")] += Pi12
        _f_[idx("my")] += Pi22
        _f_[idx("mz")] += Pi23
        _f_[idx("E")] += vx * Pi12 + vy * Pi22 + vz * Pi23
    elif dim == "z":
        Pi33 = -2 * dvdx9[0, ...] - 2 * dvdx9[4, ...] + 4 * dvdx9[8, ...]  # TEMP ARRAY
        Pi33 *= minus_nu_rho / 3
        Pi13 = minus_nu_rho * (dvdx9[2, ...] + dvdx9[6, ...])  # TEMP ARRAY
        Pi23 = minus_nu_rho * (dvdx9[5, ...] + dvdx9[7, ...])  # TEMP ARRAY

        _f_[idx("mx")] += Pi13
        _f_[idx("my")] += Pi23
        _f_[idx("mz")] += Pi33
        _f_[idx("E")] += vx * Pi13 + vy * Pi23 + vz * Pi33
    else:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    # thermal fluxes
    if Chi > 0.0:
        dPdx = xp.empty_like(_wf_[idx("rho")])  # TEMP ARRAY
        stencil_sweep(_wf_[idx("P")], normal_first_derivative_stencil, dPdx, dim)
        _f_[idx("E")] += Chi * dPdx


def update_weno_fluxes(
    _u_: ArrayLike,
    _w_: ArrayLike,
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
):
    """
    Update the finite volume fluxes with a WENO reconstruction and the Zhang-Shu limiter.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Input array of conservative cell averages
        with ghost cells along active dimensions. Is not modified.
    _w_: _u_.shape - Input array of primitive cell averages with ghost cells.
        Should represent the same physical state as _u_. Is not modified.
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
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_u_, cp.ndarray) else np
    na = xp.newaxis
    fv = fv_params
    hp = hydro_params
    nghost = mesh.nghost
    use_GL = fv_params.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE

    # 1) Interpolate nodes at each face from either primitives or conservatives
    _q_ = _w_ if fv.flux_recipe == FluxRecipe.PRIM_PRIM_LIM else _u_

    node_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}
    n_nodes = _get_n_nodes_per_face(len(active_dims), fv)

    for dim in active_dims:
        _nodes_ = xp.empty(_w_.shape + (2 * n_nodes,))  # TEMP ARRAY
        interpolate_face_nodes(_q_, _nodes_, dim, active_dims, fv.p, use_GL)
        node_dict[dim] = _nodes_

    # 2) Ensure nodes are primitive and apply a priori limiting. Apply negative guard.
    if fv.flux_recipe == FluxRecipe.CONS_PRIM_LIM:
        # Convert conservative nodal values to primitives before slope limiting
        for dim in active_dims:
            cons_to_prim(node_dict[dim], node_dict[dim], idx, hp.gamma, hp.isothermal, hp.iso_cs)

    if fv.zhang_shu_params.use_ZS:
        # a priori slope limiting
        apply_zhang_shu_limiter(
            _u_ if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _w_,
            node_dict["x"] if "x" in active_dims else np.array([]),
            node_dict["y"] if "y" in active_dims else np.array([]),
            node_dict["z"] if "z" in active_dims else np.array([]),
            _qcc_,
            _theta_,
            _alpha_,
            idx,
            False if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM else True,
            active_dims,
            fv.p,
            fv.zhang_shu_params,
        )

    if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
        # Convert conservative nodal values to primitives after slope limiting
        for dim in active_dims:
            cons_to_prim(node_dict[dim], node_dict[dim], idx, hp.gamma, hp.isothermal, hp.iso_cs)

    # Fall back to first-order where nodes are negative
    for dim in active_dims:
        _positivity_guard(node_dict[dim], _w_, idx, hp)

    # 3) Solve Riemann problem at each face node and integrate to get fluxes
    for dim in active_dims:
        axis = DIM_TO_AXIS[dim]
        minus, plus = _get_riemann_solver_slices(axis, n_nodes, nghost)

        _left_nodes_ = node_dict[dim][minus]
        _right_nodes_ = node_dict[dim][plus]
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim]
        if len(active_dims) == 1:
            _fnodes_ = _F_out_[..., na]
        else:
            _fnodes_ = xp.empty_like(_left_nodes_)  # TEMP ARRAY

        solve_riemann_problem(
            _left_nodes_,
            _right_nodes_,
            _fnodes_,
            hp.riemann_solver,
            dim,
            idx,
            hp.gamma,
            hp.isothermal,
            hp.iso_cs,
        )

        # Compute viscuous fluxes
        if hydro_params.Re is not None:
            add_viscuous_fluxes(
                _w_,
                _fnodes_,
                idx,
                dim,
                active_dims,
                fv.p,
                use_GL,
                hydro_params.Re,
                hydro_params.Chi,
                mesh.boxlen,
            )

        if len(active_dims) == 1:
            break

        if use_GL:
            integrate_gauss_legendre_face_nodes(_fnodes_, _F_out_, dim, active_dims, fv_params.p)
        else:
            integrate_transverse_nodes(_fnodes_, _F_out_, dim, active_dims, fv_params.p)


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
):
    """
    Update the finite volume fluxes with a MUSCL scheme.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Input array of conservative cell averages
        with ghost cells along active dimensions. Is not modified.
    _w_: _u_.shape - Input array of primitive cell averages with ghost cells.
        Should represent the same physical state as _u_. Is not modified.
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
    hancock_dt: Time step to use for MUSCL-Hancock predictor step. If 0, the predictor step is
        skipped.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(_u_, cp.ndarray) else np
    na = xp.newaxis
    fv = fv_params
    hp = hydro_params
    nghost = mesh.nghost

    if not fv.muscl_params.use_MUSCL:
        raise ValueError("update_fluxes_with_muscl_scheme should only be called for MUSCL schemes.")

    # 1) Compute slopes from either conservatives or primitives
    _q_ = _u_ if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM else _w_

    slope_dict: Dict[Literal["x", "y", "z"], ArrayLike] = {}

    if fv.muscl_params.SED_params.use_SED:
        compute_alpha(_q_, _alpha_, active_dims, fv.muscl_params.SED_params.clip_zero_tol)

    if fv.muscl_params.MUSCL_limiter == MUSCL_SlopeLimiter.PP2D:
        if len(active_dims) != 2:
            raise ValueError("PP2D MUSCL slopes can only be used in 2D.")

        _slopes1_ = xp.empty_like(_q_)  # TEMP ARRAY
        _slopes2_ = xp.empty_like(_q_)  # TEMP ARRAY
        compute_PP2D_slopes(
            _q_,
            _alpha_,
            _slopes1_,
            _slopes2_,
            active_dims,
            1e-20,
            fv.muscl_params.SED_params.use_SED,
        )
        slope_dict[active_dims[0]] = _slopes1_
        slope_dict[active_dims[1]] = _slopes2_
    else:
        for dim in active_dims:
            _slopes_ = xp.empty_like(_q_)  # TEMP ARRAY
            compute_MUSCL_slopes(
                _q_,
                _alpha_,
                _slopes_,
                dim,
                fv.muscl_params.MUSCL_limiter,
                fv.muscl_params.SED_params.use_SED,
            )
            slope_dict[dim] = _slopes_

    # 2) Compute predictor step
    _predictor_q_ = _q_.copy()  # TEMP ARRAY

    if hancock_dt > 0:
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
                primitives=fv.flux_recipe != FluxRecipe.CONS_LIM_PRIM,
            )
            _predictor_q_ -= 0.5 * _jvp_ * hancock_dt / h

    # 3) # Corrector step
    for dim in active_dims:
        _F_out_ = {"x": _F_, "y": _G_, "z": _H_}[dim][..., na]

        # Allocate some temp arrays
        _slopes_ = slope_dict[dim]
        _left_face_ = _predictor_q_ - 0.5 * _slopes_  # TEMP ARRAY
        _right_face_ = _predictor_q_ + 0.5 * _slopes_  # TEMP ARRAY
        _faces_ = xp.concatenate(
            [_left_face_[..., na], _right_face_[..., na]], axis=4
        )  # TEMP ARRAY

        # Ensure faces are positive and primitive
        if fv.flux_recipe == FluxRecipe.CONS_LIM_PRIM:
            cons_to_prim(_faces_, _faces_, idx, hp.gamma, hp.isothermal, hp.iso_cs)
        _positivity_guard(_faces_, _w_, idx, hp)

        # Solve Riemann problem at faces
        minus, plus = _get_riemann_solver_slices(DIM_TO_AXIS[dim], 1, nghost)
        solve_riemann_problem(
            _faces_[minus],
            _faces_[plus],
            _F_out_,
            hp.riemann_solver,
            dim,
            idx,
            hp.gamma,
            hp.isothermal,
            hp.iso_cs,
        )

        # Compute viscuous fluxes
        if hydro_params.Re is not None:
            add_viscuous_fluxes(
                _w_,
                _F_out_,
                idx,
                dim,
                active_dims,
                fv.p,
                False,
                hydro_params.Re,
                hydro_params.Chi,
                mesh.boxlen,
            )


def update_fv_fluxes(
    _u_: ArrayLike,
    _w_: ArrayLike,
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
):
    """
    Update the finite volume fluxes with a WENO or MUSCL scheme.

    _u_: shape (nvars, mesh._nx_, mesh._ny_, mesh._nz_) - Array to which u is written with boundary
        conditions applied to ghost cells along active dimensions.
    _w_: _u_.shape - Array to which primitive cell averages are written with ghost cells.
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
        )
    else:
        update_weno_fluxes(
            _u_,
            _w_,
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
        )


def compute_fv_dudt(
    F: ArrayLike,
    G: ArrayLike,
    H: ArrayLike,
    S: ArrayLike,
    mesh: UniformFiniteVolumeMesh,
) -> ArrayLike:
    """
    Compute the finite volume time derivative of the conserved variables.

    F: shape (nvars, nx + 1, mesh.ny, mesh.nz) - Array to which the x-fluxes are written if
        "x" is in active_dims.
    G: shape (nvars, mesh.nx, ny + 1, mesh.nz) - Array to which the y-fluxes are written if
        "y" is in active_dims.
    H: shape (nvars, mesh.nx, mesh.ny, nz + 1) - Array to which the z-fluxes are written if
        "z" is in active_dims.
    S: shape (nvars, mesh.nx, mesh.ny, mesh.nz) - Array of source terms.
    mesh: Mesh object containing information about the mesh and its dimensions.
    """
    dudt = S.copy()  # TEMP ARRAY
    for dim in mesh.active_dims:
        left = crop(DIM_TO_AXIS[dim], (None, -1), ndim=4)
        right = crop(DIM_TO_AXIS[dim], (1, None), ndim=4)
        fluxes = {"x": F, "y": G, "z": H}[dim]
        h = getattr(mesh, "h" + dim)

        dudt -= (fluxes[right] - fluxes[left]) / h

    return dudt


def compute_fv_dt(
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

    return dt


def compute_fv_nghost(fv_scheme: FV_SchemeParameters, ndim: int) -> int:
    """
    Returns the number of ghost cells required in the padded arrays of the HydroSolver based on
    the provided `fv_scheme`.
    """
    left_right_reach = conservative_interpolation.left_right(fv_scheme.p).shape[1] // 2
    cell_center_reach = conservative_interpolation.cell_center(fv_scheme.p).shape[1] // 2
    transverse_reach = transverse_integration(fv_scheme.p).shape[1] // 2

    nghost = left_right_reach  # Interpolating left/right face values

    # Cost of update_fv_workspace
    if fv_scheme.lazy_primitive_mode in (LazyPrimitiveMode.NONE, LazyPrimitiveMode.ADAPTIVE):
        interpolates_from_high_order_primitives = False
        if fv_scheme.flux_recipe == FluxRecipe.PRIM_PRIM_LIM:
            interpolates_from_high_order_primitives = True
        if fv_scheme.zhang_shu_params.use_ZS and fv_scheme.flux_recipe == FluxRecipe.CONS_PRIM_LIM:
            interpolates_from_high_order_primitives = True

        if interpolates_from_high_order_primitives:
            nghost += cell_center_reach + transverse_reach  # Interpolating primitives before faces
        else:
            nghost = max(nghost, cell_center_reach)  # Interpolating cell centers

        if (
            fv_scheme.lazy_primitive_mode == LazyPrimitiveMode.ADAPTIVE
            and fv_scheme.shock_detection_params.use_shock_detection
        ):
            nghost += 2  # Shock detection on top of primitive cell averages

    # Ghost cell cost of integrating fluxes and Riemann Solver
    if fv_scheme.flux_quadrature == FluxQuadrature.TRANSVERSE and ndim >= 2:
        nghost += max(transverse_reach, 1)  # Transverse integration + Riemann solver
    else:
        nghost += 1  # Riemann solver

    # Ghost cell cost of slope limiter
    mood_cost = 0
    if fv_scheme.mood_params.use_MOOD:
        if fv_scheme.mood_params.NAD_params.use_NAD:
            mood_cost += 1
        mood_cost += 2 if fv_scheme.mood_params.blend_troubles else 1

    nghost += max(
        max(1, cell_center_reach) if fv_scheme.zhang_shu_params.use_ZS else 0,
        mood_cost,
        3 if fv_scheme.muscl_params.SED_params.use_SED else 0,
        3 if fv_scheme.zhang_shu_params.SED_params.use_SED else 0,
        3 if fv_scheme.mood_params.NAD_params.SED_params.use_SED else 0,
    )

    return nghost
