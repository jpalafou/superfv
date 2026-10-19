from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Tuple

import numpy as np

from superfv.tools.slicing import crop, merge_slices
from superfv.tools.step_history import MultiTimer

from .axes import DIM_TO_AXIS
from .boundary_conditions import apply_bc
from .configs import BoundaryConditionParameters, HydroParameters
from .mesh import UniformFiniteVolumeMesh
from .quadrature import perform_quadrature
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


class Stencil(Enum):
    CONSERVATIVE_INTERP_CENTER = 0
    CONSERVATIVE_INTERP_LEFT_RIGHT = 1
    CONSERVATIVE_INTERP_GAUSS_LEGNEDRE = 2
    CONSERVATIVE_INTERP_INTERFACE = 3
    CONSERVATIVE_INTERP_INTERFACE_1ST_DER = 4
    TRANSVERSE_INTEGRATION = 5
    FINITE_DIFFERENCE_FIRST_DERIVATIVE = 6


@lru_cache
def _stencil_cache(interpolation: Stencil, p: int, cupy: bool = False) -> ArrayLike:
    match interpolation:
        case Stencil.CONSERVATIVE_INTERP_CENTER:
            stencil = conservative_interpolation.cell_center(p)
        case Stencil.CONSERVATIVE_INTERP_LEFT_RIGHT:
            stencil = conservative_interpolation.left_right(p)
        case Stencil.CONSERVATIVE_INTERP_GAUSS_LEGNEDRE:
            stencil = conservative_interpolation.gauss_legendre_nodes(p)
        case Stencil.CONSERVATIVE_INTERP_INTERFACE:
            stencil = conservative_interpolation.interface(p)
        case Stencil.CONSERVATIVE_INTERP_INTERFACE_1ST_DER:
            stencil = conservative_interpolation.interface_first_derivative(p)
        case Stencil.TRANSVERSE_INTEGRATION:
            stencil = transverse_integration(p)
        case Stencil.FINITE_DIFFERENCE_FIRST_DERIVATIVE:
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
    timer: Optional[MultiTimer] = None,
):
    """
    Interpolate finite volume cell centers from `_q_` and write them to `_qcc_`.
    """
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    weights = _stencil_cache(Stencil.CONSERVATIVE_INTERP_CENTER, p, cupy)
    xp = cp if cupy else np
    na = xp.newaxis
    ndim = len(active_dims)

    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape != _qcc_.shape:
        raise ValueError("_q_ and _qcc_ must have the same shape.")

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START

    if p < 2:
        _qcc_[...] = _q_
        timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP
        return

    if ndim == 1:
        stencil_sweep(_q_[..., na], weights, _qcc_[..., na], active_dims[0])
    elif ndim == 2:
        _qtemp1_ = xp.empty_like(_q_[..., na])

        stencil_sweep(_q_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qcc_[..., na], active_dims[1])
    elif ndim == 3:
        _qtemp1_ = xp.empty_like(_q_[..., na])
        _qtemp2_ = xp.empty_like(_q_[..., na])

        stencil_sweep(_q_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _qcc_[..., na], active_dims[2])

    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def integrate_cell_averages(
    _qcc_: ArrayLike,
    _q_: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    timer: Optional[MultiTimer] = None,
):
    """
    Interpolate finite volume cell averages from `_qcc_` and write them to `_q_`.
    """
    cupy = CUPY_AVAILABLE and isinstance(_qcc_, cp.ndarray)
    weights = _stencil_cache(Stencil.TRANSVERSE_INTEGRATION, p, cupy)
    xp = cp if cupy else np
    na = xp.newaxis
    ndim = len(active_dims)

    if _qcc_.ndim != 4:
        raise ValueError("_qcc_ must be 4D.")
    if _qcc_.shape != _q_.shape:
        raise ValueError("_qcc_ and _q_ must have the same shape.")

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START

    if p < 2:
        _q_[...] = _qcc_
        timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP
        return

    if ndim == 1:
        stencil_sweep(_qcc_[..., na], weights, _q_[..., na], active_dims[0])
    elif ndim == 2:
        _qtemp1_ = xp.empty_like(_q_[..., na])

        stencil_sweep(_qcc_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _q_[..., na], active_dims[1])
    elif ndim == 3:
        _qtemp1_ = xp.empty_like(_q_[..., na])
        _qtemp2_ = xp.empty_like(_q_[..., na])

        stencil_sweep(_qcc_[..., na], weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _q_[..., na], active_dims[2])

    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def interpolate_face_nodes(
    _q_: ArrayLike,
    _qj_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    gauss_legendre: bool = False,
    timer: Optional[MultiTimer] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    lr_stencil = _stencil_cache(Stencil.CONSERVATIVE_INTERP_LEFT_RIGHT, p, cupy)
    xp = cp if cupy else np
    na = xp.newaxis
    ndim = len(active_dims)

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _q_.shape[:4] != _qj_.shape[:4]:
        raise ValueError("The first 4 dimensions of _q_ and _qj_ must match.")

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START

    if p < 1:
        if _qj_.shape[4] != 2:
            raise ValueError("The 5th dimension of _qj_ must be 2 for 1D interpolation.")
        _qj_[...] = _q_[..., na]
        timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP
        return

    base_shape = _q_.shape

    if ndim == 1:
        if gauss_legendre:
            raise ValueError("No Gauss-Legendre face nodes in 1D.")
        if _qj_.shape[4] != 2:
            raise ValueError("The 5th dimension of _qj_ must be 2 for 1D interpolation.")
        stencil_sweep(_q_[..., na], lr_stencil, _qj_, dim)
        timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP
        return

    if gauss_legendre:
        gl_stencil = _stencil_cache(Stencil.CONSERVATIVE_INTERP_GAUSS_LEGNEDRE, p, cupy)
        ngl = gl_stencil.shape[0]

        if ndim == 2:
            if _qj_.shape[4] != 2 * ngl:
                raise ValueError(
                    "The 5th dimension of _qj_ must be 2 two times the number of GL nodes."
                )

            _qtemp1_ = xp.empty((*base_shape, 2))

            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_[..., na], lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qj_, trans_dim)
        elif ndim == 3:
            if _qj_.shape[4] != 2 * ngl**2:
                raise ValueError(
                    "The 5th dimension of _qj_ must be 2 two times the number of GL nodes squared."
                )

            _qtemp1_ = xp.empty((*base_shape, 2))
            _qtemp2_ = xp.empty((*base_shape, 2 * ngl))

            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_[..., na], lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, gl_stencil, _qj_, trans_dim2)
    else:
        cc_stencil = _stencil_cache(Stencil.CONSERVATIVE_INTERP_CENTER, p, cupy)
        if _qj_.shape[4] != 2:
            raise ValueError(
                "The 5th dimension of _qj_ must be 2 for transverse integration in 2D or 3D."
            )

        if ndim == 2:
            _qtemp1_ = xp.empty_like(_q_[..., na])

            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_[..., na], cc_stencil, _qtemp1_, trans_dim)
            stencil_sweep(_qtemp1_, lr_stencil, _qj_, dim)
        elif ndim == 3:
            _qtemp1_ = xp.empty_like(_q_[..., na])
            _qtemp2_ = xp.empty_like(_q_[..., na])

            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_[..., na], cc_stencil, _qtemp1_, trans_dim2)
            stencil_sweep(_qtemp1_, cc_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, lr_stencil, _qj_, dim)

    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def interpolate_interface_nodes(
    _q_: ArrayLike,
    _qf_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    gauss_legendre: bool = False,
    first_derivative: bool = False,
    h: Optional[float] = None,
    timer: Optional[MultiTimer] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    xp = cp if cupy else np
    if first_derivative:
        if h is None:
            raise ValueError("h must be provided for first_derivative=True.")
        interface_stencil = (
            _stencil_cache(Stencil.CONSERVATIVE_INTERP_INTERFACE_1ST_DER, p, cupy) / h
        )
    else:
        interface_stencil = _stencil_cache(Stencil.CONSERVATIVE_INTERP_INTERFACE, p, cupy)
    center_stencil = _stencil_cache(Stencil.CONSERVATIVE_INTERP_CENTER, p, cupy)
    na = xp.newaxis
    ndim = len(active_dims)

    if gauss_legendre:
        raise NotImplementedError("Gauss-Legendre interface nodes are not implemented yet.")
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _q_.ndim != 4:
        raise ValueError("_q_ must be 4D.")
    if _qf_.ndim != 5:
        raise ValueError("_qf_ must be 5D.")
    if _q_.shape[:4] != _qf_.shape[:4]:
        raise ValueError("The first 4 dimensions of _q_ and _qf_ must match.")
    if _qf_.shape[4] != 1:
        raise ValueError("The 5th dimension of _qf_ must be 1 for interface nodes.")

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START

    if ndim == 1:
        stencil_sweep(_q_[..., na], interface_stencil, _qf_, dim)
    elif ndim == 2:
        trans_dim = [d for d in active_dims if d != dim][0]
        _qtemp_ = xp.empty((*_q_.shape, 1))
        stencil_sweep(_q_[..., na], interface_stencil, _qtemp_, dim)
        stencil_sweep(_qtemp_, center_stencil, _qf_, trans_dim)
    elif ndim == 3:
        trans_dims = [d for d in active_dims if d != dim]
        trans_dim1 = trans_dims[0]
        trans_dim2 = trans_dims[1]
        _qtemp1_ = xp.empty((*_q_.shape, 1))
        _qtemp2_ = xp.empty((*_q_.shape, 1))
        stencil_sweep(_q_[..., na], interface_stencil, _qtemp1_, dim)
        stencil_sweep(_qtemp1_, center_stencil, _qtemp2_, trans_dim1)
        stencil_sweep(_qtemp2_, center_stencil, _qf_, trans_dim2)

    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def integrate_gauss_legendre_face_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    timer: Optional[MultiTimer] = None,
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

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START
    perform_quadrature(_qj_, weights, _qF_)
    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def integrate_transverse_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    timer: Optional[MultiTimer] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    stencil = _stencil_cache(Stencil.TRANSVERSE_INTEGRATION, p, cupy)
    xp = cp if cupy else np
    na = xp.newaxis
    ndim = len(active_dims)

    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")
    if _qj_.ndim != 5 and _qj_.shape[4] == 1:
        raise ValueError("_qj_ must be 5D with the 5th dimension equal to 1.")
    if _qF_.shape != _qj_.shape[:4]:
        raise ValueError("The shape of _qF_ must match the first 4 dimensions of _qj_.")

    timer is not None and timer.start("stencil_sweep", cupy)  # TIMER START

    if p < 2:
        _qF_[...] = _qj_[..., 0]
        timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP
        return

    if ndim == 1:
        raise ValueError("Cannot integrate transverse face nodes in 1D.")
    if ndim == 2:
        trans_dim = [d for d in active_dims if d != dim][0]
        stencil_sweep(_qj_, stencil, _qF_[..., na], trans_dim)
    elif ndim == 3:
        _qtemp_ = xp.empty_like(_qj_)

        trans_dims = [d for d in active_dims if d != dim]
        trans_dim1 = trans_dims[0]
        trans_dim2 = trans_dims[1]
        stencil_sweep(_qj_, stencil, _qtemp_, trans_dim1)
        stencil_sweep(_qtemp_, stencil, _qF_[..., na], trans_dim2)

    timer is not None and timer.stop("stencil_sweep", cupy)  # TIMER STOP


def apply_fv_bc(
    _u_: ArrayLike,
    idx: VariableIndexMap,
    mesh: UniformFiniteVolumeMesh,
    t: float,
    bc_params: BoundaryConditionParameters,
    hydro_params: Optional[HydroParameters] = None,
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
        hydro_params.gamma if hydro_params is not None else None,
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
