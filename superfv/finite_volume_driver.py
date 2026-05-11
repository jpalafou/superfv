from enum import Enum
from functools import lru_cache
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .quadrature import perform_quadrature
from .slope_limiting import compute_dmp
from .slope_limiting.smooth_extrema_detection import compute_alpha
from .slope_limiting.zhang_and_shu import compute_theta
from .stencils import conservative_interpolation, transverse_integration
from .sweep import stencil_sweep
from .tools.device_management import CUPY_AVAILABLE, ArrayLike

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


class FV_Stencil(Enum):
    CONSERVATIVE_INTERPOLATION_CENTER = 0
    CONSERVATIVE_INTERPOLATION_LEFT_RIGHT = 1
    CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE = 2
    TRANSVERSE_INTEGRATION = 3


@lru_cache
def _stencil_cache(interpolation: FV_Stencil, p: int, cupy: bool = False):
    match interpolation:
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER:
            stencil = conservative_interpolation.cell_center(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_LEFT_RIGHT:
            stencil = conservative_interpolation.left_right(p)
        case FV_Stencil.CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE:
            stencil = conservative_interpolation.gauss_legendre_nodes(p)
        case FV_Stencil.TRANSVERSE_INTEGRATION:
            stencil = transverse_integration(p)

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
    _qtemp1_: Optional[ArrayLike] = None,
    _qtemp2_: Optional[ArrayLike] = None,
):
    """
    Interpolate finite volume cell centers from `_q_` and write them to `_qcc_`.
    """
    ndim = len(active_dims)

    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    weights = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER, p, cupy)

    if ndim == 1:
        stencil_sweep(_q_, weights, _qcc_, active_dims[0])
        return
    elif ndim == 2:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 2D interpolation")
        stencil_sweep(_q_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qcc_, active_dims[1])
        return
    elif ndim == 3:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 3D interpolation")
        if _qtemp2_ is None:
            raise ValueError("_qtemp2_ must be provided for 3D interpolation")
        stencil_sweep(_q_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _qcc_, active_dims[2])
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def integrate_cell_averages(
    _qcc_: ArrayLike,
    _q_: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    _qtemp1_: Optional[ArrayLike] = None,
    _qtemp2_: Optional[ArrayLike] = None,
):
    """
    Interpolate finite volume cell averages from `_qcc_` and write them to `_q_`.
    """
    ndim = len(active_dims)

    cupy = CUPY_AVAILABLE and isinstance(_qcc_, cp.ndarray)
    weights = _stencil_cache(FV_Stencil.TRANSVERSE_INTEGRATION, p, cupy)

    if ndim == 1:
        stencil_sweep(_qcc_, weights, _q_, active_dims[0])
        return
    elif ndim == 2:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 2D interpolation")
        stencil_sweep(_qcc_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _q_, active_dims[1])
        return
    elif ndim == 3:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 3D interpolation")
        if _qtemp2_ is None:
            raise ValueError("_qtemp2_ must be provided for 3D interpolation")
        stencil_sweep(_qcc_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _q_, active_dims[2])
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def interpolate_face_nodes(
    _q_: ArrayLike,
    _qj_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    gauss_legendre: bool = False,
    _qtemp1_: Optional[ArrayLike] = None,
    _qtemp2_: Optional[ArrayLike] = None,
):
    ndim = len(active_dims)
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    lr_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_LEFT_RIGHT, p, cupy)

    if ndim == 1:
        if gauss_legendre:
            raise ValueError("No Gauss-Legendre face nodes in 1D.")
        stencil_sweep(_q_, lr_stencil, _qj_, dim)
        return

    if gauss_legendre:
        gl_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_GAUSS_LEGNEDRE, p, cupy)
        if ndim == 2:
            if _qtemp1_ is None:
                raise ValueError("_qtemp1_ must be provided for 2D interpolation")
            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_, lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qj_, trans_dim)
            return
        elif ndim == 3:
            if _qtemp1_ is None:
                raise ValueError("_qtemp1_ must be provided for 3D interpolation")
            if _qtemp2_ is None:
                raise ValueError("_qtemp2_ must be provided for 3D interpolation")
            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_, lr_stencil, _qtemp1_, dim)
            stencil_sweep(_qtemp1_, gl_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, gl_stencil, _qj_, trans_dim2)
            return
    else:
        cc_stencil = _stencil_cache(FV_Stencil.CONSERVATIVE_INTERPOLATION_CENTER, p, cupy)
        if ndim == 2:
            if _qtemp1_ is None:
                raise ValueError("_qtemp1_ must be provided for 2D interpolation")
            trans_dim = [d for d in active_dims if d != dim][0]
            stencil_sweep(_q_, cc_stencil, _qtemp1_, trans_dim)
            stencil_sweep(_qtemp1_, lr_stencil, _qj_, dim)
            return
        elif ndim == 3:
            if _qtemp1_ is None:
                raise ValueError("_qtemp1_ must be provided for 3D interpolation")
            if _qtemp2_ is None:
                raise ValueError("_qtemp2_ must be provided for 3D interpolation")
            trans_dims = [d for d in active_dims if d != dim]
            trans_dim1 = trans_dims[0]
            trans_dim2 = trans_dims[1]
            stencil_sweep(_q_, cc_stencil, _qtemp1_, trans_dim2)
            stencil_sweep(_qtemp1_, cc_stencil, _qtemp2_, trans_dim1)
            stencil_sweep(_qtemp2_, lr_stencil, _qj_, dim)
            return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def integrate_gauss_legendre_face_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
):
    ndim = len(active_dims)
    if ndim == 1:
        raise ValueError("No Gauss-Legendre face nodes in 1D.")
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    weights = _gauss_legendre_weights_cache(p, ndim, cupy)

    perform_quadrature(_qj_, weights, _qF_)


def integrate_transverse_nodes(
    _qj_: ArrayLike,
    _qF_: ArrayLike,
    dim: Literal["x", "y", "z"],
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    p: int,
    _qtemp_: Optional[ArrayLike] = None,
):
    ndim = len(active_dims)
    if dim not in active_dims:
        raise ValueError(f"Dimension {dim} is not in active_dims {active_dims}.")

    cupy = CUPY_AVAILABLE and isinstance(_qj_, cp.ndarray)
    stencil = _stencil_cache(FV_Stencil.TRANSVERSE_INTEGRATION, p, cupy)

    if ndim == 1:
        raise ValueError("Cannot integrate transverse face nodes in 1D.")
    if ndim == 2:
        trans_dim = [d for d in active_dims if d != dim][0]
        stencil_sweep(_qj_, stencil, _qF_, trans_dim)
        return
    elif ndim == 3:
        if _qtemp_ is None:
            raise ValueError("_qtemp_ must be provided for 3D transverse integration")
        trans_dims = [d for d in active_dims if d != dim]
        trans_dim1 = trans_dims[0]
        trans_dim2 = trans_dims[1]
        stencil_sweep(_qj_, stencil, _qtemp_, trans_dim1)
        stencil_sweep(_qtemp_, stencil, _qF_, trans_dim2)
        return

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def apply_zhang_shu_limiter(
    _q_: ArrayLike,
    _qcc_: ArrayLike,
    _theta_: ArrayLike,
    eps: float,
    include_corners: bool,
    detect_smooth_extrema: bool,
    physical_admissibility_detection: bool,
    _x_nodes_: Optional[ArrayLike] = None,
    _y_nodes_: Optional[ArrayLike] = None,
    _z_nodes_: Optional[ArrayLike] = None,
    _alpha_: Optional[ArrayLike] = None,
    PAD_bounds: Optional[Dict[int, Tuple[Optional[float], Optional[float]]]] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_q_, cp.ndarray)
    na = cp.newaxis if cupy else np.newaxis
    active_dims = tuple(
        d
        for d, nodes in zip(["x", "y", "z"], [_x_nodes_, _y_nodes_, _z_nodes_])
        if nodes is not None
    )

    # allocate arrays
    _qj_shape_ = _q_.shape[:4] + (
        1 + sum(nodes.shape[4] for nodes in [_x_nodes_, _y_nodes_, _z_nodes_] if nodes is not None),
    )
    _qj_ = cp.empty(_qj_shape_) if cupy else np.empty(_qj_shape_)
    _M_ = cp.empty_like(_q_) if cupy else np.empty_like(_q_)
    _m_ = cp.empty_like(_q_) if cupy else np.empty_like(_q_)
    _Mj_ = cp.empty_like(_q_) if cupy else np.empty_like(_q_)
    _mj_ = cp.empty_like(_q_) if cupy else np.empty_like(_q_)

    # stack all nodes into into _qj_
    _qj_[..., 0] = _qcc_
    for i, nodes in enumerate([x for x in [_x_nodes_, _y_nodes_, _z_nodes_] if x is not None]):
        n = nodes.shape[4]
        idx1 = 1 + i * n
        idx2 = 1 + (i + 1) * n
        _qj_[..., slice(idx1, idx2)] = nodes

    compute_dmp(_q_, _M_, _m_, active_dims, include_corners)  # update _M_ and _m_ with DMP
    compute_theta(_q_, _qj_, _M_, _m_, _Mj_, _mj_, _theta_, eps)  # update _Mj_, _mj_, and _theta_

    if detect_smooth_extrema:
        if _alpha_ is None:
            raise ValueError("_alpha_ must be provided if detect_smooth_extrema is True")
        compute_alpha(_q_, _alpha_, active_dims)

        if physical_admissibility_detection:
            _physical_ = cp.ones_like(_alpha_[0]) if cupy else np.ones_like(_alpha_[0])
            for i, (lb, ub) in PAD_bounds.items():
                if ub is not None:
                    _physical_ &= _Mj_[i] <= ub
                if lb is not None:
                    _physical_ &= _mj_[i] >= lb
            _alpha_ *= _physical_
        if cupy:
            cp.maximum(_theta_, _alpha_, out=_theta_)
        else:
            np.maximum(_theta_, _alpha_, out=_theta_)

    # apply limiter to face nodes
    if _x_nodes_ is not None:
        _x_nodes_[...] = _theta_[..., na] * (_x_nodes_ - _q_[..., na]) + _q_[..., na]
    if _y_nodes_ is not None:
        _y_nodes_[...] = _theta_[..., na] * (_y_nodes_ - _q_[..., na]) + _q_[..., na]
    if _z_nodes_ is not None:
        _z_nodes_[...] = _theta_[..., na] * (_z_nodes_ - _q_[..., na]) + _q_[..., na]
