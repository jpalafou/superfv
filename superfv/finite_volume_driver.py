from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Tuple, cast

import numpy as np

from superfv.tools.variable_index_map import VariableIndexMap

from .configs import ZhangShuParameters
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


def apply_zhang_shu_limiter(
    _w_: ArrayLike,
    _wcc_: ArrayLike,
    _theta_: ArrayLike,
    _alpha_: ArrayLike,
    idx: VariableIndexMap,
    params: ZhangShuParameters,
    _x_nodes_: Optional[ArrayLike] = None,
    _y_nodes_: Optional[ArrayLike] = None,
    _z_nodes_: Optional[ArrayLike] = None,
):
    cupy = CUPY_AVAILABLE and isinstance(_w_, cp.ndarray)
    na = cp.newaxis if cupy else np.newaxis
    active_dims = cast(
        Tuple[Literal["x", "y", "z"], ...],
        tuple(
            d
            for d, nodes in zip(["x", "y", "z"], [_x_nodes_, _y_nodes_, _z_nodes_])
            if nodes is not None
        ),
    )

    if _w_.ndim != 4:
        raise ValueError("_w_ must be 4D.")
    if _w_.shape != _wcc_.shape:
        raise ValueError("_w_ and _wcc_ must have the same shape.")
    if _theta_.shape != _w_.shape:
        raise ValueError("_theta_ must have the same shape as _w_.")
    if _alpha_.shape != _w_.shape:
        raise ValueError("_alpha_ must have the same shape as _w_.")

    # allocate arrays
    _qj_shape_ = _w_.shape[:4] + (
        1 + sum(nodes.shape[4] for nodes in [_x_nodes_, _y_nodes_, _z_nodes_] if nodes is not None),
    )
    _qj_ = cp.empty(_qj_shape_) if cupy else np.empty(_qj_shape_)
    _M_ = cp.empty_like(_w_) if cupy else np.empty_like(_w_)
    _m_ = cp.empty_like(_w_) if cupy else np.empty_like(_w_)
    _Mj_ = cp.empty_like(_w_) if cupy else np.empty_like(_w_)
    _mj_ = cp.empty_like(_w_) if cupy else np.empty_like(_w_)

    # stack all nodes into into _qj_
    _qj_[..., 0] = _wcc_
    for i, nodes in enumerate([x for x in [_x_nodes_, _y_nodes_, _z_nodes_] if x is not None]):
        n = nodes.shape[4]
        idx1 = 1 + i * n
        idx2 = 1 + (i + 1) * n
        _qj_[..., slice(idx1, idx2)] = nodes

    compute_dmp(_w_, _M_, _m_, active_dims, params.include_corners)  # update _M_ and _m_ with DMP
    compute_theta(
        _w_, _qj_, _M_, _m_, _Mj_, _mj_, _theta_, params.theta_denom_tol
    )  # update _Mj_, _mj_, and _theta_

    if params.SED_params.use_SED:
        if _alpha_ is None:
            raise ValueError("_alpha_ must be provided if detect_smooth_extrema is True")
        compute_alpha(_w_, _alpha_, active_dims)

        if params.PAD_params.use_PAD:
            _physical_ = (
                cp.ones_like(_alpha_[0], dtype=bool)
                if cupy
                else np.ones_like(_alpha_[0], dtype=bool)
            )
            for v, (lb, ub) in params.PAD_params.bounds.items():
                if ub is not None:
                    _physical_ &= _Mj_[idx(v)] <= ub
                if lb is not None:
                    _physical_ &= _mj_[idx(v)] >= lb
            _alpha_ *= _physical_
        if cupy:
            cp.maximum(_theta_, _alpha_ >= 1, out=_theta_)
        else:
            np.maximum(_theta_, _alpha_ >= 1, out=_theta_)

    # omit variables from limiting
    if "omit_ZS" in idx.group_var_map:
        _theta_[idx("omit_ZS")] = 1.0

    # apply limiter to face nodes
    if _x_nodes_ is not None:
        _x_nodes_[...] = _theta_[..., na] * (_x_nodes_ - _w_[..., na]) + _w_[..., na]
    if _y_nodes_ is not None:
        _y_nodes_[...] = _theta_[..., na] * (_y_nodes_ - _w_[..., na]) + _w_[..., na]
    if _z_nodes_ is not None:
        _z_nodes_[...] = _theta_[..., na] * (_z_nodes_ - _w_[..., na]) + _w_[..., na]
