from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Tuple

from .quadrature import perform_quadrature
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
