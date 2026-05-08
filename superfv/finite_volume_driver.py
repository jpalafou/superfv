from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, Tuple

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
    elif ndim == 2:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 2D interpolation")
        stencil_sweep(_q_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qcc_, active_dims[1])
    elif ndim == 3:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 3D interpolation")
        if _qtemp2_ is None:
            raise ValueError("_qtemp2_ must be provided for 3D interpolation")
        stencil_sweep(_q_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _qcc_, active_dims[2])


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
    elif ndim == 2:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 2D interpolation")
        stencil_sweep(_qcc_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _q_, active_dims[1])
    elif ndim == 3:
        if _qtemp1_ is None:
            raise ValueError("_qtemp1_ must be provided for 3D interpolation")
        if _qtemp2_ is None:
            raise ValueError("_qtemp2_ must be provided for 3D interpolation")
        stencil_sweep(_qcc_, weights, _qtemp1_, active_dims[0])
        stencil_sweep(_qtemp1_, weights, _qtemp2_, active_dims[1])
        stencil_sweep(_qtemp2_, weights, _q_, active_dims[2])
