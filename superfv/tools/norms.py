import numpy as np

from .device_management import CUPY_AVAILABLE, ArrayLike

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def l1_norm(array: ArrayLike) -> float:
    """
    Compute the L1 norm of an array.
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.mean(cp.abs(array)).item()
    else:
        return np.mean(np.abs(array)).item()


def l2_norm(array: ArrayLike) -> float:
    """
    Compute the L2 norm of an array.
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.sqrt(cp.mean(cp.square(array))).item()
    else:
        return np.sqrt(np.mean(np.square(array))).item()


def linf_norm(array: ArrayLike) -> float:
    """
    Compute the L-infinity norm of an array.
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.max(cp.abs(array)).item()
    else:
        return np.max(np.abs(array)).item()
