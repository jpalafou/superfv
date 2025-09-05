import numpy as np

from .device_management import ArrayLike


def l1_norm(array: ArrayLike) -> float:
    """
    Compute the L1 norm of an array.
    """
    return np.mean(np.abs(array))


def l2_norm(array: ArrayLike) -> float:
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.mean(np.square(array)))


def linf_norm(array: ArrayLike) -> float:
    """
    Compute the L-infinity norm of an array.
    """
    return np.max(np.abs(array))
