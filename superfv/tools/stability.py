import numpy as np

from .device_management import ArrayLike


def avoid0(x: ArrayLike, eps: float = 1e-15) -> ArrayLike:
    """
    Robust small-denominator guard: clamp magnitude to at least eps, preserving sign.

    Args:
        x: Array to be clamped.
        eps: Small value to avoid division by zero.
    """
    return np.where(x >= 0, np.maximum(x, eps), np.minimum(x, -eps))
