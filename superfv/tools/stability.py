from types import ModuleType

from .device_management import ArrayLike


def avoid0(xp: ModuleType, x: ArrayLike, eps: float = 1e-15) -> ArrayLike:
    """
    Robust small-denominator guard: clamp magnitude to at least eps, preserving sign.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        x: Array to be clamped.
        eps: Small value to avoid division by zero.
    """
    return xp.where(x >= 0, xp.maximum(x, eps), xp.minimum(x, -eps))
