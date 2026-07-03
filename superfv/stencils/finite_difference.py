import numpy as np


def first_derivative(p: int) -> np.ndarray:
    """
    Returns stencil weights for finite difference for polynomial degree `p`.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Stencil weight array of shape (1, n).
    """
    if p in (0, 1):
        return np.array([[-1 / 2, 0.0, 1 / 2]])
    if p in (2, 3):
        return np.array([[1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]])
    if p in (4, 5):
        return np.array([[-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60]])
    if p in (6, 7):
        return np.array([[1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]])
    raise NotImplementedError(f"Unsupported polynomial degree: {p}")
