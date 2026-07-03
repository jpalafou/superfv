import numpy as np


def transverse_integration(p: int) -> np.ndarray:
    """
    Returns stencil weights for transverse integration for polynomial degree `p`.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Stencil weight array of shape (1, n).
    """
    if p in (0, 1):
        return np.array([[1.0]])
    if p in (2, 3):
        return np.array([[1 / 24, 11 / 12, 1 / 24]])
    if p in (4, 5):
        return np.array([[-17 / 5760, 77 / 1440, 863 / 960, 77 / 1440, -17 / 5760]])
    if p in (6, 7):
        return np.array(
            [
                [
                    367 / 967680,
                    -281 / 53760,
                    6361 / 107520,
                    215641 / 241920,
                    6361 / 107520,
                    -281 / 53760,
                    367 / 967680,
                ]
            ]
        )
    raise NotImplementedError(f"Unsupported polynomial degree: {p}")
