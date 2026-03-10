import numpy as np


def cell_center(p: int) -> np.ndarray:
    """
    Returns stencil weights for conservative interpolation of cell center for
    polynomial degree `p`.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Stencil weight array of shape (1, n).
    """
    if p in (0, 1):
        return np.array([[1.0]])
    if p in (2, 3):
        return np.array([[-1 / 24, 13 / 12, -1 / 24]])
    if p in (4, 5):
        return np.array([[3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]])
    if p in (6, 7):
        return np.array(
            [
                [
                    -5 / 7168,
                    159 / 17920,
                    -7621 / 107520,
                    30251 / 26880,
                    -7621 / 107520,
                    159 / 17920,
                    -5 / 7168,
                ]
            ]
        )
    raise NotImplementedError(f"Unsupported polynomial degree: {p}")


def left_right(p: int) -> np.ndarray:
    """
    Returns stencil weights for conservative interpolation of left and right cell faces
    for polynomial degree `p`.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Stencil weight array of shape (2, n).
    """
    if p == 0:
        wl = [1.0]
    elif p == 1:
        wl = [1 / 4, 1, -1 / 4]
    elif p == 2:
        wl = [1 / 3, 5 / 6, -1 / 6]
    elif p == 3:
        wl = [-1 / 24, 5 / 12, 5 / 6, -1 / 4, 1 / 24]
    elif p == 4:
        wl = [-1 / 20, 9 / 20, 47 / 60, -13 / 60, 1 / 30]
    elif p == 5:
        wl = [1 / 120, -1 / 12, 59 / 120, 47 / 60, -31 / 120, 1 / 15, -1 / 120]
    elif p == 6:
        wl = [1 / 105, -19 / 210, 107 / 210, 319 / 420, -101 / 420, 5 / 84, -1 / 140]
    elif p == 7:
        wl = [
            -1 / 560,
            17 / 840,
            -97 / 840,
            449 / 840,
            319 / 420,
            -223 / 840,
            71 / 840,
            -1 / 56,
            1 / 560,
        ]
    else:
        raise NotImplementedError(f"Unsupported polynomial degree: {p}")

    wr = wl[::-1]
    return np.array([wl, wr])


def n_gauss_legendre_nodes(p: int) -> int:
    """
    Returns the number of nodes for a Gauss-Legendre quadrature that is exact for
    polynomial degree `p` or lower.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Number of quadrature nodes.
    """
    return -(-(p + 1) // 2)


def gauss_legendre_nodes(p: int) -> np.ndarray:
    """
    Returns stencil weights for interpolating the nodes of a Gauss-Legendre quadrature
    for polynomial degree `p`.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Stencil weight array of shape (n_nodes, stencil_size).
    """
    if p in (0, 1):
        weights = np.array([[1.0]])
    elif p == 2:
        w0 = [np.sqrt(3) / 12, 1, -np.sqrt(3) / 12]
        w1 = w0[::-1]
        weights = np.array([w0, w1])
    elif p == 3:
        w0 = [
            -7 * np.sqrt(3) / 432,
            25 * np.sqrt(3) / 216,
            1,
            -25 * np.sqrt(3) / 216,
            7 * np.sqrt(3) / 432,
        ]
        w1 = w0[::-1]
        weights = np.array([w0, w1])
    elif p == 4:
        w0 = [
            -11 * np.sqrt(15) / 1200 - 3 / 800,
            29 / 600 + 41 * np.sqrt(15) / 600,
            1093 / 1200,
            29 / 600 - 41 * np.sqrt(15) / 600,
            -3 / 800 + 11 * np.sqrt(15) / 1200,
        ]
        w1 = [3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]
        w2 = w0[::-1]
        weights = np.array([w0, w1, w2])
    elif p == 5:
        w0 = [
            1363 * np.sqrt(15) / 720000,
            -3013 * np.sqrt(15) / 180000 - 3 / 800,
            29 / 600 + 11203 * np.sqrt(15) / 144000,
            1093 / 1200,
            29 / 600 - 11203 * np.sqrt(15) / 144000,
            -3 / 800 + 3013 * np.sqrt(15) / 180000,
            -1363 * np.sqrt(15) / 720000,
        ]
        w1 = [0, 3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640, 0]
        w2 = w0[::-1]
        weights = np.array([w0, w1, w2])
    elif p == 6:
        w0 = [
            -59 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 12348000
            + 307 / 1646400
            + 307 * np.sqrt(30) / 2744000
            + 7039 * np.sqrt(70 * np.sqrt(30) + 525) / 24696000,
            -15439 * np.sqrt(70 * np.sqrt(30) + 525) / 6174000
            - 1971 * np.sqrt(30) / 1372000
            - 657 / 274400
            + 223 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 6174000,
            -143 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 2469600
            + 6521 / 329280
            + 6521 * np.sqrt(30) / 548800
            + 55759 * np.sqrt(70 * np.sqrt(30) + 525) / 4939200,
            79423 / 82320 - 2897 * np.sqrt(30) / 137200,
            -55759 * np.sqrt(70 * np.sqrt(30) + 525) / 4939200
            + 143 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 2469600
            + 6521 / 329280
            + 6521 * np.sqrt(30) / 548800,
            -1971 * np.sqrt(30) / 1372000
            - 223 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 6174000
            - 657 / 274400
            + 15439 * np.sqrt(70 * np.sqrt(30) + 525) / 6174000,
            -7039 * np.sqrt(70 * np.sqrt(30) + 525) / 24696000
            + 307 / 1646400
            + 307 * np.sqrt(30) / 2744000
            + 59 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 12348000,
        ]
        w1 = [
            -307 * np.sqrt(30) / 2744000
            + 307 / 1646400
            + 59 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 12348000
            + 7039 * np.sqrt(525 - 70 * np.sqrt(30)) / 24696000,
            -15439 * np.sqrt(525 - 70 * np.sqrt(30)) / 6174000
            - 657 / 274400
            - 223 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 6174000
            + 1971 * np.sqrt(30) / 1372000,
            -6521 * np.sqrt(30) / 548800
            + 143 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 2469600
            + 6521 / 329280
            + 55759 * np.sqrt(525 - 70 * np.sqrt(30)) / 4939200,
            2897 * np.sqrt(30) / 137200 + 79423 / 82320,
            -55759 * np.sqrt(525 - 70 * np.sqrt(30)) / 4939200
            - 6521 * np.sqrt(30) / 548800
            - 143 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 2469600
            + 6521 / 329280,
            -657 / 274400
            + 223 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 6174000
            + 1971 * np.sqrt(30) / 1372000
            + 15439 * np.sqrt(525 - 70 * np.sqrt(30)) / 6174000,
            -7039 * np.sqrt(525 - 70 * np.sqrt(30)) / 24696000
            - 307 * np.sqrt(30) / 2744000
            - 59 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 12348000
            + 307 / 1646400,
        ]
        w2 = w1[::-1]
        w3 = w0[::-1]
        weights = np.array([w0, w1, w2, w3])
    elif p == 7:
        w0 = [
            -37831 * np.sqrt(70 * np.sqrt(30) + 525) / 605052000
            + 3177 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 2689120000,
            -143599 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 12101040000
            + 307 / 1646400
            + 307 * np.sqrt(30) / 2744000
            + 798883 * np.sqrt(70 * np.sqrt(30) + 525) / 1210104000,
            -9119 * np.sqrt(70 * np.sqrt(30) + 525) / 2701125
            - 1971 * np.sqrt(30) / 1372000
            - 657 / 274400
            + 91033 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 1728720000,
            -128693 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 1728720000
            + 6521 / 329280
            + 6521 * np.sqrt(30) / 548800
            + 700963 * np.sqrt(70 * np.sqrt(30) + 525) / 57624000,
            79423 / 82320 - 2897 * np.sqrt(30) / 137200,
            -700963 * np.sqrt(70 * np.sqrt(30) + 525) / 57624000
            + 128693 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 1728720000
            + 6521 / 329280
            + 6521 * np.sqrt(30) / 548800,
            -91033 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 1728720000
            - 1971 * np.sqrt(30) / 1372000
            - 657 / 274400
            + 9119 * np.sqrt(70 * np.sqrt(30) + 525) / 2701125,
            -798883 * np.sqrt(70 * np.sqrt(30) + 525) / 1210104000
            + 307 / 1646400
            + 307 * np.sqrt(30) / 2744000
            + 143599 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 12101040000,
            -3177 * np.sqrt(30) * np.sqrt(70 * np.sqrt(30) + 525) / 2689120000
            + 37831 * np.sqrt(70 * np.sqrt(30) + 525) / 605052000,
        ]
        w1 = [
            -37831 * np.sqrt(525 - 70 * np.sqrt(30)) / 605052000
            - 3177 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 2689120000,
            -307 * np.sqrt(30) / 2744000
            + 307 / 1646400
            + 143599 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 12101040000
            + 798883 * np.sqrt(525 - 70 * np.sqrt(30)) / 1210104000,
            -9119 * np.sqrt(525 - 70 * np.sqrt(30)) / 2701125
            - 91033 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 1728720000
            - 657 / 274400
            + 1971 * np.sqrt(30) / 1372000,
            -6521 * np.sqrt(30) / 548800
            + 128693 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 1728720000
            + 6521 / 329280
            + 700963 * np.sqrt(525 - 70 * np.sqrt(30)) / 57624000,
            2897 * np.sqrt(30) / 137200 + 79423 / 82320,
            -700963 * np.sqrt(525 - 70 * np.sqrt(30)) / 57624000
            - 6521 * np.sqrt(30) / 548800
            - 128693 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 1728720000
            + 6521 / 329280,
            -657 / 274400
            + 91033 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 1728720000
            + 1971 * np.sqrt(30) / 1372000
            + 9119 * np.sqrt(525 - 70 * np.sqrt(30)) / 2701125,
            -798883 * np.sqrt(525 - 70 * np.sqrt(30)) / 1210104000
            - 143599 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 12101040000
            - 307 * np.sqrt(30) / 2744000
            + 307 / 1646400,
            3177 * np.sqrt(30) * np.sqrt(525 - 70 * np.sqrt(30)) / 2689120000
            + 37831 * np.sqrt(525 - 70 * np.sqrt(30)) / 605052000,
        ]
        w2 = w1[::-1]
        w3 = w0[::-1]
        weights = np.array([w0, w1, w2, w3])
    else:
        raise NotImplementedError(
            "Conservative interpolation of Gauss-Legendre nodes not implemented for "
            f"{p=}"
        )

    n_computed, _ = weights.shape
    n_expected = n_gauss_legendre_nodes(p)
    if n_computed != n_expected:
        raise ValueError(
            f"Assigned {n_computed} nodes for a degree-{p} Gauss-Legendre "
            f"quadrature, but expected {n_expected} nodes."
        )

    return weights


def gauss_legendre_weights(p: int):
    """
    Returns the weights of a Gauss-Legendre quadrature that is exact for polynomial
    degree `p` or lower.

    Args:
        p: Polynomial degree (0 to 7).

    Returns:
        Quadrature weight array of shape (n,).
    """
    n = n_gauss_legendre_nodes(p)

    if n < 1:
        raise ValueError(
            f"Cannot return Gauss-Legendre weights for {p=} with {n} nodes."
        )

    match n:
        case 1:
            weights = np.array([1.0])
        case 2:
            weights = np.array([0.5, 0.5])
        case 3:
            weights = np.array([5 / 18, 4 / 9, 5 / 18])
        case 4:
            w0 = (18 - np.sqrt(30)) / 72
            w1 = (18 + np.sqrt(30)) / 72
            w2 = w1
            w3 = w0
            weights = np.array([w0, w1, w2, w3])
        case 5:
            w0 = (322 - 13 * np.sqrt(70)) / 1800
            w1 = (322 + 13 * np.sqrt(70)) / 1800
            w2 = 64.0 / 225
            w3 = w1
            w4 = w0
            weights = np.array([w0, w1, w2, w3, w4])
        case _:
            raise NotImplementedError(
                f"Gauss-Legendre weights not implemented for {p=} with {n} nodes."
            )

    n_computed = weights.size
    n_expected = n_gauss_legendre_nodes(p)
    if n_computed != n_expected:
        raise ValueError(
            f"Assigned {n_computed} weights for a degree-{p} Gauss-Legendre "
            f"quadrature, but expected {n_expected} weights."
        )

    return weights
