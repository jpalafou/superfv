import numpy as np
import pytest

from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import CUPY_AVAILABLE, xp


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
@pytest.mark.parametrize("cupy", [False, True])
def test_compute_dmp(dims, include_corners, cupy):
    """
    Test that compute_dmp returns the discrete maximum principle in all dimensions.
    """
    if cupy and not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available")

    N = 64

    shape = (
        5,
        N if "x" in dims else 1,
        N if "y" in dims else 1,
        N if "z" in dims else 1,
    )
    arr = xp.random.rand(*shape) if cupy else np.random.rand(*shape)
    M = xp.empty(shape) if cupy else np.empty(shape)
    m = xp.empty(shape) if cupy else np.empty(shape)

    modified = compute_dmp(
        xp if cupy else np, arr, tuple(dims), include_corners, M=M, m=m
    )

    if cupy:
        assert xp.all(xp.less_equal(m, arr)[modified])
        assert xp.all(xp.greater_equal(M, arr)[modified])
    else:
        assert np.all(np.less_equal(m, arr)[modified])
        assert np.all(np.greater_equal(M, arr)[modified])
