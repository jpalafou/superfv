import pytest

from superfv.slope_limiting import compute_dmp
from superfv.tools.device_management import xp


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims, include_corners):
    """
    Test that compute_dmp returns the discrete maximum principle in all dimensions.
    """
    N = 64

    shape = (
        5,
        N if "x" in dims else 1,
        N if "y" in dims else 1,
        N if "z" in dims else 1,
    )
    arr = xp.random.rand(*shape)
    M = xp.empty(shape)
    m = xp.empty(shape)

    modified = compute_dmp(arr, M, m, tuple(dims), include_corners)

    assert xp.all(xp.less_equal(m, arr)[modified])
    assert xp.all(xp.greater_equal(M, arr)[modified])
