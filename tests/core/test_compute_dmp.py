import numpy as np
import pytest

from superfv.slope_limiting import compute_dmp


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
    arr = np.random.rand(*shape)
    out = np.empty(shape + (2,))

    modified = compute_dmp(
        np, arr, tuple(dims), out=out, include_corners=include_corners
    )
    min_vals = out[:, :, :, :, 0]
    max_vals = out[:, :, :, :, 1]

    assert np.all(np.less_equal(min_vals, arr)[modified[:-1]])
    assert np.all(np.greater_equal(max_vals, arr)[modified[:-1]])
