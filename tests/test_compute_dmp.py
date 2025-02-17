import numpy as np
import pytest

from superfv.slope_limiting import compute_dmp
from superfv.tools.array_management import crop_to_center


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims, include_corners):
    shape = (
        3,
        10 if "x" in dims else 1,
        10 if "y" in dims else 1,
        10 if "z" in dims else 1,
    )
    arr = np.random.rand(*shape)
    min_vals, max_vals = compute_dmp(arr, dims=dims, include_corners=include_corners)
    assert min_vals.shape == (
        shape[0],
        shape[1] - 2 * int("x" in dims),
        shape[2] - 2 * int("y" in dims),
        shape[3] - 2 * int("z" in dims),
    )
    assert max_vals.shape == (
        shape[0],
        shape[1] - 2 * int("x" in dims),
        shape[2] - 2 * int("y" in dims),
        shape[3] - 2 * int("z" in dims),
    )
    assert np.all(min_vals <= crop_to_center(arr, min_vals.shape, axes=(1, 2, 3)))
    assert np.all(max_vals >= crop_to_center(arr, min_vals.shape, axes=(1, 2, 3)))
