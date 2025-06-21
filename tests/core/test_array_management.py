import numpy as np
import pytest
from numpy.testing import assert_array_equal

from superfv.tools.array_management import (
    _crop_to_center,
    crop,
    crop_to_center,
    intersection_shape,
    l1_norm,
    l2_norm,
    linf_norm,
)


def test_l1_norm():
    arr = np.array([-1, 2, -3, 4])
    assert l1_norm(arr) == pytest.approx(2.5)


def test_l2_norm():
    arr = np.array([3, 4])
    assert l2_norm(arr) == pytest.approx(5.0 / np.sqrt(2))


def test_linf_norm():
    arr = np.array([-1, -9, 5])
    assert linf_norm(arr) == 9


def test_crop_single_axis():
    result = crop(1, (2, 5), step=1, ndim=3)
    assert result == (slice(None), slice(2, 5, 1), slice(None))


def test_crop_multiple_axes():
    result = crop((0, 2), (1, 4), step=None, ndim=4)
    assert result == (slice(1, 4, None), slice(None), slice(1, 4, None), slice(None))


def test_crop_invalid_axis():
    with pytest.raises(ValueError):
        crop(5, (1, 2), ndim=3)


def test__crop_to_center_even_crop():
    in_shape = (10, 10)
    target = (6, 6)
    expected = (slice(2, -2), slice(2, -2))
    result = _crop_to_center(in_shape, target)
    assert result == expected


def test__crop_to_center_invalid_shape():
    with pytest.raises(ValueError):
        _crop_to_center((5, 5), (6, 5))


def test__crop_to_center_ignore_axis():
    in_shape = (8, 10)
    target = (8, 6)
    result = _crop_to_center(in_shape, target, ignore_axes=0)
    assert result == (slice(None), slice(2, -2))


def test_crop_to_center():
    arr = np.arange(100).reshape(10, 10)
    cropped = crop_to_center(arr, (6, 6))
    assert cropped.shape == (6, 6)
    assert_array_equal(cropped, arr[2:-2, 2:-2])


def test_crop_to_center_ignore_axis():
    arr = np.arange(100).reshape(10, 10)
    cropped = crop_to_center(arr, (10, 6), ignore_axes=0)
    assert cropped.shape == (10, 6)
    assert_array_equal(cropped, arr[:, 2:-2])


def test_intersection_shape():
    s1 = (5, 10, 8)
    s2 = (6, 9, 7)
    s3 = (4, 8, 9)
    expected = (4, 8, 7)
    result = intersection_shape(s1, s2, s3)
    assert result == expected
