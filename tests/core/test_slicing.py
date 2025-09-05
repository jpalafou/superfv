import numpy as np
import pytest
from numpy.testing import assert_array_equal

from superfv.tools.slicing import (
    _crop_to_center,
    crop,
    crop_to_center,
    intersection_shape,
    merge_indices,
    merge_slices,
)

# ---------- Tests for crop ----------


def test_crop_single_axis():
    result = crop(1, (2, 5), step=1, ndim=3)
    assert result == (slice(None), slice(2, 5, 1), slice(None))


def test_crop_multiple_axes():
    result = crop((0, 2), (1, 4), step=None, ndim=4)
    assert result == (slice(1, 4, None), slice(None), slice(1, 4, None), slice(None))


def test_crop_invalid_axis():
    with pytest.raises(ValueError):
        crop(5, (1, 2), ndim=3)


# ----------- Tests for _crop_to_center ----------


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


# ---------- Tests for crop_to_center ----------


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


# ---------- Tests for intersection_shape ----------


def test_intersection_shape():
    s1 = (5, 10, 8)
    s2 = (6, 9, 7)
    s3 = (4, 8, 9)
    expected = (4, 8, 7)
    result = intersection_shape(s1, s2, s3)
    assert result == expected


# ---------- Tests for merge_indices ----------


def test_merge_indices_ints():
    assert merge_indices(1, 2, 3) == slice(1, 4)
    assert np.array_equal(merge_indices(3, 1, 2, as_array=True), np.array([1, 2, 3]))


def test_merge_indices_slices():
    assert merge_indices(slice(0, 3), 3) == slice(0, 4)
    assert np.array_equal(
        merge_indices(slice(0, 2), 4, as_array=True), np.array([0, 1, 4])
    )


def test_merge_indices_sequences():
    assert merge_indices([0, 1], (2, 3), np.array([4, 5])) == slice(0, 6)
    assert np.array_equal(
        merge_indices([0, 1], (2, 3), np.array([4, 5]), as_array=True), np.arange(6)
    )


def test_merge_empty():
    assert merge_indices(()) == slice(0, 0)
    assert merge_indices([]) == slice(0, 0)
    assert np.array_equal(merge_indices([], as_array=True), np.array([], dtype=np.int_))


def test_merge_indices_invalid():
    with pytest.raises(ValueError):
        merge_indices(-1)
    with pytest.raises(ValueError):
        merge_indices(slice(-1, 3))
    with pytest.raises(ValueError):
        merge_indices(slice(1, None))
    with pytest.raises(ValueError):
        merge_indices(slice(1, 3, 2))
    with pytest.raises(ValueError):
        merge_indices([1.0, 2.0])
    with pytest.raises(TypeError):
        merge_indices("bad")
    with pytest.raises(ValueError):
        merge_indices(np.array([[1, 2, 3], [4, 5, 6]]))


# ---------- Tests for merge_slices ----------


def test_merge_slices_basic_1d():
    s1 = (slice(2, 5),)
    s2 = (slice(1, 4),)
    assert merge_slices(s1, s2) == (slice(2, 4),)
    assert merge_slices(s1, s2, union=True) == (slice(1, 5),)


def test_merge_slices_basic_2d():
    s1 = (slice(2, 5), slice(1, 3))
    s2 = (slice(1, 4), slice(2, 8))
    assert merge_slices(s1, s2) == (slice(2, 4), slice(2, 3))
    assert merge_slices(s1, s2, union=True) == (slice(1, 5), slice(1, 8))


def test_merge_slices_different_lengths():
    s1 = (slice(1, 3),)
    s2 = (slice(2, 5), slice(3, 9))
    assert merge_slices(s1, s2) == (slice(2, 3), slice(3, 9))
    assert merge_slices(s1, s2, union=True) == (slice(1, 5), slice(None, None))


def test_merge_slices_none_start_or_stop():
    s1 = (slice(None, 5), slice(2, 8))
    s2 = (slice(1, None), slice(1, 4))
    result = merge_slices(s1, s2)
    assert result[0].start == 1
    assert result[0].stop == 5
    assert result[1].start == 2
    assert result[1].stop == 4
    result_union = merge_slices(s1, s2, union=True)
    assert result_union[0].start is None
    assert result_union[0].stop is None
    assert result_union[1].start == 1
    assert result_union[1].stop == 8


def test_merge_slices_all_none():
    s1 = (slice(None, None), slice(None, None))
    s2 = (slice(None, None), slice(None, None))
    assert merge_slices(s1, s2) == (slice(None, None), slice(None, None))
    assert merge_slices(s1, s2, union=True) == (slice(None, None), slice(None, None))


def test_merge_slices_single_slice():
    s = (slice(3, 7), slice(4, 9))
    assert merge_slices(s) == s
    assert merge_slices(s, union=True) == s


def test_merge_slices_step_is_ignored():
    s1 = (slice(0, 10, 2),)
    s2 = (slice(5, 15, 3),)
    result = merge_slices(s1, s2)
    assert result == (slice(5, 10),)  # step is intentionally ignored
    assert merge_slices(s1, s2, union=True) == (slice(0, 15),)  # step ignored in union
