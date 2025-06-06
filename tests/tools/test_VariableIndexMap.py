import numpy as np
import pytest

from superfv.tools.array_management import VariableIndexMap, merge_indices

# ---------- Tests for merge_indices ----------


def test_merge_single_int():
    assert merge_indices(5) == slice(5, 6)


def test_merge_single_int_as_array():
    assert merge_indices(5, as_array=True) == np.array(5)


def test_merge_multiple_ints_contiguous():
    assert merge_indices(1, 2, 3) == slice(1, 4)


def test_merge_multiple_ints_non_contiguous():
    result = merge_indices(1, 3, 5)
    np.testing.assert_array_equal(result, np.array([1, 3, 5]))


def test_merge_slices_contiguous():
    assert merge_indices(slice(0, 3), slice(3, 5)) == slice(0, 5)


def test_merge_slices_non_contiguous():
    result = merge_indices(slice(0, 2), slice(3, 5))
    np.testing.assert_array_equal(result, np.array([0, 1, 3, 4]))


def test_merge_mixed_indices():
    result = merge_indices(0, slice(1, 3), np.array([3, 4]))
    assert isinstance(result, slice)
    assert result == slice(0, 5)


def test_merge_as_array_forces_array_output():
    result = merge_indices(1, 2, 3, as_array=True)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_merge_invalid_int_negative():
    with pytest.raises(ValueError):
        merge_indices(-1)


def test_merge_invalid_slice_step():
    with pytest.raises(ValueError):
        merge_indices(slice(0, 5, 2))


def test_merge_invalid_ndarray_dtype():
    with pytest.raises(ValueError):
        merge_indices(np.array([1.0, 2.0]))


# ---------- Tests for VariableIndexMap ----------


def test_add_var_and_get_index():
    idx = VariableIndexMap()
    idx.add_var("u", 0)
    assert idx("u") == 0
    assert idx("u", keepdims=True) == slice(0, 1)


def test_add_var_duplicate_raises():
    idx = VariableIndexMap()
    idx.add_var("u", 0)
    with pytest.raises(KeyError):
        idx.add_var("u", 1)


def test_add_var_negative_index():
    idx = VariableIndexMap()
    with pytest.raises(ValueError):
        idx.add_var("bad", -1)


def test_add_group_and_query():
    idx = VariableIndexMap()
    idx.add_var("u", 0)
    idx.add_var("v", 1)
    idx.add_var_to_group(["u", "v"], "momentum")
    out = idx("momentum")
    assert out == slice(0, 2)


def test_add_group_with_invalid_var():
    idx = VariableIndexMap()
    idx.add_var("u", 0)
    with pytest.raises(ValueError):
        idx.add_var_to_group(["u", "ghost"], "badgroup")


def test_group_name_conflicts_with_variable():
    idx = VariableIndexMap()
    idx.add_var("density", 0)
    with pytest.raises(KeyError):
        idx.add_var_to_group("density", "density")


def test_group_membership_ordering_is_sorted():
    idx = VariableIndexMap()
    idx.add_var("x", 0)
    idx.add_var("y", 1)
    idx.add_var("z", 2)
    idx.add_var_to_group(["z", "x", "y"], "coord")
    out = idx("coord")
    assert out == slice(0, 3)


def test_group_with_duplicate_vars_raises():
    idx = VariableIndexMap()
    idx.add_var("x", 0)
    idx.add_var("y", 1)
    with pytest.raises(ValueError):
        idx.add_var_to_group(["x", "y", "x"], "duplicate_group")


def test_call_with_unknown_name_raises():
    idx = VariableIndexMap()
    with pytest.raises(ValueError):
        idx("unknown")


def test_cache_behavior():
    idx = VariableIndexMap()
    idx.add_var("u", 0)
    # First call populates cache
    _ = idx("u")
    # Now delete from map directly (simulate corruption)
    idx.var_idx_map.pop("u")
    # Cached result still valid
    assert idx("u") == 0
    # Invalidate cache
    idx._invalidate_cache()
    with pytest.raises(ValueError):
        idx("u")


def test_hydro():
    idx = VariableIndexMap(
        var_idx_map={
            "rho": 0,
            "vx": 1,
            "vy": 2,
            "vz": 3,
            "P": 4,
            "mx": 1,
            "my": 2,
            "mz": 3,
            "E": 4,
        },
        group_var_map={
            "v": ["vx", "vy", "vz"],
            "m": ["mx", "my", "mz"],
            "primitives": ["rho", "vx", "vy", "vz", "P"],
            "conservatives": ["rho", "mx", "my", "mz", "E"],
            "noncontiguous": ["vx", "mx", "E"],
        },
    )
    assert idx("rho") == 0
    assert idx("rho", keepdims=True) == slice(0, 1)
    assert idx("v") == slice(1, 4)
    assert idx("m") == slice(1, 4)
    np.testing.assert_array_equal(idx("noncontiguous"), np.array([1, 4]))
    np.testing.assert_array_equal(idx("primitives"), idx("conservatives"))
