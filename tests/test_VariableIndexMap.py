import numpy as np
import pytest

from superfv.tools.variable_index_map import VariableIndexMap


def test_init_trivial():
    idx = VariableIndexMap()
    assert idx.idxs == []
    assert idx.nvars == 0
    assert idx.is_contiguous is True


def test_init_empty():
    idx = VariableIndexMap({}, {})
    assert idx.idxs == []
    assert idx.nvars == 0
    assert idx.is_contiguous is True


def test_init_with_no_var_idx_map():
    with pytest.raises(KeyError, match="Member 'var1' does not exist."):
        _ = VariableIndexMap({}, {"group1": ["var1", "var2"]})


def test_add_var_and_get_index():
    idx = VariableIndexMap({}, {})
    idx.add_var("u", 0)
    assert idx("u") == 0
    assert idx("u", keepdims=True) == slice(0, 1)


def test_add_member_to_group_and_get_index():
    idx = VariableIndexMap({"u": 0, "v": 1, "w": 2}, {})

    idx.add_member_to_group("u", "g1")
    idx.add_member_to_group("v", "g1")

    idx.add_member_to_group("u", "g2")
    idx.add_member_to_group("w", "g2")

    assert idx("g1") == slice(0, 2)
    assert np.array_equal(idx("g2"), np.array([0, 2]))


def test_variable_index_map_nested_group():
    vmap = VariableIndexMap({"a": 0, "b": 1, "c": 2, "d": 3}, {})

    vmap.add_member_to_group("a", "ab")
    vmap.add_member_to_group("b", "ab")

    vmap.add_member_to_group("c", "cd")
    vmap.add_member_to_group("d", "cd")

    vmap.add_member_to_group("ab", "abcd")
    vmap.add_member_to_group("cd", "abcd")

    assert vmap("abcd") == slice(0, 4)


def test_hydro_groups():
    idx = VariableIndexMap(
        {
            "rho": 0,
            "vx": 1,
            "vy": 2,
            "vz": 3,
            "P": 4,
            "E": 4,
            "mx": 1,
            "my": 2,
            "mz": 3,
            "passive1": 5,
        },
        {
            "v": ["vx", "vy", "vz"],
            "m": ["mx", "my", "mz"],
            "primitives": ["rho", "v", "P"],
            "conservatives": ["rho", "m", "E"],
            "passives": ["passive1"],
        },
    )

    assert idx.nvars == 6
    assert idx("rho") == 0
    assert idx("rho", keepdims=True) == slice(0, 1)
    assert idx("v") == slice(1, 4)
    assert idx("m") == slice(1, 4)
    assert idx("primitives") == slice(0, 5)
    assert idx("conservatives") == slice(0, 5)
    assert idx("passive1") == 5
    assert idx("passives") == slice(5, 6)
    assert idx.flattened_var_names("v") == ["vx", "vy", "vz"]
    assert idx.flattened_var_names("m") == ["mx", "my", "mz"]
    assert idx.flattened_var_names("primitives") == ["rho", "vx", "vy", "vz", "P"]
    assert idx.flattened_var_names("conservatives") == ["rho", "mx", "my", "mz", "E"]
    assert idx.flattened_var_names("passives") == ["passive1"]
    assert idx.is_in_group("vx", "v")
    assert idx.is_in_group("mx", "m")
    assert idx.is_in_group("vx", "primitives")
    assert idx.is_in_group("mx", "conservatives")
    assert idx.is_contiguous


def test_cycle_raises():
    idx = VariableIndexMap({"a": 0, "b": 1}, {})
    idx.add_member_to_group("a", "g1")
    idx.add_member_to_group("b", "g2")
    idx.add_member_to_group("g1", "g2")

    with pytest.raises(ValueError, match="Circular group reference detected."):
        idx.add_member_to_group("g2", "g1")


def test_add_var_with_same_name_and_index_does_nothing():
    idx1 = VariableIndexMap({"a": 0, "b": 1}, {})

    idx2 = idx1.copy()
    idx2.add_var("a", 0)
    idx2.add_var("b", 1)

    assert idx1 == idx2


def test_add_var_with_same_name_different_index_raises():
    idx = VariableIndexMap({"a": 0, "b": 1}, {})
    with pytest.raises(KeyError, match="Variable 'a' already exists with a different index."):
        idx.add_var("a", 2)


def test_add_var_with_group_name_raises():
    idx = VariableIndexMap({"a": 0, "b": 1}, {})
    idx.add_member_to_group("a", "g1")
    with pytest.raises(KeyError, match="Name 'g1' already exists as a group."):
        idx.add_var("g1", 2)


def test_add_redundant_group_member_does_nothing():
    idx1 = VariableIndexMap({"a": 0, "b": 1}, {})
    idx1.add_member_to_group("a", "g1")
    idx1.add_member_to_group("b", "g1")

    idx2 = idx1.copy()
    idx2.add_member_to_group("a", "g1")
    idx2.add_member_to_group("b", "g1")

    assert idx1 == idx2


def test_add_group_with_var_name_raises():
    idx = VariableIndexMap({"a": 0, "b": 1}, {})
    with pytest.raises(KeyError, match="Name 'a' already exists as a variable."):
        idx.add_member_to_group("b", "a")


def test_add_group_with_invalid_var():
    idx = VariableIndexMap({"a": 0, "b": 1}, {})
    with pytest.raises(KeyError, match="Member 'nonexistent' does not exist."):
        idx.add_member_to_group("nonexistent", "g1")


def test_equality_does_not_depend_on_cache():
    idx1 = VariableIndexMap({"a": 0, "b": 1}, {})
    idx1.add_member_to_group("a", "g1")
    idx2 = idx1.copy()  # cache is not be copied

    _ = idx1("g1")  # populate cache in just idx1
    assert idx1._cache != idx2._cache  # caches are different
    assert idx1 == idx2  # equality should not depend on cache


def test_call_with_unknown_name_raises():
    idx = VariableIndexMap({}, {})
    with pytest.raises(KeyError):
        idx("unknown")


def test_cache_behavior():
    idx = VariableIndexMap({"u": 0}, {})
    # First call populates cache
    _ = idx("u")
    # Now delete from map directly (simulate corruption)
    idx.var_idx_map.pop("u")
    # Cached result still valid
    assert idx("u") == 0
    # Invalidate cache
    idx._cache.clear()
    with pytest.raises(KeyError):
        idx("u")
