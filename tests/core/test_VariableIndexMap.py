import numpy as np
import pytest

from superfv.tools.variable_index_map import VariableIndexMap


def test_add_var_and_get_index():
    idx = VariableIndexMap({}, {})
    idx.add_var("u", 0)
    assert idx("u") == 0
    assert idx("u", keepdims=True) == slice(0, 1)


def test_add_var_to_group_and_get_index():
    idx = VariableIndexMap({"u": 0, "v": 1, "w": 2}, {})

    idx.add_var_to_group("u", "g1")
    idx.add_var_to_group("v", "g1")

    idx.add_var_to_group("u", "g2")
    idx.add_var_to_group("w", "g2")

    assert idx("g1") == slice(0, 2)
    assert np.array_equal(idx("g2"), np.array([0, 2]))


def test_init():
    idx = VariableIndexMap(
        {"a": 0, "b": 1, "c": 2}, {"ab": ["a", "b"], "bc": ["b", "c"], "ac": ["a", "c"]}
    )
    assert idx("a") == 0
    assert idx("b") == 1
    assert idx("c") == 2
    assert idx("ab") == slice(0, 2)
    assert idx("bc") == slice(1, 3)
    assert np.array_equal(idx("ac"), np.array([0, 2]))


def test_variable_index_map_nested_group():
    vmap = VariableIndexMap({"a": 0, "b": 1, "c": 2, "d": 3}, {})

    vmap.add_var_to_group("a", "ab")
    vmap.add_var_to_group("b", "ab")

    vmap.add_var_to_group("c", "cd")
    vmap.add_var_to_group("d", "cd")

    vmap.add_var_to_group("ab", "abcd")
    vmap.add_var_to_group("cd", "abcd")

    assert vmap("abcd") == slice(0, 4)


def test_empty_groups():
    idx = VariableIndexMap(
        {"rho": 0, "vx": 1, "vy": 2, "vz": 3},
        {
            "primitives": ["rho", "v"],
            "conservatives": [],
            "v": ["v" + dim for dim in "xyz"],
        },
    )

    assert idx.all_names == [
        "rho",
        "vx",
        "vy",
        "vz",
        "primitives",
        "conservatives",
        "v",
    ]
    assert np.array_equal(idx.idxs, np.arange(4))
    assert idx.nvars == 4

    assert idx("rho") == 0
    assert idx("rho", keepdims=True) == slice(0, 1)
    assert idx("v") == slice(1, 4)
    assert idx("primitives") == slice(0, 4)

    with pytest.raises(ValueError, match="Group 'conservatives' has no members."):
        idx("conservatives")


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
        },
        {},
    )

    assert idx.all_names == ["rho", "vx", "vy", "vz", "P", "E", "mx", "my", "mz"]
    assert np.array_equal(idx.idxs, np.arange(5))
    assert idx.nvars == 5

    idx.add_var_to_group("vx", "v")
    idx.add_var_to_group("vy", "v")
    idx.add_var_to_group("vz", "v")

    idx.add_var_to_group("mx", "m")
    idx.add_var_to_group("my", "m")
    idx.add_var_to_group("mz", "m")

    idx.add_var_to_group("rho", "primitives")
    idx.add_var_to_group("v", "primitives")
    idx.add_var_to_group("P", "primitives")

    idx.add_var_to_group("rho", "conservatives")
    idx.add_var_to_group("m", "conservatives")
    idx.add_var_to_group("E", "conservatives")

    idx.add_var_to_group("rho", "state")
    idx.add_var_to_group("E", "state")
    idx.add_var_to_group("P", "state")

    assert idx("rho") == 0
    assert idx("rho", keepdims=True) == slice(0, 1)
    assert idx("v") == slice(1, 4)
    assert idx("m") == slice(1, 4)
    assert idx("primitives") == slice(0, 5)
    assert idx("conservatives") == slice(0, 5)
    assert np.array_equal(idx("state"), np.array([0, 4]))

    assert idx.all_names == [
        "rho",
        "vx",
        "vy",
        "vz",
        "P",
        "E",
        "mx",
        "my",
        "mz",
        "v",
        "m",
        "primitives",
        "conservatives",
        "state",
    ]
    assert np.array_equal(idx.idxs, np.arange(5))
    assert idx.nvars == 5


def test_add_var_duplicate_raises():
    idx = VariableIndexMap({}, {})
    idx.add_var("u", 0)
    with pytest.raises(KeyError):
        idx.add_var("u", 1)


def test_add_group_with_invalid_var():
    idx = VariableIndexMap({}, {})
    idx.add_var("u", 0)
    with pytest.raises(KeyError, match="Member 'nonexistent' not found as variable or group."):
        idx.add_var_to_group("u", "badgroup")
        idx.add_var_to_group("nonexistent", "badgroup")


def test_group_name_conflicts_with_variable():
    idx = VariableIndexMap({}, {})
    idx.add_var("density", 0)
    with pytest.raises(KeyError, match="Name 'density' already exists."):
        idx.add_var_to_group("density", "density")


def test_variable_name_is_group_name():
    with pytest.raises(KeyError, match="Variables and groups cannot share names."):
        VariableIndexMap({"a": 0}, {"a": ["a"]})


def test_group_membership_ordering_is_sorted():
    idx = VariableIndexMap({}, {})
    idx.add_var("x", 0)
    idx.add_var("y", 1)
    idx.add_var("z", 2)

    idx.add_var_to_group("z", "coord")
    idx.add_var_to_group("x", "coord")
    idx.add_var_to_group("y", "coord")

    out = idx("coord")
    assert out == slice(0, 3)


def test_call_with_unknown_name_raises():
    idx = VariableIndexMap({}, {})
    with pytest.raises(KeyError):
        idx("unknown")


def test_cache_behavior():
    idx = VariableIndexMap({}, {})
    idx.add_var("u", 0)
    # First call populates cache
    _ = idx("u")
    # Now delete from map directly (simulate corruption)
    idx.var_idx_map.pop("u")
    # Cached result still valid
    assert idx("u") == 0
    # Invalidate cache
    idx.__post_init__()
    with pytest.raises(KeyError):
        idx("u")


def test_variable_index_map_cycle():
    vmap = VariableIndexMap({"x": 0, "y": 1, "z": 2}, {"g1": ["x", "g2"], "g2": ["y"]})
    with pytest.raises(ValueError, match="Circular group reference detected."):
        vmap.add_var_to_group("g1", "g2")  # create cycle
