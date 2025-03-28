import numpy as np
import pytest

from superfv.tools.array_management import ArraySlicer


def test_initialization():
    var_idx_map = {"density": 0, "momentum": 1}
    slicer = ArraySlicer(var_idx_map, ndim=3)
    assert slicer.var_idx_map == var_idx_map
    assert slicer.ndim == 3
    assert slicer.var_names == {"density", "momentum"}
    assert slicer.group_names == set()
    assert slicer.all_names == {"density", "momentum"}


def test_add_var():
    slicer = ArraySlicer({"density": 0}, ndim=3)
    slicer.add_var("momentum", 1)
    assert slicer.var_idx_map["momentum"] == 1
    assert slicer.var_names == {"density", "momentum"}
    assert slicer.group_names == set()
    assert slicer.all_names == {"density", "momentum"}
    assert slicer.idxs == {0, 1}

    with pytest.raises(ValueError, match="Variable 'momentum' already exists."):
        slicer.add_var("momentum", 2)


def test_create_var_group():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)

    # Create a contiguous group
    slicer.create_var_group("fluid", ("density", "momentum"))
    assert isinstance(slicer.var_idx_map["fluid"], slice)
    assert slicer.var_idx_map["fluid"] == slice(0, 2)
    assert slicer.var_names == {"density", "momentum", "energy"}
    assert slicer.group_names == {"fluid"}
    assert slicer.all_names == {"density", "momentum", "energy", "fluid"}
    assert slicer.idxs == {0, 1, 2}

    # Create a non-contiguous group
    slicer.create_var_group("custom", ("density", "energy"))
    assert isinstance(slicer.var_idx_map["custom"], np.ndarray)
    assert slicer.var_names == {"density", "momentum", "energy"}
    assert slicer.group_names == {"fluid", "custom"}
    assert slicer.all_names == {"density", "momentum", "energy", "fluid", "custom"}
    assert slicer.idxs == {0, 1, 2}
    np.testing.assert_array_equal(slicer.var_idx_map["custom"], np.array([0, 2]))

    # Ensure error is raised if a variable is not found
    with pytest.raises(ValueError, match="Variables not found: .*"):
        slicer.create_var_group("invalid_group", ("density", "pressure"))

    # Ensure error is raised if the group name already exists
    with pytest.raises(ValueError, match="Name 'fluid' already exists."):
        slicer.create_var_group("fluid", ("density", "momentum"))


def test_add_to_var_group():
    slicer = ArraySlicer({"rho": 0, "mx": 1, "my": 2, "mz": 3, "E": 4}, ndim=4)
    slicer.create_var_group("m", ("mx",))

    # Add variables
    slicer.add_to_var_group("m", ("my", "mz"))
    assert slicer.var_idx_map["m"] == slice(1, 4)
    assert slicer.var_names == {"rho", "mx", "my", "mz", "E"}
    assert slicer.group_names == {"m"}
    assert slicer.all_names == {"rho", "mx", "my", "mz", "E", "m"}
    assert slicer.idxs == {0, 1, 2, 3, 4}

    # Ensure error is raised if a variable is not found
    with pytest.raises(ValueError, match="Variables not found: .*"):
        slicer.add_to_var_group("m", ("P",))

    # Ensure error is raised if the group name does not exist
    with pytest.raises(ValueError, match="Group 'invalid_group' not found."):
        slicer.add_to_var_group("invalid_group", ("E",))


def test_call_single_variable():
    slicer = ArraySlicer({"density": 0, "momentum": 1}, ndim=3)
    result = slicer(variable="density")
    assert result == 0

    slicer = ArraySlicer({"fluid": slice(0, 2)}, ndim=3)
    result = slicer(variable="fluid")
    assert result == slice(0, 2)


def test_call_multiple_variables():
    slicer = ArraySlicer(
        {"density": 0, "momentum": 1, "energy": 2},
        groups={"fluid": ("density", "momentum")},
        ndim=3,
    )

    # Case 1: Contiguous indices
    result = slicer(variable=("density", "momentum"))
    assert result == slice(0, 2)

    # Case 2: Non-contiguous indices
    result = slicer(variable=("density", "energy"))
    np.testing.assert_array_equal(result, np.array([0, 2]))


def test_call_with_axes():
    slicer = ArraySlicer({"density": 0}, ndim=4)
    result = slicer(
        variable="density", x=(0, 5), y=(10, 20), axis=3, cut=(20, 40), step=2
    )
    assert result == (0, slice(None, 5), slice(10, 20), slice(20, 40, 2))

    slicer = ArraySlicer({"density": 0}, ndim=2)
    with pytest.raises(
        ValueError, match="Invalid axis .* for array with .* dimensions."
    ):
        slicer(variable="density", z=(None, 5))


def test_call_mixed_cases():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)

    # Mixed slices with step
    result = slicer(variable=("density", "momentum"), x=(0, 5), y=(10, 20))
    assert result == (slice(0, 2), slice(None, 5), slice(10, 20))

    # Single variable with full slices
    result = slicer(variable="density", x=(5, 15))
    assert result == (0, slice(5, 15), slice(None))

    # Invalid variable
    with pytest.raises(ValueError, match="Variable 'pressure' not found."):
        slicer(variable="pressure")


def test_call_keepdims():
    arr = np.empty((3, 10, 10))
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)
    assert arr[slicer("density")].shape == (10, 10)
    assert arr[slicer("density", keepdims=True)].shape == (1, 10, 10)


def test_call_lru_cache():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)
    initial_hits = slicer.__call__.cache_info().hits
    for _ in range(10):
        slicer(variable="density", x=(0, 5), y=(10, 20))
    assert slicer.__call__.cache_info().hits == initial_hits + 9


def test_hash():
    slicer = ArraySlicer({}, ndim=3)
    assert hash(slicer) == id(slicer)


def test_copy():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)
    slicer.create_var_group("fluid", ("density", "momentum"))

    slicer_copy = slicer.copy()
    assert slicer_copy.var_idx_map == slicer.var_idx_map
    assert slicer_copy.ndim == slicer.ndim
    assert slicer_copy.var_names == slicer.var_names
    assert slicer_copy.group_names == slicer.group_names
    assert slicer_copy.all_names == slicer.all_names
    assert slicer_copy.idxs == slicer.idxs
    assert slicer_copy.var_idx_map["fluid"] == slicer.var_idx_map["fluid"]
    assert slicer_copy is not slicer
    assert slicer_copy.var_idx_map is not slicer.var_idx_map
    assert slicer_copy.var_names is not slicer.var_names
    assert slicer_copy.group_names is not slicer.group_names
    assert slicer_copy.idxs is not slicer.idxs

    slicer.add_var("pressure", 3)
    assert "pressure" not in slicer_copy.var_names
    assert "pressure" not in slicer_copy.all_names
    assert 3 not in slicer_copy.idxs
