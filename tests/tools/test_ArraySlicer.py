import numpy as np
import pytest

from shine.tools.array_management import ArraySlicer


def test_initialization():
    var_idx_map = {"density": 0, "momentum": 1}
    slicer = ArraySlicer(var_idx_map, ndim=3)
    assert slicer.var_idx_map == var_idx_map
    assert slicer.ndim == 3


def test_add_var():
    slicer = ArraySlicer({"density": 0}, ndim=3)
    slicer.add_var("momentum", 1)
    assert slicer.var_idx_map["momentum"] == 1
    assert slicer.var_names == {"density", "momentum"}
    assert slicer.idxs == {0, 1}

    with pytest.raises(ValueError, match="Variable 'momentum' already exists."):
        slicer.add_var("momentum", 2)


def test_create_var_group():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)

    # Create a contiguous group
    slicer.create_var_group("fluid", ("density", "momentum"))
    assert isinstance(slicer.var_idx_map["fluid"], slice)
    assert slicer.var_idx_map["fluid"] == slice(0, 2)
    assert slicer.var_names == {"density", "momentum", "energy", "fluid"}
    assert slicer.idxs == {0, 1, 2}

    # Create a non-contiguous group
    slicer.create_var_group("custom", ("density", "energy"))
    assert isinstance(slicer.var_idx_map["custom"], np.ndarray)
    assert slicer.var_names == {"density", "momentum", "energy", "fluid", "custom"}
    assert slicer.idxs == {0, 1, 2}
    np.testing.assert_array_equal(slicer.var_idx_map["custom"], np.array([0, 2]))

    # Ensure error is raised if a variable is not found
    with pytest.raises(ValueError, match="Variables not found: .*"):
        slicer.create_var_group("invalid_group", ("density", "pressure"))

    # Ensure error is raised if the group name already exists
    with pytest.raises(ValueError, match="Variable group 'fluid' already exists."):
        slicer.create_var_group("fluid", ("density", "momentum"))


def test_call_single_variable():
    slicer = ArraySlicer({"density": 0, "momentum": 1}, ndim=3)
    result = slicer(var="density")
    assert result == 0

    slicer = ArraySlicer({"fluid": slice(0, 2)}, ndim=3)
    result = slicer(var="fluid")
    assert result == slice(0, 2)


def test_call_multiple_variables():
    slicer = ArraySlicer(
        {"density": 0, "momentum": 1, "energy": 2, "fluid": np.array([0, 1])}, ndim=3
    )

    # Case 1: Contiguous indices
    result = slicer(var=("density", "momentum"))
    assert result == slice(0, 2)

    # Case 2: Non-contiguous indices
    result = slicer(var=("density", "energy"))
    np.testing.assert_array_equal(result, np.array([0, 2]))


def test_call_with_axes():
    slicer = ArraySlicer({"density": 0}, ndim=4)
    result = slicer(var="density", x=(0, 5), y=(10, 20), axis=3, cut=(20, 40), step=2)
    assert result == (0, slice(None, 5), slice(10, 20), slice(20, 40, 2))

    slicer = ArraySlicer({"density": 0}, ndim=2)
    with pytest.raises(
        ValueError, match="Invalid axis .* for array with .* dimensions."
    ):
        slicer(var="density", z=(None, 5))


def test_call_mixed_cases():
    slicer = ArraySlicer({"density": 0, "momentum": 1, "energy": 2}, ndim=3)

    # Mixed slices with step
    result = slicer(var=("density", "momentum"), x=(0, 5), y=(10, 20))
    assert result == (slice(0, 2), slice(None, 5), slice(10, 20))

    # Single variable with full slices
    result = slicer(var="density", x=(5, 15))
    assert result == (0, slice(5, 15), slice(None))

    # Invalid variable
    with pytest.raises(ValueError, match="Variable 'pressure' not found."):
        slicer(var="pressure")


def test_hash():
    slicer = ArraySlicer({}, ndim=3)
    assert hash(slicer) == id(slicer)
