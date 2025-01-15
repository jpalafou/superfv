import numpy as np
import pytest

from shine.tools.array_management import CUPY_AVAILABLE, ArrayManager


@pytest.fixture
def array_manager():
    """Fixture to create an ArrayManager instance."""
    return ArrayManager()


def test_initial_state(array_manager):
    """Test that the initial state is correct."""
    assert array_manager.arrays == {}
    assert not array_manager.using_cupy


def test_enable_cupy(array_manager):
    """Test enabling CuPy."""
    array_manager.enable_cupy()
    if CUPY_AVAILABLE:
        assert array_manager.using_cupy
    else:
        assert not array_manager.using_cupy


def test_disable_cupy(array_manager):
    """Test disabling CuPy."""
    array_manager.enable_cupy()
    array_manager.disable_cupy()
    assert not array_manager.using_cupy


def test_add_array(array_manager):
    """Test adding a new NumPy array."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    assert "test_array" in array_manager.arrays


def test_add_existing_array_raises_error(array_manager):
    """Test adding an array with an existing name raises a KeyError."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    with pytest.raises(KeyError):
        array_manager.add("test_array", array)


def test_add_non_numpy_array_raises_error(array_manager):
    """Test adding a non-NumPy array raises a TypeError."""
    with pytest.raises(TypeError):
        array_manager.add("invalid_array", "this is not an array")


def test_rm_array(array_manager):
    """Test removing an array."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    array_manager.rm("test_array")
    assert "test_array" not in array_manager.arrays


def test_rm_non_existing_array_raises_error(array_manager):
    """Test removing a non-existing array raises a KeyError."""
    with pytest.raises(KeyError):
        array_manager.rm("non_existent_array")


def test_clear_all_arrays(array_manager):
    """
    Test clearing all arrays.
    """
    for i in range(20):
        array = np.random.rand(5, 5)
        array_manager.add(f"test_array_{i}", array)
    array_manager.clear()
    assert len(array_manager.arrays) == 0


def test_clear_all_arrays_except_some(array_manager):
    """
    Test clearing all arrays except some.
    """
    for i in range(20):
        array = np.random.rand(5, 5)
        array_manager.add(f"test_array_{i}", array)
    array_manager.clear(all_but=["test_array_5", "test_array_10"])
    assert array_manager.arrays.keys() == {"test_array_5", "test_array_10"}


def test_rename(array_manager):
    """Test renaming an array."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    array_manager.rename("test_array", "new_name")
    assert "test_array" not in array_manager.arrays
    assert "new_name" in array_manager.arrays


def test_get_numpy(array_manager):
    """Test retrieving an array as a NumPy array."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    np_array = array_manager.get_numpy("test_array")
    assert isinstance(np_array, np.ndarray)


def test_get_cupy_as_numpy(array_manager):
    """Test converting a CuPy array to NumPy when using CuPy."""
    if CUPY_AVAILABLE:
        array_manager.enable_cupy()
        array = np.random.rand(5, 5)
        array_manager.add("test_array", array)
        np_array = array_manager.get_numpy("test_array")
        assert isinstance(np_array, np.ndarray)


def test_getitem(array_manager):
    """Test the __call__ method."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    assert np.all(array_manager["test_array"] == array)


def test_setitem(array_manager):
    """Test the __setitem__ method."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", np.zeros((5, 5)))
    array_manager["test_array"] = array
    assert np.all(array_manager["test_array"] == array)


def test_contains(array_manager):
    """Test the __contains__ method."""
    array = np.random.rand(5, 5)
    array_manager.add("test_array", array)
    assert "test_array" in array_manager
    assert "non_existent_array" not in array_manager


def test_to_dict(array_manager):
    """Test the to_dict method."""
    array_manager.add("test_array", np.random.rand(5, 5))
    info = array_manager.to_dict()
    assert info["names"] == ["test_array"]
    assert info["using_cupy"] == array_manager.using_cupy
