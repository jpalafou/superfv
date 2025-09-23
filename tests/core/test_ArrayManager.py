import numpy as np
import pytest

from superfv.tools.device_management import CUPY_AVAILABLE, ArrayManager, xp


def test_initialization():
    manager = ArrayManager()
    assert manager.device == "cpu"
    assert len(manager.arrays) == 0


def test_add_array():
    manager = ArrayManager()
    array = np.ones((3, 3))
    manager.add("test_array", array)
    assert "test_array" in manager.arrays
    assert np.array_equal(manager["test_array"], array)


def test_remove_array():
    manager = ArrayManager()
    array = np.zeros((2, 2))
    manager.add("zeros", array)
    manager.remove("zeros")
    assert "zeros" not in manager.arrays


def test_rename_array():
    manager = ArrayManager()
    array = np.random.rand(4, 4)
    manager.add("old_name", array)
    manager.rename("old_name", "new_name")
    assert "old_name" not in manager.arrays
    assert "new_name" in manager.arrays
    assert np.array_equal(manager["new_name"], array)


def test_clear_arrays():
    manager = ArrayManager({"a": np.array([1, 2, 3])})
    manager.clear()
    assert len(manager.arrays) == 0


def test_transfer_to():
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available")

    manager = ArrayManager()
    manager.add("cpu_array", np.ones((5, 5)))
    manager.transfer_to("gpu")
    assert isinstance(manager.arrays["cpu_array"], xp.ndarray)
    manager.transfer_to("cpu")
    assert isinstance(manager.arrays["cpu_array"], np.ndarray)


def test_get_numpy_copy():
    manager = ArrayManager()
    array = np.array([1, 2, 3])
    manager.add("data", array)
    copy = manager.get_numpy_copy("data")
    assert np.array_equal(copy, array)
    assert copy is not array  # Ensure it's a copy


def test_setitem_inplace():
    manager = ArrayManager()
    array = np.array([[1, 2], [3, 4]])
    manager.add("matrix", array)
    new_values = np.array([[5, 6], [7, 8]])
    manager["matrix"] = new_values
    assert np.array_equal(manager["matrix"], new_values)


def test_setitem_invalid_shape():
    manager = ArrayManager()
    manager.add("array", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Cannot assign array with shape"):
        manager["array"] = np.zeros((3, 3))


def test_setitem_invalid_dtype():
    manager = ArrayManager()
    manager.add("array", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Cannot assign array with dtype"):
        manager["array"] = np.zeros((2, 2), dtype=np.int32)


def test_transfer_without_cupy():
    if CUPY_AVAILABLE:
        pytest.skip("CuPy is available, skipping fallback test")

    manager = ArrayManager()
    manager.add("data", np.array([1, 2, 3]))
    with pytest.warns(UserWarning, match="CuPy is not available"):
        manager.transfer_to("gpu")  # Should warn and do nothing
    assert isinstance(manager["data"], np.ndarray)  # Still a NumPy array
