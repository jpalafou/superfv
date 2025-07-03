import warnings
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import numpy as np

# determine if CuPy is available
xp = np
cp_array = np.ndarray
cp_array_to_numpy_array = np.asarray
np_array_to_cp_array = np.asarray
CUPY_AVAILABLE = False
if not TYPE_CHECKING:
    try:
        import cupy as cp

        xp = cp
        cp_array = cp.ndarray
        cp_array_to_numpy_array = cp.asnumpy
        np_array_to_cp_array = cp.asarray
        CUPY_AVAILABLE = True
    except Exception:
        pass

# define custom types
ArrayLike = Union[np.ndarray, xp.ndarray]


class ArrayManager:
    """
    Class for managing arrays which can be transferred between CPU and GPU.
    """

    def _transfer_array(self, name: str, device: Literal["cpu", "gpu"]):
        """Transfers a specific array between CPU and GPU."""
        self._check_name_exists(name)
        is_cupy = isinstance(self.arrays[name], cp_array)
        if device == "cpu" and is_cupy:
            self.arrays[name] = cp_array_to_numpy_array(self.arrays[name])
        elif device == "gpu" and not is_cupy:
            self.arrays[name] = np_array_to_cp_array(self.arrays[name])

    def _dummy_transfer_array(self, name: str, device: Literal["cpu", "gpu"]):
        pass

    def __init__(self, arrays: Optional[Dict[str, np.ndarray]] = None):
        """
        Initializes the array manager.

        Args:
            arrays: Dictionary of NumPy arrays.
        """
        self.arrays: Dict[str, ArrayLike] = arrays if arrays else {}
        self.device: Literal["cpu", "gpu"] = "cpu"
        self.transfer_array = (
            self._transfer_array if CUPY_AVAILABLE else self._dummy_transfer_array
        )

    def __repr__(self) -> str:
        return f"ArrayManager({self.arrays.keys()})"

    def __str__(self) -> str:
        return f"ArrayManager with arrays: {list(self.arrays.keys())}"

    def _check_name_exists(self, name: str):
        if name not in self.arrays:
            raise KeyError(f"Array with name '{name}' not found.")

    def _check_name_available(self, name: str):
        if name in self.arrays:
            raise KeyError(f"Array with name '{name}' already exists.")

    def transfer_to_device(self, device: Literal["cpu", "gpu"]):
        """
        Transfer all arrays to a specific device.

        Args:
            device: Device to transfer arrays to ("cpu" or "gpu").
        """
        if self.device == device:
            raise ValueError(f"ArrayManager is already using {device}.")
        if not CUPY_AVAILABLE:
            warnings.warn("CuPy is not available. Falling back to NumPy.")
            return
        for name in self.arrays:
            self.transfer_array(name, device)
        self.device = device

    def add(self, name: str, array: ArrayLike):
        """
        Add an array to the manager.

        Args:
            name: Name of the array.
            array: NumPy or CuPy array.
        """
        self._check_name_available(name)
        self.arrays[name] = array
        self.transfer_array(name, self.device)

    def remove(self, name: str):
        """
        Remove an array from the manager.

        Args:
            name: Name of the array.
        """
        self._check_name_exists(name)
        del self.arrays[name]

    def rename(self, name: str, new_name: str):
        """
        Rename an array.

        Args:
            name: Current name of the array.
            new_name: New name of the array.
        """
        self._check_name_exists(name)
        self._check_name_available(new_name)
        self.add(new_name, self.arrays[name])
        self.remove(name)

    def clear(self):
        """
        Remove all arrays from the manager
        """
        self.arrays.clear()

    def __getitem__(
        self, name: str, copy: bool = False, asnumpy: bool = False
    ) -> ArrayLike:
        """
        Get an array from the manager.

        Args:
            name: Name of the array.
            copy: If True, return a copy of the array.
            asnumpy: If True, return a NumPy array.

        Returns:
            NumPy or CuPy array.
        """
        self._check_name_exists(name)
        array = self.arrays[name]
        if copy:
            array = array.copy()
        if asnumpy and self.device == "gpu":
            array = cp_array_to_numpy_array(array)
        return array

    def get_numpy_copy(self, name: str) -> np.ndarray:
        """
        Get a NumPy copy of an array.

        Args:
            name: Name of the array.
        """
        return self.__getitem__(name, True, True)

    def __setitem__(self, name: str, array: ArrayLike):
        """
        Modify an existing array in place.

        This prevents users from replacing arrays entirely, ensuring that
        preallocated memory is used efficiently.

        Args:
            name: Name of the array.
            array: New values to assign in-place.
        """
        self._check_name_exists(name)
        if self.arrays[name].shape != array.shape:
            raise ValueError(
                f"Cannot assign array with shape {array.shape} to array with shape {self.arrays[name].shape}."
            )
        if self.arrays[name].dtype != array.dtype:
            raise ValueError(
                f"Cannot assign array with dtype {array.dtype} to array with dtype {self.arrays[name].dtype}."
            )
        self.arrays[name][...] = array

    def __contains__(self, name: str) -> bool:
        """
        Check if an array exists in the manager.

        Args:
            name: Name of the array.
        """
        return name in self.arrays

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the manager.
        """
        return dict(names=list(self.arrays.keys()), device=self.device)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Get the state of the ArrayManager for serialization.
        """
        self.transfer_to_device("cpu")
        return self.__dict__.copy()
