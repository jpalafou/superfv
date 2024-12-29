import warnings
from typing import Dict, List, Union

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    ArrayType = Union[np.ndarray, cp.ndarray]
except Exception:
    CUPY_AVAILABLE = False
    ArrayType = np.ndarray


class ArrayManager:
    """
    Class for managing arrays in a dictionary and their transfer between NumPy and
    CuPy.
    """

    def __init__(self, arrays: Dict[str, ArrayType] = None):
        """
        Initializes the array manager.

        Args:
            arrays: Dictionary of arrays.
        """
        self.arrays = arrays if arrays else {}
        self.using_cupy = False

    def __repr__(self) -> str:
        return f"ArrayManager({self.arrays.keys()})"

    def __str__(self) -> str:
        return f"ArrayManager({self.arrays.keys()})"

    def _check_name_exists(self, name: str):
        if name not in self.arrays:
            raise KeyError(f"Array with name '{name}' not found.")

    def _check_name_available(self, name: str):
        if name in self.arrays:
            raise KeyError(f"Array with name '{name}' already exists.")

    def _check_numpy_array(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Array must be of type numpy.ndarray, not {type(array)}.")

    def enable_cupy(self):
        """
        Enable the use of CuPy.

        Note:
            If CuPy is available, this method transfers all existing arrays to the
            GPU as CuPy arrays and sets the `using_cupy` attribute to `True`. Any
            new arrays added will also be stored as CuPy arrays.

            If CuPy is not available, a warning is printed, and the arrays remain
            as NumPy arrays on the CPU.
        """
        if CUPY_AVAILABLE:
            self.using_cupy = True
            for name in self.arrays:
                self.transfer_device(name, "gpu")
        else:
            warnings.warn("CuPy is not available. Falling back to NumPy.")

    def transfer_device(self, name: str, to_device: str):
        """
        Transfer an array to a different device.

        Args:
            name: Name of the array.
            to_device: Device to transfer the array to. Options are "cpu" and "gpu".
        """
        self._check_name_exists(name)
        if self.using_cupy:
            if to_device == "cpu" and isinstance(self.arrays[name], cp.ndarray):
                self.arrays[name] = cp.asnumpy(self.arrays[name])
            elif to_device == "gpu" and isinstance(self.arrays[name], np.ndarray):
                self.arrays[name] = cp.asarray(self.arrays[name])
            else:
                raise ValueError(f"Invalid device: {to_device}")

    def disable_cupy(self):
        """
        Disable the use of CuPy.

        Note:
            This method transfers all existing arrays to the CPU as NumPy arrays and
            sets the `using_cupy` attribute to `False`. Any new arrays added will also
            be stored as NumPy arrays.
        """
        for name in self.arrays:
            self.transver_device(name, "cpu")
        self.using_cupy = False

    def add(self, name: str, array: np.ndarray):
        """
        Add an array to the manager.

        Args:
            name: Name of the array.
            array: Array to add.
        """
        self._check_name_available(name)
        self._check_numpy_array(array)
        if self.using_cupy:
            self.arrays[name] = cp.asarray(array)
        else:
            self.arrays[name] = array

    def rm(self, name: str):
        """
        Remove an array from the manager.

        Args:
            name: Name of the array.
        """
        self._check_name_exists(name)
        del self.arrays[name]

    def clear(self, all_but: List[str] = None):
        """
        Clear all arrays from the manager.

        Args:
            all_but: List of array names to keep.
        """
        if all_but:
            self.arrays = {
                name: array for name, array in self.arrays.items() if name in all_but
            }
        else:
            self.arrays = {}

    def __call__(
        self, name: str, asnumpy: bool = False, copy: bool = False
    ) -> ArrayType:
        """
        Get an array from the manager.

        Args:
            name: Name of the array.
            asnumpy: If True, return the array as a NumPy array.
            copy: If True, return a copy of the array.

        Returns:
            Array.
        """
        self._check_name_exists(name)
        if self.using_cupy and asnumpy:
            return cp.asnumpy(self.arrays[name])
        return self.arrays[name].copy() if copy else self.arrays[name]

    def to_dict(self) -> dict:
        return dict(names=list(self.arrays.keys()), using_cupy=self.using_cupy)
