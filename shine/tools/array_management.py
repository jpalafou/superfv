import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np

# determine if CuPy is available
if TYPE_CHECKING:
    import numpy as cp

    CUPY_AVAILABLE = False
else:
    try:
        import cupy as cp

        CUPY_AVAILABLE = True
    except Exception:
        import numpy as cp

        CUPY_AVAILABLE = False

# define custom types
ArrayLike = Union[np.ndarray, cp.ndarray]
SliceBounds = Tuple[Union[None, int], Union[None, int]]


@dataclass
class ArraySlicer:
    """
    Class for slicing multivariable fields.

    Args:
        var_idx_map (Dict[str, int]): Dictionary mapping variable names to indices.
        ndim (int): Number of dimensions in the field.
    """

    var_idx_map: Dict[str, int]
    ndim: int

    def __hash__(self):
        return id(self)

    def __call__(
        self,
        var: Optional[Union[str, Tuple[str, ...]]] = None,
        x: Optional[SliceBounds] = None,
        y: Optional[SliceBounds] = None,
        z: Optional[SliceBounds] = None,
        axis: Optional[int] = None,
        cut: Optional[SliceBounds] = None,
        step: Optional[int] = None,
    ) -> Union[slice, int, np.ndarray, Tuple[Union[slice, int, np.ndarray], ...]]:
        """
        Generate a slice object for a multivariable field.

        Args:
            var (Optional[Union[str, Tuple[str, ...]]]): Variable name or tuple of variable names.
            x (Optional[SliceBounds]: Start and stop indices for the x-axis.
            y (Optional[SliceBounds]): Start and stop indices for the y-axis.
            z (Optional[SliceBounds]): Start and stop indices for the z-axis.
            axis (Optional[int]): Axis along which to slice the field.
            cut (Optional[SliceBounds]): Start and stop indices for the axis.
            step (Optional[int]): Step size for the axis.

        Returns:
            Slice object if only the first axis is sliced, or tuple of slice objects.
        """
        slices: List[Union[slice, int, np.ndarray]] = [slice(None)] * self.ndim

        if var is not None:
            if isinstance(var, str):
                # retrieve single variable index
                if var not in self.var_idx_map:
                    raise ValueError(f"Variable '{var}' not found.")
                var_idx = self.var_idx_map[var]
                if not isinstance(var_idx, int):
                    raise ValueError(
                        f"Variable index must be an integer, not {type(var_idx)}."
                    )
                slices[0] = var_idx
            elif isinstance(var, tuple):
                # retrieve multiple variable indices
                missing_vars = set(var) - set(self.var_idx_map.keys())
                if missing_vars:
                    raise ValueError(f"Variables not found: {missing_vars}")
                slices[0] = np.array(list(map(self.var_idx_map.get, var)))
            else:
                raise ValueError(f"Invalid type for var: {type(var)}")

        axes = [1, 2, 3, axis]
        axis_slices = [x, y, z, cut]
        for i, axis_slice in zip(axes, axis_slices):
            if axis_slice is None:
                continue
            if cast(int, i) >= self.ndim:
                raise ValueError(
                    f"Invalid axis {i} for array with {self.ndim} dimensions."
                )
            if not isinstance(axis_slice, tuple) or len(axis_slice) != 2:
                raise ValueError(f"Expected a tuple (start, stop) for axis {i}.")
            slices[cast(int, i)] = slice(
                axis_slice[0] or None,
                axis_slice[1] or None,
                step if i == axis else None,
            )

        if len(slices) == 1 or all(s == slice(None) for s in slices[1:]):
            return slices[0]
        return tuple(slices)


class ArrayManager:
    """
    Class for managing arrays in a dictionary and their transfer between NumPy and
    CuPy.
    """

    def __init__(self, arrays: Optional[Dict[str, ArrayLike]] = None):
        """
        Initializes the array manager.

        Args:
            arrays (Optional[Dict[str, ArrayLike]]): Dictionary of arrays.
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
            name (str): Name of the array.
            to_device (str): Device to transfer the array to. Options are "cpu" and "gpu".
        """
        self._check_name_exists(name)
        if self.using_cupy:
            if to_device == "cpu" and isinstance(self.arrays[name], cp.ndarray):
                if TYPE_CHECKING:
                    raise ValueError("Cannot import CuPy in type-checking mode.")
                else:
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
            name (str): Name of the array.
            array (np.ndarray): Array to add.
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
            name (str): Name of the array.
        """
        self._check_name_exists(name)
        del self.arrays[name]

    def clear(self, all_but: Optional[Iterable[str]] = None):
        """
        Clear all arrays from the manager.

        Args:
            all_but (Optional[Iterable[str]]): List of array names to keep.
        """
        if all_but:
            self.arrays = {
                name: array for name, array in self.arrays.items() if name in all_but
            }
        else:
            self.arrays = {}

    def __getitem__(
        self, name: str, asnumpy: bool = False, copy: bool = False
    ) -> ArrayLike:
        """
        Get an array from the manager.

        Args:
            name (str): Name of the array.
            asnumpy (bool): If True, return the array as a NumPy array.
            copy (bool): If True, return a copy of the array.

        Returns:
            ArrayLike
        """
        self._check_name_exists(name)
        if self.using_cupy and asnumpy:
            if TYPE_CHECKING:
                raise ValueError("Cannot import CuPy in type-checking mode.")
            else:
                return cp.asnumpy(self.arrays[name])
        return self.arrays[name].copy() if copy else self.arrays[name]

    def get_numpy(self, name: str, copy: bool = False) -> np.ndarray:
        """
        Get an array as a NumPy array.

        Args:
            name (str): Name of the array.
            copy (bool): If True, return a copy of the array.

        Returns:
            np.ndarray
        """
        if TYPE_CHECKING:
            raise ValueError("Cannot import CuPy in type-checking mode.")
        return self.__getitem__(name, asnumpy=True, copy=copy)

    def __setitem__(self, name: str, array: ArrayLike):
        """
        Set an array in the manager.

        Args:
            name (str): Name of the array.
            array (ArrayLike): Array to set.
        """
        self.arrays[name] = array

    def to_dict(self) -> dict:
        return dict(names=list(self.arrays.keys()), using_cupy=self.using_cupy)
