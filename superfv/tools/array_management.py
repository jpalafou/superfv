import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

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


@lru_cache(maxsize=None)
def _cached_crop_to_center(
    in_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    axes: Optional[Tuple[int, ...]],
) -> Tuple[slice, ...]:
    """
    Cached helper function for crop_to_center.

    Args:
        in_shape (Tuple[int, ...]): The shape of the input array.
        target_shape (Tuple[int, ...]): The desired shape of the output array.
        axes (Optional[Tuple[int, ...]]): The axes along which to crop. If None,
            `target_shape` must have length equal to in_shape. Otherwise `target_shape`
            must have length equal to len(`axes`).

    Returns:
        Tuple[slice, ...]: A tuple of slices that can be used to crop the input array.
    """
    _axes = tuple(range(len(in_shape))) if axes is None else axes

    if len(target_shape) != len(in_shape):
        raise ValueError(
            "Target shape must have the same number of dimensions as the input array."
        )

    slices = [slice(None)] * len(in_shape)
    for axis in _axes:
        if axis not in _axes:
            continue
        dim_length = in_shape[axis]
        target_length = target_shape[axis]
        crop = dim_length - target_length
        if crop < 0:
            raise ValueError(
                "Target shape must be less than or equal to the input array's shape in all dimensions."
            )
        if crop == 0:
            continue
        elif crop % 2 == 0:
            start = crop // 2
            end = dim_length - crop // 2
            slices[axis] = slice(start, end)
        else:
            raise ValueError(
                f"Cannot evenly crop dimension from {dim_length} to {target_length}."
            )

    return tuple(slices)


def crop_to_center(
    array: np.ndarray,
    target_shape: Tuple[int, ...],
    axes: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Crops the input array to the target shape by removing an equal amount from both
    ends along each axis.

    Args:
        array (np.ndarray): The input array to be cropped.
        target_shape (Tuple[int, ...]): The desired shape of the output array. Must
            have the same number of dimensions as array.
        axes (Optional[Tuple[int, ...]]): The axes along which to crop.

    Returns:
        np.ndarray: A cropped version of the input array with the target shape.

    Raises:
        ValueError: If the target shape is invalid or cropping cannot remove an even
            amount along any axis.
    """
    slices = _cached_crop_to_center(array.shape, target_shape, axes)
    return array[slices]


def intersection_shape(*args: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
    """
    Compute the intersection of the shapes of multiple arrays.

    Args:
        *args (Tuple[Tuple[int, ...], ...]): Tuple of shapes.

    Returns:
        Tuple[int, ...]: Intersection shape.
    """
    return tuple(min(s) for s in zip(*args))


def _idxs_to_slice_or_array(
    idxs: List[int],
) -> Union[slice, np.ndarray[Any, np.dtype[np.int_]]]:
    """
    Convert a list of indices to a slice or numpy array.

    Args:
        idxs (List[int]): List of indices.
    """
    _idxs = sorted(list(set(idxs)))
    if _idxs == list(range(_idxs[0], _idxs[-1] + 1)):
        return slice(_idxs[0], _idxs[-1] + 1)
    return np.array(_idxs)


@dataclass
class ArraySlicer:
    """
    Class for slicing multivariable fields.

    Args:
        var_idx_map (
            Dict[str, Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]]
            ): Dictionary mapping variable names to slices along the first axis.
        ndim (int): Number of dimensions in the field.
    """

    var_idx_map: Dict[str, Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]]
    ndim: int

    def __post_init__(self):
        self.var_names = set()
        self.group_names = set()
        self.idxs = set()

        for name, idx in self.var_idx_map.items():
            if isinstance(idx, int):
                self.var_names.add(name)
                self.idxs.add(idx)
            else:
                self.group_names.add(name)

        self.all_names = self.var_names.union(self.group_names)

    def add_var(self, name: str, idx: int):
        """
        Add a variable to the slicer.

        Args:
            name (str): Variable name.
            idx (int): Variable index.
        """
        if name in self.all_names:
            raise ValueError(f"Variable '{name}' already exists.")
        self.var_idx_map[name] = idx
        self.__post_init__()

    def create_var_group(self, group_name: str, variables: Tuple[str, ...]):
        """
        Create a group of variables.

        Args:
            group_name (str): Name of the group.
            variables (Tuple[str, ...]): Tuple of variable names.
        """
        if any(name not in self.var_names for name in variables):
            raise ValueError(f"Variables not found: {variables}")
        if group_name in self.all_names:
            raise ValueError(f"Name '{group_name}' already exists.")
        group_idxs = [self.var_idx_map[v] for v in variables]
        if any(not isinstance(i, int) for i in group_idxs):
            raise ValueError("Variables must be indexed by integers.")
        self.var_idx_map[group_name] = _idxs_to_slice_or_array(
            [cast(int, self.var_idx_map[v]) for v in variables]
        )
        self.__post_init__()

    def __hash__(self):
        """
        Returns the hash of the object based on its memory address.
        """
        return id(self)

    @lru_cache(maxsize=None)
    def __call__(
        self,
        variable: Optional[Union[str, Tuple[str, ...]]] = None,
        x: Optional[SliceBounds] = None,
        y: Optional[SliceBounds] = None,
        z: Optional[SliceBounds] = None,
        axis: Optional[int] = None,
        cut: Optional[SliceBounds] = None,
        step: Optional[int] = None,
        keep_dims: bool = False,
    ) -> Union[
        slice,
        int,
        np.ndarray,
        Tuple[Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]], ...],
    ]:
        """
        Generate a slice object for a multivariable field.

        Args:
            variable (Optional[Union[str, Tuple[str, ...]]]): Variable name or tuple of
                variable names.
            x (Optional[SliceBounds]: Start and stop indices for the x-axis.
            y (Optional[SliceBounds]): Start and stop indices for the y-axis.
            z (Optional[SliceBounds]): Start and stop indices for the z-axis.
            axis (Optional[int]): Axis along which to slice the field.
            cut (Optional[SliceBounds]): Start and stop indices for the axis.
            step (Optional[int]): Step size for the axis.
            keep_dims (bool): If True, keep the dimensions of the sliced array.

        Returns:
            Slice object if only the first axis is sliced, or tuple of slice objects.
        """
        slices: List[Union[slice, int, np.ndarray]] = [slice(None)] * self.ndim

        if variable is not None:
            if isinstance(variable, str):
                # retrieve single variable index
                if variable not in self.var_idx_map:
                    raise ValueError(f"Variable '{variable}' not found.")
                var_idx = self.var_idx_map[variable]
                if not isinstance(var_idx, (int, slice, np.ndarray)):
                    raise ValueError(
                        f"Variable index must be an integer, slice, or numpy.ndarray, not {type(var_idx)}."
                    )
                slices[0] = var_idx
            elif isinstance(variable, tuple):
                # retrieve multiple variable indices
                missing_vars = set(variable) - set(self.var_idx_map.keys())
                if missing_vars:
                    raise ValueError(f"Variables not found: {missing_vars}")
                if any(not isinstance(self.var_idx_map[v], int) for v in variable):
                    raise ValueError("Multiple variables must be indexed by integers.")
                slices[0] = _idxs_to_slice_or_array(
                    [cast(int, self.var_idx_map[v]) for v in variable]
                )
            else:
                raise ValueError(f"Invalid type for var: {type(variable)}")

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

        if keep_dims:
            for i in range(len(slices)):
                if isinstance(slices[i], int):
                    slices[i] = slice(slices[i], cast(int, slices[i]) + 1)
        if len(slices) == 1 or all(s == slice(None) for s in slices[1:]):
            return slices[0]
        return tuple(slices)

    def copy(self) -> "ArraySlicer":
        """
        Create a copy of the slicer.
        """
        return ArraySlicer(self.var_idx_map.copy(), self.ndim)


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
            self.transfer_device(name, "cpu")
        self.using_cupy = False

    def add(self, name: str, array: np.ndarray, overwrite: bool = False):
        """
        Add an array to the manager.

        Args:
            name (str): Name of the array.
            array (np.ndarray): Array to add.
            overwrite (bool): If True, overwrite the array if it already exists.
        """
        if not overwrite:
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

    def rename(self, old_name: str, new_name: str):
        """
        Rename an array.

        Args:
            old_name (str): Old name of the array.
            new_name (str): New name of the array.
        """
        self._check_name_exists(old_name)
        self._check_name_available(new_name)
        self.arrays[new_name] = self.arrays.pop(old_name)

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

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def to_dict(self) -> dict:
        return dict(names=list(self.arrays.keys()), using_cupy=self.using_cupy)
