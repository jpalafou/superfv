import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

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
SliceBounds = Tuple[Union[None, int], Union[None, int]]


def l1_norm(array: ArrayLike) -> float:
    """
    Compute the L1 norm of an array.

    Args:
        array (ArrayLike): Array of any shape.

    Returns:
        float: L1 norm of the array.
    """
    return np.mean(np.abs(array))


def l2_norm(array: ArrayLike) -> float:
    """
    Compute the L2 norm of an array.

    Args:
        array (ArrayLike): Array of any shape.

    Returns:
        float: L2 norm of the array.
    """
    return np.sqrt(np.mean(np.square(array)))


def linf_norm(array: ArrayLike) -> float:
    """
    Compute the L-infinity norm of an array.

    Args:
        array (ArrayLike): Array of any shape.

    Returns:
        float: L-infinity norm of the array.
    """
    return np.max(np.abs(array))


def chop(
    cut: Tuple[Optional[int], Optional[int]],
    axis: int,
    step: Optional[int] = None,
) -> Tuple[slice, ...]:
    """
    Create a slice object for a given axis.

    Args:
        cut (Tuple[Optional[int], Optional[int]]): Start and stop indices for the
            slice.
        axis (int): Axis along which to slice.
        step (Optional[int]): Step size for the slice.

    Returns:
        Tuple[slice, ...]: A tuple of slices for the given axis.
    """
    return (slice(None),) * axis + (slice(cut[0] or None, cut[1] or None, step),)


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
    idxs: Iterable[int],
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
        variables (Dict[str, int]): Dictionary of variable names and indices.
        ndim (int): Number of dimensions.
        groups (Dict[str, Tuple[str, ...]]): Dictionary of group names and the variable
            names they contain.
    """

    variables: Dict[str, int]
    ndim: int
    groups: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize variable index map
        self.var_idx_map = cast(
            Dict[str, Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]],
            self.variables.copy(),
        )

        # Store variable and index sets
        self.var_names = set(self.variables.keys())
        self.idxs = set(self.variables.values())

        # Validate provided groups
        validated_groups = {}
        if self.groups:
            for group_name, var_names in self.groups.items():
                if any(v not in self.variables for v in var_names):
                    raise ValueError(f"Variables not found: {var_names}")
                if group_name in self.variables:
                    raise ValueError(f"Name '{group_name}' already exists.")
                group_idxs = [self.variables[v] for v in var_names]

                validated_groups[group_name] = var_names
                self.var_idx_map[group_name] = _idxs_to_slice_or_array(group_idxs)
        self.groups = validated_groups
        self.group_names = set(self.groups.keys())
        self.all_names = self.var_names | self.group_names

    def add_var(self, name: str, idx: int):
        """
        Add a variable to the slicer.

        Args:
            name (str): Variable name.
            idx (int): Variable index.
        """
        if name in self.all_names:
            raise ValueError(f"Variable '{name}' already exists.")
        self.variables[name] = idx
        self.__post_init__()

    def create_var_group(
        self, group_name: str, variables: Tuple[str, ...], overwrite=False
    ):
        """
        Create a group of variables (private method).

        Args:
            group_name (str): Name of the group.
            variables (Tuple[str, ...]): Tuple of variable names.
        """
        if group_name in self.all_names:
            raise ValueError(f"Name '{group_name}' already exists.")
        self.groups[group_name] = variables
        self.__post_init__()

    def add_to_var_group(self, group_name: str, variables: Tuple[str, ...]):
        """
        Add variables to an existing group.

        Args:
            group_name (str): Name of the group.
            variables (Tuple[str, ...]): Tuple of variable names.
        """
        if group_name not in self.group_names:
            raise ValueError(f"Group '{group_name}' not found.")
        self.groups[group_name] += variables
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
        keepdims: bool = False,
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
            x (Optional[SliceBounds]: Start and stop indices for the x-axis (axis 1)
            y (Optional[SliceBounds]): Start and stop indices for the y-axis (axis 2).
            z (Optional[SliceBounds]): Start and stop indices for the z-axis (axis 3).
            axis (Optional[int]): Axis along which to slice the field.
            cut (Optional[SliceBounds]): Start and stop indices for the axis.
            step (Optional[int]): Step size for the axis.
            keepdims (bool): If True, keep the dimensions of the sliced array.

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

        if keepdims:
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
        return ArraySlicer(
            variables=self.variables.copy(),
            ndim=self.ndim,
            groups=self.groups.copy() if self.groups is not None else None,
        )


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
            arrays (Optional[Dict[str, np.ndarray]]): Dictionary of NumPy arrays.
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
            device (Literal["cpu", "gpu"]): Device to transfer arrays to.
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
            name (str): Name of the array.
            array (ArrayLike): NumPy or CuPy array.
        """
        self._check_name_available(name)
        self.arrays[name] = array
        self.transfer_array(name, self.device)

    def remove(self, name: str):
        """
        Remove an array from the manager.

        Args:
            name (str): Name of the array.
        """
        self._check_name_exists(name)
        del self.arrays[name]

    def rename(self, name: str, new_name: str):
        """
        Rename an array.

        Args:
            name (str): Current name of the array.
            new_name (str): New name of the array.
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
            name (str): Name of the array.
            copy (bool): If True, return a copy of the array.
            asnumpy (bool): If True, return a NumPy array.

        Returns:
            ArrayLike: NumPy or CuPy array.
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
            name (str): Name of the array.
        """
        return self.__getitem__(name, True, True)

    def __setitem__(self, name: str, array: ArrayLike):
        """
        Modify an existing array in place.

        This prevents users from replacing arrays entirely, ensuring that
        preallocated memory is used efficiently.

        Args:
            name (str): Name of the array.
            array (ArrayLike): New values to assign in-place.
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
            name (str): Name of the array.
        """
        return name in self.arrays

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the manager.
        """
        return dict(names=list(self.arrays.keys()), device=self.device)
