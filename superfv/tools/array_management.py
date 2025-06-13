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
    Set,
    Tuple,
    Union,
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
IndexLike = Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]
SliceBounds = Tuple[Union[None, int], Union[None, int]]


def l1_norm(array: ArrayLike) -> float:
    """
    Compute the L1 norm of an array.
    """
    return np.mean(np.abs(array))


def l2_norm(array: ArrayLike) -> float:
    """
    Compute the L2 norm of an array.
    """
    return np.sqrt(np.mean(np.square(array)))


def linf_norm(array: ArrayLike) -> float:
    """
    Compute the L-infinity norm of an array.
    """
    return np.max(np.abs(array))


@lru_cache(maxsize=None)
def crop(
    axis: Union[int, Tuple[int, ...]],
    cut: Tuple[int, int],
    step: Optional[int] = None,
    ndim: Optional[int] = None,
) -> Tuple[slice, ...]:
    """
    Create an array slice for a given axis or axes.

    Args:
        axis: Axis or axes along which to slice.
        cut: Start and stop indices for the slice.
        step: Step size for the slice. Default is None.
        ndim: Number of dimensions of the array. If None, uses the maximum axis value
            to determine the number of dimensions.

    Returns:
        Tuple of slices for the given axis or axes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    rank = ndim if ndim is not None else max(axis) + 1
    out = [slice(None)] * rank
    for ax in axis:
        if ax < 0 or ax >= rank:
            raise ValueError(
                f"Axis {ax} is out of bounds for array with {len(out)} dimensions."
            )
        out[ax] = slice(cut[0] or None, cut[1] or None, step)
    return tuple(out)


@lru_cache(maxsize=None)
def _crop_to_center(
    in_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    ignore_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Tuple[slice, ...]:
    """
    Create an array slice to crop an input array to a target shape by removing an equal
    amount from both ends along each axis.

    Args:
        in_shape: The shape of the input array.
        target_shape: The desired shape of the output array.
        ignore_axes: Axes to ignore when cropping. If None, all axes are considered. If
            an int, it is treated as a single axis. If a tuple, it contains multiple
            axes to ignore.

    Returns:
        Tuple of slices that can be used to crop the input array to the target shape.
    """
    out = [slice(None)] * len(in_shape)
    if len(target_shape) != len(in_shape):
        raise ValueError(
            "Target shape must have the same number of dimensions as the input array."
        )
    if ignore_axes is None:
        ignore_axes = tuple()
    elif isinstance(ignore_axes, int):
        ignore_axes = (ignore_axes,)
    for i, (dim_length, target_length) in enumerate(zip(in_shape, target_shape)):
        if i in ignore_axes:
            out[i] = slice(None)
            continue
        if target_length > dim_length:
            raise ValueError(
                f"Target shape {target_shape} must be less than or equal to the input array's shape {in_shape} in all dimensions."
            )
        elif (dim_length - target_length) % 2 == 0:
            margin = (dim_length - target_length) // 2
            out[i] = slice(margin or None, -margin or None)
        else:
            raise ValueError(
                f"Cannot evenly crop dimension from {dim_length} to {target_length}."
            )
    return tuple(out)


def crop_to_center(
    arr: ArrayLike,
    target_shape: Tuple[int, ...],
    ignore_axes: Optional[Union[int, Tuple[int, ...]]] = None,
) -> ArrayLike:
    """
    Crop an array to a target shape by removing an equal amount from both ends along each axis.

    Args:
        arr: The input array to be cropped.
        target_shape: The desired shape of the output array.
        ignore_axes: Axes to ignore when cropping.

    Returns:
        A cropped version of the input array with the target shape.
    """
    slices = _crop_to_center(arr.shape, target_shape, ignore_axes)
    return arr[slices]


def intersection_shape(*args: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
    """
    Compute the intersection of the shapes of multiple arrays.

    Args:
        *args: Tuple of shapes.

    Returns:
        Intersection shape.
    """
    return tuple(min(s) for s in zip(*args))


def merge_indices(
    *slices: Union[
        int, slice, Tuple[int, ...], List[int], np.ndarray[Any, np.dtype[np.int_]]
    ],
    as_array: bool = False,
) -> IndexLike:
    """
    Merge indices, slices, or sequences of integers into a single slice or numpy array.

    Args:
        *slices: Indices, slices, or sequences of integers to merge.
        as_array: If True, return a numpy array instead of a slice, even if the indices
            are contiguous.

    Returns:
        Merged slice if merged indices form a contiguous range, otherwise a numpy array
            of indices.
    """
    idxs = []
    for s in slices:
        if isinstance(s, int):
            if s < 0:
                raise ValueError(f"Index must be non-negative, got {s}.")
            idxs.append(s)
        elif isinstance(s, slice):
            if s.start is not None and s.start < 0:
                raise ValueError(f"Slice start cannot be negative, got {s.start}.")
            if s.stop is None or s.stop < 0:
                raise ValueError("Slice stop must be specified and non-negative.")
            if s.step is not None and s.step != 1:
                raise ValueError("Only slices with step=1 are supported.")
            idxs.extend(range(s.start if s.start is not None else 0, s.stop))
        elif isinstance(s, tuple) or isinstance(s, list) or isinstance(s, np.ndarray):
            if len(s) > 0:
                s_arr = np.asarray(s)
                if s_arr.ndim != 1:
                    raise ValueError("Only 1D arrays/lists of integers are supported.")
                if not np.issubdtype(s_arr.dtype, np.integer):
                    raise ValueError(
                        f"Expected 1D sequence of integers, got array with dtype {s_arr.dtype}."
                    )
                idxs.extend([int(i) for i in s_arr])
            else:
                s_arr = np.array([], dtype=np.int_)
        else:
            raise TypeError(
                f"Unsupported type {type(s)} for merging indices. Expected int, slice, or numpy array."
            )
    idxs = sorted(set(idxs))
    if not as_array:
        if len(idxs) == 0:
            return slice(0, 0)
        if idxs == list(range(idxs[0], idxs[-1] + 1)):
            return slice(idxs[0], idxs[-1] + 1)
    return np.array(idxs, dtype=np.int_)


@dataclass
class VariableIndexMap:
    """
    A class to manage a mapping of variable names to indices and groups of variables.

    Args:
        var_idx_map: A dictionary mapping variable names to their indices.
        group_var_map: A dictionary mapping group names to lists of variable and
            subgroup names.

    Attributes:
        all_names: A set of all variable and group names.
        idxs: A numpy array of unique indices for the variables.
        nvars: The number of unique variables.
    """

    var_idx_map: Dict[str, int] = field(default_factory=dict)
    group_var_map: Dict[str, List[str]] = field(default_factory=dict)
    var_names: Set[str] = field(init=False)
    all_names: Set[str] = field(init=False)
    idxs: np.ndarray[Any, np.dtype[np.int_]] = field(init=False)
    nvars: int = field(init=False)
    _cache: Dict[Tuple[str, bool], IndexLike] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._cache: Dict[Tuple[str, bool], IndexLike] = {}
        self._refresh_and_validate_mappings()

    def _refresh_and_validate_mappings(self):
        self.var_names: Set[str] = set(self.var_idx_map.keys())
        self.all_names: Set[str] = set(self.var_idx_map.keys()) | set(
            self.group_var_map.keys()
        )
        self.idxs: np.ndarray[Any, np.dtype[np.int_]] = np.array(
            sorted(set(self.var_idx_map.values()))
        )
        self.nvars: int = len(self.idxs)

        # indices must be contiguous starting from 0
        if self.idxs.size > 0 and not np.array_equal(self.idxs, np.arange(self.nvars)):
            raise ValueError(
                "Variable indices must be contiguous starting from 0. "
                f"Current indices: {self.idxs}"
            )

        # validate group variable map
        for group_name in self.group_var_map.keys():
            members = set(self.group_var_map[group_name])
            members.update(set(self._resolve_group_variables(group_name, seen=set())))
            if group_name in members:
                raise ValueError(f"Group '{group_name}' contains itself as a variable.")
            if members - self.all_names:
                raise ValueError(
                    f"Group '{group_name}' contains variables not in the variable index map: "
                    f"{set(members) - self.all_names}"
                )
            self.group_var_map[group_name] = sorted(members)

    def _resolve_group_variables(self, group_name: str, seen: Set[str]) -> List[str]:
        """
        Resolve a group name to a list of its variable members.

        Args:
            group_name: Name of the group to resolve.
            seen: Set of already seen group names to avoid infinite recursion.

        Returns:
            List of variable names that are members of the group.
        """
        if group_name in seen:
            raise ValueError(f"Group '{group_name}' contains a circular reference.")
        seen.add(group_name)

        if group_name not in self.group_var_map:
            raise KeyError(f"Group '{group_name}' not found in group variable map.")

        members = self.group_var_map[group_name]
        resolved = []
        for member in members:
            if member in self.group_var_map:
                resolved.extend(self._resolve_group_variables(member, seen))
            else:
                resolved.append(member)
        return resolved

    def _invalidate_cache(self):
        self._cache.clear()

    def add_var(self, name: str, idx: int):
        """
        Add a variable to the variable index map.

        Args:
            name: Variable name.
            idx: Variable index (must be non-negative).
        """

        # validate variable name and index
        self._check_name_available(name)
        if idx < 0:
            raise ValueError("Index must be non-negative.")
        self.var_idx_map[name] = idx

        self._refresh_and_validate_mappings()
        self._invalidate_cache()

    def add_var_to_group(self, group_name: str, var_names: Union[str, Iterable[str]]):
        """
        Add variables to a group in the group variable map.

        Args:
            group_name: Name of the group to which the variables will be added.
            var_names: Name or iterable of variable names to add to the group.
        """
        var_names = [var_names] if isinstance(var_names, str) else list(var_names)

        if group_name not in self.group_var_map:
            self._check_name_available(group_name)
            self.group_var_map[group_name] = []

        self.group_var_map[group_name].extend(var_names)

        self._refresh_and_validate_mappings()
        self._invalidate_cache()

    def _check_name_available(self, name: str):
        if name in self.all_names:
            raise KeyError(
                f"Name '{name}' already exists in variable index map or group variable map."
            )

    def __call__(self, name: str, keepdims: bool = False) -> IndexLike:
        """
        Get the index or slice for a variable or group.

        Args:
            name: Name of the variable or group.
            keepdims: If True, indexes for a single variable will be returned as a
                slice object to keep the dimensions consistent.

        Returns:
            Index, slice, or numpy array of indices for the variable or group.
        """
        key = (name, keepdims)
        out: IndexLike

        if key in self._cache:
            return self._cache[key]

        if name in self.var_idx_map:
            idx = self.var_idx_map[name]
            out = slice(idx, idx + 1) if keepdims else idx
        else:
            variables = self._resolve_group_variables(name, seen=set())
            out = merge_indices([self.var_idx_map[v] for v in variables])

        self._cache[key] = out
        return out


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
