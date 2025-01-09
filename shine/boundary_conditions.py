from dataclasses import dataclass
from itertools import product
from typing import Tuple, Union

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer

IMPLEMENTED_BCS = ["periodic"]


@dataclass
class BoundaryConditions:
    """
    Class for managing finite-volume boundary conditions in 3D.
    """

    def __init__(
        self,
        array_slicer: ArraySlicer,
        x: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        y: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        z: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
    ):
        """
        Initialize boundary conditions.

        Args:
            array_slicer: ArraySlicer object for slicing arrays by x, y, z and variable
                name.
            x (Union[str, Tuple[str, str]]): Boundary condition string indicating the
                type of boundary condition to use in the x-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.
            y (Union[str, Tuple[str, str]]): Boundary condition string indicating the
                type of boundary condition to use in the y-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.
            z (Union[str, Tuple[str, str]]): Boundary condition string indicating the
                type of boundary condition to use in the z-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.

        Note:
            See `IMPLEMENTED_BCS` for the implemented boundary condition codes.
        """
        for dim in [x, y, z]:
            if not (isinstance(dim, str) or (isinstance(dim, tuple) and len(dim) == 2)):
                raise ValueError(
                    "Boundary conditions must be a string or tuple of two strings."
                )
        self.array_slicer = array_slicer
        self.x = (x, x) if isinstance(x, str) else x
        self.y = (y, y) if isinstance(y, str) else y
        self.z = (z, z) if isinstance(z, str) else z

    def __call__(self, arr: ArrayLike, pad_width: Tuple[int, int, int]) -> ArrayLike:
        """
        Apply boundary conditions to an array.

        Args:
            arr (ArrayLike): Array to which to apply boundary conditions.
            pad_width (Tuple[int, int, int]): Tuple of integers indicating the padding
                width for each dimension.

        Returns:
            (ArrayLike): Array with boundary conditions applied.
        """
        if len(pad_width) != 3 or not all(isinstance(w, int) for w in pad_width):
            raise ValueError("Padding width must be a tuple of three integers.")

        # initialize array with boundary conditions
        out = np.pad(
            arr,
            pad_width=(
                (0, 0),
                (pad_width[0], pad_width[0]),
                (pad_width[1], pad_width[1]),
                (pad_width[2], pad_width[2]),
            ),
            mode="empty",
        )

        # loop over boundary slabs
        for (i, dim), (j, pos) in product(enumerate("xyz"), enumerate("lr")):
            bc_type = getattr(self, dim)[j]
            slab_thickness = pad_width[i]
            axis = i + 1

            if slab_thickness == 0:
                continue

            match bc_type:
                case "periodic":
                    self._apply_periodic_bc(out, slab_thickness, axis, pos)

        return out

    def _apply_periodic_bc(
        self, arr: ArrayLike, slab_thickness: int, axis: int, pos: str
    ):
        """
        Apply periodic boundary conditions to arr, modifying it in place.

        Args:
            arr (ArrayLike): Array to which to apply boundary conditions.
            slab_thickness (int): Thickness of the boundary condition slab along the
                axis.
            axis (int): Axis along which to apply boundary conditions.
            pos (str): Position of the boundary condition slab ("l" or "r").

        """
        _slc = self.array_slicer
        _st = slab_thickness
        if pos == "l":
            outer_slc = _slc(axis=axis, cut=(None, _st))
            inner_slc = _slc(axis=axis, cut=(-2 * _st, -_st))
        else:
            outer_slc = _slc(axis=axis, cut=(-_st, None))
            inner_slc = _slc(axis=axis, cut=(_st, 2 * _st))
        arr[outer_slc] = arr[inner_slc]
