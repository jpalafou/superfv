from dataclasses import dataclass
from itertools import product
from typing import Callable, Literal, Optional, Tuple, Union, cast

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer, crop_to_center

# define custom type annotation for Dirichlet boundary conditions
DirichletBC = Union[
    ArrayLike,
    Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
    Tuple[
        Optional[
            Union[
                ArrayLike,
                Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
            ]
        ],
        Optional[
            Union[
                ArrayLike,
                Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
            ]
        ],
    ],
]


def _is_two_tuple(x):
    return isinstance(x, tuple) and len(x) == 2


@dataclass
class BoundaryConditions:
    """
    Class for managing finite-volume boundary conditions in 3D.
    """

    def __init__(
        self,
        array_slicer: ArraySlicer,
        bcx: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcy: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcz: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        x_dirichlet: Optional[DirichletBC] = None,
        y_dirichlet: Optional[DirichletBC] = None,
        z_dirichlet: Optional[DirichletBC] = None,
        x_slab: Optional[
            Tuple[
                Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike],
            ]
        ] = None,
        y_slab: Optional[
            Tuple[
                Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike],
            ]
        ] = None,
        z_slab: Optional[
            Tuple[
                Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike],
            ]
        ] = None,
    ):
        """
        Initialize boundary conditions.

        Args:
            array_slicer: ArraySlicer object for slicing arrays by x, y, z and variable
                name.
            bcx (Union[str, Tuple[str, str]]): Boundary condition string indicating
                the type of boundary condition to use in the x-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.
            bcy (Union[str, Tuple[str, str]]): Boundary condition string indicating
                the type of boundary condition to use in the y-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.
            bcz (Union[str, Tuple[str, str]]): Boundary condition string indicating
                the type of boundary condition to use in the z-direction. Can use one
                string for both boundaries or a tuple of two strings for the left and
                right boundaries, respectively.
            x_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                x-direction.
            y_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                y-direction.
            z_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                z-direction.
            x_slab (Optional[Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike]]]): Slab boundary coordinates
                for the x-direction slabs. The tuple should contain two tuples of three
                arrays each, representing the left and right slabs, respectively.
            y_slab (Optional[Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike]]]): Slab boundary coordinates
                for the y-direction slabs. The tuple should contain two tuples of three
                arrays each, representing the left and right slabs, respectively.
            z_slab (Optional[Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike],
                Tuple[ArrayLike, ArrayLike, ArrayLike]]]): Slab boundary coordinates
                for the z-direction slabs. The tuple should contain two tuples of three
                arrays each, representing the left and right slabs, respectively.
        Note:
            See `IMPLEMENTED_BCS` for the implemented boundary condition codes.
        """
        for bc in [bcx, bcy, bcz]:
            if not (isinstance(bc, str) or (isinstance(bc, tuple) and len(bc) == 2)):
                raise ValueError(
                    "Boundary conditions must be a string or tuple of two strings."
                )
        self.array_slicer = array_slicer
        self.bcx = bcx if _is_two_tuple(bcx) else (bcx, bcx)
        self.bcy = bcy if _is_two_tuple(bcy) else (bcy, bcy)
        self.bcz = bcz if _is_two_tuple(bcz) else (bcz, bcz)
        self.x_dirichlet = (
            x_dirichlet if _is_two_tuple(x_dirichlet) else (x_dirichlet, x_dirichlet)
        )
        self.y_dirichlet = (
            y_dirichlet if _is_two_tuple(y_dirichlet) else (y_dirichlet, y_dirichlet)
        )
        self.z_dirichlet = (
            z_dirichlet if _is_two_tuple(z_dirichlet) else (z_dirichlet, z_dirichlet)
        )
        self.x_slab = x_slab if _is_two_tuple(x_slab) else (x_slab, x_slab)
        self.y_slab = y_slab if _is_two_tuple(y_slab) else (y_slab, y_slab)
        self.z_slab = z_slab if _is_two_tuple(z_slab) else (z_slab, z_slab)
        self.slab_slicer = ArraySlicer({}, ndim=3)

        # configure dirichlet functions
        for dim in "xyz":
            configured_dirichlet = list(getattr(self, f"{dim}_dirichlet"))
            for i, f in enumerate(configured_dirichlet):
                if f is None:
                    continue
                if not callable(f):
                    arr = f
                    configured_dirichlet[i] = lambda x, y, z, value=arr: value.reshape(
                        -1, 1, 1, 1
                    )
            setattr(self, f"{dim}_dirichlet", tuple(configured_dirichlet))

    def __call__(
        self,
        arr: ArrayLike,
        pad_width: Tuple[int, int, int],
        check_for_NaNs: bool = False,
    ) -> ArrayLike:
        """
        Apply boundary conditions to an array.

        Args:
            arr (ArrayLike): Array to which to apply boundary conditions.
            pad_width (Tuple[int, int, int]): Tuple of integers indicating the padding
                width for each dimension (x, y, z).
            check_for_NaNs (bool): Whether to check for NaN values in the array after
                applying boundary conditions.
        Returns:
            (ArrayLike): Array with boundary conditions applied.
        """
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
        for i, j in product(range(3), range(2)):
            dim: Literal["x", "y", "z"] = cast(Literal["x", "y", "z"], "xyz"[i])
            pos: Literal["l", "r"] = "l" if j == 0 else "r"
            bc_type = getattr(self, f"bc{dim}")[j]
            slab_thickness = pad_width[i]

            if slab_thickness == 0:
                continue

            match bc_type:
                case "periodic":
                    self._apply_periodic_bc(out, slab_thickness, dim, pos)
                case "dirichlet":
                    self._apply_dirichlet_bc(out, slab_thickness, dim, pos)
                case _:
                    raise ValueError(f"Boundary condition {bc_type} not implemented.")

        # check for NAN values
        if check_for_NaNs and np.any(np.isnan(out)):
            raise ValueError("Boundary conditions resulted in NaN values.")

        return out

    def _apply_periodic_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
    ):
        """
        Apply periodic boundary conditions to arr, modifying it in place.

        Args:
            arr (ArrayLike): Array to which to apply boundary conditions.
            slab_thickness (int): Thickness of the boundary condition slab along the
                axis.
            dim (Literal["x", "y", "z"]): Dimension along which to apply boundary
                conditions ("x", "y", or "z").
            pos (Literal["x", "y", "z"]): Position of the boundary condition slab
                ("l" or "r").
        """
        _slc = self.array_slicer
        _st = slab_thickness
        _axis = "xyz".index(dim) + 1
        if pos == "l":
            outer_slc = _slc(axis=_axis, cut=(None, _st))
            inner_slc = _slc(axis=_axis, cut=(-2 * _st, -_st))
        else:
            outer_slc = _slc(axis=_axis, cut=(-_st, None))
            inner_slc = _slc(axis=_axis, cut=(_st, 2 * _st))
        arr[outer_slc] = arr[inner_slc]

    def _apply_dirichlet_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
    ):
        """
        Apply Dirichlet boundary conditions to arr, modifying it in place.

        Args:
            arr (ArrayLike): Array to which to apply boundary conditions.
            slab_thickness (int): Thickness of the boundary condition slab along the
                axis.
            dim (Literal["x", "y", "z"]): Dimension along which to apply boundary
                conditions ("x", "y", or "z").
            pos (Literal["x", "y", "z"]): Position of the boundary condition slab
                ("l" for left or "r" for right).
        """
        _slc = self.array_slicer
        _axis = "xyz".index(dim) + 1
        f = getattr(self, f"{dim}_dirichlet")["lr".index(pos)]
        if f is None:
            raise ValueError(f"No {dim}{pos}-dirichlet function defined.")
        cut = (None, slab_thickness) if pos == "l" else (-slab_thickness, None)
        shape = arr[_slc(axis=_axis, cut=cut)].shape
        slab_coords = self._get_slab_coords((shape[1], shape[2], shape[3]), dim, pos)
        if "passives" in _slc.group_names:
            bc_arr = f(*slab_coords)
            if bc_arr.shape[0] == len(_slc.idxs):
                # f includes all variables
                arr[_slc(axis=_axis, cut=cut)] = bc_arr
            elif bc_arr.shape[0] == arr[_slc("actives")].shape[0]:
                # f includes only active variables
                arr[_slc("actives", axis=_axis, cut=cut)] = bc_arr
                arr[_slc("passives", axis=_axis, cut=cut)] = bc_arr[_slc("rho")]
            else:
                raise ValueError(
                    "Dirichlet boundary condition function must return an array with "
                    "the same number of variables as the input array."
                )
        else:
            arr[_slc(axis=_axis, cut=cut)] = f(*slab_coords)

    def _get_slab_coords(
        self,
        shape: Tuple[int, int, int],
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the coordinates of the slab along the given axis and position, trimmed to
        the given thickness.

        Args:
            shape (Tuple[int, int, int]): Desired shape of the slab.
            dim (Literal["x", "y", "z"]): Dimension of the slab
                ("x", "y", or "z").
            pos (Literal["l", "r"]): Position of the slab ("l" or "r").
        Returns:
            Tuple[ArrayLike, ArrayLike, ArrayLike]: Coordinates of the slab.
        """
        _slc = self.slab_slicer
        _axis = "xyz".index(dim)
        _slab_thickness = shape[_axis]
        arrs = getattr(self, f"{dim}_slab")["lr".index(pos)]
        if arrs is None:
            raise ValueError(f"No {dim}{pos}-slab defined.")
        cut = (None, _slab_thickness) if pos == "l" else (-_slab_thickness, None)
        out = (
            crop_to_center(arrs[0][_slc(axis=_axis, cut=cut)], shape),
            crop_to_center(arrs[1][_slc(axis=_axis, cut=cut)], shape),
            crop_to_center(arrs[2][_slc(axis=_axis, cut=cut)], shape),
        )
        return out
