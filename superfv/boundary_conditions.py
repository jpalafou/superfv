from dataclasses import dataclass
from itertools import product
from typing import Callable, Literal, Optional, Tuple, Union, cast

import numpy as np

from superfv.mesh import UniformFVMesh

from .tools.array_management import ArrayLike, VariableIndexMap, crop, crop_to_center

# define custom type annotations
Field = Callable[
    [VariableIndexMap, ArrayLike, ArrayLike, ArrayLike, Optional[float]],
    ArrayLike,
]
DirichletBC = Union[Field, Tuple[Optional[Field], Optional[Field]]]
FieldWrapper = Callable[[Field], Field]


def _is_two_tuple(x):
    return isinstance(x, tuple) and len(x) == 2


@dataclass
class BoundaryConditions:
    """
    Class for managing finite-volume boundary conditions in 3D.
    """

    def __init__(
        self,
        variable_index_map: Optional[VariableIndexMap] = None,
        mesh: Optional[UniformFVMesh] = None,
        bcx: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcy: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcz: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        x_dirichlet: Optional[DirichletBC] = None,
        y_dirichlet: Optional[DirichletBC] = None,
        z_dirichlet: Optional[DirichletBC] = None,
        conservatives_wrapper: Optional[FieldWrapper] = None,
        fv_average_wrapper: Optional[FieldWrapper] = None,
    ):
        """
        Initialize boundary conditions.

        Args:
            variable_index_map: VariableIndexMap object.
            mesh: Optional UniformFVMesh object. If provided, it will be used to
                determine the slab coordinates for the Dirichlet functions.
            bcx, bcy, bcz: Boundary conditions for the x, y, and z directions. Each can
                be specified as a single string to apply the same condition on both
                sides, or as a tuple of two strings to apply different conditions on
                the lower and upper (left and right) boundaries, respectively.
                Supported boundary condition names include: "periodic", "dirichlet",
                "free", "reflective", "zeros", and "ones".
            x_dirichlet, y_dirichlet, z_dirichlet: Additional argument for "dirichlet"
                boundary conditions. Must be a callable that takes following arguments:
                - idx: VariableIndexMap object.
                - x: x-coordinate array. Has shape (nx, ny, nz).
                - y: y-coordinate array. Has shape (nx, ny, nz).
                - z: z-coordinate array. Has shape (nx, ny, nz).
                - t: Optional time at which the boundary condition is applied.
                And returns an array with shape (nvars, nx, ny, nz). Can also be given
                as a tuple of two callables, one for the left and one for the right
                boundary condition. If a single callable is provided, it will be used
                for both boundaries.
            conservatives_wrapper: Wrapper to convert output of the Dirichlet functions
                from primitive variables to conservative variables. This is required if
                any of the boundary conditions is "dirichlet". If not provided, it will
                raise an error when a Dirichlet boundary condition is applied.
            fv_average_wrapper: Wrapper to convert output of the Dirichlet functions
                from pointwise values to finite-volume average values. This is required
                if any of the boundary conditions is "dirichlet". If not provided, it
                will raise an error when a Dirichlet boundary condition is applied.

        Note:
            See `IMPLEMENTED_BCS` for the implemented boundary condition codes.
        """
        for bc in [bcx, bcy, bcz]:
            if not (isinstance(bc, str) or (isinstance(bc, tuple) and len(bc) == 2)):
                raise ValueError(
                    "Boundary conditions must be a string or tuple of two strings."
                )
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

        # validate and assign variable index map, mesh, and wrappers
        if any(
            bc in ("dirichlet", "reflective")
            for bc in (*self.bcx, *self.bcy, *self.bcz)
        ):
            if variable_index_map is None:
                raise ValueError(
                    "VariableIndexMap must be provided for Dirichlet or reflective BCs."
                )
        self.variable_index_map = variable_index_map

        if any(bc == "dirichlet" for bc in (*self.bcx, *self.bcy, *self.bcz)):
            if mesh is None:
                raise ValueError(
                    "UniformFVMesh must be provided for Dirichlet BCs to determine slab coordinates."
                )
            if conservatives_wrapper is None:
                raise ValueError(
                    "Conservative wrapper must be provided for Dirichlet BCs."
                )
            if fv_average_wrapper is None:
                raise ValueError(
                    "Finite-volume average wrapper must be provided for Dirichlet BCs."
                )
        self.mesh = mesh
        self.conservatives_wrapper = cast(FieldWrapper, conservatives_wrapper)
        self.fv_average_wrapper = cast(FieldWrapper, fv_average_wrapper)

    def __call__(
        self,
        arr: ArrayLike,
        pad_width: Tuple[int, int, int],
        t: Optional[float] = None,
        primitives: bool = False,
        pointwise: bool = False,
        check_for_NaNs: bool = False,
    ) -> ArrayLike:
        """
        Apply boundary conditions to an array.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            pad_width: Tuple of integers indicating the padding width for each
                dimension: x (axis 1), y (axis 2), and z (axis 3).
            t: Time at which boundary conditions are applied. This is only used for
                Dirichlet boundary conditions.
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables.
            pointwise: Whether `arr` contains pointwise values. If False, it is assumed
                that `arr` contains finite-volume average values.
            check_for_NaNs: Whether to check for NaN values in the array after applying
                boundary conditions.
        Returns:
           Array with boundary conditions applied. Has shape
           (nvars, nx + 2*pad_width[0], ny + 2*pad_width[1], nz + 2*pad_width[2]).
        """
        # initialize array with boundary conditions
        if arr.ndim != 4:
            raise ValueError("Array must be 4D (nvars, nx, ny, nz).")
        _pad_width = (
            (0, 0),
            (pad_width[0], pad_width[0]),
            (pad_width[1], pad_width[1]),
            (pad_width[2], pad_width[2]),
        )
        out = np.pad(arr, pad_width=_pad_width, mode="empty")

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
                    self._apply_dirichlet_bc(
                        out,
                        slab_thickness,
                        dim,
                        pos,
                        t,
                        primitives,
                        pointwise,
                    )
                case "free":
                    self._apply_free_bc(out, slab_thickness, dim, pos)
                case "symmetric":
                    self._apply_symmetric_bc(out, slab_thickness, dim, pos)
                case "reflective":
                    self._apply_reflective_bc(out, slab_thickness, dim, pos, primitives)
                case "zeros":
                    self._apply_constant_bc(out, slab_thickness, dim, pos, 0.0)
                case "ones":
                    self._apply_constant_bc(out, slab_thickness, dim, pos, 1.0)
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
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.

        Returns:
            None: The array is modified in place.
        """
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        if pos == "l":
            outer_slice = crop(axis, (None, st))
            inner_slice = crop(axis, (-2 * st, -st))
        else:
            outer_slice = crop(axis, (-st, None))
            inner_slice = crop(axis, (st, 2 * st))
        arr[outer_slice] = arr[inner_slice]

    def _apply_dirichlet_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
        t: Optional[float] = None,
        primitives: bool = False,
        pointwise: bool = False,
    ):
        """
        Apply Dirichlet boundary conditions to arr, modifying it in place.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.
            t: Time at which boundary conditions are applied. This is only used for
                Dirichlet boundary conditions.
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables.
            pointwise: Whether `arr` contains pointwise values. If False, it is assumed
                that `arr` contains finite-volume average values.

        Returns:
            None: The array is modified in place.
        """
        idx = cast(VariableIndexMap, self.variable_index_map)

        # configure slice
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        outer_slice = crop(axis, (None, st) if pos == "l" else (-st, None))

        # configure dirichlet function
        conservative, fv_average = not primitives, not pointwise
        f = getattr(self, f"{dim}_dirichlet")["lr".index(pos)]
        if f is None:
            raise ValueError(f"No {dim}{pos}-dirichlet function defined.")
        shape = arr[outer_slice].shape
        slab_coords = self._get_slab_coords((shape[1], shape[2], shape[3]), dim, pos)

        # apply the dirichlet function
        if conservative and fv_average:
            arr[outer_slice] = self.fv_average_wrapper(self.conservatives_wrapper(f))(
                idx, *slab_coords, t
            )
        elif conservative:
            arr[outer_slice] = self.conservatives_wrapper(f)(idx, *slab_coords, t)
        elif fv_average:
            arr[outer_slice] = self.fv_average_wrapper(f)(idx, *slab_coords, t)
        else:
            arr[outer_slice] = f(idx, *slab_coords, t)

    def _apply_free_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
    ):
        """
        Apply free boundary conditions to arr, modifying it in place.

        Args:
             arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.

        Returns:
            None: The array is modified in place.
        """
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        if pos == "l":
            outer_slice = crop(axis, (0, st))
            inner_slice = crop(axis, (st, st + 1))
        else:
            outer_slice = crop(axis, (-st, 0))
            inner_slice = crop(axis, (-st - 1, -st))
        arr[outer_slice] = arr[inner_slice]

    def _apply_symmetric_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
    ):
        """
        Apply symmetric boundary conditions to arr, modifying it in place.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.

        Returns:
            None: The array is modified in place.
        """
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        flipper_slice = crop(axis, (None, None), step=-1)
        if pos == "l":
            outer_slice = crop(axis, (0, st))
            inner_slice = crop(axis, (st, 2 * st))
        else:
            outer_slice = crop(axis, (-st, 0))
            inner_slice = crop(axis, (-2 * st, -st))
        arr[outer_slice] = arr[inner_slice][flipper_slice]

    def _apply_reflective_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
        primitives: bool = False,
    ):
        """
        Apply reflective boundary conditions to arr, modifying it in place.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables.

        Returns:
            None: The array is modified in place.
        """
        idx = cast(VariableIndexMap, self.variable_index_map)
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        if pos == "l":
            outer_slice = crop(axis, (0, st))
        else:
            outer_slice = crop(axis, (-st, 0))
        self._apply_symmetric_bc(arr, slab_thickness, dim, pos)
        arr[outer_slice][idx(("v" if primitives else "m") + dim)] *= -1

    def _apply_constant_bc(
        self,
        arr: ArrayLike,
        slab_thickness: int,
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
        value: float,
    ):
        """
        Apply zero boundary conditions to arr, modifying it in place.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.
            value: Value to set the boundary condition to.

        Returns:
            None: The array is modified in place.
        """
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        arr[crop(axis, (None, st) if pos == "l" else (-st, None))] = value

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
            shape: Desired shape of the slab (nx, ny, nz).
            dim: Dimension along which to get the slab: "x", "y", or "z".
            pos: Position of the slab: "l" for left or "r" for right.

        Returns:
            Tuple of three arrays representing the slab coordinates in the order
            (x-coordinates, y-coordinates, z-coordinates).
        """
        arrs = getattr(cast(UniformFVMesh, self.mesh), f"{dim}{pos}_slab")
        axis = "xyz".index(dim)
        st = shape[axis]
        slab_slice = crop(axis, (-st, None) if pos == "l" else (None, st))
        out = (
            crop_to_center(arrs[0][slab_slice], shape),
            crop_to_center(arrs[1][slab_slice], shape),
            crop_to_center(arrs[2][slab_slice], shape),
        )
        return out
