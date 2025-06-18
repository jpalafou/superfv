from dataclasses import dataclass
from itertools import product
from typing import Callable, Literal, Optional, Tuple, Union, cast

import numpy as np

from superfv.mesh import UniformFVMesh

from .tools.array_management import (
    CUPY_AVAILABLE,
    ArrayLike,
    VariableIndexMap,
    crop,
    crop_to_center,
    xp,
)

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
        cupy: bool = False,
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
            cupy: If True, use CuPy for array operations instead of NumPy. This is
                useful for GPU acceleration. If False, NumPy will be used.

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

        if "dirichlet" in (*self.bcx, *self.bcy, *self.bcz):
            if mesh is None:
                raise ValueError(
                    "UniformFVMesh must be provided for Dirichlet BCs to determine slab coordinates."
                )
            if conservatives_wrapper is None:
                raise ValueError(
                    "Conservative wrapper must be provided for Dirichlet BCs."
                )
        self.mesh = mesh
        self.conservatives_wrapper = cast(FieldWrapper, conservatives_wrapper)

        # assign numpy namespace
        if cupy and CUPY_AVAILABLE:
            self.xp = xp
        else:
            self.xp = np

    def __call__(
        self,
        arr: ArrayLike,
        pad_width: Tuple[int, int, int],
        primitives: bool = False,
        fv_averages: bool = True,
        t: Optional[float] = None,
        face_quadrature: Optional[Literal["xl", "xr", "yl", "yr", "zl", "zr"]] = None,
        p: int = 0,
    ) -> ArrayLike:
        """
        Apply boundary conditions to an array.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz).
            pad_width: Tuple of integers indicating the padding width for each
                dimension: x (axis 1), y (axis 2), and z (axis 3).
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables. Only used for
                Dirichlet and reflective boundary conditions.
            fv_averages: Whether to compute finite-volume averages of the Dirichlet
                function. If True, the Dirichlet function will be averaged over the
                quadrature points and `arr.ndim` is expected to be 4
                (nvar, nx, ny, nz). If False, the Dirichlet function will be evaluated
                at the quadrature points and the result will be assigned to the
                boundary slab directly, meaning `arr.ndim` is expected to be 5
                (nvar, nx, ny, nz, n_quadrature_points).
            t: Time at which boundary conditions are applied as an argument to the
                Dirichlet function. May be None if the Dirichlet function does not
                depend on time.
            face_quadrature: Optional; if provided, it specifies the face location for
                which to compute quadrature points which are used to evaluate the
                Dirichlet function. Can be one of "xl", "xr", "yl", "yr", "zl", "zr".
                If not provided, the returned qauadrature will span the interior of the
                cell.
            p: Argument for the polynomial degree of the quadrature rule which
                determines the points and weights used to evaluate the Dirichlet
                function. Defaults to 0, in which case the returned quadrature is a
                single face-centered point if `face_quadrature` is provided, or a
                single point in the center of the cell if `face_quadrature` is not
                provided.

        Returns:
           Array with boundary conditions applied. Has shape
           (nvars, nx + 2*pad_width[0], ny + 2*pad_width[1], nz + 2*pad_width[2]).
        """
        # initialize array with boundary conditions
        _pad_width: Tuple[Tuple[int, int], ...] = (
            (0, 0),
            (pad_width[0], pad_width[0]),
            (pad_width[1], pad_width[1]),
            (pad_width[2], pad_width[2]),
        )
        if arr.ndim == 5:
            _pad_width += ((0, 0),)
        elif arr.ndim != 4:
            raise ValueError(
                "Array must be 4D or 5D (nvars, nx, ny, nz[, n_quadrature])"
            )
        out = self.xp.pad(arr, pad_width=_pad_width, mode="empty")

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
                        primitives=primitives,
                        fv_averages=fv_averages,
                        t=t,
                        face_quadrature=face_quadrature,
                        p=p,
                    )
                case "free":
                    self._apply_free_bc(out, slab_thickness, dim, pos)
                case "symmetric":
                    self._apply_symmetric_bc(out, slab_thickness, dim, pos)
                case "reflective":
                    self._apply_reflective_bc(
                        out, slab_thickness, dim, pos, primitives=primitives
                    )
                case "zeros":
                    self._apply_constant_bc(out, slab_thickness, dim, pos, 0.0)
                case "ones":
                    self._apply_constant_bc(out, slab_thickness, dim, pos, 1.0)
                case _:
                    raise ValueError(f"Boundary condition {bc_type} not implemented.")

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
        primitives: bool = False,
        fv_averages: bool = True,
        t: Optional[float] = None,
        face_quadrature: Optional[Literal["xl", "xr", "yl", "yr", "zl", "zr"]] = None,
        p: int = 0,
    ):
        """
        Apply Dirichlet boundary conditions to arr, modifying it in place.

        Args:
            arr: Array to which to apply boundary conditions. Has shape
                (nvars, nx, ny, nz) or (nvars, nx, ny, nz, n_quadrature_points).
            slab_thickness: Number of cells to apply periodic boundary conditions to
                along the specified axis.
            dim: Dimension along which to apply boundary conditions: "x" (axis 1), "y"
                (axis 2), or "z" (axis 3).
            pos: Position of the boundary condition slab: "l" for left or "r" for
                right.
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables.
            fv_averages: Whether to compute finite-volume averages of the Dirichlet
                function. If True, the Dirichlet function will be averaged over the
                quadrature points and `arr.ndim` is expected to be 4
                (nvar, nx, ny, nz). If False, the Dirichlet function will be evaluated
                at the quadrature points and the result will be assigned to the
                boundary slab directly, meaning `arr.ndim` is expected to be 5
                (nvar, nx, ny, nz, n_quadrature_points).
            t: Time at which boundary conditions are applied as an argument to the
                Dirichlet function. May be None if the Dirichlet function does not
                depend on time.
            face_quadrature: Optional; if provided, it specifies the face location for
                which to compute quadrature points which are used to evaluate the
                Dirichlet function. Can be one of "xl", "xr", "yl", "yr", "zl", "zr".
                If not provided, the returned qauadrature will span the interior of the
                cell.
            p: Argument for the polynomial degree of the quadrature rule which
                determines the points and weights used to evaluate the Dirichlet
                function. Defaults to 0, in which case the returned quadrature is a
                single face-centered point if `face_quadrature` is provided, or a
                single point in the center of the cell if `face_quadrature` is not
                provided.

        Returns:
            None: The array is modified in place.
        """
        idx = cast(VariableIndexMap, self.variable_index_map)

        # configure slice
        st = slab_thickness
        axis = "xyz".index(dim) + 1
        outer_slice = crop(axis, (None, st) if pos == "l" else (-st, None))

        # retrieve the appropriate Dirichlet function
        f = getattr(self, f"{dim}_dirichlet")["lr".index(pos)]
        if f is None:
            raise ValueError(f"No {dim}{pos}-dirichlet function defined.")

        # get slab coordinates
        shape = arr[outer_slice].shape
        X, Y, Z, w = self._get_slab_face_quadrature_coords(
            (shape[1], shape[2], shape[3]),
            dim,
            pos,
            face_quadrature,
            p,
        )

        # evaluate the Dirichlet function
        f_wrapped = f if primitives else self.conservatives_wrapper(f)
        f_eval = f_wrapped(idx, X, Y, Z, t)
        if fv_averages:
            f_eval = self.xp.sum(f_eval * w, axis=4)
        arr[outer_slice] = f_eval

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

    def _get_slab_face_quadrature_coords(
        self,
        shape: Tuple[int, int, int],
        dim: Literal["x", "y", "z"],
        pos: Literal["l", "r"],
        face_quadrature: Optional[Literal["xl", "xr", "yl", "yr", "zl", "zr"]] = None,
        p: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the coordinates of the slab along the given axis and position, trimmed to
        the given thickness, using Gauss-Legendre quadrature points.

        Args:
            shape: Desired shape of the slab (nx, ny, nz).
            dim: Dimension along which to get the slab coordinates: "x", "y", or "z".
            pos: Position of the slab: "l" for left or "r" for right.
            face_quadrature: Face identifier for quadrature points, e.g., "xl", "xr",
                "yl", "yr", "zl", "zr".
            p: Polynomial degree of the quadrature rule.

        Returns:
            ...
        """
        X, Y, Z, w = cast(UniformFVMesh, self.mesh).get_gauss_legendre_quadrature(
            p, dim + pos, face_quadrature
        )
        axis = "xyz".index(dim)
        st = shape[axis]
        slab_slice = crop(axis, (-st, None) if pos == "l" else (None, st))
        X, Y, Z = (
            crop_to_center(X[slab_slice], shape),
            crop_to_center(Y[slab_slice], shape),
            crop_to_center(Z[slab_slice], shape),
        )
        return X, Y, Z, w
