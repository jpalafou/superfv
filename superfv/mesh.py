from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Literal, Tuple, cast

import numpy as np

from .fv import gauss_legendre_mesh
from .tools.device_management import ArrayLike, ArrayManager, xp


def _scaled_gauss_legendre_points_and_weights(p: int) -> Tuple[np.ndarray, np.ndarray]:
    unscaled_points, unscaled_weights = np.polynomial.legendre.leggauss(
        -(-(p + 1) // 2)
    )
    scaling = np.sum(unscaled_weights)
    return unscaled_points / scaling, unscaled_weights / scaling


def uniform_3D_mesh(
    nx: int,
    ny: int,
    nz: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_interface = np.linspace(xlim[0], xlim[1], nx + 1)
    y_interface = np.linspace(ylim[0], ylim[1], ny + 1)
    z_interface = np.linspace(zlim[0], zlim[1], nz + 1)
    x_center = 0.5 * (x_interface[1:] + x_interface[:-1])
    y_center = 0.5 * (y_interface[1:] + y_interface[:-1])
    z_center = 0.5 * (z_interface[1:] + z_interface[:-1])
    X, Y, Z = np.meshgrid(x_center, y_center, z_center, indexing="ij")
    return X, Y, Z


@dataclass
class UniformFVMesh:
    """
    A uniform finite volume mesh in 3D space.

    Args:
        nx, ny, nz: Number of cells in the x, y, and z dimensions, respectively.
        xlim, ylim, zlim: Limits of the mesh in the x, y, and z dimensions,
            respectively, as tuples (min, max).
        slab_depth: The depth of the slab in each active dimension, which are
            defined as the dimensions with nx, ny, or nz greater than 1.
        array_manager: An ArrayManager instance for managing arrays on the appropriate
            device (CPU or GPU). This is used to store and manage the mesh arrays
            efficiently, especially when using GPU acceleration.

    Attributes:
        shape: The shape of the core mesh as a tuple (nx, ny, nz).
        size: The total number of cells in the core mesh (nx * ny * nz).
        hx, hy, hz: The cell sizes in the x, y, and z dimensions, respectively.
        active_dims: A tuple of active dimensions (e.g., ('x', 'y')) based on the
            number of cells in each dimension.
        inactive_dims: A tuple of inactive dimensions (e.g., ('z')) based on the
            number of cells in each dimension.
        ndim: The number of active dimensions in the mesh.
        x_is_active, y_is_active, z_is_active: Boolean flags indicating whether the
            x, y, and z dimensions are active (i.e., have more than one cell).
        x_slab_depth, y_slab_depth, z_slab_depth: The slab depth for each active
            dimension, which is equal to slab_depth if the dimension is active,
            otherwise 0.
        _nx_, _ny_, _nz_: The number of cells in the padded mesh, accounting for
            slab_depth in each active dimension.
        _shape_: The shape of the padded mesh as a tuple (_nx_, _ny_, _nz_).
        _size_: The total number of cells in the padded mesh (_nx_ * _ny_ * _nz_).
        x_centers, y_centers, z_centers: The coordinates of the cell centers in the
            x, y, and z dimensions, respectively.
        x_interfaces, y_interfaces, z_interfaces: The coordinates of the cell
            interfaces in the x, y, and z dimensions, respectively.
        X, Y, Z: 3D arrays representing the coordinates of the cell centers in
            the x, y, and z dimensions, respectively.
    """

    nx: int = 1
    ny: int = 1
    nz: int = 1
    xlim: Tuple[float, float] = (0, 1)
    ylim: Tuple[float, float] = (0, 1)
    zlim: Tuple[float, float] = (0, 1)
    slab_depth: int = 1
    array_manager: ArrayManager = field(default_factory=ArrayManager)

    def __post_init__(self):
        self._validate_args()
        self._assign_scalar_attributes()
        self._set_interfaces_and_centers()
        self._init_core()
        self._init_slabs()

    def _validate_args(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        xlim, ylim, zlim = self.xlim, self.ylim, self.zlim
        slab_depth = self.slab_depth

        if any(x < 1 or not isinstance(x, int) for x in (nx, ny, nz)):
            raise ValueError("Mesh dimensions (nx, ny, nz) must be positive integers.")
        if any(
            lower >= upper
            or not (isinstance(lower, (int, float)) and isinstance(upper, (int, float)))
            for lower, upper in (xlim, ylim, zlim)
        ):
            raise ValueError(
                "Limits must be tuples of two values (min, max) with min < max."
            )
        if slab_depth < 0 or not isinstance(slab_depth, int):
            raise ValueError("Slab depth must be a non-negative integer.")

    def _assign_scalar_attributes(self):
        self.shape: Tuple[int, int, int] = (self.nx, self.ny, self.nz)
        self.size: int = self.nx * self.ny * self.nz
        self.hx: float = (self.xlim[1] - self.xlim[0]) / self.nx
        self.hy: float = (self.ylim[1] - self.ylim[0]) / self.ny
        self.hz: float = (self.zlim[1] - self.zlim[0]) / self.nz
        self.active_dims: Tuple[Literal["x", "y", "z"], ...] = tuple(
            cast(Literal["x", "y", "z"], dim)
            for dim, n in zip(["x", "y", "z"], (self.nx, self.ny, self.nz))
            if n > 1
        )
        self.inactive_dims: Tuple[Literal["x", "y", "z"], ...] = tuple(
            cast(Literal["x", "y", "z"], dim)
            for dim in ["x", "y", "z"]
            if dim not in self.active_dims
        )
        self.ndim = len(self.active_dims)
        self.x_is_active: bool = "x" in self.active_dims
        self.y_is_active: bool = "y" in self.active_dims
        self.z_is_active: bool = "z" in self.active_dims
        self.x_slab_depth: int = self.slab_depth if self.x_is_active else 0
        self.y_slab_depth: int = self.slab_depth if self.y_is_active else 0
        self.z_slab_depth: int = self.slab_depth if self.z_is_active else 0
        self._nx_: int = self.nx + 2 * self.x_slab_depth
        self._ny_: int = self.ny + 2 * self.y_slab_depth
        self._nz_: int = self.nz + 2 * self.z_slab_depth
        self._shape_: Tuple[int, int, int] = (self._nx_, self._ny_, self._nz_)
        self._size_: int = self._nx_ * self._ny_ * self._nz_

    def _set_interfaces_and_centers(self):
        x_interfaces = np.linspace(self.xlim[0], self.xlim[1], self.nx + 1)
        y_interfaces = np.linspace(self.ylim[0], self.ylim[1], self.ny + 1)
        z_interfaces = np.linspace(self.zlim[0], self.zlim[1], self.nz + 1)
        x_centers = 0.5 * (x_interfaces[:-1] + x_interfaces[1:])
        y_centers = 0.5 * (y_interfaces[:-1] + y_interfaces[1:])
        z_centers = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])

        self.array_manager.add("core_x_interfaces", x_interfaces)
        self.array_manager.add("core_y_interfaces", y_interfaces)
        self.array_manager.add("core_z_interfaces", z_interfaces)
        self.array_manager.add("core_x_centers", x_centers)
        self.array_manager.add("core_y_centers", y_centers)
        self.array_manager.add("core_z_centers", z_centers)

        # convenient copies
        self.x_centers = self.array_manager.get_numpy_copy("core_x_centers")
        self.y_centers = self.array_manager.get_numpy_copy("core_y_centers")
        self.z_centers = self.array_manager.get_numpy_copy("core_z_centers")
        self.x_interfaces = self.array_manager.get_numpy_copy("core_x_interfaces")
        self.y_interfaces = self.array_manager.get_numpy_copy("core_y_interfaces")
        self.z_interfaces = self.array_manager.get_numpy_copy("core_z_interfaces")

    def _init_core(self):
        X, Y, Z = uniform_3D_mesh(
            self.nx, self.ny, self.nz, self.xlim, self.ylim, self.zlim
        )

        self.array_manager.add("core_X", X)
        self.array_manager.add("core_Y", Y)
        self.array_manager.add("core_Z", Z)

        # convenient copies
        self.X = self.array_manager.get_numpy_copy("core_X")
        self.Y = self.array_manager.get_numpy_copy("core_Y")
        self.Z = self.array_manager.get_numpy_copy("core_Z")

    def _init_slabs(self):
        slab_depth = {
            "x": self.slab_depth if self.x_is_active else 0,
            "y": self.slab_depth if self.y_is_active else 0,
            "z": self.slab_depth if self.z_is_active else 0,
        }
        lim1 = {"x": self.xlim[0], "y": self.ylim[0], "z": self.zlim[0]}
        lim2 = {"x": self.xlim[1], "y": self.ylim[1], "z": self.zlim[1]}
        h = {"x": self.hx, "y": self.hy, "z": self.hz}
        n = {"x": self.nx, "y": self.ny, "z": self.nz}

        for dim, pos in product("xyz", "lr"):
            if dim not in self.active_dims:
                continue
            shape = {
                ax: (slab_depth[ax] if ax == dim else n[ax] + 2 * slab_depth[ax])
                for ax in "xyz"
            }
            bounds = {
                ax: (
                    (
                        (lim1[ax] - slab_depth[ax] * h[ax], lim1[ax])
                        if pos == "l"
                        else (lim2[ax], lim2[ax] + slab_depth[ax] * h[ax])
                    )
                    if ax == dim
                    else (
                        lim1[ax] - slab_depth[ax] * h[ax],
                        lim2[ax] + slab_depth[ax] * h[ax],
                    )
                )
                for ax in "xyz"
            }
            X, Y, Z = uniform_3D_mesh(
                shape["x"],
                shape["y"],
                shape["z"],
                bounds["x"],
                bounds["y"],
                bounds["z"],
            )

            self.array_manager.add(f"{dim}{pos}_slab_X", X)
            self.array_manager.add(f"{dim}{pos}_slab_Y", Y)
            self.array_manager.add(f"{dim}{pos}_slab_Z", Z)

    def get_cell_centers(
        self, region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"] = "core"
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the cell centers of the core mesh or a specific slab.

        Args:
            region: A string indicating which region to return. Must be one of:
                - "core": The main interior mesh (default)
                - "xl", "xr", "yl", "yr", "zl", "zr": The left/right slab in x/y/z

        Returns:
            Tuple of 3D arrays (X, Y, Z) representing cell centers.
        """
        if region == "core":
            return (
                self.array_manager["core_X"],
                self.array_manager["core_Y"],
                self.array_manager["core_Z"],
            )
        return self.get_slab_cell_centers(region)

    def get_slab_cell_centers(
        self,
        region: Literal["xl", "xr", "yl", "yr", "zl", "zr"],
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the cell centers of a specific slab.

        Args:
            region: A string indicating which region to return. Must be one of:
                - "xl", "xr", "yl", "yr", "zl", "zr": The left/right slab in x/y/z

        Returns:
            Tuple of 3D arrays (X, Y, Z) representing cell centers.
        """
        if any(f"{region}_slab_{dim}" not in self.array_manager for dim in "XYZ"):
            raise ValueError(f"{region}_slab not found in arrays.")
        X = self.array_manager[f"{region}_slab_X"]
        Y = self.array_manager[f"{region}_slab_Y"]
        Z = self.array_manager[f"{region}_slab_Z"]
        return X, Y, Z

    def get_GaussLegendre_quadrature(
        self,
        mesh_region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"],
        cell_region: Literal["interior", "xl", "xr", "yl", "yr", "zl", "zr"],
        p: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Get Gauss-Legendre quadrature points and weights for a specific mesh region and
        cell region.

        Args:
            mesh_region: The region of the mesh to use for the quadrature. Must be one
                of "core", "xl", "xr", "yl", "yr", "zl", "zr".
            cell_region: The region of the cell to use for the quadrature. Must be
                "interior" to compute the quadrature points in the interior of each
                cell or one of "xl", "xr", "yl", "yr", "zl", "zr" to compute the
                quadrature points on one of each cell's faces.
            p: Polynomial degree of the quadrature rule. This determines the number of
                quadrature points in each active dimension.

        Returns:
            X, Y, Z: Arrays of the quadrature points in the x, y, and z dimensions,
                respectively, where the quadrature nodes are flattened along axis 3 of
                each array. In other words, each has shape
                (nx, ny, nz, n_quadrature_points), where `n_quadrature_points` depends
                on the polynomial degree `p`.
            w: Array of quadrature weights flattened along axis 3 with shape
                (1, 1, 1, n_quadrature_points).
        """
        key = f"{mesh_region}_{cell_region}_p{p}"
        keys = [f"{key}_{ax}" for ax in "XYZ"] + [f"{key}_w"]

        if all(key in self.array_manager for key in keys):
            X = self.array_manager[keys[0]]
            Y = self.array_manager[keys[1]]
            Z = self.array_manager[keys[2]]
            w = self.array_manager[keys[3]]
            return X, Y, Z, w

        _xp = xp if self.array_manager.device == "gpu" else np
        px = p if self.x_is_active else 0
        py = p if self.y_is_active else 0
        pz = p if self.z_is_active else 0
        h = (self.hx, self.hy, self.hz)

        X, Y, Z = self.get_cell_centers(mesh_region)

        if cell_region == "interior":
            Xp, Yp, Zp, w = gauss_legendre_mesh(_xp, X, Y, Z, h, (px, py, pz))
        else:
            dim, pos = cell_region[0], cell_region[1]
            match dim:
                case "x":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(_xp, X, Y, Z, h, (0, py, pz))
                    Xp = Xp + (-0.5 * self.hx if pos == "l" else 0.5 * self.hx)
                case "y":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(_xp, X, Y, Z, h, (px, 0, pz))
                    Yp = Yp + (-0.5 * self.hy if pos == "l" else 0.5 * self.hy)
                case "z":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(_xp, X, Y, Z, h, (px, py, 0))
                    Zp = Zp + (-0.5 * self.hz if pos == "l" else 0.5 * self.hz)

        self.array_manager.add(keys[0], Xp)
        self.array_manager.add(keys[1], Yp)
        self.array_manager.add(keys[2], Zp)
        self.array_manager.add(keys[3], w)

        return Xp, Yp, Zp, w

    def perform_GaussLegendre_quadrature(
        self,
        f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        node_axis: int,
        mesh_region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"],
        cell_region: Literal["interior", "xl", "xr", "yl", "yr", "zl", "zr"],
        p: int = 0,
    ):
        """
        Perform Gauss-Legendre quadrature on a function over the specified mesh region
        and cell region.

        Args:
            f: A callable function that takes the quadrature points (X, Y, Z)
                and returns an array of values to be integrated.
            node_axis: The axis along which to sum the quadrature results. This is
                typically the axis corresponding to the nodes of the mesh.
            mesh_region: The region of the mesh to use for the quadrature. Must be one
                of "core", "xl", "xr", "yl", "yr", "zl", "zr".
            cell_region: The region of the cell to use for the quadrature. Must be
                "interior" to compute the quadrature points in the interior of each
                cell or one of "xl", "xr", "yl", "yr", "zl", "zr" to compute the
                quadrature points on one of each cell's faces.
            p: Polynomial degree of the quadrature rule. This determines the number of
                quadrature points in each active dimension.
        """
        _xp = xp if self.array_manager.device == "gpu" else np

        X, Y, Z, w = self.get_GaussLegendre_quadrature(mesh_region, cell_region, p)
        f_eval = f(X, Y, Z)
        return _xp.sum(f_eval * w, axis=node_axis)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["gauss_legendre_quadrature_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
