from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Literal, Tuple

import numpy as np

from .quadrature import perform_quadrature
from .stencils import conservative_interpolation as ci
from .sweep import stencil_sweep
from .tools.device_management import ArrayLike, ArrayManager

xyz_tup: Tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")


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
        active_dims: A tuple of active dimensions containing "x", "y", and/or "z".
            Active dimensions are those that get assigned a slab array. Inactive
            dimensions must have a size of 1.
        slab_depth: The depth of the slab in each active dimension, which are
            defined as the dimensions with nx, ny, or nz greater than 1.
        array_manager: An ArrayManager instance for managing arrays on the appropriate
            device (CPU or GPU). This is used to store and manage the mesh arrays
            efficiently, especially when using GPU acceleration.

    Attributes:
        shape: The shape of the core mesh as a tuple (nx, ny, nz).
        size: The total number of cells in the core mesh (nx * ny * nz).
        hx, hy, hz: The cell sizes in the x, y, and z dimensions, respectively.
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
    active_dims: Tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
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
        active_dims = self.active_dims
        slab_depth = self.slab_depth

        if any(x < 1 or not isinstance(x, int) for x in (nx, ny, nz)):
            raise ValueError("Mesh dimensions (nx, ny, nz) must be positive integers.")
        if any(
            lower >= upper
            or not (isinstance(lower, (int, float)) and isinstance(upper, (int, float)))
            for lower, upper in (xlim, ylim, zlim)
        ):
            raise ValueError("Limits must be tuples of two values (min, max) with min < max.")
        if any(dim not in xyz_tup for dim in active_dims):
            raise ValueError("Active dimensions must be 'x', 'y', and/or 'z'.")
        if any(n > 1 and dim not in active_dims for dim, n in zip(xyz_tup, (nx, ny, nz))):
            raise ValueError("Inactive dimensions must have only one cell.")
        if slab_depth < 0 or not isinstance(slab_depth, int):
            raise ValueError("Slab depth must be a non-negative integer.")

    def _assign_scalar_attributes(self):
        self.shape: Tuple[int, int, int] = (self.nx, self.ny, self.nz)
        self.size: int = self.nx * self.ny * self.nz
        self.hx: float = (self.xlim[1] - self.xlim[0]) / self.nx
        self.hy: float = (self.ylim[1] - self.ylim[0]) / self.ny
        self.hz: float = (self.zlim[1] - self.zlim[0]) / self.nz
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
        arrays = self.array_manager

        x_interfaces = np.linspace(self.xlim[0], self.xlim[1], self.nx + 1)
        y_interfaces = np.linspace(self.ylim[0], self.ylim[1], self.ny + 1)
        z_interfaces = np.linspace(self.zlim[0], self.zlim[1], self.nz + 1)
        x_centers = 0.5 * (x_interfaces[:-1] + x_interfaces[1:])
        y_centers = 0.5 * (y_interfaces[:-1] + y_interfaces[1:])
        z_centers = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])

        arrays.add("core_x_interfaces", x_interfaces)
        arrays.add("core_y_interfaces", y_interfaces)
        arrays.add("core_z_interfaces", z_interfaces)
        arrays.add("core_x_centers", x_centers)
        arrays.add("core_y_centers", y_centers)
        arrays.add("core_z_centers", z_centers)

        # convenient copies
        self.x_centers = arrays.get_numpy_copy("core_x_centers")
        self.y_centers = arrays.get_numpy_copy("core_y_centers")
        self.z_centers = arrays.get_numpy_copy("core_z_centers")
        self.x_interfaces = arrays.get_numpy_copy("core_x_interfaces")
        self.y_interfaces = arrays.get_numpy_copy("core_y_interfaces")
        self.z_interfaces = arrays.get_numpy_copy("core_z_interfaces")

    def _init_core(self):
        arrays = self.array_manager

        X, Y, Z = uniform_3D_mesh(self.nx, self.ny, self.nz, self.xlim, self.ylim, self.zlim)

        arrays.add("core_X", X)
        arrays.add("core_Y", Y)
        arrays.add("core_Z", Z)

        # convenient copies
        self.X = arrays.get_numpy_copy("core_X")
        self.Y = arrays.get_numpy_copy("core_Y")
        self.Z = arrays.get_numpy_copy("core_Z")

    def _init_slabs(self):
        arrays = self.array_manager

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
                ax: (slab_depth[ax] if ax == dim else n[ax] + 2 * slab_depth[ax]) for ax in "xyz"
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

            arrays.add(f"{dim}{pos}_slab_X", X)
            arrays.add(f"{dim}{pos}_slab_Y", Y)
            arrays.add(f"{dim}{pos}_slab_Z", Z)

    def get_cell_centers(
        self,
        region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"] = "core",
        numpy_copy: bool = False,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the cell centers of the core mesh or a specific slab.

        Args:
            region: A string indicating which region to return. Must be one of:
                - "core": The main interior mesh (default)
                - "xl", "xr", "yl", "yr", "zl", "zr": The left/right slab in x/y/z
            numpy_copy: If True, return a NumPy array copy of the cell centers.

        Returns:
            Tuple of 3D arrays (X, Y, Z) representing cell centers.
        """
        arrays = self.array_manager

        if region == "core" and numpy_copy:
            return (
                arrays.get_numpy_copy("core_X"),
                arrays.get_numpy_copy("core_Y"),
                arrays.get_numpy_copy("core_Z"),
            )
        elif region == "core":
            return (
                arrays["core_X"],
                arrays["core_Y"],
                arrays["core_Z"],
            )
        else:
            return self.get_slab_cell_centers(region, numpy_copy=numpy_copy)

    def get_slab_cell_centers(
        self,
        region: Literal["xl", "xr", "yl", "yr", "zl", "zr"],
        numpy_copy: bool = False,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get the cell centers of a specific slab.

        Args:
            region: A string indicating which region to return. Must be one of:
                - "xl", "xr", "yl", "yr", "zl", "zr": The left/right slab in x/y/z
            numpy_copy: If True, return a NumPy array copy of the slab cell centers.

        Returns:
            Tuple of 3D arrays (X, Y, Z) representing cell centers.
        """
        arrays = self.array_manager

        if any(f"{region}_slab_{dim}" not in arrays for dim in "XYZ"):
            raise ValueError(f"{region}_slab not found in array manager.")

        if numpy_copy:
            return (
                arrays.get_numpy_copy(f"{region}_slab_X"),
                arrays.get_numpy_copy(f"{region}_slab_Y"),
                arrays.get_numpy_copy(f"{region}_slab_Z"),
            )
        else:
            return (
                arrays[f"{region}_slab_X"],
                arrays[f"{region}_slab_Y"],
                arrays[f"{region}_slab_Z"],
            )

    def get_GaussLegendre_mesh(
        self,
        mesh_region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"],
        cell_region: Literal["interior", "xl", "xr", "yl", "yr", "zl", "zr"],
        p: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Get Gauss-Legendre quadrature points for a specific mesh region and for either
        the interior of each cell or one of the faces of each cell.

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
        """
        arrays = self.array_manager
        key = f"{mesh_region}_{cell_region}_p{p}"
        keys = [f"{key}_{ax}" for ax in "XYZ"]

        # load from array manager if already computed
        if all(key in arrays for key in keys):
            Xgl = arrays[keys[0]]
            Ygl = arrays[keys[1]]
            Zgl = arrays[keys[2]]
            return Xgl, Ygl, Zgl

        # must compute the mesh, start by loading necessary attributes
        active_dims = self.active_dims
        h = {"x": self.hx, "y": self.hy, "z": self.hz}
        na = np.newaxis

        # get stencils which might be needed
        lr_stencils = ci.left_right(p)
        gl_stencils = ci.gauss_legendre_nodes(p)
        _, lr_stencil_size = gl_stencils.shape
        ngl_nodes, gl_stencil_size = gl_stencils.shape
        stencil_size = max(lr_stencil_size, gl_stencil_size)
        reach = (stencil_size - 1) // 2

        # get limits and resolution for the specified mesh region
        interior_lims = {"x": self.xlim, "y": self.ylim, "z": self.zlim}

        if mesh_region == "core":
            region_lims = {dim: interior_lims[dim] for dim in "xyz"}
            resolution = {"x": self.nx, "y": self.ny, "z": self.nz}
        else:
            slab_dim = mesh_region[0]
            slab_side = mesh_region[1]

            region_lims = {}
            resolution = {}
            for dim in "xyz":
                lim0, lim1 = interior_lims[dim]
                n_slab = getattr(self, f"{dim}_slab_depth")
                slab_range = n_slab * h[dim]

                if dim == slab_dim:
                    if slab_side == "l":
                        region_lims[dim] = (lim0 - slab_range, lim0)
                    else:
                        region_lims[dim] = (lim1, lim1 + slab_range)
                    resolution[dim] = n_slab
                else:
                    region_lims[dim] = (lim0 - slab_range, lim1 + slab_range)
                    resolution[dim] = getattr(self, f"_n{dim}_")

        # expand limits to account for reach of the interpolation stencil
        for dim in active_dims:
            extension = reach * h[dim]
            lim0, lim1 = region_lims[dim]
            region_lims[dim] = (lim0 - extension, lim1 + extension)
            resolution[dim] += 2 * reach

        # get the padded mesh for the specified region
        _X3d_, _Y3d_, _Z3d_ = uniform_3D_mesh(
            resolution["x"],
            resolution["y"],
            resolution["z"],
            region_lims["x"],
            region_lims["y"],
            region_lims["z"],
        )
        _X_, _Y_, _Z_ = _X3d_[na, ..., na], _Y3d_[na, ..., na], _Z3d_[na, ..., na]

        # decide where to sweep
        sweep_instructions = {}
        for dim in active_dims:  # the lr sweep is first
            if cell_region != "interior" and cell_region[0] == dim:
                sweep_instructions[dim] = "lr"
        sweep_instructions = {}
        for dim in active_dims:
            if cell_region == "interior" or cell_region[0] != dim:
                sweep_instructions[dim] = "gl"

        in_arrays = {"x": _X_, "y": _Y_, "z": _Z_}
        out_arrays = {}

        # perform sweeps to get the Gauss-Legendre mesh
        for sweep_dim, sweep_type in sweep_instructions.items():
            stencils = gl_stencils if sweep_type == "gl" else lr_stencils
            nouterps = ngl_nodes if sweep_type == "gl" else 2

            for mesh_dim, in_mesh in in_arrays.items():
                _, _, _, _, ninterps = in_mesh.shape
                out_mesh = np.empty((*in_mesh.shape[:4], ninterps * nouterps))
                stencil_sweep(in_mesh, stencils, out_mesh, sweep_dim)
                out_arrays[mesh_dim] = out_mesh
            in_arrays = out_arrays

        inner = (
            slice(reach, -reach) if self.x_is_active and reach else slice(None),
            slice(reach, -reach) if self.y_is_active and reach else slice(None),
            slice(reach, -reach) if self.z_is_active and reach else slice(None),
            slice(None),
        )
        Xgl = out_arrays["x"][0, *inner]
        Ygl = out_arrays["y"][0, *inner]
        Zgl = out_arrays["z"][0, *inner]

        # add arrays to array manager
        arrays.add(keys[0], Xgl)
        arrays.add(keys[1], Ygl)
        arrays.add(keys[2], Zgl)

        # call them back now that they're on the correct device
        Xgl, Ygl, Zgl = arrays[keys[0]], arrays[keys[1]], arrays[keys[2]]

        return Xgl, Ygl, Zgl

    def perform_GaussLegendre_quadrature(
        self,
        f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        out: ArrayLike,
        mesh_region: Literal["core", "xl", "xr", "yl", "yr", "zl", "zr"],
        cell_region: Literal["interior", "xl", "xr", "yl", "yr", "zl", "zr"],
        p: int = 0,
    ):
        """
        Perform Gauss-Legendre quadrature on a function over the specified mesh region
        and cell region, writing the result to `out`.

        Args:
            f: A callable function that takes the quadrature points (X, Y, Z)
                and returns an array of values to be integrated.
            out: The array to which the quadrature result will be written.
            mesh_region: The region of the mesh to use for the quadrature. Must be one
                of "core", "xl", "xr", "yl", "yr", "zl", "zr".
            cell_region: The region of the cell to use for the quadrature.
                "interior" to compute the quadrature points in the interior of each
                cell or one of "xl", "xr", "yl", "yr", "zl", "zr" to compute the
                quadrature points on one of each cell's faces.
            p: Polynomial degree of the quadrature rule. This determines the number of
                quadrature points in each active dimension.
        """
        arrays = self.array_manager
        gl_ndim = self.ndim if cell_region == "interior" else self.ndim - 1

        weights_key = f"GL_WEIGHTS_p{p}_ndim{gl_ndim}"
        if weights_key not in arrays:
            arr = ci.gauss_legendre_weights(p, gl_ndim)
            arrays.add(weights_key, arr)
        weights = arrays[weights_key]

        X, Y, Z = self.get_GaussLegendre_mesh(mesh_region, cell_region, p)
        f_eval = f(X, Y, Z)

        perform_quadrature(f_eval, weights, out)

    def to_dict(self) -> dict:
        return dict(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            xlim=self.xlim,
            ylim=self.ylim,
            zlim=self.zlim,
            active_dims=self.active_dims,
            slab_depth=self.slab_depth,
        )

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
