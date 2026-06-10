from dataclasses import dataclass, field, fields
from functools import cached_property
from itertools import product
from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.tools.slicing import crop, merge_slices, replace_slice

from .quadrature import perform_quadrature
from .stencils import conservative_interpolation as ci
from .sweep import stencil_sweep
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager

xyz_tup: Tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")


def uniform_3D_mesh(
    nx: int,
    ny: int,
    nz: int,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    zlims: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_interface = np.linspace(xlims[0], xlims[1], nx + 1)
    y_interface = np.linspace(ylims[0], ylims[1], ny + 1)
    z_interface = np.linspace(zlims[0], zlims[1], nz + 1)
    x_center = 0.5 * (x_interface[1:] + x_interface[:-1])
    y_center = 0.5 * (y_interface[1:] + y_interface[:-1])
    z_center = 0.5 * (z_interface[1:] + z_interface[:-1])
    X, Y, Z = np.meshgrid(x_center, y_center, z_center, indexing="ij")
    return X, Y, Z


@dataclass
class UniformFVMesh:
    nx: int
    ny: int
    nz: int
    nghost: int
    xlims: Tuple[float, float]
    ylims: Tuple[float, float]
    zlims: Tuple[float, float]
    active_dims: Tuple[Literal["x", "y", "z"], ...]
    array_manager: ArrayManager = field(default_factory=ArrayManager)

    @property
    def ndim(self) -> int:
        return len(self.active_dims)

    @property
    def _nx_(self) -> int:
        return self.nx + 2 * self.nghost if "x" in self.active_dims else self.nx

    @property
    def _ny_(self) -> int:
        return self.ny + 2 * self.nghost if "y" in self.active_dims else self.ny

    @property
    def _nz_(self) -> int:
        return self.nz + 2 * self.nghost if "z" in self.active_dims else self.nz

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def _shape_(self) -> Tuple[int, int, int]:
        return (self._nx_, self._ny_, self._nz_)

    @property
    def hx(self) -> float:
        return (self.xlims[1] - self.xlims[0]) / self.nx

    @property
    def hy(self) -> float:
        return (self.ylims[1] - self.ylims[0]) / self.ny

    @property
    def hz(self) -> float:
        return (self.zlims[1] - self.zlims[0]) / self.nz

    def __post_init__(self):
        if len(self.active_dims) not in (1, 2, 3):
            raise ValueError("active_dims must have length 1, 2, or 3.")
        if len(set(self.active_dims)) != len(self.active_dims):
            raise ValueError("active_dims must not contain duplicates.")

        self._set_interfaces_and_centers()
        self._init_core()
        self._init_slabs()

    def _set_interfaces_and_centers(self):
        arrays = self.array_manager

        x_interfaces = np.linspace(self.xlims[0], self.xlims[1], self.nx + 1)
        y_interfaces = np.linspace(self.ylims[0], self.ylims[1], self.ny + 1)
        z_interfaces = np.linspace(self.zlims[0], self.zlims[1], self.nz + 1)
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

        X, Y, Z = uniform_3D_mesh(self.nx, self.ny, self.nz, self.xlims, self.ylims, self.zlims)

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
            "x": self.nghost if "x" in self.active_dims else 0,
            "y": self.nghost if "y" in self.active_dims else 0,
            "z": self.nghost if "z" in self.active_dims else 0,
        }
        lim1 = {"x": self.xlims[0], "y": self.ylims[0], "z": self.zlims[0]}
        lim2 = {"x": self.xlims[1], "y": self.ylims[1], "z": self.zlims[1]}
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
        interior_lims = {"x": self.xlims, "y": self.ylims, "z": self.zlims}

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
                n_slab = self.nghost if dim in active_dims else 0
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
            slice(reach, -reach) if "x" in self.active_dims and reach else slice(None),
            slice(reach, -reach) if "y" in self.active_dims and reach else slice(None),
            slice(reach, -reach) if "z" in self.active_dims and reach else slice(None),
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

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


@dataclass
class UniformFiniteVolumeMeshRegion:
    xlims: Tuple[float, float]
    ylims: Tuple[float, float]
    zlims: Tuple[float, float]
    nx: int
    ny: int
    nz: int
    cupy: bool = False

    @cached_property
    def hx(self) -> float:
        return (self.xlims[1] - self.xlims[0]) / self.nx

    @cached_property
    def hy(self) -> float:
        return (self.ylims[1] - self.ylims[0]) / self.ny

    @cached_property
    def hz(self) -> float:
        return (self.zlims[1] - self.zlims[0]) / self.nz

    @cached_property
    def faces(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        xp = cp if CUPY_AVAILABLE and self.cupy else np
        x_faces = xp.linspace(self.xlims[0], self.xlims[1], self.nx + 1)
        y_faces = xp.linspace(self.ylims[0], self.ylims[1], self.ny + 1)
        z_faces = xp.linspace(self.zlims[0], self.zlims[1], self.nz + 1)
        return x_faces, y_faces, z_faces

    @cached_property
    def centers(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        x_faces, y_faces, z_faces = self.faces
        x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
        y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
        z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
        return x_centers, y_centers, z_centers

    @cached_property
    def Centers(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        xp = cp if CUPY_AVAILABLE and self.cupy else np
        x_centers, y_centers, z_centers = self.centers
        X, Y, Z = xp.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
        return X, Y, Z

    @cached_property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def gauss_legendre_nodes(
        self, p: int, active_dims: Tuple[Literal["x", "y", "z"], ...]
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        xp = cp if CUPY_AVAILABLE and self.cupy else np
        na = xp.newaxis

        GL_stencil = ci.gauss_legendre_nodes(p)
        nnodes = GL_stencil.shape[0]
        nghost = (GL_stencil.shape[1] - 1) // 2

        if CUPY_AVAILABLE and self.cupy:
            GL_stencil = cp.asarray(GL_stencil)

        _mesh_ = UniformFiniteVolumeMeshRegion(
            (
                (self.xlims[0] - nghost * self.hx, self.xlims[1] + nghost * self.hx)
                if "x" in active_dims
                else (self.xlims[0], self.xlims[1])
            ),
            (
                (self.ylims[0] - nghost * self.hy, self.ylims[1] + nghost * self.hy)
                if "y" in active_dims
                else (self.ylims[0], self.ylims[1])
            ),
            (
                (self.zlims[0] - nghost * self.hz, self.zlims[1] + nghost * self.hz)
                if "z" in active_dims
                else (self.zlims[0], self.zlims[1])
            ),
            self.nx + 2 * nghost if "x" in active_dims else self.nx,
            self.ny + 2 * nghost if "y" in active_dims else self.ny,
            self.nz + 2 * nghost if "z" in active_dims else self.nz,
            cupy=self.cupy,
        )
        _Xcc_, _Ycc_, _Zcc_ = _mesh_.Centers
        interior = merge_slices(
            *[crop(DIM_TO_AXIS[dim], (nghost, -nghost), ndim=5) for dim in active_dims]
        )
        interior = replace_slice(interior, 0, 0)

        _Xgl1_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes))
        _Ygl1_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes))
        _Zgl1_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes))

        stencil_sweep(_Xcc_[na, ..., na], GL_stencil, _Xgl1_, active_dims[0])
        stencil_sweep(_Ycc_[na, ..., na], GL_stencil, _Ygl1_, active_dims[0])
        stencil_sweep(_Zcc_[na, ..., na], GL_stencil, _Zgl1_, active_dims[0])

        if len(active_dims) == 1:
            return _Xgl1_[interior], _Ygl1_[interior], _Zgl1_[interior]

        _Xgl2_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**2))
        _Ygl2_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**2))
        _Zgl2_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**2))

        stencil_sweep(_Xgl1_, GL_stencil, _Xgl2_, active_dims[1])
        stencil_sweep(_Ygl1_, GL_stencil, _Ygl2_, active_dims[1])
        stencil_sweep(_Zgl1_, GL_stencil, _Zgl2_, active_dims[1])

        if len(active_dims) == 2:
            return _Xgl2_[interior], _Ygl2_[interior], _Zgl2_[interior]

        _Xgl3_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**3))
        _Ygl3_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**3))
        _Zgl3_ = xp.empty((1, _mesh_.nx, _mesh_.ny, _mesh_.nz, nnodes**3))

        stencil_sweep(_Xgl2_, GL_stencil, _Xgl3_, active_dims[2])
        stencil_sweep(_Ygl2_, GL_stencil, _Ygl3_, active_dims[2])
        stencil_sweep(_Zgl2_, GL_stencil, _Zgl3_, active_dims[2])

        if len(active_dims) == 3:
            return _Xgl3_[interior], _Ygl3_[interior], _Zgl3_[interior]

        raise ValueError("Invalid number of active dimensions.")

    def __getstate__(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state):
        self.__dict__.update(state)


@dataclass
class UniformFinteVolumeMesh:
    xlims: Tuple[float, float]
    ylims: Tuple[float, float]
    zlims: Tuple[float, float]
    nx: int
    ny: int
    nz: int
    nghost: int
    active_dims: Tuple[Literal["x", "y", "z"], ...]
    cupy: bool = False

    @property
    def hx(self) -> float:
        return self.core.hx

    @property
    def hy(self) -> float:
        return self.core.hy

    @property
    def hz(self) -> float:
        return self.core.hz

    @property
    def faces(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        return self.core.faces

    @property
    def centers(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        return self.core.centers

    @property
    def Centers(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        return self.core.Centers

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def gauss_legendre_nodes(
        self,
        p: int,
        region: Literal[
            "core", "xl_slab", "xr_slab", "yl_slab", "yr_slab", "zl_slab", "zr_slab"
        ] = "core",
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        return getattr(self, region).gauss_legendre_nodes(p, self.active_dims)

    @property
    def _nx_(self) -> int:
        return self.nx + 2 * self.nghost if "x" in self.active_dims else self.nx

    @property
    def _ny_(self) -> int:
        return self.ny + 2 * self.nghost if "y" in self.active_dims else self.ny

    @property
    def _nz_(self) -> int:
        return self.nz + 2 * self.nghost if "z" in self.active_dims else self.nz

    @property
    def _shape_(self) -> Tuple[int, int, int]:
        return (self._nx_, self._ny_, self._nz_)

    def __post_init__(self):
        self.core = UniformFiniteVolumeMeshRegion(
            self.xlims, self.ylims, self.zlims, self.nx, self.ny, self.nz, cupy=self.cupy
        )

        nghostx = self.nghost if "x" in self.active_dims else 0
        nghosty = self.nghost if "y" in self.active_dims else 0
        nghostz = self.nghost if "z" in self.active_dims else 0

        self.xl_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[0] - nghostx * self.hx, self.xlims[0]),
            (self.ylims[0] - nghosty * self.hy, self.ylims[1] + nghosty * self.hy),
            (self.zlims[0] - nghostz * self.hz, self.zlims[1] + nghostz * self.hz),
            nghostx,
            self.ny + 2 * nghosty,
            self.nz + 2 * nghostz,
            cupy=self.cupy,
        )
        self.xr_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[1], self.xlims[1] + nghostx * self.hx),
            (self.ylims[0] - nghosty * self.hy, self.ylims[1] + nghosty * self.hy),
            (self.zlims[0] - nghostz * self.hz, self.zlims[1] + nghostz * self.hz),
            nghostx,
            self.ny + 2 * nghosty,
            self.nz + 2 * nghostz,
            cupy=self.cupy,
        )
        self.yl_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[0] - nghostx * self.hx, self.xlims[1] + nghostx * self.hx),
            (self.ylims[0] - nghosty * self.hy, self.ylims[0]),
            (self.zlims[0] - nghostz * self.hz, self.zlims[1] + nghostz * self.hz),
            self.nx + 2 * nghostx,
            nghosty,
            self.nz + 2 * nghostz,
            cupy=self.cupy,
        )
        self.yr_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[0] - nghostx * self.hx, self.xlims[1] + nghostx * self.hx),
            (self.ylims[1], self.ylims[1] + nghosty * self.hy),
            (self.zlims[0] - nghostz * self.hz, self.zlims[1] + nghostz * self.hz),
            self.nx + 2 * nghostx,
            nghosty,
            self.nz + 2 * nghostz,
            cupy=self.cupy,
        )
        self.zl_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[0] - nghostx * self.hx, self.xlims[1] + nghostx * self.hx),
            (self.ylims[0] - nghosty * self.hy, self.ylims[1] + nghosty * self.hy),
            (self.zlims[0] - nghostz * self.hz, self.zlims[0]),
            self.nx + 2 * nghostx,
            self.ny + 2 * nghosty,
            nghostz,
            cupy=self.cupy,
        )
        self.zr_slab = UniformFiniteVolumeMeshRegion(
            (self.xlims[0] - nghostx * self.hx, self.xlims[1] + nghostx * self.hx),
            (self.ylims[0] - nghosty * self.hy, self.ylims[1] + nghosty * self.hy),
            (self.zlims[1], self.zlims[1] + nghostz * self.hz),
            self.nx + 2 * nghostx,
            self.ny + 2 * nghosty,
            nghostz,
            cupy=self.cupy,
        )

    def __getstate__(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()
