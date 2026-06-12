import warnings
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Callable, Dict, Literal, Tuple, Union

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.tools.slicing import crop, merge_slices, replace_slice

from .stencils import conservative_interpolation as ci
from .sweep import stencil_sweep
from .tools.device_management import CUPY_AVAILABLE, ArrayLike

xyz_tup: Tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")

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

    def __post_init__(self):
        if self.cupy and not CUPY_AVAILABLE:
            warnings.warn("Cupy is not available. Falling back to NumPy.")
            self.cupy = False

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

    def GaussLegendre_nodes(
        self, sampling_p: int, active_dims: Tuple[Literal["x", "y", "z"], ...]
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        xp = cp if CUPY_AVAILABLE and self.cupy else np
        na = xp.newaxis

        GL_stencil = ci.gauss_legendre_nodes(sampling_p)
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
        interior: Tuple[Union[int, slice], ...] = merge_slices(
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

    def perform_GaussLegendre_quadrature(
        self,
        f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        sampling_p: int,
        active_dims: Tuple[Literal["x", "y", "z"], ...],
    ) -> ArrayLike:
        xp = cp if CUPY_AVAILABLE and self.cupy else np
        ndim = len(active_dims)

        GL_weights = ci.gauss_legendre_weights(sampling_p, ndim)
        if CUPY_AVAILABLE and self.cupy:
            GL_weights = cp.asarray(GL_weights)

        Xgl, Ygl, Zgl = self.GaussLegendre_nodes(sampling_p, active_dims)
        f_eval = f(Xgl, Ygl, Zgl)

        ndim_eval = f_eval.ndim
        return xp.sum(f_eval * GL_weights.reshape(*[1] * (ndim_eval - 1), -1), axis=-1)

    def __getstate__(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()


@dataclass
class UniformFiniteVolumeMesh:
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

    def GaussLegendre_nodes(
        self,
        sampling_p: int,
        region: Literal[
            "core", "xl_slab", "xr_slab", "yl_slab", "yr_slab", "zl_slab", "zr_slab"
        ] = "core",
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        return getattr(self, region).GaussLegendre_nodes(sampling_p, self.active_dims)

    def perform_GaussLegendre_quadrature(
        self,
        f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        sampling_p: int,
        region: Literal[
            "core", "xl_slab", "xr_slab", "yl_slab", "yr_slab", "zl_slab", "zr_slab"
        ] = "core",
    ) -> ArrayLike:
        return getattr(self, region).perform_GaussLegendre_quadrature(
            f, sampling_p, self.active_dims
        )

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

    @property
    def ndim(self) -> int:
        return len(self.active_dims)

    def __post_init__(self):
        if self.cupy and not CUPY_AVAILABLE:
            warnings.warn("Cupy is not available. Falling back to NumPy.")
            self.cupy = False

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
