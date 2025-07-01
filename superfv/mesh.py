from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import numpy as np

from superfv.fv import gauss_legendre_mesh


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
    A class to represent a uniform finite volume mesh in 3D space.

    Args:
        nx, ny, nz: Number of cells in the x, y, and z dimensions.
        xlim, ylim, zlim: Limits of the mesh in the x, y, and z dimensions.
        ignore_x, ignore_y, ignore_z: If True, this dimension will no longer be
            considered 'active' in the mesh, meaning it will not be considered when
            generating slabs or quadrature points.
        slab_depth: Number of cells to add as slabs on each side of the mesh along each
            active dimension.

    Attributes:
        coords: Tuple of 3D arrays representing the coordinates of the mesh
            `(X, Y, Z)`.
        h: Dictionary containing cell sizes in each dimension, e.g.,
            {"x": hx, "y": hy, "z": hz}.
        hx, hy, hz: Cell sizes in the x, y, and z dimensions.
        pad_x, pad_y, pad_z: Depth of each slab along each dimension.
        shape: Shape of the mesh as (nx, ny, nz).
        size: Total number of cells in the mesh (nx * ny * nz).
        X, Y, Z: Mesh grid arrays for the x, y, and z dimensions as 3D arrays.
        x_active, y_active, z_active: Boolean flags indicating whether the x, y, and z
            dimensions are active in the mesh (i.e., not ignored).
        x_centers, y_centers, z_centers (ndarray): Centers of the cells in the x,
            dimensions as 1D arrays.
        x_interfaces, y_interfaces, z_interfaces: Interfaces in the x, y, and z
            and z dimensions as 1D arrays.
        xl_slab, xr_slab, yl_slab, yr_slab, zl_slab, zr_slab: Slab meshes on each of
            the six sides of the meshes, represented as tuples of 3D arrays for the x,
            y, and z coordinates.
        _nx_, _ny_, _nz_: Effective number of cells in the x, y, and z dimensions,
            including slabs if applicable.
    """

    nx: int = 1
    ny: int = 1
    nz: int = 1
    xlim: Tuple[float, float] = (0, 1)
    ylim: Tuple[float, float] = (0, 1)
    zlim: Tuple[float, float] = (0, 1)
    ignore_x: bool = False
    ignore_y: bool = False
    ignore_z: bool = False
    slab_depth: int = 1

    def __post_init__(self):
        # validate mesh dimensions
        if any((not isinstance(n, int) or n < 1) for n in (self.nx, self.ny, self.nz)):
            raise ValueError("Mesh dimensions (nx, ny, nz) must be positive integers.")
        # validate limits
        if any(
            (not isinstance(lim, tuple) or len(lim) != 2 or lim[0] >= lim[1])
            for lim in (self.xlim, self.ylim, self.zlim)
        ):
            raise ValueError(
                "Limits must be tuples of two values (min, max) with min < max."
            )
        # validate slab depths
        if self.slab_depth < 0 or not isinstance(self.slab_depth, int):
            raise ValueError("Slab depth must be non-negative integer.")

        # assign mesh properties
        self.shape = (self.nx, self.ny, self.nz)
        self.size = self.nx * self.ny * self.nz
        self.hx = (self.xlim[1] - self.xlim[0]) / self.nx
        self.hy = (self.ylim[1] - self.ylim[0]) / self.ny
        self.hz = (self.zlim[1] - self.zlim[0]) / self.nz
        self.h = {"x": self.hx, "y": self.hy, "z": self.hz}
        self.x_active = not self.ignore_x
        self.y_active = not self.ignore_y
        self.z_active = not self.ignore_z
        self.pad_x = self.slab_depth if self.x_active else 0
        self.pad_y = self.slab_depth if self.y_active else 0
        self.pad_z = self.slab_depth if self.z_active else 0
        self._nx_ = self.nx + 2 * self.pad_x
        self._ny_ = self.ny + 2 * self.pad_y
        self._nz_ = self.nz + 2 * self.pad_z

        self._set_interfaces_and_centers()
        self._init_mesh()
        self._init_slabs()

        self.gauss_legendre_quadrature_cache: dict[
            Tuple[int, Optional[str], Optional[str]],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def _set_interfaces_and_centers(self):
        self.x_interfaces = np.linspace(self.xlim[0], self.xlim[1], self.nx + 1)
        self.y_interfaces = np.linspace(self.ylim[0], self.ylim[1], self.ny + 1)
        self.z_interfaces = np.linspace(self.zlim[0], self.zlim[1], self.nz + 1)
        self.x_centers = 0.5 * (self.x_interfaces[:-1] + self.x_interfaces[1:])
        self.y_centers = 0.5 * (self.y_interfaces[:-1] + self.y_interfaces[1:])
        self.z_centers = 0.5 * (self.z_interfaces[:-1] + self.z_interfaces[1:])

    def _init_mesh(self):
        self.X, self.Y, self.Z = uniform_3D_mesh(
            self.nx, self.ny, self.nz, self.xlim, self.ylim, self.zlim
        )
        self.coords = (self.X, self.Y, self.Z)

    def _init_slabs(self):
        self.xl_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.xr_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.yl_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.yr_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.zl_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.zr_slab: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        slab_depth = {
            "x": self.slab_depth if self.x_active else 0,
            "y": self.slab_depth if self.y_active else 0,
            "z": self.slab_depth if self.z_active else 0,
        }
        lim1 = {"x": self.xlim[0], "y": self.ylim[0], "z": self.zlim[0]}
        lim2 = {"x": self.xlim[1], "y": self.ylim[1], "z": self.zlim[1]}
        h = {"x": self.hx, "y": self.hy, "z": self.hz}
        n = {"x": self.nx, "y": self.ny, "z": self.nz}

        for dim, pos in product("xyz", "lr"):
            if getattr(self, f"ignore_{dim}"):
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
            setattr(self, f"{dim}{pos}_slab", (X, Y, Z))

    def get_slab(self, dim: str, pos: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = getattr(self, f"{dim}{pos}_slab", None)
        if out is None:
            raise ValueError(f"Slab {dim}{pos} is not defined in this mesh.")
        return out

    def get_gauss_legendre_quadrature(
        self, p: int, slab: Optional[str] = None, face: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Gauss-Legendre quadrature points and weights for the mesh.

        Args:
            p: Polynomial degree of the quadrature rule. This determines the number of
                quadrature points in each dimension.
            slab: Slab to use for the quadrature. May be one of "xl", "xr", "yl", "yr",
                "zl", "zr". If None, the core mesh will be used.
            face: Face over which to integrate. May be one of "xl", "xr", "yl", "yr",
                "zl", "zr". If None, the quadrature will span the interior of each
                cell.

        Returns:
            Xp, Yp, Zp: Arrays of the quadrature points in the x, y, and z dimensions,
                respectively. Each has shape (nx, ny, nz, n_quadrature_points), where
                `n_quadrature_points` depends on the polynomial degree `p`.
            w: Array of quadrature weights with shape (1, 1, 1, n_quadrature_points).
        """
        key = (p, slab, face)
        if key in self.gauss_legendre_quadrature_cache:
            return self.gauss_legendre_quadrature_cache[key]

        if slab is None:
            X, Y, Z = self.X, self.Y, self.Z
        else:
            slab_dim, slab_pos = slab[0], slab[1]
            X, Y, Z = self.get_slab(slab_dim, slab_pos)

        px = p if self.x_active else 0
        py = p if self.y_active else 0
        pz = p if self.z_active else 0
        h = (self.hx, self.hy, self.hz)
        if face is None:
            Xp, Yp, Zp, w = gauss_legendre_mesh(X, Y, Z, h, (px, py, pz))
        else:
            dim, pos = face[0], face[1]
            match dim:
                case "x":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(X, Y, Z, h, (0, py, pz))
                    Xp = Xp + (-0.5 * self.hx if pos == "l" else 0.5 * self.hx)
                case "y":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(X, Y, Z, h, (px, 0, pz))
                    Yp = Yp + (-0.5 * self.hy if pos == "l" else 0.5 * self.hy)
                case "z":
                    Xp, Yp, Zp, w = gauss_legendre_mesh(X, Y, Z, h, (px, py, 0))
                    Zp = Zp + (-0.5 * self.hz if pos == "l" else 0.5 * self.hz)

        val = Xp, Yp, Zp, w
        self.gauss_legendre_quadrature_cache[key] = val
        return val

    def __getstate__(self):
        state = self.__dict__.copy()
        state["gauss_legendre_quadrature_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
