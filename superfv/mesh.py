from dataclasses import dataclass
from itertools import product
from typing import Tuple

import numpy as np


def uniform_3D_mesh(
    nx: int,
    ny: int,
    nz: int,
    xlim: Tuple[int, int],
    ylim: Tuple[int, int],
    zlim: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_interface = np.linspace(xlim[0], xlim[1], nx + 1)
    y_interface = np.linspace(ylim[0], ylim[1], ny + 1)
    z_interface = np.linspace(zlim[0], zlim[1], nz + 1)
    x_center = 0.5 * (x_interface[1:] + x_interface[:-1])
    y_center = 0.5 * (y_interface[1:] + y_interface[:-1])
    z_center = 0.5 * (z_interface[1:] + z_interface[:-1])
    return np.meshgrid(x_center, y_center, z_center, indexing="ij")


@dataclass
class UniformFVMesh:
    """
    A class to represent a uniform finite volume mesh in 3D space.

    Args:
        nx, ny, nz (int): Number of cells in the x, y, and z dimensions.
        xlim, ylim, zlim (tuple): Limits of the mesh in the x, y, and z dimensions.
        x_slab_depth, y_slab_depth, z_slab_depth (int): Depth of the slabs on each side
            of the mesh in the x, y, and z dimensions. In total, the mesh will have
            (nx + 2 * x_slab_depth, ny + 2 * y_slab_depth, nz + 2 * z_slab_depth)
            cells in the x, y, and z dimensions.

    Attributes:
        shape (tuple): Shape of the mesh as (nx, ny, nz).
        hx, hy, hz (float): Cell sizes in the x, y, and z dimensions.
        h (tuple): Tuple of cell sizes (hx, hy, hz).
        x_interfaces, y_interfaces, z_interfaces (ndarray): Interfaces in the x, y,
            and z dimensions as 1D arrays.
        x_centers, y_centers, z_centers (ndarray): Centers of the cells in the x,
            y, and z dimensions as 1D arrays.
        X, Y, Z (ndarray): Mesh grid arrays for the x, y, and z dimensions as 3D
            arrays.
        xl_slab, xr_slab, yl_slab, yr_slab, zl_slab, zr_slab (tuple): Slab meshes on
            each of the six sides of the meshes, represented as tuples of 3D arrays for
            the x, y, and z coordinates.
    """

    nx: int = 1
    ny: int = 1
    nz: int = 1
    xlim: Tuple[float, float] = (0, 1)
    ylim: Tuple[float, float] = (0, 1)
    zlim: Tuple[float, float] = (0, 1)
    x_slab_depth: int = 1
    y_slab_depth: int = 1
    z_slab_depth: int = 1

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
        if any(
            (not isinstance(depth, int) or depth < 0)
            for depth in (self.x_slab_depth, self.y_slab_depth, self.z_slab_depth)
        ):
            raise ValueError("Slab depths must be non-negative integers.")

        # assign mesh properties
        self.shape = (self.nx, self.ny, self.nz)
        self.hx = (self.xlim[1] - self.xlim[0]) / self.nx
        self.hy = (self.ylim[1] - self.ylim[0]) / self.ny
        self.hz = (self.zlim[1] - self.zlim[0]) / self.nz
        self.h = (self.hx, self.hy, self.hz)

        self._set_interfaces_and_centers()
        self._init_mesh()
        self._init_slabs()

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

    def _init_slabs(self):
        slab_depth = {
            "x": self.x_slab_depth,
            "y": self.y_slab_depth,
            "z": self.z_slab_depth,
        }
        lim1 = {"x": self.xlim[0], "y": self.ylim[0], "z": self.zlim[0]}
        lim2 = {"x": self.xlim[1], "y": self.ylim[1], "z": self.zlim[1]}
        h = {"x": self.hx, "y": self.hy, "z": self.hz}
        n = {"x": self.nx, "y": self.ny, "z": self.nz}

        for dim, pos in product("xyz", "lr"):
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
