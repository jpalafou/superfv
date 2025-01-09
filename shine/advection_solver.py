from typing import Callable, Tuple, Union

import numpy as np

from .finite_volume_solver import FiniteVolumeSolver
from .tools.array_management import ArrayLike, crop_to_center


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
    """

    def conserved_vars(self) -> Tuple[str, ...]:
        return ("u", "vx", "vy", "vz")

    def __init__(
        self,
        ic: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        xbc: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        ybc: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        zbc: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        zlim: Tuple[float, float] = (0, 1),
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        p: int = 0,
        CFL: float = 0.8,
        cupy: bool = False,
    ):
        super().__init__(ic, xbc, ybc, zbc, xlim, ylim, zlim, nx, ny, nz, p, CFL, cupy)

    def compute_dt_and_fluxes(
        self, t: float, u: ArrayLike, p: int
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        # compute dt
        dt = self.get_dt(u)

        # get number of required ghost cells
        n_ghost_cells = max(-(-p // 2) + 1, 2 * -(-p // 2))

        # apply boundary conditions
        u_padded = self.bc(
            u,
            pad_width=(
                int(self.using_xdim) * n_ghost_cells,
                int(self.using_ydim) * n_ghost_cells,
                int(self.using_zdim) * n_ghost_cells,
            ),
        )

        # interpolate face nodes
        x_face_nodes, y_face_nodes, z_face_nodes = self.interpolate_face_nodes(
            u_padded, p=p, mode="face-centers"
        )

        # x-direction fluxes
        if self.using_xdim:
            xl, xr = x_face_nodes
            f = self.upwinding_riemann_solver(
                (xr[:, :-1, ...], xl[:, 1:, ...]), dim="x"
            )
            F = self.interpolate(
                f, p=p, stencil_type="uniform-quadrature", sweep_order="yz"
            )
            F = crop_to_center(F, self.F_shape)
            self.interpolation_cache.clear()
        else:
            F = np.array([])

        # y-direction fluxes
        if self.using_ydim:
            yl, yr = y_face_nodes
            g = self.upwinding_riemann_solver(
                (yr[:, :, :-1, ...], yl[:, :, 1:, ...]), dim="y"
            )
            G = self.interpolate(
                g, p=p, stencil_type="uniform-quadrature", sweep_order="xz"
            )
            G = crop_to_center(G, self.G_shape)
            self.interpolation_cache.clear()
        else:
            G = np.array([])

        # z-direction fluxes
        if self.using_zdim:
            zl, zr = z_face_nodes
            h = self.upwinding_riemann_solver(
                (zr[:, :, :, :-1], zl[:, :, :, 1:]), dim="z"
            )
            H = self.interpolate(
                h, p=p, stencil_type="uniform-quadrature", sweep_order="xy"
            )
            H = crop_to_center(H, self.H_shape)
            self.interpolation_cache.clear()
        else:
            H = np.array([])

        return dt, (F, G, H)

    def get_dt(self, u: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            u (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        _h = min(self.h)
        _vx = np.max(np.abs(u[_slc("vx")]))
        _vy = np.max(np.abs(u[_slc("vy")]))
        _vz = np.max(np.abs(u[_slc("vz")]))
        return (self.CFL * _h / (_vx + _vy + _vz)).item()

    def upwinding_riemann_solver(
        self, riemann_problem: Tuple[ArrayLike, ArrayLike], dim: str
    ) -> ArrayLike:
        """
        Solve the advection of a scalar in up to three dimensions.

        Args:
            riemann_problem (Tuple[ArrayLike, ArrayLike]): Tuple of left and right
                states. Each state has shape (n_vars, nx, ny, nz).
            dim (str): Dimension to solve in ("x", "y", or "z").

        Returns:
            out (ArrayLike): Point-wise solution to the Riemann problem. Has shape
                (n_vars, nx, ny, nz).
        """
        _slc = self.array_slicer
        yl, yr = riemann_problem
        vl, vr = yl[_slc("v" + dim)], yr[_slc("v" + dim)]
        v = np.where(np.abs(vl) > np.abs(vr), vl, vr)

        out = np.zeros_like(yl)
        out[_slc("u")] = v * np.where(
            v > 0, yl[_slc("u")], np.where(v < 0, yr[_slc("u")], 0)
        )
        return out
