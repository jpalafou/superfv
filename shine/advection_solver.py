from functools import partial
from typing import Tuple, Union

import numpy as np

from .finite_volume_solver import FiniteVolumeSolver
from .initial_conditions import InitialCondition
from .tools.array_management import ArrayLike, crop_to_center
from .tools.timer import method_timer


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
    """

    def conserved_vars(self) -> Tuple[str, ...]:
        return ("u", "vx", "vy", "vz")

    def __init__(
        self,
        ic: InitialCondition,
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
        """
        Initialize the advection solver.

        Args:
            ic (InitialCondition): Initial condition object.
            xbc (Union[str, Tuple[str, str]]): Boundary conditions in the x-direction.
            ybc (Union[str, Tuple[str, str]]): Boundary conditions in the y-direction.
            zbc (Union[str, Tuple[str, str]]): Boundary conditions in the z-direction.
            xlim (Tuple[float, float]): x-limits of the domain.
            ylim (Tuple[float, float]): y-limits of the domain.
            zlim (Tuple[float, float]): z-limits of the domain.
            nx (int): Number of cells in the x-direction.
            ny (int): Number of cells in the y-direction.
            nz (int): Number of cells in the z-direction.
            p (int): Polynomial degree of the spatial discretization.
            CFL (float): CFL number.
            cupy (bool): Whether to use CuPy for array operations.
        """
        super().__init__(ic, xbc, ybc, zbc, xlim, ylim, zlim, nx, ny, nz, p, CFL, cupy)

    @partial(method_timer, cat="AdvectionSolver.compute_dt_and_fluxes")
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
            F = crop_to_center(
                self.compute_flux_integral(f, dim="x", p=p, mode="transverse"),
                self.F_shape,
            )
        else:
            F = np.array([])

        # y-direction fluxes
        if self.using_ydim:
            yl, yr = y_face_nodes
            g = self.upwinding_riemann_solver(
                (yr[:, :, :-1, ...], yl[:, :, 1:, ...]), dim="y"
            )
            G = crop_to_center(
                self.compute_flux_integral(g, dim="y", p=p, mode="transverse"),
                self.G_shape,
            )
        else:
            G = np.array([])

        # z-direction fluxes
        if self.using_zdim:
            zl, zr = z_face_nodes
            h = self.upwinding_riemann_solver(
                (zr[:, :, :, :-1], zl[:, :, :, 1:]), dim="z"
            )
            H = crop_to_center(
                self.compute_flux_integral(h, dim="z", p=p, mode="transverse"),
                self.H_shape,
            )
        else:
            H = np.array([])

        return dt, (F, G, H)

    @partial(method_timer, cat="AdvectionSolver.get_dt")
    def get_dt(self, u: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            u (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        h = min(self.h)
        vx = np.max(np.abs(u[_slc("vx")]))
        vy = np.max(np.abs(u[_slc("vy")]))
        vz = np.max(np.abs(u[_slc("vz")]))
        return (self.CFL * h / (vx + vy + vz)).item()

    @partial(method_timer, cat="AdvectionSolver.upwinding_riemann_solver")
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
