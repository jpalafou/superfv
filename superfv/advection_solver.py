from functools import partial
from typing import Callable, Literal, Tuple

import numpy as np

from .finite_volume_solver import FiniteVolumeSolver
from .tools.array_management import ArrayLike, ArraySlicer
from .tools.timer import method_timer


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
    """

    def define_vars(self) -> ArraySlicer:
        """
        Returns an ArraySlicer object with the following variables:
            - rho: Density.
            - vx: x-component of the velocity.
            - vy: y-component of the velocity.
            - vz: z-component of the velocity.
        """
        return ArraySlicer({"rho": 0, "vx": 1, "vy": 2, "vz": 3}, ndim=4)

    def define_riemann_solver(
        self,
    ) -> Callable[[ArrayLike, ArrayLike, Literal["x", "y", "z"]], ArrayLike]:
        """
        Returns an upwinding Riemann solver for the advection equation.
        """

        def upwinding_riemann_solver(yl, yr, dim):
            _slc = self.array_slicer
            vl, vr = yl[_slc("v" + dim)], yr[_slc("v" + dim)]
            v = np.where(np.abs(vl) > np.abs(vr), vl, vr)

            out = np.zeros_like(yl)
            out[_slc("rho")] = v * np.where(
                v > 0, yl[_slc("rho")], np.where(v < 0, yr[_slc("rho")], 0)
            )
            if self.has_passives:
                out[_slc("passives")] = v * np.where(
                    v > 0,
                    yl[_slc("passives")],
                    np.where(v < 0, yr[_slc("passives")], 0),
                )
            return out

        return upwinding_riemann_solver

    @partial(method_timer, cat="AdvectionSolver.compute_dt_and_fluxes")
    def compute_dt_and_fluxes(
        self, t: float, u: ArrayLike, p: int
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Compute the time-step size and the fluxes for the advection equation.

        Args:
            t (float): Current time.
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).
            p (int): Polynomial degree.

        Returns:
            dt (float): Time-step size.
            fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): Fluxes in the x, y, and z
                directions.
        """
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
        xl, xr, yl, yr, zl, zr = self.interpolate_face_nodes(
            u_padded,
            p=p,
            mode={"transverse": "face-centers", "gauss-legendre": "gauss-legendre"}[
                self.interpolation_scheme
            ],
        )

        # initialize empty fluxes
        F, G, H = np.array([]), np.array([]), np.array([])

        # compute numerical fluxes in each direction
        if self.using_xdim:
            riemann_problem_x = xr[:, :-1, ...], xl[:, 1:, ...]
            F = self.compute_numerical_fluxes(
                *riemann_problem_x, p=p, dim="x", mode=self.interpolation_scheme
            )
        if self.using_ydim:
            riemann_problem_y = yr[:, :, :-1, ...], yl[:, :, 1:, ...]
            G = self.compute_numerical_fluxes(
                *riemann_problem_y, p=p, dim="y", mode=self.interpolation_scheme
            )
        if self.using_zdim:
            riemann_problem_z = zr[:, :, :, :-1, ...], zl[:, :, :, 1:, ...]
            H = self.compute_numerical_fluxes(
                *riemann_problem_z, p=p, dim="z", mode=self.interpolation_scheme
            )

        return dt, (F, G, H)

    @partial(method_timer, cat="AdvectionSolver.get_dt")
    def get_dt(self, u: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        h = min(self.h)
        vx = np.max(np.abs(u[_slc("vx")]))
        vy = np.max(np.abs(u[_slc("vy")]))
        vz = np.max(np.abs(u[_slc("vz")]))
        return (self.CFL * h / (vx + vy + vz)).item()
