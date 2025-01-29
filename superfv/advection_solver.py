from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

from .boundary_conditions import DirichletBC
from .finite_volume_solver import FiniteVolumeSolver
from .riemann_solvers import advection_upwind
from .tools.array_management import ArrayLike, ArraySlicer
from .tools.timer import method_timer


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
    """

    def __init__(
        self,
        ic: Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ] = None,
        bcx: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcy: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcz: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        x_dirichlet: Optional[DirichletBC] = None,
        y_dirichlet: Optional[DirichletBC] = None,
        z_dirichlet: Optional[DirichletBC] = None,
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        zlim: Tuple[float, float] = (0, 1),
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        p: int = 0,
        interpolation_scheme: Literal["gauss-legendre", "transverse"] = "transverse",
        riemann_solver: str = "advection_upwind",
        CFL: float = 0.8,
        cupy: bool = False,
    ):
        """
        Initialize the advection solver.

        Args:
            ic (Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike]):
                Initial condition function. The function must accept the following
                arguments:
                - array_slicer (ArraySlicer): ArraySlicer object.
                - x (ArrayLike): x-coordinates. Has shape (nx, ny, nz).
                - y (ArrayLike): y-coordinates. Has shape (nx, ny, nz).
                - z (ArrayLike): z-coordinates. Has shape (nx, ny, nz).
                The function must return an array with shape (nvars, nx, ny, nz).
            ic_passives (Optional[Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]]):
                Initial condition functions for passive variables. The dictionary keys
                are the names of the passive variables, and the values are the
                corresponding initial condition functions. Each function must accept
                the following arguments:
                - x (ArrayLike): x-coordinates. Has shape (nx, ny, nz).
                - y (ArrayLike): y-coordinates. Has shape (nx, ny, nz).
                - z (ArrayLike): z-coordinates. Has shape (nx, ny, nz).
                The function must return an array with shape (nx, ny, nz).
            bcx (Union[str, Tuple[str, str]]): Boundary conditions in the x-direction.
            bcy (Union[str, Tuple[str, str]]): Boundary conditions in the y-direction.
            bcz (Union[str, Tuple[str, str]]): Boundary conditions in the z-direction.
            x_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                x-direction.
            y_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                y-direction.
            z_dirichlet (Optional[DirichletBC]): Dirichlet boundary conditions in the
                z-direction.
            xlim (Tuple[float, float]): x-limits of the domain.
            ylim (Tuple[float, float]): y-limits of the domain.
            zlim (Tuple[float, float]): z-limits of the domain.
            nx (int): Number of cells in the x-direction.
            ny (int): Number of cells in the y-direction.
            nz (int): Number of cells in the z-direction.
            p (int): Maximum polynomial degree of the spatial discretization.
            interpolation_scheme (str): Interpolation scheme to use for the
                interpolation of face nodes. Possible values:
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
            riemann_solver (str): Name of the Riemann solver function. Must be
                implemented in the derived class.
            CFL (float): CFL number.
            cupy (bool): Whether to use CuPy for array operations.
            kwarg_attributes: kwargs to be stored as attributes.
        """
        super().__init__(
            ic=ic,
            ic_passives=ic_passives,
            bcx=bcx,
            bcy=bcy,
            bcz=bcz,
            x_dirichlet=x_dirichlet,
            y_dirichlet=y_dirichlet,
            z_dirichlet=z_dirichlet,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            nx=nx,
            ny=ny,
            nz=nz,
            p=p,
            interpolation_scheme=interpolation_scheme,
            riemann_solver=riemann_solver,
            CFL=CFL,
            cupy=cupy,
        )

    def define_vars(self) -> ArraySlicer:
        """
        Returns an ArraySlicer object with the following variables:
            - rho: Density.
            - vx: x-component of the velocity.
            - vy: y-component of the velocity.
            - vz: z-component of the velocity.
        """
        return ArraySlicer({"rho": 0, "vx": 1, "vy": 2, "vz": 3}, ndim=4)

    @partial(method_timer, cat="AdvectionSolver.advection_upwind")
    def advection_upwind(
        self, yl: ArrayLike, yr: ArrayLike, dim: Literal["x", "y", "z"]
    ) -> ArrayLike:
        """
        Upwinding Riemann solver for the advection equation.

        Args:
            yl (ArrayLike): Left state. Has shape (nvars, nx, ny, nz, ...).
            yr (ArrayLike): Right state. Has shape (nvars, nx, ny, nz, ...).
            dim (Literal["x", "y", "z"]): Dimension.

        Returns:
            ArrayLike: Flux. Has shape (nvars, nx, ny, nz, ...).
        """
        return advection_upwind(self.array_slicer, yl, yr, dim)

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
