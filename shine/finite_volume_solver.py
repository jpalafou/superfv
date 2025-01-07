from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np

from .boundary_conditions import BoundaryConditions
from .explicit_ODE_solver import ExplicitODESolver
from .tools.array_management import ArrayLike, ArrayManager, ArraySlicer


class FiniteVolumeSolver(ExplicitODESolver):
    """
    Solve a nonlinear conservation law using the finite volume method in up to three
    dimensions.
    """

    @abstractmethod
    def conserved_vars(self) -> Tuple[str, ...]:
        """
        Define the names of the conserved variables.

        Returns:
            A tuple of strings, each representing a conserved variable.
        """
        pass

    @abstractmethod
    def get_dt_and_fluxes(
        self, t: float, y: ArrayLike
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Compute the time-step size and the fluxes.

        Args:
            t (float): Time value.
            y (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
            fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]]): Tuple of fluxes. The
                fluxes have the following shapes:
                - F: (n_vars, nx + 1, ny, nz)
                - G: (n_vars, nx, ny + 1, nz)
                - H: (n_vars, nx, ny, nz + 1)
        """
        pass

    def __init__(
        self,
        ic: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        bc: BoundaryConditions,
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        zlim: Tuple[float, float] = (0, 1),
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        px: int = 0,
        py: int = 0,
        pz: int = 0,
        CFL: float = 0.8,
        cupy: bool = False,
    ):
        self._init_array_slicer()
        self._init_array_manager(cupy)
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, px, py, pz, CFL)
        self._init_ic(ic)

    def _init_array_slicer(self):
        self.vars = self.conserved_vars()
        self.array_slicer = ArraySlicer({var: i for i, var in enumerate(self.vars)})

    def _init_array_manager(self, cupy: bool):
        self.arrays = ArrayManager()
        self.interpolation_cache = ArrayManager()
        self.integration_cache = ArrayManager()
        if cupy:
            self.arrays.enable_cupy()
            self.interpolation_cache.enable_cupy()
            self.integration_cache.enable_cupy()

    def _init_mesh(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        nx: int,
        ny: int,
        nz: int,
        px: int,
        py: int,
        pz: int,
        CFL: float,
    ):
        if any([xlim[1] < xlim[0], ylim[1] < ylim[0], zlim[1] < zlim[0]]):
            raise ValueError("The upper limit must be greater than the lower limit.")
        if any([nx < 1, ny < 1, nz < 1]):
            raise ValueError("The number of cells must be at least 1.")
        if any([px < 0, py < 0, pz < 0]):
            raise ValueError("The polynomial degree must be non-negative.")
        if CFL <= 0:
            raise ValueError("The CFL number must be positive.")
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.hx = (xlim[1] - xlim[0]) / nx
        self.hy = (ylim[1] - ylim[0]) / ny
        self.hz = (zlim[1] - zlim[0]) / nz
        self.px = px
        self.py = py
        self.pz = pz
        self.using_xdim = nx > 1
        self.using_ydim = ny > 1
        self.using_zdim = nz > 1
        self.n = (nx, ny, nz)
        self.h = (self.hx, self.hy, self.hz)
        self.p = (px, py, pz)
        self.using_dim = (nx > 1, ny > 1, nz > 1)
        self.CFL = CFL

        def _get_uniform_3D_mesh(xlim, ylim, zlim, nx, ny, nz):
            x_interface = np.linspace(xlim[0], xlim[1], nx + 1)
            y_interface = np.linspace(ylim[0], ylim[1], ny + 1)
            z_interface = np.linspace(zlim[0], zlim[1], nz + 1)
            x_center = 0.5 * (x_interface[1:] + x_interface[:-1])
            y_center = 0.5 * (y_interface[1:] + y_interface[:-1])
            z_center = 0.5 * (z_interface[1:] + z_interface[:-1])
            return np.meshgrid(x_center, y_center, z_center, indexing="ij")

        self.X, self.Y, self.Z = _get_uniform_3D_mesh(xlim, ylim, zlim, nx, ny, nz)

    def _init_ic(self, ic: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]):
        self.arrays.add("y", ic(self.X, self.Y, self.Z))

    def _init_bc(self, bc: BoundaryConditions):
        self.bc = bc

    def f(self, t: float, y: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Compute the right-hand side of the ODE.

        Args:
            t (float): Time value.
            y (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
            dydt (ArrayLike): Right-hand side of the ODE. Has shape
                (n_vars, nx, ny, nz).
        """
        dt, fluxes = self.get_dt_and_fluxes(t, y)
        return dt, self.RHS(y, *fluxes)

    def RHS(self, y: ArrayLike, F: ArrayLike, G: ArrayLike, H: ArrayLike) -> ArrayLike:
        """
        Compute the right-hand side of the conservation law.

        Args:
            y (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).
            F (ArrayLike): Flux in the x-direction. Has shape (n_vars, nx + 1, ny, nz).
            G (ArrayLike): Flux in the y-direction. Has shape (n_vars, nx, ny + 1, nz).
            H (ArrayLike): Flux in the z-direction. Has shape (n_vars, nx, ny, nz + 1).

        Returns:
            dydt (ArrayLike): Right-hand side of the conservation law. Has shape
                (n_vars, nx, ny, nz).
        """
        dydt = np.zeros_like(y)
        if self.using_xdim:
            dydt += -(1 / self.hx) * (F[:, 1:, :, :] - F[:, :-1, :, :])
        if self.using_ydim:
            dydt += -(1 / self.hy) * (G[:, :, 1:, :] - G[:, :, :-1, :])
        if self.using_zdim:
            dydt += -(1 / self.hz) * (H[:, :, :, 1:] - H[:, :, :, :-1])
        return dydt

    def run(self, *args, **kwargs):
        """
        Solve the conservation law using a Runge-Kutta method whose order matches the
        chosen polynomial degree for the spatial discretization, up to RK4.

        Args:
            *args: Arguments to pass to the Runge-Kutta method.
            **kwargs: Keyword arguments to pass to the Runge-Kutta method.
        """
        q = min(max(self.p), 3)
        match q:
            case 0:
                self.euler(*args, **kwargs)
            case 1:
                self.ssprk2(*args, **kwargs)
            case 2:
                self.ssprk3(*args, **kwargs)
            case 3:
                self.rk4(*args, **kwargs)
            case _:
                raise ValueError(f"Runge-Kutta method not implemented for {q=}")
