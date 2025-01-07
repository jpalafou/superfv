from typing import Callable, Tuple

import numpy as np

from .boundary_conditions import BoundaryConditions
from .finite_volume_solver import FiniteVolumeSolver
from .tools.array_management import ArrayLike


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
    """

    def conserved_vars(self) -> Tuple[str, ...]:
        return ("u", "vx", "vy", "vz")

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
        super().__init__(ic, bc, xlim, ylim, zlim, nx, ny, nz, px, py, pz, CFL, cupy)

    def get_dt_and_fluxes(
        self, t: float, y: ArrayLike
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        F = np.zeros((4, self.nx + 1, self.ny, self.nz))
        G = np.zeros((4, self.nx, self.ny + 1, self.nz))
        H = np.zeros((4, self.nx, self.ny, self.nz + 1))
        dt = np.inf
        return dt, (F, G, H)

    def get_dt(self, y: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            y (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        _h = min(self.h)
        _vx = np.max(np.abs(y[_slc("vx")]))
        _vy = np.max(np.abs(y[_slc("vy")]))
        _vz = np.max(np.abs(y[_slc("vz")]))
        return self.CFL * _h / (_vx + _vy + _vz)
