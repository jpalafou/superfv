from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Union

import numpy as np

from .boundary_conditions import BoundaryConditions
from .explicit_ODE_solver import ExplicitODESolver
from .initial_conditions import InitialCondition
from .stencil import (
    conservative_interpolation_weights,
    stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.array_management import ArrayLike, ArrayManager, ArraySlicer
from .tools.timer import method_timer


class FiniteVolumeSolver(ExplicitODESolver, ABC):
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
    @partial(method_timer, cat="?.compute_dt_and_fluxes")
    def compute_dt_and_fluxes(
        self, t: float, u: ArrayLike, p: int
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Compute the time-step size and the fluxes.

        Args:
            t (float): Time value.
            u (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).
            p (int): Polynomial degree of the spatial discretization.

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
        Initialize the finite volume solver.

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
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, p, CFL)
        self._init_array_slicer()
        self._init_ic(ic, cupy)
        self._init_array_manager(cupy)
        self._init_bc(xbc, ybc, zbc)

    def _init_mesh(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        nx: int,
        ny: int,
        nz: int,
        p: int,
        CFL: float,
    ):
        if any([xlim[1] < xlim[0], ylim[1] < ylim[0], zlim[1] < zlim[0]]):
            raise ValueError("The upper limit must be greater than the lower limit.")
        if any([nx < 1, ny < 1, nz < 1]):
            raise ValueError("The number of cells must be at least 1.")
        if p < 0:
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
        self.using_xdim = nx > 1
        self.using_ydim = ny > 1
        self.using_zdim = nz > 1
        self.n = (nx, ny, nz)
        self.h = (self.hx, self.hy, self.hz)
        self.using_dim = (nx > 1, ny > 1, nz > 1)
        self.ndim = sum(self.using_dim)
        self.dims = "".join(dim for dim, using in zip("xyz", self.using_dim) if using)
        self.p = p
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

        self.F_shape = (len(self.conserved_vars()), nx + 1, ny, nz)
        self.G_shape = (len(self.conserved_vars()), nx, ny + 1, nz)
        self.H_shape = (len(self.conserved_vars()), nx, ny, nz + 1)

    def _init_array_slicer(self):
        self.vars = self.conserved_vars()
        self.array_slicer = ArraySlicer({var: i for i, var in enumerate(self.vars)}, 4)

    def _init_array_manager(self, cupy: bool):
        self.interpolation_cache = ArrayManager()
        if cupy:
            self.interpolation_cache.enable_cupy()

    def _init_ic(self, ic: InitialCondition, cupy):
        _ic = ic(self.array_slicer, self.dims)
        super().__init__(_ic(self.X, self.Y, self.Z), cupy=cupy)

    def _init_bc(
        self,
        xbc: Union[str, Tuple[str, str]],
        ybc: Union[str, Tuple[str, str]],
        zbc: Union[str, Tuple[str, str]],
    ):
        self.bc = BoundaryConditions(self.array_slicer, xbc, ybc, zbc)

    @partial(method_timer, cat="FiniteVolumeSolver.f")
    def f(self, t: float, u: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Compute the right-hand side of the ODE.

        Args:
            t (float): Time value.
            u (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
            dudt (ArrayLike): Right-hand side of the ODE. Has shape
                (n_vars, nx, ny, nz).
        """
        dt, fluxes = self.compute_dt_and_fluxes(t, u, p=self.p)
        return dt, self.RHS(u, *fluxes)

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate")
    def interpolate(
        self,
        u: ArrayLike,
        x: Union[int, float, str] = 0,
        y: Union[int, float, str] = 0,
        z: Union[int, float, str] = 0,
        p: int = 0,
        sweep_order: str = "xyz",
        stencil_type: str = "conservative-interpolation",
    ) -> ArrayLike:
        """
        Interpolates a node value from the finite-volume cell averages.

        Args:
            u (ArrayLike): Array to interpolate. Has shape (n_vars, nx, ny, nz).
            x (Union[int, float, str]): x-coordinate of the desired node. Must be
                between -1 (leftmost cell face) and nx (rightmost cell face).
                Alternatively, can be "l", "r", or "c" for left, right, or center.
            y (Union[int, float, str]): y-coordinate of the desired node. Must be
                between -1 (leftmost cell face) and nx (rightmost cell face).
                Alternatively, can be "l", "r", or "c" for left, right, or center.
            z (Union[int, float, str]): z-coordinate of the desired node. Must be
                between -1 (leftmost cell face) and nx (rightmost cell face).
                Alternatively, can be "l", "r", or "c" for left, right, or center.
            p (int): Polynomial degree of the interpolation.
            sweep_order (str): Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            stencil_type (str): Type of stencil weights to use for the interpolation.
                - "conservative-interpolation": Uses conservative interpolation weights.
                - "uniform-quadrature": Uses uniform quadrature weights.

        Returns:
            ArrayLike: Interpolated node values. Has shape
                (n_vars, <= nx, <= ny, <= nz).

        Notes:
            - This function utilizes a caching system (`self.interpolation_cache`) to
                store intermediate results (planes and lines) for efficiency. Use
                `self.interpolation_cache.clear()` to clear the cache.
        """
        coordinates = {"x": x, "y": y, "z": z}
        axes = {"x": 1, "y": 2, "z": 3}
        weight_functions = {
            "conservative-interpolation": lambda p, coord: conservative_interpolation_weights(
                p, coord
            ),
            "uniform-quadrature": lambda p, _: uniform_quadrature_weights(p),
        }
        key_functions = {
            "conservative-interpolation": lambda direction, coord: f"{direction}={coord}",
            "uniform-quadrature": lambda direction, _: direction,
        }
        if stencil_type not in weight_functions:
            raise ValueError(f"Unsupported stencil_type: {stencil_type}")

        # initialize the interpolation
        current_data = u
        key = None
        weight_function = weight_functions[stencil_type]
        key_function = key_functions[stencil_type]

        # loop over directions
        for direction in sweep_order:
            key = (
                key_function(direction, coordinates[direction])
                if key is None
                else (f"{key}, {key_function(direction, coordinates[direction])}")
            )

            # if the interpolated data is not in the cache, compute it
            if key not in self.interpolation_cache:
                if getattr(self, f"using_{direction}dim"):
                    self.interpolation_cache[key] = stencil_sweep(
                        current_data,
                        stencil_weights=weight_function(p, coordinates[direction]),
                        axis=axes[direction],
                    )
                else:
                    self.interpolation_cache[key] = current_data

            # update the current data
            current_data = self.interpolation_cache[key]

        return current_data

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_face_nodes")
    def interpolate_face_nodes(
        self, averages: ArrayLike, p: int, mode: str
    ) -> List[Tuple[ArrayLike, ArrayLike]]:
        """
        Interpolate face nodes from cell averages.

        Args:
            averages (ArrayLike): Cell averages. Has shape (n_vars, nx, ny, nz).
            p (int): Polynomial degree of the interpolation.
            mode (str): Mode of interpolation. Possible values:
                - "face-centers": Interpolate nodes at the cell face centers.

        Returns:
            List[Tuple[ArrayLike, ArrayLike]]: Interpolated face nodes.
                - Each face node is a tuple of two arrays: the left and right face nodes.
                - The left and right face nodes have shape
                    (n_vars, <= nx, <= ny, <= nz).
                - If a face node does not exist in a given direction, the corresponding
                    element in the list will be `None`.
        """
        self.interpolation_cache.clear()
        sweep_orders = {"x": "yzx", "y": "zxy", "z": "xyz"}
        face_nodes = []
        if mode == "face-centers":
            for dim in "xyz":
                face_nodes.append(
                    (
                        self.interpolate(
                            averages,
                            **{dim: "l"},
                            p=p,
                            sweep_order=sweep_orders[dim],
                            stencil_type="conservative-interpolation",
                        ),
                        self.interpolate(
                            averages,
                            **{dim: "r"},
                            p=p,
                            sweep_order=sweep_orders[dim],
                            stencil_type="conservative-interpolation",
                        ),
                    )
                    if getattr(self, f"using_{dim}dim")
                    else (np.array([]), np.array([]))
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return face_nodes

    @partial(method_timer, cat="FiniteVolumeSolver.compute_flux_integral")
    def compute_flux_integral(
        self, node_values: ArrayLike, dim: str, p: int, mode: str
    ) -> ArrayLike:
        """
        Compute the flux integral in a given direction.

        Args:
            node_values (ArrayLike): Node values. Has shape (n_vars, nx, ny, nz).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".
            p (int): Polynomial degree of the interpolation.
            mode (str): Mode of the flux. Possible values:
                - "transverse": Compute the transverse flux integral.

        Returns:
            ArrayLike: Flux integral. Has shape (n_vars, <= nx, <= ny, <= nz).
        """
        if mode == "transverse":
            flux_integrals = self.interpolate(
                node_values,
                p=p,
                stencil_type="uniform-quadrature",
                sweep_order={"x": "yz", "y": "xz", "z": "xy"}[dim],
            )
        self.interpolation_cache.clear()
        return flux_integrals

    def RHS(self, u: ArrayLike, F: ArrayLike, G: ArrayLike, H: ArrayLike) -> ArrayLike:
        """
        Compute the right-hand side of the conservation law.

        Args:
            u (ArrayLike): Solution value. Has shape (n_vars, nx, ny, nz).
            F (ArrayLike): Flux in the x-direction. Has shape (n_vars, nx + 1, ny, nz).
            G (ArrayLike): Flux in the y-direction. Has shape (n_vars, nx, ny + 1, nz).
            H (ArrayLike): Flux in the z-direction. Has shape (n_vars, nx, ny, nz + 1).

        Returns:
            dydt (ArrayLike): Right-hand side of the conservation law. Has shape
                (n_vars, nx, ny, nz).
        """
        dydt = np.zeros_like(u)
        if self.using_xdim:
            dydt += -(1 / self.hx) * (F[:, 1:, :, :] - F[:, :-1, :, :])
        if self.using_ydim:
            dydt += -(1 / self.hy) * (G[:, :, 1:, :] - G[:, :, :-1, :])
        if self.using_zdim:
            dydt += -(1 / self.hz) * (H[:, :, :, 1:] - H[:, :, :, :-1])
        return dydt

    @partial(method_timer, cat="!FiniteVolumeSolver.run")
    def run(self, *args, **kwargs):
        """
        Solve the conservation law using a Runge-Kutta method whose order matches the
        chosen polynomial degree for the spatial discretization, up to RK4.

        Args:
            *args: Arguments to pass to the Runge-Kutta method.
            **kwargs: Keyword arguments to pass to the Runge-Kutta method.
        """
        q = min(self.p, 3)
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
