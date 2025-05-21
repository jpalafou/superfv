from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import wtflux

from .boundary_conditions import DirichletBC
from .finite_volume_solver import FiniteVolumeSolver
from .hydro import conservatives_from_primitives, primitives_from_conservatives
from .riemann_solvers import call_riemann_solver
from .tools.array_management import ArrayLike, ArraySlicer
from .tools.timer import method_timer


class EulerSolver(FiniteVolumeSolver):
    """
    Solve the system of equations in up to three dimensions.
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
        CFL: float = 0.8,
        interpolation_scheme: Literal["gauss-legendre", "transverse"] = "transverse",
        flux_recipe: Literal[1, 2, 3] = 1,
        lazy_primitives: bool = False,
        riemann_solver: Literal["llf", "hllc"] = "llf",
        MUSCL: bool = False,
        ZS: bool = False,
        adaptive_timestepping: bool = True,
        max_adaptive_timesteps: Optional[int] = None,
        MOOD: bool = False,
        max_MOOD_iters: Optional[int] = None,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = ("rho",),
        NAD: Optional[float] = None,
        PAD: Optional[Dict[str, Tuple[float, float]]] = None,
        PAD_tol: float = 1e-15,
        SED: bool = False,
        cupy: bool = False,
        gamma: float = 1.4,
    ):
        """
        Initialize the finite volume solver.

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
            CFL (float): CFL number.
            interpolation_scheme (str): Interpolation scheme to use for the
                interpolation of face nodes. Possible values:
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
            flux_recipe (Literal[1,2,3]): Recipe for interpolating flux nodes.
                - 1: Interpolate conservative nodes from conservative cell averages.
                    Apply slope limiting to the conservative nodes. Transform to
                    primitive variables.
                - 2: Interpolate conservative nodes from conservative cell averages.
                    Transform to primitive variables. Apply slope limiting to the
                    primitive nodes.
                - 3: Interpolate primitive cell averages from conservative cell
                    averages, either by interpolating to cell-centered values
                    intermittently or transforming directly with `lazy_primtives=True`.
                    Interpolate primitive nodes from primitive cell averages.
                    Apply slope limiting to the primitive nodes.
            lazy_primitives (bool): Whether to transform conservative cell averages
                directly to primitive cell averages. Note that this is a second order
                operation. If
                - `flux_recipe=1`: This argument is ignored.
                - `flux_recipe=2`: The lazy primitives become the fallback option.
                - `flux_recipe=3`: The lazy primitives are used to interpolate the
                    primitive flux nodes.
            riemann_solver (str): Name of the Riemann solver function. Must be
                implemented in the derived class.
            MUSCL (bool): Whether to use the MUSCL scheme for a priori slope limiting.
            ZS (bool): Whether to use Zhang and Shu's maximum-principle-satisfying a
                priori slope limiter.
            adaptive_timestepping (bool): Option for `ZS=True` to half the time-step
                size if a maximum principle violation is detected. If True, MOOD is
                overwritten to only modify the time-step size and not the fluxes.
                Ignored if `ZS=False`.
            max_adaptive_timesteps (Optional[int]): Maximum number of adaptive time
                steps. If `ZS=True` and `adaptive_timestepping=True`, the default value
                is 10. Otherwise, this argument is ignored.
            MOOD (bool): Whether to use MOOD for a posteriori flux revision. Ignored if
                `ZS=True` and `adaptive_timestepping=True`.
            max_MOOD_iters (Optional[int]): Maximum number of MOOD iterations. Ignored
                if `ZS=True` and `adaptive_timestepping=True`. Otherwise, the default
                value is 1.
            limiting_vars (Union[Literal["all", "actives"], Tuple[str, ...]]):
                Specifies which variables are subject to slope limiting.
                - "all": All variables are subject to slope limiting.
                - "actives": Only active variables are subject to slope limiting.
                - Tuple[str, ...]: A tuple of variable names that are subject to slope
                    limiting. Must be defined in `self.define_vars()`.
                For the Zhang-Shu limiter, all variables are always limited, but
                `limiting_vars` determines which variables are checked for PAD when
                using adaptive timestepping.
            NAD (Optional[float]): The NAD tolerance. If None, NAD is not checked.
            PAD (Optional[Dict[str, Tuple[float, float]]]): Dict of `limiting_vars` and
                their corresponding PAD tolerances. If a limiting variable is not in
                the dict, it is given a PAD tolerance of (-np.inf, np.inf).
            PAD_tol (float): Tolerance for the PAD check as an absolute value from the
                minimum and maximum values of the variable.
            SED (bool): Whether to use smooth extrema detection for slope limiting.
            cupy (bool): Whether to use CuPy for array operations.
            gamma (float): Adiabatic index.
        """
        # init hydro
        self.gamma = gamma
        if cupy:
            wtflux.backend.set_backend("cupy")
        self.hydro = wtflux.hydro
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
            CFL=CFL,
            interpolation_scheme=interpolation_scheme,
            riemann_solver=riemann_solver,
            flux_recipe=flux_recipe,
            lazy_primitives=lazy_primitives,
            MUSCL=MUSCL,
            ZS=ZS,
            adaptive_timestepping=adaptive_timestepping,
            max_adaptive_timesteps=max_adaptive_timesteps,
            MOOD=MOOD,
            max_MOOD_iters=max_MOOD_iters,
            limiting_vars=limiting_vars,
            NAD=NAD,
            PAD=PAD,
            PAD_tol=PAD_tol,
            SED=SED,
            cupy=cupy,
        )

    def _init_snapshots(self):
        super()._init_snapshots()
        self.minisnapshots["min_rho"] = []
        self.minisnapshots["max_rho"] = []
        self.minisnapshots["min_E"] = []
        self.minisnapshots["max_E"] = []

    def define_vars(self) -> ArraySlicer:
        """
        Returns an ArraySlicer object with the following variables:
            - rho: Density.
            - vx: x-component of the velocity.
            - vy: y-component of the velocity.
            - vz: z-component of the velocity.
        """
        var_map = {
            "rho": 0,
            "vx": 1,
            "vy": 2,
            "vz": 3,
            "P": 4,
            "E": 4,
            "mx": 1,
            "my": 2,
            "mz": 3,
        }

        active_dims = [dim for dim in "xyz" if self.using[dim]]
        passive_dims = [dim for dim in "xyz" if not self.using[dim]]

        groups = {
            "v": tuple(f"v{dim}" for dim in active_dims),
            "m": tuple(f"m{dim}" for dim in active_dims),
        } | (
            {
                "passives": tuple(f"v{dim}" for dim in passive_dims)
                + tuple(f"m{dim}" for dim in passive_dims)
            }
            if passive_dims
            else {}
        )

        return ArraySlicer(var_map, groups=groups, ndim=4)

    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Convert primitive variables to conservative variables.

        Args:
            w (ArrayLike): Primitive variables.

        Returns:
            ArrayLike: Conservative variables.
        """
        return conservatives_from_primitives(
            self.hydro, self.array_slicer, w, self.gamma
        )

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u (ArrayLike): Conservative variables.

        Returns:
            ArrayLike: Primitive variables.
        """
        return primitives_from_conservatives(
            self.hydro, self.array_slicer, u, self.gamma
        )

    @partial(method_timer, cat="EulerSolver.llf")
    def llf(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        primitives: bool = True,
    ) -> ArrayLike:
        """
        LLF implementation. See FiniteVolumeSolver.dummy_riemann_solver for details.
        """
        return call_riemann_solver(self, wl, wr, dim, self.gamma, primitives, "llf")

    @partial(method_timer, cat="EulerSolver.hllc")
    def hllc(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        primitives: bool = True,
    ) -> ArrayLike:
        """
        HLLC implementation. See FiniteVolumeSolver.dummy_riemann_solver for details.
        """
        return call_riemann_solver(self, wl, wr, dim, self.gamma, primitives, "hllc")

    @partial(method_timer, cat="EulerSolver.compute_dt")
    def compute_dt(self, w: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            w (ArrayLike): Primitive solution values. Has shape (nvars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        h = min(self.h.values())
        c = self.hydro.sound_speed(rho=w[_slc("rho")], P=w[_slc("P")], gamma=self.gamma)
        out = (
            self.CFL * h / np.max(np.sum(np.abs(w[_slc("v")]), axis=0) + self.ndim * c)
        )
        return out.item()

    @partial(method_timer, cat="EulerSolver.minisnapshot")
    def minisnapshot(self):
        super().minisnapshot()
        _slc = self.array_slicer
        self.minisnapshots["min_rho"].append(self.arrays["u"][_slc("rho")].min().item())
        self.minisnapshots["max_rho"].append(self.arrays["u"][_slc("rho")].max().item())
        self.minisnapshots["min_E"].append(self.arrays["u"][_slc("E")].min().item())
        self.minisnapshots["max_E"].append(self.arrays["u"][_slc("E")].max().item())
