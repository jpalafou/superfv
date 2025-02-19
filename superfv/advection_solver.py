from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np

from .boundary_conditions import DirichletBC
from .finite_volume_solver import FiniteVolumeSolver
from .riemann_solvers import advection_upwind
from .slope_limiting import compute_dmp
from .slope_limiting.zhang_and_shu import zhang_shu_advection
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
        CFL: float = 0.8,
        interpolation_scheme: Literal["gauss-legendre", "transverse"] = "transverse",
        riemann_solver: str = "advection_upwind",
        MUSCL: bool = False,
        ZS: bool = False,
        MOOD: bool = False,
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
            CFL (float): CFL number.
            interpolation_scheme (str): Interpolation scheme to use for the
                interpolation of face nodes. Possible values:
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
            riemann_solver (str): Name of the Riemann solver function. Must be
                implemented in the derived class.
            MUSCL (bool): Whether to use the MUSCL scheme for a priori slope limiting.
            ZS (bool): Whether to use Zhang and Shu's maximum-principle-satisfying slope
                limiter.
            MOOD (bool): Whether to use MOOD for a posteriori flux revision.
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
            CFL=CFL,
            interpolation_scheme=interpolation_scheme,
            riemann_solver=riemann_solver,
            MUSCL=MUSCL,
            ZS=ZS,
            MOOD=MOOD,
            cupy=cupy,
        )

    def _init_snapshots(self):
        self.minisnapshots["min_rho"] = []
        self.minisnapshots["max_rho"] = []

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
        Riemann solver implementation. See FiniteVolumeSolver.dummy_riemann_solver.
        """
        return advection_upwind(self.array_slicer, yl, yr, dim)

    def zhang_shu_limiter(
        self,
        averages: ArrayLike,
        dim: Literal["x", "y", "z"],
        interpolation_scheme: Literal["transverse", "gauss-legendre"],
        p: int,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Returns a slope-limited interpolation using Zhang and Shu's
        maximum-principle-satisfying slope limiter.

        Args:
            averages (ArrayLike): Cell averages. Has shape (nvars, nx, ny, nz).
            dim (Literal["x", "y", "z"]): Dimension of the interpolation.
            interpolation_scheme (Literal["transverse", "gauss-legendre"]): Mode of
                interpolation.
            p (int): Polynomial degree of the interpolation.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Limited face values (left, right). Each has
                shape (nvars, <nx, <ny, <nz).
        """
        return zhang_shu_advection(self, averages, dim, interpolation_scheme, p)

    @partial(method_timer, cat="AdvectionSolver.MOOD_violation_check")
    def MOOD_violation_check(self, u: ArrayLike, ustar: ArrayLike) -> ArrayLike:
        """
        Mood violation check implementation. See
        FiniteVolumeSolver.MOOD_violation_check.
        """
        _slc = self.array_slicer
        dmp_min, dmp_max = compute_dmp(
            self.apply_bc(u, 1)[_slc("rho")][np.newaxis, ...],
            dims=self.dims,
            include_corners=True,
        )
        return (
            np.minimum(ustar[_slc("rho")] - dmp_min[0], dmp_max[0] - ustar[_slc("rho")])
            < -1e-10
        )

    @partial(method_timer, cat="AdvectionSolver.compute_dt_and_fluxes")
    def compute_dt_and_fluxes(
        self,
        t: float,
        u: ArrayLike,
        mode: Literal["transverse", "gauss-legendre"],
        p: int,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Compute the time-step size and the fluxes.

        Args:
            t (float): Time value.
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).
            mode (str): Mode of interpolation. Possible values:
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
            p (int): Polynomial degree of the spatial discretization. If
                `slope_limiting` is "muscl", p is ignored and assumed to be 1.
            limiting_scheme (Optional[Literal["muscl"]]): Overwrites interpolation mode
                to use a slope-limiting scheme. Possible values:
                - "muscl": Use the MUSCL scheme.
                - "zhang-shu": Use Zhang and Shu's maximum-principle-satisfying slope
                    limiter.
            slope_limiter (Optional[Literal["minmod", "moncen"]]): Slope limiter to
                use if `slope_limiting` is "muscl". Possible values:
                - "minmod": Minmod limiter.
                - "moncen": Moncen limiter.

        Returns:
            dt (float): Time-step size.
            fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]]): Tuple of fluxes. The
                fluxes have the following shapes:
                - F: (nvars, nx + 1, ny, nz)
                - G: (nvars, nx, ny + 1, nz)
                - H: (nvars, nx, ny, nz + 1)
        """
        # set p
        _p = 1 if limiting_scheme == "muscl" else cast(int, p)

        # compute dt
        dt = self.get_dt(u)

        # get number of required ghost cells
        n_ghost_cells = max(-(-_p // 2) + 1, 2 * -(-_p // 2))

        # apply boundary conditions
        u_padded = self.apply_bc(u, n_ghost_cells)

        # initialize empty flux arrays
        F, G, H = np.array([]), np.array([]), np.array([])

        # x-fluxes
        if self.using_xdim:
            xl, xr = self.interpolate_face_nodes(
                u_padded,
                dim="x",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            riemann_problem_x = xr[:, :-1, ...], xl[:, 1:, ...]
            F = self.compute_numerical_fluxes(
                *riemann_problem_x, dim="x", quadrature=mode, p=_p
            )

        # y-fluxes
        if self.using_ydim:
            yl, yr = self.interpolate_face_nodes(
                u_padded,
                dim="y",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            riemann_problem_y = yr[:, :, :-1, ...], yl[:, :, 1:, ...]
            G = self.compute_numerical_fluxes(
                *riemann_problem_y, dim="y", quadrature=mode, p=_p
            )

        # z-fluxes
        if self.using_zdim:
            zl, zr = self.interpolate_face_nodes(
                u_padded,
                dim="z",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            riemann_problem_z = zr[:, :, :, :-1, ...], zl[:, :, :, 1:, ...]
            H = self.compute_numerical_fluxes(
                *riemann_problem_z, dim="z", quadrature=mode, p=_p
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

    @partial(method_timer, cat="AdvectionSolver.minisnapshot")
    def minisnapshot(self):
        super().minisnapshot()
        _slc = self.array_slicer
        self.minisnapshots["min_rho"].append(self.arrays["u"][_slc("rho")].min())
        self.minisnapshots["max_rho"].append(self.arrays["u"][_slc("rho")].max())
