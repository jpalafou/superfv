from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
import wtflux.hydro as hydro

from .boundary_conditions import DirichletBC
from .finite_volume_solver import FiniteVolumeSolver
from .fv import fv_average
from .hydro import conservatives_from_primitives, primitives_from_conservatives
from .riemann_solvers import llf
from .slope_limiting.zhang_and_shu import zhang_shu_advection
from .tools.array_management import ArrayLike, ArraySlicer
from .tools.timer import method_timer


class EulerSolver(FiniteVolumeSolver):
    """
    Solve the euler system of equations using a finite volume method.
    """

    def __init__(
        self,
        ic: Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ] = None,
        gamma: float = 5 / 3,
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
        riemann_solver: str = "llf",
        MUSCL: bool = False,
        ZS: bool = False,
        adaptive_timestepping: bool = True,
        max_adaptive_timesteps: Optional[int] = None,
        MOOD: bool = False,
        max_MOOD_iters: Optional[int] = None,
        NAD: Optional[float] = None,
        NAD_vars: Optional[Tuple[str]] = None,
        PAD: Optional[Dict[str, Tuple[float, float]]] = None,
        cupy: bool = False,
        **kwarg_attributes,
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
            gamma (float): Ratio of specific heats.
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
            NAD (Optional[float]): The NAD tolerance. If None, NAD is not checked.
            NAD_vars (Optional[Tuple[str]]): The variables to check for NAD. If None,
                all "active" variables are checked.
            PAD (Optional[Dict[str, Tuple[float, float]]]): Dict of variable names to
                (lower, upper) bounds. If None, PAD is not checked.
            cupy (bool): Whether to use CuPy for array operations.
            kwarg_attributes: kwargs to be stored as attributes.
        """
        super().__init__(
            ic=ic,
            ic_passives=ic_passives,
            gamma=gamma,
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
            adaptive_timestepping=adaptive_timestepping,
            max_adaptive_timesteps=max_adaptive_timesteps,
            MOOD=MOOD,
            max_MOOD_iters=max_MOOD_iters,
            NAD=NAD,
            NAD_vars=NAD_vars,
            PAD=PAD,
            cupy=cupy,
        )

    def _ic_transform(
        self, f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
        """
        Modify IC callable to convert to conservative variables and compute the cell
        average.
        """
        _slc = self.array_slicer

        def conservative_wrapper(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
            return conservatives_from_primitives(_slc, f(x, y, z), self.gamma)

        def cell_average_wrapper(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
            return fv_average(
                conservative_wrapper,
                x,
                y,
                z,
                h=self.h,
                p=(
                    self.p if self.using_xdim else 0,
                    self.p if self.using_ydim else 0,
                    self.p if self.using_zdim else 0,
                ),
            )

        return cell_average_wrapper

    def _init_array_manager(self, cupy: bool):
        super()._init_array_manager(cupy)
        self.arrays["w"] = np.empty_like(self.arrays["u"])

    def _init_snapshots(self):
        super()._init_snapshots()
        self.minisnapshots["min_rho"] = []
        self.minisnapshots["max_rho"] = []
        self.minisnapshots["min_E"] = []
        self.minisnapshots["max_E"] = []

    def _init_hydro(self, gamma: float):
        self.gamma = gamma

    def define_vars(self) -> ArraySlicer:
        """
        Returns an ArraySlicer object with the following variables:
            - rho: Density.
            - vx: x-component of the velocity.
            - vy: y-component of the velocity.
            - vz: z-component of the velocity.
        """
        array_slicer = ArraySlicer(
            {
                "rho": 0,
                "vx": 1,
                "vy": 2,
                "vz": 3,
                "P": 4,
                "mx": 1,
                "my": 2,
                "mz": 3,
                "E": 4,
            },
            ndim=4,
        )
        array_slicer.create_var_group("v", ("vx", "vy", "vz"))
        array_slicer.create_var_group("m", ("mx", "my", "mz"))
        return array_slicer

    @partial(method_timer, cat="EulerSolver.llf")
    def llf(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        ul: Optional[ArrayLike] = None,
        ur: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Riemann solver implementation. See FiniteVolumeSolver.dummy_riemann_solver.
        """
        return llf(self.array_slicer, wl, wr, dim, self.gamma, ul, ur)

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

    @partial(method_timer, cat="EulerSolver.compute_dt_and_fluxes")
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
        _slc = self.array_slicer

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
            u_xl, u_xr = self.interpolate_face_nodes(
                u_padded,
                dim="x",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            w_xl = primitives_from_conservatives(_slc, u_xl, self.gamma)
            w_xr = primitives_from_conservatives(_slc, u_xr, self.gamma)
            F = self.compute_numerical_fluxes(
                left_nodes=w_xr[:, :-1, ...],
                right_nodes=w_xl[:, 1:, ...],
                dim="x",
                quadrature=mode,
                p=_p,
                left_transformed_nodes=u_xr[:, :-1, ...],
                right_transformed_nodes=u_xl[:, 1:, ...],
            )

        # y-fluxes
        if self.using_ydim:
            u_yl, u_yr = self.interpolate_face_nodes(
                u_padded,
                dim="y",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            w_yl = primitives_from_conservatives(_slc, u_yl, self.gamma)
            w_yr = primitives_from_conservatives(_slc, u_yr, self.gamma)
            G = self.compute_numerical_fluxes(
                left_nodes=w_yr[:, :, :-1, ...],
                right_nodes=w_yl[:, :, 1:, ...],
                dim="y",
                quadrature=mode,
                p=_p,
                left_transformed_nodes=u_yr[:, :, :-1, ...],
                right_transformed_nodes=u_yl[:, :, 1:, ...],
            )

        # z-fluxes
        if self.using_zdim:
            u_zl, u_zr = self.interpolate_face_nodes(
                u_padded,
                dim="z",
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=_p,
                slope_limiter=slope_limiter,
            )
            w_zl = primitives_from_conservatives(_slc, u_zl, self.gamma)
            w_zr = primitives_from_conservatives(_slc, u_zr, self.gamma)
            H = self.compute_numerical_fluxes(
                left_nodes=w_zr[:, :, :, :-1, ...],
                right_nodes=w_zl[:, :, :, 1:, ...],
                dim="z",
                quadrature=mode,
                p=_p,
                left_transformed_nodes=u_zr[:, :, :, :-1, ...],
                right_transformed_nodes=u_zl[:, :, :, 1:, ...],
            )

        return dt, (F, G, H)

    @partial(method_timer, cat="EulerSolver.get_dt")
    def get_dt(self, w: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            w (ArrayLike): Primitive solution values. Has shape (nvars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
        """
        _slc = self.array_slicer
        h = min(self.h)
        c = hydro.sound_speed(rho=w[_slc("rho")], P=w[_slc("P")], gamma=self.gamma)
        out = (
            self.CFL * h / np.max(np.sum(np.abs(w[_slc("v")]), axis=0) + self.ndim * c)
        )
        return out.item()

    @partial(method_timer, cat="EulerSolver.minisnapshot")
    def minisnapshot(self):
        super().minisnapshot()
        _slc = self.array_slicer
        self.minisnapshots["min_rho"].append(self.arrays["u"][_slc("rho")].min())
        self.minisnapshots["max_rho"].append(self.arrays["u"][_slc("rho")].max())
        self.minisnapshots["min_E"].append(self.arrays["u"][_slc("E")].min())
        self.minisnapshots["max_E"].append(self.arrays["u"][_slc("E")].max())

    @partial(method_timer, cat="EulerSolver.snapshot")
    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        self.arrays["w"] = primitives_from_conservatives(
            self.array_slicer, self.arrays["u"], self.gamma
        )
        data = {
            "u": self.arrays.get_numpy("u", copy=True),
            "w": self.arrays.get_numpy("w", copy=True),
        }
        self.snapshots.log(self.t, data)
