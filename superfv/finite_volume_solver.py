from abc import ABC, abstractmethod
from functools import partial
from itertools import product
from typing import Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np

from .boundary_conditions import BoundaryConditions, DirichletBC
from .explicit_ODE_solver import ExplicitODESolver
from .slope_limiting import minmod, moncen, muscl
from .slope_limiting.MOOD import detect_troubles, init_MOOD, revise_fluxes
from .stencil import (
    conservative_interpolation_weights,
    get_gauss_legendre_face_nodes,
    get_gauss_legendre_face_weights,
    stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.array_management import ArrayLike, ArrayManager, ArraySlicer, crop_to_center
from .tools.timer import method_timer
from .visualization import plot_1d_slice, plot_2d_slice


class FiniteVolumeSolver(ExplicitODESolver, ABC):
    """
    Solve a nonlinear conservation law using the finite volume method in up to three
    dimensions.
    """

    @abstractmethod
    def define_vars(self) -> ArraySlicer:
        """
        Define the names of the solver variables.

        Returns:
            ArraySlicer: ArraySlicer object.
        """
        pass

    def dummy_riemann_solver(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        ur: Optional[ArrayLike] = None,
        ul: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Dummy Riemann solver to give an example of the required signature.

        Args:
            wl (ArrayLike): Left state primitive variables. Has shape
                (nvars, nx, ny, nz, ...).
            wr (ArrayLike): Right state primitive variables. Has shape
                (nvars, nx, ny, nz, ...).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".\
            ul (Optional[ArrayLike]): Left state conserved variables. Has shape
                (nvars, nx, ny, nz, ...).
            ur (Optional[ArrayLike]): Right state conserved variables. Has shape
                (nvars, nx, ny, nz, ...).

        Returns:
            ArrayLike: Numerical fluxes. Has shape (nvars, nx, ny, nz, ...).
        """
        raise NotImplementedError("Riemann solver not implemented.")

    @abstractmethod
    @partial(method_timer, cat="?.compute_dt_and_fluxes")
    def compute_dt_and_fluxes(
        self,
        t: float,
        u: ArrayLike,
        mode: Literal["transverse", "gauss-legendre"],
        p: int,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
    ) -> Tuple[
        float, Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]
    ]:
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
            fluxes (
                Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]
            ):
                Tuple of fluxes. The fluxes have the following shapes:
                - F: (nvars, nx + 1, ny, nz)
                - G: (nvars, nx, ny + 1, nz)
                - H: (nvars, nx, ny, nz + 1)
        """
        pass

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
        raise NotImplementedError("Zhang and Shu limiter not implemented.")

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
        riemann_solver: str = "dummy_riemann_solver",
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
        """
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, p, CFL, interpolation_scheme)
        self._init_ic(ic, ic_passives, cupy)
        self._init_bc(bcx, bcy, bcz, x_dirichlet, y_dirichlet, z_dirichlet)
        self._init_snapshots()
        self._init_array_manager(cupy)
        self._init_riemann_solver(riemann_solver)
        self._init_slope_limiting(
            MUSCL,
            ZS,
            adaptive_timestepping,
            max_adaptive_timesteps,
            MOOD,
            max_MOOD_iters,
            NAD,
            NAD_vars,
            PAD,
        )

    def _init_kwarg_attributes(self, kwarg_attributes):
        for key, value in kwarg_attributes.items():
            setattr(self, key, value)

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
        interpolation_scheme: Literal["gauss-legendre", "transverse"],
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
        self.n = {"x": nx, "y": ny, "z": nz}
        self.hx = (xlim[1] - xlim[0]) / nx
        self.hy = (ylim[1] - ylim[0]) / ny
        self.hz = (zlim[1] - zlim[0]) / nz
        self.h = {"x": self.hx, "y": self.hy, "z": self.hz}
        self.using_xdim = nx > 1
        self.using_ydim = ny > 1
        self.using_zdim = nz > 1
        self.using = {"x": self.using_xdim, "y": self.using_ydim, "z": self.using_zdim}
        self.dims = "".join(dim for dim, using in self.using.items() if using)
        self.ndim = len(self.dims)
        self.p = p
        self.CFL = CFL
        self.interpolation_scheme = interpolation_scheme

        if self.interpolation_scheme == "gauss-legendre" and self.ndim == 1:
            raise ValueError(
                "Gauss-Legendre interpolation scheme is not supported in 1D."
            )

        def _get_uniform_3D_mesh(xlim, ylim, zlim, nx, ny, nz, as_mesh: bool = True):
            x_interface = np.linspace(xlim[0], xlim[1], nx + 1)
            y_interface = np.linspace(ylim[0], ylim[1], ny + 1)
            z_interface = np.linspace(zlim[0], zlim[1], nz + 1)
            x_center = 0.5 * (x_interface[1:] + x_interface[:-1])
            y_center = 0.5 * (y_interface[1:] + y_interface[:-1])
            z_center = 0.5 * (z_interface[1:] + z_interface[:-1])
            if as_mesh:
                return np.meshgrid(x_center, y_center, z_center, indexing="ij")
            return x_center, y_center, z_center

        # core mesh
        self.x, self.y, self.z = _get_uniform_3D_mesh(
            xlim, ylim, zlim, nx, ny, nz, as_mesh=False
        )
        self.X, self.Y, self.Z = _get_uniform_3D_mesh(xlim, ylim, zlim, nx, ny, nz)

        # slab meshes
        slab_thickness = max(-(-p // 2) + 1, 2 * -(-p // 2))

        def _get_slab_limits(lim, spacing, thickness, pos=None):
            if pos is None:
                return (lim[0] - spacing * thickness, lim[1] + spacing * thickness)
            if pos == "l":
                return (lim[0] - spacing * thickness, lim[0])
            if pos == "r":
                return (lim[1], lim[1] + spacing * thickness)

        self.slab_meshes = {}
        for dim, pos in product("xyz", "lr"):
            slab_mesh = _get_uniform_3D_mesh(
                xlim=_get_slab_limits(
                    self.xlim, self.hx, slab_thickness, pos if dim == "x" else None
                ),
                ylim=_get_slab_limits(
                    self.ylim, self.hy, slab_thickness, pos if dim == "y" else None
                ),
                zlim=_get_slab_limits(
                    self.zlim, self.hz, slab_thickness, pos if dim == "z" else None
                ),
                nx=slab_thickness if dim == "x" else self.nx + 2 * slab_thickness,
                ny=slab_thickness if dim == "y" else self.ny + 2 * slab_thickness,
                nz=slab_thickness if dim == "z" else self.nz + 2 * slab_thickness,
            )
            self.slab_meshes[f"{dim}{pos}"] = slab_mesh

    def _init_ic(
        self,
        ic: Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ],
        cupy,
    ):
        # active variable initial conditions
        self.array_slicer = self.define_vars()
        self.n_active_vars = max(self.array_slicer.idxs) + 1
        _active_array_slicer = self.array_slicer.copy()

        # check indices are consecutive
        if self.array_slicer.idxs != set(range(self.n_active_vars)):
            raise ValueError("Variable indices must be consecutive.")

        # define callable
        def _callable_active_ic(x, y, z):
            return ic(_active_array_slicer, x, y, z)

        # determine if passive variables are used
        self.has_passives = bool(ic_passives)
        _n_passive_vars = len(cast(dict, ic_passives)) if self.has_passives else 0
        self.nvars = self.n_active_vars + _n_passive_vars

        if self.has_passives:
            self._init_ic_passives(ic_passives)

            # define callable to get the full ic array
            def _callable_ic_with_passives(
                x: ArrayLike, y: ArrayLike, z: ArrayLike
            ) -> ArrayLike:
                _slc = self.array_slicer
                out = np.empty((self.nvars, *x.shape), dtype=float)
                out[_slc("actives")] = _callable_active_ic(x, y, z)
                for name, f in cast(dict, ic_passives).items():
                    out[_slc(name)] = f(x, y, z)
                return out

            callable_ic = _callable_ic_with_passives

        else:
            callable_ic = _callable_active_ic

        # apply transformation
        self.callable_ic = self._ic_transform(callable_ic)

        # initialize the ODE solver and array manager
        ic_arr = self.callable_ic(self.X, self.Y, self.Z)
        self.maximum_principle = np.min(ic_arr, axis=(1, 2, 3), keepdims=True), np.max(
            ic_arr, axis=(1, 2, 3), keepdims=True
        )
        super().__init__(ic_arr, state_array_name="u", cupy=cupy)

    def _init_ic_passives(
        self,
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ],
    ):
        # gather indices of active variables
        _active_vars = self.array_slicer.var_names
        self.array_slicer.create_var_group("actives", tuple(_active_vars))

        # assign indices to passive variables
        for i, name in enumerate(cast(dict, ic_passives).keys()):
            self.array_slicer.add_var(name, i + self.n_active_vars)
        self.array_slicer.create_var_group(
            "passives", tuple(cast(dict, ic_passives).keys())
        )

        # check the union of active and passive variables
        _slc = self.array_slicer
        test_arr = np.arange(self.nvars)
        if not np.array_equal(
            np.concatenate((test_arr[_slc("actives")], test_arr[_slc("passives")])),
            test_arr,
        ):
            raise ValueError(
                "The intersection of active and passive variables must be the set of all variables."
            )

    def _ic_transform(
        self, f: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
        """
        Transform the initial condition function.
        """
        return f

    def _init_bc(
        self,
        bcx: Union[str, Tuple[str, str]],
        bcy: Union[str, Tuple[str, str]],
        bcz: Union[str, Tuple[str, str]],
        x_dirichlet: Optional[DirichletBC],
        y_dirichlet: Optional[DirichletBC],
        z_dirichlet: Optional[DirichletBC],
    ):
        # configure "ic" boundary conditions as "dirichlet" boundary conditions
        ic_configed_bc = {"x": bcx, "y": bcy, "z": bcz}
        ic_configed_dirichlet = {"x": x_dirichlet, "y": y_dirichlet, "z": z_dirichlet}
        for dim in "xyz":
            bc = ic_configed_bc[dim]
            f = ic_configed_dirichlet[dim]
            _bc = list(bc) if isinstance(bc, tuple) else [bc, bc]
            _f = list(f) if isinstance(f, tuple) else [f, f]
            for i in range(2):
                if _bc[i] == "ic":
                    _bc[i] = "dirichlet"
                    _f[i] = self.callable_ic
            ic_configed_bc[dim] = (_bc[0], _bc[1])
            ic_configed_dirichlet[dim] = (_f[0], _f[1])

        # initialize boundary conditions
        bcx = ic_configed_bc["x"]
        bcy = ic_configed_bc["y"]
        bcz = ic_configed_bc["z"]
        x_dirichlet = ic_configed_dirichlet["x"]
        y_dirichlet = ic_configed_dirichlet["y"]
        z_dirichlet = ic_configed_dirichlet["z"]
        _sm = self.slab_meshes
        self.bc = BoundaryConditions(
            self.array_slicer,
            bcx,
            bcy,
            bcz,
            x_dirichlet,
            y_dirichlet,
            z_dirichlet,
            (
                (_sm["xl"], _sm["xr"])
                if x_dirichlet is not None and self.using_xdim
                else None
            ),
            (
                (_sm["yl"], _sm["yr"])
                if y_dirichlet is not None and self.using_xdim
                else None
            ),
            (
                (_sm["zl"], _sm["zr"])
                if z_dirichlet is not None and self.using_xdim
                else None
            ),
        )

    def _init_snapshots(self):
        self.minisnapshots["MOOD_iters"] = []

    def _init_array_manager(self, cupy: bool):
        self.interpolation_cache = ArrayManager()
        self.MOOD_cache = ArrayManager()
        if cupy:
            self.interpolation_cache.enable_cupy()
            self.MOOD_cache.enable_cupy()

        # initialize flux arrays
        nvars, nx, ny, nz = self.nvars, self.nx, self.ny, self.nz
        self.arrays.add("F", np.zeros((nvars, nx + 1, ny, nz)))
        self.arrays.add("G", np.zeros((nvars, nx, ny + 1, nz)))
        self.arrays.add("H", np.zeros((nvars, nx, ny, nz + 1)))

    def _init_riemann_solver(self, riemann_solver: str):
        if not hasattr(self, riemann_solver):
            raise ValueError(f"Riemann solver {riemann_solver} not implemented.")
        self.riemann_solver: Callable[
            ...,
            ArrayLike,
        ] = getattr(self, riemann_solver)

    def _init_slope_limiting(
        self,
        MUSCL: bool,
        ZS: bool,
        adaptive_timestepping: bool,
        max_adaptive_timesteps: Optional[int],
        MOOD: bool,
        max_MOOD_iters: Optional[int],
        NAD: Optional[float],
        NAD_vars: Optional[Tuple[str]],
        PAD: Optional[Dict[str, Tuple[float, float]]],
    ):
        # a priori limiting
        self.a_priori_slope_limiting_scheme: Optional[Literal["muscl", "zhang-shu"]]
        self.a_priori_slope_limiter: Optional[Literal["minmod", "moncen"]]
        self.ZS_adaptive_timestep = False

        if all([MUSCL, ZS]):
            raise ValueError("Cannot use both Zhang-Shu and MUSCL slope limiting.")
        elif MUSCL:
            self.a_priori_slope_limiting_scheme = "muscl"
            self.a_priori_slope_limiter = "minmod"
        elif ZS:
            self.a_priori_slope_limiting_scheme = "zhang-shu"
            self.a_priori_slope_limiter = None
            if adaptive_timestepping:
                self.ZS_adaptive_timestep = True
                self.MOOD = True
                self.MOOD_cascade = ["fv" + str(self.p), "half-dt"]
                self.MOOD_max_iter_count = (
                    max_adaptive_timesteps if max_adaptive_timesteps is not None else 10
                )
        else:
            self.a_priori_slope_limiting_scheme = None
            self.a_priori_slope_limiter = None

        # a posteriori limiting
        if not self.ZS_adaptive_timestep:
            self.MOOD = MOOD
            self.MOOD_cascade = ["fv" + str(self.p), "muscl"]
            self.MOOD_max_iter_count = (
                max_MOOD_iters if max_MOOD_iters is not None else 1
            )

        # MOOD detection config
        self.MOOD_iter_count = 0
        self.NAD = NAD
        self.NAD_vars = NAD_vars
        self.PAD = PAD

        if self.PAD is None and self.ZS_adaptive_timestep:
            raise ValueError("PAD must be provided if adaptive_timestepping=True.")

    @partial(method_timer, cat="FiniteVolumeSolver.f")
    def f(self, t: float, u: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Compute the right-hand side of the ODE.

        Args:
            t (float): Time value.
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).

        Returns:
            dt (float): Time-step size.
            dudt (ArrayLike): Right-hand side of the ODE. Has shape
                (nvars, nx, ny, nz).
        """
        self.substep_count += 1
        dt, fluxes = self.compute_dt_and_fluxes(
            t=t,
            u=u,
            mode=self.interpolation_scheme,
            p=self.p,
            limiting_scheme=self.a_priori_slope_limiting_scheme,
            slope_limiter=self.a_priori_slope_limiter,
        )
        if self.MOOD and not (self.ZS_adaptive_timestep and self.substep_count > 1):
            dt, fluxes = self.MOOD_loop(t, dt, u, fluxes)
        return dt, self.RHS(u, *fluxes)

    @partial(method_timer, cat="FiniteVolumeSolver.MOOD_loop")
    def MOOD_loop(
        self,
        t: float,
        dt: float,
        u: ArrayLike,
        fluxes: Tuple[ArrayLike, ArrayLike, ArrayLike],
    ) -> Tuple[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]:
        """
        Revise fluxes using MOOD.

        Args:
            t (float): Time value.
            dt (float): Time-step size.
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).
            fluxes (Tuple[ArrayLike, ArrayLike, ArrayLike]): Tuple of fluxes. The
                fluxes have the following shapes:
                - F: (nvars, nx + 1, ny, nz)
                - G: (nvars, nx, ny + 1, nz)
                - H: (nvars, nx, ny, nz + 1)

        Returns:
            Tuple[ArrayLike, ArrayLike, ArrayLike]: Revised fluxes.
        """
        init_MOOD(self, u, fluxes)
        while detect_troubles(
            self, t, dt, u, fluxes, NAD=self.NAD, NAD_vars=self.NAD_vars, PAD=self.PAD
        ):
            dt, fluxes = revise_fluxes(
                self,
                t,
                dt,
                u,
                fluxes,
                mode=self.interpolation_scheme,
                slope_limiter="minmod",
            )
        return dt, fluxes

    @partial(method_timer, cat="FiniteVolumeSolver.apply_bc")
    def apply_bc(self, arr: ArrayLike, n_ghost_cells: int) -> ArrayLike:
        """
        Apply boundary conditions to an array by padding it with ghost cells along the
        active dimensions.

        Args:
            arr (ArrayLike): Field array. Has shape (nvars, nx, ny, nz).
            n_ghost_cells (int): Number of ghost cells to apply boundary conditions to.

        Returns:
            ArrayLike: Padded array. Has shape (nvars, >= nx, >= ny, >= nz). Axes
                corresponding to inactive dimensions are not padded and maintain length
                n[dim]. Axes corresponding to active dimensions are padded with
                n_ghost_cells ghost cells on each side, resulting in length
                n[dim] + 2 * n_ghost_cells.

        """
        return self.bc(
            arr,
            (
                int(self.using_xdim) * n_ghost_cells,
                int(self.using_ydim) * n_ghost_cells,
                int(self.using_zdim) * n_ghost_cells,
            ),
        )

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate")
    def interpolate(
        self,
        u: ArrayLike,
        x: Union[int, float, str] = 0,
        y: Union[int, float, str] = 0,
        z: Union[int, float, str] = 0,
        p: int = 0,
        sweep_order: str = "xyz",
        stencil_type: Literal[
            "conservative-interpolation", "uniform-quadrature"
        ] = "conservative-interpolation",
    ) -> ArrayLike:
        """
        Interpolates a node value from the finite-volume cell averages.

        Args:
            u (ArrayLike): Array to interpolate. Has shape (nvars, nx, ny, nz).
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
            ArrayLike: Interpolated node values. Has shape (nvars, <=nx, <=ny, <=nz).

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
                if self.using[direction]:
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

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_cell_centers")
    def interpolate_cell_centers(
        self,
        averages: ArrayLike,
        interpolation_scheme: Literal["transverse", "gauss-legendre"],
        p: Optional[int] = None,
        sweep_order: str = "xyz",
        clear_cache: bool = True,
    ) -> ArrayLike:
        """
        Interpolate face nodes from cell averages.

        Args:
            averages (ArrayLike): Cell averages. Has shape (nvars, nx, ny, nz).
            interpolation_scheme (Literal["transverse", "gauss-legendre"]): Mode of
                interpolation. Possible values:
                - "transverse": Interpolate nodes at the cell face centers.
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points.
                - "muscl": Interpolate nodes using the MUSCL scheme.
            p (Optional[int]): Polynomial degree of the interpolation for "transverse" and
                "gauss-legendre" modes. For "muscl" mode, p is ignored.
            sweep_order (str): Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            clear_cache (bool): Whether to clear the interpolation cache before
                performing the interpolation.

        Returns:
            ArrayLike: Cell centers. Has shape (nvars, <=nx, <=ny, <=nz).
        """
        if clear_cache:
            self.interpolation_cache.clear()

        cell_centers = self.interpolate(
            averages,
            x=0,
            y=0,
            z=0,
            p=p,
            sweep_order=sweep_order,
            stencil_type="conservative-interpolation",
        )
        return cell_centers

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_face_nodes")
    def interpolate_face_nodes(
        self,
        averages: ArrayLike,
        dim: Literal["x", "y", "z"],
        interpolation_scheme: Optional[Literal["transverse", "gauss-legendre"]] = None,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        p: Optional[int] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
        clear_cache: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Interpolate face nodes from cell averages.

        Args:
            averages (ArrayLike): Cell averages. Has shape (nvars, nx, ny, nz).
            dim (Literal["x", "y", "z"]): Direction of the face nodes. Can be "x", "y",
                or "z".
            interpolation_scheme (Optional[Literal["transverse", "gauss-legendre"]]):
                Mode of interpolation. Possible values:
                - None: Only valid when `limiting_scheme` is "muscl".
                - "transverse": Interpolate nodes at the cell face centers.
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points.
            limiting_scheme (Optional[Literal["muscl", "zhang-shu"]]): Slope limiting
                scheme. Possible values:
                - None: No slope limiting.
                - "muscl": Use the MUSCL scheme.
                - "zhang-shu": Use Zhang and Shu's maximum-principle-satisfying slope
                    limiter.
            p (Optional[int]): Polynomial degree of the interpolation schemes. For
                "muscl" slope limiting, p is ignored.
            slope_limiter (str): Additional option for slope limiting. Possible values:
                - "minmod": Minmod limiter for the MUSCL scheme.
                - "moncen": Moncen limiter for the MUSCL scheme.
            clear_cache (bool): Whether to clear the interpolation cache before
                performing the interpolation.
        Returns:
            Tuple[ArrayLike, ArrayLike]: Two node arrays. The first element is the left
                face nodes, and the second element is the right face nodes. Each face
                node has shape (nvars, <=nx, <=ny, <=nz, ninterpolations), unless the
                dimension is not used, in which case the face node is an empty array.
                ninterpolations is the number of Gauss-Legendre interpolation points
                for a degree `p` reconstruction if `interpolation_scheme` is
                "gauss-legendre". If `interpolation_scheme` is "face-centers",
                ninterpolations is 1.
        """
        if clear_cache:
            self.interpolation_cache.clear()
        sweep_orders = {"x": "yzx", "y": "zxy", "z": "xyz"}

        faces = []
        for pos in "lr":
            nodes = []
            if not self.using[dim]:
                raise ValueError(f"Dimension {dim} is not used in the mesh.")
            if interpolation_scheme == "gauss-legendre" and limiting_scheme is None:
                all_coords = get_gauss_legendre_face_nodes(self.dims, dim, pos, p)
                for coords in all_coords:
                    nodes.append(
                        self.interpolate(
                            averages,
                            **coords,
                            p=cast(int, p),
                            sweep_order=sweep_orders[dim],
                            stencil_type="conservative-interpolation",
                        )
                    )
                faces.append(np.stack(nodes, axis=-1))
            elif interpolation_scheme == "transverse" and limiting_scheme is None:
                (coords,) = get_gauss_legendre_face_nodes(self.dims, dim, pos, 0)
                node = self.interpolate(
                    averages,
                    **coords,
                    p=cast(int, p),
                    sweep_order=sweep_orders[dim],
                    stencil_type="conservative-interpolation",
                )
                faces.append(node[..., np.newaxis])
            elif limiting_scheme == "muscl":
                # early escape with MUSCL scheme
                return muscl(
                    averages,
                    dim=dim,
                    slope_limiter=minmod if slope_limiter == "minmod" else moncen,
                )
            elif limiting_scheme == "zhang-shu":
                # early escape with Zhang-Shu scheme
                return self.zhang_shu_limiter(
                    averages,
                    dim=dim,
                    interpolation_scheme=cast(
                        Literal["transverse", "gauss-legendre"], interpolation_scheme
                    ),
                    p=cast(int, p),
                )
            else:
                raise ValueError(
                    f"Unsupported interpolation and limiting schemes: {interpolation_scheme}, {limiting_scheme}."
                )

        return faces[0], faces[1]

    def compute_transverse_flux_integral(
        self,
        node_values: ArrayLike,
        dim: Literal["x", "y", "z"],
        p: int,
        clear_cache: bool = True,
    ) -> ArrayLike:
        """
        Compute the flux integral in a given direction.

        Args:
            node_values (ArrayLike): Node values. Has shape (nvars, nx, ny, nz, 1).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".
            p (int): Polynomial degree of the interpolation.
            clear_cache (bool): Whether to clear the interpolation cache before
                performing the quadrature.
        Returns:
            ArrayLike: Flux integral. Has shape (nvars, <=nx, <=ny, <=nz).
        """
        if clear_cache:
            self.interpolation_cache.clear()
        flux_integrals = self.interpolate(
            node_values[..., 0],
            p=p,
            stencil_type="uniform-quadrature",
            sweep_order={"x": "yz", "y": "xz", "z": "xy"}[dim],
        )
        return flux_integrals

    def compute_gauss_legendre_flux_integral(
        self, node_values: ArrayLike, dim: Literal["x", "y", "z"], p: int
    ) -> ArrayLike:
        """
        Compute the flux integral using Gauss-Legendre quadrature.

        Args:
            node_values (ArrayLike): Node values. Has shape
                (nvars, nx, ny, nz, ninterpolations).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".
            p (int): Polynomial degree of the interpolation.

        Returns:
            ArrayLike: Flux integral. Has shape (nvars, nx, ny, nz).
        """
        weights = get_gauss_legendre_face_weights(self.dims, dim, p)
        return np.sum(node_values * weights.reshape(1, 1, 1, 1, -1), axis=4)

    @partial(method_timer, cat="FiniteVolumeSolver.compute_numerical_fluxes")
    def compute_numerical_fluxes(
        self,
        left_nodes: ArrayLike,
        right_nodes: ArrayLike,
        dim: Literal["x", "y", "z"],
        quadrature: Literal["transverse", "gauss-legendre"],
        p: int,
        clear_cache: bool = True,
        crop: bool = True,
        left_transformed_nodes: Optional[ArrayLike] = None,
        right_transformed_nodes: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Compute the numerical fluxes.

        Args:
            left_nodes (ArrayLike): Value of nodes to the left of the discontinuity.
                Has shape (nvars, nx, ny, nz, ninterpolations).
            right_nodes (ArrayLike): Value of nodes to the right of the discontinuity.
                Has shape (nvars, nx, ny, nz, ninterpolations).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".
            quadrature (str): Mode of the numerical flux computation. Possible values:
                - "transverse": Compute the flux integral using a transverse
                    quadrature.
                - "gauss-legendre": Compute the flux integral using Gauss-Legendre
                    quadrature.
            p (int): Polynomial degree of the interpolation.
            clear_cache (bool): Whether to clear the interpolation cache before
                performing the quadrature.
            crop (bool): Whether to crop the numerical fluxes to the shape of the
                finite-volume mesh.
            left_transformed_nodes (Optional[ArrayLike]): Transformed value of nodes to
                the left of the discontinuity. Has shape
                (nvars, nx, ny, nz, ninterpolations).
            right_transformed_nodes (Optional[ArrayLike]): Transformed value of nodes to
                the right of the discontinuity. Has shape
                (nvars, nx, ny, nz, ninterpolations).
        """
        nodal_fluxes = self.riemann_solver(
            wl=left_nodes,
            wr=right_nodes,
            dim=dim,
            ul=left_transformed_nodes,
            ur=right_transformed_nodes,
        )

        if quadrature == "transverse":
            numerical_fluxes = self.compute_transverse_flux_integral(
                nodal_fluxes, dim=dim, p=p, clear_cache=clear_cache
            )
        elif quadrature == "gauss-legendre":
            numerical_fluxes = self.compute_gauss_legendre_flux_integral(
                nodal_fluxes, dim=dim, p=p
            )
        else:
            raise ValueError(f"Unsupported quadrature: {quadrature}")
        if crop:
            return crop_to_center(
                numerical_fluxes, self.arrays[{"x": "F", "y": "G", "z": "H"}[dim]].shape
            )
        return numerical_fluxes

    def RHS(
        self,
        u: ArrayLike,
        F: Optional[ArrayLike],
        G: Optional[ArrayLike],
        H: Optional[ArrayLike],
    ) -> ArrayLike:
        """
        Compute the right-hand side of the conservation law.

        Args:
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).
            F (Optional[ArrayLike]): Flux in the x-direction. Has shape
                (nvars, nx + 1, ny, nz).
            G (Optional[ArrayLike]): Flux in the y-direction. Has shape
                (nvars, nx, ny + 1, nz).
            H (Optional[ArrayLike]): Flux in the z-direction. Has shape
                (nvars, nx, ny, nz + 1).

        Returns:
            dydt (ArrayLike): Right-hand side of the conservation law. Has shape
                (nvars, nx, ny, nz).
        """
        _slc = self.array_slicer
        dydt = np.zeros_like(u)
        for dim, _F in zip(["x", "y", "z"], [F, G, H]):
            if self.using[dim]:
                dydt += -(1 / self.h[dim]) * (
                    cast(ArrayLike, _F)[_slc(**{dim: (1, None)})]
                    - cast(ArrayLike, _F)[_slc(**{dim: (None, -1)})]
                )
        return dydt

    @partial(method_timer, cat="FiniteVolumeSolver.snapshot")
    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        data = {"u": self.arrays.get_numpy("u", copy=True)}
        self.snapshots.log(self.t, data)

    @partial(method_timer, cat="FiniteVolumeSolver.minisnapshot")
    def minisnapshot(self):
        super().minisnapshot()
        self.minisnapshots["MOOD_iters"].append(self.MOOD_iter_count)

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

    def plot_1d_slice(self, *args, **kwargs):
        return plot_1d_slice(self, *args, **kwargs)

    def plot_2d_slice(self, *args, **kwargs):
        return plot_2d_slice(self, *args, **kwargs)
