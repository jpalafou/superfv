import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast

import numpy as np

from superfv.fv import gauss_legendre_for_finite_volume

from .boundary_conditions import BoundaryConditions, DirichletBC, Field
from .explicit_ODE_solver import ExplicitODESolver
from .fv import fv_average
from .mesh import UniformFVMesh
from .slope_limiting import minmod, moncen, muscl
from .slope_limiting.MOOD import detect_troubles, init_MOOD, revise_fluxes
from .slope_limiting.zhang_and_shu import zhang_shu_limiter
from .stencil import (
    conservative_interpolation_weights,
    stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.array_management import (
    CUPY_AVAILABLE,
    ArrayLike,
    ArrayManager,
    VariableIndexMap,
    crop,
    crop_to_center,
    xp,
)
from .tools.timer import method_timer
from .visualization import plot_1d_slice, plot_2d_slice


class FiniteVolumeSolver(ExplicitODESolver, ABC):
    """
    Solve a nonlinear conservation law using the finite volume method in up to three
    dimensions.
    """

    @abstractmethod
    def define_vars(self) -> VariableIndexMap:
        """
        Define the names of the solver variables.

        Returns:
            VariableIndexMap object.
        """
        pass

    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Convert primitive variables to conservative variables.

        Args:
            w: Array of primitive variables.

        Returns:
            Array of conservative variables.
        """
        raise NotImplementedError("Conservative variables not implemented.")

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u: Array of conservative variables.

        Returns:
            Array of primitive variables.
        """
        raise NotImplementedError("Primitive variables not implemented.")

    def dummy_riemann_solver(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
    ) -> ArrayLike:
        """
        Dummy Riemann solver to give an example of the required signature.

        Args:
            wl: Array of primitive variables to the left of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            wr: Array of primitive variables to the right of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            dim: Direction in which the Riemann problem is solved: "x", "y", or "z".

        Returns:
            Array of numerical fluxes at the interface. Has shape
                (nvars, nx, ny, nz, ...).
        """
        raise NotImplementedError("Riemann solver not implemented.")

    @abstractmethod
    @partial(method_timer, cat="?.compute_dt")
    def compute_dt(self, w: ArrayLike) -> float:
        """
        Compute the time-step size.

        Args:
            w: Array of primitive variables. Has shape (nvars, nx, ny, nz).

        Returns:
            Time-step size.
        """
        pass

    def __init__(
        self,
        ic: Callable[[VariableIndexMap, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
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
        riemann_solver: str = "dummy_riemann_solver",
        MUSCL: bool = False,
        ZS: bool = False,
        adaptive_timestepping: bool = True,
        max_adaptive_timesteps: Optional[int] = None,
        MOOD: bool = False,
        max_MOOD_iters: Optional[int] = None,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "all",
        NAD: Optional[float] = None,
        PAD: Optional[Dict[str, Tuple[float, float]]] = None,
        PAD_tol: float = 1e-15,
        SED: bool = False,
        cupy: bool = False,
    ):
        """
        Initialize the finite volume solver.

        Args:
            ic: Initial condition function of pointwise, primitive variables. The
                function must accept the following arguments:
                - idx: VariableIndexMap object.
                - x: x-coordinate array. Has shape (nx, ny, nz).
                - y: y-coordinate array. Has shape (nx, ny, nz).
                - z: z-coordinate array. Has shape (nx, ny, nz).
                The function must return an array with shape (nvars, nx, ny, nz).
            ic_passives: Dictionary of initial condition functions for passive
                variables. The dictionary keys are the names of the passive variables
                and the values are the corresponding initial condition functions.
                Each function must accept the following arguments:
                - x: x-coordinate array. Has shape (nx, ny, nz).
                - y: y-coordinate array. Has shape (nx, ny, nz).
                - z: z-coordinate array. Has shape (nx, ny, nz).
                The function must return an array with shape (nx, ny, nz).
            bcx, bcy, bcz: Boundary conditions for the x, y, and z directions. Each can
                be specified as a single string to apply the same condition on both
                sides, or as a tuple of two strings to apply different conditions on
                the lower and upper (left and right) boundaries, respectively.
                Supported boundary condition names include: "periodic", "dirichlet",
                "free", "reflective", "zeros", and "ones".
            x_dirichlet, y_dirichlet, z_dirichlet: Additional argument for "dirichlet"
                boundary conditions. Must be a callable that takes following arguments:
                - idx: VariableIndexMap object.
                - x: x-coordinate array. Has shape (nx, ny, nz).
                - y: y-coordinate array. Has shape (nx, ny, nz).
                - z: z-coordinate array. Has shape (nx, ny, nz).
                - t: Optional time at which the boundary condition is applied.
                And returns an array with shape (nvars, nx, ny, nz). Can also be given
                as a tuple of two callables, one for the left and one for the right
                boundary condition. If a single callable is provided, it will be used
                for both boundaries.
            xlim, ylim, zlim: Limits of the domain in the x, y, and z-directions.
            nx, ny, nz: Number of cells in the x, y, and z-directions.
            p: Maximum polynomial degree of the spatial discretization.
            CFL: CFL number.
            interpolation_scheme: Scheme to use for the interpolation of face nodes.
                Possible values:
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
            flux_recipe: Recipe for interpolating flux nodes. Possible values:
                - 1: Interpolate conservative nodes from conservative cell averages.
                    Apply slope limiting to the conservative nodes. Transform to
                    primitive variables.
                - 2: Interpolate conservative nodes from conservative cell averages.
                    Transform to primitive variables. Apply slope limiting to the
                    primitive nodes.
                - 3: Interpolate primitive cell averages from conservative cell
                    averages, either by interpolating to cell-centered values
                    intermittently or transforming directly with `lazy_primitives=True`.
                    Interpolate primitive nodes from primitive cell averages.
                    Apply slope limiting to the primitive nodes.
            lazy_primitives: Whether to transform conservative cell averages
                directly to primitive cell averages. Note that this is a second order
                operation. If
                - `flux_recipe=1`: This argument is ignored.
                - `flux_recipe=2`: The lazy primitives become the fallback option.
                - `flux_recipe=3`: The lazy primitives are used to interpolate the
                    primitive flux nodes.
            riemann_solver: Name of the Riemann solver function. Must be implemented in
                the derived class.
            MUSCL: Whether to use the MUSCL scheme for a priori slope limiting.
            ZS: Whether to use Zhang and Shu's maximum-principle-satisfying a priori
                slope limiter.
            adaptive_timestepping: Option for `ZS=True` to half the time-step size if a
                maximum principle violation is detected. If True, MOOD is overwritten
                to only modify the time-step size and not the fluxes. Ignored if
                `ZS=False`.
            max_adaptive_timesteps: Maximum number of adaptive time steps if both
                `ZS=True` and `adaptive_timestepping=True`. If None, defaults to 10.
                Ignored if either `ZS` or `adaptive_timestepping` is False.
            MOOD: Whether to use MOOD for a posteriori flux revision. Ignored if
                `ZS=True` and `adaptive_timestepping=True`.
            max_MOOD_iters: Maximum number of MOOD iterations if `MOOD=True`. If None,
                defaults to 1. If 'MOOD=False', this argument is ignored.
            limiting_vars: Specifies which variables are subject to slope limiting.
                - "all": All variables are subject to slope limiting.
                - "actives": Only active variables are subject to slope limiting.
                - Tuple[str, ...]: A tuple of variable names that are subject to slope
                    limiting. Must be defined in `self.define_vars()`.
                For the Zhang-Shu limiter, all variables are always limited, but
                `limiting_vars` determines which variables are checked for PAD when
                using adaptive timestepping.
            NAD: The NAD tolerance. If None, NAD is not checked.
            PAD: Dict of `limiting_vars` and their corresponding PAD tolerances. If a
                limiting variable is not in the dict, it is given a PAD tolerance of
                (-np.inf, np.inf). If None, PAD is not checked.
            PAD_tol: Tolerance for the PAD check as an absolute value from the minimum
                and maximum values of the variable.
            SED: Whether to use smooth extrema detection for slope limiting.
            cupy: Whether to use CuPy for array operations.
        """
        self._init_cupy(cupy)
        self._init_mesh(
            xlim,
            ylim,
            zlim,
            nx,
            ny,
            nz,
            p,
            CFL,
            interpolation_scheme,
            flux_recipe,
            lazy_primitives,
        )
        self._init_ic(ic, ic_passives)
        self._init_bc(bcx, bcy, bcz, x_dirichlet, y_dirichlet, z_dirichlet)
        self._init_snapshots()
        self._init_array_manager()
        self._init_riemann_solver(riemann_solver)
        self._init_slope_limiting(
            MUSCL,
            ZS,
            adaptive_timestepping,
            max_adaptive_timesteps,
            MOOD,
            max_MOOD_iters,
            limiting_vars,
            NAD,
            PAD,
            PAD_tol,
            SED,
        )

    def _init_cupy(self, cupy: bool):
        self.cupy = False
        if cupy and CUPY_AVAILABLE:
            self.cupy = True
        elif cupy:
            warnings.warn("CuPy is not available. Using NumPy instead.")
        self.xp = xp if self.cupy else np

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
        flux_recipe: Literal[1, 2, 3],
        lazy_primitives: bool,
    ):
        # determine which dimensions are used
        self.using_xdim = nx > 1
        self.using_ydim = ny > 1
        self.using_zdim = nz > 1
        self.using = {"x": self.using_xdim, "y": self.using_ydim, "z": self.using_zdim}
        self.dims = "".join(dim for dim, using in self.using.items() if using)
        self.axes = tuple("xyz".index(dim) + 1 for dim in self.dims)
        self.ndim = len(self.dims)

        # assign slab thickness
        slab_thickness = -2 * (-p // 2) + 1

        # init mesh object
        self.mesh = UniformFVMesh(
            nx=nx,
            ny=ny,
            nz=nz,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            ignore_x=not self.using_xdim,
            ignore_y=not self.using_ydim,
            ignore_z=not self.using_zdim,
            slab_depth=slab_thickness,
        )

        # assign p and CFL
        if p < 0:
            raise ValueError("The polynomial degree must be non-negative.")
        if CFL <= 0:
            raise ValueError("The CFL number must be positive.")
        self.p = p
        self.px = p if self.using_xdim else 0
        self.py = p if self.using_ydim else 0
        self.pz = p if self.using_zdim else 0
        self.CFL = CFL

        # assign flux recipe and interpolation scheme
        if flux_recipe not in (1, 2, 3):
            raise ValueError(
                "flux_recipe must be 1, 2, or 3. See the documentation for details."
            )
        if interpolation_scheme == "gauss-legendre" and self.ndim == 1:
            raise ValueError(
                "Gauss-Legendre interpolation scheme is not supported in 1D."
            )
        self.interpolation_scheme = interpolation_scheme
        self.flux_recipe = flux_recipe
        self.lazy_primitives = lazy_primitives

    def _init_ic(
        self,
        ic: Callable[[VariableIndexMap, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ],
    ):
        # Define the following attributes:
        self.variable_index_map: VariableIndexMap
        self.nvars: int
        self.n_passive_vars: int
        self.active_vars: Set[str]
        self.passive_vars: Set[str]
        self.callable_ic: Callable[
            [VariableIndexMap, ArrayLike, ArrayLike, ArrayLike, Optional[float]],
            ArrayLike,
        ]

        # Define variable index map
        idx = self.define_vars()

        if ic_passives:
            for v in ic_passives.keys():
                if v not in idx.var_idx_map:
                    idx.add_var(v, idx.nvars)
            idx.add_var_to_group("passives", ic_passives.keys())

            def callable_ic(idx, x, y, z, t=None):
                out = ic(idx, x, y, z)
                for v, f in ic_passives.items():
                    out[idx(v)] = f(x, y, z)
                return out

        else:

            def callable_ic(idx, x, y, z, t=None):
                return ic(idx, x, y, z)

        self.callable_ic = callable_ic

        self.variable_index_map = idx
        self.nvars = idx.nvars
        self.n_passive_vars = len(np.arange(self.nvars)[idx("passives")])

        # Initialize the ODE solver with the initial condition array
        ic_arr = self.fv_average_wrapper(self.conservatives_wrapper(self.callable_ic))(
            self.variable_index_map, *self.mesh.coords, 0.0
        )
        self.maximum_principle = np.min(ic_arr, axis=(1, 2, 3), keepdims=True), np.max(
            ic_arr, axis=(1, 2, 3), keepdims=True
        )
        super().__init__(ic_arr, state_array_name="u")

    def conservatives_wrapper(
        self,
        f: Field,
    ) -> Field:
        """
        Wrapper to convert output of a function from primitive variables to
        conservative variables.
        """

        def wrapped_func(
            variable_index_map: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
        ) -> ArrayLike:
            return self.conservatives_from_primitives(f(variable_index_map, x, y, z, t))

        return wrapped_func

    def fv_average_wrapper(
        self,
        f: Field,
    ) -> Field:
        """
        Wrapper to convert output of a function from pointwise to cell-averaged.
        """

        def wrapped_func(
            variable_index_map: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
        ) -> ArrayLike:
            return fv_average(
                lambda x, y, z: f(variable_index_map, x, y, z, t),
                x,
                y,
                z,
                h=(self.mesh.hx, self.mesh.hy, self.mesh.hz),
                p=(self.px, self.py, self.pz),
            )

        return wrapped_func

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
        def as_two_list(x):
            if isinstance(x, list) or isinstance(x, tuple):
                if len(x) != 2:
                    raise TypeError("Expected a two-element iterable.")
                return list(x)
            else:
                return [x, x]

        ic_configed_bc = {"x": bcx, "y": bcy, "z": bcz}
        ic_configed_dirichlet = {"x": x_dirichlet, "y": y_dirichlet, "z": z_dirichlet}
        for dim in "xyz":
            bc2 = as_two_list(ic_configed_bc[dim])
            f2 = as_two_list(ic_configed_dirichlet[dim])
            for i in range(2):
                if bc2[i] == "ic":
                    bc2[i] = "dirichlet"
                    f2[i] = lambda idx, x, y, z, t: self.callable_ic(idx, x, y, z, 0.0)
            ic_configed_bc[dim] = (bc2[0], bc2[1])
            ic_configed_dirichlet[dim] = (f2[0], f2[1])

        # initialize boundary conditions
        self.bc = BoundaryConditions(
            self.variable_index_map,
            self.mesh,
            bcx=ic_configed_bc["x"],
            bcy=ic_configed_bc["y"],
            bcz=ic_configed_bc["z"],
            x_dirichlet=ic_configed_dirichlet["x"],
            y_dirichlet=ic_configed_dirichlet["y"],
            z_dirichlet=ic_configed_dirichlet["z"],
            conservatives_wrapper=self.conservatives_wrapper,
            cupy=self.cupy,
        )

        # this is used to apply ghost cells to alpha in the smooth extrema detector
        periodic = ("periodic", "periodic")
        self.bc_for_smooth_extrema_detection = BoundaryConditions(
            self.variable_index_map,
            self.mesh,
            bcx="periodic" if self.bc.bcx == periodic else "ones",
            bcy="periodic" if self.bc.bcy == periodic else "ones",
            bcz="periodic" if self.bc.bcz == periodic else "ones",
            cupy=self.cupy,
        )

        # this is used to apply ghost cells to the troubled cell mask in MOOD
        self.bc_for_troubled_cell_mask = BoundaryConditions(
            self.variable_index_map,
            self.mesh,
            bcx="periodic" if self.bc.bcx == periodic else "free",
            bcy="periodic" if self.bc.bcy == periodic else "free",
            bcz="periodic" if self.bc.bcz == periodic else "free",
            cupy=self.cupy,
        )

    def _init_snapshots(self):
        self.minisnapshots["MOOD_iters"] = []

    def _init_array_manager(self):
        self.interpolation_cache = ArrayManager()
        self.MOOD_cache = ArrayManager()
        self.ZS_cache = ArrayManager()
        if self.cupy:
            self.arrays.transfer_to_device("gpu")
            self.interpolation_cache.transfer_to_device("gpu")
            self.MOOD_cache.transfer_to_device("gpu")
            self.ZS_cache.transfer_to_device("gpu")

        # initialize flux arrays
        nvars, nx, ny, nz = self.nvars, self.mesh.nx, self.mesh.ny, self.mesh.nz
        self.arrays.add("F", np.zeros((nvars, nx + 1, ny, nz)))
        self.arrays.add("G", np.zeros((nvars, nx, ny + 1, nz)))
        self.arrays.add("H", np.zeros((nvars, nx, ny, nz + 1)))

        # initialize snapshot arrays
        self.arrays.add("ucc", np.empty((nvars, nx, ny, nz)))
        self.arrays.add("wcc", np.empty((nvars, nx, ny, nz)))
        self.arrays.add("w", np.empty((nvars, nx, ny, nz)))

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
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]],
        NAD: Optional[float],
        PAD: Optional[Dict[str, Tuple[float, float]]],
        PAD_tol: float,
        SED: bool,
    ):
        # Initial setup
        self.a_priori_slope_limiter: Optional[str] = None
        self.a_priori_slope_limiting_scheme: Optional[str] = None
        self.MOOD: bool = False
        self.MOOD_cascade: List[str] = []
        self.MOOD_iter_count: int = 0
        self.MOOD_max_iter_count: int = np.iinfo(np.int64).max
        self.NAD: Optional[float] = None
        self.PAD_tol: float = np.nan
        self.SED: bool = False
        self.ZS_adaptive_timestep: bool = False

        if self.p < 1:
            return

        # Validate and assign limiting schemes
        if MUSCL and ZS:
            raise ValueError("Cannot use both Zhang-Shu and MUSCL slope limiting.")

        # Set a priori slope limiting schemes
        if MUSCL:
            self.a_priori_slope_limiting_scheme = "muscl"
            self.a_priori_slope_limiter = "minmod"
        elif ZS:
            self.a_priori_slope_limiting_scheme = "zhang-shu"
            self.ZS_adaptive_timestep = adaptive_timestepping

        # Determine if MOOD is enabled
        self.MOOD = MOOD or self.ZS_adaptive_timestep

        if self.MOOD:
            self.MOOD_cascade = (
                ["fv" + str(self.p), "half-dt"]
                if self.ZS_adaptive_timestep
                else ["fv" + str(self.p), "muscl"]
            )
            self.MOOD_max_iter_count = (
                max_adaptive_timesteps if self.ZS_adaptive_timestep else max_MOOD_iters
            ) or (10 if self.ZS_adaptive_timestep else 1)

        # Prespare limiting variables
        if limiting_vars == "actives":
            self.variable_index_map.add_var_to_group("limiting_vars", "primitives")
        elif limiting_vars == "all":
            self.variable_index_map.add_var_to_group(
                "limiting_vars", self.variable_index_map.var_idx_map.keys()
            )
        elif isinstance(limiting_vars, tuple):
            self.variable_index_map.add_var_to_group("limiting_vars", limiting_vars)
        else:
            raise ValueError(
                "limiting_vars must be a tuple of variable names, 'actives', or 'all'."
            )

        # Set SED
        self.SED = SED

        # If no MOOD, early return
        if not self.MOOD:
            return

        # Set NAD
        self.NAD = NAD

        # Handle PAD
        self.PAD_tol = PAD_tol
        if PAD is None:
            if self.ZS_adaptive_timestep:
                raise ValueError("PAD must be provided if adaptive_timestepping=True.")
        else:
            # Create and register PAD array
            PAD_arr = np.array([[-np.inf, np.inf]] * self.nvars)
            for v, (lb, ub) in PAD.items():
                PAD_arr[self.variable_index_map(v)] = [lb, ub]
            PAD_arr = PAD_arr.reshape(self.nvars, 1, 1, 1, 2)
            self.arrays.add("PAD", PAD_arr)  # Register PAD array

    @partial(method_timer, cat="FiniteVolumeSolver.f")
    def f(self, t: float, u: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Compute the right-hand side of the ODE.

        Args:
            t: Time value.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).

        Returns:
            dt: Time-step size.
            dudt (ArrayLike): Right-hand side of the ODE as an array. Has shape
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
        self.f_cleanup()
        return dt, self.RHS(u, *fluxes)

    def f_cleanup(self):
        if self.a_priori_slope_limiting_scheme == "zhang-shu":
            self.ZS_cache.clear()

    @partial(method_timer, cat="FiniteVolumeSolver.compute_dt_and_fluxes")
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
            t: Time value.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            mode: Mode of interpolation. Possible values:
                - "transverse": Interpolate nodes at the cell face centers. Compute the
                    flux integral using a transverse quadrature.
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points. Compute the flux integral using Gauss-Legendre quadrature.
            p: Polynomial degree of the spatial discretization. If `slope_limiting` is
                "muscl", p is ignored and assumed to be 1.
            limiting_scheme: Scheme to use for slope limiting. If None, no slope
                limiting is applied. Possible values:
                - None: No slope limiting is applied.
                - "muscl": Use the MUSCL scheme.
                - "zhang-shu": Use Zhang and Shu's maximum-principle-satisfying slope
                    limiter.
            slope_limiter: Slope limiter to use if `limiting_scheme` is "muscl".
                If None, no slope limiter is applied. Possible values:
                - None: No slope limiter is applied.
                - "minmod": Minmod limiter.
                - "moncen": Moncen limiter.

        Returns:
            dt: Time-step size.
            fluxes: Tuple of flux arrays or None if the corresponding dimension is
                unused. The fluxes have the following shapes if not None:
            - F: (nvars, nx + 1, ny, nz)
            - G: (nvars, nx, ny + 1, nz)
            - H: (nvars, nx, ny, nz + 1)
        """
        p = 1 if limiting_scheme == "muscl" else cast(int, p)
        ZS = slope_limiter == "zhang-shu"

        # compute primitive averages
        if self.lazy_primitives:
            w = self.primitives_from_conservatives(u)
        else:
            u_padded = self.apply_bc(u, -2 * (-p // 2), t=t, p=p)
            w = self.interpolate_cell_centers(u_padded, p)
            w[...] = self.primitives_from_conservatives(w)
            w = self.interpolate_cell_averages(w, p)

        # apply ghost cells
        nghost1 = -(-p // 2)
        u_padded = self.apply_bc(u, nghost1, t=t, p=p)
        w_padded = self.apply_bc(w, nghost1, t=t, p=p, primitives=True)

        # compute dt
        dt = self.compute_dt(w_padded)

        # compute fluxes
        fluxes: List[Optional[ArrayLike]] = []
        for axis, dim in zip([1, 2, 3], ["x", "y", "z"]):
            # skip unused dimensions
            if not self.using[dim]:
                fluxes.append(None)
                continue

            # interpolate face nodes as primitive variables
            w_xl, w_xr = self.interpolate_face_nodes(
                w_padded if self.flux_recipe == 3 else u_padded,
                dim=dim,
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=p,
                slope_limiter=slope_limiter,
                convert_to_primitives=ZS and self.flux_recipe == 2,
                primitive_fallback=w_padded if ZS and self.flux_recipe == 2 else None,
            )
            if self.flux_recipe == 1 or (self.flux_recipe == 2 and not ZS):
                w_xl[...] = self.primitives_from_conservatives(w_xl)
                w_xr[...] = self.primitives_from_conservatives(w_xr)

            # apply extra ghost cells for transverse quadrature and riemann solve
            nghost2 = p // 2 + 1 if mode == "transverse" else 1
            w_xl = self.apply_bc(
                w_xl,
                nghost2,
                primitives=True,
                fv_averages=False,
                t=t,
                face_quadrature=dim + "l",
                p=0 if mode == "transverse" else p,
            )
            w_xr = self.apply_bc(
                w_xr,
                nghost2,
                primitives=True,
                fv_averages=False,
                t=t,
                face_quadrature=dim + "r",
                p=0 if mode == "transverse" else p,
            )

            # compute fluxes from the primitive variables
            F = self.compute_numerical_fluxes(
                w_xr[crop(axis, (None, -1))],
                w_xl[crop(axis, (1, None))],
                dim=dim,
                quadrature=mode,
                p=p,
            )
            fluxes.append(F)

        # return dt and fluxes
        return dt, (fluxes[0], fluxes[1], fluxes[2])

    @partial(method_timer, cat="FiniteVolumeSolver.MOOD_loop")
    def MOOD_loop(
        self,
        t: float,
        dt: float,
        u: ArrayLike,
        fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
    ) -> Tuple[
        float, Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]
    ]:
        """
        Revise fluxes using MOOD.

        Args:
            t: Time value.
            dt: Time step size.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            fluxes: Tuple of flux arrays (F, G, H). None if the corresponding dimension
                is unused. Otherwise, is an array with shape:
                - F: (nvars, nx+1, ny, nz)
                - G: (nvars, nx, ny+1, nz)
                - H: (nvars, nx, ny, nz+1)

        Returns:
            dt: Revised time step size.
            fluxes: Tuple of revised flux arrays (F, G, H). None if the corresponding
                dimension is unused. Otherwise, is an array with shape:
                - F: (nvars, nx+1, ny, nz)
                - G: (nvars, nx, ny+1, nz)
                - H: (nvars, nx, ny, nz+1)
        """
        init_MOOD(self, u, fluxes)
        while detect_troubles(
            self,
            t=t,
            dt=dt,
            u=u,
            fluxes=fluxes,
            NAD=self.NAD,
            PAD=self.arrays["PAD"] if "PAD" in self.arrays else None,
            PAD_tol=self.PAD_tol,
            SED=self.SED,
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
    def apply_bc(
        self,
        u: ArrayLike,
        n: int,
        primitives: bool = False,
        fv_averages: bool = True,
        t: Optional[float] = None,
        face_quadrature: Optional[Literal["xl", "xr", "yl", "yr", "zl", "zr"]] = None,
        p: int = 0,
    ) -> ArrayLike:
        """
        Apply boundary conditions to an array.

        Args:
            u: Field array. Has shape (nvars, nx, ny, nz).
            n: Number of ghost cells to add on each side along each active dimension.
            primitives: Whether `arr` contains primitive variables. If False, it is
                assumed that `arr` contains conservative variables. Only used for
                Dirichlet and reflective boundary conditions.
            fv_averages: Whether to compute finite-volume averages of the Dirichlet
                function. If True, the Dirichlet function will be averaged over the
                quadrature points and `arr.ndim` is expected to be 4
                (nvar, nx, ny, nz). If False, the Dirichlet function will be evaluated
                at the quadrature points and the result will be assigned to the
                boundary slab directly, meaning `arr.ndim` is expected to be 5
                (nvar, nx, ny, nz, n_quadrature_points).
            t: Time at which boundary conditions are applied as an argument to the
                Dirichlet function. May be None if the Dirichlet function does not
                depend on time.
            face_quadrature: Optional; if provided, it specifies the face location for
                which to compute quadrature points which are used to evaluate the
                Dirichlet function. Can be one of "xl", "xr", "yl", "yr", "zl", "zr".
                If not provided, the returned qauadrature will span the interior of the
                cell.
            p: Argument for the polynomial degree of the quadrature rule which
                determines the points and weights used to evaluate the Dirichlet
                function. Defaults to 0, in which case the returned quadrature is a
                single face-centered point if `face_quadrature` is provided, or a
                single point in the center of the cell if `face_quadrature` is not
                provided.

        Returns:
            Padded array. Has shape (nvars, >= nx, >= ny, >= nz). Axes corresponding to
                inactive dimensions are not padded and maintain length 1. Axes
                corresponding to active dimensions are padded with 'n' ghost cells on
                each side, resulting in length 'n[dim] + 2 * n' along that axis.

        """
        using = self.using
        return self.bc(
            u,
            (int(using["x"]) * n, int(using["y"]) * n, int(using["z"]) * n),
            primitives=primitives,
            fv_averages=fv_averages,
            t=t,
            face_quadrature=face_quadrature,
            p=p,
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
        Interpolates a value from an array using a stencil of polynomial degree `p`.

        Args:
            u: Array to interpolate. Has shape (nvars, nx, ny, nz).
            x, y, z: Coordinates of the desired node. Can be specified as:
                - Integers or floats: Must be bounded by -1 (leftmost cell face) and 1
                    (rightmost cell face).
                - Strings: Can be "l", "r", or "c" for left, right, or center of the
                  cell, respectively.
            p: Polynomial degree of the interpolation.
            sweep_order: Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            stencil_type: Type of stencil weights to use for the interpolation.
                - "conservative-interpolation": Uses conservative interpolation weights.
                - "uniform-quadrature": Uses uniform quadrature weights.

        Returns:
            Array of interpolated node values. Has shape (nvars, <=nx, <=ny, <=nz).

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
                    # prepare stencil
                    stencil_name = f"{stencil_type}, p={p}, x={coordinates[direction]}"
                    if stencil_name not in self.arrays:
                        self.arrays.add(
                            stencil_name, weight_function(p, coordinates[direction])
                        )

                    # interpolate
                    self.interpolation_cache.add(
                        key,
                        stencil_sweep(
                            self.xp,
                            current_data,
                            stencil_weights=self.arrays[stencil_name],
                            axis=axes[direction],
                        ),
                    )
                else:
                    self.interpolation_cache.add(key, current_data)

            # update the current data
            current_data = self.interpolation_cache[key]

        return current_data

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_cell_centers")
    def interpolate_cell_centers(
        self,
        averages: ArrayLike,
        p: int,
        sweep_order: str = "xyz",
        clear_cache: bool = True,
    ) -> ArrayLike:
        """
        Interpolate cell-centered values from finite-volume cell averages.

        Args:
            averages: Array of FV cell averages. Has shape (nvars, nx, ny, nz).
            p: Polynomial degree of the interpolation for "transverse" and
                "gauss-legendre" modes.
            sweep_order: Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            clear_cache: Whether to clear the interpolation cache before performing the
                interpolation.

        Returns:
            Array of FV cell-centered values. Has shape (nvars, <=nx, <=ny, <=nz).
        """
        return self._interpolate_cell_quantity(
            averages,
            centers=True,
            p=p,
            sweep_order=sweep_order,
            clear_cache=clear_cache,
        )

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_cell_averages")
    def interpolate_cell_averages(
        self,
        centers: ArrayLike,
        p: int,
        sweep_order: str = "xyz",
        clear_cache: bool = True,
    ) -> ArrayLike:
        """
        Interpolate finite-volume cell averages from cell-centered values.

        Args:
            centers: Array of cell center values. Has shape (nvars, nx, ny, nz).
            p: Polynomial degree of the interpolation for "transverse" and
                "gauss-legendre" modes.
            sweep_order: Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            clear_cache: Whether to clear the interpolation cache before performing the
                interpolation.

        Returns:
            Array of FV cell averages. Has shape (nvars, <=nx, <=ny, <=nz).
        """
        return self._interpolate_cell_quantity(
            centers,
            centers=False,
            p=p,
            sweep_order=sweep_order,
            clear_cache=clear_cache,
        )

    def _interpolate_cell_quantity(
        self,
        u: ArrayLike,
        p: int,
        centers: bool = True,
        sweep_order: str = "xyz",
        clear_cache: bool = True,
    ) -> ArrayLike:
        """
        Interpolate a cell quantity (cell centers or cell averages) from finite-volume
        cell averages or cell centers.
        Args:
            u: Array of cell averages or cell centers. Has shape (nvars, nx, ny, nz).
            p: Polynomial degree of the interpolation for "transverse" and
                "gauss-legendre" modes.
            centers: Whether to interpolate cell centers (True) or cell averages
                (False).
            sweep_order: Order of the direction of the interpolation sweeps. Any
                combination of "x", "y", and "z".
            clear_cache: Whether to clear the interpolation cache before performing the
                interpolation.

        Returns:
            Array of interpolated cell quantity. Has shape (nvars, <=nx, <=ny, <=nz)
        """
        if clear_cache:
            self.interpolation_cache.clear()

        cell_quantity = self.interpolate(
            u,
            p=p,
            sweep_order=sweep_order,
            stencil_type=(
                "conservative-interpolation" if centers else "uniform-quadrature"
            ),
        )
        return cell_quantity

    @partial(method_timer, cat="FiniteVolumeSolver.interpolate_face_nodes")
    def interpolate_face_nodes(
        self,
        averages: ArrayLike,
        dim: Literal["x", "y", "z"],
        interpolation_scheme: Optional[Literal["transverse", "gauss-legendre"]] = None,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        p: Optional[int] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
        convert_to_primitives: bool = False,
        primitive_fallback: Optional[ArrayLike] = None,
        clear_cache: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Interpolate face nodes from finite-volume cell averages.

        Args:
            averages: Array of FV cell averages. Has shape (nvars, nx, ny, nz).
            dim: Face dimension to interpolate. Can be "x", "y", or "z".
            interpolation_scheme: Mode of interpolation. Possible values:
                - None: Only valid when `limiting_scheme` is "muscl".
                - "transverse": Interpolate nodes at the cell face centers.
                - "gauss-legendre": Interpolate nodes at the Gauss-Legendre quadrature
                    points.
            limiting_scheme: Slope limiting scheme. Possible values:
                - None: No slope limiting.
                - "muscl": Use the MUSCL scheme.
                - "zhang-shu": Use Zhang and Shu's maximum-principle-satisfying slope
                    limiter.
            p: Polynomial degree of the interpolation schemes. For "muscl" slope
                limiting, p is ignored and assumed to be 1.
            slope_limiter: Additional option for the MUSCL scheme. Possible values:
                - "minmod": Minmod limiter.
                - "moncen": Moncen limiter.
            convert_to_primitives: Option for the Zhang-Shu limiter. If True, the
                interpolated face nodes are transformed to primitive variables. If
                False, the face nodes are not transformed and are returned as is.
            primitive_fallback: Optional fallback values for the Zhang-Shu limiter. If
                provided, it should be an array of primitive variables with shape
                (nvars, nx, ny, nz). If not provided, the 'averages' array is used as
                the fallback values for the Zhang-Shu limiter.
            clear_cache: Whether to clear the interpolation cache before performing the
                interpolation.
        Returns:
            Tuple of node arrays for the left and right faces along the specified
            dimension ('dim'). Each face node has shape
            (nvars, <=nx, <=ny, <=nz, ninterpolations). If `interpolation_scheme` is
            "gauss-legendre", `ninterpolations` is the number of Gauss-Legendre
            interpolation points for a degree `p` reconstruction. If
            `interpolation_scheme` is "face-centers", `ninterpolations` is 1.
        """
        if clear_cache:
            self.interpolation_cache.clear()
        sweep_orders = {"x": "yzx", "y": "zxy", "z": "xyz"}

        # MUSCL limiter escape
        if limiting_scheme == "muscl":
            return muscl(
                self.xp,
                averages,
                dim=dim,
                slope_limiter=minmod if slope_limiter == "minmod" else moncen,
            )
        elif p is None:
            raise ValueError(
                "Polynomial degree 'p' must be specified for interpolation schemes "
                "other than 'muscl'."
            )

        # Zhang-Shu limiter escape
        if limiting_scheme == "zhang-shu":
            return zhang_shu_limiter(
                self,
                averages,
                dim=dim,
                interpolation_scheme=cast(
                    Literal["transverse", "gauss-legendre"], interpolation_scheme
                ),
                p=cast(int, p),
                SED=self.SED,
                convert_to_primitives=convert_to_primitives,
                primitive_fallback=primitive_fallback,
            )

        # Interpolate unlimited face nodes
        faces = []
        for pos in "lr":
            nodes = []
            if not self.using[dim]:
                raise ValueError(f"Dimension {dim} is not used in the mesh.")
            if (
                interpolation_scheme in ("gauss-legendre", "transverse")
                and limiting_scheme is None
            ):
                px = p if self.using_xdim else 0
                py = p if self.using_ydim else 0
                pz = p if self.using_zdim else 0
                if interpolation_scheme == "transverse":
                    xp, yp, zp, _ = gauss_legendre_for_finite_volume(0, 0, 0)
                else:
                    xp, yp, zp, _ = gauss_legendre_for_finite_volume(
                        0 if dim == "x" else px,
                        0 if dim == "y" else py,
                        0 if dim == "z" else pz,
                    )
                coords = {"x": 2 * xp, "y": 2 * yp, "z": 2 * zp}  # scale to [-1, 1]
                coords[dim] += -1 if pos == "l" else 1
                for xpi, ypi, zpi in zip(coords["x"], coords["y"], coords["z"]):
                    nodes.append(
                        self.interpolate(
                            averages,
                            xpi.item(),
                            ypi.item(),
                            zpi.item(),
                            p=cast(int, p),
                            sweep_order=sweep_orders[dim],
                            stencil_type="conservative-interpolation",
                        )
                    )
                faces.append(self.xp.stack(nodes, axis=-1))
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
            node_values: Array of node values. Has shape (nvars, nx, ny, nz, 1).
            dim: Direction of the flux integral: "x", "y", or "z".
            p: Polynomial degree of the interpolation.
            clear_cache: Whether to clear the interpolation cache before performing the
                quadrature.

        Returns:
            Array of flux integrals. Has shape (nvars, <=nx, <=ny, <=nz).
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
            node_values: Array of node values. Has shape
                (nvars, nx, ny, nz, ninterpolations).
            dim: Direction of the flux integral: "x", "y", or "z".
            p: Polynomial degree of the interpolation.

        Returns:
            Array of flux integrals. Has shape (nvars, <=nx, <=ny, <=nz).
        """
        px, py, pz = self.px, self.py, self.pz
        match dim:
            case "x":
                _, _, _, w = gauss_legendre_for_finite_volume(0, py, pz)
            case "y":
                _, _, _, w = gauss_legendre_for_finite_volume(px, 0, pz)
            case "z":
                _, _, _, w = gauss_legendre_for_finite_volume(px, py, 0)
        return self.xp.sum(node_values * w, axis=4)

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
    ) -> ArrayLike:
        """
        Compute the numerical fluxes.

        Args:
            left_nodes: Array of primitive, pointwise variable nodes to the left of the
                discontinuity. Has shape (nvars, nx, ny, nz, ninterpolations).
            right_nodes: Array of primitive, pointwise variable nodes to the right of
                the discontinuity. Has shape (nvars, nx, ny, nz, ninterpolations).
            dim: Direction of the flux integral: "x", "y", or "z".
            quadrature: Mode of the numerical flux computation. Possible values:
                - "transverse": Compute the flux integral using a transverse
                    quadrature.
                - "gauss-legendre": Compute the flux integral using Gauss-Legendre
                    quadrature.
            p: Polynomial degree of the interpolation.
            clear_cache: Whether to clear the interpolation cache before performing the
                quadrature.
            crop: Whether to crop the numerical fluxes to the shape of the FV mesh.

        Returns:
            Array of numerical fluxes. Has shape (nvars, <=nx, <=ny, <=nz).
            If `crop` is True, the shape is cropped to the shape of the FV mesh.
            Otherwise, it retains the full shape of the numerical fluxes.
        """
        nodal_fluxes = self.riemann_solver(left_nodes, right_nodes, dim)

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
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            F: Array of fluxes in the x-direction. Has shape
                (nvars, nx + 1, ny, nz).
            G: Array of fluxes in the y-direction. Has shape
                (nvars, nx, ny + 1, nz).
            H: Array of fluxes in the z-direction. Has shape
                (nvars, nx, ny, nz + 1).

        Returns:
            dudt: Right-hand side of the ODE at (t, y) as an array. Has shape
                (nvars, nx, ny, nz).
        """
        dudt = self.xp.zeros_like(u)
        for axis, dim, _F in zip([1, 2, 3], ["x", "y", "z"], [F, G, H]):
            if self.using[dim]:
                dudt += -(1 / getattr(self.mesh, "h" + dim)) * (
                    cast(ArrayLike, _F)[crop(axis, (1, None))]
                    - cast(ArrayLike, _F)[crop(axis, (None, -1))]
                )
        return dudt

    @partial(method_timer, cat="FiniteVolumeSolver.snapshot")
    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        p = self.p
        ucc = self.interpolate_cell_centers(
            self.apply_bc(self.arrays["u"], -(-p // 2), t=self.t, p=p), p=p
        )
        wcc = self.primitives_from_conservatives(ucc)
        if self.lazy_primitives:
            w = self.primitives_from_conservatives(self.arrays["u"])
        else:
            w = self.interpolate_cell_averages(
                self.apply_bc(
                    wcc, -(-p // 2), primitives=True, fv_averages=False, t=self.t, p=0
                ),
                p=p,
            )

        # update the arrays
        self.arrays["ucc"][...] = ucc
        self.arrays["wcc"][...] = wcc
        self.arrays["w"][...] = w

        # store the snapshot
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "w": self.arrays.get_numpy_copy("w"),
            "ucc": self.arrays.get_numpy_copy("ucc"),
            "wcc": self.arrays.get_numpy_copy("wcc"),
        }
        self.snapshots.log(self.t, data)

    @partial(method_timer, cat="FiniteVolumeSolver.minisnapshot")
    def minisnapshot(self):
        super().minisnapshot()
        self.minisnapshots["MOOD_iters"].append(self.MOOD_iter_count)

    @partial(method_timer, cat="!FiniteVolumeSolver.run")
    def run(self, *args, q_max=3, **kwargs):
        """
        Solve the conservation law using a Runge-Kutta method whose order matches the
        chosen polynomial degree for the spatial discretization, up to RK4.

        Args:
            *args: Arguments to pass to the Runge-Kutta method.
            q_max: Maximum degree of the Runge-Kutta method to use.
            **kwargs: Keyword arguments to pass to the Runge-Kutta method.
        """
        q = min(self.p, q_max)
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
