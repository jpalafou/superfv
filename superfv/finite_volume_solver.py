import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast

import numpy as np

from .boundary_conditions import BoundaryConditions, DirichletBC, Field
from .explicit_ODE_solver import ExplicitODESolver
from .fv import fv_average
from .slope_limiting import minmod, moncen, muscl
from .slope_limiting.MOOD import detect_troubles, init_MOOD, revise_fluxes
from .slope_limiting.zhang_and_shu import zhang_shu_limiter
from .stencil import (
    conservative_interpolation_weights,
    get_gauss_legendre_face_nodes,
    get_gauss_legendre_face_weights,
    stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.array_management import (
    CUPY_AVAILABLE,
    ArrayLike,
    ArrayManager,
    ArraySlicer,
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
    def define_vars(self) -> ArraySlicer:
        """
        Define the names of the solver variables.

        Returns:
            ArraySlicer: ArraySlicer object.
        """
        pass

    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Convert primitive variables to conservative variables.

        Args:
            w (ArrayLike): Primitive variables.

        Returns:
            ArrayLike: Conservative variables.
        """
        raise NotImplementedError("Conservative variables not implemented.")

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u (ArrayLike): Conservative variables.

        Returns:
            ArrayLike: Primitive variables.
        """
        raise NotImplementedError("Primitive variables not implemented.")

    def dummy_riemann_solver(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        primitives: bool = True,
    ) -> ArrayLike:
        """
        Dummy Riemann solver to give an example of the required signature.

        Args:
            wl (ArrayLike): Left state primitive variables. Has shape
                (nvars, nx, ny, nz, ...).
            wr (ArrayLike): Right state primitive variables. Has shape
                (nvars, nx, ny, nz, ...).
            dim (str): Direction of the flux integral. Can be "x", "y", or "z".
            primitives (bool): Whether the input variables are primitive. If False, the
                input variables are conservative.

        Returns:
            ArrayLike: Numerical fluxes. Has shape (nvars, nx, ny, nz, ...).
        """
        raise NotImplementedError("Riemann solver not implemented.")

    @abstractmethod
    @partial(method_timer, cat="?.compute_dt")
    def compute_dt(self, w: ArrayLike) -> float:
        """
        Compute the time-step size.

        Args:
            w (ArrayLike): Primitive variables. Has shape (nvars, nx, ny, nz, ...).

        Returns:
            float: Time-step size.
        """
        pass

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
            ic (Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike]):
                Initial condition function of pointwise, primitive variables. The
                function must accept the following arguments:
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
        """
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
        self._init_array_manager(cupy)
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
        flux_recipe: Literal[1, 2, 3],
        lazy_primitives: bool,
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
        self.axes = tuple("xyz".index(dim) + 1 for dim in self.dims)
        self.ndim = len(self.dims)
        self.p = p
        self.CFL = CFL
        self.interpolation_scheme = interpolation_scheme
        self.flux_recipe = flux_recipe
        self.lazy_primitives = lazy_primitives

        # Validation
        if self.flux_recipe not in (1, 2, 3):
            raise ValueError(
                "flux_recipe must be 1, 2, or 3. See the documentation for details."
            )
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
        slab_thickness = -2 * (-p // 2) + 1

        def _get_slab_limits(lim, spacing, thickness, pos=None):
            if pos is None:
                return (lim[0] - spacing * thickness, lim[1] + spacing * thickness)
            if pos == "l":
                return (lim[0] - spacing * thickness, lim[0])
            if pos == "r":
                return (lim[1], lim[1] + spacing * thickness)

        self.slab_meshes = {}
        for dim in "xyz":
            self.slab_meshes[dim] = (
                tuple(
                    (
                        _get_uniform_3D_mesh(
                            *(
                                _get_slab_limits(
                                    getattr(self, _dim + "lim"),
                                    self.h[_dim],
                                    slab_thickness if self.using[_dim] else 0,
                                    pos if dim == _dim else None,
                                )
                                for _dim in "xyz"
                            ),
                            *(
                                (
                                    {_dim: slab_thickness}.get(
                                        dim, self.n[_dim] + 2 * slab_thickness
                                    )
                                    if self.using[_dim]
                                    else 1
                                )
                                for _dim in "xyz"
                            ),
                        )
                        for pos in "lr"
                    )
                )
                if self.using[dim]
                else None
            )

    def _init_ic(
        self,
        ic: Callable[[ArraySlicer, ArrayLike, ArrayLike, ArrayLike], ArrayLike],
        ic_passives: Optional[
            Dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]]
        ],
    ):
        # Define the following attributes:
        self.active_vars: Set[str]
        self.array_slicer: ArraySlicer
        self.callable_ic: Callable[
            [ArraySlicer, ArrayLike, ArrayLike, ArrayLike, Optional[float]], ArrayLike
        ]
        self.n_active_vars: int
        self.n_passive_vars: int
        self.nvars: int
        self.passive_vars: Set[str]
        self.user_defined_passive_vars: Set[str]
        self.vars: Set[str]

        # Define active and passive variables
        self.array_slicer = self.define_vars()
        if "passives" in self.array_slicer.group_names:
            self.passive_vars = set(self.array_slicer.groups["passives"])
            self.active_vars = self.array_slicer.var_names - self.passive_vars
        else:
            self.active_vars = self.array_slicer.var_names
            self.passive_vars = set()
        self.array_slicer.create_var_group("actives", tuple(self.active_vars))

        # Count active and passive variables
        self.n_active_vars = len(
            {self.array_slicer.variables[v] for v in self.active_vars}
        )
        self.n_passive_vars = len(
            {self.array_slicer.variables[v] for v in self.passive_vars}
        )

        # Include user-defined passive variables
        self.user_defined_passive_vars = set()
        if ic_passives is not None:
            starting_passive_idx = len(self.array_slicer.idxs)
            for i, v in enumerate(ic_passives.keys()):
                if v in self.active_vars | self.passive_vars:
                    raise ValueError("Variable already defined.")
                self.array_slicer.add_var(v, starting_passive_idx + i)
            if "passives" in self.array_slicer.group_names:
                self.array_slicer.add_to_var_group(
                    "passives", tuple(ic_passives.keys())
                )
            else:
                self.array_slicer.create_var_group(
                    "passives", tuple(ic_passives.keys())
                )
            self.n_user_defined_passive_vars = len(ic_passives)
            self.n_passive_vars += self.n_user_defined_passive_vars
            self.passive_vars |= set(ic_passives.keys())

            self.user_defined_passive_vars = set(ic_passives.keys())
            self.array_slicer.create_var_group(
                "user_defined_passives", tuple(self.user_defined_passive_vars)
            )

        # Define all variables
        self.nvars = self.n_active_vars + self.n_passive_vars
        self.vars = self.active_vars | self.passive_vars

        # Define callable initial condition function
        def callable_ic(array_slicer, x, y, z, t=None):
            return ic(array_slicer, x, y, z)

        if ic_passives is not None:

            def passive_wrapper(f):
                def wrapped_ic_func(array_slicer, x, y, z, t=None):
                    actives = f(array_slicer, x, y, z)
                    out = np.concatenate(
                        [
                            actives,
                            np.empty(
                                (self.n_user_defined_passive_vars, *actives[0].shape)
                            ),
                        ]
                    )
                    for pv, pf in ic_passives.items():
                        out[array_slicer(pv)] = pf(x, y, z)
                    return out

                return wrapped_ic_func

        # Apply wrappers
        if ic_passives is None:
            self.callable_ic = self.fv_average_wrapper(
                self.conservatives_wrapper(callable_ic)
            )
        else:
            self.callable_ic = self.fv_average_wrapper(
                self.conservatives_wrapper(passive_wrapper(callable_ic))
            )

        # Test array slicing
        test_arr = np.arange(self.nvars)
        selected_actives = set(
            cast(
                List[str],
                test_arr[self.array_slicer("actives", keepdims=True)].tolist(),
            )
        )
        selected_passives = (
            set(
                cast(
                    List[str],
                    test_arr[self.array_slicer("passives", keepdims=True)].tolist(),
                )
            )
            if "passives" in self.array_slicer.group_names
            else None
        )
        all_selected_vars = set(test_arr.tolist())
        if (
            self.n_passive_vars > 0
            and not selected_actives | cast(Set[str], selected_passives)
            == all_selected_vars
        ):
            raise ValueError(
                "The intersection of active and passive variables must be the set of all variables."
            )
        elif (
            self.n_passive_vars == 0
            and not selected_actives == selected_actives == all_selected_vars
        ):
            raise ValueError("Active variables must be the set of all variables.")

        # Initialize the ODE solver with the initial condition array
        ic_arr = self.callable_ic(self.array_slicer, self.X, self.Y, self.Z, 0.0)
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
            array_slicer: ArraySlicer,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
        ) -> ArrayLike:
            return self.conservatives_from_primitives(f(array_slicer, x, y, z, t))

        return wrapped_func

    def fv_average_wrapper(
        self,
        f: Field,
    ) -> Field:
        """
        Wrapper to convert output of a function from pointwise to cell-averaged.
        """

        def wrapped_func(
            array_slicer: ArraySlicer,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
        ) -> ArrayLike:
            return fv_average(
                lambda x, y, z: f(array_slicer, x, y, z, t),
                x,
                y,
                z,
                h=(self.hx, self.hy, self.hz),
                p=(
                    self.p if self.using_xdim else 0,
                    self.p if self.using_ydim else 0,
                    self.p if self.using_zdim else 0,
                ),
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
                    _f[i] = lambda array_slicer, x, y, z, t: self.callable_ic(
                        array_slicer, x, y, z, 0.0
                    )
            ic_configed_bc[dim] = (_bc[0], _bc[1])
            ic_configed_dirichlet[dim] = (_f[0], _f[1])

        # initialize boundary conditions
        bcx = ic_configed_bc["x"]
        bcy = ic_configed_bc["y"]
        bcz = ic_configed_bc["z"]
        x_dirichlet = ic_configed_dirichlet["x"]
        y_dirichlet = ic_configed_dirichlet["y"]
        z_dirichlet = ic_configed_dirichlet["z"]
        self.bc = BoundaryConditions(
            self.array_slicer,
            bcx,
            bcy,
            bcz,
            x_dirichlet,
            y_dirichlet,
            z_dirichlet,
            self.slab_meshes["x"],
            self.slab_meshes["y"],
            self.slab_meshes["z"],
            self.conservatives_wrapper,
            self.fv_average_wrapper,
        )

    def _init_snapshots(self):
        self.minisnapshots["MOOD_iters"] = []

    def _init_array_manager(self, cupy: bool):
        self.cupy = False
        self.interpolation_cache = ArrayManager()
        self.MOOD_cache = ArrayManager()
        self.ZS_cache = ArrayManager()
        if cupy and CUPY_AVAILABLE:
            self.cupy = True
            self.arrays.transfer_to_device("gpu")
            self.interpolation_cache.transfer_to_device("gpu")
            self.MOOD_cache.transfer_to_device("gpu")
            self.ZS_cache.transfer_to_device("gpu")
        elif cupy:
            warnings.warn("CuPy is not available. Using NumPy instead.")

        # initialize flux arrays
        nvars, nx, ny, nz = self.nvars, self.nx, self.ny, self.nz
        self.arrays.add("F", np.zeros((nvars, nx + 1, ny, nz)))
        self.arrays.add("G", np.zeros((nvars, nx, ny + 1, nz)))
        self.arrays.add("H", np.zeros((nvars, nx, ny, nz + 1)))

        # initialize primitive arrays
        self.arrays.add("w", np.empty((nvars, nx, ny, nz)))

        # initialize numpy namespace
        self.xp = xp if self.cupy else np

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
        # Initial setup for a priori slope limiting
        self.a_priori_slope_limiting_scheme = None
        self.a_priori_slope_limiter = None
        self.ZS_adaptive_timestep = False

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
        self.MOOD_iter_count = 0  # Initialize MOOD iteration count

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
            self.limiting_vars = self.active_vars
        elif limiting_vars == "all":
            self.limiting_vars = self.array_slicer.var_names
        elif isinstance(limiting_vars, tuple):
            self.limiting_vars = set(limiting_vars)
        else:
            raise ValueError(
                "limiting_vars must be a tuple of variable names, 'actives', or 'all'."
            )

        # Assign limiting variables
        self.array_slicer.create_var_group("limiting_vars", tuple(self.limiting_vars))

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
            _PAD_arr = np.array([[-np.inf, np.inf]] * self.nvars)
            for v, (lb, ub) in PAD.items():
                _PAD_arr[self.array_slicer(v)] = [lb, ub]
            _PAD_arr = _PAD_arr.reshape(self.nvars, 1, 1, 1, 2)
            self.arrays.add("PAD", _PAD_arr)  # Register PAD array

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
            fluxes (Tuple[Optional[ArrayLike], ...]): Tuple of flux arrays or None if
                the corresponding dimension is unused. The fluxes have the following
                shapes if not None:
                - F: (nvars, nx + 1, ny, nz)
                - G: (nvars, nx, ny + 1, nz)
                - H: (nvars, nx, ny, nz + 1)
        """
        _slc = self.array_slicer
        p = 1 if limiting_scheme == "muscl" else cast(int, p)

        # apply boundary conditions
        u_padded = self.apply_bc(u, -(-p // 2) + 1, t)

        # assign primitive averages to 'w'
        if self.lazy_primitives:
            w_padded = self.primitives_from_conservatives(u_padded)
        else:
            w = self.interpolate_cell_centers(
                self.apply_bc(u, -2 * (-p // 2), t), interpolation_scheme=mode, p=p
            )
            w[...] = self.primitives_from_conservatives(w)
            w = self.interpolate(w, p=p, stencil_type="uniform-quadrature")
            w_padded = self.apply_bc(w, -(-p // 2) + 1, t, conservatives=False)

        # compute dt
        dt = self.compute_dt(w_padded)

        # compute fluxes
        fluxes: List[Optional[ArrayLike]] = []
        for dim in ["x", "y", "z"]:
            if not self.using[dim]:
                fluxes.append(None)
                continue
            w_xl, w_xr = self.interpolate_face_nodes(
                w_padded if self.flux_recipe == 3 else u_padded,
                dim=dim,
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=p,
                slope_limiter=slope_limiter,
                convert_to_primitives=limiting_scheme == "zhang-shu"
                and self.flux_recipe == 2,
                primitive_fallback=(
                    w_padded
                    if limiting_scheme == "zhang-shu" and self.flux_recipe == 2
                    else None
                ),
            )
            if self.flux_recipe == 1 or (
                self.flux_recipe == 2 and limiting_scheme != "zhang-shu"
            ):
                w_xl = self.primitives_from_conservatives(w_xl)
                w_xr = self.primitives_from_conservatives(w_xr)
            F = self.compute_numerical_fluxes(
                w_xr[_slc(**{dim: (None, -1)})],
                w_xl[_slc(**{dim: (1, None)})],
                dim=dim,
                quadrature=mode,
                p=p,
                primitive=True,
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
            t (float): Time value.
            dt (float): Time-step size.
            u (ArrayLike): Solution value. Has shape (nvars, nx, ny, nz).
            fluxes (Tuple[Optional[ArrayLike], ...]): Tuple of flux arrays or None if
                the corresponding dimension is unused. The fluxes have the following
                shapes if not None:
                - F: (nvars, nx + 1, ny, nz)
                - G: (nvars, nx, ny + 1, nz)
                - H: (nvars, nx, ny, nz + 1)

        Returns:
            Tuple[float, Tuple[Optional[ArrayLike]], ...]: Tuple composed of:
                - The revised time step.
                - The revised fluxes (F, G, H). None if the corresponding dimension is
                unused. Otherwise, is an array with shape:
                    - F: (nvars, nx+1, ny, nz, ...)
                    - G: (nvars, nx, ny+1, nz, ...)
                    - H: (nvars, nx, ny, nz+1, ...)
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
        arr: ArrayLike,
        n_ghost_cells: int,
        t: Optional[float] = None,
        conservatives: bool = True,
        averages: bool = True,
    ) -> ArrayLike:
        """
        Apply boundary conditions to an array by padding it with ghost cells along the
        active dimensions.

        Args:
            arr (ArrayLike): Field array. Has shape (nvars, nx, ny, nz).
            n_ghost_cells (int): Number of ghost cells to apply boundary conditions to.
            t (Optional[float]): Time at which boundary conditions are applied.
            conservatives (bool): Whether the input is conservative variables. If
                False, the input is assumed to be primitive variables.
            averages (bool): Whether the input is finite volume averages. If False, the
                input is assumed to be pointwise values.

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
            t=t,
            conservatives=conservatives,
            averages=averages,
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
        convert_to_primitives: bool = False,
        primitive_fallback: Optional[ArrayLike] = None,
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
            convert_to_primitives (bool): Convert nodes to primitive variables before
                returning. Only used if `limiting_scheme` is "zhang-shu".
            primitive_fallback (Optional[ArrayLike]): Argument of the Zhang-Shu
                limiter. It gives the fallback values used by the limiter if
            clear_cache (bool): Whether to clear the interpolation cache before
                performing the interpolation.
        Returns:
            Tuple[ArrayLike, ArrayLike]: Two node arrays. The first element is the left
                face nodes, and the second element is the right face nodes. Each face
                node has shape (nvars, <=nx, <=ny, <=nz, ninterpolations), unless the
                dimension is not used, in which case the face node is None.
                ninterpolations is the number of Gauss-Legendre interpolation points
                for a degree `p` reconstruction if `interpolation_scheme` is
                "gauss-legendre". If `interpolation_scheme` is "face-centers",
                ninterpolations is 1.
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
                faces.append(self.xp.stack(nodes, axis=-1))
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
        weights_name = f"gauss-legendre-weights-{dim}-{p}"
        if weights_name not in self.arrays:
            self.arrays.add(
                weights_name, get_gauss_legendre_face_weights(self.dims, dim, p)
            )
        weights = self.arrays[weights_name]
        return self.xp.sum(node_values * weights.reshape(1, 1, 1, 1, -1), axis=4)

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
        primitive: bool = True,
    ) -> ArrayLike:
        """
        Compute the numerical fluxes.

        Args:
            left_nodes (ArrayLike): Value of primitive variable nodes to the
                left of the discontinuity. Has shape
                (nvars, nx, ny, nz, ninterpolations).
            right_nodes (ArrayLike): Value of primitive variable nodes to the
                right of the discontinuity. Has shape
                (nvars, nx, ny, nz, ninterpolations).
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
            primitive (bool): Whether the input nodes are primitive variables. If
                False, the input nodes are assumed to be conservative variables.
        """
        nodal_fluxes = self.riemann_solver(left_nodes, right_nodes, dim, primitive)

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
        dydt = self.xp.zeros_like(u)
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
        # compute the average of the primitive variables
        if self.lazy_primitives:
            self.arrays["w"] = self.primitives_from_conservatives(self.arrays["u"])
        else:
            u_padded = self.apply_bc(
                self.arrays["u"], n_ghost_cells=2 * -(-self.p // 2), t=self.t
            )
            u_centers = self.interpolate_cell_centers(
                u_padded, self.interpolation_scheme, self.p
            )
            w = self.primitives_from_conservatives(u_centers)
            self.arrays["w"] = self.interpolate(
                w, p=self.p, stencil_type="uniform-quadrature"
            )

        # store the snapshot
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "w": self.arrays.get_numpy_copy("w"),
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
            q_max (int): Maximum degree of the Runge-Kutta method to use.
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
