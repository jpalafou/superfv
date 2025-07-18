import warnings
from abc import ABC, abstractmethod
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from . import fv
from .boundary_conditions import DirichletBC, Field, apply_bc
from .explicit_ODE_solver import ExplicitODESolver
from .fv import DIM_TO_AXIS, fv_average, gauss_legendre_for_finite_volume
from .initial_conditions import _uninitialized
from .mesh import UniformFVMesh
from .slope_limiting import minmod, moncen, muscl
from .slope_limiting.MOOD import detect_troubles, init_MOOD, revise_fluxes
from .slope_limiting.zhang_and_shu import (
    compute_theta,
    zhang_shu_limiter,
    zhang_shu_operator,
)
from .stencil import (
    conservative_interpolation_weights,
    stencil_sweep,
    uniform_quadrature_weights,
)
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager, xp
from .tools.dummy_module import DummyModule
from .tools.slicing import VariableIndexMap, crop, crop_to_center, merge_slices
from .tools.timer import MethodTimer
from .visualization import plot_1d_slice, plot_2d_slice

Directions = Literal["x", "y", "z"]
Faces = Literal["xl", "xr", "yl", "yr", "zl", "zr"]


class FieldFunction(Protocol):
    def __call__(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float] = None,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...


class PassiveFieldFunction(Protocol):
    def __call__(
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float] = None,
        *,
        xp: ModuleType,
    ) -> ArrayLike: ...


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

    @abstractmethod
    def conservatives_to_primitives(self, u: ArrayLike, w: ArrayLike):
        """
        Convert conservative variables to primitive variables and write them to `w`.
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
        out: ArrayLike,
    ):
        """
        Dummy Riemann solver to give an example of the required signature.

        Args:
            wl: Array of primitive variables to the left of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            wr: Array of primitive variables to the right of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            dim: Direction in which the Riemann problem is solved: "x", "y", or "z".
            out: Output array to store the numerical fluxes. Has shape
                (nvars, nx, ny, nz, ...).
        """
        raise NotImplementedError("Riemann solver not implemented.")

    @abstractmethod
    def compute_dt(self, t: float, u: ArrayLike) -> float:
        """
        Compute the time-step size.

        Args:
            t: Time value.
            u: Array of finite volume averaged conservative variables. Has shape
                (nvars, nx, ny, nz).

        Returns:
            Time-step size.
        """
        pass

    @abstractmethod
    def log_quantity(self, u: ArrayLike, t: float) -> Dict[str, Any]:
        """
        Log a quantity at the end of each time step.

        Args:
            u: Array of finite volume averaged conservative variables. Has shape
                (nvars, nx, ny, nz).
            t: Time value.

        Returns:
            Dictionary of logged quantities.
        """
        pass

    def __init__(
        self,
        ic: FieldFunction,
        ic_passives: Optional[Dict[str, PassiveFieldFunction]] = None,
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
        adaptive_dt: bool = True,
        max_dt_revisions: int = 8,
        MOOD: bool = False,
        max_MOOD_iters: int = 1,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "all",
        NAD: Optional[float] = None,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        PAD_tol: float = 1e-15,
        SED: bool = True,
        cupy: bool = False,
        log_every_step: bool = True,
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
            adaptive_dt: Option for the Zhang and Shu limiter; Whether to iteratively
                halve the timestep size if the proposed solution fails PAD.
            max_dt_revisions: Option for the Zhang and Shu limiter; The maximum number
                of timestep size revisions that may be attempted in an update step
                if `adaptive_dt=True`. Defaults to 8.
            MOOD: Whether to use MOOD for a posteriori flux revision. Ignored if
                `ZS=True` and `adaptive_timestepping=True`.
            max_MOOD_iters: Option for the MOOD limiter; The maximum number of MOOD
                iterations that may be performed in an update step. Defaults to 1.
            limiting_vars: Specifies which variables are subject to slope limiting.
                - "all": All variables are subject to slope limiting.
                - "actives": Only active variables are subject to slope limiting.
                - Tuple[str, ...]: A tuple of variable names that are subject to slope
                    limiting. Must be defined in `self.define_vars()`.
                For the Zhang-Shu limiter, all variables are always limited, but
                `limiting_vars` determines which variables are checked for PAD when
                using adaptive timestepping.
            NAD: The NAD tolerance. If None, NAD is not checked.
            PAD: Dict of `limiting_vars` and their corresponding PAD tolerances as a
                tuple: (lower_bound, upper_bound). Any variable or bound not provided
                in `PAD` is given a lower and upper bound of `-np.inf` and `np.inf`
                respectively.
            PAD_tol: Tolerance for the PAD check as an absolute value from the minimum
                and maximum values of the variable.
            SED: Whether to use smooth extrema detection for slope limiting.
            cupy: Whether to use CuPy for array operations.
            log_every_step: Whether to call `log_quantity` at the end of each timestep.
        """
        self._init_array_management(cupy)
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, p, CFL)
        self._init_spatial_discretization(
            interpolation_scheme, flux_recipe, lazy_primitives
        )
        self._init_ic(ic, ic_passives)
        self._init_bc(bcx, bcy, bcz, x_dirichlet, y_dirichlet, z_dirichlet)
        self._init_snapshots(log_every_step)
        self._init_array_allocation()
        self._init_riemann_solver(riemann_solver)
        self._init_slope_limiting(
            MUSCL,
            ZS,
            adaptive_dt,
            max_dt_revisions,
            MOOD,
            max_MOOD_iters,
            limiting_vars,
            NAD,
            PAD,
            PAD_tol,
            SED,
        )

    def _init_array_management(self, cupy: bool):
        # init cupy boolean and numpy namespace
        self.cupy = False
        if cupy and CUPY_AVAILABLE:
            self.cupy = True
        elif cupy:
            warnings.warn("CuPy is not available. Using NumPy instead.")
        self.xp = xp if self.cupy else np

        # init array manager
        self.arrays = ArrayManager()
        if self.cupy:
            self.arrays.transfer_to_device("gpu")

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
        if p < 0:
            raise ValueError("The polynomial degree must be non-negative.")
        if CFL <= 0:
            raise ValueError("The CFL number must be positive.")

        # init mesh object
        self.mesh = UniformFVMesh(
            nx=nx,
            ny=ny,
            nz=nz,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            slab_depth=max(-(-p // 2), 3),
            array_manager=self.arrays,
        )

        # assign attributes
        self.p: int = p
        self.CFL: float = CFL

        # interior workspace view
        self.interior = merge_slices(
            *[
                crop(DIM_TO_AXIS[dim], (self.mesh.slab_depth, -self.mesh.slab_depth))
                for dim in self.mesh.active_dims
            ]
        )

        # interior workspace view for fluxes
        self.flux_interior = {
            dim: (
                merge_slices(
                    *[
                        crop(
                            DIM_TO_AXIS[d],
                            (self.mesh.slab_depth, -self.mesh.slab_depth),
                        )
                        for d in self.mesh.active_dims
                        if d != dim
                    ],
                )
                if self.mesh.ndim > 1
                else slice(None)
            )
            for dim in self.mesh.active_dims
        }

    def _init_spatial_discretization(
        self,
        interpolation_scheme: Literal["gauss-legendre", "transverse"],
        flux_recipe: Literal[1, 2, 3],
        lazy_primitives: bool,
    ):
        # assign flux recipe and interpolation scheme
        if flux_recipe not in (1, 2, 3):
            raise ValueError(
                "flux_recipe must be 1, 2, or 3. See the documentation for details."
            )
        if interpolation_scheme == "gauss-legendre" and self.mesh.ndim == 1:
            raise ValueError(
                "Gauss-Legendre interpolation scheme is not supported in 1D."
            )
        self.interpolation_scheme = interpolation_scheme
        self.GL = interpolation_scheme == "gauss-legendre"
        self.interpolation_func = (
            self._interpolate_GaussLegendre_nodes
            if self.GL
            else self._interpolate_transverse_nodes
        )
        self.integration_func = (
            self._integrate_trivial_nodes
            if self.mesh.ndim == 1
            else (
                self._integrate_GaussLegendre_nodes
                if self.GL
                else self._integrate_transverse_nodes
            )
        )
        self.flux_recipe = flux_recipe
        self.lazy_primitives = lazy_primitives

    def _interpolate_GaussLegendre_nodes(self, u, face_dim, p, buffer, out):
        fv.interpolate_GaussLegendre_nodes(
            self.xp, u, face_dim, self.mesh.active_dims, p, buffer, out
        )

    def _integrate_GaussLegendre_nodes(self, f, face_dim, p, buffer, out):
        fv.integrate_GaussLegendre_nodes(
            self.xp, f, face_dim, self.mesh.active_dims, p, out[..., 0]
        )

    def _interpolate_transverse_nodes(self, u, face_dim, p, buffer, out):
        fv.interpolate_face_centers(
            self.xp, u, face_dim, self.mesh.active_dims, p, buffer, out
        )

    def _integrate_transverse_nodes(self, f, face_dim, p, buffer, out):
        fv.transversely_integrate_nodes(
            self.xp, f[..., 0], face_dim, self.mesh.active_dims, p, buffer, out
        )

    def _integrate_trivial_nodes(self, f, face_dim, p, buffer, out):
        out[...] = f

    def _init_ic(
        self,
        ic: FieldFunction,
        ic_passives: Optional[Dict[str, PassiveFieldFunction]],
    ):
        # Define the following attributes:
        self.variable_index_map: VariableIndexMap
        self.nvars: int
        self.n_passive_vars: int
        self.active_vars: Set[str]
        self.passive_vars: Set[str]
        self.ic: FieldFunction
        self.callable_ic: Callable[
            [VariableIndexMap, ArrayLike, ArrayLike, ArrayLike, Optional[float]],
            ArrayLike,
        ]

        # Define variable index map
        idx = self.define_vars()

        self.ic = ic
        if ic_passives:
            for v in ic_passives.keys():
                if v not in idx.var_idx_map:
                    idx.add_var(v, idx.nvars)
            idx.add_var_to_group("passives", ic_passives.keys())
            self.ic_passives = ic_passives
            self.callable_ic = self._callable_ic_with_passives
        else:
            self.callable_ic = self._callable_ic

        self.variable_index_map = idx
        self.nvars = idx.nvars
        self.n_passive_vars = len(np.arange(self.nvars)[idx("passives")])

        # Initialize the ODE solver with the initial condition array
        ic_arr = self.mesh.perform_GaussLegendre_quadrature(
            lambda x, y, z: self.conservative_callable_ic(idx, x, y, z, 0.0),
            node_axis=4,
            mesh_region="core",
            cell_region="interior",
            p=self.p,
        )

        lower_bound = self.xp.min(ic_arr, axis=(1, 2, 3), keepdims=True)
        upper_bound = self.xp.max(ic_arr, axis=(1, 2, 3), keepdims=True)
        self.maximum_principle = (lower_bound, upper_bound)

        super().__init__(ic_arr, self.arrays)

    def _callable_ic(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float],
    ) -> ArrayLike:
        return self.ic(idx, x, y, z, t, xp=self.xp)

    def _callable_ic_with_passives(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float],
    ) -> ArrayLike:
        out = self.ic(idx, x, y, z, t, xp=self.xp)
        if hasattr(self, "ic_passives"):
            for v, f in self.ic_passives.items():
                out[idx(v)] = f(x, y, z, t, xp=self.xp)
        return out

    def conservative_callable_ic(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float],
    ) -> ArrayLike:
        """
        Callable for initial conditions that returns conservative variables.
        """
        return self.conservatives_from_primitives(self.callable_ic(idx, x, y, z, t))

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
        def as_pair(x: Union[Any, Tuple[Any, Any]]) -> Tuple[Any, Any]:
            if isinstance(x, tuple):
                if len(x) != 2:
                    raise TypeError("Expected a two-element tuple.")
                return x
            return (x, x)

        mode_list: List[Tuple[str, str]] = []
        primitive_dirichlet_list: List[Tuple[Callable, Callable]] = []

        for bci, fi in zip((bcx, bcy, bcz), (x_dirichlet, y_dirichlet, z_dirichlet)):
            mode_list_i: List[str] = []
            primitive_dirichlet_list_i: List[Callable] = []
            for j, (bc, f) in enumerate(zip(as_pair(bci), as_pair(fi))):
                if bc == "ic":
                    mode_list_i.append("dirichlet")
                    primitive_dirichlet_list_i.append(
                        lambda idx, x, y, z, t: self.callable_ic(idx, x, y, z, 0.0)
                    )
                else:
                    mode_list_i.append(bc)
                    primitive_dirichlet_list_i.append(f)

            mode_list.append(tuple(mode_list_i))
            primitive_dirichlet_list.append(tuple(primitive_dirichlet_list_i))

        mode = tuple(mode_list)
        primitive_dirichlet_arg = tuple(primitive_dirichlet_list)
        conservative_dirichlet_arg = tuple(
            tuple(self.conservatives_wrapper(f) if f is not None else None for f in f2)
            for f2 in primitive_dirichlet_arg
        )

        mesh = self.mesh
        self.primitive_bc_kwargs = dict(
            pad_width=(mesh.x_slab_depth, mesh.y_slab_depth, mesh.z_slab_depth),
            mode=mode,
            f=primitive_dirichlet_arg,
            variable_index_map=self.variable_index_map,
            mesh=self.mesh,
        )
        self.conservative_bc_kwargs = dict(
            pad_width=(mesh.x_slab_depth, mesh.y_slab_depth, mesh.z_slab_depth),
            mode=mode,
            f=conservative_dirichlet_arg,
            variable_index_map=self.variable_index_map,
            mesh=self.mesh,
        )

    def _init_snapshots(self, log_every_step: bool):
        self.log_every_step = log_every_step
        self.n_updates = 0
        self.minisnapshots["n_updates"] = []
        self.minisnapshots["MOOD_iters"] = []
        dummy_log = self.log_quantity(self.arrays["u"], 0.0)
        for key in dummy_log.keys():
            self.minisnapshots[key] = []

    def _init_array_allocation(self):
        self.interpolation_cache = ArrayManager()
        self.MOOD_cache = ArrayManager()
        self.ZS_cache = ArrayManager()
        if self.cupy:
            self.arrays.transfer_to_device("gpu")
            self.interpolation_cache.transfer_to_device("gpu")
            self.MOOD_cache.transfer_to_device("gpu")
            self.ZS_cache.transfer_to_device("gpu")

        arrays = self.arrays

        # initialize flux arrays
        nvars, nx, ny, nz = self.nvars, self.mesh.nx, self.mesh.ny, self.mesh.nz
        arrays.add("F", np.empty((nvars, nx + 1, ny, nz)))
        arrays.add("G", np.empty((nvars, nx, ny + 1, nz)))
        arrays.add("H", np.empty((nvars, nx, ny, nz + 1)))

        # initialize snapshot arrays
        assert arrays["u"].shape == (nvars, nx, ny, nz)
        arrays.add("ucc", np.empty((nvars, nx, ny, nz)))
        arrays.add("wcc", np.empty((nvars, nx, ny, nz)))
        arrays.add("w", np.empty((nvars, nx, ny, nz)))
        arrays.add("dudt", np.empty((nvars, nx, ny, nz)))

        # initialize workspace arrays
        _nx_, _ny_, _nz_ = self.mesh._nx_, self.mesh._ny_, self.mesh._nz_
        max_nodes = self.nodes_per_face(self.p)
        max_ninterps = 2 * max_nodes
        SED_buffer_size = {1: 4, 2: 6, 3: 7}[self.mesh.ndim] + 1
        buffer_size = max(SED_buffer_size, max_nodes)

        arrays.add("u_workspace", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("buffer", np.empty((nvars, _nx_, _ny_, _nz_, buffer_size)))
        arrays.add("x_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("y_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("z_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("centroid", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("theta", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("F_wrkspce", np.empty((nvars, nx + 1, _ny_, _nz_, max_nodes)))
        arrays.add("G_wrkspce", np.empty((nvars, _nx_, ny + 1, _nz_, max_nodes)))
        arrays.add("H_wrkspce", np.empty((nvars, _nx_, _ny_, nz + 1, max_nodes)))

        # fill arrays with NaNs to sabotage unpermitted use
        arrays["F"].fill(np.nan)
        arrays["G"].fill(np.nan)
        arrays["H"].fill(np.nan)
        arrays["ucc"].fill(np.nan)
        arrays["wcc"].fill(np.nan)
        arrays["w"].fill(np.nan)
        arrays["dudt"].fill(np.nan)
        arrays["u_workspace"].fill(np.nan)
        arrays["buffer"].fill(np.nan)
        arrays["x_nodes"].fill(np.nan)
        arrays["y_nodes"].fill(np.nan)
        arrays["z_nodes"].fill(np.nan)
        arrays["centroid"].fill(np.nan)
        arrays["theta"].fill(np.nan)
        arrays["F_wrkspce"].fill(np.nan)
        arrays["G_wrkspce"].fill(np.nan)
        arrays["H_wrkspce"].fill(np.nan)

        # helper attribute
        self.flux_names = {"x": "F", "y": "G", "z": "H"}

    def _init_riemann_solver(self, riemann_solver: str):
        if not hasattr(self, riemann_solver):
            raise ValueError(f"Riemann solver {riemann_solver} not implemented.")
        self.riemann_func = getattr(self, riemann_solver)

    @MethodTimer(cat="FiniteVolumeSolver.riemann_solver")
    def riemann_solver(
        self, wl: ArrayLike, wr: ArrayLike, dim: Literal["x", "y", "z"], out: ArrayLike
    ):
        """
        Compute the numerical fluxes at the interface using the specified Riemann solver.

        Args:
            wl: Primitive variables to the left of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            wr: Primitive variables to the right of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            dim: Direction in which the Riemann problem is solved: "x", "y", or "z".
            out: Output array to store the numerical fluxes. Has shape
                (nvars, nx, ny, nz, ...).
        """
        return self.riemann_func(wl, wr, dim, out)

    def _init_slope_limiting(
        self,
        MUSCL: bool,
        ZS: bool,
        adaptive_dt: bool,
        max_dt_revisions: int,
        MOOD: bool,
        max_MOOD_iters: int,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]],
        NAD: Optional[float],
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        PAD_tol: float,
        SED: bool,
    ):
        # reset slope limiting state
        self.a_priori_slope_limiter = None
        self.a_priori_slope_limiting_scheme = None
        self.adaptive_dt = False
        self.max_dt_revisions = 0
        self.MOOD = False
        self.MOOD_cascade = []
        self.MOOD_iter_count = 0
        self.MOOD_max_iter_count = 0
        self.NAD = None
        self.PAD = None
        self.PAD_tol = np.nan
        self.SED = False
        self.ZS = False

        # skip slope limiting for first-order or if nothing is active
        if self.p == 0 or (not ZS and not MOOD):
            return

        # common settings ---
        self.PAD = PAD
        self.PAD_tol = PAD_tol
        self.SED = SED

        # PAD
        if PAD:
            idx = self.variable_index_map.idx
            PAD_arr = np.full((self.nvars, 2), [-np.inf, np.inf])
            for var, (lb, ub) in PAD.items():
                PAD_arr[idx(var)] = [
                    lb if lb is not None else -np.inf,
                    ub if ub is not None else np.inf,
                ]
            PAD_arr = np.expand_dims(
                PAD_arr, axis=(1, 2, 3)
            )  # shape (nvars, 1, 1, 1, 2)
            self.arrays.add("PAD", PAD_arr)

        # Zhang-Shu (a priori limiting)
        if ZS:
            self.ZS = True
            self.adaptive_dt = adaptive_dt
            if adaptive_dt:
                self._dt_criterion = self.adaptive_dt_criterion
                self._compute_revised_dt = self.adaptive_dt_revision
                self.max_dt_revisions = max_dt_revisions

        # MOOD (a posteriori limiting)
        if MOOD:
            self.MOOD = True
            self.NAD = NAD
            self.MOOD_max_iter_count = max_MOOD_iters

    def adaptive_dt_criterion(self, tnew: float, unew: ArrayLike) -> bool:
        """
        Returns whether PAD violations are present.
        """
        xp = self.xp

        PAD_lower = self.arrays["PAD"][..., 0]
        PAD_upper = self.arrays["PAD"][..., 1]

        nontrivial_mask = xp.logical_or(
            PAD_lower > -np.inf, PAD_upper < np.inf
        ).flatten()
        unew_filt = unew[nontrivial_mask]

        PAD_lower_filt = PAD_lower[nontrivial_mask]
        PAD_upper_filt = PAD_upper[nontrivial_mask]

        lower_violations = unew_filt < PAD_lower_filt - self.PAD_tol
        upper_violations = unew_filt > PAD_upper_filt + self.PAD_tol
        return not xp.any(xp.logical_or(lower_violations, upper_violations))

    def adaptive_dt_revision(self, t, u, dt):
        """
        Returns dt/2 if the maximum number of adaptive timestep revisions hasn't been
        exceeded.
        """
        if self.dt_revision_count < self.max_dt_revisions:
            return dt / 2
        raise ValueError(
            f"Failed to satisfy `dt_criterion` in {self.max_dt_revisions} iterations."
        )

    @MethodTimer(cat="FiniteVolumeSolver.f")
    def f(self, t: float, u: ArrayLike) -> ArrayLike:
        """
        Compute the right-hand side of the ODE:

            du/dt = -∇·F(u)

        Args:
            t: Time value.
            u: Array of conservative, cell-averaged values. Shape: (nvars, nx, ny, nz).

        Returns:
            dudt: Right-hand side of the ODE. Shape: (nvars, nx, ny, nz).
        """
        out = self.arrays["dudt"]

        self.inplace_compute_fluxes(t, u, self.p)

        out[...] = 0.0
        for dim in self.mesh.active_dims:
            self._add_flux_divergence(dim, out)
        self.n_updates += self.mesh.size
        return out.copy()

    @MethodTimer(cat="FiniteVolumeSolver.inplace_compute_fluxes")
    def inplace_compute_fluxes(self, t: float, u: ArrayLike, p: int):
        """
        Updates the interior of `self.arrays["u_workspace"]` with the provided
        conservative values `u` and applies boundary conditions in place.

        Interpolates the face nodes in each active dimension with polynomial degree `p`
        and writes the results to `self.arrays["x_nodes"]`, `self.arrays["y_nodes"]`,
        and/or `self.arrays["z_nodes"]`.

        Computes the face flux integral in each active dimension with polynomial degree
        `p` and writes the results to `self.arrays["F"]`, `self.arrays["G"]`, and/or
        `self.arrays["H"]`.

        Args:
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            p: Polynomial degree of the spatial discretization.
        """
        _u_ = self.arrays["u_workspace"]
        _u_[self.interior] = u
        self.inplace_apply_bc(t, _u_, p=p)

        for dim in self.mesh.active_dims:
            self.inplace_interpolate_faces(t, _u_, dim, p, primitives=False)

        if self.ZS:
            self.zhang_shu_limiter(p)

        for dim in self.mesh.active_dims:
            self.apply_nodal_bc(t, dim, p, primitives=False)
            self.inplace_integrate_fluxes(dim, p)

    @MethodTimer(cat="FiniteVolumeSolver.inplace_apply_bc")
    def inplace_apply_bc(
        self,
        t: float,
        u: ArrayLike,
        primitives: bool = False,
        pointwise: bool = False,
        cell_region: Optional[
            Literal["xl", "xr", "yl", "yr", "zl", "zr", "center"]
        ] = None,
        p: Optional[int] = None,
    ):
        """
        Apply boundary conditions to the provided array `u` in place.
        """
        dirichlet_mode = (
            "fv-averages"
            if not pointwise
            else ("cell-centers" if cell_region == "center" else "face-nodes")
        )
        apply_bc(
            _u_=u,
            dirichlet_mode=dirichlet_mode,
            t=t,
            face_dim=cell_region[0] if dirichlet_mode == "face-nodes" else None,
            face_pos=cell_region[1] if dirichlet_mode == "face-nodes" else None,
            p=p if self.GL else 0,
            **(self.primitive_bc_kwargs if primitives else self.conservative_bc_kwargs),
        )

    @MethodTimer(cat="FiniteVolumeSolver.inplace_interpolate_faces")
    def inplace_interpolate_faces(
        self,
        t: float,
        u: ArrayLike,
        dim: Literal["x", "y", "z"],
        p: int,
        primitives: bool = True,
    ):
        """
        Interpolate the face nodes at the opposing face centers and optionally convert
        the interpolated values to primitive variables, then update boundaries.

        Args:
            t: Time value.
            u: Workspace array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            dim: Dimension in which to interpolate the face nodes. Possible values:
                - "x": Interpolate in the x-direction.
                - "y": Interpolate in the y-direction.
                - "z": Interpolate in the z-direction.
            p: Polynomial degree of the spatial discretization.
            primitives: Whether to convert the interpolated values to primitive
                variables.

        Returns:
            None. Modifies `self.arrays[dim + "_nodes"]` in place.

        Notes:
            - `self.arrays["buffer"]` is used as a temporary buffer for sweeps.
        """
        out = self.arrays[dim + "_nodes"]
        buffer = self.arrays["buffer"]

        self.interpolation_func(
            u,
            face_dim=dim,
            p=p,
            buffer=buffer,
            out=out,
        )

        if primitives:
            self.conservatives_to_primitives(out, buffer)
            out[...] = buffer

    def apply_nodal_bc(
        self, t: float, dim: Literal["x", "y", "z"], p: int, primitives: bool
    ):
        """
        Apply boundary conditions to arrays of nodal variables
        ("x_nodes", "y_nodes", "z_nodes").

        Args:
            t: Time value.
            dim: Dimension in which to interpolate the face nodes. Possible values:
                - "x": Interpolate in the x-direction.
                - "y": Interpolate in the y-direction.
                - "z": Interpolate in the z-direction.
            p: Polynomial degree of the spatial discretization.
            primitives: Whether the nodal variables arrays are of primitive variables.

        Returns:
            None. Modifies `self.arrays[dim + "_nodes"]` in place.
        """
        n = self.nodes_per_face(p)

        nodes = self.arrays[dim + "_nodes"]
        ul = nodes[crop(4, (None, n))]
        ur = nodes[crop(4, (n, 2 * n))]

        self.inplace_apply_bc(
            t,
            ul,
            primitives=primitives,
            pointwise=True,
            cell_region=dim + "l",
            p=p,
        )
        self.inplace_apply_bc(
            t,
            ur,
            primitives=primitives,
            pointwise=True,
            cell_region=dim + "r",
            p=p,
        )

    @MethodTimer(cat="FiniteVolumeSolver.zhang_shu_limiter")
    def zhang_shu_limiter(self, p: int):
        """
        Limit the face node arrays.

        Args:
            p: Polynomial degree for interpolating the cell centroid.
        """

        xp = self.xp
        mesh = self.mesh

        # define array references
        u = self.arrays["u_workspace"]
        centroid = self.arrays["centroid"]
        x = self.arrays["x_nodes"] if mesh.x_is_active else None
        y = self.arrays["y_nodes"] if mesh.y_is_active else None
        z = self.arrays["z_nodes"] if mesh.z_is_active else None
        theta = self.arrays["theta"]
        buffer = self.arrays["buffer"]

        # compute centroid then compute theta
        fv.interpolate_cell_centers(xp, u, mesh.active_dims, p, buffer, out=centroid)
        compute_theta(xp, u, centroid, x, y, z, buffer, SED=self.SED, out=theta)

        # limit the face nodes
        if mesh.x_is_active:
            x[...] = zhang_shu_operator(x, u[..., np.newaxis], theta)
        if mesh.y_is_active:
            y[...] = zhang_shu_operator(y, u[..., np.newaxis], theta)
        if mesh.z_is_active:
            z[...] = zhang_shu_operator(z, u[..., np.newaxis], theta)

    @MethodTimer(cat="FiniteVolumeSolver.inplace_integrate_fluxes")
    def inplace_integrate_fluxes(self, dim: Literal["x", "y", "z"], p: int):
        """
        Integrate the flux nodes at the face centers using the transverse quadrature or
        Gauss-Legendre quadrature.

        Args:
            dim: Dimension in which to integrate the flux nodes. Possible values:
                - "x": Integrate in the x-direction.
                - "y": Integrate in the y-direction.
                - "z": Integrate in the z-direction.
            p: Polynomial degree of the spatial discretization.

        Returns:
            None. Modifies `self.arrays[{"x": "F", ...}[dim]]` in place.

        Notes:
            - `self.arrays["buffer"]` is used as a temporary buffer for the sweeps.
            - `self.arrays[{"x": "F_wrkspce", ...}[dim]]` is used as a workspace for
            the flux integration.
            - `self.arrays[dim + "_nodes"]` contains the interpolated primitive
                variables at the face nodes.

        """
        n = self.nodes_per_face(p)
        pad = getattr(self.mesh, f"{dim}_slab_depth")
        axis = DIM_TO_AXIS[dim]
        flux_name = self.flux_names[dim]
        flux_workspace_name = flux_name + "_wrkspce"

        out = self.arrays[flux_name]
        flux_workspace = self.arrays[flux_workspace_name]

        nodes = self.arrays[dim + "_nodes"]
        wl = nodes[crop(4, (None, n))]
        wr = nodes[crop(4, (n, 2 * n))]

        left_state = wr[crop(axis, (pad - 1, -pad))]
        right_state = wl[crop(axis, (pad, -pad + 1))]
        self.riemann_solver(left_state, right_state, dim, left_state)  # overwrite wl

        self.integration_func(left_state, dim, p, right_state, flux_workspace)

        out[...] = flux_workspace[self.flux_interior[dim]][..., 0]

    def nodes_per_face(self, p: int) -> int:
        """
        Returns the number of nodes per face for a given polynomial degree `p`.

        Args:
            p: Polynomial degree of the spatial discretization.
        """
        if self.GL:
            return (-(-(p + 1) // 2)) ** (self.mesh.ndim - 1)
        return 1

    def _add_flux_divergence(self, dim: Literal["x", "y", "z"], out: ArrayLike):
        """
        Add the flux divergence in the specified dimension to the output array.

        Args:
            dim: Dimension in which to compute the flux divergence. Possible values:
                - "x": Compute in the x-direction.
                - "y": Compute in the y-direction.
                - "z": Compute in the z-direction.
            out: Output array to which the flux divergence is added. Has shape
                (nvars, nx, ny, nz).
        """
        flux_name = self.flux_names[dim]
        h = getattr(self.mesh, "h" + dim)
        axis = DIM_TO_AXIS[dim]

        F = self.arrays[flux_name]
        self.xp.add(
            out,
            -(1 / h) * (F[crop(axis, (1, None))] - F[crop(axis, (None, -1))]),
            out=out,
        )

    def compute_fluxes(
        self,
        t: float,
        u: ArrayLike,
        mode: Literal["transverse", "gauss-legendre"],
        p: int,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
    ) -> Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]:
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
            fluxes: Tuple of flux arrays or None if the corresponding dimension is
                unused. The fluxes have the following shapes if not None:
            - F: (nvars, nx + 1, ny, nz)
            - G: (nvars, nx, ny + 1, nz)
            - H: (nvars, nx, ny, nz + 1)
        """
        match self.flux_recipe:
            case 1:
                return self._flux_recipe1(
                    t=t,
                    u=u,
                    mode=mode,
                    p=p,
                    limiting_scheme=limiting_scheme,
                    slope_limiter=slope_limiter,
                )
            case _:
                raise NotImplementedError(
                    "Flux recipes 2 and 3 are not implemented yet."
                )

    def _flux_recipe1(
        self,
        t: float,
        u: ArrayLike,
        mode: Literal["transverse", "gauss-legendre"],
        p: int,
        limiting_scheme: Optional[Literal["muscl", "zhang-shu"]] = None,
        slope_limiter: Optional[Literal["minmod", "moncen"]] = None,
    ) -> Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]:
        """
        Helper function for variant of `compute_fluxes` that has the following steps:
        - compute conservative face nodes
        - limit conservative face nodes
        - transform limited conservative face nodes to primitive face nodes
        - compute fluxes from primitive face nodes
        """
        p = 1 if limiting_scheme == "muscl" else cast(int, p)
        u_padded = self.apply_bc(u, -(-p // 2), t=t, p=p)

        fluxes: List[Optional[ArrayLike]] = []
        for dim in ["x", "y", "z"]:
            # skip unused dimensions
            if dim not in self.mesh.active_dims:
                fluxes.append(None)
                continue

            # interpolate face nodes as primitive variables
            u_xl, u_xr = self.interpolate_face_nodes(
                u_padded,
                dim=dim,
                interpolation_scheme=mode,
                limiting_scheme=limiting_scheme,
                p=p,
                slope_limiter=slope_limiter,
            )
            w_xl = self.primitives_from_conservatives(u_xl)
            w_xr = self.primitives_from_conservatives(u_xr)

            # apply extra ghost cells for transverse quadrature and riemann solver
            nghost2 = p // 2 + 1 if mode == "transverse" else 1
            w_xl = self.apply_bc(
                w_xl,
                nghost2,
                primitives=True,
                fv_averages=False,
                t=t,
                face_quadrature=cast(Faces, dim + "l"),
                p=0 if mode == "transverse" else p,
            )
            w_xr = self.apply_bc(
                w_xr,
                nghost2,
                primitives=True,
                fv_averages=False,
                t=t,
                face_quadrature=cast(Faces, dim + "r"),
                p=0 if mode == "transverse" else p,
            )

            # compute fluxes from the primitive variables
            F = self.integrate_fluxes(
                w_xr[crop(DIM_TO_AXIS[dim], (None, -1))],
                w_xl[crop(DIM_TO_AXIS[dim], (1, None))],
                dim=dim,
                quadrature=mode,
                p=p,
            )
            fluxes.append(F)

        return fluxes[0], fluxes[1], fluxes[2]

    def MOOD_loop(
        self,
        t: float,
        u: ArrayLike,
        fluxes: Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]],
    ) -> Tuple[Optional[ArrayLike], Optional[ArrayLike], Optional[ArrayLike]]:
        """
        Revise fluxes using MOOD.

        Args:
            t: Time value.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            fluxes: Tuple of flux arrays (F, G, H). None if the corresponding dimension
                is unused. Otherwise, is an array with shape:
                - F: (nvars, nx+1, ny, nz)
                - G: (nvars, nx, ny+1, nz)
                - H: (nvars, nx, ny, nz+1)

        Returns:
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
            u=u,
            fluxes=fluxes,
            NAD=self.NAD,
            PAD=self.arrays["PAD"] if "PAD" in self.arrays else None,
            PAD_tol=self.PAD_tol,
            SED=self.SED,
        ):
            fluxes = revise_fluxes(
                self,
                t,
                u,
                fluxes,
                mode=self.interpolation_scheme,
                slope_limiter="minmod",
            )
        return fluxes

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
                px = p if self.mesh.x_is_active else 0
                py = p if self.mesh.y_is_active else 0
                pz = p if self.mesh.z_is_active else 0
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

    def integrate_fluxes(
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
        for dim, _F in zip(["x", "y", "z"], [F, G, H]):
            if dim not in self.mesh.active_dims:
                continue
            dudt += -(1 / getattr(self.mesh, "h" + dim)) * (
                cast(ArrayLike, _F)[crop(DIM_TO_AXIS[dim], (1, None))]
                - cast(ArrayLike, _F)[crop(DIM_TO_AXIS[dim], (None, -1))]
            )
        return dudt

    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        _u_ = self.arrays["u_workspace"]
        ucc = self.arrays["ucc"]
        wcc = self.arrays["wcc"]
        w = self.arrays["w"]
        buffer1 = self.arrays["buffer"][..., :1]
        buffer2 = self.arrays["buffer"][..., 1:]

        p = self.p
        dims = self.mesh.active_dims
        interior = self.interior

        # fill workspace interior
        _u_[interior] = self.arrays["u"]
        self.inplace_apply_bc(self.t, _u_, p=self.p)

        # conservative cell centers
        fv.interpolate_cell_centers(self.xp, _u_, dims, p, buffer2, buffer1)
        ucc[...] = buffer1[interior][..., 0]

        # primitive cell centers
        self.conservatives_to_primitives(ucc, wcc)

        # primitive cell averages
        if self.lazy_primitives:
            self.conservatives_to_primitives(self.arrays["u"], w)
        else:
            _u_[interior] = wcc
            self.inplace_apply_bc(
                self.t,
                _u_,
                p=-(-p // 2),
                primitives=True,
                pointwise=True,
                cell_region="center",
            )
            fv.integrate_fv_averages(
                self.xp, _u_, self.mesh.active_dims, p, buffer2, buffer1
            )
            w[...] = buffer1[interior][..., 0]

        # store the snapshot
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "w": self.arrays.get_numpy_copy("w"),
            "ucc": self.arrays.get_numpy_copy("ucc"),
            "wcc": self.arrays.get_numpy_copy("wcc"),
        }
        self.snapshots.log(self.t, data)

    def minisnapshot(self):
        super().minisnapshot()
        self.minisnapshots["n_updates"].append(self.n_updates)
        self.minisnapshots["MOOD_iters"].append(self.MOOD_iter_count)
        if self.log_every_step:
            log = self.log_quantity(self.arrays["u"], self.t)
            for key, value in log.items():
                self.minisnapshots[key].append(value)
        self.n_updates = 0

    def called_at_end_of_step(self):
        """
        Called at the end of each time step to synchronize the GPU if using CuPy and to
        perform any additional end-of-step operations.

        """
        if self.cupy:
            self.xp.cuda.Device().synchronize()
        super().called_at_end_of_step()

    @MethodTimer(cat="FiniteVolumeSolver.run")
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["xp"] = DummyModule
        state["ic"] = _uninitialized
        state["ic_passives"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.cupy:
            self.xp = xp
            self.arrays.transfer_to_device("gpu")
        else:
            self.xp = np
