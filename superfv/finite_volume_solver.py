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
)

import numpy as np

from . import fv
from .boundary_conditions import DirichletBC, apply_bc
from .explicit_ODE_solver import ExplicitODESolver
from .fv import DIM_TO_AXIS
from .initial_conditions import _uninitialized
from .mesh import UniformFVMesh
from .slope_limiting import MOOD
from .slope_limiting.MOOD import MOODConfig
from .slope_limiting.zhang_and_shu import (
    ZhangShuConfig,
    compute_theta,
    zhang_shu_operator,
)
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager, xp
from .tools.dummy_module import DummyModule
from .tools.slicing import VariableIndexMap, crop, merge_slices
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
    def log_quantity(self) -> Dict[str, Any]:
        """
        Log a quantity at the end of each time step.

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
        NAD: bool = False,
        NAD_rtol: float = 1.0,
        NAD_atol: float = 0.0,
        global_dmp: bool = False,
        include_corners: bool = False,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        PAD_atol: float = 1e-15,
        SED: bool = False,
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
            NAD: Whether to use nuerical admissibility detection (NAD) when determining
                if a cell is troubled in the MOOD loop.
            NAD_rtol: Relative tolerance for the NAD violations.
            NAD_atol: Absolute tolerance for the NAD violations.
            global_dmp: Whether to use a global DMP check for NAD violations.
            include_corners: Whether to include corner nodes in the slope limiting.
            PAD: Dict of `limiting_vars` and their corresponding PAD tolerances as a
                tuple: (lower_bound, upper_bound). Any variable or bound not provided
                in `PAD` is given a lower and upper bound of `-np.inf` and `np.inf`
                respectively.
            PAD_atol: Tolerance for the PAD check as an absolute value from the minimum
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
            NAD_atol,
            NAD_rtol,
            global_dmp,
            include_corners,
            PAD,
            PAD_atol,
            SED,
        )
        self._init_array_allocation()
        self._init_snapshots(log_every_step)

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
        elif interpolation_scheme not in ("gauss-legendre", "transverse"):
            raise ValueError(
                "interpolation_scheme must be 'gauss-legendre' or 'transverse'."
            )
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
            tuple(self.conservative_callable_ic if f is not None else None for f in f2)
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

        def normalize_troubles_bc(
            bc_tuple: Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]],
        ) -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
            def map_bc(bc: str) -> str:
                if bc == "none":
                    return "none"
                elif bc == "periodic":
                    return "periodic"
                else:
                    return "zeros"

            return tuple(tuple(map_bc(bc) for bc in dim) for dim in bc_tuple)

        self.troubles_bc_kwargs = dict(
            pad_width=(mesh.x_slab_depth, mesh.y_slab_depth, mesh.z_slab_depth),
            mode=normalize_troubles_bc(mode),
            variable_index_map=VariableIndexMap({"troubles": 0}),
        )

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
        NAD: bool,
        NAD_atol: float,
        NAD_rtol: float,
        global_dmp: bool,
        include_corners: bool,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        PAD_atol: float,
        SED: bool,
    ):
        # reset slope limiting state
        self.MOOD: bool = False
        self.MOOD_config: MOODConfig = MOODConfig()
        self.ZS = False
        self.ZhangShu_config = ZhangShuConfig()

        # configure variable indices
        idx = self.variable_index_map
        idx.add_var_to_group("limiting", [])

        # skip slope limiting for first-order or if nothing is active
        if self.p == 0 or (not ZS and not MOOD):
            return

        # PAD
        if PAD:
            PAD_bounds = np.full((self.nvars, 2), [-np.inf, np.inf])
            for var, (lb, ub) in PAD.items():
                PAD_bounds[idx(var)] = [
                    lb if lb is not None else -np.inf,
                    ub if ub is not None else np.inf,
                ]
            PAD_bounds = np.expand_dims(
                PAD_bounds, axis=(1, 2, 3)
            )  # shape (nvars, 1, 1, 1, 2)
            self.arrays.add("PAD_bounds", PAD_bounds)

        # Zhang-Shu (a priori limiting)
        if ZS:
            self.ZS = True
            self.ZhangShu_config = ZhangShuConfig(
                include_corners=include_corners,
                SED=SED,
                adaptive_dt=adaptive_dt,
                max_dt_revisions=max_dt_revisions,
                PAD_bounds=self.arrays["PAD_bounds"] if PAD else None,
                PAD_atol=PAD_atol,
            )
            if adaptive_dt:
                self._dt_criterion = self.adaptive_dt_criterion
                self._compute_revised_dt = self.adaptive_dt_revision

        # MOOD (a posteriori limiting)
        if MOOD:
            self.MOOD = True
            self.MOOD_config = MOODConfig(
                max_iters=max_MOOD_iters,
                cascade=["fv" + str(self.p)] + ["fv0"],
                NAD=NAD,
                NAD_rtol=NAD_rtol,
                NAD_atol=NAD_atol,
                global_dmp=global_dmp,
                include_corners=include_corners,
                PAD=PAD is not None,
                PAD_bounds=self.arrays["PAD_bounds"] if PAD else None,
                PAD_atol=PAD_atol,
                SED=SED,
            )

        # add limiing variables to the index map
        if limiting_vars == "all":
            limiting_vars = tuple(idx.var_idx_map.keys())
        elif limiting_vars == "actives":
            limiting_vars = tuple(idx.group_var_map["primitives"])
        idx.add_var_to_group("limiting", limiting_vars)

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
        MOOD_buffer_size = {1: 4, 2: 6, 3: 7}[self.mesh.ndim] + 4
        buffer_size = max(SED_buffer_size, max_nodes)
        if self.MOOD:
            buffer_size = max(buffer_size, MOOD_buffer_size)

        arrays.add("_u_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("buffer", np.empty((nvars, _nx_, _ny_, _nz_, buffer_size)))
        arrays.add("x_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("y_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("z_nodes", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("centroid", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("theta", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("theta_log", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("troubles", np.zeros((1, nx, ny, nz), dtype=bool))
        arrays.add("troubles_log", np.zeros((1, nx, ny, nz), dtype=int))
        arrays.add("_troubles_", np.zeros((1, _nx_, _ny_, _nz_), dtype=bool))
        arrays.add("F_wrkspce", np.empty((nvars, nx + 1, _ny_, _nz_, max_nodes)))
        arrays.add("G_wrkspce", np.empty((nvars, _nx_, ny + 1, _nz_, max_nodes)))
        arrays.add("H_wrkspce", np.empty((nvars, _nx_, _ny_, nz + 1, max_nodes)))

        # MOOD arrays
        if self.MOOD:
            for scheme in self.MOOD_config.cascade:
                arrays.add("F_" + scheme, np.empty((nvars, nx + 1, ny, nz)))
                arrays.add("G_" + scheme, np.empty((nvars, nx, ny + 1, nz)))
                arrays.add("H_" + scheme, np.empty((nvars, nx, ny, nz + 1)))
            arrays.add(
                "_cascade_idx_array_", np.zeros((1, _nx_, _ny_, _nz_), dtype=int)
            )
            arrays.add("_mask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1), dtype=int))

        # fill arrays with NaNs to sabotage unpermitted use
        arrays["F"].fill(np.nan)
        arrays["G"].fill(np.nan)
        arrays["H"].fill(np.nan)
        arrays["ucc"].fill(np.nan)
        arrays["wcc"].fill(np.nan)
        arrays["w"].fill(np.nan)
        arrays["dudt"].fill(np.nan)
        arrays["_u_"].fill(np.nan)
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

    def _init_snapshots(self, log_every_step: bool):
        self.log_every_step = log_every_step
        self.reset_substepwise_counters()

        # allocate keys from collect_minisnapshot_data
        keys = self.collect_minisnapshot_data().keys()
        for key in keys:
            self.minisnapshots[key] = []

    def adaptive_dt_criterion(self, tnew: float, unew: ArrayLike) -> bool:
        """
        Returns whether unew is free of PAD violations.
        """
        xp = self.xp
        PAD_bounds = self.ZhangShu_config.PAD_bounds
        PAD_atol = self.ZhangShu_config.PAD_atol

        PAD_violations = self.arrays["buffer"][self.interior][..., 0]
        MOOD.inplace_PAD_violations(xp, unew, PAD_bounds, PAD_atol, out=PAD_violations)

        return not xp.any(PAD_violations < 0)

    def adaptive_dt_revision(self, t, u, dt):
        """
        Returns dt/2 if the maximum number of adaptive timestep revisions hasn't been
        exceeded.
        """
        n_dt_revisions = self.n_dt_revisions
        max_dt_revisions = self.ZhangShu_config.max_dt_revisions

        if n_dt_revisions < max_dt_revisions:
            return dt / 2
        raise ValueError(
            f"Failed to satisfy `dt_criterion` in {max_dt_revisions} iterations."
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
            Right-hand side of the ODE. Shape: (nvars, nx, ny, nz).
        """
        self.update_workspace(t, u, self.p)
        self.inplace_compute_fluxes(t, self.p)

        if self.MOOD:
            self.MOOD_loop(t)

        out = self.compute_RHS().copy()

        self.increment_substepwise_counters()
        self.accumulate_substepwise_logarrays()

        return out

    def update_workspace(self, t: float, u: ArrayLike, p: int):
        """
        Update the workspace array with the provided state u and apply boundary
        conditions in place.

        Args:
            t: Time value.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            p: Polynomial degree of the spatial discretization.
        """
        _u_ = self.arrays["_u_"]
        _u_[self.interior] = u
        self.inplace_apply_bc(t, _u_, p=p)

    @MethodTimer(cat="FiniteVolumeSolver.inplace_compute_fluxes")
    def inplace_compute_fluxes(self, t: float, p: int):
        """
        Compute the fluxes at the cell faces from the workspace array `u_workspace`.

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
        _u_ = self.arrays["_u_"]

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

    def inplace_troubles_bc(self, troubles: ArrayLike):
        """
        Apply boundary conditions to the troubles array in place.
        """
        apply_bc(_u_=troubles, **self.troubles_bc_kwargs)

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
        ZhangShu_config = self.ZhangShu_config

        # define array references
        u = self.arrays["_u_"]
        centroid = self.arrays["centroid"]
        x = self.arrays["x_nodes"] if mesh.x_is_active else None
        y = self.arrays["y_nodes"] if mesh.y_is_active else None
        z = self.arrays["z_nodes"] if mesh.z_is_active else None
        theta = self.arrays["theta"]
        buffer = self.arrays["buffer"]

        # compute centroid then compute theta
        fv.interpolate_cell_centers(xp, u, mesh.active_dims, p, buffer, out=centroid)
        compute_theta(
            xp,
            u,
            centroid,
            x,
            y,
            z,
            buffer,
            SED=ZhangShu_config.SED,
            include_corners=ZhangShu_config.include_corners,
            tol=ZhangShu_config.tol,
            out=theta,
        )

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

    def MOOD_loop(self, t: float):
        """
        Perform the MOOD loop to detect and revise troubled cells.

        Args:
            t: Time value.
        """
        self.MOOD_config.reset_MOOD_loop()
        while True:
            if MOOD.detect_troubled_cells(self, t):
                MOOD.inplace_revise_fluxes(self, t)
                self.MOOD_config.increment_MOOD_iteration()
            else:
                return

    def compute_RHS(self) -> ArrayLike:
        """
        Compute the right-hand side of the ODE .

        Returns:
            ArrayLike: The right-hand side of the ODE.
        """
        out = self.arrays["dudt"]
        out[...] = 0.0

        for dim in self.mesh.active_dims:
            self._add_flux_divergence(dim, out)

        return out

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

    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        u = self.arrays["u"]
        _u_ = self.arrays["_u_"]
        ucc = self.arrays["ucc"]
        wcc = self.arrays["wcc"]
        w = self.arrays["w"]
        buffer1 = self.arrays["buffer"][..., :1]
        buffer2 = self.arrays["buffer"][..., 1:]

        p = self.p
        dims = self.mesh.active_dims
        interior = self.interior

        # update workspace interior
        self.update_workspace(self.t, u, self.p)

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
            "theta": self.arrays.get_numpy_copy("theta_log"),
            "troubles": self.arrays.get_numpy_copy("troubles_log"),
        }
        self.snapshots.log(self.t, data)

    def minisnapshot(self):
        """
        Overwrite the `minisnapshot` method to collect additional data.
        """
        super().minisnapshot()

        data = self.collect_minisnapshot_data()
        for key, value in data.items():
            self.minisnapshots[key].append(value)

    def collect_minisnapshot_data(self) -> Dict[str, Any]:
        """
        Collect additional data for a minisnapshot.

        Returns:
            Dictionary mapping minisnapshot keys to their values.
        """
        n_updates = self.n_updates
        run_time = (
            self.minisnapshots["run_time"][-1]
            if len(self.minisnapshots["run_time"]) > 0
            and self.minisnapshots["run_time"][-1] > 0
            else np.nan
        )
        data = {"n_updates": n_updates, "update_rate": n_updates / run_time}

        if self.MOOD:
            data["MOOD_iters"] = self.MOOD_config.iter_count

        if self.log_every_step:
            data.update(self.log_quantity())

        return data

    def reset_substepwise_counters(self):
        """
        Reset the counters that increment in each substep of the integration.
        """
        self.n_updates = 0

    def increment_substepwise_counters(self):
        """
        Increment counters at the end of each substep of the integration.
        """
        self.n_updates += self.mesh.size

    def reset_substepwise_logarrays(self):
        """
        Reset the log arrays that accumulate data over substeps.
        """
        if self.MOOD:
            troubles = self.arrays["troubles_log"]
            troubles[...] = 0

        if self.ZS:
            theta = self.arrays["theta_log"]
            theta[...] = 0

    def accumulate_substepwise_logarrays(self):
        """
        Accumulate the log arrays over substeps.
        """
        xp = self.xp

        if self.ZS:
            theta = self.arrays["theta"]
            theta_log = self.arrays["theta_log"]
            xp.add(theta_log, theta, out=theta_log)

        if self.MOOD:
            troubles = self.arrays["troubles"]
            troubles_log = self.arrays["troubles_log"]
            xp.add(troubles_log, troubles, out=troubles_log)

    def called_at_beginning_of_step(self):
        """
        Overwrite `called_at_beginning_of_step` of the ODE solver to reset various
        internal state variables.
        """
        super().called_at_beginning_of_step()
        self.reset_substepwise_counters()
        self.reset_substepwise_logarrays()
        if self.MOOD:
            self.MOOD_config.reset_iter_count()

    def called_at_end_of_step(self):
        """
        Overwrite `called_at_end_of_step` of the ODE solver to synchronize the GPU if
        using CuPy, and to perform any additional cleanup or logging.
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
