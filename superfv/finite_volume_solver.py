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
from .boundary_conditions import BCs, DirichletBC, Field, apply_bc
from .explicit_ODE_solver import ExplicitODESolver
from .fv import DIM_TO_AXIS
from .initial_conditions import _uninitialized
from .interpolation_schemes import (
    InterpolationScheme,
    musclInterpolationScheme,
    polyInterpolationScheme,
)
from .mesh import UniformFVMesh
from .slope_limiting import MOOD, minmod, moncen
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
        self,
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
        GL: bool = False,
        flux_recipe: Literal[1, 2, 3] = 1,
        lazy_primitives: bool = False,
        riemann_solver: str = "dummy_riemann_solver",
        MUSCL: bool = False,
        ZS: bool = False,
        adaptive_dt: bool = True,
        max_dt_revisions: int = 8,
        MOOD: bool = False,
        cascade: Literal["first-order", "muscl", "full"] = "first-order",
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
            GL: Whether to use Gauss-Legendre quadrature for flux integration. If
                `False`, the transverse quadrature is used.
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
            cascade: A string indicating which type of MOOD cascade to use:
                - "first-order": Fall back directly to a first-order scheme.
                - "muscl": Fall back directly to a MUSCL scheme.
                - "full": Fall back to a full cascade of scheme in descending order.
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
        self._init_spatial_discretization(
            GL,
            p,
            flux_recipe,
            lazy_primitives,
        )
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, CFL)
        self._init_ic(ic, ic_passives)
        self._init_bc(bcx, bcy, bcz, x_dirichlet, y_dirichlet, z_dirichlet)
        self._init_riemann_solver(riemann_solver)
        self._init_slope_limiting(
            MUSCL,
            ZS,
            adaptive_dt,
            max_dt_revisions,
            MOOD,
            cascade,
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

    def _init_spatial_discretization(
        self,
        GL: bool,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        lazy_primitives: bool,
    ):
        # validate flux recipe
        if flux_recipe not in (1, 2, 3):
            raise ValueError(
                "flux_recipe must be 1, 2, or 3. See the documentation for details."
            )

        # assign polynomial degree
        if p < 0:
            raise ValueError("The polynomial degree must be non-negative.")
        self.p = p

        # assign base scheme
        self.base_scheme = polyInterpolationScheme(
            flux_recipe=flux_recipe,
            limiter=None,
            p=p,
            lazy_primitives=lazy_primitives,
            gauss_legendre=GL,
        )

    def _init_mesh(
        self,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zlim: Tuple[float, float],
        nx: int,
        ny: int,
        nz: int,
        CFL: float,
    ):
        if CFL <= 0:
            raise ValueError("The CFL number must be positive.")

        # determine slab depth
        stencil_depth = -2 * (-self.p // 2)
        SED_depth = 3
        slab_depth = max(stencil_depth, SED_depth)

        # init mesh object
        self.mesh = UniformFVMesh(
            nx=nx,
            ny=ny,
            nz=nz,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            slab_depth=slab_depth,
            array_manager=self.arrays,
        )

        # assign attributes
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
        """
        Helper function to call the initial condition function.

        Args:
            idx: VariableIndexMap object.
            x: x-coordinate array. Has shape (nx, ny, nz).
            y: y-coordinate array. Has shape (nx, ny, nz).
            z: z-coordinate array. Has shape (nx, ny, nz).
            t: Optional time at which the initial condition is evaluated.

        Returns:
            Array of initial conditions with shape (nvars, nx, ny, nz).
        """
        return self.ic(idx, x, y, z, t, xp=self.xp)

    def _callable_ic_with_passives(
        self,
        idx: VariableIndexMap,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        t: Optional[float],
    ) -> ArrayLike:
        """
        Helper function to call the initial condition function with passive variables.

        Args:
            idx: VariableIndexMap object.
            x: x-coordinate array. Has shape (nx, ny, nz).
            y: y-coordinate array. Has shape (nx, ny, nz).
            z: z-coordinate array. Has shape (nx, ny, nz).
            t: Optional time at which the initial condition is evaluated.

        Returns:
            Array of initial conditions with shape (nvars, nx, ny, nz).
        """
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
        Callable for initial conditions as conservative variables.

        Args:
            idx: VariableIndexMap object.
            x: x-coordinate array. Has shape (nx, ny, nz).
            y: y-coordinate array. Has shape (nx, ny, nz).
            z: z-coordinate array. Has shape (nx, ny, nz).
            t: Optional time at which the initial condition is evaluated.

        Returns:
            Array of initial conditions in conservative variables with shape
                (nvars, nx, ny, nz).
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

        mode_list: List[Tuple[BCs, BCs]] = []
        primitive_dirichlet_list: List[Tuple[Optional[Field], Optional[Field]]] = []

        def f0(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float],
        ) -> ArrayLike:
            return self.callable_ic(idx, x, y, z, 0.0)

        for bci, fi in zip((bcx, bcy, bcz), (x_dirichlet, y_dirichlet, z_dirichlet)):
            mode_list_i: List[BCs] = []
            prim_drch_list_i: List[Optional[Field]] = []
            for j, (bc, f) in enumerate(zip(as_pair(bci), as_pair(fi))):
                if bc == "ic":
                    mode_list_i.append("dirichlet")
                    prim_drch_list_i.append(f0)
                else:
                    mode_list_i.append(bc)
                    prim_drch_list_i.append(f)

            mode_list.append((mode_list_i[0], mode_list_i[1]))
            primitive_dirichlet_list.append((prim_drch_list_i[0], prim_drch_list_i[1]))

        def normalize_conservative_dirichlet(f: Optional[Field]) -> Optional[Field]:
            if f is None:
                return None

            def f_conservative(
                idx: VariableIndexMap,
                x: ArrayLike,
                y: ArrayLike,
                z: ArrayLike,
                t: Optional[float],
            ):
                return self.conservatives_from_primitives(f(idx, x, y, z, t))

            return f_conservative

        mode = (mode_list[0], mode_list[1], mode_list[2])
        primitive_dirichlet_arg = (
            primitive_dirichlet_list[0],
            primitive_dirichlet_list[1],
            primitive_dirichlet_list[2],
        )
        conservative_dirichlet_arg = (
            (
                normalize_conservative_dirichlet(primitive_dirichlet_arg[0][0]),
                normalize_conservative_dirichlet(primitive_dirichlet_arg[0][1]),
            ),
            (
                normalize_conservative_dirichlet(primitive_dirichlet_arg[1][0]),
                normalize_conservative_dirichlet(primitive_dirichlet_arg[1][1]),
            ),
            (
                normalize_conservative_dirichlet(primitive_dirichlet_arg[2][0]),
                normalize_conservative_dirichlet(primitive_dirichlet_arg[2][1]),
            ),
        )

        mesh = self.mesh

        self.bc_pad_width: Tuple[int, int, int] = (
            mesh.x_slab_depth,
            mesh.y_slab_depth,
            mesh.z_slab_depth,
        )
        self.bc_mode: Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]] = mode
        self.bc_f: Dict[
            str,
            Tuple[
                Tuple[Optional[Field], Optional[Field]],
                Tuple[Optional[Field], Optional[Field]],
                Tuple[Optional[Field], Optional[Field]],
            ],
        ] = {
            "primitive": primitive_dirichlet_arg,
            "conservative": conservative_dirichlet_arg,
        }

    def _init_riemann_solver(self, riemann_solver: str):
        if not hasattr(self, riemann_solver):
            raise ValueError(f"Riemann solver {riemann_solver} not implemented.")
        self.riemann_func = getattr(self, riemann_solver)

    @MethodTimer(cat="FiniteVolumeSolver.riemann_solver")
    def riemann_solver(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        *,
        out: ArrayLike,
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
        out[...] = self.riemann_func(wl, wr, dim)

    def _init_slope_limiting(
        self,
        MUSCL: bool,
        ZS: bool,
        adaptive_dt: bool,
        max_dt_revisions: int,
        MOOD: bool,
        cascade: Literal["first-order", "muscl", "full"],
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
            PAD_bounds = np.array([[-np.inf, np.inf] for _ in range(self.nvars)])
            for var, (lb, ub) in PAD.items():
                PAD_bounds[idx(var), 0] = lb if lb is not None else -np.inf
                PAD_bounds[idx(var), 1] = ub if ub is not None else np.inf
            PAD_bounds = np.expand_dims(
                PAD_bounds, axis=(1, 2, 3)
            )  # shape (nvars, 1, 1, 1, 2)
            self.arrays.add("PAD_bounds", PAD_bounds)

        # Zhang-Shu (a priori limiting)
        if ZS:
            self.ZS = True
            self.base_scheme.limiter = "zhang-shu"
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

            # define MOOD cascade
            fallback_schemes: List[InterpolationScheme]
            if cascade == "first-order":
                fallback_schemes = [
                    polyInterpolationScheme(
                        p=0,
                        flux_recipe=self.base_scheme.flux_recipe,
                        gauss_legendre=self.base_scheme.gauss_legendre,
                    )
                ]
            elif cascade == "muscl":
                fallback_schemes = [
                    musclInterpolationScheme(
                        flux_recipe=self.base_scheme.flux_recipe,
                        limiter="minmod",
                    )
                ]
            elif cascade == "full":
                fallback_schemes = [
                    polyInterpolationScheme(
                        p=pi,
                        flux_recipe=self.base_scheme.flux_recipe,
                        gauss_legendre=self.base_scheme.gauss_legendre,
                    )
                    for pi in range(self.p - 1, -1, -1)
                ]
            cascade_list = [self.base_scheme] + fallback_schemes

            # initialize MOOD configuration
            self.MOOD_config = MOODConfig(
                max_iters=max_MOOD_iters,
                cascade=cascade_list,
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
        if self.cupy:
            self.arrays.transfer_to_device("gpu")

        arrays = self.arrays

        # initialize flux arrays
        nvars, nx, ny, nz = self.nvars, self.mesh.nx, self.mesh.ny, self.mesh.nz
        arrays.add("F", np.empty((nvars, nx + 1, ny, nz)))
        arrays.add("G", np.empty((nvars, nx, ny + 1, nz)))
        arrays.add("H", np.empty((nvars, nx, ny, nz + 1)))

        # initialize snapshot arrays
        assert arrays["u"].shape == (nvars, nx, ny, nz)
        arrays.add("dudt", np.empty((nvars, nx, ny, nz)))

        # initialize workspace arrays
        _nx_, _ny_, _nz_ = self.mesh._nx_, self.mesh._ny_, self.mesh._nz_
        max_nodes = self.nodes_per_face(self.base_scheme)
        max_ninterps = 2 * max_nodes
        SED_buffer_size = {1: 4, 2: 6, 3: 7}[self.mesh.ndim] + 1
        MOOD_buffer_size = {1: 4, 2: 6, 3: 7}[self.mesh.ndim] + 4
        buffer_size = max(SED_buffer_size, max_nodes)
        if self.MOOD:
            buffer_size = max(buffer_size, MOOD_buffer_size)

        arrays.add("_u_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_ucc_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_wcc_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_w_", np.empty((nvars, _nx_, _ny_, _nz_)))
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
                arrays.add("F_" + scheme.key(), np.empty((nvars, nx + 1, ny, nz)))
                arrays.add("G_" + scheme.key(), np.empty((nvars, nx, ny + 1, nz)))
                arrays.add("H_" + scheme.key(), np.empty((nvars, nx, ny, nz + 1)))
            arrays.add(
                "_cascade_idx_array_", np.zeros((1, _nx_, _ny_, _nz_), dtype=int)
            )
            arrays.add("_mask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1), dtype=int))

        # fill arrays with NaNs to sabotage unpermitted use
        arrays["F"].fill(np.nan)
        arrays["G"].fill(np.nan)
        arrays["H"].fill(np.nan)
        arrays["dudt"].fill(np.nan)
        arrays["_u_"].fill(np.nan)
        arrays["_ucc_"].fill(np.nan)
        arrays["_wcc_"].fill(np.nan)
        arrays["_w_"].fill(np.nan)
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

    def nodes_per_face(self, scheme: InterpolationScheme) -> int:
        """ """
        if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
            p = scheme.p
            return (-(-(p + 1) // 2)) ** (self.mesh.ndim - 1)
        return 1

    def _init_snapshots(self, log_every_step: bool):
        self.log_every_step = log_every_step

        # allocate keys from collect_minisnapshot_data
        keys = self.collect_minisnapshot_data().keys()
        for key in keys:
            self.minisnapshots[key] = []

    @abstractmethod
    def define_vars(self) -> VariableIndexMap:
        """
        Define the names of the solver variables.

        Returns:
            VariableIndexMap object.
        """
        pass

    @abstractmethod
    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Convert primitive variables to conservative variables.

        Args:
            w: Array of primitive variables.

        Returns:
            Array of conservative variables.
        """
        pass

    @abstractmethod
    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u: Array of conservative variables.

        Returns:
            Array of primitive variables.
        """
        pass

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
        """
        raise NotImplementedError("Riemann solver not implemented.")

    @abstractmethod
    def log_quantity(self) -> Dict[str, Any]:
        """
        Log a quantity at the end of each time step.

        Returns:
            Dictionary of logged quantities.
        """
        pass

    def update_workspaces(
        self,
        t: float,
        u: ArrayLike,
        scheme: InterpolationScheme,
    ):
        """
        Update the workspace arrays `_u_`, '_ucc_', '_wcc_', and `_w_` based on the
        provided conservative variables `u` and the interpolation scheme.

        Args:
            t: Time value.
            u: Array of conservative, cell-averaged values. Has shape
                (nvars, nx, ny, nz).
            scheme: Interpolation scheme to use.

        Returns:
            None. The workspace arrays `_u_` and maybe `_w_` are updated in place.
        """
        active_dims = self.mesh.active_dims
        interior = self.interior
        interior0 = interior + (0,)
        p = scheme.p
        xp = self.xp

        _u_ = self.arrays["_u_"]
        _ucc_ = self.arrays["_ucc_"]  # shape (..., 1)
        _wcc_ = self.arrays["_wcc_"]  # shape (..., 1)
        _w_ = self.arrays["_w_"]  # shape (..., 1)
        _wbf_ = self.arrays["buffer"][..., :1]
        buffer = self.arrays["buffer"][..., 1:]

        # 0) conservatives + BC
        _u_[interior] = u
        self.inplace_apply_bc(t, _u_, scheme=scheme)

        # 1) centroids
        fv.interpolate_cell_centers(xp, _u_, active_dims, p, out=_ucc_, buffer=buffer)
        _wcc_[...] = self.primitives_from_conservatives(_ucc_)

        # 3) primitive FV averages
        if scheme.lazy_primitives:
            _wbf_[..., 0] = self.primitives_from_conservatives(_u_)
        else:
            fv.integrate_fv_averages(
                xp, _wcc_, active_dims, p, out=_wbf_, buffer=buffer
            )

        # 4) write interior, then primitive BCs
        _w_[interior] = _wbf_[interior0]
        self.inplace_apply_bc(t, _w_, scheme=scheme, primitives=True)

    @MethodTimer(cat="FiniteVolumeSolver.inplace_apply_bc")
    def inplace_apply_bc(
        self,
        t: float,
        u: ArrayLike,
        scheme: InterpolationScheme,
        primitives: bool = False,
        pointwise: bool = False,
        cell_region: Optional[
            Literal["xl", "xr", "yl", "yr", "zl", "zr", "center"]
        ] = None,
    ):
        """
        Apply boundary conditions to the provided array `u` in place.

        Args:
            t: Time value.
            u: Array of conservative or primitive variables. Has shape
                (nvars, nx, ny, nz).
            scheme: Interpolation scheme to use for the boundary conditions.
            primitives: Whether the provided array `u` contains primitive variables.
            pointwise: Whether to apply pointwise boundary conditions.
            cell_region: Optional cell region to apply the boundary conditions to.
                Possible values:
                - "xl": Left x-face.
                - "xr": Right x-face.
                - "yl": Left y-face.
                - "yr": Right y-face.
                - "zl": Left z-face.
                - "zr": Right z-face.
                - "center": Apply to cell centers.

        Returns:
            None. The array `u` is modified in place with the boundary conditions
                applied.
        """
        # determine polynomial degree
        if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
            p = scheme.p
        else:
            p = 0

        # determine Dirichlet mode
        dirichlet_mode: Literal["fv-averages", "cell-centers", "face-nodes"] = (
            "fv-averages"
            if not pointwise
            else ("cell-centers" if cell_region == "center" else "face-nodes")
        )

        region_to_face_dim: dict[str, Literal["x", "y", "z"]] = {
            "xl": "x",
            "xr": "x",
            "yl": "y",
            "yr": "y",
            "zl": "z",
            "zr": "z",
        }

        region_to_face_pos: dict[str, Literal["l", "r"]] = {
            "xl": "l",
            "yl": "l",
            "zl": "l",
            "xr": "r",
            "yr": "r",
            "zr": "r",
        }

        region = str(cell_region) if cell_region is not None else ""

        face_dim = region_to_face_dim.get(region, None)
        face_pos = region_to_face_pos.get(region, None)

        # apply boundary conditions
        apply_bc(
            _u_=u,
            pad_width=self.bc_pad_width,
            mode=self.bc_mode,
            dirichlet_mode=dirichlet_mode,
            f=self.bc_f["primitive" if primitives else "conservative"],
            variable_index_map=self.variable_index_map,
            mesh=self.mesh,
            t=t,
            face_dim=face_dim,
            face_pos=face_pos,
            p=p,
        )

    @MethodTimer(cat="FiniteVolumeSolver.inplace_interpolate_faces")
    def inplace_interpolate_faces(
        self,
        u: ArrayLike,
        dim: Literal["x", "y", "z"],
        scheme: InterpolationScheme,
        conv_to_prim: bool = False,
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
            scheme: Interpolation scheme to use for the interpolation.
            conv_to_prim: Whether to convert the interpolated values to primitive
                variables after interpolation.

        Returns:
            None. Modifies `self.arrays[dim + "_nodes"]` in place.

        Notes:
            - `self.arrays["buffer"]` is used as a temporary buffer for sweeps.
        """
        xp = self.xp
        adims = self.mesh.active_dims

        out = self.arrays[dim + "_nodes"]
        buffer = self.arrays["buffer"]

        # perform interpolation
        if isinstance(scheme, polyInterpolationScheme):
            p = scheme.p
            if scheme.gauss_legendre:
                fv.interpolate_GaussLegendre_nodes(
                    xp, u, dim, adims, p, out=out, buffer=buffer
                )
            else:
                fv.interpolate_face_centers(
                    xp, u, dim, adims, p, out=out, buffer=buffer
                )
        elif isinstance(scheme, musclInterpolationScheme):
            limiter: Callable[[ModuleType, ArrayLike, ArrayLike], ArrayLike]
            if scheme.limiter == "minmod":
                limiter = minmod
            elif scheme.limiter == "moncen":
                limiter = moncen
            else:
                raise ValueError(f"Unknown MUSCL limiter: {scheme.limiter}")
            fv.interpolate_muscl_faces(xp, limiter, u, dim, out=out, buffer=buffer)
        else:
            raise ValueError(f"Unknown interpolation scheme: {scheme}")

        if conv_to_prim:
            # convert to primitive variables if requested
            out[...] = self.primitives_from_conservatives(out)

    @MethodTimer(cat="FiniteVolumeSolver.zhang_shu_limiter")
    def zhang_shu_limiter(self, p: int, primitives: bool = False):
        """
        Limit the face node arrays.

        Args:
            p: Polynomial degree of the spatial discretization.
            primitives: Whether the face nodes are in primitive variables.

        Returns:
            None. Modifies `self.arrays["x_nodes"]`, `self.arrays["y_nodes"]`, and
            `self.arrays["z_nodes"]` in place.
        """
        xp = self.xp
        mesh = self.mesh
        ZhangShu_config = self.ZhangShu_config

        # define array references
        u = self.arrays["_w_"] if primitives else self.arrays["_u_"]
        centroid = self.arrays["_wcc_"] if primitives else self.arrays["_ucc_"]
        x = self.arrays["x_nodes"] if mesh.x_is_active else None
        y = self.arrays["y_nodes"] if mesh.y_is_active else None
        z = self.arrays["z_nodes"] if mesh.z_is_active else None
        theta = self.arrays["theta"]
        buffer = self.arrays["buffer"]

        # compute centroid then compute theta
        fv.interpolate_cell_centers(
            xp, u, mesh.active_dims, p, out=centroid, buffer=buffer
        )
        compute_theta(
            xp,
            u,
            centroid,
            x,
            y,
            z,
            out=theta,
            buffer=buffer,
            tol=ZhangShu_config.tol,
            include_corners=ZhangShu_config.include_corners,
            SED=ZhangShu_config.SED,
        )

        # limit the face nodes
        if x is not None:
            x[...] = zhang_shu_operator(x, u[..., np.newaxis], theta)
        if y is not None:
            y[...] = zhang_shu_operator(y, u[..., np.newaxis], theta)
        if z is not None:
            z[...] = zhang_shu_operator(z, u[..., np.newaxis], theta)

    def validate_nodal_bc(
        self,
        t: float,
        dim: Literal["x", "y", "z"],
        scheme: InterpolationScheme,
    ):
        """
        Ensure cell face nodes are primitive variables and apply boundary conditions.

        Args:
            t: Time value.
            dim: Dimension in which to interpolate the face nodes. Possible values:
                - "x": Interpolate in the x-direction.
                - "y": Interpolate in the y-direction.
                - "z": Interpolate in the z-direction.
            p: Polynomial degree of the spatial discretization.
            scheme: Interpolation scheme to use for the interpolation.

        Returns:
            None. Modifies `self.arrays[dim + "_nodes"]` in place.
        """
        n = self.nodes_per_face(scheme)
        conv_to_prim = scheme.flux_recipe == 1

        nodes = self.arrays[dim + "_nodes"]

        if conv_to_prim:
            nodes[...] = self.primitives_from_conservatives(nodes)

        ul = nodes[crop(4, (None, n))]
        ur = nodes[crop(4, (n, 2 * n))]

        self.inplace_apply_bc(
            t,
            ul,
            scheme=scheme,
            primitives=True,
            pointwise=True,
            cell_region=dim + "l",
        )
        self.inplace_apply_bc(
            t,
            ur,
            scheme=scheme,
            primitives=True,
            pointwise=True,
            cell_region=dim + "r",
        )

    @MethodTimer(cat="FiniteVolumeSolver.inplace_integrate_fluxes")
    def inplace_integrate_fluxes(
        self, dim: Literal["x", "y", "z"], scheme: InterpolationScheme
    ):
        """
        Integrate the flux nodes at the face centers using the transverse quadrature or
        Gauss-Legendre quadrature.

        Args:
            dim: Dimension in which to integrate the flux nodes. Possible values:
                - "x": Integrate in the x-direction.
                - "y": Integrate in the y-direction.
                - "z": Integrate in the z-direction.
            scheme: Interpolation scheme to use for the integration.


        Returns:
            None. Modifies `self.arrays[{"x": "F", ...}[dim]]` in place.

        Notes:
            - `self.arrays["buffer"]` is used as a temporary buffer for the sweeps.
            - `self.arrays[{"x": "F_wrkspce", ...}[dim]]` is used as a workspace for
            the flux integration.
            - `self.arrays[dim + "_nodes"]` contains the interpolated primitive
                variables at the face nodes.

        """
        n = self.nodes_per_face(scheme)
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
        self.riemann_solver(left_state, right_state, dim, out=left_state)

        # perform the integration
        if self.mesh.ndim == 1:
            flux_workspace[...] = left_state
        else:
            if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
                fv.integrate_GaussLegendre_nodes(
                    xp,
                    left_state,
                    dim,
                    self.mesh.active_dims,
                    scheme.p,
                    out=flux_workspace[..., 0],
                )
            elif isinstance(
                scheme, (polyInterpolationScheme, musclInterpolationScheme)
            ):
                fv.transversely_integrate_nodes(
                    self.xp,
                    left_state[..., 0],
                    dim,
                    self.mesh.active_dims,
                    scheme.p,
                    out=flux_workspace,
                    buffer=right_state,
                )
            else:
                raise ValueError(
                    f"Unknown interpolation scheme: {scheme}. "
                    "Expected polyInterpolationScheme or musclInterpolationScheme."
                )

        out[...] = flux_workspace[self.flux_interior[dim]][..., 0]

    @MethodTimer(cat="FiniteVolumeSolver.inplace_compute_fluxes")
    def inplace_compute_fluxes(self, t: float, scheme: InterpolationScheme):
        """
        Write fluxes based on the current workspaces `_u_` and maybe `_w_` to the flux
            arrays `F`, `G`, and `H`.

        Args:
            t: Time value.
            scheme: Interpolation scheme to use for the flux computation.

        Returns:
            None. The flux arrays `F`, `G`, and `H` are updated in place.
        """
        _u_ = self.arrays["_u_"]
        _w_ = self.arrays["_w_"]

        # update the '_nodes' arrays with the interpolated face nodes
        primitive_nodes = scheme.flux_recipe in (2, 3)
        for dim in self.mesh.active_dims:
            if scheme.flux_recipe == 1:
                self.inplace_interpolate_faces(_u_, dim, scheme)
            elif scheme.flux_recipe == 2:
                self.inplace_interpolate_faces(_u_, dim, scheme, conv_to_prim=True)
            elif scheme.flux_recipe == 3:
                self.inplace_interpolate_faces(_w_, dim, scheme)
            else:
                raise ValueError(f"Unknown scheme flux_recipe: {scheme.flux_recipe}")

        # Zhang-Shu limiter
        if scheme.limiter == "zhang-shu":
            self.zhang_shu_limiter(scheme.p, primitive_nodes)

        # integrate the fluxes at the cell faces
        for dim in self.mesh.active_dims:
            self.validate_nodal_bc(t, dim, scheme)
            self.inplace_integrate_fluxes(dim, scheme)

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
            self._add_flux_divergence(dim, out=out)

        return out

    def _add_flux_divergence(self, dim: Literal["x", "y", "z"], *, out: ArrayLike):
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
        self.update_workspaces(t, u, self.base_scheme)
        self.inplace_compute_fluxes(t, self.base_scheme)

        if self.MOOD:
            self.MOOD_loop(t)

        out = self.compute_RHS().copy()
        return out

    def build_opening_message(self) -> str:
        """
        Build the opening message for the FV solver.
        """
        return ""

    def build_update_message(self) -> str:
        """
        Build the update message for the FV solver.
        """
        return f"Step #{self.n_steps} @ t={self.t:<.2e} | dt={self.dt:<.2e}"

    def build_closing_message(self) -> str:
        """
        Build the closing message for the FV solver.
        """
        return self.build_update_message() + " | (done)"

    def adaptive_dt_criterion(self, tnew: float, unew: ArrayLike) -> bool:
        """
        Returns whether unew is free of PAD violations.

        Args:
            tnew: Unused time value for the new state.
            unew: Array of conservative, cell-averaged values. Shape:
                (nvars, nx, ny, nz).

        Returns:
            True if unew satisfies the PAD criterion, False otherwise.
        """
        xp = self.xp
        PAD_bounds = self.ZhangShu_config.PAD_bounds
        PAD_atol = self.ZhangShu_config.PAD_atol

        if PAD_bounds is None:
            raise ValueError(
                "PAD bounds are not set. Please provide PAD bounds in the "
                "ZhangShuConfig."
            )

        PAD_violations = self.arrays["buffer"][self.interior][..., 0]
        MOOD.inplace_PAD_violations(xp, unew, PAD_bounds, PAD_atol, out=PAD_violations)

        return not xp.any(PAD_violations < 0)

    def adaptive_dt_revision(self, t, u, dt):
        """
        Returns dt/2 if the maximum number of adaptive timestep revisions hasn't been
        exceeded.

        Args:
            t: Unused time value.
            u: Unused array of conservative, cell-averaged values.
            dt: Proposed timestep size.

        Returns:
            Half the proposed timestep size if the maximum number of revisions hasn't
            been exceeded, otherwise raises a ValueError.
        """
        n_dt_revisions = self.n_dt_revisions
        max_dt_revisions = self.ZhangShu_config.max_dt_revisions

        if n_dt_revisions < max_dt_revisions:
            return dt / 2
        raise ValueError(
            f"Failed to satisfy `dt_criterion` in {max_dt_revisions} iterations."
        )

    def called_at_end_of_step(self):
        """
        Overwrite `called_at_end_of_step` of the ODE solver to synchronize the GPU if
        using CuPy, and to perform any additional cleanup or logging.
        """
        if self.cupy:
            self.xp.cuda.Device().synchronize()
        super().called_at_end_of_step()

    def snapshot(self):
        """
        Simple snapshot method that writes the solution to `self.snapshots` keyed by
        the current time value.
        """
        interior = self.interior
        interior0 = interior + (0,)

        self.update_workspaces(self.t, self.arrays["u"], self.base_scheme)

        # store the snapshot
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "ucc": self.arrays.get_numpy_copy("_ucc_")[interior0],
            "wcc": self.arrays.get_numpy_copy("_wcc_")[interior0],
            "w": self.arrays.get_numpy_copy("_w_")[interior],
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
            data["n_MOOD_iters"] = self.MOOD_config.iter_count
            data["nfine_MOOD_iters"] = self.MOOD_config.iter_count_hist

        if self.log_every_step:
            data.update(self.log_quantity())

        return data

    def reset_stepwise_logs(self):
        """
        Reset logs that are incremented at the end of each step.
        """
        super().reset_stepwise_logs()

        if hasattr(self, "MOOD") and self.MOOD:
            self.MOOD_config.reset_iter_count_hist()

    def reset_substepwise_logs(self):
        """
        Reset logs that are incremented at the end of each substep.
        """
        super().reset_substepwise_logs()

        self.n_updates = 0

        if hasattr(self, "ZS") and self.ZS:
            self.arrays["theta_log"].fill(0.0)

        if hasattr(self, "MOOD") and self.MOOD:
            self.arrays["troubles_log"].fill(0)
            self.MOOD_config.reset_iter_count()

    def increment_substepwise_logs(self):
        """
        Increment logs at the end of each substep.
        """
        super().increment_substepwise_logs()

        xp = self.xp

        self.n_updates += self.mesh.size

        if self.ZS:
            theta = self.arrays["theta"]
            theta_log = self.arrays["theta_log"]
            xp.add(theta_log, theta, out=theta_log)

        if self.MOOD:
            troubles = self.arrays["troubles"]
            troubles_log = self.arrays["troubles_log"]
            xp.add(troubles_log, troubles, out=troubles_log)

            self.MOOD_config.increment_iter_count_hist()

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
