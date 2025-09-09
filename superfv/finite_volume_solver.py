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
from .interpolation_schemes import InterpolationScheme, polyInterpolationScheme
from .mesh import UniformFVMesh, xyz_tup
from .slope_limiting import MOOD
from .slope_limiting.MOOD import MOODConfig, MOODState
from .slope_limiting.muscl import (
    compute_limited_slopes,
    musclConfig,
    musclInterpolationScheme,
)
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
        MUSCL_limiter: Literal["minmod", "moncen"] = "minmod",
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
            MUSCL: Whether to use the MUSCL scheme as the base scheme. Overrides `p`,
                `flux_recipe`, and `lazy_primitives`. The `flux_recipe` options become:
                - `flux_recipe=1`: Slope limiting is performed on conservative slopes.
                - `flux_recipe=2`: Slope limiting is performed on primitive slopes.
                - `flux_recipe=3`: `flux_recipe=2` is used.
            MUSCL_limiter: Slope limiter used for the MUSCL scheme, either for the base
                scheme or the MOOD cascade. Options include:
                - "minmod"
                - "moncen"
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
        self._init_active_dims(nx, ny, nz)
        self._init_ic_callables(ic, ic_passives)
        self._init_array_management(cupy)
        self._init_spatial_discretization(
            p,
            flux_recipe,
            GL,
            lazy_primitives,
            MUSCL,
            MUSCL_limiter,
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
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, CFL)
        self._init_bc(bcx, bcy, bcz, x_dirichlet, y_dirichlet, z_dirichlet)
        self._init_array_allocation()
        self._init_snapshots(log_every_step)
        self._init_riemann_solver(riemann_solver)

    def _init_active_dims(self, nx: int, ny: int, nz: int):
        self.active_dims = tuple(d for d, n in zip(xyz_tup, (nx, ny, nz)) if n > 1)
        self.inactive_dims = tuple(d for d in xyz_tup if d not in self.active_dims)

    def _init_ic_callables(
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
        p: int,
        flux_recipe: Literal[1, 2, 3],
        GL: bool,
        lazy_primitives: bool,
        MUSCL: bool,
        MUSCL_limiter: Literal["minmod", "moncen"],
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
        self.base_scheme: InterpolationScheme

        idx = self.variable_index_map
        idx.add_var_to_group("limiting", [])

        null_MOOD_config = MOODConfig(
            cascade=[], max_iters=0, NAD=False, PAD=False, SED=False
        )
        self.MOOD_state = MOODState(config=null_MOOD_config)
        self.MOOD = False

        # first-order scheme early escape
        if p == 0:
            self._init_unlimited_scheme(0, flux_recipe, GL, lazy_primitives)
            return

        # init a priori scheme
        if MUSCL:
            if ZS:
                raise ValueError("MUSCL scheme cannot be combined with ZS.")
            self._init_muscl_scheme(p, flux_recipe, MUSCL_limiter, SED)
        elif ZS:
            self._init_PAD(PAD)
            self._init_zhang_shu_scheme(
                p,
                flux_recipe,
                GL,
                lazy_primitives,
                include_corners,
                PAD,
                SED,
                adaptive_dt,
                max_dt_revisions,
                PAD_atol,
            )
        else:
            self._init_unlimited_scheme(p, flux_recipe, GL, lazy_primitives)

        # init a posteriori scheme
        if MOOD:
            if not ZS:
                self._init_PAD(PAD)
            self._init_MOOD(
                MOOD,
                cascade,
                MUSCL_limiter,
                max_MOOD_iters,
                NAD,
                PAD,
                SED,
                NAD_rtol,
                NAD_atol,
                PAD_atol,
                global_dmp,
                include_corners,
            )

        # add limiting variables to the index map
        if limiting_vars == "all":
            limiting_vars = tuple(idx.var_idx_map.keys())
        elif limiting_vars == "actives":
            limiting_vars = tuple(idx.group_var_map["primitives"])
        idx.add_var_to_group("limiting", limiting_vars)

    def _init_unlimited_scheme(
        self, p: int, flux_recipe: Literal[1, 2, 3], GL: bool, lazy_primitives: bool
    ):
        self.p = p
        self.base_scheme = polyInterpolationScheme(
            p=p,
            flux_recipe=flux_recipe,
            limiter_config=None,
            gauss_legendre=GL,
            lazy_primitives=lazy_primitives,
        )

    def _init_muscl_scheme(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        MUSCL_limiter: Optional[Literal["minmod", "moncen"]],
        SED: bool,
    ):
        if p != 1:
            warnings.warn("MUSCL overrides p to be 1.")
        if flux_recipe == 3:
            warnings.warn("MUSCL overrides flux_recipe 3 to be 2.")
            flux_recipe = 2

        self.p = 1
        self.base_scheme = musclInterpolationScheme(
            flux_recipe=flux_recipe,
            limiter_config=musclConfig(limiter=MUSCL_limiter, SED=SED),
        )

    def _init_zhang_shu_scheme(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        GL: bool,
        lazy_primitives: bool,
        include_corners: bool,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        SED: bool,
        adaptive_dt: bool,
        max_dt_revisions: int,
        PAD_atol: float,
    ):
        self.p = p
        self.base_scheme = polyInterpolationScheme(
            p=p,
            flux_recipe=flux_recipe,
            limiter_config=ZhangShuConfig(
                SED=SED,
                adaptive_dt=adaptive_dt,
                max_dt_revisions=max_dt_revisions,
                include_corners=include_corners,
                PAD_bounds=None if PAD is None else self.arrays["PAD_bounds"],
                PAD_atol=PAD_atol,
            ),
            gauss_legendre=GL,
            lazy_primitives=lazy_primitives,
        )

    def _init_MOOD(
        self,
        MOOD: bool,
        cascade: Literal["first-order", "muscl", "full"],
        MUSCL_limiter: Literal["minmod", "moncen"],
        max_MOOD_iters: int,
        NAD: bool,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        SED: bool,
        NAD_rtol: float,
        NAD_atol: float,
        PAD_atol: float,
        global_dmp: bool,
        include_corners: bool,
    ):
        if not MOOD:
            return

        base_scheme = self.base_scheme
        if not isinstance(base_scheme, polyInterpolationScheme):
            raise ValueError(
                "Base scheme must be an instance of polyInterpolationScheme."
            )

        fallback_schemes: List[InterpolationScheme]
        if cascade == "first-order":
            fallback_schemes = [
                polyInterpolationScheme(
                    p=0,
                    flux_recipe=base_scheme.flux_recipe,
                    gauss_legendre=base_scheme.gauss_legendre,
                    lazy_primitives=base_scheme.lazy_primitives,
                )
            ]
        elif cascade in ("muscl", "muscl1"):
            if base_scheme.flux_recipe == 3:
                warnings.warn("MUSCL overrides flux_recipe 3 to be 2.")
            muscl_flux_recipe: Literal[1, 2] = (
                2 if base_scheme.flux_recipe == 3 else base_scheme.flux_recipe
            )
            fallback_schemes = [
                musclInterpolationScheme(
                    flux_recipe=muscl_flux_recipe,
                    limiter_config=musclConfig(limiter=MUSCL_limiter, SED=False),
                )
            ]
            if cascade == "muscl1":
                fallback_schemes += [
                    polyInterpolationScheme(
                        p=0,
                        flux_recipe=base_scheme.flux_recipe,
                        gauss_legendre=base_scheme.gauss_legendre,
                        lazy_primitives=base_scheme.lazy_primitives,
                    )
                ]
        elif cascade == "full":
            fallback_schemes = [
                polyInterpolationScheme(
                    p=p,
                    flux_recipe=base_scheme.flux_recipe,
                    gauss_legendre=base_scheme.gauss_legendre,
                    lazy_primitives=base_scheme.lazy_primitives,
                )
                for p in range(base_scheme.p - 1, -1, -1)
            ]
        cascade_list = [self.base_scheme] + fallback_schemes

        MOOD_config = MOODConfig(
            cascade=cascade_list,
            max_iters=max_MOOD_iters,
            NAD=NAD,
            PAD=PAD is not None,
            SED=SED,
            NAD_rtol=NAD_rtol,
            NAD_atol=NAD_atol,
            PAD_atol=PAD_atol,
            PAD_bounds=None if PAD is None else self.arrays["PAD_bounds"],
            global_dmp=global_dmp,
            include_corners=include_corners,
        )
        self.MOOD_state = MOODState(config=MOOD_config)
        self.MOOD = True

    def _init_PAD(
        self, PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]]
    ):
        if PAD is None:
            return
        idx = self.variable_index_map
        PAD_bounds = np.array([[-np.inf, np.inf] for _ in range(self.nvars)])
        for var, (lb, ub) in PAD.items():
            PAD_bounds[idx(var), 0] = lb if lb is not None else -np.inf
            PAD_bounds[idx(var), 1] = ub if ub is not None else np.inf
        PAD_bounds = np.expand_dims(
            PAD_bounds, axis=(1, 2, 3)
        )  # shape (nvars, 1, 1, 1, 2)
        self.arrays.add("PAD_bounds", PAD_bounds)

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

        p = self.base_scheme.p

        # determine slab depth
        stencil_depth = -2 * (-p // 2)
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
            active_dims=self.active_dims,
            slab_depth=slab_depth,
            array_manager=self.arrays,
        )

        # assign attributes
        self.CFL: float = CFL

        # interior workspace view
        self.interior = merge_slices(
            *[
                crop(DIM_TO_AXIS[dim], (self.mesh.slab_depth, -self.mesh.slab_depth))
                for dim in self.active_dims
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
                        for d in self.active_dims
                        if d != dim
                    ],
                )
                if self.mesh.ndim > 1
                else slice(None)
            )
            for dim in self.active_dims
        }

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

    def _init_array_allocation(self):
        if self.cupy:
            self.arrays.transfer_to_device("gpu")

        idx = self.variable_index_map
        scheme = self.base_scheme
        mesh = self.mesh
        arrays = self.arrays

        # initialize regular mesh arrays
        nvars, nx, ny, nz = self.nvars, mesh.nx, mesh.ny, mesh.nz
        arrays.add("dudt", np.empty((nvars, nx, ny, nz)))
        arrays.add("sum_of_s_over_h", np.empty((nx, ny, nz)))

        # initialize flux arrays
        arrays.add("F", np.empty((nvars, nx + 1, ny, nz)))
        arrays.add("G", np.empty((nvars, nx, ny + 1, nz)))
        arrays.add("H", np.empty((nvars, nx, ny, nz + 1)))

        # initialize workspace arrays
        max_nodes = self.nodes_per_face(scheme)
        max_ninterps = 2 * max_nodes
        buffer_size = self._compute_buffer_size(scheme)

        _nx_, _ny_, _nz_ = mesh._nx_, mesh._ny_, mesh._nz_
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
        arrays.add("theta_log", np.empty((nvars, nx, ny, nz)))
        arrays.add("troubles", np.zeros((1, nx, ny, nz), dtype=bool))
        arrays.add("troubles_log", np.zeros((1, nx, ny, nz), dtype=float))
        arrays.add("cascade_idx_log", np.zeros((1, nx, ny, nz), dtype=float))
        arrays.add("_troubles_", np.zeros((1, _nx_, _ny_, _nz_), dtype=bool))
        arrays.add("F_wrkspce", np.empty((nvars, nx + 1, _ny_, _nz_, max_nodes)))
        arrays.add("G_wrkspce", np.empty((nvars, _nx_, ny + 1, _nz_, max_nodes)))
        arrays.add("H_wrkspce", np.empty((nvars, _nx_, _ny_, nz + 1, max_nodes)))

        # MOOD arrays
        if self.MOOD:
            for scheme in self.MOOD_state.config.cascade:
                arrays.add("F_" + scheme.key(), np.empty((nvars, nx + 1, ny, nz)))
                arrays.add("G_" + scheme.key(), np.empty((nvars, nx, ny + 1, nz)))
                arrays.add("H_" + scheme.key(), np.empty((nvars, nx, ny, nz + 1)))
            arrays.add(
                "_cascade_idx_array_", np.zeros((1, _nx_, _ny_, _nz_), dtype=int)
            )
            arrays.add("_mask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1), dtype=int))

        # fill arrays with NaNs to sabotage unpermitted use
        arrays["dudt"].fill(np.nan)
        arrays["sum_of_s_over_h"].fill(np.nan)
        arrays["F"].fill(np.nan)
        arrays["G"].fill(np.nan)
        arrays["H"].fill(np.nan)
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

        # initialize the ODE solver with the initial condition array
        ic_arr = mesh.perform_GaussLegendre_quadrature(
            lambda x, y, z: self.conservative_callable_ic(idx, x, y, z, 0.0),
            node_axis=4,
            mesh_region="core",
            cell_region="interior",
            p=self.p,
        )

        super().__init__(ic_arr, self.arrays)  # defines "u"
        assert arrays["u"].shape == (nvars, nx, ny, nz)

        if getattr(self.base_scheme.limiter_config, "adaptive_dt", False):
            self._dt_criterion = self.adaptive_dt_criterion
            self._compute_revised_dt = self.adaptive_dt_revision

    def _compute_buffer_size(self, scheme: InterpolationScheme) -> int:
        mesh = self.mesh
        max_nodes = self.nodes_per_face(scheme)

        SED_buffer_cost = {1: 4, 2: 6, 3: 7}[mesh.ndim]
        base_buffer_size = SED_buffer_cost + 1

        buffer_size = max(base_buffer_size, max_nodes)

        MOOD_buffer_size = SED_buffer_cost + 4
        MUSCL_buffer_size = SED_buffer_cost + 10

        if isinstance(scheme.limiter_config, ZhangShuConfig):
            buffer_size = max(buffer_size, SED_buffer_cost)
        elif self.MOOD:
            buffer_size = max((buffer_size, MOOD_buffer_size, MUSCL_buffer_size))
        elif isinstance(scheme, musclInterpolationScheme):
            buffer_size = max(buffer_size, MUSCL_buffer_size)

        return buffer_size

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
    def log_quantity(self) -> Dict[str, Any]:
        """
        Log a quantity at the end of each time step.

        Returns:
            Dictionary of logged quantities.
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

    def flux_jvp(
        self,
        w: ArrayLike,
        vec: ArrayLike,
        dim: Literal["x", "y", "z"],
        *,
        primitives: bool = True,
    ) -> ArrayLike:
        """
        Jacobian-vector product for the primitive-variable quasilinear,
        dimensionally-split system
            dW/dt + A(W; [dim]) dW/d[dim] = 0,  W=[primitive_var1, ...]
        if `primitives=True`, or the conservative-variable quasilinear system
            dU/dt + A(U; [dim]) dU/d[dim] = 0,  U=[conservative_var1, ...]
        if `primitives=False`.

        Args:
            w: State array with shape (nvars, nx, ny, nz).
            vec: Vector to multiply with the flux Jacobian. Has shape (nvars,).
            dim: Dimension along which the flux Jacobian is computed. Can be "x", "y",
                or "z".
            primitives: Whether the state array `w` contains primitive variables.

        Returns:
            ArrayLike: The flux Jacobian-vector product A @ vec.
        """
        raise NotImplementedError(
            f"No `flux_jvp` method has been implemented in class {self.__class__.__name__}."
        )

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
        active_dims = self.active_dims
        interior = self.interior
        interior0 = interior + (0,)
        p = scheme.p
        xp = self.xp

        # determine if lazy primitives should be used
        if isinstance(scheme, polyInterpolationScheme):
            use_lazy_primitives = False
            if scheme.p in (0, 1) or scheme.lazy_primitives:
                use_lazy_primitives = True
        elif isinstance(scheme, musclInterpolationScheme):
            use_lazy_primitives = True
        else:
            raise ValueError("Unknown scheme type.")

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
        if use_lazy_primitives:
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
        scheme: polyInterpolationScheme,
        *,
        convert_to_primitives: bool = False,
    ):
        """
        Interpolate the face nodes at the opposing face centers and assign them to the
        appropriate array (`x_nodes`, `y_nodes`, or `z_nodes`).

        Args:
            t: Time value.
            u: Array of cell-averaged values including ghost cells. Has shape
                (nvars, nx, ny, nz).
            dim: Dimension along which to interpolate the opposing face nodes. Can be
                "x", "y", or "z".
            scheme: Interpolation scheme object.
            convert_to_primitives: Whether to convert the interpolated values to
                primitives.

        Notes:
            Let 'ni' be the number of interpolations on a given face. Then the
            appropriate face node array must have shape (nvars, nx, ny, nz, 2*ni). The
            left face interpolations are written along [:, :, :, :, :ni] and the right
            face interpolations are written along [:, :, :, :, ni:2*ni].
        """
        xp = self.xp
        adims = self.active_dims

        out = self.arrays[dim + "_nodes"]
        buffer = self.arrays["buffer"]

        # perform interpolation
        p = scheme.p
        if scheme.gauss_legendre:
            fv.interpolate_GaussLegendre_nodes(
                xp, u, dim, adims, p, out=out, buffer=buffer
            )
        else:
            fv.interpolate_face_centers(xp, u, dim, adims, p, out=out, buffer=buffer)

        if convert_to_primitives:
            # convert to primitive variables if requested
            out[...] = self.primitives_from_conservatives(out)

    @MethodTimer(cat="FiniteVolumeSolver.zhang_shu_limiter")
    def zhang_shu_limiter(
        self, scheme: polyInterpolationScheme, primitives: bool = False
    ):
        """
        Limit the face node arrays (`x_nodes`, `y_nodes`, `z_nodes`) using the
        Zhang-Shu limiter, theta, which is written to the `theta` array.

        Args:
            scheme: Interpolation scheme to use.
            primitives: Whether the face node arrays represent primitive variables.
        """
        if not isinstance(scheme.limiter_config, ZhangShuConfig):
            raise ValueError("Zhang-Shu limiter configuration is required.")

        xp = self.xp
        mesh = self.mesh

        # define array references
        average = self.arrays["_w_"] if primitives else self.arrays["_u_"]
        centroid = self.arrays["_wcc_"] if primitives else self.arrays["_ucc_"]
        x = self.arrays["x_nodes"] if mesh.x_is_active else None
        y = self.arrays["y_nodes"] if mesh.y_is_active else None
        z = self.arrays["z_nodes"] if mesh.z_is_active else None
        theta = self.arrays["theta"]
        buffer = self.arrays["buffer"]

        # compute centroid and theta
        fv.interpolate_cell_centers(
            xp, average, self.active_dims, scheme.p, out=centroid, buffer=buffer
        )
        compute_theta(
            xp,
            average,
            centroid,
            x,
            y,
            z,
            out=theta,
            buffer=buffer,
            config=scheme.limiter_config,
        )

        # limit the face nodes
        if x is not None:
            x[...] = zhang_shu_operator(x, average[..., np.newaxis], theta)
        if y is not None:
            y[...] = zhang_shu_operator(y, average[..., np.newaxis], theta)
        if z is not None:
            z[...] = zhang_shu_operator(z, average[..., np.newaxis], theta)

    def normalize_node_arrays(
        self,
        t: float,
        dim: Literal["x", "y", "z"],
        scheme: InterpolationScheme,
        *,
        primitives: bool = True,
    ):
        """
        Ensure the face node arrays (`x_nodes`, `y_nodes`, `z_nodes`) represent
        primitive variables and have valid boundary conditions.

        Args:
            t: Time value.
            dim: Dimension of the face node array to normalize ("x", "y", or "z").
            scheme: Interpolation scheme to use for the interpolation.
            convert_to_primitives: Whether to the face node arrays represent primitive
                variables. If not, they will be converted to conservative variables
                before boundary conditions are applied.
        """
        n = self.nodes_per_face(scheme)

        nodes = self.arrays[dim + "_nodes"]

        if not primitives:
            nodes[...] = self.primitives_from_conservatives(nodes)

        wl = nodes[crop(4, (None, n))]
        wr = nodes[crop(4, (n, 2 * n))]

        self.inplace_apply_bc(
            t,
            wl,
            scheme=scheme,
            primitives=True,
            pointwise=True,
            cell_region=dim + "l",
        )
        self.inplace_apply_bc(
            t,
            wr,
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
        Compute the flux nodes based on the primitive face node arrays
        (`x_nodes`, `y_nodes`, or `z_nodes`), perform the integration, and assign the
        result to the appropriate flux array (`F`, `G`, or `H`).

        Args:
            dim: Dimension along which to compute the fluxes of opposing faces. Can be
                "x", "y", or "z".
            scheme: Interpolation scheme to use for the integration.
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
                    self.active_dims,
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
                    self.active_dims,
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
        Write fluxes to their respective arrays (`F`, `G`, and `H`) based on the
        workspace arrays `_u_` and `_w_`.

        Args:
            t: Time value.
            scheme: Interpolation scheme to use for the flux computation.
        """
        interp_primitives = scheme.flux_recipe == 3
        limit_primitives = scheme.flux_recipe in (2, 3)
        averages = self.arrays["_w_"] if interp_primitives else self.arrays["_u_"]

        # update the face node arrays with the interpolated face nodes
        if isinstance(scheme, polyInterpolationScheme):
            for dim in self.active_dims:
                self.inplace_interpolate_faces(
                    averages,
                    dim,
                    scheme,
                    convert_to_primitives=limit_primitives and not interp_primitives,
                )
        elif isinstance(scheme, musclInterpolationScheme):
            self.inplace_reconstruct_muscl_faces(
                scheme, limit_primitives=limit_primitives
            )
        else:
            raise ValueError("Unknown interpolation scheme.")

        # limit the face nodes arrays with the Zhang-Shu limiter
        if isinstance(scheme.limiter_config, ZhangShuConfig):
            self.zhang_shu_limiter(scheme, primitives=limit_primitives)

        # compute the fluxes and assign them to their respective arrays
        for dim in self.active_dims:
            self.normalize_node_arrays(t, dim, scheme, primitives=limit_primitives)
            self.inplace_integrate_fluxes(dim, scheme)

    def MOOD_loop(self, t: float):
        """
        Perform the MOOD loop to detect and revise troubled cells.

        Args:
            t: Time value.
        """
        state = self.MOOD_state

        state.reset_MOOD_loop()
        for _ in range(state.config.max_iters):
            n_revisable, n_total = MOOD.detect_troubled_cells(self, t)
            if n_revisable > 0:
                MOOD.inplace_revise_fluxes(self, t)
                state.increment_MOOD_iteration()
            else:
                break
        state.update_troubled_cell_count(n_total)

    def compute_RHS(self) -> ArrayLike:
        """
        Compute the right-hand side of the ODE and write it to `self.arrays["dudt"]`.

        Returns:
            ArrayLike: The right-hand side of the ODE.
        """
        out = self.arrays["dudt"]
        out[...] = 0.0

        for dim in self.active_dims:
            self._add_flux_divergence(dim, out=out)

        return out

    def _add_flux_divergence(self, dim: Literal["x", "y", "z"], *, out: ArrayLike):
        """
        Add the flux divergence in the specified dimension to the output array.

        Args:
            dim: Dimension along which to compute the flux divergence. Can be
                "x", "y", or "z".
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
        scheme = self.base_scheme

        if not isinstance(scheme.limiter_config, ZhangShuConfig):
            raise ValueError(
                "PAD criterion requires a Zhang-Shu limiter configuration."
            )
        if scheme.limiter_config.PAD_bounds is None:
            raise ValueError(
                "PAD bounds are not set. Please provide PAD bounds in the "
                "ZhangShuConfig."
            )

        PAD_violations = self.arrays["buffer"][self.interior][..., 0]
        MOOD.inplace_PAD_violations(
            xp,
            self.primitives_from_conservatives(unew),
            scheme.limiter_config.PAD_bounds,
            scheme.limiter_config.PAD_atol,
            out=PAD_violations,
        )

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
        if not isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            raise ValueError(
                "PAD criterion requires a Zhang-Shu limiter configuration."
            )

        max_dt_revisions = self.base_scheme.limiter_config.max_dt_revisions
        n_dt_revisions = self.n_dt_revisions

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

        if (
            isinstance(self.base_scheme.limiter_config, ZhangShuConfig)
            and self.n_substeps > 1
        ):
            theta = self.arrays["theta_log"]
            theta[...] = theta / self.n_substeps
        if self.MOOD and self.n_substeps > 1:
            troubles = self.arrays["troubles_log"]
            cascade_idx = self.arrays["cascade_idx_log"]

            troubles[...] = troubles / self.n_substeps
            cascade_idx[...] = cascade_idx / self.n_substeps

        # store the snapshot
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "ucc": self.arrays.get_numpy_copy("_ucc_")[interior0],
            "wcc": self.arrays.get_numpy_copy("_wcc_")[interior0],
            "w": self.arrays.get_numpy_copy("_w_")[interior],
            "theta": self.arrays.get_numpy_copy("theta_log"),
            "troubles": self.arrays.get_numpy_copy("troubles_log"),
            "cascade": self.arrays.get_numpy_copy("cascade_idx_log"),
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
            state = self.MOOD_state

            if sum(state.iter_count_hist) != state.iter_count:
                raise ValueError(
                    "MOOD iteration count mismatch: "
                    f"{state.iter_count_hist} != {state.iter_count}"
                )
            if sum(state.troubled_cell_count_hist) != state.troubled_cell_count:
                raise ValueError(
                    "MOOD troubled cell count mismatch: "
                    f"{state.troubled_cell_count_hist} != {state.troubled_cell_count}"
                )

            data["n_MOOD_iters"] = state.iter_count
            data["nfine_MOOD_iters"] = state.iter_count_hist
            data["n_troubled_cells"] = state.troubled_cell_count
            data["nfine_troubled_cells"] = state.troubled_cell_count_hist

        if self.log_every_step:
            data.update(self.log_quantity())

        return data

    def reset_stepwise_logs(self):
        """
        Reset logs that are incremented at the end of each step.
        """
        super().reset_stepwise_logs()

        if self.MOOD:
            self.MOOD_state.reset_stepwise_counters()

    def reset_substepwise_logs(self):
        """
        Reset logs that are incremented at the end of each substep.
        """
        super().reset_substepwise_logs()

        self.n_updates = 0

        if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            self.arrays["theta_log"].fill(0.0)

        if self.MOOD:
            self.arrays["troubles_log"].fill(0)
            self.arrays["cascade_idx_log"].fill(0)

    def increment_substepwise_logs(self):
        """
        Increment logs at the end of each substep.
        """
        super().increment_substepwise_logs()

        xp = self.xp

        self.n_updates += self.mesh.size

        if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            theta = self.arrays["theta"][self.interior][..., 0]
            theta_log = self.arrays["theta_log"]
            xp.add(theta_log, theta, out=theta_log)

        if self.MOOD:
            troubles = self.arrays["troubles"]
            troubles_log = self.arrays["troubles_log"]
            cascade_idx = self.arrays["_cascade_idx_array_"][self.interior]
            cascade_idx_log = self.arrays["cascade_idx_log"]

            xp.add(troubles_log, troubles, out=troubles_log)
            xp.add(cascade_idx_log, cascade_idx, out=cascade_idx_log)

            self.MOOD_state.increment_substep_hists()

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

    def musclhancock(self, *args, **kwargs):
        """
        Apply the MUSCL-Hancock method for time integration.
        """
        self.integrator = "musclhancock"
        self.stepper = self._musclhancock_step
        self.integrate(*args, **kwargs)

    def _musclhancock_step(self, t: float, u: ArrayLike, dt: float):
        """
        Implementation of `ExplicitODESolver.stepper` for the MUSCL-Hancock method.

        Args:
            t: Current time.
            u: Current solution vector.
            dt: Time step size.

        Notes:
            This method requires `self.flux_jvp` to be implemented.
        """
        scheme = self.base_scheme
        limit_primitives = scheme.flux_recipe in (2, 3)
        self.substep_dt = dt

        if not isinstance(scheme, musclInterpolationScheme):
            raise ValueError("The scheme must be a MUSCL interpolation scheme.")

        # update workspace
        self.update_workspaces(t, u, scheme)

        # predictor step
        self.inplace_reconstruct_muscl_faces(
            scheme, limit_primitives=limit_primitives, hancock=True, dt=dt
        )

        self.increment_substepwise_logs()

        # corrector step
        for dim in self.active_dims:
            self.normalize_node_arrays(t, dim, scheme, primitives=limit_primitives)
            self.inplace_integrate_fluxes(dim, scheme)

        dudt = self.compute_RHS().copy()
        self.arrays["unew"][...] = u + dt * dudt

        self.increment_substepwise_logs()

    def inplace_reconstruct_muscl_faces(
        self,
        scheme: musclInterpolationScheme,
        *,
        limit_primitives: bool,
        hancock: bool = False,
        dt: Optional[float] = None,
    ):
        """
        Interpolate face nodes using either a MUSCL or MUSCL-Hancock method and write
        them to the appropriate arrays (`x_nodes`, `y_nodes`, `z_nodes`).

        Args:
            scheme: The MUSCL interpolation scheme to use.
            limit_primitives: Whether to limit the primitive variables.
            hancock: Whether to use the MUSCL-Hancock method.
            dt: The time step size (required if hancock is True).
        """
        xp = self.xp
        mesh = self.mesh
        arrays = self.arrays

        # allocate arrays
        x_nodes = arrays["x_nodes"]
        y_nodes = arrays["y_nodes"]
        z_nodes = arrays["z_nodes"]
        w_center = arrays["_w_"] if limit_primitives else arrays["_u_"]
        dwx = arrays["buffer"][..., 0:1]
        dwy = arrays["buffer"][..., 1:2]
        dwz = arrays["buffer"][..., 2:3]
        w_center_for_nodes = arrays["buffer"][..., 3]
        buffer = arrays["buffer"][..., 4:]

        # compute limited slopes
        for slope_arr, dim in zip([dwx, dwy, dwz], xyz_tup):
            if dim not in self.active_dims:
                continue
            compute_limited_slopes(
                xp,
                w_center,
                dim,
                self.active_dims,
                out=slope_arr,
                buffer=buffer,
                limiter=scheme.limiter_config.limiter,
                SED=scheme.limiter_config.SED,
            )

        # evolve the cell-center by 1/2 dt
        if hancock:
            if dt is None:
                raise ValueError(
                    "dt must be provided for MUSCL-Hancock reconstruction."
                )

            w_center_for_nodes[...] = w_center
            for slope_arr, dim in zip([dwx, dwy, dwz], xyz_tup):
                if dim not in self.active_dims:
                    continue
                h = getattr(mesh, "h" + dim)
                ds = self.flux_jvp(
                    w_center, slope_arr[..., 0], dim, primitives=limit_primitives
                )
                w_center_for_nodes[...] -= ds * dt / 2 / h
        else:
            w_center_for_nodes[...] = w_center

        # update the face nodes using the limited slopes
        for node_arr, slope_arr, dim in zip(
            [x_nodes, y_nodes, z_nodes], [dwx, dwy, dwz], xyz_tup
        ):
            if dim not in self.active_dims:
                continue
            node_arr[..., 0] = w_center_for_nodes - slope_arr[..., 0] / 2
            node_arr[..., 1] = w_center_for_nodes + slope_arr[..., 0] / 2

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
