import pickle
import warnings
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from superfv.tools.buffer import check_buffer_slots

from . import fv
from .axes import DIM_TO_AXIS
from .boundary_conditions import BCs, PatchBC, apply_bc
from .explicit_ODE_solver import ExplicitODESolver
from .field import MultivarField, UnivarField
from .interpolation_schemes import (
    InterpolationScheme,
    LimiterConfig,
    polyInterpolationScheme,
)
from .mesh import UniformFVMesh, xyz_tup
from .slope_limiting import MOOD, compute_dmp, compute_vis
from .slope_limiting.MOOD import (
    MOODConfig,
    MOODState,
    append_troubled_cell_scalar_statistics,
    clear_troubled_cell_scalar_statistics,
    detect_PAD_violations,
    init_troubled_cell_scalar_statistics,
    log_troubled_cell_scalar_statistics,
)
from .slope_limiting.muscl import (
    compute_limited_slopes,
    compute_PP2D_slopes,
    musclConfig,
    musclInterpolationScheme,
)
from .slope_limiting.shock_detection import compute_shock_detector
from .slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from .slope_limiting.zhang_and_shu import (
    ZhangShuConfig,
    append_zhang_shu_scalar_statistics,
    clear_zhang_shu_scalar_statistics,
    compute_theta,
    compute_theta_kernel_helper,
    init_zhang_shu_scalar_statistics,
    log_zhang_shu_scalar_statistics,
    zhang_shu_operator,
)
from .tools.device_management import CUPY_AVAILABLE, ArrayLike, ArrayManager, xp
from .tools.slicing import VariableIndexMap, crop, insert_slice, merge_slices
from .tools.timer import MethodTimer, StepperTimer
from .tools.yaml_helper import yaml_dump
from .visualization import plot_1d_slice, plot_2d_slice

warnings.filterwarnings("ignore", category=RuntimeWarning)


Directions = Literal["x", "y", "z"]
Faces = Literal["xl", "xr", "yl", "yr", "zl", "zr"]


class FiniteVolumeSolver(ExplicitODESolver, ABC):
    """
    Solve a nonlinear conservation law using the finite volume method in up to three
    dimensions.
    """

    def __init__(
        self,
        ic: MultivarField,
        ic_passives: Optional[Dict[str, UnivarField]] = None,
        bcx: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcy: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcz: Union[str, Tuple[str, str]] = ("periodic", "periodic"),
        bcx_callable: Optional[Tuple[MultivarField, PatchBC]] = None,
        bcy_callable: Optional[Tuple[MultivarField, PatchBC]] = None,
        bcz_callable: Optional[Tuple[MultivarField, PatchBC]] = None,
        xlim: Tuple[float, float] = (0, 1),
        ylim: Tuple[float, float] = (0, 1),
        zlim: Tuple[float, float] = (0, 1),
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        p: int = 0,
        CFL: float = 0.8,
        dt_min: float = 1e-15,
        GL: bool = False,
        flux_recipe: Literal[1, 2, 3] = 2,
        lazy_primitives: Literal["none", "full", "adaptive"] = "none",
        eta_max: float = 0.025,
        riemann_solver: str = "dummy_riemann_solver",
        face_fallback: bool = False,
        MUSCL: bool = False,
        MUSCL_limiter: Literal["minmod", "moncen", "PP2D"] = "minmod",
        ZS: bool = False,
        adaptive_dt: bool = True,
        log_limiter_scalars: bool = True,
        MOOD: bool = False,
        cascade: Literal["first-order", "muscl", "full", "none"] = "muscl",
        blend: bool = False,
        max_MOOD_iters: int = 1,
        skip_trouble_counts: bool = False,
        detect_closing_troubles: bool = True,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "all",
        NAD: bool = True,
        NAD_delta: bool = True,
        NAD_rtol: Optional[Union[Dict[str, float], float]] = None,
        NAD_gtol: Optional[Union[Dict[str, float], float]] = None,
        NAD_atol: Optional[Union[Dict[str, float], float]] = 1e-14,
        scale_NAD_rtol_by_dt: bool = False,
        include_corners: bool = True,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        PAD_atol: float = 1e-15,
        SED: bool = True,
        check_uniformity: bool = True,
        uniformity_tol: float = 1e-15,
        vis_rtol: float = 1e-1,
        vis_atol: float = 1e-10,
        cupy: bool = False,
        sync_timing: bool = True,
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
            bcx_callable, bcy_callable, bcz_callable: Additional argument for
                "dirichlet" or "patch" boundary conditions. If "dirichlet" is used,
                the corresponding entry in the tuple must be a callable that takes the
                following arguments:
                - idx: VariableIndexMap object.
                - x: x-coordinate array.
                - y: y-coordinate array.
                - z: z-coordinate array.
                - t: Optional time at which the boundary condition is applied.
                And returns an array with shape as x, y, and z. If "patch" is used, the
                corresponding entry in the tuple must be a callable that takes the
                following arguments:
                - _u_: Array to which the boundary condition is applied.
                - context: BCcontext object containing parameters for applying the BC.
                and modifies _u_ in place.
            xlim, ylim, zlim: Limits of the domain in the x, y, and z-directions.
            nx, ny, nz: Number of cells in the x, y, and z-directions.
            p: Maximum polynomial degree of the spatial discretization.
            CFL: CFL number.
            dt_min: Minimum allowable timestep size.
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
                    averages (see `lazy_primitives` below). Interpolate primitive nodes
                    from primitive cell averages. Apply slope limiting to the primitive
                    nodes.
            lazy_primitives: Option for lazy evaluation of primitive variables.
                Possible values include:
                - "none": Do not use second-order evaluation for primitive cell
                    averages.
                - "full": Always use second-order evaluation for primitive cell
                    averages.
                - "adaptive": Based on a shock-detection criterion, adaptively reduce
                    the order of conservative cell centers, primitive cell centers, and
                    primitive cell averages to second order.
            eta_max: Threshold for shock detection when `lazy_primitives` is "adaptive".
            riemann_solver: Name of the Riemann solver function. Must be implemented in
                the derived class.
            face_fallback: Whether to enable face state fallback based on floor
                violations.
            MUSCL: Whether to use the MUSCL scheme as the base scheme. Overrides `p`,
                `flux_recipe`, and `lazy_primitives`. The `flux_recipe` options become:
                - `flux_recipe=1`: Slope limiting is performed on conservative slopes.
                - `flux_recipe=2`: Slope limiting is performed on primitive slopes.
                - `flux_recipe=3`: `flux_recipe=2` is used.
            MUSCL_limiter: Slope limiter used for the MUSCL scheme, either for the base
                scheme or the MOOD cascade. Options include:
                - "minmod"
                - "moncen"
                - "PP2D": Only valid for 2D problems.
            ZS: Whether to use Zhang and Shu's maximum-principle-satisfying a priori
                slope limiter.
            adaptive_dt: Option for the Zhang and Shu limiter; Whether to iteratively
                halve the timestep size if the proposed solution fails PAD.
            log_limiter_scalars: Whether to log scalar statistics for the Zhang-Shu
                limiter and MOOD.
            MOOD: Whether to use MOOD for a posteriori flux revision. Ignored if
                `ZS=True` and `adaptive_timestepping=True`.
            cascade: A string indicating which type of MOOD cascade to use:
                - "first-order": Fall back directly to a first-order scheme.
                - "muscl": Fall back directly to a MUSCL scheme.
                - "full": Fall back to a full cascade of scheme in descending order.
                - "none": Do not use any fallback schemes.
            blend: Whether to blend the troubled cell indicator with neighboring
                cells following Vilar and Abgrall 2022. Only valid for "first-order"
                and "muscl" cascades.
            max_MOOD_iters: Option for the MOOD limiter; The maximum number of MOOD
                iterations that may be performed in an update step. Defaults to 1.
            skip_trouble_counts: Whether to skip counting the number of troubled cells.
            detect_closing_troubles: Whether to detect closing troubles at the end of
                the MOOD loop if revisable troubled cells were found during the last
                iteration. If False, the troubles array will represent the troubled
                cells that determined the closing cascade index.
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
            NAD_delta: Whether to use the local DMP range to relax the bounds for NAD. If
                False, only NAD_rtol is used to relax the bounds.
            NAD_rtol, NAD_gtol, NAD_atol: Tolerance values used to relax the bounds for
                numerical admissibility detection (see the `detect_NAD_violations`).
                May be provided as one of the following:
                - Dict[str, float]: A dictionary mapping variable names to their
                    corresponding tolerance values. Limiting variables not provided in
                    the dictionary are treated as having a tolerance of 0.
                - float: A single float value that is applied to all limiting
                    variables.
                - None: All limiting variables are treated as having a tolerance of 0.
            scale_NAD_rtol_by_dt: Whether to scale the NAD rtol by dt.
            include_corners: Whether to include corner nodes in the slope limiting.
            PAD: Dict of `limiting_vars` and their corresponding PAD tolerances as a
                tuple: (lower_bound, upper_bound). Any variable or bound not provided
                in `PAD` is given a lower and upper bound of `-np.inf` and `np.inf`
                respectively.
            PAD_atol: Tolerance for the PAD check as an absolute value from the minimum
                and maximum values of the variable.
            SED: Whether to use smooth extrema detection for slope limiting.
            check_uniformity: Whether to relax alpha to 1.0 in uniform regions if smooth
            extrema detection is enabled. Uniform regions satisfy:
                max(u_{i-1}, u_i, u_{i+1}) - min(u_{i-1}, u_i, u_{i+1})
                    <= uniformity_tol * |u_i|
            uniformity_tol: Tolerance for uniformity check when check_uniformity is True.
            vis_rtol, vis_atol: Relative and absolute tolerances for the visualization
                threshold. See `compute_vis`.
            cupy: Whether to use CuPy for array operations.
            sync_timing: Whether to synchronize the GPU after each timed method call if
                using CuPy. This ensures accurate timing measurements when profiling.
        """
        self._init_active_dims(nx, ny, nz)
        self._init_ic_callables(ic, ic_passives)
        self._init_array_management(cupy)
        self._init_spatial_discretization(
            p,
            flux_recipe,
            GL,
            lazy_primitives,
            eta_max,
            MUSCL,
            MUSCL_limiter,
            ZS,
            adaptive_dt,
            MOOD,
            cascade,
            blend,
            max_MOOD_iters,
            skip_trouble_counts,
            detect_closing_troubles,
            limiting_vars,
            NAD,
            NAD_delta,
            NAD_rtol,
            NAD_gtol,
            NAD_atol,
            scale_NAD_rtol_by_dt,
            include_corners,
            PAD,
            PAD_atol,
            SED,
            check_uniformity,
            uniformity_tol,
            face_fallback,
        )
        self._init_snapshots(log_limiter_scalars)
        self._init_mesh(xlim, ylim, zlim, nx, ny, nz, CFL)
        self._init_bc(bcx, bcy, bcz, bcx_callable, bcy_callable, bcz_callable)
        self._init_array_allocation()
        self._init_ODE_solver(dt_min)
        self._init_timer(sync_timing)
        self._init_riemann_solver(riemann_solver)
        self._init_visualization(vis_rtol, vis_atol)

    def _init_active_dims(self, nx: int, ny: int, nz: int):
        self.active_dims = tuple(d for d, n in zip(xyz_tup, (nx, ny, nz)) if n > 1)
        self.inactive_dims = tuple(d for d in xyz_tup if d not in self.active_dims)

    def _init_ic_callables(
        self,
        ic: MultivarField,
        ic_passives: Optional[Dict[str, UnivarField]],
    ):
        # Define the following attributes:
        self.variable_index_map: VariableIndexMap
        self.nvars: int
        self.n_passive_vars: int
        self.active_vars: Set[str]
        self.passive_vars: Set[str]
        self.ic: MultivarField
        self.callable_ic: MultivarField

        # Define variable index map
        idx = self.define_vars()

        self.ic = ic
        if ic_passives:
            for v in ic_passives.keys():
                if v not in idx.var_idx_map:
                    idx.add_var(v, idx.nvars)
            idx.add_var_to_group("passives", ic_passives.keys())
            self.ic_passives = ic_passives
            self.callable_ic = self._make_callable_ic_with_passives()
        else:
            self.callable_ic = self._make_callable_ic()

        self.variable_index_map = idx
        self.nvars = idx.nvars
        self.n_passive_vars = (
            len(np.arange(self.nvars)[idx("passives")]) if "passives" in idx else 0
        )

    def _make_callable_ic(self) -> MultivarField:
        """
        Returns a MultivarField callable for the initial condition function.
        """

        def f(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            return self.ic(idx, x, y, z, t, xp=xp)

        return f

    def _make_callable_ic_with_passives(self) -> MultivarField:
        """
        Returns a MultivarField callable for the initial condition function with
        passive variables.
        """

        def f(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            out = self.ic(idx, x, y, z, t, xp=xp)
            if hasattr(self, "ic_passives"):
                for v, func in self.ic_passives.items():
                    out[idx(v)] = func(x, y, z, t, xp=xp)
            return out

        return f

    def _make_conservative_field(self, f: MultivarField) -> MultivarField:
        """
        Returns a MultivarField callable that converts a primitive variable field to
        conservative variables.

        Args:
            f: MultivarField callable that returns primitive variables.

        Returns:
            A MultivarField callable that returns conservative variables.
        """

        def g(
            idx: VariableIndexMap,
            x: ArrayLike,
            y: ArrayLike,
            z: ArrayLike,
            t: Optional[float] = None,
            *,
            xp: ModuleType,
        ) -> ArrayLike:
            return self.conservatives_from_primitives(f(idx, x, y, z, t, xp=xp))

        return g

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
        self.mesh_arrays = ArrayManager()
        if self.cupy:
            self.arrays.transfer_to("gpu")
            self.mesh_arrays.transfer_to("gpu")

    def _init_spatial_discretization(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        GL: bool,
        lazy_primitives: Literal["none", "full", "adaptive"],
        eta_max: float,
        MUSCL: bool,
        MUSCL_limiter: Literal["minmod", "moncen", "PP2D"],
        ZS: bool,
        adaptive_dt: bool,
        MOOD: bool,
        cascade: Literal["first-order", "muscl", "full", "none"],
        blend: bool,
        max_MOOD_iters: int,
        skip_trouble_counts: bool,
        detect_closing_troubles: bool,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]],
        NAD: bool,
        NAD_delta: bool,
        NAD_rtol: Optional[Union[Dict[str, float], float]],
        NAD_gtol: Optional[Union[Dict[str, float], float]],
        NAD_atol: Optional[Union[Dict[str, float], float]],
        scale_NAD_rtol_by_dt: bool,
        include_corners: bool,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]],
        PAD_atol: float,
        SED: bool,
        check_uniformity: bool,
        uniformity_tol: float,
        face_fallback: bool,
    ):
        self.base_scheme: InterpolationScheme
        self.using_PAD: bool
        self.MOOD: bool
        self.MOOD_config: MOODConfig
        self.MOOD_state: MOODState

        # add limiting variables to the index map
        idx = self.variable_index_map
        idx.add_var_to_group("limiting", [])
        if limiting_vars == "all":
            limiting_vars = tuple(idx.var_idx_map.keys())
        elif limiting_vars == "actives":
            limiting_vars = tuple(idx.group_var_map["primitives"])
        idx.add_var_to_group("limiting", limiting_vars)

        self._init_PAD(PAD)  # defines `self.using_PAD`

        # init a priori base scheme
        if p == 0:
            self._init_unlimited_scheme(
                0,
                flux_recipe,
                GL,
                lazy_primitives,
                eta_max,
                SED,
                check_uniformity,
                uniformity_tol,
                PAD_atol,
            )
        elif MUSCL:
            if ZS:
                raise ValueError("MUSCL scheme cannot be combined with ZS.")
            self._init_muscl_scheme(
                p, flux_recipe, MUSCL_limiter, SED, check_uniformity, uniformity_tol
            )
        elif ZS:
            self._init_zhang_shu_scheme(
                p,
                flux_recipe,
                GL,
                lazy_primitives,
                eta_max,
                include_corners,
                SED,
                check_uniformity,
                uniformity_tol,
                adaptive_dt,
                PAD_atol,
            )
        else:
            self._init_unlimited_scheme(
                p,
                flux_recipe,
                GL,
                lazy_primitives,
                eta_max,
                SED,
                check_uniformity,
                uniformity_tol,
                PAD_atol,
            )

        # init a posteriori scheme
        self.MOOD_state = MOODState()
        if MOOD and self.base_scheme.p > 0:
            self.MOOD = True

            self._init_MOOD(
                cascade,
                blend,
                MUSCL_limiter,
                max_MOOD_iters,
                skip_trouble_counts,
                detect_closing_troubles,
                NAD,
                NAD_delta,
                NAD_rtol,
                NAD_gtol,
                NAD_atol,
                scale_NAD_rtol_by_dt,
                SED,
                check_uniformity,
                uniformity_tol,
                PAD_atol,
                include_corners,
            )
        else:
            self.MOOD = False
            self.MOOD_config = MOODConfig(
                shock_detection=False,
                smooth_extrema_detection=False,
                check_uniformity=False,
                physical_admissibility_detection=False,
            )

        # init face fallback
        self.face_fallback = face_fallback

    def _init_PAD(
        self, PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]]
    ):
        if PAD is None:
            self.using_PAD = False
            return

        idx = self.variable_index_map
        self.using_PAD = True

        PAD_bounds = np.array([[-np.inf, np.inf] for _ in range(self.nvars)])
        for var, (lb, ub) in PAD.items():
            PAD_bounds[idx(var), 0] = lb if lb is not None else -np.inf
            PAD_bounds[idx(var), 1] = ub if ub is not None else np.inf
        self.arrays.add("PAD_bounds", PAD_bounds)

    def _init_unlimited_scheme(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        GL: bool,
        lazy_primitives: Literal["none", "full", "adaptive"],
        eta_max: float,
        SED: bool,
        check_uniformity: bool,
        uniformity_tol: float,
        PAD_atol: float,
    ):
        self.p = p
        self.base_scheme = polyInterpolationScheme(
            p=p,
            flux_recipe=flux_recipe,
            limiter_config=LimiterConfig(
                shock_detection=lazy_primitives == "adaptive",
                smooth_extrema_detection=SED,
                check_uniformity=check_uniformity,
                physical_admissibility_detection=self.using_PAD,
                eta_max=eta_max,
                PAD_bounds=self.arrays["PAD_bounds"] if self.using_PAD else None,
                PAD_atol=PAD_atol,
                uniformity_tol=uniformity_tol,
            ),
            gauss_legendre=GL,
            lazy_primitives=lazy_primitives,
        )

    def _init_muscl_scheme(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        MUSCL_limiter: Optional[Literal["minmod", "moncen", "PP2D"]],
        SED: bool,
        check_uniformity: bool,
        uniformity_tol: float,
    ):
        if p != 1:
            warnings.warn("MUSCL overrides p to be 1.")
        if flux_recipe == 3:
            warnings.warn("MUSCL overrides flux_recipe 3 to be 2.")
            flux_recipe = 2

        self.p = 1
        self.base_scheme = musclInterpolationScheme(
            flux_recipe=flux_recipe,
            limiter_config=musclConfig(
                shock_detection=False,
                smooth_extrema_detection=SED,
                check_uniformity=check_uniformity,
                physical_admissibility_detection=False,
                limiter=MUSCL_limiter,
                uniformity_tol=uniformity_tol,
            ),
        )

    def _init_zhang_shu_scheme(
        self,
        p: int,
        flux_recipe: Literal[1, 2, 3],
        GL: bool,
        lazy_primitives: Literal["none", "full", "adaptive"],
        eta_max: float,
        include_corners: bool,
        SED: bool,
        check_uniformity: bool,
        uniformity_tol: float,
        adaptive_dt: bool,
        PAD_atol: float,
    ):
        self.p = p
        self.base_scheme = polyInterpolationScheme(
            p=p,
            flux_recipe=flux_recipe,
            limiter_config=ZhangShuConfig(
                shock_detection=lazy_primitives == "adaptive",
                smooth_extrema_detection=SED,
                check_uniformity=check_uniformity,
                physical_admissibility_detection=self.using_PAD,
                eta_max=eta_max,
                PAD_bounds=self.arrays["PAD_bounds"] if self.using_PAD else None,
                PAD_atol=PAD_atol,
                uniformity_tol=uniformity_tol,
                include_corners=include_corners,
                adaptive_dt=adaptive_dt,
                theta_denom_tol=1e-16,
            ),
            gauss_legendre=GL,
            lazy_primitives=lazy_primitives,
        )

    def _init_MOOD(
        self,
        cascade: Literal["first-order", "muscl", "full", "none"],
        blend: bool,
        MUSCL_limiter: Literal["minmod", "moncen", "PP2D"],
        max_MOOD_iters: int,
        skip_trouble_counts: bool,
        detect_closing_troubles: bool,
        NAD: bool,
        NAD_delta: bool,
        NAD_rtol: Optional[Union[Dict[str, float], float]],
        NAD_gtol: Optional[Union[Dict[str, float], float]],
        NAD_atol: Optional[Union[Dict[str, float], float]],
        scale_NAD_rtol_by_dt: bool,
        SED: bool,
        check_uniformity: bool,
        uniformity_tol: float,
        PAD_atol: float,
        include_corners: bool,
    ):
        base_scheme = self.base_scheme
        if not isinstance(base_scheme, polyInterpolationScheme):
            raise ValueError(
                "Base scheme must be an instance of polyInterpolationScheme."
            )

        # define list of fallback schemes
        fallback_schemes: List[InterpolationScheme]
        if cascade == "first-order":
            fallback_schemes = [
                polyInterpolationScheme(
                    p=0,
                    flux_recipe=base_scheme.flux_recipe,
                    limiter_config=LimiterConfig(
                        shock_detection=False,
                        smooth_extrema_detection=False,
                        check_uniformity=False,
                        physical_admissibility_detection=False,
                    ),
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
                    limiter_config=musclConfig(
                        shock_detection=False,
                        smooth_extrema_detection=False,
                        check_uniformity=False,
                        physical_admissibility_detection=False,
                        limiter=MUSCL_limiter,
                    ),
                )
            ]
            if cascade == "muscl1":
                fallback_schemes += [
                    polyInterpolationScheme(
                        p=0,
                        flux_recipe=base_scheme.flux_recipe,
                        limiter_config=LimiterConfig(
                            shock_detection=False,
                            smooth_extrema_detection=False,
                            check_uniformity=False,
                            physical_admissibility_detection=False,
                        ),
                        gauss_legendre=base_scheme.gauss_legendre,
                        lazy_primitives=base_scheme.lazy_primitives,
                    )
                ]
        elif cascade == "full":
            fallback_schemes = [
                polyInterpolationScheme(
                    p=p,
                    flux_recipe=base_scheme.flux_recipe,
                    limiter_config=LimiterConfig(
                        shock_detection=False,
                        smooth_extrema_detection=False,
                        check_uniformity=False,
                        physical_admissibility_detection=False,
                    ),
                    gauss_legendre=base_scheme.gauss_legendre,
                    lazy_primitives=base_scheme.lazy_primitives,
                )
                for p in range(base_scheme.p - 1, -1, -1)
            ]
        elif cascade == "none":
            fallback_schemes = []
        else:
            raise ValueError(f"Unknown cascade type: {cascade}.")
        cascade_list = [self.base_scheme] + fallback_schemes

        # init NAD arrays if needed
        if NAD:
            self._init_NAD(NAD_rtol, NAD_gtol, NAD_atol)

        # assign MOOD config
        self.MOOD_config = MOODConfig(
            shock_detection=False,
            smooth_extrema_detection=SED,
            check_uniformity=check_uniformity,
            physical_admissibility_detection=self.using_PAD,
            PAD_bounds=self.arrays["PAD_bounds"] if self.using_PAD else None,
            PAD_atol=PAD_atol,
            uniformity_tol=uniformity_tol,
            numerical_admissibility_detection=NAD,
            delta=NAD_delta,
            cascade=cascade_list,
            blend=blend,
            max_iters=max_MOOD_iters,
            include_corners=include_corners,
            NAD_rtol=self.arrays["NAD_rtol"] if NAD and NAD_rtol else None,
            NAD_gtol=self.arrays["NAD_gtol"] if NAD and NAD_gtol else None,
            NAD_atol=self.arrays["NAD_atol"] if NAD and NAD_atol else None,
            scale_NAD_rtol_by_dt=scale_NAD_rtol_by_dt,
            skip_trouble_counts=skip_trouble_counts,
            detect_closing_troubles=detect_closing_troubles,
        )

    def _init_NAD(
        self,
        NAD_rtol: Optional[Union[Dict[str, float], float]],
        NAD_gtol: Optional[Union[Dict[str, float], float]],
        NAD_atol: Optional[Union[Dict[str, float], float]],
    ):
        xp = self.xp
        idx = self.variable_index_map

        rtol = xp.zeros(self.nvars)
        gtol = xp.zeros(self.nvars)
        atol = xp.zeros(self.nvars)

        def validate_and_assign(
            tols: Optional[Union[Dict[str, float], float]], arr: ArrayLike
        ):
            if tols is None:
                return

            if isinstance(tols, (int, float)):
                arr.fill(tols)
                return

            for var, tol in tols.items():
                if var not in idx.var_idx_map:
                    raise ValueError(f"{var} not defined in variable index map.")
                if var not in idx.group_var_map["limiting"]:
                    raise ValueError(f"{var} not in limiting variable group.")
                arr[idx(var)] = tol

        validate_and_assign(NAD_rtol, rtol)
        validate_and_assign(NAD_gtol, gtol)
        validate_and_assign(NAD_atol, atol)

        self.arrays.add("NAD_rtol", rtol)
        self.arrays.add("NAD_gtol", gtol)
        self.arrays.add("NAD_atol", atol)

    def _init_snapshots(self, log_limiter_scalars: bool):
        self.step_log: Dict[str, List[float]] = {}
        self.log_limiter_scalars = log_limiter_scalars

        # init emergency face fallback statistics
        self.step_log["nfine_emergency_fallbacks"] = []
        self.n_emergency_fallbacks: int = 0

        if self.log_limiter_scalars:
            if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
                init_zhang_shu_scalar_statistics(self)

            if self.MOOD:
                init_troubled_cell_scalar_statistics(self)

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

        bs = self.base_scheme
        p = bs.p

        # determine slab depth
        if isinstance(bs, musclInterpolationScheme):
            node_cost = 0  # included in limiter
            limiting_cost = 3 if bs.limiter_config.smooth_extrema_detection else 1
            flux_integral_cost = 0
        elif isinstance(bs, polyInterpolationScheme):
            s = -(-p // 2)  # ghost cell cost of stencil with degree 0

            if bs.lazy_primitives == "full":
                node_cost = s
            else:
                node_cost = 3 * s if bs.flux_recipe == 3 else 2 * s
            if bs.lazy_primitives == "adaptive":
                node_cost += 2

            limiting_cost = 0
            if isinstance(bs.limiter_config, ZhangShuConfig):
                limiting_cost = 3 if bs.limiter_config.smooth_extrema_detection else 1
            if self.MOOD:
                NAD_cost = 1
                if self.MOOD_config.smooth_extrema_detection:
                    NAD_cost = 3
                trouble_map_cost = 3 if self.MOOD_config.blend else 1
                limiting_cost = NAD_cost + trouble_map_cost

            flux_integral_cost = s if len(self.active_dims) > 1 else 0
        else:
            raise ValueError("Unknown interpolation scheme.")

        slab_depth = node_cost + limiting_cost + max(flux_integral_cost, 1)

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
            array_manager=self.mesh_arrays,
        )

        # assign attributes
        self.CFL: float = CFL

        # interior workspace view
        self.interior = merge_slices(
            *[
                crop(
                    DIM_TO_AXIS[dim],
                    (self.mesh.slab_depth, -self.mesh.slab_depth),
                    ndim=4,
                )
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
        bcx_callable: Optional[Tuple[MultivarField, PatchBC]],
        bcy_callable: Optional[Tuple[MultivarField, PatchBC]],
        bcz_callable: Optional[Tuple[MultivarField, PatchBC]],
    ):
        def as_pair(x: Union[Any, Tuple[Any, Any]]) -> Tuple[Any, Any]:
            if isinstance(x, tuple):
                if len(x) != 2:
                    raise TypeError("Expected a two-element tuple.")
                return x
            return (x, x)

        mode_list: List[Tuple[BCs, BCs]] = []
        callable_bc_list: List[
            Tuple[
                Optional[Union[MultivarField, PatchBC]],
                Optional[Union[MultivarField, PatchBC]],
            ]
        ] = []

        for dim, bci, fi in zip(
            xyz_tup, (bcx, bcy, bcz), (bcx_callable, bcy_callable, bcz_callable)
        ):
            mode_pair: List[BCs] = []
            callable_bc_pair: List[Optional[Union[MultivarField, PatchBC]]] = []

            if dim not in self.active_dims:
                mode_list.append(("none", "none"))
                callable_bc_list.append((None, None))
                continue

            for j, (bc, f) in enumerate(zip(as_pair(bci), as_pair(fi))):
                if bc == "ic":
                    mode_pair.append("dirichlet")
                    callable_bc_pair.append(
                        self._make_conservative_field(self.callable_ic)
                    )
                else:
                    mode_pair.append(bc)
                    callable_bc_pair.append(
                        self._make_conservative_field(f) if bc == "dirichlet" else f
                    )

            mode_list.append((mode_pair[0], mode_pair[1]))
            callable_bc_list.append((callable_bc_pair[0], callable_bc_pair[1]))

        self.bc_mode: Tuple[Tuple[BCs, BCs], Tuple[BCs, BCs], Tuple[BCs, BCs]] = (
            mode_list[0],
            mode_list[1],
            mode_list[2],
        )
        self.bc_callables: Tuple[
            Tuple[
                Optional[Union[MultivarField, PatchBC]],
                Optional[Union[MultivarField, PatchBC]],
            ],
            Tuple[
                Optional[Union[MultivarField, PatchBC]],
                Optional[Union[MultivarField, PatchBC]],
            ],
            Tuple[
                Optional[Union[MultivarField, PatchBC]],
                Optional[Union[MultivarField, PatchBC]],
            ],
        ] = (
            callable_bc_list[0],
            callable_bc_list[1],
            callable_bc_list[2],
        )

        mesh = self.mesh

        self.bc_pad_width: Tuple[int, int, int] = (
            mesh.x_slab_depth,
            mesh.y_slab_depth,
            mesh.z_slab_depth,
        )

    def _init_riemann_solver(self, riemann_solver: str):
        self.init_riemann_solver(riemann_solver)

    def _init_visualization(self, vis_rtol: float, vis_atol: float):
        self.vis_rtol = vis_rtol
        self.vis_atol = vis_atol

    def _init_array_allocation(self):
        if self.cupy:
            self.arrays.transfer_to("gpu")

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
        arrays.add("_wp_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_w_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_buffer_", np.empty((nvars, _nx_, _ny_, _nz_, buffer_size)))
        arrays.add("_x_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("_y_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("_z_nodes_", np.empty((nvars, _nx_, _ny_, _nz_, max_ninterps)))
        arrays.add("_centroid_", np.empty((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_F_", np.empty((nvars, nx + 1, _ny_, _nz_, 1)))
        arrays.add("_G_", np.empty((nvars, _nx_, ny + 1, _nz_, 1)))
        arrays.add("_H_", np.empty((nvars, _nx_, _ny_, nz + 1, 1)))
        arrays.add("_f_nodes_", np.empty((nvars, nx + 1, _ny_, _nz_, max_nodes)))
        arrays.add("_g_nodes_", np.empty((nvars, _nx_, ny + 1, _nz_, max_nodes)))
        arrays.add("_h_nodes_", np.empty((nvars, _nx_, _ny_, nz + 1, max_nodes)))

        # General slope-limiting arrays
        arrays.add("_dmp_", np.empty((nvars, _nx_, _ny_, _nz_, 2)))
        arrays.add("_alpha_", np.ones((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("_eta_", np.zeros((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_shockless_", np.ones((1, _nx_, _ny_, _nz_), dtype=bool))

        # Visualization array
        arrays.add("visualize", np.ones((nvars, nx, ny, nz), dtype=bool))

        # Zhang-Shu limiter arrays
        arrays.add("_theta_", np.ones((nvars, _nx_, _ny_, _nz_, 1)))
        arrays.add("theta_vis", np.ones((nvars, nx, ny, nz)))
        arrays.add("theta_log", np.ones((nvars, nx, ny, nz)))
        arrays.add("theta_vis_log", np.ones((nvars, nx, ny, nz)))
        arrays.add("_node_mp_", np.empty((nvars, _nx_, _ny_, _nz_, 2)))

        ntotal = _nx_ * _ny_ * _nz_
        flat_ninterpolations = 1 + self.mesh.ndim * max_ninterps
        arrays.add("_flat_w_", np.empty((nvars, ntotal)))
        arrays.add("_flat_wj_", np.empty((nvars, ntotal, flat_ninterpolations)))
        arrays.add("_flat_M_", np.empty((nvars, ntotal)))
        arrays.add("_flat_m_", np.empty((nvars, ntotal)))
        arrays.add("_flat_Mj_", np.empty((nvars, ntotal)))
        arrays.add("_flat_mj_", np.empty((nvars, ntotal)))
        arrays.add("_flat_theta_", np.ones((nvars, ntotal)))

        # MOOD arrays
        arrays.add("troubles", np.zeros((nvars, nx, ny, nz), dtype=bool))
        arrays.add("troubles_vis", np.zeros((nvars, nx, ny, nz), dtype=bool))
        arrays.add("troubles_log", np.zeros((nvars, nx, ny, nz)))
        arrays.add("troubles_vis_log", np.zeros((nvars, nx, ny, nz)))
        arrays.add("revisable_troubles", np.zeros((1, nx, ny, nz), dtype=bool))
        arrays.add("cascade_idx", np.zeros((1, nx, ny, nz), dtype=int))
        arrays.add("cascade_idx_log", np.zeros((1, nx, ny, nz)))

        for scheme in self.MOOD_config.cascade:
            arrays.add("F_" + scheme.key(), np.empty((nvars, nx + 1, ny, nz)))
            arrays.add("G_" + scheme.key(), np.empty((nvars, nx, ny + 1, nz)))
            arrays.add("H_" + scheme.key(), np.empty((nvars, nx, ny, nz + 1)))

        arrays.add("_unew_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_wnew_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_NAD_violations_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_PAD_violations_", np.empty((nvars, _nx_, _ny_, _nz_)))
        arrays.add("_troubles_", np.zeros((nvars, _nx_, _ny_, _nz_), dtype=bool))
        arrays.add("_any_troubles_", np.zeros((1, _nx_, _ny_, _nz_), dtype=bool))
        arrays.add("_cascade_idx_", np.zeros((1, _nx_, _ny_, _nz_), dtype=int))
        arrays.add("_blended_cascade_idx_", np.zeros((1, _nx_, _ny_, _nz_)))
        arrays.add("_mask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1), dtype=int))
        arrays.add("_fmask_", np.zeros((1, _nx_ + 1, _ny_ + 1, _nz_ + 1)))

        # helper attribute
        self.flux_names = {"x": "F", "y": "G", "z": "H"}

    def _init_ODE_solver(self, dt_min: float):
        idx = self.variable_index_map
        xp = self.xp
        mesh = self.mesh
        nvars = self.nvars
        arrays = self.arrays

        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        # initialize the ODE solver with the initial condition array
        u0 = self._make_conservative_field(self.callable_ic)
        ic_arr = mesh.perform_GaussLegendre_quadrature(
            lambda x, y, z: u0(idx, x, y, z, 0.0, xp=xp),
            node_axis=4,
            mesh_region="core",
            cell_region="interior",
            p=self.p,
        )

        # call superclass initializer
        super().__init__(ic_arr, self.arrays, dt_min=dt_min)  # defines "u"
        assert arrays["u"].shape == (nvars, nx, ny, nz)

        # set adaptive dt functions if applicable
        if getattr(self.base_scheme.limiter_config, "adaptive_dt", False):
            self._dt_criterion = self.adaptive_dt_criterion
            self._compute_revised_dt = self.adaptive_dt_revision

    def _compute_buffer_size(
        self, scheme: InterpolationScheme, check_MOOD: bool = True
    ) -> int:
        """
        Compute the required buffer size, which is the length along the fifth axis of
        the "_buffer_", for the given interpolation scheme.

        Args:
            scheme: Interpolation scheme.
            check_MOOD: Whether to check fallback schemes in MOOD configuration.

        Returns:
            Required buffer size.
        """
        ndim = self.mesh.ndim
        p = scheme.p

        # buffer cost of interpolation nodes
        GL_nodes_per_dim = -(-(p + 1) // 2)
        if ndim == 1:
            interpolation_buffer_cost = 0
        elif ndim == 2:
            interpolation_buffer_cost = 2
        else:  # ndim == 3
            if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
                interpolation_buffer_cost = 2 + 2 * GL_nodes_per_dim
            else:
                interpolation_buffer_cost = 3

        # buffer cost of `update_workspaces` function
        if getattr(scheme, "lazy_primitives", "none") == "adaptive":
            adaptive_primitive_cost = {1: 7, 2: 9, 3: 10}[ndim]
            update_workspaces_buffer_cost = max(
                interpolation_buffer_cost, adaptive_primitive_cost
            )
        else:
            update_workspaces_buffer_cost = interpolation_buffer_cost

        # buffer cost of slope-limiting functions
        if isinstance(scheme.limiter_config, ZhangShuConfig):
            limiting_buffer_cost = 3
        elif isinstance(scheme, musclInterpolationScheme):
            limiting_buffer_cost = 4
            if scheme.limiter_config.limiter == "PP2D":
                limiting_buffer_cost += 4
            else:
                limiting_buffer_cost += 5
        elif check_MOOD and self.MOOD:
            limiting_buffer_cost = 8
        else:
            limiting_buffer_cost = 0

        # add SED buffer cost if applicable
        if getattr(scheme.limiter_config, "smooth_extrema_detection", False):
            limiting_buffer_cost += {1: 10, 2: 12, 3: 13}[ndim]

        # buffer size requirement before checking fallback schemes
        total_buffer_cost = max(update_workspaces_buffer_cost, limiting_buffer_cost)

        # check fallback schemes in MOOD configuration
        if check_MOOD:
            for fallback_scheme in self.MOOD_config.cascade[1:]:
                fallback_buffer_cost = self._compute_buffer_size(
                    fallback_scheme, check_MOOD=False
                )
                total_buffer_cost = max(total_buffer_cost, fallback_buffer_cost)

        return total_buffer_cost

    def nodes_per_face(self, scheme: InterpolationScheme) -> int:
        """ """
        if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
            p = scheme.p
            return (-(-(p + 1) // 2)) ** (self.mesh.ndim - 1)
        return 1

    def _init_timer(self, sync_timing: bool):
        self.sync_timing = sync_timing

        new_timer_cats = [
            "compute_dt",
            "apply_bc",
            "update_workspaces",
            "interpolate_faces",
            "zhang_shu_limiter",
            "integrate_fluxes",
            "MOOD_loop",
            "compute_RHS",
            "update_workspaces:shock_detector",
            "slope_limiter:detect_smooth_extrema",
            "integrate_fluxes:fallback",
            "integrate_fluxes:riemann_solver",
            "integrate_fluxes:quadrature",
            "MOOD_loop:compute_fallback_fluxes",
            "MOOD_loop:detect_troubled_cells",
            "MOOD_loop:revise_fluxes",
        ]
        new_stepper_timer = StepperTimer(self.stepper_timer.cats + new_timer_cats)
        self.stepper_timer = new_stepper_timer

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
    def init_riemann_solver(self, riemann_solver: str):
        """
        Define `self.arraywise_riemann_solver` and `self.elemewise_riemann_solver`.

        Args:
            riemann_solver: Name of the Riemann solver to use.
        """
        pass

    @MethodTimer(cat="integrate_fluxes:riemann_solver")
    @abstractmethod
    def riemann_solver(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        *,
        out: ArrayLike,
    ):
        """
        Compute the numerical flux at the interfaces using the Riemann solver and write
        the result to the `out` array.

        Args:
            wl: Array of primitive variables to the left of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            wr: Array of primitive variables to the right of the interface. Has shape
                (nvars, nx, ny, nz, ...).
            dim: Direction in which the Riemann problem is solved: "x", "y", or "z".
            out: Output array to write the numerical fluxes to. Has shape
                (nvars, nx, ny, nz, ...).
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

    @MethodTimer(cat="compute_dt")
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

    @MethodTimer(cat="update_workspaces")
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
        p = scheme.p
        xp = self.xp
        arrays = self.arrays
        lazy_primitives = getattr(scheme, "lazy_primitives", "none")

        # allocate arrays
        _u_ = arrays["_u_"]
        _ucc_ = arrays["_ucc_"]  # shape (..., 1)
        _wcc_ = arrays["_wcc_"]  # shape (..., 1)
        _wp_ = arrays["_wp_"]  # shape (..., 1)
        _w_ = arrays["_w_"]
        _shockless_ = arrays["_shockless_"]
        _PAD_violations_ = arrays["_PAD_violations_"]
        _any_PAD_violations_ = arrays["_any_troubles_"]
        _buff_ = arrays["_buffer_"]

        # 0) conservatives FV averages + BC
        _u_[self.interior] = u
        self.apply_bc(t, _u_, scheme=scheme)

        # 1) conservative and primitive centroids
        fv.interpolate_cell_centers(xp, _u_, active_dims, p, out=_ucc_, buffer=_buff_)
        _wcc_[...] = self.primitives_from_conservatives(_ucc_)

        # 2) primitive FV averages
        if lazy_primitives == "none":
            fv.integrate_fv_averages(xp, _wcc_, active_dims, p, out=_wp_, buffer=_buff_)
            _w_[...] = _wp_[..., 0]
        elif lazy_primitives == "full":
            _w_[...] = self.primitives_from_conservatives(_u_)
        elif lazy_primitives == "adaptive":
            if not isinstance(scheme, polyInterpolationScheme):
                raise ValueError(
                    "Adaptive lazy primitives can only be used with polyInterpolationScheme."
                )

            fv.integrate_fv_averages(xp, _wcc_, active_dims, p, out=_wp_, buffer=_buff_)
            _w_[...] = self.primitives_from_conservatives(_u_)

            primitives = scheme.flux_recipe in (2, 3)
            self.shock_detector(scheme, primitives)  # writes to _eta_, _shockless_

            if scheme.limiter_config.physical_admissibility_detection:
                detect_PAD_violations(
                    xp,
                    _w_,
                    self.arrays["PAD_bounds"],
                    physical_tols=scheme.limiter_config.PAD_atol,
                    out=_PAD_violations_,
                )
                _any_PAD_violations_[...] = xp.any(
                    _PAD_violations_ < 0, axis=0, keepdims=True
                )

                _w_[...] = xp.where(
                    xp.logical_and(_shockless_, _any_PAD_violations_ >= 0),
                    _wp_[..., 0],
                    _w_,
                )
            else:
                _w_[...] = xp.where(_shockless_, _wp_[..., 0], _w_)
        else:
            raise ValueError(f"Unknown lazy_primitives option: {lazy_primitives}")

    @MethodTimer(cat="apply_bc")
    def apply_bc(
        self,
        t: float,
        u: ArrayLike,
        scheme: InterpolationScheme,
    ):
        """
        Apply boundary conditions to the provided array `u` in place.

        Args:
            t: Time value.
            u: Array of conservative or primitive variables. Has shape
                (nvars, nx, ny, nz).
            scheme: Interpolation scheme to use for the boundary conditions.

        Returns:
            None. The array `u` is modified in place with the boundary conditions
                applied.
        """
        apply_bc(
            xp=self.xp,
            _u_=u,
            pad_width=self.bc_pad_width,
            mode=self.bc_mode,
            f=self.bc_callables,
            variable_index_map=self.variable_index_map,
            mesh=self.mesh,
            t=t,
            p=scheme.p,
        )

    @MethodTimer(cat="update_workspaces:shock_detector")
    def shock_detector(self, scheme: InterpolationScheme, primitives: bool):
        """
        Compute the shock detector based on the `_w_` or `_u_` workspaces depending on
        the flux recipe and write the result to the `_eta_` array.

        Args:
            scheme: Interpolation scheme to use.
            primitives: Whether to use primitive variables for shock detection.
                Otherwise, conservative variables are used.
        """
        xp = self.xp
        arrays = self.arrays
        active_dims = self.active_dims

        if not scheme.limiter_config.shock_detection:
            raise ValueError("Shock detection is not enabled in the scheme.")

        eta = arrays["_eta_"]
        shockless = arrays["_shockless_"]
        w1 = arrays["_w_"] if primitives else arrays["_u_"]
        buffer = arrays["_buffer_"]

        compute_shock_detector(
            xp,
            w1,
            w1,
            active_dims,
            scheme.limiter_config.eta_max,
            out=shockless,
            eta=eta,
            buffer=buffer,
        )

    @MethodTimer(cat="interpolate_faces")
    def interpolate_faces(
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

        out = self.arrays[f"_{dim}_nodes_"]
        buffer = self.arrays["_buffer_"]

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

    @MethodTimer(cat="zhang_shu_limiter")
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
        interior = self.interior
        lim_slc = self.variable_index_map("limiting", keepdims=True)
        limiter_config = scheme.limiter_config

        # define array references
        w = self.arrays["_w_"][lim_slc] if primitives else self.arrays["_u_"][lim_slc]
        wcc = (
            self.arrays["_wcc_"][lim_slc]
            if primitives
            else self.arrays["_ucc_"][lim_slc]
        )
        wx = self.arrays["_x_nodes_"][lim_slc] if mesh.x_is_active else None
        wy = self.arrays["_y_nodes_"][lim_slc] if mesh.y_is_active else None
        wz = self.arrays["_z_nodes_"][lim_slc] if mesh.z_is_active else None
        theta = self.arrays["_theta_"][lim_slc]
        dmp = self.arrays["_dmp_"][lim_slc]
        node_mp = self.arrays["_node_mp_"][lim_slc]
        alpha = self.arrays["_alpha_"][lim_slc]
        PAD_violations = self.arrays["_PAD_violations_"][lim_slc]
        buffer = self.arrays["_buffer_"][lim_slc]
        theta_vis = self.arrays["theta_vis"][lim_slc]
        visualize = self.arrays["visualize"][lim_slc]
        wflat = self.arrays["_flat_w_"][lim_slc]
        wjflat = self.arrays["_flat_wj_"][lim_slc]
        Mflat = self.arrays["_flat_M_"][lim_slc]
        mflat = self.arrays["_flat_m_"][lim_slc]
        Mjflat = self.arrays["_flat_Mj_"][lim_slc]
        mjflat = self.arrays["_flat_mj_"][lim_slc]
        thetaflat = self.arrays["_flat_theta_"][lim_slc]

        # compute theta
        if hasattr(xp, "cuda"):
            nvars = w.shape[0]
            nx = w.shape[1]
            ny = w.shape[2]
            nz = w.shape[3]
            ntotal = nx * ny * nz
            max_nodes = self.nodes_per_face(scheme)
            max_ninterps = 2 * max_nodes

            wflat[...] = w.reshape(nvars, ntotal)
            wjflat[..., 0] = wcc.reshape(nvars, ntotal)

            for i, wj in enumerate([wx, wy, wz]):
                if wj is None:
                    continue
                idx1 = 1 + i * max_ninterps
                idx2 = 1 + (i + 1) * max_ninterps
                wjflat[..., slice(idx1, idx2)] = wj.reshape(nvars, ntotal, max_ninterps)

            # compute DMP
            compute_dmp(
                xp,
                w,
                self.active_dims,
                out=dmp,
                include_corners=limiter_config.include_corners,
            )
            Mflat[...] = dmp[..., 1].reshape(nvars, ntotal)
            mflat[...] = dmp[..., 0].reshape(nvars, ntotal)

            compute_theta_kernel_helper(
                wflat,
                wjflat,
                Mflat,
                mflat,
                Mjflat,
                mjflat,
                thetaflat,
                limiter_config.theta_denom_tol,
            )
            theta[...] = thetaflat.reshape(nvars, nx, ny, nz, 1)
            node_mp[..., 0] = mjflat.reshape(nvars, nx, ny, nz)
            node_mp[..., 1] = Mjflat.reshape(nvars, nx, ny, nz)
        else:
            compute_theta(
                xp,
                w,
                wcc,
                wx,
                wy,
                wz,
                out=theta,
                dmp=dmp,
                node_mp=node_mp,
                buffer=buffer,
                config=limiter_config,
            )

        # SED
        if limiter_config.smooth_extrema_detection:
            self.detect_smooth_extrema(w, scheme)

            # unrelax with PAD
            if limiter_config.physical_admissibility_detection:
                PAD_bounds = limiter_config.PAD_bounds
                PAD_atol = limiter_config.PAD_atol

                if PAD_bounds is None:
                    raise ValueError(
                        "PAD_bounds must be provided when "
                        "physical_admissibility_detection is True."
                    )

                detect_PAD_violations(
                    xp, node_mp, PAD_bounds[lim_slc], PAD_atol, out=PAD_violations
                )
                alpha[..., 0] = xp.where(PAD_violations < 0, -1.0, alpha[..., 0])

            theta[...] = xp.where(alpha < 1.0, theta, 1.0)

        # limit the face nodes
        if wx is not None:
            wx[...] = zhang_shu_operator(wx, w[..., np.newaxis], theta)
        if wy is not None:
            wy[...] = zhang_shu_operator(wy, w[..., np.newaxis], theta)
        if wz is not None:
            wz[...] = zhang_shu_operator(wz, w[..., np.newaxis], theta)
        # compute theta for visualization (ignore cells with small dmp ranges)
        compute_vis(xp, dmp[interior], self.vis_rtol, self.vis_atol, out=visualize)
        theta_vis[...] = xp.where(visualize, theta[insert_slice(interior, 4, 0)], 1.0)

    @MethodTimer(cat="slope_limiter:detect_smooth_extrema")
    def detect_smooth_extrema(self, u: ArrayLike, scheme: InterpolationScheme):
        """
        Detect smooth extrema and write the result to the limiting variable slice of
        the `_alpha_` array.

        Args:
            u: Array on which smooth extrema are to be detected. Has shape
                (nvars, nx, ny, nz).
            scheme: Interpolation scheme to use for the detection.
        """
        xp = self.xp
        active_dims = self.active_dims
        arrays = self.arrays
        lim_slc = self.variable_index_map("limiting", keepdims=True)

        alpha = arrays["_alpha_"][lim_slc]
        buffer = arrays["_buffer_"][lim_slc]

        smooth_extrema_detector(
            xp,
            u[lim_slc],
            active_dims,
            scheme.limiter_config.check_uniformity,
            out=alpha,
            buffer=buffer,
            uniformity_tol=scheme.limiter_config.uniformity_tol,
        )

    @MethodTimer(cat="integrate_fluxes")
    def integrate_fluxes(
        self,
        dim: Literal["x", "y", "z"],
        scheme: InterpolationScheme,
        *,
        convert_to_primitives: bool,
    ):
        """
        Compute the flux nodes based on the primitive face node arrays
        (`x_nodes`, `y_nodes`, or `z_nodes`), perform the integration, and assign the
        result to the appropriate flux array (`F`, `G`, or `H`).

        Args:
            dim: Dimension along which to compute the fluxes of opposing faces. Can be
                "x", "y", or "z".
            scheme: Interpolation scheme to use for the integration.
            convert_to_primitives: Whether to convert the face node arrays to
                primitives before computing the fluxes.
        """
        n = self.nodes_per_face(scheme)
        pad = getattr(self.mesh, f"{dim}_slab_depth")
        axis = DIM_TO_AXIS[dim]
        flux_name = self.flux_names[dim]

        # allocate arrays
        F = self.arrays[flux_name]
        _F_ = self.arrays[f"_{flux_name}_"]
        nodes = self.arrays[f"_{dim}_nodes_"]
        fnodes = self.arrays[f"_{flux_name.lower()}_nodes_"]

        # convert to primitives if requested
        if convert_to_primitives:
            nodes[...] = self.primitives_from_conservatives(nodes)

        # sanitize face reconstruction
        self.primitive_reconstruction_fallback(
            nodes[crop(4, (None, 2 * n))], self.arrays["_w_"], scheme
        )

        wl = nodes[crop(4, (None, n))]
        wr = nodes[crop(4, (n, 2 * n))]

        left_state = wr[crop(axis, (pad - 1, -pad))]
        right_state = wl[crop(axis, (pad, -pad + 1))]
        self.riemann_solver(left_state, right_state, dim, out=fnodes)

        # perform the integration
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        self.stepper_timer.start("integrate_fluxes:quadrature")

        if self.mesh.ndim == 1:
            _F_[...] = fnodes
        else:
            if isinstance(scheme, polyInterpolationScheme) and scheme.gauss_legendre:
                fv.integrate_GaussLegendre_nodes(
                    self.xp,
                    fnodes,
                    dim,
                    self.active_dims,
                    scheme.p,
                    out=_F_[..., 0],
                )
            elif isinstance(
                scheme, (polyInterpolationScheme, musclInterpolationScheme)
            ):
                fv.transversely_integrate_nodes(
                    self.xp,
                    fnodes[..., 0],
                    dim,
                    self.active_dims,
                    scheme.p,
                    out=_F_,
                    buffer=right_state,
                )
            else:
                raise ValueError(
                    f"Unknown interpolation scheme: {scheme}. "
                    "Expected polyInterpolationScheme or musclInterpolationScheme."
                )

        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        self.stepper_timer.stop("integrate_fluxes:quadrature")

        F[...] = _F_[self.flux_interior[dim]][..., 0]

    @MethodTimer(cat="integrate_fluxes:fallback")
    def primitive_reconstruction_fallback(
        self, wp: ArrayLike, w0: ArrayLike, scheme: InterpolationScheme
    ):
        """
        Overwrite the reconstructed face states `wp` with fallback values `w0` in place
        based on physical constraints which must be enforced in a subclass.

        Args:
            wp: Array of primitive reconstructed face states. Has shape
                (nvars, nx, ny, nz, ninterpolations).
            w0: Array of primitive fallback values. Has shape (nvars, nx, ny, nz).
            scheme: Interpolation scheme to use for the reconstruction.
        """
        if not self.face_fallback:
            return

        xp = self.xp
        mesh = self.mesh

        violations = self.reconstruction_fallback_mask(wp)

        if self.log_limiter_scalars:
            n = xp.sum(violations[self.interior]).item()
            self.n_emergency_fallbacks += n

            total_nodes = self.nodes_per_face(scheme) * 2 * mesh.ndim * mesh.size
            freq = n / total_nodes
            if freq > 0.05:
                warnings.warn(
                    f"{freq * 100:.2f}% of face nodes required emergency fallback "
                    "reconstruction."
                )

        wp[...] = xp.where(violations, w0[..., xp.newaxis], wp)

    def reconstruction_fallback_mask(self, wp: ArrayLike) -> ArrayLike:
        """
        Determine where to apply an emergency reconstruction fallback to first order.

        Args:
            wp: Array of primitive reconstructed face states. Has shape
                (nvars, nx, ny, nz, ninterpolations).

        Returns:
            ArrayLike: Boolean mask indicating where to apply the fallback. Has shape
                (1, nx, ny, nz, ninterpolations).
        """
        return xp.zeros_like(wp[:1, :, :, :, :], dtype=bool)

    def compute_fluxes(self, t: float, scheme: InterpolationScheme):
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
                self.interpolate_faces(
                    averages,
                    dim,
                    scheme,
                    convert_to_primitives=limit_primitives and not interp_primitives,
                )
        elif isinstance(scheme, musclInterpolationScheme):
            self.reconstruct_muscl_faces(scheme, limit_primitives=limit_primitives)
        else:
            raise ValueError("Unknown interpolation scheme.")

        # limit the face nodes arrays with the Zhang-Shu limiter
        if isinstance(scheme.limiter_config, ZhangShuConfig):
            self.zhang_shu_limiter(scheme, primitives=limit_primitives)

        # compute the fluxes and assign them to their respective arrays
        for dim in self.active_dims:
            self.integrate_fluxes(
                dim, scheme, convert_to_primitives=not limit_primitives
            )

    @MethodTimer(cat="MOOD_loop")
    def MOOD_loop(self, t: float):
        """
        Perform the MOOD loop to detect and revise troubled cells.

        Args:
            t: Time value.
        """
        config = self.MOOD_config
        state = self.MOOD_state

        state.reset_MOOD_loop()

        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        self.stepper_timer.start("MOOD_loop:compute_fallback_fluxes")

        MOOD.compute_fallback_fluxes(self, t)

        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        self.stepper_timer.stop("MOOD_loop:compute_fallback_fluxes")

        for _ in range(config.max_iters):
            if hasattr(self.xp, "cuda"):
                self.xp.cuda.Device().synchronize()
            self.stepper_timer.start("MOOD_loop:detect_troubled_cells")

            n_revisable, n_total = MOOD.detect_troubled_cells(self, t)

            if hasattr(self.xp, "cuda"):
                self.xp.cuda.Device().synchronize()
            self.stepper_timer.stop("MOOD_loop:detect_troubled_cells")

            if n_revisable:
                if hasattr(self.xp, "cuda"):
                    self.xp.cuda.Device().synchronize()
                self.stepper_timer.start("MOOD_loop:revise_fluxes")

                MOOD.revise_fluxes(self, t)

                if hasattr(self.xp, "cuda"):
                    self.xp.cuda.Device().synchronize()
                self.stepper_timer.stop("MOOD_loop:revise_fluxes")

                state.increment_MOOD_iteration()
            else:
                break

        if n_revisable and config.detect_closing_troubles:
            if hasattr(self.xp, "cuda"):
                self.xp.cuda.Device().synchronize()
            self.stepper_timer.start("MOOD_loop:detect_troubled_cells")

            n_revisable, n_total = MOOD.detect_troubled_cells(self, t)

            if hasattr(self.xp, "cuda"):
                self.xp.cuda.Device().synchronize()
            self.stepper_timer.stop("MOOD_loop:detect_troubled_cells")

        state.update_troubled_cell_count(n_total)

    @MethodTimer(cat="compute_RHS")
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
        self.compute_fluxes(t, self.base_scheme)

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
        runtime = self.wall_timer.data["wall"].cum_time
        return self.build_update_message() + f" | (ran in {runtime:.2f}s)"

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
            raise ValueError("PAD bounds are required to enable adaptive dt.")

        PAD_violations = self.arrays["_PAD_violations_"][self.interior]
        MOOD.detect_PAD_violations(
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

        n_dt_revisions = self.n_dt_revisions
        dt_min = self.dt_min

        revised_dt = dt / 2
        if revised_dt < dt_min:
            raise RuntimeError(
                f"Adaptive dt revision resulted in dt={revised_dt} < {dt_min=} after"
                f"{n_dt_revisions} revisions."
            )

        return revised_dt

    def called_at_beginning_of_step(self):
        """
        Helper function called at the beginning of each step starting with a timer
        start preceded by a CUDA synchronization if using CuPy.
        """
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        super().called_at_beginning_of_step()

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step ending with a timer stop
        followed by a CUDA synchronization if using CuPy.
        """
        super().called_at_end_of_step()
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()

    def take_snapshot(self):
        """
        Log and time snapshot data at time `self.t` and write it to `self.path` if not
        None, all wrapped in CUDA synchronizations if using CuPy.
        """
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()
        super().take_snapshot()
        if hasattr(self.xp, "cuda"):
            self.xp.cuda.Device().synchronize()

    def prepare_snapshot_data(self) -> Dict[str, np.ndarray]:
        """
        Returns the arrays to be saved in the snapshot at time `self.t`.
        """
        interior = self.interior
        interior0 = interior + (0,)

        self.update_workspaces(self.t, self.arrays["u"], self.base_scheme)

        # write the snapshot dict
        data = {
            "u": self.arrays.get_numpy_copy("u"),
            "ucc": self.arrays.get_numpy_copy("_ucc_")[interior0],
            "wcc": self.arrays.get_numpy_copy("_wcc_")[interior0],
            "w": self.arrays.get_numpy_copy("_w_")[interior],
        }

        def normalize_across_substeps(x: ArrayLike) -> ArrayLike:
            return x / self.n_substeps if self.n_substeps > 1 else x

        # include limiting arrays in snapshot dict
        if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            theta = self.arrays["theta_log"]
            theta_vis = self.arrays["theta_vis_log"]

            theta[...] = normalize_across_substeps(theta)
            theta_vis[...] = normalize_across_substeps(theta_vis)

            data["theta"] = self.arrays.get_numpy_copy("theta_log")
            data["theta_vis"] = self.arrays.get_numpy_copy("theta_vis_log")

        if self.MOOD:
            troubles = self.arrays["troubles_log"]
            troubles_vis = self.arrays["troubles_vis_log"]
            cascade_idx = self.arrays["cascade_idx_log"]

            troubles[...] = normalize_across_substeps(troubles)
            troubles_vis[...] = normalize_across_substeps(troubles_vis)
            cascade_idx[...] = normalize_across_substeps(cascade_idx)

            data["troubles"] = self.arrays.get_numpy_copy("troubles_log")
            data["troubles_vis"] = self.arrays.get_numpy_copy("troubles_vis_log")
            data["cascade"] = self.arrays.get_numpy_copy("cascade_idx_log")

        return data

    def prepare_minisnapshot_data(self) -> Dict[str, Any]:
        """
        Returns the data to be saved in a minisnapshot, including cell update rate and
        MOOD data.
        """
        data = super().prepare_minisnapshot_data()

        n_updates = self.n_updates
        step_time = self.stepper_timer.steps[-1].data["take_step"].cum_time
        if self.n_steps > 0:
            substep_update_rate = n_updates / step_time
            update_rate = self.mesh.size / step_time
        else:
            substep_update_rate = 0.0
            update_rate = 0.0

        data.update(
            {
                "n_updates": n_updates,
                "substep_update_rate": substep_update_rate,
                "update_rate": update_rate,
            }
        )

        # log emergency face fallback statistics
        data.update(
            {
                "nfine_emergency_fallbacks": self.step_log[
                    "nfine_emergency_fallbacks"
                ].copy()
            }
        )

        if self.log_limiter_scalars:
            if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
                log_zhang_shu_scalar_statistics(self, data)

            if self.MOOD:
                log_troubled_cell_scalar_statistics(self, data)

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

        # clear emergency face fallback statistics
        self.step_log["nfine_emergency_fallbacks"].clear()

        if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            self.arrays["theta_log"].fill(0.0)
            self.arrays["theta_vis_log"].fill(0.0)
            if self.log_limiter_scalars:
                clear_zhang_shu_scalar_statistics(self)

        if self.MOOD:
            self.arrays["troubles_log"].fill(0)
            self.arrays["troubles_vis_log"].fill(0)
            self.arrays["cascade_idx_log"].fill(0)
            if self.log_limiter_scalars:
                clear_troubled_cell_scalar_statistics(self)

    def increment_substepwise_logs(self):
        """
        Increment logs at the end of each substep.
        """
        super().increment_substepwise_logs()

        xp = self.xp

        self.n_updates += self.mesh.size

        # append emergency face fallback statistics
        self.step_log["nfine_emergency_fallbacks"].append(self.n_emergency_fallbacks)
        self.n_emergency_fallbacks = 0

        if isinstance(self.base_scheme.limiter_config, ZhangShuConfig):
            theta = self.arrays["_theta_"][self.interior][..., 0]
            theta_vis = self.arrays["theta_vis"]
            theta_log = self.arrays["theta_log"]
            theta_vis_log = self.arrays["theta_vis_log"]

            xp.add(theta_log, theta, out=theta_log)
            xp.add(theta_vis_log, theta_vis, out=theta_vis_log)

            if self.log_limiter_scalars:
                append_zhang_shu_scalar_statistics(self)

        if self.MOOD:
            troubles = self.arrays["troubles"]
            troubles_vis = self.arrays["troubles_vis"]
            troubles_log = self.arrays["troubles_log"]
            troubles_vis_log = self.arrays["troubles_vis_log"]
            cascade_idx = self.arrays["cascade_idx"]
            cascade_idx_log = self.arrays["cascade_idx_log"]

            xp.add(troubles_log, troubles, out=troubles_log)
            xp.add(troubles_vis_log, troubles_vis, out=troubles_vis_log)
            xp.add(cascade_idx_log, cascade_idx, out=cascade_idx_log)

            self.MOOD_state.increment_substep_hists()

            if self.log_limiter_scalars:
                append_troubled_cell_scalar_statistics(self)

    def run(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
        q_max: int = 3,
        time_degree: Optional[int] = None,
        muscl_hancock: bool = False,
        reduce_CFL: bool = False,
    ):
        """
        Integrate the ODE system forward in time using a specified time integrator,
        either a Runge-Kutta method of order up to 4, or a MUSCL-Hancock scheme.

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
            q_max: Maximum time-integration degree (i.e., highest-order Runge-Kutt
                scheme) the solver is allowed to use. Options are:
                - 0: Forward Euler (1st order).
                - 1: SSPRK2 (2nd order).
                - 2: SSPRK3 (3rd order).
                - 3: Classical RK4 (4th order).
                The actual degree used will be the minimum of `self.p` and `q_max`.
            time_degree: If specified, override `self.p` with this value when selecting
                the time integrator.
            muscl_hancock: If True, use a MUSCL-Hancock scheme instead of a
                Runge-Kutta method. This option overrides `q_max` and `time_degree`.
                The base scheme must be a `musclInterpolationScheme`.
            reduce_CFL: If True, reduce the CFL to emulate a higher-order time
                integrator matching the order of the spatial discretization.
        """
        q = min(self.p if time_degree is None else time_degree, q_max)
        if muscl_hancock:
            self.musclhancock(
                T=T,
                n=n,
                snapshot_mode=snapshot_mode,
                allow_overshoot=allow_overshoot,
                verbose=verbose,
                log_freq=log_freq,
                max_steps=max_steps,
                path=path,
                overwrite=overwrite,
                discard=discard,
            )
            return
        match q:
            case 0:
                if reduce_CFL:
                    self.reduce_CFL(0)
                self.euler(
                    T=T,
                    n=n,
                    snapshot_mode=snapshot_mode,
                    allow_overshoot=allow_overshoot,
                    verbose=verbose,
                    log_freq=log_freq,
                    max_steps=max_steps,
                    path=path,
                    overwrite=overwrite,
                    discard=discard,
                )
            case 1:
                if reduce_CFL:
                    self.reduce_CFL(1)
                self.ssprk2(
                    T=T,
                    n=n,
                    snapshot_mode=snapshot_mode,
                    allow_overshoot=allow_overshoot,
                    verbose=verbose,
                    log_freq=log_freq,
                    max_steps=max_steps,
                    path=path,
                    overwrite=overwrite,
                    discard=discard,
                )
            case 2:
                if reduce_CFL:
                    self.reduce_CFL(2)
                self.ssprk3(
                    T=T,
                    n=n,
                    snapshot_mode=snapshot_mode,
                    allow_overshoot=allow_overshoot,
                    verbose=verbose,
                    log_freq=log_freq,
                    max_steps=max_steps,
                    path=path,
                    overwrite=overwrite,
                    discard=discard,
                )
            case 3:
                if reduce_CFL:
                    self.reduce_CFL(3)
                self.rk4(
                    T=T,
                    n=n,
                    snapshot_mode=snapshot_mode,
                    allow_overshoot=allow_overshoot,
                    verbose=verbose,
                    log_freq=log_freq,
                    max_steps=max_steps,
                    path=path,
                    overwrite=overwrite,
                    discard=discard,
                )
            case _:
                raise ValueError(f"Runge-Kutta method not implemented for {q=}")

    def musclhancock(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time using a MUSCL-Hancock scheme.

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.integrator = "musclhancock"
        self.stepper = self._musclhancock_step
        self.integrate(
            T=T,
            n=n,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
            max_steps=max_steps,
            path=path,
            overwrite=overwrite,
            discard=discard,
        )

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
        self.reconstruct_muscl_faces(
            scheme, limit_primitives=limit_primitives, hancock=True, dt=dt
        )

        # corrector step
        for dim in self.active_dims:
            self.integrate_fluxes(
                dim, scheme, convert_to_primitives=not limit_primitives
            )

        dudt = self.compute_RHS().copy()
        self.arrays["unew"][...] = u + dt * dudt

        self.increment_substepwise_logs()

    def reconstruct_muscl_faces(
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

        Notes:
            - "_buffer_" array has different shape requirements depending on whether
            SED is used and the number (length) of active dimensions:
                - without SED: (nvars, nx, ny, nz, >=4)
                - with SED, 1D: (nvars, nx, ny, nz, >=11)
                - with SED, 2D: (nvars, nx, ny, nz, >=13)
                - with SED, 3D: (nvars, nx, ny, nz, >=14)
        """
        xp = self.xp
        mesh = self.mesh
        active_dims = self.active_dims
        arrays = self.arrays

        # allocate arrays
        wx = arrays["_x_nodes_"]
        wy = arrays["_y_nodes_"]
        wz = arrays["_z_nodes_"]
        wcc = arrays["_w_"] if limit_primitives else arrays["_u_"]
        alpha = arrays["_alpha_"]

        check_buffer_slots(arrays["_buffer_"], required=4)
        dwx = arrays["_buffer_"][..., 0:1]
        dwy = arrays["_buffer_"][..., 1:2]
        dwz = arrays["_buffer_"][..., 2:3]
        wcc_for_nodes = arrays["_buffer_"][..., 3]
        lim_buff = arrays["_buffer_"][..., 4:]

        # compute limited slopes
        if scheme.limiter_config.limiter == "PP2D":
            dw1 = {"x": dwx, "y": dwy, "z": dwz}[active_dims[0]]
            dw2 = {"x": dwx, "y": dwy, "z": dwz}[active_dims[1]]

            compute_PP2D_slopes(
                xp,
                wcc,
                active_dims,
                Sx=dw1,
                Sy=dw2,
                buffer=lim_buff,
                alpha=alpha,
                config=scheme.limiter_config,
            )
        else:
            for slope_arr, dim in zip([dwx, dwy, dwz], xyz_tup):
                if dim not in active_dims:
                    continue
                compute_limited_slopes(
                    xp,
                    wcc,
                    dim,
                    active_dims,
                    out=slope_arr,
                    buffer=lim_buff,
                    alpha=alpha,
                    config=scheme.limiter_config,
                )

        # evolve the cell-center by 1/2 dt
        if hancock:
            if dt is None:
                raise ValueError(
                    "dt must be provided for MUSCL-Hancock reconstruction."
                )

            wcc_for_nodes[...] = wcc
            for slope_arr, dim in zip([dwx, dwy, dwz], xyz_tup):
                if dim not in active_dims:
                    continue
                h = getattr(mesh, "h" + dim)
                ds = self.flux_jvp(
                    wcc, slope_arr[..., 0], dim, primitives=limit_primitives
                )
                wcc_for_nodes[...] -= ds * dt / 2 / h
        else:
            wcc_for_nodes[...] = wcc

        # update the face nodes using the limited slopes
        for node_arr, slope_arr, dim in zip([wx, wy, wz], [dwx, dwy, dwz], xyz_tup):
            if dim not in active_dims:
                continue
            node_arr[..., 0] = wcc_for_nodes - slope_arr[..., 0] / 2
            node_arr[..., 1] = wcc_for_nodes + slope_arr[..., 0] / 2

    def reduce_CFL(self, q: int):
        """
        Reduce the CFL to emulate a higher-order time integrator matching the order of
        the spatial discretization.

        Args:
            q: The polynomial degree of the time integrator being used.
        """
        mesh = self.mesh
        p = self.base_scheme.p

        if p <= q:
            return

        hx, hy, hz = mesh.hx, mesh.hy, mesh.hz
        Lx = mesh.xlim[1] - mesh.xlim[0]
        Ly = mesh.ylim[1] - mesh.ylim[0]
        Lz = mesh.zlim[1] - mesh.zlim[0]
        self.CFL *= min(hx / Lx, hy / Ly, hz / Lz) ** ((p - q) / (q + 1))

    def plot_1d_slice(self, *args, **kwargs):
        return plot_1d_slice(self, *args, **kwargs)

    def plot_2d_slice(self, *args, **kwargs):
        return plot_2d_slice(self, *args, **kwargs)

    def print_timings(self, total_time_spec: str = ".2f"):
        """
        Print the timing statistics for the solver.

        Args:
            total_time_spec: Format specification for the total time column.
        """
        print(self.get_timings_message(total_time_spec=total_time_spec))

    def get_timings_message(self, total_time_spec: str) -> str:
        """
        Return a string of the timing statistics for the solver.

        Args:
            total_time_spec: Format specification for the total time column.
        """
        df = self.get_timings_df()
        df["% time"] = (
            df["Total time (s)"]
            / df.loc[df["Routine"] == "wall", "Total time (s)"].values[0]
        )

        # Nicer labels + hierarchy
        children = {
            "MOOD_loop": [
                "MOOD_loop:compute_fallback_fluxes",
                "MOOD_loop:detect_troubled_cells",
                "MOOD_loop:revise_fluxes",
            ]
        }

        for parent, kids in children.items():
            for k in kids:
                sel = df["Routine"].eq(k)
                df.loc[sel, "Routine"] = "      " + df.loc[sel, "Routine"]

        # Formats
        calls = df["# of calls"].astype(int).astype(str)
        totals = df["Total time (s)"].map(
            lambda x: "-" if pd.isna(x) or x == 0 else f"{x:{total_time_spec}}"
        )
        pct = df["% time"].map(
            lambda x: "-" if pd.isna(x) or x == 0 else f"{100*x:.1f}"
        )

        # Dynamic column widths
        w1 = max(len("Routine"), int(df["Routine"].str.len().max()))
        w2 = max(len("# of calls"), int(calls.str.len().max()))
        w3 = max(len("Total time (s)"), int(totals.str.len().max()))
        w4 = max(len("% time"), int(pd.Series(pct).str.len().max()))

        # Write string
        out = f"{'Routine':<{w1}}  {'# of calls':>{w2}}  {'Total time (s)':>{w3}} {'% time':>{w4}}\n"
        out += f"{'-'*w1}  {'-'*w2}  {'-'*w3}  {'-'*w4}\n"
        for name, c, tot, p in zip(df["Routine"], calls, totals, pct):
            out += f"{name:<{w1}}  {c:>{w2}}  {tot:>{w3}}  {p:>{w4}}\n"
        return out

    def get_timings_df(self) -> pd.DataFrame:
        """
        Get the timing statistics for the solver as a DataFrame.
        """
        wall_timer = self.wall_timer
        stepper_timer = self.stepper_timer

        data = [
            {
                "Routine": "wall",
                "# of calls": wall_timer.data["wall"].n_calls,
                "Total time (s)": wall_timer.data["wall"].cum_time,
            }
        ]
        for cat in sorted(stepper_timer.cats):
            data.append(
                {
                    "Routine": cat,
                    "# of calls": stepper_timer.total_calls(cat),
                    "Total time (s)": stepper_timer.total_time(cat),
                }
            )
        df = pd.DataFrame(data)

        return df

    def write_metadata(self):
        """
        Write commit details and config before the solver runs.
        """
        super().write_metadata()
        self.write_config()
        self.write_mesh()

    def write_config(self):
        """
        Write `self.to_dict()` to 'config.yaml'.
        """
        if self.path is None:
            return

        with open(self.path / "config.yaml", "w") as f:
            f.write(yaml_dump(self.to_dict()))

    def write_mesh(self):
        """
        Write the mesh to 'mesh.pkl'.
        """
        if self.path is None:
            return

        if self.cupy:
            self.mesh_arrays.transfer_to("cpu")

        with open(self.path / "mesh.pkl", "wb") as f:
            pickle.dump(self.mesh, f)

        if self.cupy:
            self.mesh_arrays.transfer_to("gpu")

    def to_dict(self) -> dict:
        """
        Return a dict of solver parameters independent of results.
        """
        return dict(
            active_dims=self.active_dims,
            base_scheme=self.base_scheme.to_dict(),
            bc_mode=self.bc_mode,
            CFL=self.CFL,
            dt_min=self.dt_min,
            ic=getattr(self.ic, "__name__", "unknown name"),
            integrator=self.integrator if hasattr(self, "integrator") else None,
            MOOD_config=self.MOOD_config.to_dict() if self.MOOD else None,
            face_fallback=self.face_fallback,
            mesh=self.mesh.to_dict(),
            nvars=self.nvars,
            n_passive_vars=self.n_passive_vars,
            arraywise_riemann_solver=(
                self.arraywise_riemann_solver.__name__
                if hasattr(self, "arraywise_riemann_solver")
                else None
            ),
            elemewise_riemann_solver=(
                self.elemewise_riemann_solver.__name__
                if hasattr(self, "elemewise_riemann_solver")
                and self.elemewise_riemann_solver is not None
                else None
            ),
            variable_index_map=self.variable_index_map.to_dict(),
            xp=self.xp.__name__,
        )

    def write_timings(self, total_time_spec: str = ".6f"):
        """
        Postprocess IO step that writes timing results to output directory.
        """
        if self.path is None:
            raise FileNotFoundError("Path not specified.")

        with open(self.path / "timings.txt", "w") as f:
            f.write(self.get_timings_message(total_time_spec))
