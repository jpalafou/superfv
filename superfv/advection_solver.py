from typing import Callable, Dict, Literal, Optional, Tuple, Union

from . import riemann_solvers
from .axes import DIM_TO_AXIS
from .boundary_conditions import PatchBC
from .field import MultivarField, UnivarField
from .finite_volume_solver import FiniteVolumeSolver
from .riemann_solvers import AdvectionRiemannSolver
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap
from .tools.timer import MethodTimer


class AdvectionSolver(FiniteVolumeSolver):
    """
    Solve the advection of a scalar `u` in up to three dimensions.
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
        flux_recipe: Literal[1, 2, 3] = 1,
        lazy_primitives: Literal["none", "full", "adaptive"] = "full",
        eta_max: float = 0.025,
        riemann_solver: Literal["advection_upwind"] = "advection_upwind",
        MUSCL: bool = False,
        MUSCL_limiter: Literal["minmod", "moncen", "PP2D"] = "minmod",
        ZS: bool = False,
        adaptive_dt: bool = True,
        MOOD: bool = False,
        cascade: Literal["first-order", "muscl", "full", "none"] = "muscl",
        blend: bool = False,
        max_MOOD_iters: int = 1,
        skip_trouble_counts: bool = False,
        detect_closing_troubles: bool = True,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = ("rho",),
        NAD: bool = True,
        NAD_rtol: Optional[Union[Dict[str, float], float]] = None,
        NAD_gtol: Optional[Union[Dict[str, float], float]] = None,
        NAD_atol: Optional[Union[Dict[str, float], float]] = None,
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
        Initialize the finite volume solver of the advection equation.

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
                    averages, either by interpolating to cell-centered values
                    intermittently or transforming directly with `lazy_primitives=True`.
                    Interpolate primitive nodes from primitive cell averages.
                    Apply slope limiting to the primitive nodes.
            lazy_primitives: Option for lazy evaluation of primitive variables.
                Possible values include:
                - "none": Do not use second-order evaluation for primitive cell
                    averages.
                - "full": Always use second-order evaluation for primitive cell
                    averages.
                - "adaptive": Based on a shock-detection criterion, adaptively reduce
                    the order of conservative cell centers, primitive cell centers, and
                    primitive cell averages to second order.
                Defaults to `True` for the advection solver since the transformation is
                trivial.
            eta_max: Threshold for shock detection when `lazy_primitives` is "adaptive".
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
                - "PP2D": Only valid for 2D problems.
            ZS: Whether to use Zhang and Shu's maximum-principle-satisfying a priori
                slope limiter.
            adaptive_dt: Option for the Zhang and Shu limiter; Whether to iteratively
                halve the timestep size if the proposed solution fails PAD.
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
            NAD_rtol, NAD_gtol, NAD_atol: Tolerance values used to relax the bounds for
                numerical admissibility detection (see the `detect_NAD_violations`).
                May be provided as one of the following:
                - Dict[str, float]: A dictionary mapping variable names to their
                    corresponding tolerance values. Limiting variables not provided in
                    the dictionary are treated as having a tolerance of 0.
                - float: A single float value that is applied to all limiting
                    variables.
                - None: All limiting variables are treated as having a tolerance of 0.
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
        super().__init__(
            ic=ic,
            ic_passives=ic_passives,
            bcx=bcx,
            bcy=bcy,
            bcz=bcz,
            bcx_callable=bcx_callable,
            bcy_callable=bcy_callable,
            bcz_callable=bcz_callable,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            nx=nx,
            ny=ny,
            nz=nz,
            p=p,
            CFL=CFL,
            dt_min=dt_min,
            GL=GL,
            flux_recipe=flux_recipe,
            lazy_primitives=lazy_primitives,
            eta_max=eta_max,
            riemann_solver=riemann_solver,
            MUSCL=MUSCL,
            MUSCL_limiter=MUSCL_limiter,
            ZS=ZS,
            adaptive_dt=adaptive_dt,
            MOOD=MOOD,
            cascade=cascade,
            blend=blend,
            max_MOOD_iters=max_MOOD_iters,
            skip_trouble_counts=skip_trouble_counts,
            detect_closing_troubles=detect_closing_troubles,
            limiting_vars=limiting_vars,
            NAD=NAD,
            NAD_rtol=NAD_rtol,
            NAD_gtol=NAD_gtol,
            NAD_atol=NAD_atol,
            include_corners=include_corners,
            PAD=PAD,
            PAD_atol=PAD_atol,
            SED=SED,
            check_uniformity=check_uniformity,
            uniformity_tol=uniformity_tol,
            vis_rtol=vis_rtol,
            vis_atol=vis_atol,
            cupy=cupy,
            sync_timing=sync_timing,
        )

    def define_vars(self) -> VariableIndexMap:
        """
        Returns an VariableIndexMap object with the following variables:
            - rho: Density.
            - vx: x-component of the velocity.
            - vy: y-component of the velocity.
            - vz: z-component of the velocity.
        """
        return VariableIndexMap(
            {"rho": 0, "vx": 1, "vy": 2, "vz": 3},
            group_var_map={
                "v": ["vx", "vy", "vz"],
                "primitives": ["rho", "v"],
                "conservatives": [],
            },
        )

    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Trivial transformation for linear avection.
        """
        return w

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Trivial transformation for linear avection.
        """
        return u

    def init_riemann_solver(self, riemann_solver: str):
        """
        Define `self.arraywise_riemann_solver` and `self.elemewise_riemann_solver`.

        Args:
            riemann_solver: Name of the Riemann solver to use.
        """
        self.arraywise_riemann_solver: AdvectionRiemannSolver
        self.elemewise_riemann_solver: Optional[Callable] = None

        if hasattr(riemann_solvers, riemann_solver):
            tmp_arraywise = getattr(riemann_solvers, riemann_solver)
            if not isinstance(tmp_arraywise, AdvectionRiemannSolver):
                raise TypeError(
                    f"Riemann solver '{riemann_solver}' is not of type "
                    "AdvectionRiemannSolver."
                )
            self.arraywise_riemann_solver = tmp_arraywise
        else:
            raise ValueError(f"Riemann solver '{riemann_solver}' is not implemented.")

        make_kernel_name = f"make_{riemann_solver}_elementwise_kernel"
        if self.cupy and hasattr(riemann_solvers, make_kernel_name):
            self.elemewise_riemann_solver = getattr(riemann_solvers, make_kernel_name)(
                self.n_passive_vars
            )

    @MethodTimer(cat="riemann_solver")
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
        idx = self.variable_index_map

        if self.cupy and self.elemewise_riemann_solver is not None:
            result = self.elemewise_riemann_solver(
                wl[idx("rho")],
                wr[idx("rho")],
                wl[idx("vx")],
                wr[idx("vx")],
                wl[idx("vy")],
                wr[idx("vy")],
                wl[idx("vz")],
                wr[idx("vz")],
                DIM_TO_AXIS[dim],
                *[
                    x
                    for v in idx.group_var_map.get("passives", [])
                    for x in (wl[idx(v)], wr[idx(v)])
                ],
            )

            out[idx("rho")] = result[0]
            out[idx("vx")] = result[1]
            out[idx("vy")] = result[2]
            out[idx("vz")] = result[3]
            for i, v in enumerate(idx.group_var_map.get("passives", []), start=4):
                out[idx(v)] = result[i]
        else:
            out[...] = self.arraywise_riemann_solver(
                self.xp,
                idx,
                wl,
                wr,
                dim,
            )

    def log_quantity(self) -> Dict[str, float]:
        """
        Log the minimum and maximum density in the domain.

        Returns:
            Dictionary with the following keys:
            - "rho_min": Minimum density in the domain.
            - "rho_max": Maximum density in the domain.
        """
        idx = self.variable_index_map

        min_rho = self.arrays["u"][idx("rho")].min().item()
        max_rho = self.arrays["u"][idx("rho")].max().item()

        return {
            "rho_min": min_rho,
            "rho_max": max_rho,
        }

    @MethodTimer(cat="compute_dt")
    def compute_dt(self, t: float, u: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            t: Current time (not used in this implementation, but included for
                compatibility with the base class).
            u: Solution value. Has shape (nvars, nx, ny, nz).

        Returns:
            Time-step size.
        """
        xp = self.xp
        idx = self.variable_index_map
        mesh = self.mesh

        sum_of_v_over_h = self.arrays["sum_of_s_over_h"]

        sum_of_v_over_h[...] = 0.0
        for dim in self.active_dims:
            v = xp.abs(u[idx("v" + dim)])
            h = getattr(mesh, "h" + dim)

            sum_of_v_over_h[...] = sum_of_v_over_h + v / h

        out = self.CFL / xp.max(sum_of_v_over_h).item()
        return out

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
            dW/dt + A(W; [dim]) dW/d[dim] = 0,  W=[rho, vx, vy, vz, (passives)]

        Args:
            w: State array with shape (nvars, nx, ny, nz).
            vec: Vector to multiply with the flux Jacobian. Has shape (nvars,).
            dim: Dimension along which the flux Jacobian is computed. Can be "x", "y",
                or "z".
            primitives: Unused in advection, since the variables are always primitive.

        Returns:
            ArrayLike: The flux Jacobian-vector product A @ vec.
        """
        xp = self.xp
        idx = self.variable_index_map

        _rho_ = idx("rho")
        _v1_ = idx("v" + dim)
        _v_ = idx("v")
        _passives_ = idx("passives") if "passives" in idx else None

        out = xp.empty_like(w)
        out[_rho_] = w[_v1_] * vec[_rho_] + w[_rho_] * vec[_v1_]
        if _passives_ is not None:
            out[_passives_] = w[_v1_] * vec[_passives_] + w[_passives_] * vec[_v1_]
        out[_v_] = 0

        return out
