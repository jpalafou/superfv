from typing import Dict, Literal, Optional, Tuple, Union

from . import hydro, riemann_solvers
from .boundary_conditions import PatchBC
from .field import MultivarField, UnivarField
from .finite_volume_solver import FiniteVolumeSolver
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap
from .tools.timer import MethodTimer


class EulerSolver(FiniteVolumeSolver):
    """
    Solve the system of equations in up to three dimensions.
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
        GL: bool = False,
        flux_recipe: Literal[1, 2, 3] = 1,
        lazy_primitives: Literal["none", "full", "adaptive"] = "none",
        eta_max: float = 0.025,
        riemann_solver: Literal["llf", "hllc"] = "llf",
        MUSCL: bool = False,
        MUSCL_limiter: Literal["minmod", "moncen", "PP2D"] = "minmod",
        ZS: bool = False,
        adaptive_dt: bool = True,
        max_dt_revisions: int = 8,
        MOOD: bool = False,
        cascade: Literal["first-order", "muscl", "full"] = "first-order",
        blend: bool = False,
        max_MOOD_iters: int = 1,
        skip_trouble_counts: bool = False,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "actives",
        NAD: bool = False,
        NAD_rtol: float = 1.0,
        NAD_atol: float = 0.0,
        absolute_dmp: bool = False,
        include_corners: bool = False,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        PAD_atol: float = 1e-15,
        SED: bool = False,
        vis_rtol: float = 1e-1,
        vis_atol: float = 1e-10,
        cupy: bool = False,
        sync_timing: bool = True,
        gamma: float = 1.4,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        """
        Initialize the finite volume solver for the Euler equations.

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
            max_dt_revisions: Option for the Zhang and Shu limiter; The maximum number
                of timestep size revisions that may be attempted in an update step
                if `adaptive_dt=True`. Defaults to 8.
            MOOD: Whether to use MOOD for a posteriori flux revision. Ignored if
                `ZS=True` and `adaptive_timestepping=True`.
            cascade: A string indicating which type of MOOD cascade to use:
                - "first-order": Fall back directly to a first-order scheme.
                - "muscl": Fall back directly to a MUSCL scheme.
                - "full": Fall back to a full cascade of scheme in descending order.
            blend: Whether to blend the troubled cell indicator with neighboring
                cells following Vilar and Abgrall 2022. Only valid for "first-order"
                and "muscl" cascades.
            max_MOOD_iters: Option for the MOOD limiter; The maximum number of MOOD
                iterations that may be performed in an update step. Defaults to 1.
            skip_trouble_counts: Whether to skip counting the number of troubled cells.
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
            absolute_dmp: Whether to use the absolute values of the DMP instead of the
            range to set the NAD bounds. The NAD condition for each case is:
            - `absolute_dmp=False`:
                umin-rtol*(umax-umin)-atol <= u_new <= umax+rtol*(umax-umin)+atol
            - `absolute_dmp=True`:
                umin-rtol*|umin|-atol <= u_new <= umax+rtol*|umax|+atol
            include_corners: Whether to include corner nodes in the slope limiting.
            PAD: Dict of `limiting_vars` and their corresponding PAD tolerances as a
                tuple: (lower_bound, upper_bound). Any variable or bound not provided
                in `PAD` is given a lower and upper bound of `-np.inf` and `np.inf`
                respectively.
            PAD_atol: Tolerance for the PAD check as an absolute value from the minimum
                and maximum values of the variable.
            SED: Whether to use smooth extrema detection for slope limiting.
            vis_rtol, vis_atol: Relative and absolute tolerances for the visualization
                threshold. See `compute_vis`.
            cupy: Whether to use CuPy for array operations.
            sync_timing: Whether to synchronize the GPU after each timed method call if
                using CuPy. This ensures accurate timing measurements when profiling.
            gamma (float): Adiabatic index.
            isothermal (bool): If True, use an isothermal equation of state where
                pressure is directly proportional to density. If True, the `gamma`
                parameter is ignored.
            iso_cs (float): Isothermal sound speed. Used only if `isothermal=True`.
        """
        # init hydro
        self.gamma = gamma
        self.isothermal = isothermal
        self.iso_cs = iso_cs
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
            GL=GL,
            riemann_solver=riemann_solver,
            flux_recipe=flux_recipe,
            lazy_primitives=lazy_primitives,
            eta_max=eta_max,
            MUSCL=MUSCL,
            MUSCL_limiter=MUSCL_limiter,
            ZS=ZS,
            adaptive_dt=adaptive_dt,
            max_dt_revisions=max_dt_revisions,
            MOOD=MOOD,
            cascade=cascade,
            blend=blend,
            max_MOOD_iters=max_MOOD_iters,
            skip_trouble_counts=skip_trouble_counts,
            limiting_vars=limiting_vars,
            NAD=NAD,
            NAD_rtol=NAD_rtol,
            NAD_atol=NAD_atol,
            absolute_dmp=absolute_dmp,
            include_corners=include_corners,
            PAD=PAD,
            PAD_atol=PAD_atol,
            SED=SED,
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
            group_var_map={
                "v": ["vx", "vy", "vz"],
                "m": ["mx", "my", "mz"],
                "primitives": ["rho", "v", "P"],
                "conservatives": ["rho", "m", "E"],
            },
        )

    def conservatives_from_primitives(self, w: ArrayLike) -> ArrayLike:
        """
        Convert primitive variables to conservative variables.

        Args:
            w: Array of primitive variables.

        Returns:
            Array of conservative variables.
        """
        return hydro.prim_to_cons(self.xp, self.variable_index_map, w, self.gamma)

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u: Array of conservative variables.

        Returns:
            Array of primitive variables.
        """
        return hydro.cons_to_prim(
            self.xp,
            self.variable_index_map,
            u,
            self.gamma,
            self.isothermal,
            self.iso_cs,
        )

    def log_quantity(self) -> Dict[str, float]:
        """
        Log the minimum density and pressure.

        Returns:
            Dictionary with the following keys:
            - "min_rho": Minimum density.
            - "max_rho": Maximum density.
            - "min_P": Minimum pressure.
            - "max_P": Maximum pressure.
            - "total_rho": Total density.
            - "total_E": Total energy.
        """
        min_rho, max_rho, min_P, max_P, total_rho, total_E = self.get_physical_scalars()

        return {
            "min_rho": min_rho,
            "max_rho": max_rho,
            "min_P": min_P,
            "max_P": max_P,
            "total_rho": total_rho,
            "total_E": total_E,
        }

    def get_physical_scalars(self) -> Tuple[float, float, float, float, float, float]:
        """
        Compute global physical scalars from the "_w_" array.

        Returns:
            Tuple of global physical scalars:
            - min_rho: Minimum density.
            - max_rho: Maximum density.
            - min_P: Minimum pressure.
            - max_P: Maximum pressure.
            - total_rho: Total density.
            - total_E: Total energy.
        """
        idx = self.variable_index_map
        interior = self.interior

        u = self.arrays["_u_"][interior]
        w = self.arrays["_w_"][interior]

        min_rho = u[idx("rho")].min().item()
        max_rho = u[idx("rho")].max().item()
        min_P = w[idx("P")].min().item()
        max_P = w[idx("P")].max().item()
        total_rho = u[idx("rho")].sum().item()
        total_E = u[idx("E")].sum().item()

        return min_rho, max_rho, min_P, max_P, total_rho, total_E

    @MethodTimer(cat="compute_dt")
    def compute_dt(self, t: float, u: ArrayLike) -> float:
        """
        Compute the time-step size based on the CFL condition.

        Args:
            t: Current time (not used in this implementation, but included for
                compatibility with the base class).
            u: Array of finite volume averaged conservative variables.

        Returns:
            Time-step size.
        """
        xp = self.xp
        idx = self.variable_index_map
        mesh = self.mesh

        sum_of_s_over_h = self.arrays["sum_of_s_over_h"]

        w = self.primitives_from_conservatives(u)
        c = (
            self.iso_cs
            if self.isothermal
            else hydro.sound_speed(xp, idx, w, self.gamma)[0, ...]
        )

        sum_of_s_over_h[...] = 0.0
        for dim in self.active_dims:
            v = xp.abs(w[idx("v" + dim)]) + c
            h = getattr(mesh, "h" + dim)

            sum_of_s_over_h[...] = sum_of_s_over_h + v / h

        out = self.CFL / xp.max(sum_of_s_over_h).item()

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
            dW/dt + A(W; [dim]) dW/d[dim] = 0,  W=[rho, vx, vy, vz, P, (passives)]
        if `primitives=True`, or the conservative-variable quasilinear system
            dU/dt + A(U; [dim]) dU/d[dim] = 0,  U=[rho, mx, my, mz, E, (rho * passives)]
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
        if not primitives:
            raise NotImplementedError(
                "This function doesn't support conservative variables at this moment."
            )

        xp = self.xp
        idx = self.variable_index_map
        gamma = self.gamma

        _rho_ = idx("rho")
        _v1_ = idx("v" + dim)
        _vx_ = idx("vx")
        _vy_ = idx("vy")
        _vz_ = idx("vz")
        _P_ = idx("P")
        _passives_ = idx("passives") if "passives" in idx else None

        out = xp.empty_like(w)
        out[_rho_] = w[_v1_] * vec[_rho_] + w[_rho_] * vec[_v1_]
        out[_vx_] = w[_v1_] * vec[_vx_]
        out[_vy_] = w[_v1_] * vec[_vy_]
        out[_vz_] = w[_v1_] * vec[_vz_]
        out[_v1_] += (1 / w[_rho_]) * vec[_P_]
        out[_P_] = (
            self.iso_cs**2 * out[_rho_]
            if self.isothermal
            else gamma * w[_P_] * vec[_v1_] + w[_v1_] * vec[_P_]
        )
        if _passives_ is not None:
            out[_passives_] = w[_v1_] * vec[_passives_] + w[_passives_] * vec[_v1_]

        return out

    def llf(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
    ) -> ArrayLike:
        """
        LLF implementation. See FiniteVolumeSolver.dummy_riemann_solver for details.
        """
        return riemann_solvers.llf(
            self.xp,
            self.variable_index_map,
            wl,
            wr,
            dim,
            self.gamma,
            self.isothermal,
            self.iso_cs,
        )

    def hllc(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
    ) -> ArrayLike:
        """
        HLLC implementation. See FiniteVolumeSolver.dummy_riemann_solver for details.
        """
        return riemann_solvers.hllc(
            self.xp,
            self.variable_index_map,
            wl,
            wr,
            dim,
            self.gamma,
            self.isothermal,
            self.iso_cs,
        )

    def hllct(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
    ) -> ArrayLike:
        """
        HLLC Teyssier implementation. See FiniteVolumeSolver.dummy_riemann_solver for details.
        """
        if self.mesh.ndim > 1:
            raise NotImplementedError("HLLCT is only implemented for 1D problems.")
        return riemann_solvers.hllct(
            self.xp,
            self.variable_index_map,
            wl,
            wr,
            dim,
            self.gamma,
        )

    def build_update_message(self) -> str:
        """
        Build the update message for the FV solver, including the minimum density and
        pressure.
        """
        min_rho, _, min_P, _, _, _ = self.get_physical_scalars()

        message = super().build_update_message()
        message += f" | min(rho)={min_rho:.2e}, min(P)={min_P:.2e}"

        return message

    def to_dict(self) -> dict:
        """
        Return a dict of solver parameters independent of results.
        """
        out = super().to_dict()
        out["gamma"] = self.gamma
        return out
