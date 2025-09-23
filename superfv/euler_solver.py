from typing import Dict, Literal, Optional, Tuple, Union

from . import hydro, riemann_solvers
from .boundary_conditions import DirichletBC
from .finite_volume_solver import (
    FieldFunction,
    FiniteVolumeSolver,
    PassiveFieldFunction,
)
from .tools.device_management import ArrayLike
from .tools.slicing import VariableIndexMap
from .tools.timer import MethodTimer


class EulerSolver(FiniteVolumeSolver):
    """
    Solve the system of equations in up to three dimensions.
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
        riemann_solver: Literal["llf", "hllc"] = "llf",
        MUSCL: bool = False,
        MUSCL_limiter: Literal["minmod", "moncen"] = "minmod",
        ZS: bool = False,
        adaptive_dt: bool = True,
        max_dt_revisions: int = 8,
        MOOD: bool = False,
        cascade: Literal["first-order", "muscl", "full"] = "first-order",
        blend: bool = False,
        max_MOOD_iters: int = 1,
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "actives",
        NAD: bool = False,
        NAD_rtol: float = 1.0,
        NAD_atol: float = 0.0,
        absolute_dmp: bool = False,
        include_corners: bool = False,
        PAD: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        PAD_atol: float = 1e-15,
        SED: bool = False,
        cupy: bool = False,
        gamma: float = 1.4,
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
            blend: Whether to blend the troubled cell indicator with neighboring
                cells following Vilar and Abgrall 2022. Only valid for "first-order"
                and "muscl" cascades.
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
            cupy: Whether to use CuPy for array operations.
            gamma (float): Adiabatic index.
        """
        # init hydro
        self.gamma = gamma
        super().__init__(
            ic=ic,
            ic_passives=ic_passives,
            bcx=bcx,
            bcy=bcy,
            bcz=bcz,
            x_dirichlet=x_dirichlet,
            y_dirichlet=y_dirichlet,
            z_dirichlet=z_dirichlet,
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
            MUSCL=MUSCL,
            MUSCL_limiter=MUSCL_limiter,
            ZS=ZS,
            adaptive_dt=adaptive_dt,
            max_dt_revisions=max_dt_revisions,
            MOOD=MOOD,
            cascade=cascade,
            blend=blend,
            max_MOOD_iters=max_MOOD_iters,
            limiting_vars=limiting_vars,
            NAD=NAD,
            NAD_rtol=NAD_rtol,
            NAD_atol=NAD_atol,
            absolute_dmp=absolute_dmp,
            include_corners=include_corners,
            PAD=PAD,
            PAD_atol=PAD_atol,
            SED=SED,
            cupy=cupy,
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
                "v": ["v" + dim for dim in self.active_dims],
                "m": ["m" + dim for dim in self.active_dims],
                "primitives": ["rho", "v", "P"],
                "conservatives": ["rho", "m", "E"],
                "passives": ["v" + dim for dim in self.inactive_dims]
                + ["m" + dim for dim in self.inactive_dims],
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
        return hydro.prim_to_cons(
            self.xp, self.variable_index_map, w, self.active_dims, self.gamma
        )

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u: Array of conservative variables.

        Returns:
            Array of primitive variables.
        """
        return hydro.cons_to_prim(
            self.xp, self.variable_index_map, u, self.active_dims, self.gamma
        )

    def log_quantity(self) -> Dict[str, float]:
        """
        Log the minimum density and pressure.

        Returns:
            Dictionary with the following keys:
            - "min_rho": Minimum density.
            - "min_P": Minimum pressure.
        """
        min_rho, min_P = self.get_physical_mins()

        return {
            "min_rho": min_rho,
            "min_P": min_P,
        }

    def get_physical_mins(self) -> Tuple[float, float]:
        """
        Helper function for logging and printing the minimum density and pressure.

        Returns:
            Tuple of minimum density and minimum pressure from the primitive workspace
                array `self.arrays["_w_"]`.
        """
        idx = self.variable_index_map
        interior = self.interior

        min_rho = self.arrays["_w_"][interior][idx("rho")].min().item()
        min_P = self.arrays["_w_"][interior][idx("P")].min().item()

        return min_rho, min_P

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
        c = hydro.sound_speed(xp, idx, w, self.gamma)[0, ...]

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
        out[_P_] = gamma * w[_P_] * vec[_v1_] + w[_v1_] * vec[_P_]
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
            self.active_dims,
            self.gamma,
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
            self.active_dims,
            self.gamma,
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
        return riemann_solvers.hllct(
            self.xp,
            self.variable_index_map,
            wl,
            wr,
            dim,
            self.active_dims,
            self.gamma,
        )

    def build_update_message(self) -> str:
        """
        Build the update message for the FV solver, including the minimum density and
        pressure.
        """
        min_rho, min_P = self.get_physical_mins()

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
