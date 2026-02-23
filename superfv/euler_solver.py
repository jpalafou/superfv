from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np

from superfv.slope_limiting.shock_detection import compute_shock_detector

from . import riemann_solvers
from .axes import DIM_TO_AXIS
from .boundary_conditions import PatchBC
from .field import MultivarField, UnivarField
from .finite_volume_solver import FiniteVolumeSolver
from .hydro import cons_to_prim, prim_to_cons, sound_speed
from .interpolation_schemes import InterpolationScheme
from .riemann_solvers import HydroRiemannSolver
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import VariableIndexMap
from .tools.timer import MethodTimer

if CUPY_AVAILABLE:
    from .hydro import (
        make_cons_to_prim_elementwise_kernel,
        make_prim_to_cons_elementwise_kernel,
        sound_speed_cp,
    )
    from .riemann_solvers import make_hllc_elementwise_kernel


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
        dt_min: float = 1e-15,
        GL: bool = False,
        flux_recipe: Literal[1, 2, 3] = 2,
        lazy_primitives: Literal["none", "full", "adaptive"] = "none",
        eta_max: float = 0.025,
        riemann_solver: Literal["llf", "hllc"] = "hllc",
        face_fallback: bool = True,
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
        limiting_vars: Union[Literal["all", "actives"], Tuple[str, ...]] = "actives",
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
        cupy: bool = False,
        sync_timing: bool = True,
        gamma: float = 1.4,
        isothermal: bool = False,
        iso_cs: float = 1.0,
        rho_min: float = 1e-12,
        P_min: float = 1e-12,
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
            uniformity_tol: Tolerance for uniformity check when check_uniformity is
                True.
            cupy: Whether to use CuPy for array operations.
            sync_timing: Whether to synchronize the GPU after each timed method call if
                using CuPy. This ensures accurate timing measurements when profiling.
            gamma (float): Adiabatic index.
            isothermal (bool): If True, use an isothermal equation of state where
                pressure is directly proportional to density. If True, the `gamma`
                parameter is ignored.
            iso_cs (float): Isothermal sound speed. Used only if `isothermal=True`.
            rho_min, P_min (float): Density and pressure floors when
                `face_fallback=True`, ignored otherwise.
        """
        # init hydro
        self.gamma = gamma
        self.isothermal = isothermal
        self.iso_cs = iso_cs
        self.rho_min = rho_min
        self.P_min = P_min

        # init base class
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
            riemann_solver=riemann_solver,
            face_fallback=face_fallback,
            flux_recipe=flux_recipe,
            lazy_primitives=lazy_primitives,
            eta_max=eta_max,
            MUSCL=MUSCL,
            MUSCL_limiter=MUSCL_limiter,
            ZS=ZS,
            adaptive_dt=adaptive_dt,
            log_limiter_scalars=log_limiter_scalars,
            MOOD=MOOD,
            cascade=cascade,
            blend=blend,
            max_MOOD_iters=max_MOOD_iters,
            skip_trouble_counts=skip_trouble_counts,
            detect_closing_troubles=detect_closing_troubles,
            limiting_vars=limiting_vars,
            NAD=NAD,
            NAD_delta=NAD_delta,
            NAD_rtol=NAD_rtol,
            NAD_gtol=NAD_gtol,
            NAD_atol=NAD_atol,
            scale_NAD_rtol_by_dt=scale_NAD_rtol_by_dt,
            include_corners=include_corners,
            PAD=PAD,
            PAD_atol=PAD_atol,
            SED=SED,
            check_uniformity=check_uniformity,
            uniformity_tol=uniformity_tol,
            cupy=cupy,
            sync_timing=sync_timing,
        )

        # init hydro arrays
        nvars, _nx_, _ny_, _nz_ = (
            self.nvars,
            self.mesh._nx_,
            self.mesh._ny_,
            self.mesh._nz_,
        )
        self.arrays.add("_c_", np.empty((1, _nx_, _ny_, _nz_)))
        self.arrays.add("_wr_", np.empty((nvars, _nx_, _ny_, _nz_)))

        # special cupy functions
        if self.cupy:
            n_passives = self.n_passive_vars
            self.prim_to_cons_cp = make_prim_to_cons_elementwise_kernel(n_passives)
            self.cons_to_prim_cp = make_cons_to_prim_elementwise_kernel(n_passives)
            self.hllc_cp = make_hllc_elementwise_kernel(n_passives)

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
        xp = self.xp
        idx = self.variable_index_map
        gamma = self.gamma

        if self.cupy and hasattr(self, "prim_to_cons_cp"):
            return xp.asarray(
                [
                    w[idx("rho")],
                    *self.prim_to_cons_cp(
                        w[idx("rho")],
                        w[idx("vx")],
                        w[idx("vy")],
                        w[idx("vz")],
                        w[idx("P")],
                        gamma,
                        *(w[idx(v)] for v in idx.group_var_map.get("passives", [])),
                    ),
                ]
            )
        else:
            return prim_to_cons(xp, idx, w, gamma)

    def primitives_from_conservatives(self, u: ArrayLike) -> ArrayLike:
        """
        Convert conservative variables to primitive variables.

        Args:
            u: Array of conservative variables.

        Returns:
            Array of primitive variables.
        """
        xp = self.xp
        idx = self.variable_index_map
        gamma = self.gamma

        if self.cupy and hasattr(self, "cons_to_prim_cp"):
            return xp.asarray(
                [
                    u[idx("rho")],
                    *self.cons_to_prim_cp(
                        u[idx("rho")],
                        u[idx("mx")],
                        u[idx("my")],
                        u[idx("mz")],
                        u[idx("E")],
                        gamma,
                        self.isothermal,
                        self.iso_cs,
                        *(u[idx(v)] for v in idx.group_var_map.get("passives", [])),
                    ),
                ]
            )
        else:
            return cons_to_prim(
                self.xp,
                self.variable_index_map,
                u,
                self.gamma,
                self.isothermal,
                self.iso_cs,
            )

    def init_riemann_solver(self, riemann_solver: str):
        """
        Define `self.arraywise_riemann_solver` and `self.elemewise_riemann_solver`.

        Args:
            riemann_solver: Name of the Riemann solver to use.
        """
        self.arraywise_riemann_solver: HydroRiemannSolver
        self.elemewise_riemann_solver: Optional[Callable] = None

        if hasattr(riemann_solvers, riemann_solver):
            tmp_arraywise = getattr(riemann_solvers, riemann_solver)
            if not isinstance(tmp_arraywise, HydroRiemannSolver):
                raise TypeError(
                    f"Riemann solver '{riemann_solver}' is not of type "
                    "HydroRiemannSolver."
                )
            self.arraywise_riemann_solver = tmp_arraywise
        else:
            raise ValueError(f"Riemann solver '{riemann_solver}' is not implemented.")

        make_kernel_name = f"make_{riemann_solver}_elementwise_kernel"
        if self.cupy and hasattr(riemann_solvers, make_kernel_name):
            self.elemewise_riemann_solver = getattr(riemann_solvers, make_kernel_name)(
                self.n_passive_vars
            )

    @MethodTimer(cat="integrate_fluxes:riemann_solver")
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
        gamma = self.gamma

        if self.cupy and self.elemewise_riemann_solver is not None:
            cl = self.compute_sound_speed(wl)
            cr = self.compute_sound_speed(wr)
            self.elemewise_riemann_solver(
                wl[idx("rho")],
                wr[idx("rho")],
                wl[idx("vx")],
                wr[idx("vx")],
                wl[idx("vy")],
                wr[idx("vy")],
                wl[idx("vz")],
                wr[idx("vz")],
                wl[idx("P")],
                wr[idx("P")],
                cl,
                cr,
                gamma,
                DIM_TO_AXIS[dim],
                *[
                    x
                    for v in idx.group_var_map.get("passives", [])
                    for x in (wl[idx(v)], wr[idx(v)])
                ],
                out[idx("rho")],
                out[idx("mx")],
                out[idx("my")],
                out[idx("mz")],
                out[idx("E")],
                *[out[idx(v)] for v in idx.group_var_map.get("passives", [])],
            )
        else:
            out[...] = self.arraywise_riemann_solver(
                self.xp,
                idx,
                wl,
                wr,
                dim,
                gamma,
                self.isothermal,
                self.iso_cs,
            )

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
        idx = self.variable_index_map
        xp = self.xp

        rho = wp[idx("rho", keepdims=True)]
        P = wp[idx("P", keepdims=True)]

        violations = (rho < self.rho_min) | xp.isnan(rho)
        violations |= (P < self.P_min) | xp.isnan(P)

        return violations

    def log_quantity(self) -> Dict[str, float]:
        """
        Log the global physical scalars.

        Returns:
            Dictionary of global physical scalars with the following keys:
            - "rho_min": Minimum cell-averaged density.
            - "rho_max": Maximum cell-averaged density.
            - "P_min": Minimum cell-averaged pressure.
            - "P_max": Maximum cell-averaged pressure.
            - "rho_total": Total density.
            - "E_total": Total energy.
            - "E_cons": Absolute energy conservation error.
        """
        return self.get_physical_scalars()

    def get_physical_scalars(self) -> Dict[str, float]:
        """
        Compute global physical scalars from the "u" and "w" arrays.

        Returns:
            Dictionary of global physical scalars with the following keys:
            - "rho_min": Minimum cell-averaged density.
            - "rho_max": Maximum cell-averaged density.
            - "P_min": Minimum cell-averaged pressure.
            - "P_max": Maximum cell-averaged pressure.
            - "rho_total": Total density.
            - "E_total": Total energy.
            - "E_cons": Absolute energy conservation error.
        """
        idx = self.variable_index_map
        interior = self.interior

        u = self.arrays["_u_"][interior]
        w = self.arrays["_w_"][interior]

        rho_min = u[idx("rho")].min().item()
        rho_max = u[idx("rho")].max().item()
        P_min = w[idx("P")].min().item()
        P_max = w[idx("P")].max().item()
        rho_total = u[idx("rho")].sum().item()
        E_total = u[idx("E")].sum().item()

        if self.n_steps == 0:
            E_cons = 0.0
        else:
            E_cons = abs(E_total - self.minisnapshots["E_total"][0])

        scalar_packet = {
            "rho_min": rho_min,
            "rho_max": rho_max,
            "P_min": P_min,
            "P_max": P_max,
            "rho_total": rho_total,
            "E_total": E_total,
            "E_cons": E_cons,
        }
        return scalar_packet

    def compute_sound_speed(self, w: ArrayLike) -> Union[ArrayLike, float]:
        idx = self.variable_index_map
        gamma = self.gamma

        if self.isothermal:
            return self.iso_cs
        elif self.cupy:
            return sound_speed_cp(w[idx("rho")], w[idx("P")], gamma)
        else:
            return sound_speed(self.xp, idx, w, gamma)

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
        c = self.compute_sound_speed(w)

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

    @MethodTimer(cat="update_workspaces:shock_detector")
    def shock_detector(self, scheme: InterpolationScheme, primitives: bool):
        """
        Compute the hydro shock detector based on the `_w_` or `_u_` workspaces depending
        on the flux recipe and write the result to the `_eta_` array.

        Args:
            scheme: Interpolation scheme to use.
            primitives: Whether to use primitive variables for shock detection.
                Otherwise, conservative variables are used.
        """
        xp = self.xp
        arrays = self.arrays
        active_dims = self.active_dims
        idx = self.variable_index_map

        if not scheme.limiter_config.shock_detection:
            raise ValueError("Shock detection is not enabled in the scheme.")

        eta = arrays["_eta_"]
        shockless = arrays["_shockless_"]
        w1 = arrays["_w_"]
        u1 = arrays["_u_"]
        wr = arrays["_wr_"]
        c = arrays["_c_"]
        buffer = arrays["_buffer_"]

        c[...] = self.compute_sound_speed(w1)
        if primitives:
            wr[...] = w1
            wr[idx("v")] = c
        else:
            wr[...] = u1
            wr[idx("m")] = c * u1[idx("rho")]

        compute_shock_detector(
            xp,
            w1 if primitives else u1,
            wr,
            active_dims,
            scheme.limiter_config.eta_max,
            out=shockless,
            eta=eta,
            buffer=buffer,
        )

    def build_update_message(self) -> str:
        """
        Build the update message for the FV solver, including the minimum density and
        pressure.
        """
        scalar_packet = self.get_physical_scalars()

        rho_min = scalar_packet["rho_min"]
        P_min = scalar_packet["P_min"]
        E_cons = scalar_packet["E_cons"]

        message = super().build_update_message()
        message += (
            f" | min(rho)={rho_min:.2e}, min(P)={P_min:.2e} | E_cons={E_cons:.2e}"
        )

        return message

    def to_dict(self) -> dict:
        """
        Return a dict of solver parameters independent of results.
        """
        out = super().to_dict()
        out["gamma"] = self.gamma
        out["rho_min"] = self.rho_min
        out["P_min"] = self.P_min
        return out
