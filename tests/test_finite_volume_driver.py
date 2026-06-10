from typing import Literal, Tuple

import numpy as np
import pytest

from superfv import BC, FluxQuadrature, FluxRecipe, LazyPrimitiveMode
from superfv.axes import DIM_TO_AXIS
from superfv.configs import (
    BoundaryConditionParameters,
    FV_SchemeParameters,
    HydroParameters,
    MOOD_Parameters,
    MUSCL_Parameters,
    MUSCL_SlopeLimiter,
    NumericalAdmissibilityParameters,
    PhysicalAdmissibilityParameters,
    ShockDetectionParameters,
    SmoothExtremaDetectionParameters,
    ZhangShuParameters,
)
from superfv.finite_volume_driver import (
    compute_fv_dudt,
    compute_fv_nghost,
    get_interior_view,
    update_fv_fluxes,
    update_fv_workspace,
)
from superfv.hydro import prim_to_cons
from superfv.mesh import UniformFiniteVolumeMesh
from superfv.riemann_solvers import RiemannSolver
from superfv.tools.slicing import replace_slice
from superfv.tools.variable_index_map import VariableIndexMap

N = 8

idx = VariableIndexMap(
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
        "passive1": 5,
        "passive2": 6,
    },
    group_var_map={
        "v": ["vx", "vy", "vz"],
        "m": ["mx", "my", "mz"],
        "primitives": ["rho", "v", "P"],
        "conservatives": ["rho", "m", "E"],
        "passives": ["passive1", "passive2"],
    },
)

hydro_params = HydroParameters(
    gamma=1.4,
    riemann_solver=RiemannSolver.HLLC,
    CFL=0.8,
)


unlimited_FV_configs = []
for p in range(8):
    for flux_recipe in [
        FluxRecipe.CONS_LIM_PRIM,
        FluxRecipe.CONS_PRIM_LIM,
        FluxRecipe.PRIM_PRIM_LIM,
    ]:
        for flux_quad in [FluxQuadrature.TRANSVERSE, FluxQuadrature.GAUSS_LEGENDRE]:
            for lazy_prim in [
                LazyPrimitiveMode.NONE,
                LazyPrimitiveMode.FULL,
                LazyPrimitiveMode.ADAPTIVE,
            ]:
                unlimited_FV_configs.append(
                    FV_SchemeParameters(
                        name=f"FV{p+1}",
                        p=p,
                        flux_recipe=flux_recipe,
                        flux_quadrature=flux_quad,
                        lazy_primitive_mode=lazy_prim,
                        muscl_params=MUSCL_Parameters(
                            False, MUSCL_SlopeLimiter.NONE, SmoothExtremaDetectionParameters(False)
                        ),
                        zhang_shu_params=ZhangShuParameters(
                            False,
                            False,
                            SmoothExtremaDetectionParameters(False),
                            PhysicalAdmissibilityParameters(False, {}),
                            [],
                        ),
                        mood_params=MOOD_Parameters(
                            False,
                            NumericalAdmissibilityParameters(
                                False, np.nan, np.nan, SmoothExtremaDetectionParameters(False), []
                            ),
                            PhysicalAdmissibilityParameters(False, {}),
                            [],
                            -1,
                            False,
                        ),
                        shock_detection_params=ShockDetectionParameters(
                            lazy_prim == LazyPrimitiveMode.ADAPTIVE
                        ),
                    )
                )


muscl_configs = []
for flux_recipe in [
    FluxRecipe.CONS_LIM_PRIM,
    FluxRecipe.CONS_PRIM_LIM,
    FluxRecipe.PRIM_PRIM_LIM,
]:
    for name in ["MUSCL", "MUSCL-Hancock"]:
        for limiter in [MUSCL_SlopeLimiter.NONE]:
            for use_SED in [False, True]:
                muscl_configs.append(
                    FV_SchemeParameters(
                        name=name,
                        p=1,
                        flux_recipe=flux_recipe,
                        flux_quadrature=FluxQuadrature.TRANSVERSE,
                        lazy_primitive_mode=LazyPrimitiveMode.FULL,
                        muscl_params=MUSCL_Parameters(
                            True, limiter, SmoothExtremaDetectionParameters(use_SED)
                        ),
                        zhang_shu_params=ZhangShuParameters(
                            False,
                            False,
                            SmoothExtremaDetectionParameters(False),
                            PhysicalAdmissibilityParameters(False, {}),
                            [],
                        ),
                        mood_params=MOOD_Parameters(
                            False,
                            NumericalAdmissibilityParameters(
                                False, np.nan, np.nan, SmoothExtremaDetectionParameters(False), []
                            ),
                            PhysicalAdmissibilityParameters(False, {}),
                            [],
                            -1,
                            False,
                        ),
                        shock_detection_params=ShockDetectionParameters(False),
                    )
                )


ZS_configs = []
for p in [3, 7]:
    for flux_recipe in [
        FluxRecipe.CONS_LIM_PRIM,
        FluxRecipe.CONS_PRIM_LIM,
        FluxRecipe.PRIM_PRIM_LIM,
    ]:
        for flux_quad in [FluxQuadrature.TRANSVERSE, FluxQuadrature.GAUSS_LEGENDRE]:
            for lazy_prim in [
                LazyPrimitiveMode.NONE,
                LazyPrimitiveMode.FULL,
                LazyPrimitiveMode.ADAPTIVE,
            ]:
                for use_SED in [False, True]:
                    ZS_configs.append(
                        FV_SchemeParameters(
                            name=f"FV{p+1}",
                            p=p,
                            flux_recipe=flux_recipe,
                            flux_quadrature=flux_quad,
                            lazy_primitive_mode=lazy_prim,
                            muscl_params=MUSCL_Parameters(
                                False,
                                MUSCL_SlopeLimiter.NONE,
                                SmoothExtremaDetectionParameters(False),
                            ),
                            zhang_shu_params=ZhangShuParameters(
                                True,
                                False,
                                SmoothExtremaDetectionParameters(use_SED),
                                PhysicalAdmissibilityParameters(False, {}),
                                [],
                            ),
                            mood_params=MOOD_Parameters(
                                False,
                                NumericalAdmissibilityParameters(
                                    False,
                                    np.nan,
                                    np.nan,
                                    SmoothExtremaDetectionParameters(False),
                                    [],
                                ),
                                PhysicalAdmissibilityParameters(False, {}),
                                [],
                                -1,
                                False,
                            ),
                            shock_detection_params=ShockDetectionParameters(
                                lazy_prim == LazyPrimitiveMode.ADAPTIVE
                            ),
                        )
                    )


@pytest.mark.parametrize("base_scheme", unlimited_FV_configs + muscl_configs + ZS_configs)
@pytest.mark.parametrize(
    "active_dims", [("x",), ("y",), ("z"), ("x", "y"), ("x", "z"), ("y", "z"), ("x", "y", "z")]
)
def test_fv_rhs_is_finite(
    base_scheme: FV_SchemeParameters, active_dims: Tuple[Literal["x", "y", "z"], ...]
):
    if len(active_dims) == 1 and base_scheme.flux_quadrature == FluxQuadrature.GAUSS_LEGENDRE:
        pytest.skip("Gauss-Legendre quadrature is not implemented for 1D fluxes.")

    print(f"{base_scheme=}")

    # Build mesh
    mesh = UniformFiniteVolumeMesh(
        xlims=(0.0, 1.0),
        ylims=(0.0, 1.0),
        zlims=(0.0, 1.0),
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        nghost=compute_fv_nghost(base_scheme, len(active_dims)),
        active_dims=active_dims,
    )

    bc_params = BoundaryConditionParameters(
        bcx=(BC.FREE, BC.FREE) if "x" in active_dims else (BC.NONE, BC.NONE),
        bcy=(BC.FREE, BC.FREE) if "y" in active_dims else (BC.NONE, BC.NONE),
        bcz=(BC.FREE, BC.FREE) if "z" in active_dims else (BC.NONE, BC.NONE),
    )

    # Allocate arrays
    u = np.full((idx.nvars, *mesh.shape), np.nan)
    _u_ = np.full((idx.nvars, *mesh._shape_), np.nan)
    _w_ = _u_.copy()
    _has_shock_ = (
        np.full((1, *mesh._shape_), -1, dtype=np.int32)
        if base_scheme.shock_detection_params.use_shock_detection
        else np.array([])
    )
    _F_ = (
        np.full((idx.nvars, mesh.nx + 1, mesh._ny_, mesh._nz_), np.nan)
        if "x" in active_dims
        else np.array([])
    )
    _G_ = (
        np.full((idx.nvars, mesh._nx_, mesh.ny + 1, mesh._nz_), np.nan)
        if "y" in active_dims
        else np.array([])
    )
    _H_ = (
        np.full((idx.nvars, mesh._nx_, mesh._ny_, mesh.nz + 1), np.nan)
        if "z" in active_dims
        else np.array([])
    )
    source = np.zeros_like(u)
    _theta_ = _u_.copy() if base_scheme.zhang_shu_params.use_ZS else np.array([])
    _qcc_ = _u_.copy() if base_scheme.zhang_shu_params.use_ZS else np.array([])
    _alpha_ = _u_.copy()

    # Apply initial conditions
    w = np.empty_like(u)
    w[idx("rho"), ...] = 1.0
    w[idx("vx"), ...] = 0.1
    w[idx("vy"), ...] = 0.1
    w[idx("vz"), ...] = 0.1
    w[idx("P"), ...] = 1.0
    w[idx("passive1"), ...] = 0.5
    w[idx("passive2"), ...] = 0.25
    prim_to_cons(w, u, idx, hydro_params.gamma)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        update_fv_workspace(
            u,
            _u_,
            _w_,
            _has_shock_,
            0.0,
            idx,
            mesh,
            base_scheme,
            bc_params,
            hydro_params,
        )
        update_fv_fluxes(
            _u_,
            _w_,
            _F_,
            _G_,
            _H_,
            _theta_,
            _qcc_,
            _alpha_,
            idx,
            mesh.active_dims,
            mesh,
            base_scheme,
            hydro_params,
        )

    interior = get_interior_view(mesh.active_dims, mesh.nghost)
    dudt = compute_fv_dudt(
        (
            _F_[replace_slice(interior, DIM_TO_AXIS["x"], slice(None))]
            if "x" in active_dims
            else np.array([])
        ),
        (
            _G_[replace_slice(interior, DIM_TO_AXIS["y"], slice(None))]
            if "y" in active_dims
            else np.array([])
        ),
        (
            _H_[replace_slice(interior, DIM_TO_AXIS["z"], slice(None))]
            if "z" in active_dims
            else np.array([])
        ),
        source,
        mesh,
    )

    assert np.all(np.isfinite(dudt))
