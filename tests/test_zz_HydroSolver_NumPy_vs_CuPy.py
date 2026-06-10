from functools import partial

import pytest

from superfv import (
    BC,
    CUPY_AVAILABLE,
    FallbackCascade,
    FluxQuadrature,
    FluxRecipe,
    HydroSolver,
    LazyPrimitiveMode,
    MUSCL_SlopeLimiter,
    SnapshotMode,
    TimeIntegrator,
    ics,
)
from superfv.riemann_solvers import RiemannSolver

if CUPY_AVAILABLE:
    import cupy as cp

N = 32


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1),
        dict(p=1, use_MUSCL=True, MUSCL_limiter=MUSCL_SlopeLimiter.NONE),
        dict(p=2),
        dict(p=2, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=3),
        dict(p=3, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=3, riemann_solver=RiemannSolver.LLF),
        dict(p=3, flux_recipe=FluxRecipe.CONS_PRIM_LIM),
        dict(p=3, flux_recipe=FluxRecipe.PRIM_PRIM_LIM, lazy_primitive_mode=LazyPrimitiveMode.NONE),
        dict(p=4),
        dict(p=4, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=5),
        dict(p=5, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=6),
        dict(p=6, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7),
        dict(p=7, flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE),
        dict(p=7, riemann_solver=RiemannSolver.LLF),
        dict(p=7, flux_recipe=FluxRecipe.CONS_PRIM_LIM),
        dict(p=7, flux_recipe=FluxRecipe.PRIM_PRIM_LIM, lazy_primitive_mode=LazyPrimitiveMode.NONE),
    ],
)
# @pytest.mark.parametrize(
#     "active_dims", [("x",), ("y",), ("z"), ("x", "y"), ("x", "z"), ("y", "z"), ("x", "y", "z")]
# )
@pytest.mark.parametrize("active_dims", [("x",), ("y",), ("z"), ("x", "y"), ("x", "z"), ("y", "z")])
def test_square_with_unlimited_schemes(scheme, active_dims):
    if not CUPY_AVAILABLE:
        pytest.skip("Cupy is not available, skipping test.")
    if (
        len(active_dims) == 1
        and scheme.get("flux_quadrature", FluxQuadrature.NONE) == FluxQuadrature.GAUSS_LEGENDRE
    ):
        pytest.skip("Flux quadrature is not supported for 1D simulations, skipping test.")

    sim_np = HydroSolver(
        ic=partial(
            ics.square,
            vx=1.0 if "x" in active_dims else 0.0,
            vy=1.0 if "y" in active_dims else 0.0,
            vz=1.0 if "z" in active_dims else 0.0,
        ),
        gamma=1.4,
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        **scheme,
    )
    sim_cp = HydroSolver(
        ic=partial(
            ics.square,
            vx=1.0 if "x" in active_dims else 0.0,
            vy=1.0 if "y" in active_dims else 0.0,
            vz=1.0 if "z" in active_dims else 0.0,
        ),
        gamma=1.4,
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        cupy=True,
        **scheme,
    )

    sim_np.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
        snapshot_mode=SnapshotMode.EVERY,
        print_frequency=1,
    )
    sim_cp.take_n_steps(
        10,
        time_integrator=(
            TimeIntegrator.MUSCL_HANCOCK
            if scheme.get("use_MUSCL", False)
            else TimeIntegrator.SSPRK3
        ),
        snapshot_mode=SnapshotMode.EVERY,
        print_frequency=1,
    )

    for i, (snapshot_np, snapshot_cp) in enumerate(
        zip(sim_np.snapshot_history, sim_cp.snapshot_history)
    ):
        rho_err = cp.max(
            cp.abs(
                cp.asarray(snapshot_np.u[sim_cp.idx("rho"), ...])
                - cp.asarray(snapshot_cp.u[sim_np.idx("rho"), ...])
            )
        ).item()
        if rho_err >= 1e-12:
            raise AssertionError(f"Density diverges after {i} steps with max error {rho_err}.")


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(
            p=3,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode=LazyPrimitiveMode.FULL,
            flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE,
        ),
        dict(
            p=7,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode=LazyPrimitiveMode.FULL,
            flux_quadrature=FluxQuadrature.GAUSS_LEGENDRE,
        ),
        dict(
            p=7,
            use_MOOD=True,
            fallback_cascade=FallbackCascade.MUSCL0,
            max_revs=3,
            MUSCL_limiter=MUSCL_SlopeLimiter.MONCEN,
            riemann_solver=RiemannSolver.LLF,
        ),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
# @pytest.mark.parametrize(
#     "active_dims", [("x",), ("y",), ("z"), ("x", "y"), ("x", "z"), ("y", "z"), ("x", "y", "z")]
# )
@pytest.mark.parametrize("active_dims", [("x",), ("y",), ("z"), ("x", "y"), ("x", "z"), ("y", "z")])
def test_sedov(scheme, active_dims):
    if not CUPY_AVAILABLE:
        pytest.skip("Cupy is not available, skipping test.")

    sim_np = HydroSolver(
        ic=partial(ics.sedov, h=1 / 64, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        bcy=(BC.REFLECTIVE, BC.FREE),
        bcz=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        **scheme,
    )
    sim_cp = HydroSolver(
        ic=partial(ics.sedov, h=1 / 64, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        bcy=(BC.REFLECTIVE, BC.FREE),
        bcz=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        cupy=True,
        **scheme,
    )

    if scheme.get("use_MUSCL", False):
        sim_np.take_n_steps(
            10,
            time_integrator=TimeIntegrator.MUSCL_HANCOCK,
            snapshot_mode=SnapshotMode.EVERY,
            print_frequency=1,
        )
        sim_cp.take_n_steps(
            10,
            time_integrator=TimeIntegrator.MUSCL_HANCOCK,
            snapshot_mode=SnapshotMode.EVERY,
            print_frequency=1,
        )
    else:
        sim_np.take_n_steps(
            10,
            time_integrator=TimeIntegrator.SSPRK3,
            snapshot_mode=SnapshotMode.EVERY,
            print_frequency=1,
        )
        sim_cp.take_n_steps(
            10,
            time_integrator=TimeIntegrator.SSPRK3,
            snapshot_mode=SnapshotMode.EVERY,
            print_frequency=1,
        )

    for i, (snapshot_np, snapshot_cp) in enumerate(
        zip(sim_np.snapshot_history, sim_cp.snapshot_history)
    ):
        if sim_np.params.fv_scheme.mood_params.use_MOOD:
            if not cp.array_equal(
                cp.asarray(snapshot_np.cascade_idx), cp.asarray(snapshot_cp.cascade_idx)
            ):
                print(f"{sim_np.step_history[i]=}")
                print(f"{sim_cp.step_history[i]=}")
                raise AssertionError(f"Cascade indices diverge after {i} steps.")
        rho_err = cp.max(
            cp.abs(
                cp.asarray(snapshot_np.u[sim_cp.idx("rho"), ...])
                - cp.asarray(snapshot_cp.u[sim_np.idx("rho"), ...])
            )
        ).item()
        if rho_err >= 1e-12:
            raise AssertionError(f"Density diverges after {i} steps with max error {rho_err}.")

    # assert (
    #     cp.max(
    #         cp.abs(
    #             cp.asarray(sim_cp.snapshot_history[-1].u[sim_cp.idx("rho"), ...])
    #             - cp.asarray(sim_np.snapshot_history[-1].u[sim_np.idx("rho"), ...])
    #         )
    #     ).item()
    #     < 1e-12
    # )
