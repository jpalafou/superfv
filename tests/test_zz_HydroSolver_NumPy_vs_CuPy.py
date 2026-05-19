from functools import partial

import pytest

from superfv import (
    BC,
    CUPY_AVAILABLE,
    FallbackCascade,
    FluxQuadrature,
    HydroSolver,
    LazyPrimitiveMode,
    TimeIntegrator,
    ics,
)

if CUPY_AVAILABLE:
    import cupy as cp


@pytest.mark.parametrize(
    "scheme",
    [
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
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
def test_sedov(scheme):
    if not CUPY_AVAILABLE:
        pytest.skip("Cupy is not available, skipping test.")

    sim_np = HydroSolver(
        ic=partial(ics.sedov, h=1 / 64, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        bcy=(BC.REFLECTIVE, BC.FREE),
        bcz=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=64,
        ny=64,
        nz=64,
        **scheme,
    )
    sim_cp = HydroSolver(
        ic=partial(ics.sedov, h=1 / 64, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        bcy=(BC.REFLECTIVE, BC.FREE),
        bcz=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=64,
        ny=64,
        nz=64,
        cupy=True,
        **scheme,
    )

    if scheme.get("use_MUSCL", False):
        sim_np.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK, print_frequency=1)
        sim_cp.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK, print_frequency=1)
    else:
        sim_np.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3, print_frequency=1)
        sim_cp.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3, print_frequency=1)

    assert (
        cp.max(
            cp.abs(
                cp.asarray(sim_cp.snapshot_history[-1].u[sim_cp.idx("rho"), ...])
                - cp.asarray(sim_np.snapshot_history[-1].u[sim_np.idx("rho"), ...])
            )
        ).item()
        < 1e-12
    )
