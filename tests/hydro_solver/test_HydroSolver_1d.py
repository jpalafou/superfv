from functools import partial

import pytest

from superfv import (
    BC,
    FallbackCascade,
    HydroSolver,
    LazyPrimitiveMode,
    TimeIntegrator,
    ic,
)


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, max_revs=7),
    ],
)
def test_sedov(scheme):
    sim = HydroSolver(
        ic=partial(ic.sedov, h=1 / 100, gamma=1.4, P0=1e-5),
        bcx=(BC.REFLECTIVE, BC.FREE),
        gamma=1.4,
        nx=100,
        **scheme,
    )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.take_n_steps(10, time_integrator=TimeIntegrator.SSPRK3)


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=1, use_MUSCL=True),
        dict(p=3, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_ZS=True, adaptive_dt=True, lazy_primitive_mode=LazyPrimitiveMode.ADAPTIVE),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.MUSCL0, rtol=0, max_revs=64),
        dict(p=7, use_MOOD=True, fallback_cascade=FallbackCascade.FULL, rtol=0, max_revs=64),
    ],
)
def test_preservation_of_maximum_principle(scheme):
    with pytest.warns(UserWarning, match="PAD lower bound for 'rho' is 1, which is different from the hydro parameter rho_min=1e-12."):
        sim = HydroSolver(
            ic=partial(ic.square, rhomin=1, rhomax=2, vx=1),
            PAD_bounds={"rho": (1, 2)},
            nx=64,
            **scheme,
        )
    if scheme.get("use_MUSCL", False):
        sim.run(1.0, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    else:
        sim.run(1.0, time_integrator=TimeIntegrator.SSPRK3)

    assert min(sim.step_history.get_history("rho_min")) >= 1 - 1e-14