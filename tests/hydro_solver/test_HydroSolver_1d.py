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
