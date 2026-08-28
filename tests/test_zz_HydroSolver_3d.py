from functools import partial

import pytest

from superfv import CUPY_AVAILABLE, HydroSolver, ics


@pytest.mark.parametrize(
    "scheme",
    [
        dict(p=0),
        dict(p=1, use_MUSCL=True),
        dict(
            p=3,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode="full",
            flux_quadrature="gauss_legendre",
        ),
        dict(
            p=7,
            use_ZS=True,
            adaptive_dt=True,
            lazy_primitive_mode="full",
            flux_quadrature="gauss_legendre",
        ),
        dict(p=7, use_MOOD=True, fallback_cascade="muscl0", max_revs=3),
        dict(p=7, use_MOOD=True, fallback_cascade="full", max_revs=7),
    ],
)
def test_sedov(scheme):
    if not CUPY_AVAILABLE:
        pytest.skip("Cupy is not available, skipping test.")
    sim = HydroSolver(
        ic=partial(ics.sedov, h=1 / 64, gamma=1.4, P0=1e-5),
        bcx=("reflective", "free"),
        bcy=("reflective", "free"),
        bcz=("reflective", "free"),
        gamma=1.4,
        nx=64,
        ny=64,
        nz=64,
        cupy=True,
        **scheme,
    )
    if scheme.get("use_MUSCL", False):
        sim.take_n_steps(10, time_integrator="muscl_hancock", print_frequency=1)
    else:
        sim.take_n_steps(10, time_integrator="ssprk3", print_frequency=1)
