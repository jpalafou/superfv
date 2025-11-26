
from functools import partial

import matplotlib.pyplot as plt

from superfv import EulerSolver, plot_1d_slice, plot_timeseries
from superfv.initial_conditions import sinus, square
from superfv.tools.norms import linf_norm

from superfv.tools.device_management import CUPY_AVAILABLE

import pytest

@pytest.mark.parametrize("f", [square, sinus])
@pytest.mark.parametrize(
    "config",
    [
        dict(p=0),
        dict(p=1, MUSCL=True, MUSCL_limiter=None),
        dict(p=1, MUSCL=True, MUSCL_limiter="minmod", SED=False),
        dict(p=1, MUSCL=True, MUSCL_limiter="minmod", SED=True),
        dict(p=1, MUSCL=True, MUSCL_limiter="moncen", SED=False),
        dict(p=1, MUSCL=True, MUSCL_limiter="moncen", SED=True),
        dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", SED=False),
        dict(p=1, MUSCL=True, MUSCL_limiter="PP2D", SED=True),
        dict(p=3, ZS=True, adaptive_dt=False, SED=False),
        dict(p=3, ZS=True, adaptive_dt=False, SED=True),
    ],
)
def test_hydro_advection(f: callable, config: dict):
    if not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available.")

    ic = partial(sinus, bounds=(1, 2), vx=2, vy=1, P=1)
    ic_passives = dict(passive1=lambda x, y, z, t, *, xp: xp.sin(2 * xp.pi * x))
    N = 128
    n_steps = 10

    sim1 = EulerSolver(ic=ic, ic_passives=ic_passives, nx=N, ny=N, **config)
    sim2 = EulerSolver(ic=ic, ic_passives=ic_passives, nx=N, ny=N, cupy=True, **config)

    sim1.run(n=n_steps, muscl_hancock=config.get("MUSCL", False))
    sim2.run(n=n_steps, muscl_hancock=config.get("MUSCL", False))

    tol = 1e-13 if config.get("ZS", False) else 1e-14
    assert linf_norm(sim2.snapshots[-1]["u"] - sim1.snapshots[-1]["u"]) < tol