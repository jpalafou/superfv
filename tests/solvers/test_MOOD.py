import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv.advection_solver import AdvectionSolver


@pytest.mark.parametrize("N", [32, 40, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize(
    "config",
    [
        dict(cascade="first-order", NAD_rtol=1e-2, scale_NAD_rtol_by_dt=True),
        dict(cascade="first-order", NAD=False, SED=False),
        dict(cascade="muscl1", NAD_gtol={"rho": 1e-1}),
        dict(cascade="full", NAD_atol=1e-8),
        dict(cascade="full", NAD=False, SED=False),
    ],
)
def test_mpp_1d(N: int, p: int, config: dict):
    n_steps = 10
    max_MOOD_iters = 20
    PAD_atol = 1e-14

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, xp=xp
        ),
        nx=N,
        p=p,
        MOOD=True,
        max_MOOD_iters=max_MOOD_iters,
        PAD={"rho": (0, 1)},
        PAD_atol=PAD_atol,
        **config,
    )
    sim.run(n=n_steps, q_max=2)

    assert np.min(sim.minisnapshots["rho_min"]) > -PAD_atol
    assert np.max(sim.minisnapshots["rho_max"]) < 1 + PAD_atol
    assert sim.minisnapshots["nfine_MOOD_iters"][-1][-1] <= max_MOOD_iters


@pytest.mark.parametrize("N", [32, 40, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize(
    "config",
    [
        dict(cascade="first-order", NAD_rtol={"rho": 1e-2}),
        dict(cascade="first-order", NAD=False, SED=False),
        dict(cascade="muscl1", NAD_rtol=1e-1, scale_NAD_rtol_by_dt=True),
        dict(cascade="muscl1", NAD=False, SED=False),
        dict(cascade="full"),
    ],
)
def test_mpp_2d(N: int, p: int, config: dict):
    n_steps = 10
    max_MOOD_iters = N**2 if not config.get("NAD", True) else 40
    tol = 1e-14

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, vy=1, xp=xp
        ),
        nx=N,
        ny=N,
        p=p,
        MOOD=True,
        max_MOOD_iters=max_MOOD_iters,
        NAD_atol=tol,
        NAD_gtol=tol,
        PAD={"rho": (0.0, 1.0)},
        PAD_atol=tol,
        **config,
    )
    sim.run(n=n_steps, q_max=2)

    assert np.min(sim.minisnapshots["rho_min"]) > -tol
    assert np.max(sim.minisnapshots["rho_max"]) < 1 + tol
    assert sim.minisnapshots["nfine_MOOD_iters"][-1][-1] <= max_MOOD_iters


@pytest.mark.parametrize("cascade", ["first-order", "muscl1", "full"])
def test_mpp_3d(cascade: str):
    N = 20
    p = 3

    n_steps = 5
    max_MOOD_iters = 40
    PAD_atol = 1e-14

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, vy=1, vz=1, xp=xp
        ),
        nx=N,
        ny=N,
        nz=N,
        p=p,
        MOOD=True,
        cascade=cascade,
        max_MOOD_iters=max_MOOD_iters,
        NAD=True,
        PAD={"rho": (0.0, 1.0)},
        PAD_atol=PAD_atol,
    )
    sim.run(n=n_steps, q_max=2)

    assert np.min(sim.minisnapshots["rho_min"]) > -PAD_atol
    assert np.max(sim.minisnapshots["rho_max"]) < 1 + PAD_atol
    assert sim.minisnapshots["nfine_MOOD_iters"][-1][-1] <= max_MOOD_iters
