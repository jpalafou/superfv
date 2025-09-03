import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv.advection_solver import AdvectionSolver


@pytest.mark.parametrize("N", [32, 40, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7])
def test_mpp_1d(N: int, p: int):
    n_steps = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, xp=xp
        ),
        nx=N,
        p=p,
        MOOD=True,
        cascade="first-order",
        max_MOOD_iters=3,
        PAD={"rho": (0, 1)},
        PAD_atol=1e-14,
    )
    sim.run(n=n_steps)

    assert np.min(sim.minisnapshots["min_rho"]) > -1e-14
    assert np.max(sim.minisnapshots["max_rho"]) < 1 + 1e-14
    assert sim.minisnapshots["nfine_MOOD_iters"][-1][-1] <= 3


@pytest.mark.parametrize("N", [32, 40, 64])
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 7])
def test_mpp_2d(N: int, p: int):
    n_steps = 20
    max_MOOD_iters = N * p

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, vy=1, xp=xp
        ),
        nx=N,
        ny=N,
        p=p,
        MOOD=True,
        cascade="first-order",
        max_MOOD_iters=max_MOOD_iters,
        PAD={"rho": (0.0, 1.0)},
        PAD_atol=1e-14,
    )
    sim.run(n=n_steps, q_max=2)

    assert np.min(sim.minisnapshots["min_rho"]) > -1e-14
    assert np.max(sim.minisnapshots["max_rho"]) < 1 + 1e-14
    assert sim.minisnapshots["nfine_MOOD_iters"][-1][-1] <= max_MOOD_iters


def test_mpp_3d():
    N = 20
    p = 3
    n_steps = 5

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: ic.square(
            array_slicer, x, y, z, vx=1, vy=1, vz=1, xp=xp
        ),
        nx=N,
        ny=N,
        nz=N,
        p=p,
        MOOD=True,
        cascade="first-order",
        max_MOOD_iters=N,
        PAD={"rho": (0.0, 1.0)},
        PAD_atol=1e-14,
    )
    sim.run(n=n_steps, q_max=2)
