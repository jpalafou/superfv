from functools import partial

import numpy as np
import pytest

from superfv import AdvectionSolver, initial_conditions

p_CFL_map = {
    0: 1 / 2,
    1: 1 / 2,
    2: 1 / 6,
    3: 1 / 6,
    4: 1 / 12,
    5: 1 / 12,
    6: 1 / 20,
    7: 1 / 20,
}


@pytest.mark.parametrize("adaptive_dt", [False, True])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7])
def test_1D_advection_mpp(adaptive_dt, p):
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, vx=1),
        nx=64,
        CFL=0.8 if adaptive_dt else p_CFL_map[p],
        adaptive_timestepping=adaptive_dt,
        ZS=p > 0,
        PAD={"rho": (0, 1)},
    )
    solver.run(1.0)

    upper_violations = 1 - np.array(solver.minisnapshots["max_rho"])
    lower_violations = np.array(solver.minisnapshots["min_rho"]) - 0

    assert np.all(upper_violations >= 0)
    assert np.all(lower_violations >= 0)


@pytest.mark.parametrize("adaptive_dt", [False, True])
@pytest.mark.parametrize("p", [0, 1, 2])
def test_2D_advection_mpp(adaptive_dt, p):
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, vx=2, vy=1),
        nx=32,
        ny=32,
        CFL=0.8 if adaptive_dt else p_CFL_map[p],
        adaptive_timestepping=adaptive_dt,
        interpolation_scheme="gauss-legendre",
        ZS=p > 0,
        PAD={"rho": (0, 1)},
    )
    solver.run(1.0)

    upper_violations = 1 - np.array(solver.minisnapshots["max_rho"])
    lower_violations = np.array(solver.minisnapshots["min_rho"]) - 0

    assert np.all(upper_violations >= 0)
    assert np.all(lower_violations >= 0)
