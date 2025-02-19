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


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7])
def test_1D_advection_mpp(p):
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, vx=1),
        nx=64,
        CFL=p_CFL_map[p],
        ZS=p > 0,
    )
    solver.run(1.0)

    upper_violations = 1 - np.array(solver.minisnapshots["max_rho"])
    lower_violations = np.array(solver.minisnapshots["min_rho"]) - 0

    assert np.all(upper_violations >= 0)
    assert np.all(lower_violations >= 0)


@pytest.mark.parametrize("p", [0, 1, 2])
def test_2D_advection_mpp(p):
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, vx=2, vy=1),
        nx=32,
        ny=32,
        CFL=p_CFL_map[p],
        interpolation_scheme="gauss-legendre",
        ZS=p > 0,
    )
    solver.run(1.0)

    upper_violations = 1 - np.array(solver.minisnapshots["max_rho"])
    lower_violations = np.array(solver.minisnapshots["min_rho"]) - 0

    assert np.all(upper_violations >= 0)
    assert np.all(lower_violations >= 0)
