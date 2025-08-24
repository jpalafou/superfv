import numpy as np
import pytest

from superfv.advection_solver import AdvectionSolver
from superfv.initial_conditions import square


@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_advection_of_a_1d_square(dim: str, predictor_corrector: bool):
    N = 64
    n = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim: 1}
        ),
        p=1,
        MUSCL=True,
        **{"n" + dim: N},
    )
    if predictor_corrector:
        sim.musclhancock(n=n)
    else:
        sim.ssprk2(n=n)

    assert np.min(sim.minisnapshots["min_rho"]) >= 0
    assert np.min(sim.minisnapshots["max_rho"]) <= 1


@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("x", "z"), ("y", "z")])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_advection_of_a_2d_square(dim1_dim2: tuple, predictor_corrector: bool):
    dim1, dim2 = dim1_dim2

    N = 32
    n = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim1: 2, "v" + dim2: 1}
        ),
        p=1,
        MUSCL=True,
        **{"n" + dim1: N, "n" + dim2: N},
    )
    if predictor_corrector:
        sim.musclhancock(n=n)
    else:
        sim.ssprk2(n=n)

    assert np.min(sim.minisnapshots["min_rho"]) >= 0
    assert np.min(sim.minisnapshots["max_rho"]) <= 1
