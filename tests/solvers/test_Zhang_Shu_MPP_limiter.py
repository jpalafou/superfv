from functools import partial

import numpy as np
import pytest

from superfv import AdvectionSolver, initial_conditions
from superfv.tools.device_management import CUPY_AVAILABLE

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
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7, 15])
@pytest.mark.parametrize("SED", [True])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_1D_advection_mpp(adaptive_dt, p, SED, dim):
    if p > 7 and not adaptive_dt:
        pytest.skip("High order schemes require adaptive timestepping.")
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, **{"v" + dim: 1.0}),
        **{"n" + dim: 64},
        CFL=0.8 if adaptive_dt else p_CFL_map[p],
        adaptive_dt=adaptive_dt and p > 0,
        ZS=True,
        PAD={"rho": (0, 1)},
        SED=SED,
        check_uniformity=False,
    )
    solver.run(n=10)

    violations = np.minimum(
        1 - np.array(solver.minisnapshots["rho_max"]),
        np.array(solver.minisnapshots["rho_min"]),
    )
    assert np.all(violations >= -1e-15)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 7, 15])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("y", "z"), ("x", "z")])
def test_2D_advection_mpp(p, SED, dim1_dim2):
    dim1, dim2 = dim1_dim2
    solver = AdvectionSolver(
        p=p,
        ic=partial(
            initial_conditions.square,
            **{"v" + dim1: 2.0, "v" + dim2: 1.0},
        ),
        **{"n" + dim1: 32, "n" + dim2: 32},
        GL=True,
        ZS=True,
        PAD={"rho": (0, 1)},
        SED=SED,
        check_uniformity=False,
        uniformity_tol=1e-15,
        cupy=CUPY_AVAILABLE,
    )
    solver.run(n=10)

    violations = np.minimum(
        1 - np.array(solver.minisnapshots["rho_max"]),
        np.array(solver.minisnapshots["rho_min"]),
    )
    assert np.all(violations >= -1e-15)


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_3D_advection_mpp(p):
    solver = AdvectionSolver(
        p=p,
        ic=partial(initial_conditions.square, vx=1.0, vy=1.0, vz=1.0),
        nx=16,
        ny=16,
        nz=16,
        GL=True,
        ZS=True,
        PAD={"rho": (0, 1)},
        check_uniformity=False,
        cupy=CUPY_AVAILABLE,
    )
    solver.run(n=3)

    violations = np.minimum(
        1 - np.array(solver.minisnapshots["rho_max"]),
        np.array(solver.minisnapshots["rho_min"]),
    )
    assert np.all(violations >= -1e-15)
