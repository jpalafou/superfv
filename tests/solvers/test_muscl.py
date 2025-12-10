import numpy as np
import pytest

from superfv import AdvectionSolver, EulerSolver
from superfv.initial_conditions import square
from superfv.tools.norms import l1_norm


@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_null_limiter_1d(dim: str):
    N = 64
    n = 10

    # initialize simulations
    sim1 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim: 1}
        ),
        p=1,
        **{"n" + dim: N},
    )
    sim2 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim: 1}
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=None,
        **{"n" + dim: N},
    )

    # run simulations
    sim1.ssprk2(n=n)
    sim2.ssprk2(n=n)

    # compare results
    err = l1_norm(sim1.snapshots[-1]["u"] - sim2.snapshots[-1]["u"])
    assert err < 1e-15


@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("x", "z"), ("y", "z")])
def test_null_limiter_2d(dim1_dim2: tuple):
    dim1, dim2 = dim1_dim2

    N = 32
    n = 10

    # initialize simulations
    sim1 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim1: 2, "v" + dim2: 1}
        ),
        p=1,
        **{"n" + dim1: N, "n" + dim2: N},
    )
    sim2 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim1: 2, "v" + dim2: 1}
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=None,
        **{"n" + dim1: N, "n" + dim2: N},
    )

    # run simulations
    sim1.ssprk2(n=n)
    sim2.ssprk2(n=n)

    # compare results
    err = l1_norm(sim1.snapshots[-1]["u"] - sim2.snapshots[-1]["u"])
    assert err < 1e-5


def test_null_limiter_3d():
    N = 16
    n = 10

    # initialize simulations
    sim1 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer,
            x,
            y,
            z,
            xp=xp,
            vx=1,
            vy=1,
            vz=1,
        ),
        p=1,
        nx=N,
        ny=N,
        nz=N,
    )
    sim2 = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer,
            x,
            y,
            z,
            xp=xp,
            vx=1,
            vy=1,
            vz=1,
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=None,
        nx=N,
        ny=N,
        nz=N,
    )

    # run simulations
    sim1.ssprk2(n=n)
    sim2.ssprk2(n=n)

    # compare results
    err = l1_norm(sim1.snapshots[-1]["u"] - sim2.snapshots[-1]["u"])
    assert err < 1e-5


@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_advection_of_a_1d_square(limiter: str, dim: str, predictor_corrector: bool):
    N = 64
    n = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim: 1}
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=limiter,
        SED=False,
        **{"n" + dim: N},
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= 0
    assert np.min(sim.minisnapshots["rho_max"]) <= 1


@pytest.mark.parametrize("limiter", ["minmod", "PP2D"])
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("x", "z"), ("y", "z")])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_advection_of_a_2d_square(
    limiter: str, dim1_dim2: tuple, predictor_corrector: bool
):
    if limiter == "PP2D" and not predictor_corrector:
        pytest.skip("PP2D only works with predictor-corrector")

    dim1, dim2 = dim1_dim2

    N = 32
    n = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim1: 2, "v" + dim2: 1}
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=limiter,
        SED=False,
        **{"n" + dim1: N, "n" + dim2: N},
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= -1e-15
    assert np.min(sim.minisnapshots["rho_max"]) <= 1 + 1e-15


@pytest.mark.parametrize("limiter", ["minmod"])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_advection_of_a_3d_square(limiter: str, predictor_corrector: bool):
    N = 32
    n = 10

    sim = AdvectionSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, vx=1, vy=1, vz=1
        ),
        p=1,
        MUSCL=True,
        MUSCL_limiter=limiter,
        SED=False,
        nx=N,
        ny=N,
        nz=N,
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= -1e-15
    assert np.min(sim.minisnapshots["rho_max"]) <= 1 + 1e-15


@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_hydro_advection_of_a_1d_square(
    limiter: str, dim: str, predictor_corrector: bool
):
    N = 64
    n = 10

    sim = EulerSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, **{"v" + dim: 1}, bounds=(1e-5, 1.0), P=1e-5
        ),
        p=1,
        MUSCL=True,
        SED=False,
        flux_recipe=2,
        riemann_solver="hllc",
        MUSCL_limiter=limiter,
        **{"n" + dim: N},
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= 1e-5 + (-1e-15)


@pytest.mark.parametrize("limiter", ["minmod", "PP2D"])
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("x", "z"), ("y", "z")])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_hydro_advection_of_a_2d_square(
    limiter: str, dim1_dim2: tuple, predictor_corrector: bool
):
    if limiter == "PP2D" and not predictor_corrector:
        pytest.skip("PP2D only works with predictor-corrector")

    dim1, dim2 = dim1_dim2

    N = 32
    n = 10

    sim = EulerSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer,
            x,
            y,
            z,
            xp=xp,
            **{"v" + dim1: 2, "v" + dim2: 1},
            bounds=(1e-5, 1.0),
            P=1e-5,
        ),
        p=1,
        MUSCL=True,
        SED=False,
        flux_recipe=2,
        riemann_solver="hllc",
        MUSCL_limiter=limiter,
        **{"n" + dim1: N, "n" + dim2: N},
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= 1e-5 + (-1e-15)


@pytest.mark.parametrize("limiter", ["minmod"])
@pytest.mark.parametrize("predictor_corrector", [False, True])
def test_hydro_advection_of_a_3d_square(limiter: str, predictor_corrector: bool):
    N = 32
    n = 10

    sim = EulerSolver(
        ic=lambda array_slicer, x, y, z, t, xp: square(
            array_slicer, x, y, z, xp=xp, vx=1, vy=1, vz=1, bounds=(1e-5, 1.0), P=1e-5
        ),
        p=1,
        MUSCL=True,
        SED=False,
        flux_recipe=2,
        riemann_solver="hllc",
        MUSCL_limiter=limiter,
        nx=N,
        ny=N,
        nz=N,
    )
    sim.run(n=n, muscl_hancock=predictor_corrector)

    assert np.min(sim.minisnapshots["rho_min"]) >= 1e-5 + (-1e-15)
