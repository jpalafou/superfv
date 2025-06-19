from typing import Tuple

import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv import AdvectionSolver
from superfv.tools.array_management import l1_norm


@pytest.mark.parametrize("p", [0, 1, 3, 7])
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("y", "z")])
def test_AdvectionSolver_translational_symmetry_1D(p: int, dim1_dim2: Tuple[str, str]):
    """
    Test the symmetry of the solver along each dimension in 1D.
    """
    dim1, dim2 = dim1_dim2
    N = 64
    n_steps = 10

    # set up solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, **{f"v{dim1}": 1}),
        p=p,
        **{f"n{dim1}": N},
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, **{f"v{dim2}": 1}),
        p=p,
        **{f"n{dim2}": N},
    )

    # run solvers
    solver1.run(n=n_steps)
    solver2.run(n=n_steps)

    # compare solutions
    idx1 = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx1("rho")].flatten()
        - solver2.snapshots[-1]["u"][idx1("rho")].flatten()
    )
    assert l1_error == 0


@pytest.mark.parametrize("p", [0, 1, 3, 7])
@pytest.mark.parametrize("interpolation_scheme", ["transverse", "gauss-legendre"])
@pytest.mark.parametrize("dims1_dims2", [("xy", "yz"), ("yz", "xz")])
def test_AdvectionSolver_translational_symmetry_2D(
    p: int, interpolation_scheme: str, dims1_dims2: Tuple[str, str]
):
    """
    Test the symmetry of the solver along each plane in 2D.
    """
    (d1x, d1y), (d2x, d2y) = dims1_dims2
    N = 64
    n_steps = 10

    # set up solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, **{f"v{d1x}": 1, f"v{d1y}": 1}),
        p=p,
        interpolation_scheme=interpolation_scheme,
        **{f"n{d1x}": N, f"n{d1y}": N},
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, **{f"v{d2x}": 1, f"v{d2y}": 1}),
        p=p,
        interpolation_scheme=interpolation_scheme,
        **{f"n{d2x}": N, f"n{d2y}": N},
    )

    # run solvers
    solver1.run(n=n_steps)
    solver2.run(n=n_steps)

    # compare solutions
    idx1 = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx1("rho")].flatten()
        - solver2.snapshots[-1]["u"][idx1("rho")].flatten()
    )
    assert l1_error == 0


@pytest.mark.parametrize("p", [0, 1, 2, 3, 7, 15])
@pytest.mark.parametrize("interpolation_scheme", ["transverse", "gauss-legendre"])
def test_AdvectionSolver_rotational_symmetry_xy(p: int, interpolation_scheme: str):
    """
    Assert that the result of a counter-clockwise rotation of a slotted disk is the
    same as the mirror of the result of a clockwise rotation.
    """
    N = 64
    n_steps = 10

    # initialize solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.slotted_disk(idx, x, y, z),
        bcx="ic",
        bcy="ic",
        p=p,
        nx=N,
        ny=N,
        interpolation_scheme=interpolation_scheme,
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.slotted_disk(idx, x, y, z, rotation="cw"),
        bcx="ic",
        bcy="ic",
        p=p,
        nx=N,
        ny=N,
        interpolation_scheme=interpolation_scheme,
    )

    # run solvers
    solver1.run(n=n_steps)
    solver2.run(n=n_steps)

    # compare solutions
    idx = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx("rho")]
        - np.flipud(solver2.snapshots[-1]["u"][idx("rho")]),
    )
    assert l1_error < 1e-15


@pytest.mark.parametrize("interpolation_scheme", ["transverse", "gauss-legendre"])
def test_AdvectionSolver_translational_symmetry_3D(interpolation_scheme: str):
    """
    Assert that the solution is invariant under translation in 3D.
    """
    N = 32
    p = 3
    n_steps = 1

    # set up solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, vx=1, vy=1, vz=1),
        p=p,
        interpolation_scheme=interpolation_scheme,
        nx=N,
        ny=N,
        nz=N,
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.square(idx, x, y, z, vx=-1, vy=-1, vz=-1),
        p=p,
        interpolation_scheme=interpolation_scheme,
        nx=N,
        ny=N,
        nz=N,
    )

    # run solvers
    solver1.run(n=n_steps)
    solver2.run(n=n_steps)

    # compare solutions
    idx = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx("rho")]
        - np.flip(solver2.snapshots[-1]["u"][idx("rho")], axis=(0, 1, 2)),
    )
    assert l1_error < 1e-15


def test_AdvectionSolver_passive_scalar_invariance():
    """
    Test the invariance of the solution when adding a passive scalar.
    """
    N = 64
    p = 3
    n_steps = 10

    # set up solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.sinus(idx, x, y, z, vx=1),
        nx=N,
        p=p,
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.sinus(idx, x, y, z, vx=1),
        ic_passives={
            "passive1": lambda x, y, z: np.where(np.abs(x - 0.5) < 0.25, 1, 0)
        },
        nx=N,
        p=p,
    )

    # run solvers
    solver1.run(n=n_steps)
    solver2.run(n=n_steps)

    # compare solutions
    idx = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx("rho")] - solver2.snapshots[-1]["u"][idx("rho")]
    )
    assert l1_error == 0
