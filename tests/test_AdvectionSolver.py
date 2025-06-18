from itertools import combinations

import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv import AdvectionSolver
from superfv.tools.array_management import l1_norm


@pytest.mark.parametrize("p", [0, 1, 3, 7])
def test_AdvectionSolver_symmetry_1D(p: int):
    """
    Test the symmetry of the solution in 1D.
    """
    N = 64
    n_steps = 10

    # run solver in each direction
    solution = {}
    for dim in "xyz":
        solver = AdvectionSolver(
            ic=lambda idx, x, y, z: ic.square(idx, x, y, z, **{f"v{dim}": 1}),
            p=p,
            **{f"n{dim}": N},
        )
        solver.run(n=n_steps)

        idx = solver.variable_index_map
        solution[dim] = solver.snapshots[-1]["u"][idx("rho")].flatten().copy()

    # check that the solutions are equal
    assert np.array_equal(solution["x"], solution["y"])
    assert np.array_equal(solution["x"], solution["z"])


@pytest.mark.parametrize("p", [0, 1, 3, 7])
@pytest.mark.parametrize("interpolation_scheme", ["transverse", "gauss-legendre"])
def test_AdvectionSolver_symmetry_2D(p: int, interpolation_scheme: str):
    """
    Test the symmetry of the solution in 2D.
    """
    N = 64
    n_steps = 10

    # run solver along each plane
    solution = {}
    for dims in combinations("xyz", 2):
        dim1, dim2 = dims
        solver = AdvectionSolver(
            ic=lambda idx, x, y, z: ic.square(
                idx, x, y, z, **{f"v{dim1}": 1, f"v{dim2}": 1}
            ),
            p=p,
            interpolation_scheme=interpolation_scheme,
            **{f"n{dim1}": N, f"n{dim2}": N},
        )
        solver.run(n=n_steps)

        # get the resulting 2D solution
        idx = solver.variable_index_map
        _slices2d = [slice(None), slice(None), slice(None)]
        _slices2d[{"xy": 2, "xz": 1, "yz": 0}[dim1 + dim2]] = 0
        solution[dim1 + dim2] = solver.snapshots[-1]["u"][
            idx("rho"), _slices2d[0], _slices2d[1], _slices2d[2]
        ]

    # check that the solutions are equal
    assert np.array_equal(solution["xy"], solution["yz"])
    assert np.array_equal(solution["xy"], solution["xz"])


@pytest.mark.parametrize("interpolation_scheme", ["transverse", "gauss-legendre"])
def test_AdvectionSolver_rotational_symmetry_xy(interpolation_scheme: str):
    """
    Assert that the result of a counter-clockwise rotation of a slotted disk is the
    same as the mirror of the result of a clockwise rotation.
    """
    N = 64
    p = 3
    n_steps = 10

    # initialize solvers
    ccw_solver = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.slotted_disk(idx, x, y, z),
        bcx="ic",
        bcy="ic",
        p=p,
        nx=N,
        ny=N,
        interpolation_scheme=interpolation_scheme,
    )
    cw_solver = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.slotted_disk(idx, x, y, z, rotation="cw"),
        bcx="ic",
        bcy="ic",
        p=p,
        nx=N,
        ny=N,
        interpolation_scheme=interpolation_scheme,
    )

    # run solvers
    ccw_solver.run(n=n_steps)
    cw_solver.run(n=n_steps)

    # compare solutions
    idx = ccw_solver.variable_index_map
    assert (
        l1_norm(
            cw_solver.snapshots[-1]["u"][idx("rho")]
            - np.flipud(ccw_solver.snapshots[-1]["u"][idx("rho")]),
        )
        < 1e-15
    )


def test_AdvectionSolver_passive_scalar_invariance():
    """
    Test the invariance of the solution when adding a passive scalar.
    """
    N = 64
    p = 3
    n_steps = 10

    # set up solvers
    solver = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.sinus(idx, x, y, z, vx=1),
        nx=N,
        p=p,
    )
    solver_with_passive_scalar = AdvectionSolver(
        ic=lambda idx, x, y, z: ic.sinus(idx, x, y, z, vx=1),
        ic_passives={
            "passive1": lambda x, y, z: np.where(np.abs(x - 0.5) < 0.25, 1, 0)
        },
        nx=N,
        p=p,
    )

    # run solvers
    solver.run(n=n_steps)
    solver_with_passive_scalar.run(n=n_steps)

    # compare solutions
    idx = solver.variable_index_map
    assert np.array_equal(
        solver.snapshots[-1]["u"][idx("rho")],
        solver_with_passive_scalar.snapshots[-1]["u"][idx("rho")],
    )
