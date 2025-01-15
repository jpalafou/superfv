from functools import partial
from itertools import combinations

import numpy as np
import pytest

from shine import AdvectionSolver, initial_conditions


@pytest.mark.parametrize("p", [0, 1, 3, 7])
def test_AdvectionSolver_symmetry_1D(p: int):
    """
    Test the symmetry of the solution in 1D.
    """
    N = 128
    T = 1.0

    # run solver in each direction
    solution = {}
    for dim in "xyz":
        solver = AdvectionSolver(
            ic=partial(initial_conditions.square, **{f"v{dim}": 1}),
            p=p,
            **{f"n{dim}": N},
        )
        solver.run(T)
        _slc = solver.array_slicer
        solution[dim] = solver.snapshots[T]["u"][_slc("rho")].flatten()

    # check that the solutions are equal
    assert np.array_equal(solution["x"], solution["y"])
    assert np.array_equal(solution["x"], solution["z"])


@pytest.mark.parametrize("p", [0, 1, 3, 7])
def test_AdvectionSolver_symmetry_2D(p: int):
    """
    Test the symmetry of the solution in 2D.
    """
    N = 64
    T = 1.0

    # run solver along each plane
    solution = {}
    for dims in combinations("xyz", 2):
        dim1, dim2 = dims
        solver = AdvectionSolver(
            ic=partial(initial_conditions.square, **{f"v{dim1}": 1, f"v{dim2}": 1}),
            p=p,
            **{f"n{dim1}": N, f"n{dim2}": N},
        )
        solver.run(T)

        # get the resulting 2D solution
        _slc = solver.array_slicer
        _slices2d = [slice(None), slice(None), slice(None)]
        _slices2d[{"xy": 2, "xz": 1, "yz": 0}[dim1 + dim2]] = 0
        solution[dim1 + dim2] = solver.snapshots[T]["u"][
            _slc("rho"), _slices2d[0], _slices2d[1], _slices2d[2]
        ]

    # check that the solutions are equal
    print(solution["xy"].shape, solution["yz"].shape, solution["xz"].shape)
    assert np.array_equal(solution["xy"], solution["yz"])
    assert np.array_equal(solution["xy"], solution["xz"])


def test_AdvectionSolver_passive_scalar_invariance():
    """
    Test the invariance of the solution when adding a passive scalar.
    """
    N = 128
    p = 3
    T = 1.0

    # set up solvers
    solver = AdvectionSolver(
        ic=partial(initial_conditions.sinus, vx=1),
        nx=N,
        p=p,
    )
    solver_with_passive_scalar = AdvectionSolver(
        ic=partial(initial_conditions.sinus, vx=1),
        ic_passives={
            "passive1": lambda x, y, z: np.where(np.abs(x - 0.5) < 0.25, 1, 0)
        },
        nx=N,
        p=p,
    )

    # run solvers
    solver.run(T)
    solver_with_passive_scalar.run(T)

    # compare solutions
    _slc = solver.array_slicer
    assert np.array_equal(
        solver.snapshots[T]["u"][_slc("rho")],
        solver_with_passive_scalar.snapshots[T]["u"][_slc("rho")],
    )
