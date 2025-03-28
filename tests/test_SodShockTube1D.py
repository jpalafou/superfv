import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv.euler_solver import EulerSolver


def linf_error(u1, u2):
    return np.max(np.abs(u1 - u2))


@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori"])
def test_sod_shock_tube_1d_symmetry(p, limiting):
    """
    Test that the Sod shock tube solution is symmetric in all dimensions.
    """
    solutions = {}
    for dim in "xyz":
        solver = EulerSolver(
            ic=ic.sod_shock_tube_1d,
            p=p,
            **(
                {
                    "a priori": {"ZS": True, "PAD": {"rho": (0, np.inf)}},
                    "a posteriori": {"MOOD": True, "NAD": 1e-5},
                }[limiting]
                if p > 0
                else {}
            ),
            **{"bc" + dim: "free", "n" + dim: 64},
        )
        solver.run(0.245)
        _slc = solver.array_slicer
        solutions[dim] = solver.snapshots[-1]["w"][_slc("P")].flatten().copy()

    assert np.all(solutions["x"] == solutions["y"])
    assert np.all(solutions["x"] == solutions["z"])


@pytest.mark.parametrize("dim", "xyz")
@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori"])
def test_sod_shock_tube_passive_scalars(dim, p, limiting):
    """
    Test that passive scalars don't affect the Sod shock tube solution.
    """
    config = {
        "ic": ic.sod_shock_tube_1d,
        "bc" + dim: "free",
        "n" + dim: 64,
        "p": p,
        **(
            {
                "a priori": {"ZS": True, "PAD": {"rho": (0, np.inf)}},
                "a posteriori": {"MOOD": True, "NAD": 1e-5},
            }[limiting]
            if p > 0
            else {}
        ),
    }

    # run solver with no passive scalars
    solver = EulerSolver(**config)
    solver.run(0.245)

    # run solver with passive scalars
    solver_with_passives = EulerSolver(
        **config,
        ic_passives={
            "passive_square": lambda x, y, z: np.where(np.abs(x - 0.5) < 0.25, 1, -1)
        },
    )
    solver_with_passives.run(0.245)

    # compare the solutions
    _slc = solver_with_passives.array_slicer
    assert np.all(
        solver.snapshots[-1]["u"][_slc("actives")]
        == solver_with_passives.snapshots[-1]["u"][_slc("actives")]
    )
