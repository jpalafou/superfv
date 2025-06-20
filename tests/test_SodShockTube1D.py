from typing import Tuple

import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv.euler_solver import EulerSolver
from superfv.tools.array_management import l1_norm


def linf_error(u1, u2):
    return np.max(np.abs(u1 - u2))


@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori"])
@pytest.mark.parametrize("dim1_dim2", [("x", "y"), ("y", "z")])
def test_sod_shock_tube_1d_symmetry(p: int, limiting: str, dim1_dim2: Tuple[str, str]):
    """
    Test that the Sod shock tube solution is symmetric in all dimensions.
    """
    dim1, dim2 = dim1_dim2
    N = 64

    # set up solvers
    limiting_config = (
        {"ZS": True, "PAD": {"rho": (0, None)}}
        if limiting == "a priori"
        else {"MOOD": True, "NAD": 1e-5}
    )
    solver1 = EulerSolver(
        ic=ic.sod_shock_tube_1d, **{f"n{dim1}": N}, p=p, **limiting_config
    )
    solver2 = EulerSolver(
        ic=ic.sod_shock_tube_1d, **{f"n{dim2}": N}, p=p, **limiting_config
    )

    # run solvers
    solver1.run(0.245)
    solver2.run(0.245)

    # compare solutions
    idx = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx("rho")].flatten()
        - solver2.snapshots[-1]["u"][idx("rho")].flatten()
    )
    assert l1_error == 0


@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori"])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_sod_shock_tube_passive_scalars(p: int, limiting: str, dim: str):
    """
    Test that passive scalars don't affect the Sod shock tube solution.
    """
    N = 64

    # set up solvers
    limiting_config = (
        {"ZS": True, "PAD": {"rho": (0, None)}}
        if limiting == "a priori"
        else {"MOOD": True, "NAD": 1e-5}
    )
    solver1 = EulerSolver(
        ic=ic.sod_shock_tube_1d, **{f"n{dim}": N}, p=p, **limiting_config
    )
    solver2 = EulerSolver(
        ic=ic.sod_shock_tube_1d,
        **{f"n{dim}": N},
        p=p,
        **limiting_config,
        ic_passives={
            "passive_square": lambda x, y, z: np.where(np.abs(x - 0.5) < 0.25, 1, -1)
        },
    )

    # run solvers
    solver1.run(0.245)
    solver2.run(0.245)

    # compare solutions
    idx = solver1.variable_index_map
    l1_error = l1_norm(
        solver1.snapshots[-1]["u"][idx("rho")].flatten()
        - solver2.snapshots[-1]["u"][idx("rho")].flatten()
    )
    assert l1_error == 0
