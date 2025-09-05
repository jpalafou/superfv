import numpy as np
import pytest

import superfv.initial_conditions as ic
from superfv import AdvectionSolver, EulerSolver
from superfv.tools.norms import l1_norm


def test_AdvectionSolver_passive_scalar_invariance():
    """
    Test that passive scalars don't change the advection solution.
    """
    N = 64
    p = 3
    n_steps = 10

    # set up solvers
    solver1 = AdvectionSolver(
        ic=lambda idx, x, y, z, t, xp: ic.sinus(idx, x, y, z, vx=1, xp=np),
        nx=N,
        p=p,
    )
    solver2 = AdvectionSolver(
        ic=lambda idx, x, y, z, t, xp: ic.sinus(idx, x, y, z, vx=1, xp=np),
        ic_passives={
            "passive1": lambda x, y, z, t, xp: xp.where(xp.abs(x - 0.5) < 0.25, 1, 0)
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


@pytest.mark.parametrize("p", [0, 3, 7])
@pytest.mark.parametrize("limiting", ["a priori", "a posteriori"])
@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_Sod_shock_tube_passive_scalar_invariance(p: int, limiting: str, dim: str):
    """
    Test that passive scalars don't change the Sod shock tube solution.
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
            "passive_square": lambda x, y, z, t, xp: xp.where(
                xp.abs(x - 0.5) < 0.25, 1, -1
            )
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
