import os
import pickle

import numpy as np

from superfv import AdvectionSolver
from superfv.initial_conditions import square


def test_interrupted_simulation():
    """
    Test that a simulation can be interrupted and continued later.
    """
    N = 64
    p = 0

    # init solvers
    sim_full = AdvectionSolver(
        ic=lambda idx, x, y, z, t, xp: square(idx, x, y, z, vx=2, vy=1, xp=xp),
        ic_passives={
            "passive1": lambda x, y, z, t, xp: xp.where(
                xp.abs(x - 0.5) < 0.25, 1.0, 0.0
            )
        },
        nx=N,
        ny=N,
        p=p,
        log_every_step=False,
    )
    sim_part1 = AdvectionSolver(
        ic=lambda idx, x, y, z, t, xp: square(idx, x, y, z, vx=2, vy=1, xp=xp),
        ic_passives={
            "passive1": lambda x, y, z, t, xp: xp.where(
                xp.abs(x - 0.5) < 0.25, 1.0, 0.0
            )
        },
        nx=N,
        ny=N,
        p=p,
        log_every_step=False,
    )

    # run simulations
    sim_full.run(n=10)
    sim_part1.run(n=5)

    # save the first part of the simulation
    with open("tests/solvers/sim.pkl", "wb") as f:
        pickle.dump(sim_part1, f)

    # load the first part of the simulation
    with open("tests/solvers/sim.pkl", "rb") as f:
        sim_part2 = pickle.load(f)

    # continue the simulation
    sim_part2.run(n=5)

    assert np.array_equal(sim_full.arrays["u"], sim_part2.arrays["u"])

    # remove the saved file
    os.remove("tests/solvers/sim.pkl")
