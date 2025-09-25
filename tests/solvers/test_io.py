import os
import pickle
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from superfv import AdvectionSolver, EulerSolver, OutputLoader
from superfv.initial_conditions import sinus, sod_shock_tube_1d, square
from superfv.tools.snapshots import SnapshotPlaceholder

TEST_PATH = Path("tests/data")


def test_interrupted_simulation():
    """
    Test that a simulation can be stopped, pickled, unpickled, and continued.
    """
    N = 64
    p = 0

    # ensure test path exists
    TEST_PATH.mkdir(parents=True, exist_ok=True)

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
    )

    # run simulations
    sim_full.run(n=10)
    sim_part1.run(n=5)

    # save the first part of the simulation
    with open(TEST_PATH / "sim.pkl", "wb") as f:
        pickle.dump(sim_part1, f)

    # load the first part of the simulation
    with open(TEST_PATH / "sim.pkl", "rb") as f:
        sim_part2 = pickle.load(f)

    # continue the simulation
    sim_part2.run(n=5)

    assert np.array_equal(sim_full.arrays["u"], sim_part2.arrays["u"])

    # remove the saved file
    os.remove(TEST_PATH / "sim.pkl")


@pytest.mark.parametrize("snapshot_mode", ["target", "every"])
@pytest.mark.parametrize("fixed_n_steps", [True, False])
def test_discarding_snapshots(
    snapshot_mode: Literal["target", "every"], fixed_n_steps: bool
):
    if (TEST_PATH / "out").exists():
        shutil.rmtree(TEST_PATH / "out")

    sim1 = EulerSolver(ic=sod_shock_tube_1d, bcx="free", nx=100, p=0)
    sim2 = EulerSolver(ic=sod_shock_tube_1d, bcx="free", nx=100, p=0)

    if fixed_n_steps:
        sim1.run(n=20, snapshot_mode=snapshot_mode)
        sim2.run(n=20, snapshot_mode=snapshot_mode, path=TEST_PATH / "out")
    else:
        sim1.run([0.1, 0.2], snapshot_mode=snapshot_mode)
        sim2.run([0.1, 0.2], snapshot_mode=snapshot_mode, path=TEST_PATH / "out")

    assert all(isinstance(s, dict) for s in sim1.snapshots.data.values())
    assert all(isinstance(s, SnapshotPlaceholder) for s in sim2.snapshots.data.values())

    if fixed_n_steps:
        assert np.array_equal(sim1.snapshots[0]["u"], sim2.snapshots[0]["u"])
        assert np.array_equal(sim1.snapshots[-1]["u"], sim2.snapshots[-1]["u"])
    else:
        assert np.array_equal(sim1.snapshots(0)["u"], sim2.snapshots(0)["u"])
        assert np.array_equal(sim1.snapshots(0.1)["u"], sim2.snapshots(0.1)["u"])
        assert np.array_equal(sim1.snapshots(0.2)["u"], sim2.snapshots(0.2)["u"])

    shutil.rmtree(TEST_PATH / "out")


def test_OutputLoader():
    N = 64
    p = 3

    # init solver
    sim = EulerSolver(
        ic=lambda idx, x, y, z, t, xp: sinus(
            idx, x, y, z, bounds=(1, 2), P=1, vx=1, xp=xp
        ),
        nx=N,
        p=p,
    )

    # run simulation, writing snapshots to disk
    sim.run(n=10, path=TEST_PATH / "out", overwrite=True)

    # load snapshots from disk
    loader = OutputLoader(TEST_PATH / "out")

    # check that the loaded snapshots match the simulation data
    assert np.array_equal(sim.snapshots[-1]["u"], loader.snapshots[-1]["u"])

    # remove the saved output directory
    shutil.rmtree(TEST_PATH / "out")
