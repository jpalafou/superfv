import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import sod_shock_tube_1d
from superfv.tools.snapshots import SnapshotPlaceholder

TEST_PATH = Path("tests/data")


@pytest.mark.parametrize("snapshot_mode", ["target", "every"])
@pytest.mark.parametrize("fixed_n_steps", [True, False])
def test_discarding_snapshots(
    snapshot_mode: Literal["target", "every"], fixed_n_steps: bool
):
    sim1 = EulerSolver(ic=sod_shock_tube_1d, bcx="free", nx=100, p=0)
    sim2 = EulerSolver(ic=sod_shock_tube_1d, bcx="free", nx=100, p=0)

    if fixed_n_steps:
        sim1.run(n=20, snapshot_mode=snapshot_mode)
        sim2.run(
            n=20, snapshot_mode=snapshot_mode, path=TEST_PATH / "out", overwrite=True
        )
    else:
        sim1.run([0.1, 0.2], snapshot_mode=snapshot_mode)
        sim2.run(
            [0.1, 0.2],
            snapshot_mode=snapshot_mode,
            path=TEST_PATH / "out",
            overwrite=True,
        )

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


@pytest.mark.parametrize("snapshot_mode", ["target", "every"])
@pytest.mark.parametrize("fixed_n_steps", [True, False])
def test_OutputLoader(snapshot_mode: Literal["target", "every"], fixed_n_steps: bool):
    sim = EulerSolver(
        sod_shock_tube_1d, bcx="free", nx=100, p=1, ZS=True, adaptive_dt=False
    )

    if fixed_n_steps:
        sim.run(
            n=20, snapshot_mode=snapshot_mode, path=TEST_PATH / "out", overwrite=True
        )
    else:
        sim.run(
            [0.1, 0.2],
            snapshot_mode=snapshot_mode,
            path=TEST_PATH / "out",
            overwrite=True,
        )
    sim.variable_index_map.clear_cache()

    loader = OutputLoader(TEST_PATH / "out")

    assert sim.active_dims == loader.active_dims
    assert sim.variable_index_map == loader.variable_index_map
    assert np.array_equal(sim.mesh.X, loader.mesh.X)
    assert np.array_equal(sim.mesh.Y, loader.mesh.Y)
    assert np.array_equal(sim.mesh.Z, loader.mesh.Z)

    def equal_lists(a, b):
        if a is b:
            return True
        if type(a) is not type(b):
            return False
        if isinstance(a, list):
            return len(a) == len(b) and all(equal_lists(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return np.isnan(a) and np.isnan(b) or a == b
        return a == b

    for key in sim.minisnapshots.keys():
        assert equal_lists(sim.minisnapshots[key], loader.minisnapshots[key])

    for i in range(len(sim.snapshots.data)):
        assert np.array_equal(sim.snapshots[i]["u"], loader.snapshots[i]["u"])

    shutil.rmtree(TEST_PATH / "out")
