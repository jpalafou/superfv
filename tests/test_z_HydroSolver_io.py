import os
import pickle
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from superfv import HydroSolver, HydroSolverOutput, ics
from superfv.configs import dummy_function

OUTPUT_PATH = Path("snapshot_test")


def test_fail_when_output_exists():
    f0 = partial(ics.square, vx=1)
    _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH)
    with pytest.raises(FileExistsError):
        _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH)

    shutil.rmtree(OUTPUT_PATH)


def test_overwrite():
    f0 = partial(ics.square, vx=1)
    _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH)

    # write a dummy file
    with open(OUTPUT_PATH / "dummy.txt", "w") as f:
        f.write("dummy content")

    # overwrite snapshot
    with pytest.warns(
        UserWarning, match=f"Output path '{OUTPUT_PATH}' already exists. Overwriting."
    ):
        _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH, overwrite=True)

    assert not os.path.exists(OUTPUT_PATH / "dummy.txt"), "Dummy file should be removed"

    shutil.rmtree(OUTPUT_PATH)


def test_lambda_source_does_not_break_params_files(tmp_path):
    f0 = partial(ics.square, vx=1)

    output_path = tmp_path / "snapshot_test"
    _ = HydroSolver(
        ic=f0,
        source=lambda idx, u, *, xp: xp.zeros_like(u),
        p=1,
        nx=64,
        use_MUSCL=True,
        output_path=output_path,
    )

    assert (output_path / "params.yaml").exists()
    with open(output_path / "params.pkl", "rb") as f:
        params = pickle.load(f)
    assert params.source is dummy_function
    assert params.ic.ic.func is ics.square
    assert params.ic.ic.keywords == {"vx": 1}


def test_local_ic_does_not_break_params_files(tmp_path):
    def local_ic(idx, x, y, z, t, *, xp):
        return ics.square(idx, x, y, z, t, xp=xp, vx=1)

    output_path = tmp_path / "snapshot_test"
    _ = HydroSolver(
        ic=local_ic,
        p=1,
        nx=64,
        use_MUSCL=True,
        output_path=output_path,
    )

    assert (output_path / "params.yaml").exists()
    with open(output_path / "params.pkl", "rb") as f:
        params = pickle.load(f)
    assert params.ic.ic is dummy_function


@pytest.mark.parametrize("discard_after_writing", [True, False])
def test_writing_and_reading_snapshot(discard_after_writing):
    f0 = partial(ics.square, vx=1)
    sim = HydroSolver(
        ic=f0,
        p=1,
        nx=64,
        use_MUSCL=True,
        output_path=OUTPUT_PATH,
        discard_after_writing=discard_after_writing,
    )

    sim.run(1.0)

    output = HydroSolverOutput(OUTPUT_PATH)

    # check that every attribute of params is identical except for `ic`
    for attr in sim.params.__dataclass_fields__.keys():
        if attr == "ic":
            continue
        assert getattr(sim.params, attr) == getattr(output.params, attr)

    # check that all snapshots can be loaded and are identical
    assert len(sim.snapshot_history) > 1
    for i in range(len(sim.snapshot_history)):
        snapshot_from_sim = sim.snapshot_history[i]
        snapshot_from_output = output.snapshot_history[i]

        assert snapshot_from_sim.t == snapshot_from_output.t
        assert np.array_equal(snapshot_from_sim.u, snapshot_from_output.u)

    shutil.rmtree(OUTPUT_PATH)
