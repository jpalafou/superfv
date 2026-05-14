import os
import shutil
from functools import partial
from pathlib import Path

import pytest

from superfv import HydroSolver, ic

OUTPUT_PATH = Path("snapshot_test")


def test_fail_when_output_exists():
    f0 = partial(ic.square, vx=1)
    _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH)
    with pytest.raises(FileExistsError):
        _ = HydroSolver(ic=f0, p=1, nx=64, use_MUSCL=True, output_path=OUTPUT_PATH)

    shutil.rmtree(OUTPUT_PATH)


def test_overwrite():
    f0 = partial(ic.square, vx=1)
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
