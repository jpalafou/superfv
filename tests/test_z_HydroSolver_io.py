import os
import pickle
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from superfv import HydroSolver, HydroSolverOutput, ics
from superfv.boundary_conditions import apply_free_bc
from superfv.configs import dummy_function
from superfv.hydro_solver_output import (
    dummy_multivar_field,
    dummy_patch_bc,
    dummy_source_term,
    dummy_univar_field,
)

OUTPUT_PATH = Path("snapshot_test")


def _main_module_bc(idx, x, y, z, t, *, xp):
    return ics.square(idx, x, y, z, t, xp=xp, vx=1)


def _main_module_passive(x, y, z, t, *, xp):
    return xp.zeros_like(x)


def _main_module_source(idx, u, *, xp):
    return xp.zeros_like(u)


def _main_module_patch(_u_, context):
    apply_free_bc(_u_, context)


def _mark_as_main_module(monkeypatch, *funcs):
    for func in funcs:
        monkeypatch.setattr(func, "__module__", "__main__")
        monkeypatch.setattr(sys.modules["__main__"], func.__name__, func, raising=False)


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


def test_local_passive_ic_does_not_break_params_files(tmp_path):
    f0 = partial(ics.square, vx=1)

    def local_passive_ic(x, y, z, t, *, xp):
        return xp.zeros_like(x)

    output_path = tmp_path / "snapshot_test"
    _ = HydroSolver(
        ic=f0,
        passive_ics={"dye": local_passive_ic},
        p=1,
        nx=64,
        use_MUSCL=True,
        output_path=output_path,
    )

    assert (output_path / "params.yaml").exists()
    with open(output_path / "params.pkl", "rb") as f:
        params = pickle.load(f)
    assert params.ic.passive_ics["dye"] is dummy_function


def test_main_module_bc_is_preserved_when_loadable(tmp_path, monkeypatch):
    f0 = partial(ics.square, vx=1)
    _mark_as_main_module(monkeypatch, _main_module_bc)

    output_path = tmp_path / "snapshot_test"
    _ = HydroSolver(
        ic=f0,
        p=1,
        nx=64,
        use_MUSCL=True,
        bcx=("dirichlet", "free"),
        bcx_callable_lower=_main_module_bc,
        output_path=output_path,
    )

    with open(output_path / "params.pkl", "rb") as f:
        params = pickle.load(f)
    assert params.bc.bcx_callable_lower is _main_module_bc


def test_output_loads_legacy_main_module_callables(tmp_path, monkeypatch):
    f0 = partial(ics.square, vx=1)
    _mark_as_main_module(
        monkeypatch,
        _main_module_bc,
        _main_module_passive,
        _main_module_source,
        _main_module_patch,
    )

    output_path = tmp_path / "snapshot_test"
    sim = HydroSolver(
        ic=f0,
        passive_ics={"dye": _main_module_passive},
        source=_main_module_source,
        p=1,
        nx=64,
        use_MUSCL=True,
        bcx=("patch", "dirichlet"),
        bcx_callable_lower=_main_module_patch,
        bcx_callable_upper=_main_module_bc,
        output_path=output_path,
    )
    sim.run(0.01, print_update=False)

    with open(output_path / "params.pkl", "wb") as f:
        pickle.dump(sim.params, f)
    for func in (
        _main_module_bc,
        _main_module_passive,
        _main_module_source,
        _main_module_patch,
    ):
        monkeypatch.delattr(sys.modules["__main__"], func.__name__)

    output = HydroSolverOutput(output_path)

    assert output.params is not None
    assert output.params.ic.passive_ics["dye"] is dummy_univar_field
    assert output.params.source is dummy_source_term
    assert output.params.bc.bcx_callable_lower is dummy_patch_bc
    assert output.params.bc.bcx_callable_upper is dummy_multivar_field


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
