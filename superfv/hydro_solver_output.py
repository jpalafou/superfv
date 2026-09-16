import pickle
from functools import partial
from pathlib import Path
from typing import Any, Mapping, NoReturn, Union

from .configs import SolverParameters
from .mesh import UniformFiniteVolumeMesh
from .tools.snapshot import SnapshotHistory
from .tools.step_history import StepHistory


def _raise_dummy_callable_error(kind: str) -> NoReturn:
    raise RuntimeError(
        f"This {kind} was substituted because the original callable could not be unpickled."
    )


def dummy_multivar_field(idx, x, y, z, t, *, xp):
    _raise_dummy_callable_error("MultivarField")


def dummy_univar_field(x, y, z, t, *, xp):
    _raise_dummy_callable_error("UnivarField")


def dummy_source_term(idx, u, *, xp):
    _raise_dummy_callable_error("SourceTerm")


def dummy_patch_bc(_u_, context):
    _raise_dummy_callable_error("PatchBC")


def _unresolved_pickle_callable(*args: Any, **kwargs: Any):
    _raise_dummy_callable_error("callable")


class _SolverParametersUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _unresolved_pickle_callable


def _contains_unresolved_pickle_callable(obj: Any) -> bool:
    if obj is _unresolved_pickle_callable:
        return True
    if isinstance(obj, partial):
        return (
            _contains_unresolved_pickle_callable(obj.func)
            or any(_contains_unresolved_pickle_callable(arg) for arg in obj.args)
            or any(
                _contains_unresolved_pickle_callable(value)
                for value in (obj.keywords or {}).values()
            )
        )
    if isinstance(obj, tuple):
        return any(_contains_unresolved_pickle_callable(value) for value in obj)
    if isinstance(obj, list):
        return any(_contains_unresolved_pickle_callable(value) for value in obj)
    if isinstance(obj, Mapping):
        return any(
            _contains_unresolved_pickle_callable(key) or _contains_unresolved_pickle_callable(value)
            for key, value in obj.items()
        )
    return False


def _replace_unresolved_callable(obj: Any, dummy: Any) -> Any:
    return dummy if _contains_unresolved_pickle_callable(obj) else obj


def _repair_unresolved_param_callables(params: SolverParameters) -> SolverParameters:
    params.ic.ic = _replace_unresolved_callable(params.ic.ic, dummy_multivar_field)
    params.ic.passive_ics = {
        name: _replace_unresolved_callable(passive_ic, dummy_univar_field)
        for name, passive_ic in params.ic.passive_ics.items()
    }

    for dim in ("x", "y", "z"):
        modes = getattr(params.bc, f"bc{dim}")
        for side, mode in zip(("lower", "upper"), modes):
            name = f"bc{dim}_callable_{side}"
            dummy = dummy_patch_bc if mode == "patch" else dummy_multivar_field
            setattr(
                params.bc,
                name,
                _replace_unresolved_callable(getattr(params.bc, name), dummy),
            )

    object.__setattr__(
        params, "source", _replace_unresolved_callable(params.source, dummy_source_term)
    )
    return params


class HydroSolverOutput:
    def __init__(self, output_path: Union[str, Path]):
        self.output_path: Path
        self.params: SolverParameters
        self.mesh: UniformFiniteVolumeMesh
        self.step_history: StepHistory
        self.snapshot_history: SnapshotHistory

        self.output_path = Path(output_path)
        if not self.output_path.exists():
            raise FileNotFoundError(f"Output path {self.output_path} does not exist.")

        self.params = self._unpickle("params")
        self.mesh = self._unpickle("mesh")
        self.step_history = self._unpickle("step_history")
        self.snapshot_history = self._unpickle("snapshot_history")

    def _unpickle_params(self):
        file_path = self.output_path / "params.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "rb") as f:
            return _repair_unresolved_param_callables(_SolverParametersUnpickler(f).load())

    def _unpickle(self, name: str):
        if name == "params":
            return self._unpickle_params()

        file_path = self.output_path / f"{name}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception as e:
                print(f"Error occurred while unpickling {file_path}: {e}")
                return None
