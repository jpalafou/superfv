import pickle
from pathlib import Path
from typing import Union

from .configs import SolverParameters
from .mesh import UniformFiniteVolumeMesh
from .tools.snapshot import SnapshotHistory
from .tools.step_history import StepHistory


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

    def _unpickle(self, name: str):
        file_path = self.output_path / f"{name}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, "rb") as f:
            try:
                return pickle.load(f)
            except Exception as e:
                print(f"Error occurred while unpickling {file_path}: {e}")
                return None
