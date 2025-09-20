import json
import pickle
from pathlib import Path
from typing import Dict, Literal, Union

from .slicing import VariableIndexMap
from .snapshots import Snapshots


class OutputLoader:

    def __init__(self, path: Path):
        """
        Load simulation output from the specified path.

        Args:
            path: The path to the simulation output directory.
        """
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self.config = self.load_config()

        self.active_dims = self.config["active_dims"]
        self.variable_index_map = VariableIndexMap(**self.config["variable_index_map"])

        self.mesh = self.load_mesh()
        self.minisnapshots = self.load_minisnapshots()
        self.file_index = self.load_snapshot_index()

        self.snapshots = Snapshots()

        print(f'Successfully read simulation output from "{self.path}"')

    def load_config(self):
        with open(self.path / "config.json", "r") as f:
            return json.load(f)

    def load_mesh(self):
        """
        Load the mesh from 'output_dir/snapshots/mesh.pkl'.
        """
        with open(self.path / "snapshots" / "mesh.pkl", "rb") as f:
            return pickle.load(f)

    def load_minisnapshots(self):
        """
        Load the minisnapshots from 'output_dir/snapshots/minisnapshots.pkl'.
        """
        with open(self.path / "snapshots" / "minisnapshots.pkl", "rb") as f:
            return pickle.load(f)

    def load_snapshot_index(self) -> Dict[int, float]:
        """
        Load the snapshot index from 'output_dir/snapshots/index.csv'.
        """
        out = {}
        with open(self.path / "snapshots" / "index.csv", "r") as f:
            next(f)
            for line in f:
                i, t = line.strip().split(",")
                out[int(i)] = float(t)
        return out

    def load_snapshot(self, t: Union[float, Literal["all"]]):
        """
        Load the snapshot data at time `t`.

        Args:
            t: Time at which to load the snapshot data. If 'all', load all snapshots.
        """
        if t == "all":
            for t in self.file_index.values():
                self.load_snapshot(t)
            return

        if t not in self.file_index.values():
            raise KeyError(f"No snapshot data available for time {t}.")

        if t in self.snapshots.data:
            return  # Snapshot already loaded

        file_number = [k for k, v in self.file_index.items() if v == t][0]
        filepath = Path(self.path / "snapshots" / f"snapshot_{file_number:04d}.pkl")

        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot file {filepath} does not exist.")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.snapshots.log(t, data)
