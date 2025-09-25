import json
import pickle
from pathlib import Path

from .slicing import VariableIndexMap
from .snapshots import Snapshots


class OutputLoader:

    def __init__(self, base: Path):
        """
        Load simulation output from the specified base directory.

        Args:
            base: The base directory for the simulation output.
        """
        self.base = Path(base)

        if not self.base.exists():
            raise FileNotFoundError(f"Base directory {base} does not exist.")

        self.config = self.load_config()

        self.active_dims = self.config["active_dims"]
        self.variable_index_map = VariableIndexMap(**self.config["variable_index_map"])

        self.mesh = self.load_mesh()
        self.minisnapshots = self.load_minisnapshots()
        self.snapshots = self.load_snapshots()

        print(f'Successfully read simulation output from "{self.base}"')

    def load_config(self):
        with open(self.base / "config.json", "r") as f:
            return json.load(f)

    def load_mesh(self):
        """
        Load the mesh from 'output_dir/snapshots/mesh.pkl'.
        """
        with open(self.base / "mesh.pkl", "rb") as f:
            return pickle.load(f)

    def load_minisnapshots(self):
        """
        Load the minisnapshots from 'output_dir/snapshots/minisnapshots.pkl'.
        """
        with open(self.base / "snapshots" / "minisnapshots.pkl", "rb") as f:
            return pickle.load(f)

    def load_snapshots(self):
        """
        Load the snapshots from 'output_dir/snapshots'.
        """
        return Snapshots.load(self.base / "snapshots")

    def print_timings(self, total_time_spec: str = ".2f"):
        """
        Print the timing statistics for the solver.

        Args:
            total_time_spec: Format specification for the total time column.
        """
        with open(self.base / "timings.txt", "r") as f:
            lines = f.readlines()

        for line in lines:
            print(line.strip())
