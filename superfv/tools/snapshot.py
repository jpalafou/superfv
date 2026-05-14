import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True, frozen=True)
class SnapshotData:
    t: float
    u: np.ndarray
    w: np.ndarray
    has_shock: np.ndarray
    theta: np.ndarray
    troubles: np.ndarray
    cascade_idx: np.ndarray


@dataclass
class Snapshot:
    data: Optional[SnapshotData] = None
    path: Optional[str] = None

    def __post_init__(self):
        if self.path is not None:
            if self.path.exists():
                raise FileExistsError(f"Snapshot path {self.path} already exists.")

    def load(self):
        if self.path is None:
            raise ValueError("Cannot load snapshot: no path specified.")
        if self.data is None:
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def dump(self, clear: bool = False):
        if self.path is None:
            raise ValueError("Cannot dump snapshot: no path specified.")
        if self.data is None:
            raise ValueError("Cannot dump snapshot: no data to dump.")
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
        if clear:
            self.data = None

    def __getattr__(self, name):
        if self.data is None:
            self.load()
        return getattr(self.data, name)

    def __setattr__(self, name, value):
        if name in {"data", "path"}:
            super().__setattr__(name, value)
            return
        if self.data is None:
            self.load()
        setattr(self.data, name, value)


@dataclass
class SnapshotHistory:
    snapshots: list[Snapshot]

    def __getitem__(self, i: int) -> Snapshot:
        return self.snapshots[i]

    def __call__(self, t: float) -> Snapshot:
        closest_snapshot = min(self.snapshots, key=lambda s: abs(s.t - t))
        if closest_snapshot.t != t:
            print(
                f"Warning: requested snapshot at t={t}, "
                f"but closest snapshot is at t={closest_snapshot.t}."
            )
        return closest_snapshot

    def __len__(self):
        return len(self.snapshots)

    def append(self, snapshot: Snapshot):
        self.snapshots.append(snapshot)

    def dump_all(self):
        for snapshot in self.snapshots:
            snapshot.dump()

    def load_all(self):
        for snapshot in self.snapshots:
            snapshot.load()
