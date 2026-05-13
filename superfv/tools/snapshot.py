from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    t: float
    u: np.ndarray
    w: np.ndarray
    has_shock: np.ndarray
    theta: np.ndarray
    troubles: np.ndarray
    cascade_idx: np.ndarray


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

    def append(self, snapshot: Snapshot):
        self.snapshots.append(snapshot)
