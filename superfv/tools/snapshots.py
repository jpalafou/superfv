import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


@dataclass(frozen=True)
class SnapshotPlaceholder:
    """
    Placeholder for a snapshot that is stored on disk.
    """

    path: Path

    def load(self) -> Any:
        with open(self.path, "rb") as f:
            return pickle.load(f)


@dataclass
class Snapshots:
    """
    Like a dict but with a more convenient interface for accessing snapshot data.
    """

    def __init__(self):
        self.data: Dict[float, Union[Any, SnapshotPlaceholder]] = {}
        self.file_index: Dict[int, float] = {}
        self._update_metadata()

    def _update_metadata(self):
        self.time_values = sorted(list(self.data.keys()))
        self.size = len(self.time_values)

    def _check_t_exists(self, t: float):
        if t not in self.data:
            raise KeyError(f"No snapshot data available for time {t}.")

    def log(self, t: float, data: Any):
        """
        Log snapshot data at time `t`.

        Args:
            t: Time at which to log the snapshot data.
            data: Snapshot data to log.

        Raises:
            ValueError: If snapshot data already exists for time `t`.
        """
        if t not in self.data:
            self.data[t] = data
            self._update_metadata()
        else:
            raise ValueError(f"Snapshot data already exists for time {t}.")

    def unlog(self, t: float):
        """
        Remove snapshot data at time `t`.

        Args:
            t: Time at which to remove the snapshot data.

        Raises:
            KeyError: If no snapshot data is available for time `t`.
        """
        self._check_t_exists(t)
        del self.data[t]
        self._update_metadata()

    def write(self, base: Path, t: float, discard: bool = False):
        """
        Write snapshot data at time `t` to the specified path.

        Args:
            base: Base directory to which the snapshot data is written.
            t: Time at which to write the snapshot data.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self._check_t_exists(t)
        if not base.exists():
            raise FileNotFoundError(f"Base directory {base} does not exist.")
        if t in self.file_index.values():
            raise ValueError(f"Snapshot data for time {t} already written to disk.")

        idx, filepath = self._next_index_and_path(base)

        # write snapshot to disk
        with open(filepath, "wb") as f:
            pickle.dump(self.data[t], f)
        self.file_index[idx] = t

        # update on-disk index
        index_path = base / "index.csv"
        if not index_path.exists():
            index_path.write_text("idx,t\n")
        with open(index_path, "a") as f:
            f.write(f"{idx},{t}\n")

        # discard in-memory data
        if discard:
            self.data[t] = SnapshotPlaceholder(filepath)

    def _next_index_and_path(self, base: Path) -> Tuple[int, Path]:
        """
        Get the next available snapshot directory and index.

        Args:
            base: Base directory where snapshots are stored.
            n_digits: Number of digits for the snapshot index.

        Returns:
            A tuple containing the next available index and the corresponding path.
        """
        idx = max(self.file_index.keys(), default=-1) + 1
        if idx > 9999:
            raise RuntimeError("Exceeded maximum number of snapshots (9999).")
        filename = f"snapshot_{idx:04d}.pkl"
        return idx, base / filename

    def clear(self):
        """
        Remove all snapshot data.
        """
        self.data.clear()
        self.file_index.clear()
        self._update_metadata()

    @classmethod
    def load(cls, base: Path) -> "Snapshots":
        """
        Load snapshots as placeholders from the specified base directory containing
        "index.csv" and corresponding snapshot files of the form "snapshot_XXXX.pkl".

        Args:
            base: Path to the directory containing index and snapshot files.
        """
        if not base.exists():
            raise FileNotFoundError(f"Base directory {base} does not exist.")

        index_path = base / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_path}")

        instance = cls()
        with open(index_path, "r") as f:
            next(f)  # skip header
            for line in f:
                # parse index and time
                idx_str, t_str = line.strip().split(",")
                idx = int(idx_str)
                t = float(t_str)

                # load snapshot placeholder
                filename = f"snapshot_{idx:04d}.pkl"
                filepath = base / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Snapshot file not found at {filepath}")
                instance.data[t] = SnapshotPlaceholder(filepath)

                # update file index
                instance.file_index[idx] = t

        instance._update_metadata()
        return instance

    def __call__(self, t: float) -> Any:
        """
        Returns the snapshot data at time `t`.

        Args:
            t: Time at which to retrieve the snapshot data.

        Returns:
            Snapshot data at time `t`.

        Raises:
            KeyError: If no snapshot data is available for time `t`.

        Note:
            If the snapshot data is a placeholder, it will be loaded from disk.
        """
        self._check_t_exists(t)
        if isinstance(self.data[t], SnapshotPlaceholder):
            return self.data[t].load()
        return self.data[t]

    def __getitem__(self, n: int) -> Any:
        """
        Returns the snapshot data at index `n`.

        Args:
            n (int): Index of the snapshot data.

        Returns:
            Snapshot data at index `n`.

        Raises:
            IndexError: If the index `n` is out of range.

        Note:
            If the snapshot data is a placeholder, it will be loaded from disk.
        """
        if not isinstance(n, int):
            raise TypeError(f"Index must be an integer. Got {type(n)}.")
        if n < -self.size or n >= self.size:
            raise IndexError(
                f"Index {n} out of range. Valid range: [{-self.size}, {self.size - 1}]."
            )

        snapshot = self.data[self.time_values[n]]
        if isinstance(snapshot, SnapshotPlaceholder):
            return snapshot.load()

        return snapshot

    def __iter__(self):
        """
        Allows iteration over the snapshots in order of time.

        Yields:
            A tuple containing time and snapshot data.
        """
        for t in self.time_values:
            yield t, self.data[t]

    def __contains__(self, t: float) -> bool:
        """
        Checks if snapshot data is available for time `t`.

        Args:
            t: Time to check for snapshot data.

        Returns:
            True if snapshot data is available for time `t`, False otherwise.
        """
        return t in self.data

    def times(self) -> List[float]:
        """
        Returns the list of time values for which snapshot data is available.

        Returns:
            List of time values.
        """
        return self.time_values[:]

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
