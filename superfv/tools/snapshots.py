from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Snapshots:
    """
    Like a dict but with a more convenient interface for accessing snapshot data.
    """

    def __init__(self):
        self.data: Dict[float, Any] = {}
        self._update_metadata()

    def _update_metadata(self):
        self.time_values = sorted(list(self.data.keys()))
        self.size = len(self.time_values)

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
        if t not in self.data:
            raise KeyError(f"No snapshot data available for time {t}.")
        del self.data[t]
        self._update_metadata()

    def clear(self):
        """
        Remove all snapshot data.
        """
        self.data.clear()
        self._update_metadata()

    def __call__(self, t: float) -> Any:
        """
        Returns the snapshot data at time `t`.

        Args:
            t: Time at which to retrieve the snapshot data.

        Returns:
            Snapshot data at time `t`.

        Raises:
            KeyError: If no snapshot data is available for time `t`.
        """
        if t not in self.data:
            raise KeyError(f"No snapshot data available for time {t}.")
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
        """
        if not isinstance(n, int):
            raise TypeError(f"Index must be an integer. Got {type(n)}.")
        if n < -self.size or n >= self.size:
            raise IndexError(
                f"Index {n} out of range. Valid range: [{-self.size}, {self.size - 1}]."
            )
        return self.data[self.time_values[n]]

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
