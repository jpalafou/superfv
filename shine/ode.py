import os
import subprocess
from abc import ABC, abstractmethod

import numpy as np

from .tools.array_management import ArrayManager, ArrayType
from .tools.timer import Timer


def _dt_ceil(t: float, dt: float, t_max: float = None) -> None:
    """
    Returns a reduced dt if it overshoots a target time.

    Args:
        t: Time value.
        dt: Potential time-step size.
        t_max: Time value to avoid overshooting.

    Returns:
        Time-step size that does not overshoot t_max.
    """
    return t_max - t if t_max is not None and t + dt > t_max else dt


class ExplicitODE(ABC):
    """
    Base class for explicit ODE solvers for the form y' = f(t, y).
    """

    @abstractmethod
    def f(self, t: float, y: ArrayType) -> ArrayType:
        """
        Right-hand side of the ODE.

        Args:
            t: Time value.
            y: Solution value.

        Returns:
            Right-hand side of the ODE.
        """
        pass

    def __init__(self, y0: np.ndarray, progress_bar: bool = True, cupy: bool = False):
        """
        Initializes the ODE solver.

        Args:
            y0: Initial solution value.
            progress_bar: If True, display a progress bar.
            cupy: If True, use CuPy for computations.
        """
        # initialize times
        self.t = 0.0
        self.timestamps = [0.0]
        self.step_count = 0

        # initialize array manager
        self.am = ArrayManager()
        if cupy:
            self.am.enable_cupy()
        self.am.add("y", y0)

        # initialize timer, snapshots, progress bar, and git commit details
        self.timer = Timer(cats=["TOTAL", "SNAPSHOTS"])
        self.snapshots = {}
        self.print_progress_bar = True if progress_bar else False
        self.commit_details = self._get_commit_details()

    def _get_commit_details(self) -> dict:
        """
        Returns the commit details of the repository.
        """
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            # Navigate to the repository path and get commit details
            result = subprocess.run(
                ["git", "-C", repo_path, "log", "-1", "--pretty=format:%H|%an|%ai|%D"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            commit_info = result.stdout.strip().split("|")
            commit_hash = commit_info[0]
            author_name = commit_info[1]
            commit_date = commit_info[2]
            branch_name = (
                commit_info[3].split(",")[0].strip().split()[-1]
                if len(commit_info) > 3
                else None
            )

            return {
                "commit_hash": commit_hash,
                "author_name": author_name,
                "commit_date": commit_date,
                "branch_name": branch_name,
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"An error occurred: {e.stderr.strip()}"}
