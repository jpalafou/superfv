import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .tools.array_management import ArrayLike, ArrayManager
from .tools.timer import Timer, method_timer


def _dt_ceil(t: float, dt: float, t_max: Optional[float] = None) -> float:
    """
    Returns a reduced dt if it overshoots a target time.

    Args:
        t (float): Time value.
        dt (float): Potential time-step size.
        t_max (Optional[float]): Time value to avoid overshooting.

    Returns:
        (float): Time-step size that does not overshoot t_max.
    """
    return min(t_max - t, dt) if t_max is not None else dt


class ExplicitODESolver(ABC):
    """
    Base class for explicit ODE solvers for the form y' = f(t, y).

    Attributes:
        t (float): Current time.
        timestamps (List[float]): List of timestamps.
        step_count (int): Number of steps taken.
        am (ArrayManager): Array manager.
        timer (Timer): Timer object.
        snapshots (dict): Dictionary of snapshots.
        print_progress_bar (bool): If True, display a progress bar.
        commit_details (dict): Git commit details.
        integrator (str): Name of the integrator.
        stepper (Callable): Stepper function.
        snapshot_dir (str): Directory to save snapshots.

    Notes:
        - The `f` method must be implemented by the subclass.
        - The `read_snapshots` and `write_snapshots` methods can be overridden to
          read and write additional data.
        - The `snapshot` method can be overridden to save additional data or perform
            other operations at each snapshot.
        - The `called_at_end_of_step` method can be overridden to perform additional
          routines at the end of each step.
    """

    @abstractmethod
    def f(self, t: float, y: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Right-hand side of the ODE.

        Args:
            t (float): Time value.
            y (ArrayLike): Solution value.

        Returns:
            dt (float): Time-step size.
            dydt (ArrayLike): Right-hand side of the ODE at (t, y).
        """
        pass

    @partial(method_timer, "READING")
    def read_snapshots(self) -> bool:
        """
        Read snapshots from the snapshot directory. Override to load more data.

        Returns:
            (bool): True if snapshots were read successfully.
        """
        raise NotImplementedError("read_snapshots method not implemented.")

    @partial(method_timer, "WRITING")
    def write_snapshots(self, overwrite: bool = False):
        """
        Write snapshots to the snapshot directory. Override to save more data.

        Args:
            overwrite (bool): Overwrite the snapshot directory if it exists.
        """
        raise NotImplementedError("write_snapshots method not implemented.")

    @partial(method_timer, "READING")
    def snapshot(self):
        """
        Snapshot function. Override to save more data to `self.snapshots`.
        """
        pass

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step. Override for additional
        routines.
        """
        pass

    def __init__(self, y0: np.ndarray, progress_bar: bool = True, cupy: bool = False):
        """
        Initializes the ODE solver.

        Args:
            y0 (np.ndarray): Initial solution value.
            progress_bar (bool): If True, display a progress bar.
            cupy (bool): If True, use CuPy for computations.
        """
        # declare types
        self.snapshots: Dict[float, Any]

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
        self.timer = Timer(cats=["ODE_INT", "SNAPSHOTS"])
        self.snapshots = {}
        self.print_progress_bar = True if progress_bar else False
        self.commit_details = self._get_commit_details()

    def _get_commit_details(self) -> dict:
        """
        Returns a dict summary of the commit details of the repository.
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

    def integrate(
        self,
        T: Optional[Union[int, float, Iterable[Union[int, float]]]] = None,
        n: Optional[int] = None,
        log_every_step: bool = False,
        snapshot_dir: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Integrate the ODE.

        Args:
            T (Optional[Union[int, float, Iterable[Union[int, float]]]]): Time to
                simulate until. If an iterable, take snapshots at those times.
            n (Optional[int]): Number of iterations to evolve. If defined, all other arguments are
                ignored.
            log_every_step (bool): Take a snapshot at every step.
            snapshot_dir (Optional[str]): Directory to save snapshots. If None, does not save.
            overwrite (bool): Overwrite the snapshot directory if it exists.
        """
        # if given n, perform a simple time evolution
        if n is not None:
            self.timer.start("SNAPSHOTS")
            self.snapshot()
            self.timer.stop("SNAPSHOTS")
            self.timer.start("ODE_INT")
            for _ in tqdm(range(n)):
                self.take_step()
            self.timer.stop("ODE_INT")
            self.timer.start("SNAPSHOTS")
            self.snapshot()
            self.timer.stop("SNAPSHOTS")
            return

        # try to read snapshots
        if snapshot_dir is not None:
            self.snapshot_dir = os.path.normpath(snapshot_dir)
            if not overwrite:
                if self.read_snapshots():
                    return

        if T is None:
            raise ValueError("T must be defined.")
        elif isinstance(T, int):
            target_times = [float(T)]
        elif isinstance(T, float):
            target_times = [T]
        elif isinstance(T, list) or isinstance(T, tuple):
            target_times = sorted([float(t) if isinstance(t, int) else t for t in T])
        elif isinstance(T, np.ndarray):
            target_times = sorted(T.astype(float).tolist())
        else:
            raise ValueError(f"Invalid type for T: {type(T)}")
        if min(target_times) <= 0:
            raise ValueError("Target times must be greater than 0.")

        # define simulation stop time
        T_max = target_times[-1]

        # setup progress bar
        self.progress_bar_action(action="setup", T=T_max)

        # get unique target times and ignore 0
        target_time = target_times.pop(0)

        # initial snapshot
        self.snapshot()

        # simulation loop
        self.timer.start("ODE_INT")
        while self.t < T_max:
            self.take_step(target_time=target_time)
            self.progress_bar_action(action="update")
            # target time actions
            if self.t == target_time or log_every_step:
                self.timer.start("SNAPSHOTS")
                self.snapshot()
                self.timer.stop("SNAPSHOTS")
                if self.t == target_time:
                    target_time = target_times.pop(0)
        self.timer.stop("ODE_INT")

        # clean up progress bar
        self.progress_bar_action(action="cleanup")

        # write snapshots
        if snapshot_dir is not None:
            self.write_snapshots(overwrite)

    def take_step(self, target_time: Optional[float] = None):
        """
        Take a single step in the integration.

        Args:
            target_time (Optional[float]): Time to avoid overshooting.
        """
        dt, ynext = self.stepper(self.t, self.am["y"], target_time=target_time)
        self.am["y"] = ynext
        self.timestamps.append(self.t)
        self.step_count += 1
        self.called_at_end_of_step()

    def progress_bar_action(self, action: str, T: Optional[float] = None):
        """
        Setup, update, or cleanup the progress bar.

        Args:
            action (str): "setup", "update", "cleanup".
            T (Optional[float]): Time to simulate until.
        """
        if self.print_progress_bar:
            if action == "setup":
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
                if T is None:
                    raise ValueError("T must be defined.")
                self.progress_bar = tqdm(total=T, bar_format=bar_format)
            elif action == "update":
                self.progress_bar.n = self.t
                self.progress_bar.refresh()
            elif action == "cleanup":
                self.progress_bar.close()

    def euler(self, *args, **kwargs) -> None:
        self.integrator = "euler"

        def stepper(t, u, target_time=None):
            dt, dudt = self.f(t, u)
            dt = _dt_ceil(t=t, dt=dt, t_max=target_time)
            unext = u + dt * dudt
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)
