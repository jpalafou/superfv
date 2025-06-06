import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .tools.array_management import ArrayLike, ArrayManager
from .tools.snapshots import Snapshots
from .tools.timer import Timer, method_timer


def _dt_ceil(t: float, dt: float, t_max: Optional[float] = None) -> float:
    """
    Returns a reduced dt if it overshoots a target time.

    Args:
        t: Time value.
        dt: Potential time-step size.
        t_max: Time value to avoid overshooting. If None, dt is returned as is.

    Returns:
        Time-step size that does not overshoot t_max if it is defined, otherwise
            returns dt.
    """
    return min(t_max - t, dt) if t_max is not None else dt


class ExplicitODESolver(ABC):
    """
    Base class for explicit ODE solvers for the form y' = f(t, y).

    Attributes:
        t: Current time.
        timestamps: List of timestamps.
        step_count: Number of steps taken.
        am: ArrayManager object.
        timer: Timer object.
        snapshots: Dictionary of snapshots.
        commit_details: Git commit details.
        integrator: Name of the integrator.
        stepper: Stepper function.
        snapshot_dir: Directory to save snapshots.

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
    @partial(method_timer, cat="ExplicitODESolver.f")
    def f(self, t: float, y: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Right-hand side of the ODE.

        Args:
            t: Time value.
            y: State at time t as an array.

        Returns:
            dt: Time-step size.
            dydt: Right-hand side of the ODE at (t, y) as an array.

        Notes:
            - Increment `self.substep_count` if the function is called multiple times
                per step: `self.substep_count += 1`.
        """
        pass

    @partial(method_timer, cat="ExplicitODESolver.read_snapshots")
    def read_snapshots(self) -> bool:
        """
        Read snapshots from the snapshot directory. Override to load more data.

        Returns:
            True if snapshots were read successfully, False otherwise.
        """
        raise NotImplementedError("read_snapshots method not implemented.")

    @partial(method_timer, cat="ExplicitODESolver.write_snapshots")
    def write_snapshots(self, overwrite: bool = False):
        """
        Write snapshots to the snapshot directory. Override to save more data.

        Args:
            overwrite: Whether to overwrite the snapshot directory if it exists.
        """
        raise NotImplementedError("write_snapshots method not implemented.")

    @partial(method_timer, cat="ExplicitODESolver.snapshot")
    def snapshot(self):
        """
        Snapshot function. Override to save more data to `self.snapshots`.
        """
        pass

    @partial(method_timer, cat="ExplicitODESolver.minisnapshot")
    def minisnapshot(self):
        """
        Mini snapshot function. Is executed after every step. Override to save more
        data to `self.minisnapshots`.
        """
        self.minisnapshots["t"].append(self.t)
        self.minisnapshots["dt"].append(self.dt)
        self.minisnapshots["n_substeps"].append(self.substep_count)

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step. Override for additional
        routines.
        """
        pass

    def __init__(
        self,
        y0: np.ndarray,
        state_array_name: str = "y",
    ):
        """
        Initializes the ODE solver.

        Args:
            y0: Initial state as an array.
            state_array_name: Name of the state array.
        """
        # initialize times
        self.t = 0.0
        self.dt = np.nan
        self.timestamps = [0.0]
        self.step_count = 0
        self.substep_count = 0

        # initialize array manager
        self.arrays = ArrayManager()
        self.arrays.add(state_array_name, y0)
        self.state = state_array_name

        # initialize timer, snapshots, and git commit details
        self.timer = Timer(cats=["ExplicitODESolver.integrate.body"])
        self.snapshots: Snapshots = Snapshots()
        self.minisnapshots: Dict[str, list] = {"t": [], "n_substeps": [], "dt": []}
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

    @partial(method_timer, cat="ExplicitODESolver.integrate")
    def integrate(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        log_every_step: bool = False,
        snapshot_dir: Optional[str] = None,
        overwrite: bool = False,
        allow_overshoot: bool = False,
        progress_bar: bool = True,
    ):
        """
        Integrate the ODE.

        Args:
            T: Times to simulate until. If list, snapshots are taken at each time
                in the list. If float, a single time is used. If None, `n` must be
                defined.
            n: Number of steps to take. If None, `T` must be defined.
            log_every_step: Whether to a snapshot at every step.
            snapshot_dir Directory to save snapshots. If None, does not save.
            overwrite: Whether to overwrite the snapshot directory if it exists.
            allow_overshoot: Whether to allow overshooting of 'T' if it is a float.
            progress_bar: Whether to print a progress bar during integration.
        """
        # if given n, perform a simple time evolution
        if n is not None:
            self.snapshot()
            self.minisnapshot()
            self.timer.start("ExplicitODESolver.integrate.body")
            for _ in tqdm(range(n)):
                self.take_step()
                self.minisnapshot()
            self.timer.stop("ExplicitODESolver.integrate.body")
            self.snapshot()
            return

        # try to read snapshots
        if snapshot_dir is not None:
            self.snapshot_dir = os.path.normpath(snapshot_dir)
            if not overwrite:
                if self.read_snapshots():
                    return

        # check T
        if T is None:
            raise ValueError("T and n cannot both be None.")

        # format list of target times
        target_times: List[float]
        if T is None:
            raise ValueError("T and n cannot both be None.")
        elif isinstance(T, int) or isinstance(T, float):
            target_times = [T]
        elif isinstance(T, list):
            target_times = sorted([float(t) for t in T])
        else:
            raise ValueError(f"Invalid type for T: {type(T)}")
        if min(target_times) <= 0:
            raise ValueError("Target times must be greater than 0.")
        T_max = max(target_times)
        target_time = None if allow_overshoot else target_times.pop(0)

        # setup progress bar
        self.progress_bar_action("setup", T=T_max, do_nothing=not progress_bar)

        # initial snapshot
        self.snapshot()
        self.minisnapshot()

        # simulation loop
        self.timer.start("ExplicitODESolver.integrate.body")
        while self.t < T_max:
            self.take_step(target_time=target_time)
            self.minisnapshot()

            # update progress bar
            self.progress_bar_action("update", do_nothing=not progress_bar)

            # snapshot decision and target time update
            if self.t > T_max:
                self.snapshot()  # trigger closing snapshot
            elif (not allow_overshoot and self.t == target_time) or log_every_step:
                self.snapshot()
                if self.t == target_time and self.t < T_max:
                    target_time = target_times.pop(0)
        self.timer.stop("ExplicitODESolver.integrate.body")

        # clean up progress bar
        self.progress_bar_action("cleanup", do_nothing=not progress_bar)

        # write snapshots
        if snapshot_dir is not None:
            self.write_snapshots(overwrite)

    def take_step(self, target_time: Optional[float] = None):
        """
        Take a single step in the integration.

        Args:
            target_time (Optional[float]): Time to avoid overshooting.
        """
        self.substep_count = 0
        self.step_count += 1
        dt, ynext = self.stepper(
            self.t, self.arrays[self.state], target_time=target_time
        )
        self.t += dt
        self.dt = dt
        self.arrays[self.state] = ynext
        self.timestamps.append(self.t)
        self.called_at_end_of_step()

    def progress_bar_action(
        self, action: str, T: Optional[float] = None, do_nothing: bool = False
    ):
        """
        Setup, update, or cleanup the progress bar.

        Args:
            action: Name of action to take: "setup", "update", or "cleanup".
            T: Time to simulate until. If None, no progress bar is set up.
            do_nothing: Whether to do nothing, e.g., if progress bar is disabled.
        """
        if do_nothing:
            return
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

    def ssprk2(self, *args, **kwargs) -> None:
        self.integrator = "ssprk2"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _dt_ceil(t=t, dt=dt, t_max=target_time)
            u1 = u + dt * k0
            _, k1 = self.f(t, u1)
            unext = 0.5 * u + 0.5 * (u1 + dt * k1)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def ssprk3(self, *args, **kwargs) -> None:
        self.integrator = "ssprk3"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _dt_ceil(t=t, dt=dt, t_max=target_time)
            _, k1 = self.f(t + dt, u + dt * k0)
            _, k2 = self.f(t + 0.5 * dt, u + 0.25 * dt * k0 + 0.25 * dt * k1)
            unext = u + (1 / 6) * dt * (k0 + k1 + 4 * k2)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def rk4(self, *args, **kwargs) -> None:
        self.integrator = "rk4"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _dt_ceil(t=t, dt=dt, t_max=target_time)
            _, k1 = self.f(t + 0.5 * dt, u + 0.5 * dt * k0)
            _, k2 = self.f(t + 0.5 * dt, u + 0.5 * dt * k1)
            _, k3 = self.f(t + dt, u + dt * k2)
            unext = u + (1 / 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)
