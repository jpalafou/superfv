import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .tools.array_management import ArrayLike, ArrayManager
from .tools.snapshots import Snapshots
from .tools.timer import Timer, method_timer


def clamp_dt(t: float, dt: float, target_time: Optional[float] = None) -> float:
    """
    Clamp the time-step size to avoid overshooting the target time.

    Args:
        t: Current time.
        dt: Proposed time-step size.
        target_time: Optional target time to avoid overshooting.

    Returns:
        Clamped time-step size if target_time is provided, otherwise returns dt.
    """
    return dt if target_time is None else min(target_time - t, dt)


class ExplicitODESolver(ABC):
    """
    Base class for explicit ODE solvers for the form u' = f(t, u).

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
    @partial(method_timer, cat="?.compute_dt")
    def compute_dt(self, t: float, u: ArrayLike) -> float:
        """
        Compute the time-step size.

        Args:
            t: Current time.
            u: Current state as an array.

        Returns:
            dt: Time-step size.
        """
        pass

    @abstractmethod
    @partial(method_timer, cat="?.f")
    def f(self, t: float, u: ArrayLike) -> Tuple[float, ArrayLike]:
        """
        Right-hand side of the ODE.

        Args:
            t: Current time.
            u: Current state as an array.

        Returns:
            dudt: Right-hand side of the ODE at (t, u) as an array.
        """
        pass

    def dt_criterion(self, tnew: float, unew: ArrayLike) -> bool:
        """
        Determine if dt is okay and does not need to be revised. Override
        `_dt_criterion` to implement specific behavior.

        Args:
            tnew: New time after the step.
            unew: New proposed state as an array.

        Returns:
            bool: True if the time-step size should be revised based on the new state,
                False otherwise.
        """
        return self._dt_criterion(tnew, unew)

    def default_dt_criterion(self, tnew: float, unew: ArrayLike) -> bool:
        """
        Default dt criterion that always returns True. Override `_dt_criterion` to
        implement specific behavior.

        Args:
            tnew: New time after the step.
            unew: New proposed state as an array.

        Returns:
            bool: Always returns True.
        """
        return True

    def compute_revised_dt(self, t: float, u: ArrayLike, dt: float) -> float:
        """
        Compute a revised time-step size based on the new state. Override
        `_compute_revised_dt` to implement specific behavior.

        Args:
            t: Current time.
            u: Current state as an array.
            dt: Proposed time-step size.

        Returns:
            Revised time-step size.
        """
        return self._compute_revised_dt(t, u, dt)

    def default_compute_revised_dt(self, t: float, u: ArrayLike, dt: float) -> float:
        """
        Default compute revised dt that raises NotImplementedError. Override
        `_compute_revised_dt` to implement specific behavior.

        Args:
            t: Current time.
            u: Current state as an array.
            dt: Proposed time-step size.

        Returns:
            Raises NotImplementedError.
        """
        raise NotImplementedError("compute_revised_dt method not implemented.")

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
        self.minisnapshots["dt_revisions"].append(self.dt_revision_count)

    def __init__(self, u0: np.ndarray):
        """
        Initializes the ODE solver.

        Args:
            u0: Initial state as an array.
        """
        # initialize times
        self.t = 0.0
        self.dt = np.nan
        self.step_count = 0
        self.substep_count = 0
        self.dt_revision_count = 0

        # initialize array manager
        self.arrays = ArrayManager()
        self.arrays.add("u", u0)

        # initialize timer
        self.timer = Timer(cats=["!ExplicitODESolver.integrate.body"])

        # initialize snapshots
        self.snapshots: Snapshots = Snapshots()
        self.minisnapshots: Dict[str, list] = {
            "t": [],
            "n_substeps": [],
            "dt": [],
            "dt_revisions": [],
        }

        # initialize commit details
        self.commit_details = self._get_commit_details()

        # assign default timestep revision and dt criterion
        self._dt_criterion = self.default_dt_criterion
        self._compute_revised_dt = self.default_compute_revised_dt

        # assign stepper signature
        self.stepper: Callable[[float, ArrayLike, float], ArrayLike]

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
            self.timer.start("!ExplicitODESolver.integrate.body")
            for _ in tqdm(range(n)) if progress_bar else range(n):
                self.take_step()
                self.minisnapshot()
            self.timer.stop("!ExplicitODESolver.integrate.body")
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
        self.timer.start("!ExplicitODESolver.integrate.body")
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
        self.timer.stop("!ExplicitODESolver.integrate.body")

        # clean up progress bar
        self.progress_bar_action("cleanup", do_nothing=not progress_bar)

        # write snapshots
        if snapshot_dir is not None:
            self.write_snapshots(overwrite)

    @partial(method_timer, cat="ExplicitODESolver.take_step")
    def take_step(self, target_time: Optional[float] = None):
        """
        Take a single step in the integration.

        Args:
            target_time (Optional[float]): Time to avoid overshooting.
        """
        self.called_at_beginning_of_step()

        # define current time and state
        t, u = self.t, self.arrays["u"]

        # determine time-step size and next state
        dt = clamp_dt(t, self.compute_dt(t, u), target_time)
        while True:
            unew = self.stepper(t, u, dt)
            if self.dt_criterion(t + dt, unew):
                break
            dt = clamp_dt(t, self.compute_revised_dt(t, u, dt), target_time)
            self.dt_revision_count += 1

        # update attributes
        self.arrays["u"][...] = unew
        self.t += dt
        self.dt = dt

        self.called_at_end_of_step()

    def called_at_beginning_of_step(self):
        """
        Helper function called at the beginning of each step. Override for additional
        routines.
        """
        self.substep_count = 0
        self.dt_revision_count = 0

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step. Override for additional
        routines.
        """
        self.step_count += 1
        self.minisnapshot()

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

        def stepper(t, u, dt):
            self.substep_dt = dt

            # stage 1
            k0 = self.f(t, u)
            unew = u + dt * k0
            self.substep_count += 1

            return unew

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def ssprk2(self, *args, **kwargs) -> None:
        self.integrator = "ssprk2"

        def stepper(t, u, dt):
            self.substep_dt = dt  # constant throughout all SSPRK2 stages

            # stage 1
            k0 = self.f(t, u)
            u1 = u + dt * k0
            self.substep_count += 1

            # stage 2
            k1 = self.f(t + dt, u1)
            unew = 0.5 * u + 0.5 * (u1 + dt * k1)
            self.substep_count += 1

            return unew

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def ssprk3(self, *args, **kwargs) -> None:
        self.integrator = "ssprk3"

        def stepper(t, u, dt):
            self.substep_dt = dt  # constant throughout all SSPRK3 stages

            # stage 1
            k0 = self.f(t, u)
            self.substep_count += 1

            # stage 2
            k1 = self.f(t + dt, u + dt * k0)
            self.substep_count += 1

            # stage 3
            k2 = self.f(t + 0.5 * dt, u + 0.25 * dt * k0 + 0.25 * dt * k1)
            unew = u + (1 / 6) * dt * (k0 + k1 + 4 * k2)
            self.substep_count += 1

            return unew

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def rk4(self, *args, **kwargs) -> None:
        self.integrator = "rk4"

        def stepper(t, u, dt):
            self.substep_dt = dt  # constant throughout all RK4 stages

            # stage 1
            k0 = self.f(t, u)
            self.substep_count += 1

            # stage 2
            k1 = self.f(t + 0.5 * dt, u + 0.5 * dt * k0)
            self.substep_count += 1

            # stage 3
            k2 = self.f(t + 0.5 * dt, u + 0.5 * dt * k1)
            self.substep_count += 1

            # stage 4
            k3 = self.f(t + dt, u + dt * k2)
            unew = u + (1 / 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
            self.substep_count += 1

            return unew

        self.stepper = stepper
        self.integrate(*args, **kwargs)
