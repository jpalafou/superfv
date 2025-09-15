import os
import pickle
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from .tools.device_management import ArrayLike, ArrayManager
from .tools.snapshots import Snapshots
from .tools.timer import Timer


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


def status_print(msg: str, closing: bool = False, width: int = 100):
    """
    Print a status message with a fixed width.

    Args:
        msg: Message to print.
        closing: Whether this is the closing message to print. If False, it will print the
            message without a trailing newline.
        width: Width of the printed message.
    """
    print(f"\r{msg:<{width}}", end="\n" if closing else "")


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

    Notes:
        - The `f` method must be implemented by the subclass.
        - The `snapshot` method can be overridden to save additional data or perform
            other operations at each snapshot.
        - The `called_at_end_of_step` method can be overridden to perform additional
          routines at the end of each step.
    """

    def __init__(self, u0: np.ndarray, array_manager: Optional[ArrayManager] = None):
        """
        Initializes the ODE solver.

        Args:
            u0: Initial state as an array.
            array_manager: Optional ArrayManager instance to manage arrays.
        """
        # initialize time values
        self.t = 0.0
        self.dt = 0.0

        # initialize logs
        self.reset_global_logs()
        self.reset_stepwise_logs()
        self.reset_substepwise_logs()

        # initialize array manager
        self.arrays = ArrayManager() if array_manager is None else array_manager
        self.arrays.add("u", u0)
        self.arrays.add("k0", np.empty_like(u0))
        self.arrays.add("k1", np.empty_like(u0))
        self.arrays.add("k2", np.empty_like(u0))
        self.arrays.add("k3", np.empty_like(u0))
        self.arrays.add("unew", np.empty_like(u0))
        self.arrays["k0"].fill(np.nan)
        self.arrays["k1"].fill(np.nan)
        self.arrays["k2"].fill(np.nan)
        self.arrays["k3"].fill(np.nan)
        self.arrays["unew"].fill(np.nan)

        # initialize timer
        self.timer = Timer(cats=["wall", "take_step", "snapshot", "minisnapshot"])

        # initialize snapshots
        self.snapshots: Snapshots = Snapshots()
        self.minisnapshots: Dict[str, list] = {}
        for key in self.prepare_minisnapshot_data().keys():
            self.minisnapshots[key] = []

        # initialize IO
        self.path: Optional[Path] = None

        # initialize commit details
        self.commit_details = self._get_commit_details()

        # assign default timestep revision and dt criterion
        self._dt_criterion = self.default_dt_criterion
        self._compute_revised_dt = self.default_compute_revised_dt

        # assign stepper signature
        self.stepper: Callable[[float, ArrayLike, float], None]

    @abstractmethod
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
    def f(self, t: float, u: ArrayLike) -> ArrayLike:
        """
        Right-hand side of the ODE.

        Args:
            t: Current time.
            u: Current state as an array.

        Returns:
            dudt: Right-hand side of the ODE at (t, u) as an array.
        """
        pass

    @abstractmethod
    def build_opening_message(self) -> str:
        """
        Returns a string message for the printed opening message before integration
        starts.
        """
        pass

    @abstractmethod
    def build_update_message(self) -> str:
        """
        Returns a string message for the printed update during integration.
        """
        pass

    @abstractmethod
    def build_closing_message(self) -> str:
        """
        Returns a string message for the printed closing message after integration
        ends.
        """
        pass

    @abstractmethod
    def prepare_snapshot_data(self) -> Any:
        """
        Returns the data to be saved in the snapshot at time `self.t`.
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

    def take_step(self, target_time: Optional[float] = None):
        """
        Take a single step in the integration.

        Args:
            target_time (Optional[float]): Time to avoid overshooting.
        """
        self.called_at_beginning_of_step()

        t, u = self.t, self.arrays["u"]
        dt = clamp_dt(t, self.compute_dt(t, u), target_time)

        while True:
            self.reset_substepwise_logs()
            self.stepper(t, u, dt)  # revises self.arrays["unew"]
            if self.dt_criterion(t + dt, self.arrays["unew"]):
                break
            dt = clamp_dt(t, self.compute_revised_dt(t, u, dt), target_time)
            self.n_dt_revisions += 1

        # update attributes
        self.arrays["u"][...] = self.arrays["unew"]
        self.t += dt
        self.dt = dt

        self.called_at_end_of_step()

    def called_at_beginning_of_step(self):
        """
        Helper function called at the beginning of each step starting with a timer
        start.
        """
        self.reset_stepwise_logs()
        self.timer.start("take_step")

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step ending with a timer stop.
        """
        self.timer.stop("take_step")
        self.increment_stepwise_logs()
        self.increment_global_logs()

    def integrate(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        log_every_step: bool = False,
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        no_snapshots: bool = False,
        path: Optional[str] = None,
        overwrite: bool = False,
    ):
        """
        Integrate the ODE.

        Args:
            T: Times to simulate until. If list, snapshots are taken at each time
                in the list. If float, a single time is used. If None, `n` must be
                defined.
            n: Number of steps to take. If None, `T` must be defined.
            log_every_step: Whether to a snapshot at every step.
            allow_overshoot: Whether to allow overshooting of 'T' if it is a float.
            verbose: Whether to print verbose output during integration.
            log_freq: Step frequency of logging updates to the progress bar.
            no_snapshots: Whether to skip taking snapshots.
            path: Path to which integration output is written if not None.
            overwrite: Whether to overwrite the output directory if it exists.
        """
        self.timer.start("wall")

        # prepare output directory
        self.prepare_output_directory(path, overwrite)

        # perform integration
        if n is not None and T is None:
            self._integrate_for_fixed_number_of_steps(
                n,
                log_every_step=log_every_step,
                verbose=verbose,
                log_freq=log_freq,
                no_snapshots=no_snapshots,
            )
        elif T is not None and n is None:
            self._integrate_until_target_time_is_reached(
                cast(Union[float, List[float]], T),
                log_every_step=log_every_step,
                allow_overshoot=allow_overshoot,
                verbose=verbose,
                log_freq=log_freq,
                no_snapshots=no_snapshots,
            )
        else:
            raise ValueError("Either 'n' or 'T' must be defined, but not both.")

        self.timer.stop("wall")

    def _integrate_for_fixed_number_of_steps(
        self,
        n: int,
        log_every_step: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        no_snapshots: bool = False,
    ):
        """
        Integrate the ODE for a fixed number of steps.

        Args:
            n: Number of steps to take.
            log_every_step: Whether to a snapshot at every step.
            verbose: Whether to print a progress bar during integration.
            log_freq: Step frequency of logging updates to the progress bar.
            no_snapshots: Whether to skip taking snapshots.
        """
        # print initial message
        if verbose:
            status_print(self.build_opening_message())

        # take initial snapshots
        if self.t not in self.minisnapshots["t"]:
            if not no_snapshots:
                self.take_snapshot()
            self.take_minisnapshot()

        # simulation loop
        for _ in range(n):
            self.take_step()
            self.take_minisnapshot()

            # take snapshot
            if log_every_step or self.n_steps == n:
                self.take_snapshot()

            # update printed message
            if verbose:
                if self.n_steps % log_freq == 0 or self.n_steps >= n:
                    status_print(self.build_update_message())

        # wrap up snapshots
        self.postprocess_snapshots()

        # print closing message
        if verbose:
            status_print(self.build_closing_message(), closing=True)

    def _integrate_until_target_time_is_reached(
        self,
        T: Union[float, List[float]],
        log_every_step: bool = False,
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        no_snapshots: bool = False,
    ):
        """
        Integrate the ODE until a target time is reached.

        Args:
            T: Times to simulate until. If list, snapshots are taken at each time
                in the list. If float, a single time is used.
            log_every_step: Whether to a snapshot at every step.
            allow_overshoot: Whether to allow overshooting of 'T' if it is a float.
            verbose: Whether to print a progress bar during integration.
            log_freq: Step frequency of logging updates to the progress bar.
            no_snapshots: Whether to skip taking snapshots.
        """
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
        if verbose:
            status_print(self.build_opening_message())

        # initial snapshot
        if self.t not in self.minisnapshots["t"]:
            if not no_snapshots:
                self.take_snapshot()
            self.take_minisnapshot()

        # simulation loop
        while self.t < T_max:
            self.take_step(target_time=target_time)
            self.take_minisnapshot()

            # snapshot decision and target time update
            if not no_snapshots:
                if self.t > T_max:  # trigger closing snapshot
                    self.take_snapshot()
                elif (not allow_overshoot and self.t == target_time) or log_every_step:
                    self.take_snapshot()

                    if self.t == target_time and self.t < T_max:
                        target_time = target_times.pop(0)

            # update progress bar
            if verbose:
                if self.n_steps % log_freq == 0 or self.t >= T_max:
                    status_print(self.build_update_message())

        # closing actions
        if verbose:
            status_print(self.build_closing_message(), closing=True)

    def prepare_minisnapshot_data(self) -> Dict[str, Any]:
        """
        Returns the data to be saved in a minisnapshot.
        """
        return {
            "t": self.t,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "n_substeps": self.n_substeps,
            "n_dt_revisions": self.n_dt_revisions,
            "step_time": self.timer.goldfish_lap_time["take_step"],
        }

    def prepare_output_directory(
        self, path: Optional[str] = None, overwrite: bool = False
    ):
        """
        Create output directory if it doesn't exist and throw an error or overwrite it
        if it does.

        Args:
            path: Output path as a string.
            overwrite: Whether to completely delete the path if it exists.
        """
        if path is None:
            return None

        out_path = Path(path)

        if out_path.exists() and overwrite:
            shutil.rmtree(out_path)
        elif out_path.exists() and not overwrite:
            raise FileExistsError(f"Output directory '{out_path}' already exists.")

        os.makedirs(out_path)
        os.makedirs(out_path / "snapshots")
        self.path = out_path

        # write some metadata before anything runs
        self.write_metadata()

    def write_metadata(self):
        """
        Write commit details before the solver runs.
        """
        if self.path is None:
            return
        with open(self.path / "commit_details.txt", "w") as f:
            for key, value in self.commit_details.items():
                f.write(f"{key}: {value}\n")

    def take_snapshot(self):
        """
        Log and time snapshot data at time `self.t` and write it to `self.path` if not None.
        """
        self.timer.start("snapshot")

        data = self.prepare_snapshot_data()
        self.snapshots.log(self.t, data)

        if self.path is not None:
            self.snapshots.write(self.path / "snapshots", self.t)

        self.timer.stop("snapshot")

    def take_minisnapshot(self):
        """
        Log and time minisnapshot data.
        """
        self.timer.start("minisnapshot")

        data = self.prepare_minisnapshot_data()
        for key, value in data.items():
            self.minisnapshots[key].append(value)

        self.timer.stop("minisnapshot")

    def postprocess_snapshots(self):
        """
        Write the minisnapshots and snapshot index to the snapshot path if not None.
        """
        if self.path is None:
            return

        # pickle minisnapshots lists
        with open(self.path / "snapshots" / "minisnapshots.pkl", "wb") as f:
            pickle.dump(self.minisnapshots, f)

        # write snapshot index as csv
        df = pd.DataFrame(
            [{"index": i, "t": t} for i, t in self.snapshots.file_index.items()]
        )
        df.to_csv(self.path / "snapshots" / "index.csv", index=False)

    def reset_global_logs(self):
        """
        Reset global logs that are incremented throughout the simulation.
        """
        self.n_steps = 0

    def increment_global_logs(self):
        """
        Increment global logs throughout the simulation.
        """
        self.n_steps += 1

    def reset_stepwise_logs(self):
        """
        Reset logs that are incremented at the end of each step.
        """
        self.n_dt_revisions = 0

    def increment_stepwise_logs(self):
        """
        Increment logs at the end of each step.
        """
        pass

    def reset_substepwise_logs(self):
        """
        Reset logs that are incremented at the end of each substep.
        """
        self.n_substeps = 0

    def increment_substepwise_logs(self):
        """
        Increment logs at the end of each substep.
        """
        self.n_substeps += 1

    def euler(self, *args, **kwargs) -> None:
        self.integrator = "euler"
        self.stepper = self._euler_step
        self.integrate(*args, **kwargs)

    def _euler_step(self, t: float, u: ArrayLike, dt: float):
        unew = self.arrays["unew"]
        k0 = self.arrays["k0"]
        self.substep_dt = dt

        # stage 1
        k0[...] = self.f(t, u)
        unew[...] = u + dt * k0
        self.increment_substepwise_logs()

    def ssprk2(self, *args, **kwargs) -> None:
        self.integrator = "ssprk2"
        self.stepper = self._ssprk2_step
        self.integrate(*args, **kwargs)

    def _ssprk2_step(self, t: float, u: ArrayLike, dt: float):
        unew = self.arrays["unew"]
        k0 = self.arrays["k0"]
        k1 = self.arrays["k1"]
        self.substep_dt = dt

        # stage 1
        k0[...] = self.f(t, u)
        unew[...] = u + dt * k0
        self.increment_substepwise_logs()

        # stage 2
        k1[...] = self.f(t + dt, unew)
        unew[...] = 0.5 * u + 0.5 * (unew + dt * k1)
        self.increment_substepwise_logs()

    def ssprk3(self, *args, **kwargs) -> None:
        self.integrator = "ssprk3"
        self.stepper = self._ssprk3_step
        self.integrate(*args, **kwargs)

    def _ssprk3_step(self, t: float, u: ArrayLike, dt: float):
        unew = self.arrays["unew"]
        k0 = self.arrays["k0"]
        k1 = self.arrays["k1"]
        k2 = self.arrays["k2"]
        self.substep_dt = dt

        # stage 1
        k0[...] = self.f(t, u)
        unew[...] = u + dt * k0
        self.increment_substepwise_logs()

        # stage 2
        k1[...] = self.f(t + dt, unew)
        self.increment_substepwise_logs()

        # stage 3
        k2[...] = self.f(t + 0.5 * dt, u + 0.25 * dt * k0 + 0.25 * dt * k1)
        unew[...] = u + (1 / 6) * dt * (k0 + k1 + 4 * k2)
        self.increment_substepwise_logs()

    def rk4(self, *args, **kwargs):
        self.integrator = "rk4"
        self.stepper = self._rk4_step
        self.integrate(*args, **kwargs)

    def _rk4_step(self, t: float, u: ArrayLike, dt: float):
        unew = self.arrays["unew"]
        k0 = self.arrays["k0"]
        k1 = self.arrays["k1"]
        k2 = self.arrays["k2"]
        k3 = self.arrays["k3"]
        self.substep_dt = dt

        # stage 1
        k0[...] = self.f(t, u)
        self.increment_substepwise_logs()

        # stage 2
        k1[...] = self.f(t + 0.5 * dt, u + 0.5 * dt * k0)
        self.increment_substepwise_logs()

        # stage 3
        k2[...] = self.f(t + 0.5 * dt, u + 0.5 * dt * k1)
        self.increment_substepwise_logs()

        # stage 4
        k3[...] = self.f(t + dt, u + dt * k2)
        unew[...] = u + (1 / 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
        self.increment_substepwise_logs()

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
