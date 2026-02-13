import os
import pickle
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from .tools.device_management import ArrayLike, ArrayManager
from .tools.snapshots import Snapshots
from .tools.timer import StepperTimer, Timer
from .tools.yaml_helper import yaml_dump


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

    def __init__(
        self,
        u0: np.ndarray,
        array_manager: Optional[ArrayManager] = None,
        dt_min: float = 1e-15,
    ):
        """
        Initializes the ODE solver.

        Args:
            u0: Initial state as an array.
            array_manager: Optional ArrayManager instance to manage arrays.
            dt_min: Minimum allowable time-step size.
        """
        # initialize time values
        self.t = 0.0
        self.dt = 0.0
        self.dt_min = dt_min

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
        self.wall_timer = Timer(["wall"])
        self.stepper_timer = StepperTimer(["take_step", "snapshot", "minisnapshot"])

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
        self.validate_dt(dt)

        while True:
            self.reset_substepwise_logs()
            self.stepper(t, u, dt)  # revises self.arrays["unew"]
            if self.dt_criterion(t + dt, self.arrays["unew"]):
                break

            dt = clamp_dt(t, self.compute_revised_dt(t, u, dt), target_time)
            self.n_dt_revisions += 1
            self.validate_dt(dt, revising=True)

        # update attributes
        self.arrays["u"][...] = self.arrays["unew"]
        self.t += dt
        self.dt = dt

        self.called_at_end_of_step()

    def validate_dt(self, dt: float, revising: bool = False):
        """
        Validate the computed time-step size.

        Args:
            dt: Time-step size to validate.
        """
        if dt < self.dt_min:
            msg = f"Computed time-step size smaller than {self.dt_min}"
            if revising:
                msg += f" after {self.n_dt_revisions} revisions."
            else:
                msg += "."
            raise RuntimeError(msg)

        if np.isnan(dt):
            msg = "Computed NaN time-step size"
            if revising:
                msg += f" after {self.n_dt_revisions} revisions."
            else:
                msg += "."
            raise RuntimeError(msg)

    def called_at_beginning_of_step(self):
        """
        Helper function called at the beginning of each step starting with a timer
        start.
        """
        self.reset_stepwise_logs()
        self.stepper_timer.begin_new_step()
        self.stepper_timer.start("take_step")

    def called_at_end_of_step(self):
        """
        Helper function called at the end of each step ending with a timer stop.
        """
        self.stepper_timer.stop("take_step")
        self.increment_stepwise_logs()
        self.increment_global_logs()

    def integrate(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time.

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.wall_timer.start("wall")

        # print initial message
        if verbose:
            status_print(self.build_opening_message())

        try:
            self.prepare_output_directory(path, overwrite, discard)

            if snapshot_mode not in ("target", "none", "every"):
                raise ValueError(
                    f"Invalid snapshot_mode '{snapshot_mode}'. "
                    "Must be 'target', 'none', or 'every'."
                )

            if T is None and n is not None:
                self._integrate_for_fixed_number_of_steps(
                    n=n,
                    snapshot_mode=snapshot_mode,
                    verbose=verbose,
                    log_freq=log_freq,
                )
            elif T is not None and n is None:
                self._integrate_until_target_time_is_reached(
                    T=T,
                    snapshot_mode=snapshot_mode,
                    allow_overshoot=allow_overshoot,
                    verbose=verbose,
                    log_freq=log_freq,
                    max_steps=max_steps,
                )
            else:
                raise ValueError("Specify exactly one of 'T' or 'n'.")
        finally:
            self.wall_timer.stop_all()

        # print closing message
        if verbose:
            status_print(self.build_closing_message(), closing=True)

    def _integrate_for_fixed_number_of_steps(
        self,
        n: int,
        snapshot_mode: Literal["target", "none", "every"],
        verbose: bool,
        log_freq: int,
    ):
        """
        Integrate the ODE system for a fixed number of steps.

        Args:
            n: Number of steps to take.
            snapshot_mode: When to take snapshots.
                - "target" (default): at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
        """
        # write run args to file
        self.write_run_args(
            n=n,
            snapshot_mode=snapshot_mode,
            verbose=verbose,
            log_freq=log_freq,
        )

        # take initial snapshots
        if self.t not in self.minisnapshots["t"]:
            if snapshot_mode != "none":
                self.take_snapshot()
            self.take_minisnapshot()

        # simulation loop
        for i in range(1, n + 1):
            self.take_step()
            self.take_minisnapshot()

            # take snapshot
            if (snapshot_mode == "target" and i == n) or snapshot_mode == "every":
                self.take_snapshot()

            # update printed message
            if verbose:
                if self.n_steps % log_freq == 0 or self.n_steps >= n:
                    status_print(self.build_update_message())

        # wrap up snapshots
        self.postprocess_snapshots()

    def _integrate_until_target_time_is_reached(
        self,
        T: Union[float, List[float]],
        snapshot_mode: Literal["target", "none", "every"],
        allow_overshoot: bool,
        verbose: bool,
        log_freq: int,
        max_steps: Optional[int] = None,
    ):
        """
        Integrate the ODE until a target time is reached.

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
            snapshot_mode: When to take snapshots.
                - "target" (default): at t=0 and each target time. If multiple target
                    times are crossed in a single step, only one snapshot is taken
                    at the end of the step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
        """
        # write run args to file
        self.write_run_args(
            T=T,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
        )

        # format target times
        target_times = self._get_target_time_list(T)
        T_max = max(target_times)
        target_time = target_times.pop(0)

        # initial snapshot
        if self.t not in self.minisnapshots["t"]:
            if snapshot_mode != "none":
                self.take_snapshot()
            self.take_minisnapshot()

        # simulation loop
        while self.t < T_max:
            self.take_step(target_time=None if allow_overshoot else target_time)
            self.take_minisnapshot()

            # count how many target times we crossed on this step
            crossed = 0
            while self.t >= target_time:
                crossed += 1
                if target_times:
                    target_time = target_times.pop(0)
                else:
                    break

            # determine if we reached the max number of steps
            reached_max_steps = max_steps is not None and self.n_steps >= max_steps

            # take at most one snapshot this step
            if snapshot_mode == "every":
                self.take_snapshot()
            elif snapshot_mode == "target" and crossed > 0:
                self.take_snapshot()
            elif reached_max_steps:
                self.take_snapshot()

            # update printed message
            if verbose:
                closing_message = self.t >= T_max or reached_max_steps
                if self.n_steps % log_freq == 0 or closing_message:
                    status_print(self.build_update_message())

            if reached_max_steps:
                warnings.warn("Solver reached maximum number of steps.")
                break

        # wrap up snapshots
        self.postprocess_snapshots()

    def _get_target_time_list(self, T: Union[float, List[float]]) -> List[float]:
        """
        Format the target time(s) into a sorted list.

        Args:
            T: Target simulation time(s).
                - single float: Integrate until this time.
                - list of floats: integrate until each listed time.

        Returns:
            Sorted list of target times.
        """
        if isinstance(T, int) or isinstance(T, float):
            target_times = [T]
        elif isinstance(T, list):
            target_times = sorted([float(t) for t in T])
        else:
            raise ValueError(f"Invalid type for T: {type(T)}")
        if min(target_times) <= 0:
            raise ValueError("Target times must be greater than 0.")
        return target_times

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
            "stepper_timer": (
                self.stepper_timer.steps[-1]
                if self.stepper_timer.steps
                else Timer(self.stepper_timer.cats)
            ),
        }

    def prepare_output_directory(
        self, path: Optional[str] = None, overwrite: bool = False, discard: bool = True
    ):
        """
        Create output directory if it doesn't exist and throw an error or overwrite it
        if it does.

        Args:
            path: Output path as a string.
            overwrite: Whether to completely delete the path if it exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        if path is None:
            return None

        out_path = Path(path)

        if out_path.exists() and overwrite:
            print(f'Overwriting existing output directory "{out_path}".')
            shutil.rmtree(out_path)
        elif out_path.exists() and not overwrite:
            raise FileExistsError(f"Output directory '{out_path}' already exists.")

        os.makedirs(out_path)
        os.makedirs(out_path / "snapshots")

        self.path = out_path
        self.discard = discard

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

    def write_run_args(self, **kwargs):
        """
        Write the arguments passed to the `run` method to a yaml file.

        Args:
            **kwargs: Arguments passed to the `run` method.
        """
        if self.path is None:
            return
        with open(self.path / "run.yaml", "w") as f:
            f.write(yaml_dump(kwargs))

    def take_snapshot(self):
        """
        Log and time snapshot data at time `self.t` and write it to `self.path` if not None.
        """
        self.stepper_timer.start("snapshot")

        data = self.prepare_snapshot_data()
        self.snapshots.log(self.t, data)

        if self.path is not None:
            self.snapshots.write(self.path / "snapshots", self.t, discard=self.discard)

        self.stepper_timer.stop("snapshot")

    def take_minisnapshot(self):
        """
        Log and time minisnapshot data.
        """
        self.stepper_timer.start("minisnapshot")

        data = self.prepare_minisnapshot_data()
        for key, value in data.items():
            self.minisnapshots[key].append(value)

        self.stepper_timer.stop("minisnapshot")

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

        if len(self.stepper_timer.steps) != self.n_steps + 1:
            raise RuntimeError(
                f"Number of steps in timer ({len(self.stepper_timer.steps)}) is not "
                f"one greater than n_steps ({self.n_steps})."
            )

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

    def euler(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time using the forward Euler method.

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.integrator = "euler"
        self.stepper = self._euler_step
        self.integrate(
            T=T,
            n=n,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
            max_steps=max_steps,
            path=path,
            overwrite=overwrite,
            discard=discard,
        )

    def _euler_step(self, t: float, u: ArrayLike, dt: float):
        unew = self.arrays["unew"]
        k0 = self.arrays["k0"]
        self.substep_dt = dt

        # stage 1
        k0[...] = self.f(t, u)
        unew[...] = u + dt * k0
        self.increment_substepwise_logs()

    def ssprk2(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time using the second-order
        Strong Stability Preserving Runge-Kutta method (SSPRK2).

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.integrator = "ssprk2"
        self.stepper = self._ssprk2_step
        self.integrate(
            T=T,
            n=n,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
            max_steps=max_steps,
            path=path,
            overwrite=overwrite,
            discard=discard,
        )

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

    def ssprk3(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time using the third-order
        Strong Stability Preserving Runge-Kutta method (SSPRK3).

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.integrator = "ssprk3"
        self.stepper = self._ssprk3_step
        self.integrate(
            T=T,
            n=n,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
            max_steps=max_steps,
            path=path,
            overwrite=overwrite,
            discard=discard,
        )

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

    def rk4(
        self,
        T: Optional[Union[float, List[float]]] = None,
        n: Optional[int] = None,
        snapshot_mode: Literal["target", "none", "every"] = "target",
        allow_overshoot: bool = False,
        verbose: bool = True,
        log_freq: int = 100,
        max_steps: Optional[int] = None,
        path: Optional[str] = None,
        overwrite: bool = False,
        discard: bool = True,
    ):
        """
        Integrate the ODE system forward in time using the fourth-order Runge-Kutta
        method (RK4).

        Args:
            T: Target simulation time(s).
                - Single float: Integrate until this time.
                - List of floats: Integrate until each listed time.
                - None: `n` must be specified instead.
            n: Number of steps to take. If None, `T` must be specified instead.
            snapshot_mode: When to take snapshots.
                - "target" (default):
                    * If `T` is given: at t=0 and each target time. If multiple target
                        times are crossed in a single step, only one snapshot is taken
                        at the end of the step.
                    * If `n` is given: at t=0 and the nth step.
                - "none": no snapshots.
                - "every": at t=0 and every step.
            allow_overshoot: If True, the solver may overshoot target times
                instead of shortening the last step to hit them exactly.
            verbose: Whether to print progress information.
            log_freq: Step interval between log updates (if verbose).
            max_steps: If provided, sets the maximum number of steps the solver is
                permitted to take. If the solver reaches this step count, it will stop,
                raising a warning.
            path: Directory to write snapshots. If None, snapshots are not written.
            overwrite: Whether to overwrite `path` if it already exists.
            discard: If True, discard the in-memory snapshot data after writing to
                disk.
        """
        self.integrator = "rk4"
        self.stepper = self._rk4_step
        self.integrate(
            T=T,
            n=n,
            snapshot_mode=snapshot_mode,
            allow_overshoot=allow_overshoot,
            verbose=verbose,
            log_freq=log_freq,
            max_steps=max_steps,
            path=path,
            overwrite=overwrite,
            discard=discard,
        )

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
