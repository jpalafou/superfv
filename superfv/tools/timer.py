import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class CatTimer:
    is_timing: bool = False
    start_time: float = 0.0
    cum_time: float = 0.0
    n_calls: int = 0

    def __eq__(self, other):
        if not isinstance(other, CatTimer):
            return NotImplemented
        return (
            self.is_timing == other.is_timing
            and self.start_time == other.start_time
            and self.cum_time == other.cum_time
            and self.n_calls == other.n_calls
        )


class Timer:

    def __init__(self, cats: List[str]):
        self.data = {cat: CatTimer() for cat in cats}

    def _check_cat(self, cat: str):
        if cat not in self.data:
            raise ValueError(f"Could not find timer category '{cat}'")

    def start(self, cat: str):
        self._check_cat(cat)
        if self.data[cat].is_timing:
            raise ValueError("Cannot start an aready-running timer category.")

        self.data[cat].is_timing = True
        self.data[cat].start_time = time.perf_counter()

    def stop(self, cat: str):
        self._check_cat(cat)
        if not self.data[cat].is_timing:
            raise ValueError("Cannot stop a timer category that is not active.")

        self.data[cat].is_timing = False
        self.data[cat].cum_time += time.perf_counter() - self.data[cat].start_time
        self.data[cat].n_calls += 1

    def stop_all(self):
        for cat in self.data:
            if self.data[cat].is_timing:
                self.stop(cat)

    def __eq__(self, other):
        if not isinstance(other, Timer):
            return NotImplemented
        return self.data == other.data


class StepperTimer:
    def __init__(self, cats: List[str]):
        self.cats: List[str] = cats
        self.steps: List[Timer] = []
        self.begin_new_step()  # init 0th step

    def begin_new_step(self):
        self.steps.append(Timer(self.cats))

    def start(self, cat: str):
        self.steps[-1].start(cat)

    def stop(self, cat: str):
        self.steps[-1].stop(cat)

    def stop_all(self):
        self.steps[-1].stop_all()

    def cum_time_list(self, cat: str) -> List[float]:
        return [timer.data[cat].cum_time for timer in self.steps]

    def ncalls_list(self, cat: str) -> List[int]:
        return [timer.data[cat].n_calls for timer in self.steps]

    def total_time(self, cat: str):
        return sum(self.cum_time_list(cat))

    def total_calls(self, cat: str):
        return sum(self.ncalls_list(cat))

    def to_dict(self, decimals: int = 2) -> dict:
        out = {cat: np.round(self.total_time(cat), decimals) for cat in self.cats}
        return out

    def print_report(self, precision: int = 2):
        # Sort the categories alphabetically
        sorted_cats = sorted(self.cats)

        # Fixed width for category column
        cat_width = (
            max(len(cat) for cat in sorted_cats) + 5
        )  # Slightly more for padding

        # Start building the report string
        report_str = (
            f"{'Category':<{cat_width}} {'Calls':>10} {'Cumulative Time':>20}\n"
        )
        report_str += "-" * (cat_width + 34) + "\n"

        # Add the data for each category
        for cat in sorted_cats:
            ncalls = self.total_calls(cat)
            cumtime = self.total_time(cat)
            report_str += (
                f"{cat:<{cat_width}} {ncalls:>10} {cumtime:>20.{precision}f}\n"
            )

        print(report_str)

    def __contains__(self, cat: str) -> bool:
        return cat in self.cats

    def __eq__(self, other):
        if not isinstance(other, StepperTimer):
            return NotImplemented
        return self.cats == other.cats and self.steps == other.steps


class MethodTimer:
    """
    Decorator for timing methods in a class with a Timer instance.
    """

    def __init__(self, cat):
        self.cat = cat

    def __call__(self, method):
        def wrapped(instance, *args, **kwargs):
            sync = instance.cupy and instance.sync_timing
            if sync:
                instance.xp.cuda.Device().synchronize()

            instance.stepper_timer.start(self.cat)
            result = method(instance, *args, **kwargs)

            if sync:
                instance.xp.cuda.Device().synchronize()
            instance.stepper_timer.stop(self.cat)
            return result

        return wrapped
