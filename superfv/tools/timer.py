import time
from typing import Dict, Iterable, Set, cast

import numpy as np

from .device_management import xp


class Timer:
    """
    Timer class for timing code execution.
    """

    def __init__(
        self,
        cats: Iterable[str] = ("main",),
    ):
        """
        Initializes the timer.

        Args:
            cats: Iterable of category names to initialize.
            precision: Precision of the time values.
        """
        # Public
        self.cats: Set[str] = set()
        self.goldfish_lap_time: Dict[str, float] = {}
        self.cum_time: Dict[str, float] = {}
        self.n_calls: Dict[str, int] = {}

        # Private
        self._running: Dict[str, bool] = {}
        self._start_time: Dict[str, float] = {}

        for cat in cats:
            self._add_cat(cat)

    def _add_cat(self, cat: str):
        """
        Internal method to add a new timer category without checks.

        Args:
            cat: Category name.
        """
        # Public
        self.cats.add(cat)
        self.goldfish_lap_time[cat] = np.nan
        self.cum_time[cat] = 0.0
        self.n_calls[cat] = 0

        # Private
        self._running[cat] = False
        self._start_time[cat] = np.nan

    def add_cat(self, cat: str):
        """
        Add a new timer category.

        Args:
            cat: Category name.
        """
        if cat in self.cats:
            raise ValueError(f"Category '{cat}' already exists.")
        self._add_cat(cat)

    def _check_cat_existence(self, cat: str):
        if cat not in self.cats:
            raise ValueError(f"Category '{cat}' not found in timer categories.")

    def start(self, cat: str, reset: bool = False):
        """
        Start timer of a category.

        Args:
            cat: Category name.
            reset: If True, reset the timer for this category.
            same_call: If True, do not increment the call count.
        """
        self._check_cat_existence(cat)
        if self._running[cat]:
            raise RuntimeError(
                f"Cannot start '{cat}' timer since it is already in progress."
            )
        else:
            if reset:
                self.n_calls[cat] = 0
                self.cum_time[cat] = 0.0

            self._running[cat] = True
            self._start_time[cat] = time.time()
            self.n_calls[cat] += 1

    def stop(self, cat: str):
        """
        Stop timer of a category.

        Args:
            cat: Category name.
        """
        self._check_cat_existence(cat)
        if self._running[cat]:
            lap_time = time.time() - cast(float, self._start_time[cat])
            self.goldfish_lap_time[cat] = lap_time
            self.cum_time[cat] += lap_time
            self._running[cat] = False
            self._start_time[cat] = np.nan
        else:
            raise RuntimeError(
                f"Cannot stop '{cat}' timer since it is not in progress."
            )

    def stop_all(self):
        """
        Stop all running timers.
        """
        for cat in self.cats:
            if self._running[cat]:
                self.stop(cat)

    def to_dict(self, decimals: int = 2) -> dict:
        """
        Return the cumulative times for all categories as a dictionary.

        Args:
            decimals: Number of decimal places to round to.
        """
        out = {cat: np.round(t, decimals) for cat, t in self.cum_time.items()}
        return out

    def print_report(self, precision: int = 2):
        """
        Prints a report of the timer categories with ncalls and cumtime.

        Args:
            precision: Number of decimal places to print for cumulative time.

        Returns:
            A string containing the report formatted as a table.
        """
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
            ncalls = self.n_calls[cat]
            cumtime = self.cum_time[cat]
            report_str += (
                f"{cat:<{cat_width}} {ncalls:>10} {cumtime:>20.{precision}f}\n"
            )

        print(report_str)

    def __contains__(self, cat: str) -> bool:
        return cat in self.cats


class MethodTimer:
    """
    Decorator for timing methods in a class with a Timer instance.
    """

    def __init__(self, cat):
        self.cat = cat

    def __call__(self, method):
        def wrapped(instance, *args, **kwargs):
            instance.timer.start(self.cat)
            result = method(instance, *args, **kwargs)
            if instance.profile and instance.cupy:
                xp.cuda.Device().synchronize()
            instance.timer.stop(self.cat)
            return result

        return wrapped
