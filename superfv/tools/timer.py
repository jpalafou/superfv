import time
from typing import Dict, Iterable, Optional, cast

import numpy as np


def method_timer(method, cat):
    def wrapper(self, *args, **kwargs):
        if cat not in self.timer:
            self.timer.add_cat(cat)
        ALREADY_RUNNING = self.timer._running[cat]
        if not ALREADY_RUNNING:
            self.timer.start(cat)
        result = method(self, *args, **kwargs)
        if not ALREADY_RUNNING:
            self.timer.stop(cat)
        return result

    return wrapper


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
            cats (Tuple[str, ...]): Tuple of categories.
            precision (int): Precision of the time values.
        """
        self.cats = set(cats)
        self._running: Dict[str, bool] = {cat: False for cat in cats}
        self._start_time: Dict[str, Optional[float]] = {cat: None for cat in cats}

        # Outputs
        self.cum_time: Dict[str, float] = {cat: 0.0 for cat in cats}
        self.n_calls: Dict[str, int] = {cat: 0 for cat in cats}

    def add_cat(self, cat: str):
        """
        Add a new timer category.

        Args:
            cat (str): Category.
        """
        if cat in self.cats:
            raise ValueError(f"Category '{cat}' already exists.")
        self.cats.add(cat)
        self._running[cat] = False
        self._start_time[cat] = None

        # Outputs
        self.cum_time[cat] = 0.0
        self.n_calls[cat] = 0

    def _check_cat_existence(self, cat: str):
        if cat not in self.cats:
            raise ValueError(f"Category '{cat}' not found in timer categories.")

    def start(self, cat: str):
        """
        Start timer of a category.

        Args:
            cat (str): Category.
        """
        self._check_cat_existence(cat)
        if self._running[cat]:
            raise RuntimeError(
                f"Cannot start '{cat}' timer since it is already in progress."
            )
        else:
            self._running[cat] = True
            self._start_time[cat] = time.time()
            self.n_calls[cat] += 1

    def stop(self, cat: str):
        """
        Stop timer of a category.

        Args:
            cat (str): Category.
        """
        self._check_cat_existence(cat)
        if self._running[cat]:
            self.cum_time[cat] += time.time() - cast(float, self._start_time[cat])
            self._running[cat] = False
            self._start_time[cat] = None
        else:
            raise RuntimeError(
                f"Cannot stop '{cat}' timer since it is not in progress."
            )

    def to_dict(self, decimals: int = 2) -> dict:
        """
        Return the cumulative times for all categories as a dictionary.

        Args:
            decimals (int): Number of decimal places to round to.
        """
        out = {cat: np.round(t, decimals) for cat, t in self.cum_time.items()}
        return out

    def report(self, precision: int = 2) -> str:
        """
        Generates a report of the timer categories with ncalls and cumtime.

        Args:
            precision (int): Precision of the time values.

        Returns:
            str: A string containing the report formatted as a table.
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

        return report_str

    def __contains__(self, cat: str) -> bool:
        return cat in self.cats
