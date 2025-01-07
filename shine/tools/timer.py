import time
from typing import Dict, Iterable, Union, cast

import numpy as np


def method_timer(method, cat):
    def wrapper(self, *args, **kwargs):
        self.timer.start(cat)
        result = method(self, *args, **kwargs)
        self.timer.stop(cat)
        return result

    return wrapper


class Timer:
    """
    Timer class for timing code execution.
    """

    def __init__(self, cats: Iterable[str] = ("main",), precision: int = 2):
        """
        Initializes the timer.

        Args:
            cats (Tuple[str, ...]): Tuple of categories.
            precision (int): Precision of the time values.
        """
        self.cats = set(cats)
        self.start_time: Dict[str, Union[None, float]] = {cat: None for cat in cats}
        self.cum_time: Dict[str, float] = {cat: 0.0 for cat in cats}

        self.precision = precision

    def add_cat(self, cat: str):
        """
        Add a new timer category.

        Args:
            cat (str): Category.
        """
        if cat in self.cats:
            raise ValueError(f"Category '{cat}' already exists.")
        self.cats.add(cat)
        self.start_time[cat] = None
        self.cum_time[cat] = 0.0

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
        if self.start_time[cat] is None:
            self.start_time[cat] = time.time()
        else:
            raise RuntimeError(
                f"Cannot start '{cat}' timer since it is already in progress."
            )

    def stop(self, cat: str):
        """
        Stop timer of a category.

        Args:
            cat (str): Category.
        """
        self._check_cat_existence(cat)
        if self.start_time[cat] is None:
            raise RuntimeError(
                f"Cannot stop '{cat}' timer since it is not in progress."
            )
        else:
            self.cum_time[cat] += time.time() - cast(float, self.start_time[cat])
            self.start_time[cat] = None

    def to_dict(self) -> dict:
        """
        Return the cumulative times for all categories as a dictionary.
        """
        out = {cat: np.round(t, self.precision) for cat, t in self.cum_time.items()}
        return out

    def report(self) -> str:
        """
        Return a formatted string report of the cumulative times for all categories
        with dynamic column width based on both category name length and time values.
        """
        # name headers
        cat_header = "Category"
        time_header = "Time (s)"

        # Determine the max length of the category names and the time values
        max_cat_len = (
            max(len(cat) for cat in self.cats) if self.cats else len(cat_header)
        )
        max_time_len = max(
            (
                max(len(f"{t:.{self.precision}f}") for t in self.cum_time.values())
                if self.cum_time
                else len(time_header)
            ),
            len(time_header),
        )

        # Build the report as a string with dynamically sized columns
        report_str = f"{cat_header:<{max_cat_len}} {time_header:<{max_time_len}}\n"
        report_str += "-" * (max_cat_len + max_time_len + 1) + "\n"

        # Add each category and time, formatted to the correct precision and width
        for cat, t in self.cum_time.items():
            time_str = f"{t:.{self.precision}f}"
            report_str += f"{cat:<{max_cat_len}} {time_str:>{max_time_len}}\n"

        return report_str
