import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from superfv.tools.device_management import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


@dataclass
class Timer:
    is_timing: bool = False
    start_time: float = np.nan
    cum_time: float = 0.0
    n_calls: int = 0

    def start(self, cuda_sync: bool):
        if self.is_timing:
            raise ValueError("Cannot start an aready-running timer.")

        if CUPY_AVAILABLE and cuda_sync:
            cp.cuda.Device().synchronize()

        self.is_timing = True
        self.start_time = time.perf_counter()
        self.n_calls += 1

    def stop(self, cuda_sync: bool):
        if not self.is_timing:
            raise ValueError("Cannot stop a timer that is not active.")

        if CUPY_AVAILABLE and cuda_sync:
            cp.cuda.Device().synchronize()

        self.is_timing = False
        self.cum_time += time.perf_counter() - self.start_time
        self.start_time = np.nan

    def copy(self) -> "Timer":
        return Timer(
            is_timing=self.is_timing,
            start_time=self.start_time,
            cum_time=self.cum_time,
            n_calls=self.n_calls,
        )


class MultiTimer:
    def __init__(self, cats: List[str]):
        self.timers = {cat: Timer() for cat in cats}

    def __getitem__(self, cat: str) -> Timer:
        return self.timers[cat]

    def __str__(self) -> str:
        return (
            "MultiTimer("
            + ", ".join(
                [
                    f"{cat}: {timer.cum_time:.2e}s over {timer.n_calls} call{'s' if timer.n_calls != 1 else ''}"
                    for cat, timer in self.timers.items()
                ]
            )
            + ")"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def start(self, cat: str, cuda_sync: bool):
        if cat not in self.timers:
            raise ValueError(f"Could not find timer category '{cat}'")
        self.timers[cat].start(cuda_sync)

    def stop(self, cat: str, cuda_sync: bool):
        if cat not in self.timers:
            raise ValueError(f"Could not find timer category '{cat}'")
        self.timers[cat].stop(cuda_sync)

    def copy(self) -> "MultiTimer":
        out = MultiTimer(list(self.timers.keys()))
        for cat in self.timers:
            out.timers[cat] = self.timers[cat].copy()
        return out


@dataclass
class SubstepSummary:
    substep: int
    t_wall: float
    n_MOOD_revisions: int
    n_troubles_hist: List[int]


@dataclass
class StepSummary:
    step: int
    t_sim: float
    t_wall: float
    n_dt_revisions: int
    rho_min: float
    E_total: float
    substeps: List[SubstepSummary]
    timer: MultiTimer


@dataclass
class StepHistory:
    steps: List[StepSummary]

    def get_history(self, name: str) -> List[Any]:
        out = []
        for step in self.steps:
            if name in step.__dataclass_fields__:
                out.append(getattr(step, name))
                continue
            for substep in step.substeps:
                if name in substep.__dataclass_fields__:
                    out.append(getattr(substep, name))
                    continue
                raise ValueError(f"Field '{name}' not found in StepSummary or SubstepSummary.")
        return out

    def __getitem__(self, i: int) -> StepSummary:
        return self.steps[i]

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def append(self, step_summary: StepSummary):
        self.steps.append(step_summary)
