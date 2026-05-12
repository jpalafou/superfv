from dataclasses import dataclass
from typing import Any, List


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
