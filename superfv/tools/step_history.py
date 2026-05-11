from dataclasses import dataclass
from typing import Any, List


@dataclass
class SubstepSummary:
    substep: int
    t_wall: float


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

    def get_substep_history_as_list(self, substep_attr: str) -> List[Any]:
        out = []
        for step in self.steps:
            for substep in step.substeps:
                out.append(getattr(substep, substep_attr))
        return out

    def __getitem__(self, i: int) -> StepSummary:
        return self.steps[i]

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def append(self, step_summary: StepSummary):
        self.steps.append(step_summary)
