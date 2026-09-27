"""Explicit curriculum phases whose steps sum to the requested budget."""

from __future__ import annotations

from dataclasses import dataclass

from polybot.training.config import CurriculumConfig


@dataclass(frozen=True, slots=True)
class CurriculumPhase:
    mode: str
    steps: int
    spawn_ratio: float | None = None
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    lead_in_ratio: float = 0.05

    def env_kwargs(self) -> dict[str, object]:
        return {
            "curriculum_spawn_ratio": self.spawn_ratio,
            "curriculum_lead_in_ratio": self.lead_in_ratio,
            "curriculum_start_ratio": self.start_ratio,
            "curriculum_end_ratio": self.end_ratio,
            "curriculum_start_s": self.start_s,
            "curriculum_end_s": self.end_s,
            "curriculum_random_quarters": self.mode == "quarters-randomised",
        }


@dataclass(frozen=True, slots=True)
class CurriculumPlan:
    phases: tuple[CurriculumPhase, ...]

    @property
    def total_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)


def build_plan(config: CurriculumConfig, budget: int) -> CurriculumPlan:
    if budget < 1:
        raise ValueError("training budget must be positive")
    if config.mode == "custom":
        phases = tuple(CurriculumPhase(
            phase.mode, phase.steps,
            max(0.0, phase.start_ratio - phase.lead_in_ratio)
            if phase.start_ratio is not None else None,
            phase.start_ratio, phase.end_ratio, phase.start_s, phase.end_s,
            phase.lead_in_ratio,
        ) for phase in config.phases)
        plan = CurriculumPlan(phases)
        if plan.total_steps != budget:
            raise ValueError("custom phase steps must sum to the total training budget")
        return plan
    if config.mode == "quarters":
        sections = [("section", i / 4, (i + 1) / 4) for i in range(4)]
        sections.append(("full", None, None))
    elif config.mode == "q4-full":
        sections = [("section", 0.75, 1.0), ("full", None, None)]
    else:
        sections = [(config.mode, config.start_ratio, config.end_ratio)]
    if budget < len(sections):
        raise ValueError("budget is smaller than the number of curriculum phases")
    each, remainder = divmod(budget, len(sections))
    phases = tuple(
        CurriculumPhase(
            mode, each + int(index < remainder),
            max(0.0, start - config.lead_in_ratio) if start is not None else None,
            start, end,
            config.start_s if mode == "timed" else None,
            config.end_s if mode == "timed" else None,
            config.lead_in_ratio,
        )
        for index, (mode, start, end) in enumerate(sections)
    )
    return CurriculumPlan(phases)
