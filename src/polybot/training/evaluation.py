"""Model evaluation records used for deliberate promotion decisions."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    laps_requested: int
    lap_times_s: tuple[float, ...]
    crashes: int

    @property
    def finishes(self) -> int:
        return len(self.lap_times_s)

    @property
    def finish_rate(self) -> float:
        return self.finishes / self.laps_requested if self.laps_requested else 0.0

    @property
    def best_lap_s(self) -> float | None:
        return min(self.lap_times_s, default=None)

    @property
    def mean_lap_s(self) -> float | None:
        return statistics.fmean(self.lap_times_s) if self.lap_times_s else None

    @property
    def median_lap_s(self) -> float | None:
        return statistics.median(self.lap_times_s) if self.lap_times_s else None
