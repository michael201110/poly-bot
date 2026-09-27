"""Deterministic full-track evaluation and champion ranking."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    finish_rate: float
    median_progress: float
    mean_progress: float
    best_lap_s: float | None
    median_lap_s: float | None
    crash_rate: float
    off_track_rate: float
    stall_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rank(self) -> tuple[float, ...]:
        return (
            self.finish_rate,
            self.median_progress,
            self.mean_progress,
            -(self.median_lap_s if self.median_lap_s is not None else float("inf")),
            -self.crash_rate,
            -self.off_track_rate,
            -self.stall_rate,
        )


def evaluate_model(
    model: Any, env_factory: Callable[[], Any], *, episodes: int, seed: int
) -> EvaluationResult:
    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    progress: list[float] = []
    laps: list[float] = []
    crashes = off_tracks = stalls = 0
    env = env_factory()
    model.policy.set_training_mode(False)
    try:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            events = set(info.get("events", ()))
            progress.append(min(1.0, max(0.0, info["route_progress_m"] / info["track_length_m"])))
            if "finish" in events:
                laps.append(float(info["elapsed_s"]))
            crashes += int("crash" in events or "barrier_contact" in events)
            off_tracks += int("off_track" in events)
            stalls += int("stalled" in events)
    finally:
        env.close()
        model.policy.set_training_mode(True)
    return EvaluationResult(
        episodes, len(laps) / episodes, statistics.median(progress),
        statistics.fmean(progress), min(laps) if laps else None,
        statistics.median(laps) if laps else None,
        crashes / episodes, off_tracks / episodes, stalls / episodes,
    )
