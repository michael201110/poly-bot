"""Deterministic full-track evaluation and champion ranking."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
    airborne_time_s: float = 0.0
    air_brake_time_s: float = 0.0
    air_brake_fraction: float = 0.0
    air_brake_reward: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rank(self) -> tuple[float, ...]:
        reliable = self.finish_rate == 1.0 and self.median_progress == 1.0
        return (
            self.finish_rate,
            self.median_progress,
            -(self.median_lap_s if self.median_lap_s is not None else float("inf"))
            if reliable else self.mean_progress,
            self.mean_progress if reliable else
            -(self.median_lap_s if self.median_lap_s is not None else float("inf")),
            -self.crash_rate,
            -self.off_track_rate,
            -self.stall_rate,
        )


def evaluate_model(
    model: Any, env_factory: Callable[[], Any], *, episodes: int, seed: int,
    observation_sink: list[np.ndarray] | None = None,
) -> EvaluationResult:
    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    progress: list[float] = []
    laps: list[float] = []
    crashes = off_tracks = stalls = 0
    airborne_time = air_brake_time = air_brake_reward = 0.0
    env = env_factory()
    model.policy.set_training_mode(False)
    try:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            while True:
                if observation_sink is not None:
                    observation_sink.append(np.array(observation, copy=True))
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            events = set(info.get("events", ()))
            progress.append(min(1.0, max(0.0, info["route_progress_m"] / info["track_length_m"])))
            if "finish" in events:
                laps.append(float(info["elapsed_s"]))
            crashes += int(
                "crash" in events or "barrier_contact" in events
                or "airborne_roll_failure" in events
            )
            off_tracks += int("off_track" in events)
            stalls += int("stalled" in events)
            summary = info.get("air_brake_summary", {})
            airborne_time += float(summary.get("airborne_time_s", 0.0))
            air_brake_time += float(summary.get("air_brake_time_s", 0.0))
            air_brake_reward += float(summary.get("air_brake_reward", 0.0))
    finally:
        env.close()
        model.policy.set_training_mode(True)
    return EvaluationResult(
        episodes, len(laps) / episodes, statistics.median(progress),
        statistics.fmean(progress), min(laps) if laps else None,
        statistics.median(laps) if laps else None,
        crashes / episodes, off_tracks / episodes, stalls / episodes,
        airborne_time / episodes, air_brake_time / episodes,
        air_brake_time / airborne_time if airborne_time else 0.0,
        air_brake_reward / episodes,
    )
