"""Append-only record of accepted pace champions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polybot.models.registry import git_commit


def append_pace_history(
    champion_dir: Path, *, source: str, evaluation: Any, model: Any,
    reward_profile: str | None, air_brake_bonus_per_s: float,
    learning_rate: float | None,
) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "timesteps": int(model.num_timesteps),
        "median_lap_s": evaluation.median_lap_s,
        "best_lap_s": evaluation.best_lap_s,
        "finish_rate": evaluation.finish_rate,
        "learning_rate": learning_rate,
        "action_drift": getattr(model, "_anchor_action_drift", None),
        "reward_profile": reward_profile,
        "speed_bias_schedule": getattr(model, "speed_bias_schedule", []),
        "air_brake_bonus_per_s": air_brake_bonus_per_s,
        "air_brake_time_s": evaluation.air_brake_time_s,
        "air_brake_fraction": evaluation.air_brake_fraction,
        "git_commit": git_commit(),
    }
    path = champion_dir.parent / "pace-history.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
