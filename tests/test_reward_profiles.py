from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from polybot.env import RewardConfig, summer_1_pace_reward_config
from polybot.training.reward_profiles import RewardProfileStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_custom_reward_profiles_round_trip_every_reward_field(tmp_path) -> None:
    store = RewardProfileStore(tmp_path)
    expected = replace(summer_1_pace_reward_config(), progress_per_m=12.5)
    path = store.save("My Pace Profile", expected)
    assert path.name == "my-pace-profile.json"
    actual = store.load("My Pace Profile")
    assert actual == expected
    assert {field.name for field in fields(RewardConfig)} == set(
        json.loads(path.read_text(encoding="utf-8"))["rewards"]
    )
    assert "My Pace Profile" in store.names()


def test_winter_4_failed_episode_score_increases_decisively_with_progress() -> None:
    config = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards").load(
        "Winter 4 - learning"
    )
    track_length_m = 2_000.0

    def shaped_score(progress_ratio: float, checkpoints: int) -> float:
        progress_m = track_length_m * progress_ratio
        return (
            config.progress_per_m * progress_m
            + config.failure_progress_clawback_per_m * progress_m
            + config.failure_early_penalty * (1.0 - progress_ratio)
            + config.barrier_contact_penalty
            + config.checkpoint_bonus * checkpoints
            + config.checkpoint_fast_bonus * checkpoints
        )

    middle_score = shaped_score(0.541, 3)
    late_score = shaped_score(0.96, 4)

    assert middle_score > 0.0
    assert late_score >= middle_score + 700.0
