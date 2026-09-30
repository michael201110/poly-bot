from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from pathlib import Path

from polybot.environment.rewards import RewardConfig, summer_1_pace_reward_config
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


def test_winter_4_19s_pace_profile_targets_fast_complete_laps() -> None:
    store = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards")
    config = store.load("Winter 4 - 19s pace")

    assert "Winter 4 - 19s pace" in store.names()
    assert config.finish_target_s == 19.0
    assert config.finish_fast_bonus == 5000.0
    assert config.finish_pace_decay_per_s == 0.4
    assert config.checkpoint_target_s == 5.0
    assert config.checkpoint_fast_bonus == 250.0
    assert config.failure_progress_clawback_per_m == -2.0
    assert config.failure_early_penalty == -1000.0


def test_winter_4_conservative_pace_changes_only_finish_shaping() -> None:
    store = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards")
    baseline = store.load("Winter 4 - learning")
    conservative = store.load("Winter 4 - 19s conservative")

    assert "Winter 4 - 19s conservative" in store.names()
    assert conservative.finish_target_s == 19.0
    assert conservative.finish_fast_bonus == 3000.0
    for field in fields(RewardConfig):
        if field.name not in {"finish_target_s", "finish_fast_bonus"}:
            assert getattr(conservative, field.name) == getattr(baseline, field.name)


def test_summer_1_20s_pace_rewards_faster_finishes_and_claws_back_failures() -> None:
    config = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards").load(
        "Summer 1 - 20s pace"
    )
    at_29 = config.finish_bonus + config.finish_fast_bonus * math.exp(
        -config.finish_pace_decay_per_s * (29 - config.finish_target_s)
    )
    at_20 = config.finish_bonus + config.finish_fast_bonus
    assert config.finish_target_s == 20.0
    assert at_20 > at_29 + 2_900
    assert config.failure_progress_clawback_per_m == -config.progress_per_m
    assert config.failure_early_penalty < 0


def test_summer_1_spin_control_profile_targets_grounded_yaw_spins() -> None:
    config = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards").load(
        "Summer 1 - 20s pace spin control"
    )

    assert config.finish_target_s == 20.0
    assert config.ground_spin_deadzone_radps == 5.0
    assert config.ground_spin_penalty_per_rad_s == -12.0
    assert config.ground_spin_min_grounded_wheels == 2
    assert config.barrier_collision_impulse_threshold == 1e9


def test_summer_1_impact_pace_profile_keeps_contacts_nonterminal() -> None:
    config = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards").load(
        "Summer 1 - 20s pace impact"
    )
    assert config.barrier_collision_impulse_threshold == 0.0
    assert config.barrier_contact_penalty == -1000.0
    assert config.finish_target_s == 20.0
