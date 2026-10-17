from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from polybot.control.actions import ControlDemand
from polybot.environment.env import PolyTrackEnv
from polybot.environment.rewards import RewardConfig, RewardContext, calculate_reward
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Transition
from polybot.training.reward_profiles import RewardProfileStore, pace_failure_x3_reward_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_custom_reward_profiles_round_trip_every_reward_field(tmp_path) -> None:
    store = RewardProfileStore(tmp_path)
    expected = replace(pace_failure_x3_reward_config(), progress_per_m=12.5)
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


def test_generic_pace_profile_has_effective_contact_feedback_without_ghost_or_brake_bias() -> None:
    config = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards").load("Pace")

    assert config.speed_pace_reward_per_m_per_mps == 0.0
    assert config.on_track_speed_per_m == 0.0
    assert config.elapsed_cost_per_s == -50.0
    assert config.progress_per_m == 2.0
    assert config.unsafe_speed_penalty_per_m == 0.0
    assert config.crash_penalty == -1_000.0
    assert config.finish_target_s == 30.0
    assert config.imitation_bonus_per_s == 0.0
    assert config.ground_brake_penalty_per_s == 0.0
    assert config.barrier_contact_penalty == -100.0
    assert config.barrier_speed_loss_penalty_per_mps == -50.0
    assert config.barrier_collision_impulse_threshold == 250.0


def test_pace_failure_x3_profile_only_increases_contact_and_failure_costs() -> None:
    store = RewardProfileStore(PROJECT_ROOT / "profiles" / "rewards")
    pace = store.load("Pace")
    stronger = store.load("Pace + failure x3")

    for field in fields(RewardConfig):
        if field.name not in {
            "barrier_contact_penalty", "barrier_speed_loss_penalty_per_mps", "crash_penalty", "stall_penalty",
            "off_track_penalty", "early_off_track_penalty",
        }:
            assert getattr(stronger, field.name) == getattr(pace, field.name)
    assert stronger.barrier_contact_penalty == -300.0
    assert stronger.barrier_speed_loss_penalty_per_mps == -150.0
    assert stronger.crash_penalty == pace.crash_penalty * 3.0
    assert stronger.stall_penalty == pace.stall_penalty * 3.0
    assert stronger.off_track_penalty == pace.off_track_penalty * 3.0
    assert stronger.early_off_track_penalty == pace.early_off_track_penalty * 3.0


@pytest.fixture
def pure_pace_context(tmp_path):
    store = RewardProfileStore(tmp_path)
    assert "Pure Pace" in store.names()
    config = store.load("Pure Pace")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight")
    try:
        env.reset(seed=1)
        telemetry = env.latest_telemetry
        assert telemetry is not None
        yield RewardContext(
            Transition("test", 1000, 1000, telemetry, (), {}),
            ControlDemand(0.0, 1.0, 0.0), ControlDemand(0.0, 0.0, 0.0),
            config, 0.001, 0.0, 0.0, 0.0,
        )
    finally:
        env.close()


@pytest.mark.parametrize("target", [30.0, 32.0, 40.0])
def test_pure_pace_completed_laps_rank_strictly_by_time(pure_pace_context, target):
    context = replace(pure_pace_context, config=replace(pure_pace_context.config, finish_target_s=target))
    scores = []
    for seconds in [25.0, 30.0, 32.0, 35.0, 36.0, 60.0, 90.0]:
        transition = replace(context.transition, ticks_advanced=round(seconds * 1000),
                             telemetry=replace(context.telemetry, elapsed_s=seconds), events=("finish",))
        score, terms, _ = calculate_reward(replace(context, transition=transition))
        assert all(value == 0 for name, value in terms.items() if name not in {"elapsed", "finish"})
        scores.append(score)
    assert all(faster > slower for faster, slower in zip(scores, scores[1:], strict=False))
    assert scores[-1] > 0  # Even a slow valid finish beats abandoning the lap.


def test_pure_pace_does_not_reward_line_speed_or_control_style(pure_pace_context):
    baseline = calculate_reward(pure_pace_context)[0]
    changed = replace(
        pure_pace_context,
        transition=replace(pure_pace_context.transition, events=("checkpoint",), telemetry=replace(
            pure_pace_context.telemetry, route_progress_m=20.0, local_velocity_mps=(0.0, 0.0, 45.0),
            lateral_offset_m=2.0, heading_error_rad=0.3,
        )),
        action=ControlDemand(0.8, 0.0, 1.0), barrier_contact=True,
    )
    assert calculate_reward(changed)[0] == baseline


@pytest.mark.parametrize("failure", ["crash", "stalled", "off_track", "early_off_track", "timed_out"])
def test_pure_pace_failures_have_negative_score_and_no_finish_credit(pure_pace_context, failure):
    if failure == "crash":
        context = replace(pure_pace_context, transition=replace(pure_pace_context.transition, events=("crash",)))
    else:
        context = replace(pure_pace_context, **{failure: True})
    score, terms, _ = calculate_reward(context)
    assert terms["finish"] == 0
    assert score < 0
