from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from polybot.controller import CenterlineController
from polybot.env import (
    PolyTrackEnv,
    RewardConfig,
    _airborne_brake_reward,
    _airborne_spin_penalty,
    _airborne_tilt_penalty,
    _checkpoint_reward,
    _finish_reward,
    _ghost_pose_reward,
    _ground_slip_penalty,
    _has_off_track_evidence,
)
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Action


def make_env(**kwargs: object) -> PolyTrackEnv:
    return PolyTrackEnv(MockSimulatorTransport(), **kwargs)


def test_gymnasium_contract() -> None:
    env = make_env(track_id="mock/straight")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_reset_is_deterministic_for_same_seed() -> None:
    env = make_env()
    try:
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


def test_curriculum_reset_starts_within_requested_final_section() -> None:
    env = make_env(
        track_id="mock/straight",
        curriculum_last_fraction=0.30,
        curriculum_probability=1.0,
    )
    try:
        _, info = env.reset(seed=42)

        progress_ratio = info["route_progress_m"] / info["track_length_m"]
        assert 0.70 <= progress_ratio <= 0.95
        assert info["ticks_advanced"] == 0
    finally:
        env.close()


def test_full_track_curriculum_can_start_anywhere_before_finish() -> None:
    env = make_env(
        track_id="mock/straight",
        curriculum_last_fraction=1.0,
        curriculum_probability=1.0,
    )
    try:
        _, info = env.reset(seed=42)

        progress_ratio = info["route_progress_m"] / info["track_length_m"]
        assert 0.0 <= progress_ratio <= 0.95
    finally:
        env.close()


def test_curriculum_section_starts_exactly_and_terminates_at_boundary() -> None:
    env = make_env(
        track_id="mock/straight",
        curriculum_start_ratio=0.25,
        curriculum_end_ratio=0.50,
    )
    try:
        _, reset_info = env.reset(seed=42)
        reset_ratio = reset_info["route_progress_m"] / reset_info["track_length_m"]
        assert reset_ratio == pytest.approx(0.25)

        terminated = False
        info: dict[str, object] = {}
        for _ in range(2_000):
            _, _, terminated, truncated, info = env.step(
                np.asarray([1, 1, 0], dtype=np.int64)
            )
            if terminated or truncated:
                break

        assert terminated
        assert "curriculum_section_complete" in info["events"]
        assert info["reward_terms"]["curriculum_section"] == 0.0
    finally:
        env.close()


def test_timed_curriculum_terminates_after_selected_duration() -> None:
    env = make_env(
        track_id="mock/straight",
        frame_skip=10,
        curriculum_start_s=9.5,
        curriculum_end_s=15.0,
    )
    try:
        env.reset(seed=42)
        reset_request = next(
            request
            for request in env.transport.command_log
            if request.get("op") == "reset"
        )
        assert reset_request["params"]["native_restart"] is False
        terminated = False
        info: dict[str, object] = {}
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(
                np.asarray([1, 1, 0], dtype=np.int64)
            )
            if terminated or truncated:
                break

        assert terminated
        assert info["elapsed_s"] == pytest.approx(5.5)
        assert "curriculum_section_complete" in info["events"]
    finally:
        env.close()


def test_only_ghost_pose_rewards_an_unfinished_step() -> None:
    env = make_env(track_id="mock/straight", frame_skip=4)
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(np.asarray([1, 1, 0], dtype=np.int64))
        assert info["reward_terms"]["progress"] == 0
        assert info["reward_terms"]["on_track_speed"] == 0
        assert info["reward_terms"]["ghost_imitation"] > 0
        assert reward == pytest.approx(info["reward_terms"]["ghost_imitation"])
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_speed_and_checkpoint_shaping_are_disabled() -> None:
    env = make_env(track_id="mock/straight", frame_skip=10)
    try:
        env.reset(seed=0)
        speed_reward = 0.0
        for _ in range(10):
            _, _, _, _, info = env.step(np.asarray([1, 1, 0], dtype=np.int64))
            speed_reward += info["reward_terms"]["on_track_speed"]

        assert speed_reward == 0
        assert env.reward_config.airborne_speed_per_m == 0
        assert env.reward_config.takeoff_target_speed_mps == 45.0
        assert env.reward_config.imitation_bonus_per_s == 18.0
        assert env.reward_config.checkpoint_bonus == 0
        assert env.reward_config.checkpoint_fast_bonus == 0
        assert env.reward_config.finish_bonus == 1000.0
        assert env.reward_config.finish_fast_bonus == 2000.0
        assert env.reward_config.finish_target_s == 22.0
        assert env.reward_config.finish_pace_decay_per_s == 1.5
        assert env.reward_config.unsafe_speed_penalty_per_m == 0
    finally:
        env.close()


def test_checkpoint_reward_is_disabled() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        fast = replace(env.latest_telemetry, checkpoint_index=1, elapsed_s=5.0)
        slow = replace(env.latest_telemetry, checkpoint_index=1, elapsed_s=25.0)

        assert _checkpoint_reward(fast, env.reward_config) == 0
        assert _checkpoint_reward(slow, env.reward_config) == 0
    finally:
        env.close()


def test_ghost_pose_reward_prefers_positional_and_rotational_alignment() -> None:
    config = RewardConfig()
    aligned = {
        "ghost_position_error_m": 0.0,
        "ghost_rotation_error_rad": 0.0,
    }
    displaced = {
        "ghost_position_error_m": 2.0,
        "ghost_rotation_error_rad": 0.0,
    }
    rotated = {
        "ghost_position_error_m": 0.0,
        "ghost_rotation_error_rad": np.deg2rad(20),
    }

    assert _ghost_pose_reward(aligned, config, 0.5) == pytest.approx(9.0)
    assert _ghost_pose_reward(displaced, config, 0.5) == pytest.approx(9.0 / np.e)
    assert _ghost_pose_reward(rotated, config, 0.5) == pytest.approx(9.0 / np.e)
    assert _ghost_pose_reward({}, config, 0.5) == 0.0


def test_faster_finish_receives_more_reward() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        fast = replace(env.latest_telemetry, elapsed_s=22.9)
        slow = replace(env.latest_telemetry, elapsed_s=23.1)

        assert _finish_reward(fast, env.reward_config) > _finish_reward(slow, env.reward_config)
        assert _finish_reward(slow, env.reward_config) >= 1000.0
        at_target = replace(env.latest_telemetry, elapsed_s=22.0)
        one_second_slow = replace(env.latest_telemetry, elapsed_s=23.0)
        assert _finish_reward(at_target, env.reward_config) == pytest.approx(3000.0)
        assert _finish_reward(one_second_slow, env.reward_config) == pytest.approx(
            1000.0 + 2000.0 * np.exp(-1.5)
        )
    finally:
        env.close()


def test_barrier_punishment_requires_native_collision() -> None:
    config = RewardConfig()

    assert config.barrier_contact_penalty == 0.0
    assert config.barrier_collision_impulse_threshold == 0.0
    assert config.off_track_landing_penalty == 0.0


def test_strong_airborne_spin_is_penalized_but_grounded_rotation_is_not() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        spinning = replace(
            env.latest_telemetry,
            angular_velocity_radps=(4.0, 3.0, 0.0),
            wheel_contacts=(0.0, 0.0, 0.0, 0.0),
        )
        grounded = replace(spinning, wheel_contacts=(1.0, 1.0, 1.0, 1.0))
        one_wheel_down = replace(spinning, wheel_contacts=(1.0, 0.0, 0.0, 0.0))

        assert _airborne_spin_penalty(spinning, env.reward_config, 0.1) == 0
        assert _airborne_spin_penalty(grounded, env.reward_config, 0.1) == 0
        assert _airborne_spin_penalty(one_wheel_down, env.reward_config, 0.1) == 0
        assert env.reward_config.airborne_spin_deadzone_radps == pytest.approx(
            np.deg2rad(2), rel=1e-5
        )
    finally:
        env.close()


def test_barrel_roll_orientation_is_penalized_while_airborne() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        sideways = replace(
            env.latest_telemetry,
            up_vector=(1.0, 0.0, 0.0),
            roll_rad=np.pi / 2,
            wheel_contacts=(0.0, 0.0, 0.0, 0.0),
        )
        inverted = replace(sideways, up_vector=(0.0, -1.0, 0.0), roll_rad=np.pi)
        allowed_pitch = replace(
            sideways,
            roll_rad=0.0,
            pitch_rad=np.deg2rad(60),
        )
        excessive_pitch = replace(allowed_pitch, pitch_rad=np.deg2rad(75))

        sideways_penalty = _airborne_tilt_penalty(sideways, env.reward_config, 0.1)
        inverted_penalty = _airborne_tilt_penalty(inverted, env.reward_config, 0.1)
        assert inverted_penalty == sideways_penalty == 0
        assert env.reward_config.airborne_roll_penalty_per_s == 0.0
        assert env.reward_config.airborne_roll_limit_rad == pytest.approx(np.deg2rad(60), rel=1e-5)
        assert env.reward_config.airborne_roll_failure_penalty == 0.0
        assert env.reward_config.airborne_pitch_deadzone_radps == pytest.approx(
            np.deg2rad(90), rel=1e-5
        )
        assert _airborne_tilt_penalty(allowed_pitch, env.reward_config, 0.1) == 0
        assert _airborne_tilt_penalty(excessive_pitch, env.reward_config, 0.1) == 0
        one_wheel_down = replace(sideways, wheel_contacts=(1.0, 0.0, 0.0, 0.0))
        assert _airborne_tilt_penalty(one_wheel_down, env.reward_config, 0.1) == 0
    finally:
        env.close()


def test_air_braking_receives_only_a_small_fully_airborne_reward() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        airborne = replace(
            env.latest_telemetry,
            wheel_contacts=(0.0, 0.0, 0.0, 0.0),
        )
        one_wheel_down = replace(airborne, wheel_contacts=(1.0, 0.0, 0.0, 0.0))
        braking = Action(steer=0, throttle=False, brake=True)
        coasting = Action(steer=0, throttle=False, brake=False)

        assert _airborne_brake_reward(airborne, braking, env.reward_config, 0.5) == 0
        assert _airborne_brake_reward(airborne, coasting, env.reward_config, 0.5) == 0
        assert _airborne_brake_reward(one_wheel_down, braking, env.reward_config, 0.5) == 0
    finally:
        env.close()


def test_large_slip_angle_is_penalized_only_with_four_wheels_grounded() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        sliding = replace(
            env.latest_telemetry,
            local_velocity_mps=(10.0, 0.0, 20.0),
            wheel_contacts=(1.0, 1.0, 1.0, 1.0),
        )
        controlled = replace(
            sliding,
            local_velocity_mps=(20.0 * np.tan(np.deg2rad(5)), 0.0, 20.0),
        )
        three_wheels = replace(sliding, wheel_contacts=(1.0, 1.0, 1.0, 0.0))

        assert env.reward_config.ground_slip_tolerance_rad == pytest.approx(
            np.deg2rad(5), rel=1e-5
        )
        assert env.reward_config.ground_slip_penalty_per_rad_s == 0.0
        assert _ground_slip_penalty(sliding, env.reward_config, 0.1) == 0
        assert _ground_slip_penalty(controlled, env.reward_config, 0.1) == pytest.approx(0)
        assert _ground_slip_penalty(three_wheels, env.reward_config, 0.1) == 0
    finally:
        env.close()


def test_ground_slip_penalty_is_symmetric() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        sliding_left = replace(
            env.latest_telemetry,
            local_velocity_mps=(12.0, 0.0, 40.0),
            wheel_contacts=(1.0, 1.0, 1.0, 1.0),
        )
        sliding_right = replace(sliding_left, local_velocity_mps=(-12.0, 0.0, 40.0))

        assert _ground_slip_penalty(
            sliding_left, env.reward_config, 0.1
        ) == pytest.approx(
            _ground_slip_penalty(sliding_right, env.reward_config, 0.1)
        )
    finally:
        env.close()


def test_wrapper_time_limit_truncates_without_termination() -> None:
    env = make_env(track_id="mock/straight", max_episode_steps=1)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.asarray([1, 0, 0], dtype=np.int64))
        assert not terminated
        assert truncated
        assert info["wrapper_time_limit"] is True
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.asarray([1, 0, 0], dtype=np.int64))
    finally:
        env.close()


def test_stationary_car_terminates_after_five_simulated_seconds() -> None:
    env = make_env(track_id="mock/straight", frame_skip=10, max_episode_steps=100)
    try:
        env.reset(seed=0)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.asarray([1, 0, 0], dtype=np.int64))

        assert terminated
        assert not truncated
        assert "stalled" in info["events"]
        assert info["stationary_s"] >= 5.0
        assert info["reward_terms"]["stall"] == env.reward_config.stall_penalty
    finally:
        env.close()


def test_stall_threshold_is_five_metres_per_second() -> None:
    config = RewardConfig()

    assert config.stall_speed_threshold_mps == 5.0
    assert config.stall_timeout_s == 5.0


def test_sustained_early_off_track_state_terminates_with_larger_penalty() -> None:
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport,
        track_id="mock/straight",
        frame_skip=10,
        max_episode_steps=100,
        reward_config=replace(RewardConfig(), barrier_collision_impulse_threshold=float("inf")),
    )
    try:
        env.reset(seed=0)
        assert transport.state is not None
        transport.state.lateral_offset_m = transport.track_half_width_m * 1.2
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.asarray([1, 1, 0], dtype=np.int64))

        assert terminated
        assert not truncated
        assert "off_track" in info["events"]
        assert info["off_track_s"] >= env.reward_config.off_track_timeout_s
        assert info["early_off_track"] is True
        assert info["reward_terms"]["off_track"] == env.reward_config.early_off_track_penalty
    finally:
        env.close()


def test_native_chassis_collision_terminates_episode() -> None:
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport,
        track_id="mock/straight",
        frame_skip=10,
        max_episode_steps=100,
    )
    try:
        env.reset(seed=0)
        assert transport.state is not None
        transport.state.lateral_offset_m = transport.track_half_width_m * 0.95
        _, _, terminated, truncated, info = env.step(np.asarray([1, 1, 0], dtype=np.int64))

        assert terminated
        assert not truncated
        assert "barrier_contact" in info["events"]
        assert info["reward_terms"]["barrier_contact"] == 0.0
    finally:
        env.close()


def test_airborne_jump_does_not_count_as_off_track() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        airborne = replace(
            env.latest_telemetry,
            lateral_offset_m=20.0,
            heading_error_rad=2.0,
            wheel_contacts=(0.0, 0.0, 0.0, 0.0),
        )

        assert not _has_off_track_evidence(airborne, RewardConfig())
    finally:
        env.close()


def test_bank_wall_ride_does_not_count_as_off_track() -> None:
    env = make_env(track_id="mock/straight")
    try:
        env.reset(seed=0)
        assert env.latest_telemetry is not None
        wall_riding = replace(
            env.latest_telemetry,
            lateral_offset_m=20.0,
            heading_error_rad=2.0,
            roll_rad=np.deg2rad(15),
            wheel_contacts=(1.0, 1.0, 1.0, 0.0),
        )

        assert not _has_off_track_evidence(wall_riding, RewardConfig())
        assert _has_off_track_evidence(
            replace(wall_riding, wheel_contacts=(1.0, 1.0, 0.0, 0.0)),
            RewardConfig(),
        )
    finally:
        env.close()


def test_frame_skip_matches_individual_fixed_ticks() -> None:
    held_action = np.asarray([2, 1, 0], dtype=np.int64)
    batched = make_env(track_id="mock/straight", frame_skip=4)
    individual = make_env(track_id="mock/straight", frame_skip=1)
    try:
        batched.reset(seed=7)
        individual.reset(seed=7)
        batched.step(held_action)
        for _ in range(4):
            individual.step(held_action)
        assert batched.latest_telemetry is not None
        assert individual.latest_telemetry is not None
        np.testing.assert_allclose(
            batched.latest_telemetry.position_m,
            individual.latest_telemetry.position_m,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            batched.latest_telemetry.local_velocity_mps,
            individual.latest_telemetry.local_velocity_mps,
            rtol=0,
            atol=1e-12,
        )
    finally:
        batched.close()
        individual.close()


@pytest.mark.parametrize("track_id", ["mock/straight", "mock/gentle-s"])
def test_baseline_completes_training_tracks(track_id: str) -> None:
    env = make_env(track_id=track_id, max_episode_steps=2_000)
    controller = CenterlineController()
    try:
        env.reset(seed=0)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            assert env.latest_telemetry is not None
            _, _, terminated, truncated, info = env.step(
                controller.policy_action(env.latest_telemetry)
            )
        assert "finish" in info["events"]
        assert "crash" not in info["events"]
    finally:
        env.close()
