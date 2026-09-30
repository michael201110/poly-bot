from __future__ import annotations

from dataclasses import asdict, fields, replace

import numpy as np
import pytest
from gymnasium import spaces

from polybot.algorithms.registry import backend_for
from polybot.control.actions import (
    ContinuousActionAdapter,
    ControlDemand,
    DigitalActionAdapter,
)
from polybot.environment.env import PolyTrackEnv, _mean_tick_demand
from polybot.environment.rewards import (
    COMPONENTS,
    RewardConfig,
    RewardContext,
    _airborne_terms,
    _driving_terms,
    _ghost_guidance_weight,
    _ground_spin_penalty,
    summer_1_recovery_reward_config,
)
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Action, Transition
from polybot.training.config import PPOConfig, TQCConfig, TrainingConfig
from polybot.training.reward_profiles import RewardProfileStore


def test_control_modes_use_shared_demand_and_never_overlap() -> None:
    adapters = (
        (DigitalActionAdapter(), np.array([1, 1, 1]), 1.0),
        (DigitalActionAdapter(), np.array([1, 1, 1]), 1.0),
        (ContinuousActionAdapter(), np.array([0.2, -0.05], dtype=np.float32), 0.05),
    )
    for adapter, action, brake in adapters:
        applied = adapter.apply(action, 40)
        assert applied.demand.brake == pytest.approx(brake)
        assert applied.demand.throttle == 0
        assert all(not (tick.throttle and tick.brake) for tick in applied.ticks)
        adapter.reset()
        assert applied == adapter.apply(action, 40)


def test_external_overlapping_action_resolves_in_favor_of_brake() -> None:
    demand = ControlDemand.from_action(Action(steer=1, throttle=True, brake=True))
    assert demand == ControlDemand(steer=1, throttle=0.0, brake=1.0)


def test_landing_frame_mixed_air_brake_and_drive_ticks_average_as_signed_demand() -> None:
    demand = _mean_tick_demand([
        Action(steer=1, throttle=False, brake=True),
        Action(steer=-1, throttle=True, brake=False),
    ])
    assert demand == ControlDemand(steer=0.0, throttle=0.0, brake=0.0)
    forward = _mean_tick_demand([
        Action(steer=1, throttle=True, brake=False),
        Action(steer=1, throttle=True, brake=False),
        Action(steer=1, throttle=False, brake=True),
    ])
    assert forward.steer == 1
    assert forward.throttle == pytest.approx(1 / 3)
    assert forward.brake == 0


def test_pwm_direction_changes_reset_pulse_phase() -> None:
    adapter = ContinuousActionAdapter()
    first = adapter.apply(np.array([0.25, 0.25], dtype=np.float32), 40)
    assert sum(t.steer > 0 for t in first.ticks) == 10
    assert sum(t.throttle for t in first.ticks) == 10
    reverse = adapter.apply(np.array([-0.25, -0.25], dtype=np.float32), 40)
    assert sum(t.steer < 0 for t in reverse.ticks) == 10
    assert sum(t.brake for t in reverse.ticks) == 10


def test_ppo_and_tqc_share_unquantized_continuous_actions() -> None:
    ppo = backend_for("ppo").action_adapter(
        TrainingConfig(algorithm="ppo", ppo=PPOConfig())
    )
    tqc = backend_for("tqc").action_adapter(
        TrainingConfig(algorithm="tqc", tqc=TQCConfig())
    )
    assert ppo.schema == tqc.schema == "continuous-pwm-v2"
    assert isinstance(ppo.action_space, spaces.Box)
    assert ppo.action_space.shape == tqc.action_space.shape == (2,)
    np.testing.assert_array_equal(ppo.action_space.low, [-1, -1])
    np.testing.assert_array_equal(ppo.action_space.high, [1, 1])
    for steer, longitudinal in ((0.137, 0.684), (-0.291, -0.413), (0.017, -0.003)):
        result = ppo.apply(np.asarray([steer, longitudinal], dtype=np.float32), 100)
        assert result.demand.steer == pytest.approx(steer)
        assert result.demand.throttle == pytest.approx(max(0.0, longitudinal))
        assert result.demand.brake == pytest.approx(max(0.0, -longitudinal))
        assert abs(sum(tick.steer != 0 for tick in result.ticks) - abs(steer) * 100) <= 1
        assert abs(
            sum(tick.throttle or tick.brake for tick in result.ticks)
            - abs(longitudinal) * 100
        ) <= 1


def test_environment_action_adapter_and_shared_observation() -> None:
    observations = []
    ppo = backend_for("ppo").action_adapter(
        TrainingConfig(algorithm="ppo", ppo=PPOConfig())
    )
    for adapter, action in (
        (ppo, np.array([0.137, 0.684], dtype=np.float32)),
        (ContinuousActionAdapter(), np.array([0, 1], dtype=np.float32)),
    ):
        transport = MockSimulatorTransport()
        env = PolyTrackEnv(
            transport, track_id="mock/straight", frame_skip=30,
            action_adapter=adapter,
        )
        try:
            observation, _ = env.reset(seed=4)
            observations.append(observation)
            _, _, _, _, info = env.step(action)
            assert info["ticks_advanced"] == 30
            assert info["requested_control_duty"]["throttle"] == pytest.approx(
                0.684 if adapter is ppo else 1.0
            )
            assert all(
                item["params"]["ticks"] <= transport.max_ticks_per_step
                for item in transport.command_log if item["op"] == "step"
            )
        finally:
            env.close()
    np.testing.assert_array_equal(*observations)


def test_reward_components_have_unique_complete_terms_and_group_totals() -> None:
    profile = replace(RewardConfig(), ground_brake_penalty_per_s=-10)
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight",
        action_adapter=ContinuousActionAdapter(), reward_config=profile,
        frame_skip=10,
    )
    try:
        env.reset(seed=1)
        _, reward, _, _, info = env.step(np.array([0, -0.05], dtype=np.float32))
        terms = info["reward_terms"]
        groups = info["reward_groups"]
        assert len(COMPONENTS) == len(groups)
        assert len(terms) == 29
        assert sum(terms.values()) == pytest.approx(reward)
        assert sum(groups.values()) == pytest.approx(reward)
        assert info["requested_control_duty"]["brake"] == pytest.approx(0.05)
        assert terms["ground_brake"] == pytest.approx(
            -10 * 10 / 60 * info["applied_control_fraction"]["brake"]
        )
        assert info["applied_control_fraction"]["brake"] == pytest.approx(0.0, abs=0.1)
    finally:
        env.close()


def test_recovery_ghost_guidance_teaches_from_standstill() -> None:
    profile = summer_1_recovery_reward_config()
    assert profile.guidance_reward_scale == 1.0
    assert profile.guidance_min_forward_speed_mps == 0.0
    assert _ghost_guidance_weight(
        0.0, 0.0, 1.0, profile, incomplete_failure=False,
    ) == pytest.approx(1.0)
    assert _ghost_guidance_weight(
        0.0, 0.0, 0.1, profile, incomplete_failure=False,
    ) == 0.0
    assert _ghost_guidance_weight(
        0.0, 0.0, 1.0, profile, incomplete_failure=True,
    ) == 0.0


@pytest.mark.parametrize(
    ("contacts", "brake", "air", "ground"),
    [
        ((0, 0, 0, 0), 1.0, 2.0, 0.0),
        ((0, 0, 0, 0), 0.5, 1.0, 0.0),
        ((0, 0, 0, 0), 0.0, 0.0, 0.0),
        ((1, 0, 0, 0), 1.0, 0.0, -1.5),
        ((1, 1, 1, 1), 1.0, 0.0, -1.5),
    ],
)
def test_air_braking_and_ground_braking_are_exclusive(
    contacts, brake, air, ground,
) -> None:
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight")
    try:
        env.reset(seed=1)
        assert env.latest_telemetry is not None
        telemetry = replace(env.latest_telemetry, wheel_contacts=contacts)
        action = ControlDemand(0.0, 0.0, brake)
        context = RewardContext(
            Transition("test", 60, 60, telemetry, (), {}), action,
            ControlDemand(0.0, 0.0, 0.0),
            replace(RewardConfig(), airborne_brake_bonus_per_s=2.0,
                    ground_brake_penalty_per_s=-1.5),
            1 / 60, 0.0, 0.0, 0.0,
        )
        assert _airborne_terms(context)["airborne_brake"] == pytest.approx(air)
        assert _driving_terms(context)["ground_brake"] == pytest.approx(ground)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("contacts", "yaw_rate", "expected"),
    [
        ((1, 1, 1, 1), 4.0, 0.0),  # ordinary cornering below the deadzone
        ((1, 1, 0, 0), 9.0, -20.0),  # grounded wall-spin signal
        ((0, 0, 0, 0), 9.0, 0.0),  # jump rotation is handled separately
        ((1, 0, 0, 0), 9.0, 0.0),  # a single wheel touch is not firm ground contact
    ],
)
def test_ground_spin_penalty_only_targets_excess_yaw_with_ground_contact(
    contacts, yaw_rate, expected,
) -> None:
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight")
    try:
        env.reset(seed=1)
        assert env.latest_telemetry is not None
        telemetry = replace(
            env.latest_telemetry,
            wheel_contacts=contacts,
            angular_velocity_radps=(0.0, yaw_rate, 0.0),
        )
        config = replace(
            RewardConfig(), ground_spin_deadzone_radps=5.0,
            ground_spin_penalty_per_rad_s=-5.0,
            ground_spin_min_grounded_wheels=2,
        )
        assert _ground_spin_penalty(telemetry, config, dt=1.0) == pytest.approx(expected)
    finally:
        env.close()


def test_incomplete_time_limit_gets_failure_penalty() -> None:
    profile = replace(
        RewardConfig(), failure_early_penalty=-100.0,
        failure_progress_clawback_per_m=-1.0,
    )
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight",
        action_adapter=ContinuousActionAdapter(), reward_config=profile,
        max_episode_steps=1,
    )
    try:
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step(np.array([0, 1], dtype=np.float32))
        assert not terminated and truncated
        assert "time_limit" in info["events"]
        assert info["reward_terms"]["failure_early"] < 0
    finally:
        env.close()


def test_profile_roundtrip_and_comparison(tmp_path) -> None:
    store = RewardProfileStore(tmp_path)
    balanced = store.load("Balanced")
    store.save("My balanced", balanced)
    store.duplicate("My balanced", "My copy")
    assert store.load("My copy") == balanced
    assert not store.compare("My balanced", "My copy")
    assert store.compare("Balanced", "Pace")
    assert set(asdict(balanced)) == {field.name for field in fields(RewardConfig)}
