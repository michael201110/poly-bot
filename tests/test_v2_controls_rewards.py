from __future__ import annotations

from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from polybot.control.actions import (
    ContinuousPwmActionAdapter,
    DigitalActionAdapter,
    DiscretePwmActionAdapter,
)
from polybot.environment.env import PolyTrackEnv
from polybot.environment.rewards import COMPONENTS, RewardConfig
from polybot.mock import MockSimulatorTransport
from polybot.training.reward_profiles import RewardProfileStore


def test_control_modes_use_shared_demand_and_never_overlap() -> None:
    adapters = (
        (DigitalActionAdapter(), np.array([1, 1, 1]), 1.0),
        (DiscretePwmActionAdapter(41), np.array([20, 1, 1]), 1.0),
        (ContinuousPwmActionAdapter(), np.array([0.2, -0.05], dtype=np.float32), 0.05),
    )
    for adapter, action, brake in adapters:
        applied = adapter.apply(action, 40)
        assert applied.demand.brake == pytest.approx(brake)
        assert applied.demand.throttle == 0
        assert all(not (tick.throttle and tick.brake) for tick in applied.ticks)
        adapter.reset()
        assert applied == adapter.apply(action, 40)


def test_pwm_direction_changes_reset_pulse_phase() -> None:
    adapter = ContinuousPwmActionAdapter()
    first = adapter.apply(np.array([0.25, 0.25], dtype=np.float32), 40)
    assert sum(t.steer > 0 for t in first.ticks) == 10
    assert sum(t.throttle for t in first.ticks) == 10
    reverse = adapter.apply(np.array([-0.25, -0.25], dtype=np.float32), 40)
    assert sum(t.steer < 0 for t in reverse.ticks) == 10
    assert sum(t.brake for t in reverse.ticks) == 10


def test_environment_action_adapter_and_shared_observation() -> None:
    observations = []
    for adapter, action in (
        (DiscretePwmActionAdapter(), np.array([20, 1, 0])),
        (ContinuousPwmActionAdapter(), np.array([0, 1], dtype=np.float32)),
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
            assert info["requested_control_duty"]["throttle"] == 1
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
        action_adapter=ContinuousPwmActionAdapter(), reward_config=profile,
        frame_skip=10,
    )
    try:
        env.reset(seed=1)
        _, reward, _, _, info = env.step(np.array([0, -0.05], dtype=np.float32))
        terms = info["reward_terms"]
        groups = info["reward_groups"]
        assert len(COMPONENTS) == len(groups)
        assert len(terms) == 28
        assert sum(terms.values()) == pytest.approx(reward)
        assert sum(groups.values()) == pytest.approx(reward)
        assert info["requested_control_duty"]["brake"] == pytest.approx(0.05)
        assert terms["ground_brake"] == pytest.approx(-10 * 10 / 60 * 0.05)
        assert info["applied_control_fraction"]["brake"] == pytest.approx(0.0, abs=0.1)
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
