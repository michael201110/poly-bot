from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Action
from polybot.training.config import DQNConfig


@pytest.mark.parametrize("index,expected", [
    (0, Action(0, False, False)),
    (1, Action(0, True, False)),
    (2, Action(0, False, True)),
    (3, Action(-1, False, False)),
    (4, Action(-1, True, False)),
    (5, Action(-1, False, True)),
    (6, Action(1, False, False)),
    (7, Action(1, True, False)),
    (8, Action(1, False, True)),
])
def test_all_native_digital_actions_are_exact_and_stateless(index: int, expected: Action) -> None:
    adapter = NativeDigitalActionAdapter()
    assert adapter.action_space == spaces.Discrete(9)
    assert adapter.schema == "digital-discrete-9-v2"
    assert adapter.sequence is False
    assert set(vars(adapter)) == {"action_space"}  # No pulse scheduler or phase.
    for _ in range(2):
        applied = adapter.apply(np.int64(index), 30)
        assert applied.ticks == [expected] * 30
        assert applied.demand.steer == expected.steer
        assert applied.demand.throttle == expected.throttle
        assert applied.demand.brake == expected.brake
        assert not (expected.throttle and expected.brake)
        adapter.reset()


def test_dqn_step_holds_left_throttle_for_all_thirty_ticks_without_pwm(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("DQN must not call a PWM scheduler")

    monkeypatch.setattr("polybot.control.pwm.PwmSteering.generate", fail)
    monkeypatch.setattr("polybot.control.pwm.ContinuousPwmControls.generate", fail)
    monkeypatch.setattr("polybot.control.actions.decode_pwm_level", fail)
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport, track_id="mock/straight", frame_skip=30,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        env.reset(seed=4)
        _, _, _, _, info = env.step(4)
        assert info["ticks_advanced"] == 30
        commands = [entry for entry in transport.command_log if entry["op"] == "step"]
        assert [entry["params"]["ticks"] for entry in commands] == [16, 14]
        assert all(entry["params"]["action"] == Action(-1, True, False).to_wire()
                   for entry in commands)
        assert info["requested_control_duty"] == {"steer": -1.0, "throttle": 1.0, "brake": 0.0}
        assert info["applied_control_fraction"] == {"steer": -1.0, "throttle": 1.0, "brake": 0.0}
    finally:
        env.close()


@pytest.mark.parametrize("changes", [
    {"replay_capacity": 0}, {"learning_starts": -1}, {"batch_size": 0},
    {"gamma": 0}, {"train_frequency": 0}, {"gradient_steps": 0},
    {"target_update_interval": 0}, {"exploration_fraction": 1.1},
    {"exploration_initial_eps": -0.1},
    {"exploration_final_eps": 0.5, "exploration_initial_eps": 0.2},
])
def test_dqn_config_rejects_invalid_values(changes: dict) -> None:
    with pytest.raises(ValueError, match="DQN"):
        DQNConfig(**changes)
