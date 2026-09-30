from __future__ import annotations

import gymnasium as gym
import numpy as np

from polybot.algorithms.registry import backend_for
from polybot.environment.env import AirBrakeActionWrapper, PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.compare_teacher_student import _record, first_divergences, replay_actions
from polybot.training.config import PPOConfig, TQCConfig, TrainingConfig


def test_first_divergences_reports_threshold_crossings() -> None:
    teacher = [
        {"action": [0.0, 1.0], "position_m": [0, 0, 0], "heading_deg": 0.0,
         "wheel_contacts": [1, 1, 1, 1], "speed_mps": 10.0,
         "steering": 0.0, "longitudinal": 1.0},
        {"action": [0.0, 1.0], "position_m": [1, 0, 0], "heading_deg": 0.0,
         "wheel_contacts": [1, 1, 1, 1], "speed_mps": 10.0,
         "steering": 0.0, "longitudinal": 1.0},
    ]
    student = [
        {**teacher[0], "action": [0.006, 1.0], "steering": 0.006},
        {**teacher[1], "position_m": [1.11, 0, 0], "heading_deg": 0.6,
         "wheel_contacts": [0, 1, 1, 1], "speed_mps": 9.9},
    ]
    divergence = first_divergences(teacher, student)
    assert divergence["action"]["0.005"] == 0
    assert divergence["action"]["0.01"] is None
    assert divergence["position_m"]["0.1"] == 1
    assert divergence["heading_deg"]["0.5"] == 1
    assert divergence["wheel_contacts"] == 1


def test_identical_continuous_actions_replay_identically_with_both_backends() -> None:
    action = np.asarray([0.137, 0.684], dtype=np.float32)
    tick_sequences = []
    for algorithm, kwargs in (("tqc", {"tqc": TQCConfig()}),
                              ("ppo", {"ppo": PPOConfig()})):
        config = TrainingConfig(algorithm=algorithm, **kwargs)
        adapter = backend_for(algorithm).action_adapter(config)
        env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                           frame_skip=30, action_adapter=adapter)
        env.capture_tick_controls = True
        try:
            observation, _ = env.reset(seed=17)
            records = []
            for index in range(3):
                previous = observation.copy()
                observation, _, _, _, info = env.step(action)
                records.append(_record(index, previous, action, info))
            replay = replay_actions(env, records, seed=17)
            assert first_divergences(records, replay)["position_m"]["0.01"] is None
            assert first_divergences(records, replay)["heading_deg"]["0.1"] is None
            tick_sequences.append([row["executed_tick_controls"] for row in records])
        finally:
            env.close()
    assert tick_sequences[0] == tick_sequences[1]


def test_overlapping_air_brake_resumes_previous_layer_on_touchdown() -> None:
    class CaptureEnv(gym.Env):
        def __init__(self) -> None:
            self._air_brake_request = False
            self._air_brake_base_action = None
            self.action = None
            self.base = None

        def reset(self, *, seed=None, options=None):
            del seed, options
            observation = np.zeros(105, dtype=np.float32)
            observation[12] = 0.75
            return observation, {}

        def step(self, action):
            self.action = np.asarray(action).copy()
            self.base = self._air_brake_base_action.copy()
            return np.zeros(105, dtype=np.float32), 0.0, False, False, {}

    base = CaptureEnv()
    wrapper = AirBrakeActionWrapper(base, [
        {"kind": "air_brake", "start": 0.68, "end": 0.82, "duty": 0.02},
        {"kind": "air_brake", "start": 0.68, "end": 0.81, "duty": 1.0},
    ])
    wrapper.reset()
    wrapper.step(np.asarray([0.25, 0.8], dtype=np.float32))
    assert base._air_brake_request
    np.testing.assert_allclose(base.action, [0.25, -1.0])
    np.testing.assert_allclose(base.base, [0.25, -0.02])
