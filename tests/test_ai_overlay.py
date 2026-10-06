from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from polybot.ai_overlay import (
    HUD_FRAME_SCHEMA,
    AIOverlaySettings,
    AIOverlaySettingsStore,
    AIOverlayTelemetryWrapper,
)
from polybot.control.actions import ContinuousActionAdapter
from polybot.environment.env import PolyTrackEnv
from polybot.environment.observations import FEATURE_SCHEMA, describe_observation, size
from polybot.mock import MockSimulatorTransport
from polybot.protocol import success_response
from polybot.transport import WebSocketServerTransport


class _FeatureEnv(gym.Env[np.ndarray, np.ndarray]):
    observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, *, fail_frame: bool = False) -> None:
        self.last_observation_features = [{
            "index": 0, "key": "velocity.forward", "label": "Forward velocity",
            "group": "vehicle", "unit": "m/s", "raw_value": 20.0,
            "value": 0.2, "minimum": -2.0, "maximum": 2.0,
        }]
        self.last_observation_feature_error = None
        self.fail_frame = fail_frame
        self.frames: list[dict[str, object]] = []
        self.capture_enabled: list[bool] = []

    def set_hud_feature_capture(self, enabled: bool) -> None:
        self.capture_enabled.append(enabled)

    def set_hud_frame(self, frame: dict[str, object] | None) -> None:
        if self.fail_frame:
            raise ValueError("deliberate HUD serialization failure")
        if frame is not None:
            self.frames.append(frame)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.asarray([0.1, -0.2], dtype=np.float32), {"elapsed_s": 0.0}

    def step(self, action: np.ndarray):
        next_feature = dict(self.last_observation_features[0], value=0.3, raw_value=30.0)
        self.last_observation_features = [next_feature]
        info = {
            "raw_policy_action": [0.25, -0.5],
            "transformed_policy_action": [0.2, -0.4],
            "requested_control_duty": {"steer": 0.2, "throttle": 0.0, "brake": 0.4},
            "applied_control_fraction": {"steer": 0.25, "throttle": 0.0, "brake": 0.5},
            "reward_terms": {"progress": 4.0, "finish": 0.0},
            "reward_groups": {"Progress": 4.0, "Milestones": 0.0},
            "events": ("finish",),
            "tick": 60,
            "elapsed_s": 1.0,
            "route_progress_m": 50.0,
            "track_length_m": 100.0,
        }
        return np.asarray([0.3, -0.4], dtype=np.float32), 4.0, True, False, info


def test_settings_round_trip_and_malformed_file(tmp_path) -> None:
    store = AIOverlaySettingsStore(tmp_path / "config" / "ai-overlay.json")
    settings = AIOverlaySettings(
        enabled=False,
        preset="full",
        scale=1.2,
        show_episode_status=False,
        show_labels=False,
        show_observations=False,
        show_controls=True,
        show_reward_breakdown=False,
        show_event_popups=False,
        lookahead_points=7,
    )
    store.save(settings)
    assert store.load() == settings

    store.path.write_text(json.dumps({"schema": "unknown", "settings": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported or malformed"):
        store.load()


def test_hud_frames_are_exact_inputs_actions_controls_and_rewards() -> None:
    settings = AIOverlaySettings()
    env = _FeatureEnv()
    wrapper = AIOverlayTelemetryWrapper(
        env,
        settings_provider=lambda: settings,
        context_provider=lambda: {
            "mode": "training",
            "track_name": "Summer 1",
            "algorithm": "grtqc",
            "observation_schema": "polybot.observation.v2",
            "training_step": 125_000,
            "episode": 7,
            "episode_total": None,
            "run_id": "run-1",
        },
        frame_skip=30,
        reward_scale_provider=lambda: 0.01,
    )

    wrapper.reset(seed=1)
    _, reward, terminated, truncated, _ = wrapper.step(np.asarray([0.2, -0.4], dtype=np.float32))

    frame = env.frames[-1]
    assert reward == 4.0
    assert terminated and not truncated
    assert frame["schema"] == HUD_FRAME_SCHEMA
    assert frame["training_step"] == 125_000
    assert frame["simulator_tick"] == 60
    assert frame["decision_count"] == 1
    assert frame["frame_skip"] == 30
    assert frame["run_id"] == "run-1"
    assert frame["features"][0]["value"] == 0.2
    assert frame["controls"]["model_output"] == [
        {"label": "Steer", "value": 0.25},
        {"label": "Longitudinal", "value": -0.5},
    ]
    assert frame["controls"]["transformed_output"][0]["value"] == 0.2
    assert frame["controls"]["adapter_demand"]["brake"] == 0.4
    assert frame["controls"]["applied"]["brake"] == 0.5
    assert frame["reward"]["raw_step"] == 4.0
    assert frame["reward"]["learner_step"] == 0.04
    assert frame["reward"]["episode_learner"] == 0.04
    assert frame["reward"]["terms"] == {"progress": 4.0, "finish": 0.0}
    assert frame["reward"]["groups"] == {"Progress": 4.0, "Milestones": 0.0}
    assert frame["reward"]["learner_terms"] == {"progress": 0.04, "finish": 0.0}
    assert frame["reward"]["learner_groups"] == {"Progress": 0.04, "Milestones": 0.0}
    assert frame["status"] == "finished"


def test_hud_packaging_failure_does_not_fail_environment_step() -> None:
    env = _FeatureEnv(fail_frame=True)
    wrapper = AIOverlayTelemetryWrapper(
        env,
        settings_provider=AIOverlaySettings,
        context_provider=lambda: {"mode": "manual_model_drive"},
        frame_skip=4,
        reward_scale_provider=lambda: 1.0,
    )
    wrapper.reset(seed=1)
    observation, reward, terminated, truncated, _ = wrapper.step(np.zeros(2, dtype=np.float32))
    assert observation.shape == (2,)
    assert reward == 4.0
    assert terminated and not truncated


@pytest.mark.parametrize("lookahead_count", [1, 3, 12])
def test_feature_descriptions_use_exact_policy_vector_values(lookahead_count: int) -> None:
    env = PolyTrackEnv(
        MockSimulatorTransport(),
        lookahead_count=lookahead_count,
        action_adapter=ContinuousActionAdapter(expose_controller_state=True),
        expose_training_state=True,
    )
    env.set_hud_feature_capture(True)
    try:
        observation, _ = env.reset(seed=12)
        telemetry = env.latest_telemetry
        assert telemetry is not None
        features = env.last_observation_features
        assert features is not None
        assert len(features) == size(lookahead_count) + 4 + 12
        assert all(item["value"] == pytest.approx(float(observation[item["index"]])) for item in features)
        assert features[0]["key"] == "velocity.right"
        assert features[2]["key"] == "velocity.forward"
        assert features[2]["raw_value"] == pytest.approx(telemetry.local_velocity_mps[2])
        assert features[7]["label"] == "Yaw rate (about up/Y)"
        assert features[-16]["group"] == "controller_state"
        assert features[-12]["group"] == "training_state"
        assert features[-1]["key"] == "training_state.11"
        assert FEATURE_SCHEMA == "polybot.observation-features.v1"
    finally:
        env.close()


def test_observation_feature_schema_mismatch_is_explicit() -> None:
    env = PolyTrackEnv(MockSimulatorTransport(), lookahead_count=2)
    try:
        _, _ = env.reset(seed=2)
        assert env.latest_telemetry is not None
        with pytest.raises(ValueError, match="metadata has"):
            describe_observation(env.latest_telemetry, np.zeros(1, dtype=np.float32))
    finally:
        env.close()


def test_bridge_hud_frames_are_capability_gated_and_reset_bypasses_rate_limit() -> None:
    class CaptureTransport:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []
            self.notifications: list[dict[str, object]] = []

        def request(self, message, *, timeout_s=None):
            self.messages.append(dict(message))
            return success_response(message, {})

        def notify(self, message) -> None:
            self.notifications.append(dict(message))

        def close(self) -> None:
            pass

    transport = CaptureTransport()
    env = PolyTrackEnv(transport)  # type: ignore[arg-type]
    env._handshake_complete = True
    env.simulator_capabilities = {"features": ["ai_overlay_hud"]}
    frame = {"schema": HUD_FRAME_SCHEMA, "enabled": True, "settings": {}}
    env.publish_hud_frame(frame)
    assert transport.notifications[-1]["op"] == "hud_frame"
    assert transport.notifications[-1]["params"]["frame"] == frame

    env.set_hud_frame(frame)
    env._hud_last_sent = 0.0
    env._exchange("step", {})
    assert transport.messages[-1]["params"]["hud_frame"] == frame

    env.set_hud_frame(frame)
    env._hud_last_sent = 999_999_999.0
    env._exchange("reset", {})
    assert transport.messages[-1]["params"]["hud_frame"] == frame
    env.close()


def test_bridge_without_hud_capability_drops_overlay_frame_without_failing() -> None:
    class CaptureTransport:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def request(self, message, *, timeout_s=None):
            self.messages.append(dict(message))
            return success_response(message, {})

        def close(self) -> None:
            pass

    transport = CaptureTransport()
    env = PolyTrackEnv(transport)  # type: ignore[arg-type]
    env._handshake_complete = True
    env.simulator_capabilities = {
        "simulator": "polytrack-pml-worker",
        "features": [],
    }
    env.publish_hud_frame({"enabled": True})
    assert env._exchange("step", {}) == {}
    assert "hud_frame" not in transport.messages[-1]["params"]
    env.close()


def test_failed_one_way_hud_notification_does_not_fail_training() -> None:
    class FailingTransport:
        def notify(self, message) -> None:
            raise ValueError("deliberate notification failure")

        def close(self) -> None:
            pass

    env = PolyTrackEnv(FailingTransport())  # type: ignore[arg-type]
    env.simulator_capabilities = {"features": ["ai_overlay_hud"]}
    env.publish_hud_frame({"schema": HUD_FRAME_SCHEMA, "enabled": True})
    assert env._hud_warned_unsupported
    env.close()


def test_websocket_one_way_notification_is_serialized_and_bounded() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, raw: str) -> None:
            self.sent.append(raw)

    transport = WebSocketServerTransport(max_message_bytes=128)
    connection = FakeConnection()
    transport.connect = lambda timeout_s=None: pytest.fail(  # type: ignore[method-assign]
        "one-way HUD notification must not wait to reconnect"
    )
    transport._connection = connection  # type: ignore[assignment]
    transport.notify({"op": "hud_frame", "params": {"enabled": True}})
    assert json.loads(connection.sent[0]) == {
        "op": "hud_frame",
        "params": {"enabled": True},
    }
    with pytest.raises(ValueError, match="size limit"):
        transport.notify({"large": "x" * 256})
    transport._connection = None
    with pytest.raises(ConnectionError, match="disconnected"):
        transport.notify({"op": "hud_frame"})
    assert transport._server is None
    transport.close()
