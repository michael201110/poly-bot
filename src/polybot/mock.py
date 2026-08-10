"""Deterministic kinematic simulator implementing the PolyBot protocol.

This is intentionally simple. Its job is to test the environment and training
pipeline, not to approximate PolyTrack's Bullet physics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from polybot.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Action,
    ProtocolViolation,
    error_response,
    success_response,
)
from polybot.transport import TransportClosed


@dataclass(slots=True)
class _MockState:
    episode_id: str
    seed: int
    track_id: str
    track_length_m: float
    curvature_amplitude: float
    curvature_phase: float
    tick: int = 0
    speed_mps: float = 0.0
    progress_m: float = 0.0
    lateral_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    yaw_rate_radps: float = 0.0
    checkpoint_index: int = 0
    previous_action: Action = field(default_factory=Action)
    finished: bool = False
    crashed: bool = False


class MockSimulatorTransport:
    """In-process protocol peer with seeded, fixed-step vehicle dynamics."""

    fixed_dt_s = 1.0 / 60.0
    max_ticks_per_step = 16
    track_half_width_m = 5.0

    def __init__(self, *, lookahead_spacing_m: float = 5.0) -> None:
        self.lookahead_spacing_m = lookahead_spacing_m
        self.lookahead_count: int | None = None
        self.state: _MockState | None = None
        self._episode_counter = 0
        self._closed = False
        self.command_log: list[dict[str, Any]] = []

    def request(
        self, message: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> Mapping[str, Any]:
        del timeout_s
        if self._closed:
            raise TransportClosed("mock transport is closed")

        # A JSON round trip enforces the same finite/serializable boundary as a
        # real transport and prevents callers mutating the recorded request.
        try:
            request = json.loads(json.dumps(dict(message), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ProtocolViolation("request is not valid finite JSON") from exc
        self.command_log.append(request)
        request_id = request.get("id")

        if request.get("protocol") != PROTOCOL_NAME:
            return error_response(request_id, "unsupported_protocol", "unsupported protocol name")
        if request.get("v") != PROTOCOL_VERSION:
            return error_response(request_id, "unsupported_version", "unsupported protocol version")
        params = request.get("params")
        if not isinstance(params, Mapping):
            return error_response(request_id, "invalid_params", "params must be an object")

        try:
            match request.get("op"):
                case "hello":
                    result = self._hello(params)
                case "reset":
                    result = self._reset(params)
                case "step":
                    result = self._step(params)
                case "close":
                    result = {"closed": True}
                case _:
                    return error_response(
                        request_id, "unknown_operation", "operation is unsupported"
                    )
            return success_response(request, result)
        except (ProtocolViolation, ValueError) as exc:
            return error_response(request_id, "invalid_request", str(exc))

    def _hello(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if params.get("protocol") != PROTOCOL_NAME:
            raise ProtocolViolation("hello protocol is unsupported")
        if params.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolViolation("hello protocol version is unsupported")
        lookahead_count = params.get("lookahead_count")
        if isinstance(lookahead_count, bool) or not isinstance(lookahead_count, int):
            raise ProtocolViolation("lookahead_count must be an integer")
        if not 1 <= lookahead_count <= 128:
            raise ProtocolViolation("lookahead_count must be between 1 and 128")
        self.lookahead_count = lookahead_count
        return {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "simulator": "mock-kinematic",
            "fixed_dt_s": self.fixed_dt_s,
            "max_ticks_per_step": self.max_ticks_per_step,
            "lookahead_count": lookahead_count,
            "features": ["seeded_reset", "fixed_step", "ordered_progress"],
        }

    def _reset(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if self.lookahead_count is None:
            raise ProtocolViolation("hello must be called before reset")
        seed = params.get("seed")
        track_id = params.get("track_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ProtocolViolation("seed must be a non-negative integer")
        if not isinstance(track_id, str) or not track_id:
            raise ProtocolViolation("track_id must be a non-empty string")

        track_length, amplitude = self._track_settings(track_id)
        self._episode_counter += 1
        phase = ((seed % 997) / 997.0) * math.tau if amplitude else 0.0
        self.state = _MockState(
            episode_id=f"mock-{self._episode_counter}",
            seed=seed,
            track_id=track_id,
            track_length_m=track_length,
            curvature_amplitude=amplitude,
            curvature_phase=phase,
        )
        return self._transition(ticks_advanced=0, events=[])

    @staticmethod
    def _track_settings(track_id: str) -> tuple[float, float]:
        if track_id == "mock/straight":
            return 120.0, 0.0
        if track_id == "mock/gentle-s":
            return 300.0, 0.010
        if track_id == "mock/tight-s":
            return 360.0, 0.018
        raise ProtocolViolation(f"unknown mock track: {track_id}")

    def _step(self, params: Mapping[str, Any]) -> dict[str, Any]:
        state = self.state
        if state is None:
            raise ProtocolViolation("reset must be called before step")
        if params.get("episode_id") != state.episode_id:
            raise ProtocolViolation("episode_id is stale or unknown")
        ticks = params.get("ticks")
        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise ProtocolViolation("ticks must be an integer")
        if not 1 <= ticks <= self.max_ticks_per_step:
            raise ProtocolViolation(f"ticks must be between 1 and {self.max_ticks_per_step}")
        if state.finished or state.crashed:
            raise ProtocolViolation("episode has ended; reset before stepping again")
        action = Action.from_wire(params.get("action"))

        events: list[str] = []
        start_checkpoint = state.checkpoint_index
        ticks_advanced = 0
        for _ in range(ticks):
            self._physics_tick(action)
            ticks_advanced += 1
            if state.finished or state.crashed:
                break

        while start_checkpoint < state.checkpoint_index:
            events.append("checkpoint")
            start_checkpoint += 1
        if abs(state.lateral_offset_m) > self.track_half_width_m:
            events.append("off_track")
        if state.finished:
            events.append("finish")
        if state.crashed:
            events.append("crash")
        return self._transition(ticks_advanced=ticks_advanced, events=events)

    def _physics_tick(self, action: Action) -> None:
        state = self.state
        assert state is not None
        dt = self.fixed_dt_s

        acceleration = 11.0 * int(action.throttle) - 18.0 * int(action.brake)
        if state.speed_mps > 0:
            acceleration -= 0.18 + 0.012 * state.speed_mps**2
        state.speed_mps = max(0.0, min(45.0, state.speed_mps + acceleration * dt))

        curvature = self._curvature(state.progress_m)
        steering_yaw_rate = action.steer * (0.32 + 0.018 * state.speed_mps)
        state.yaw_rate_radps = steering_yaw_rate - curvature * state.speed_mps
        state.heading_error_rad += state.yaw_rate_radps * dt
        state.heading_error_rad = math.atan2(
            math.sin(state.heading_error_rad), math.cos(state.heading_error_rad)
        )

        state.lateral_offset_m += state.speed_mps * math.sin(state.heading_error_rad) * dt
        forward_speed = max(0.0, state.speed_mps * math.cos(state.heading_error_rad))
        state.progress_m = min(state.track_length_m, state.progress_m + forward_speed * dt)
        state.tick += 1
        state.previous_action = action
        state.checkpoint_index = min(3, int((state.progress_m / state.track_length_m) * 4.0))
        state.finished = state.progress_m >= state.track_length_m
        state.crashed = (
            abs(state.lateral_offset_m) > self.track_half_width_m * 1.5
            or abs(state.heading_error_rad) > 1.55
        )

    def _curvature(self, progress_m: float) -> float:
        state = self.state
        assert state is not None
        if state.curvature_amplitude == 0:
            return 0.0
        return state.curvature_amplitude * math.sin(progress_m / 28.0 + state.curvature_phase)

    def _transition(self, *, ticks_advanced: int, events: list[str]) -> dict[str, Any]:
        state = self.state
        assert state is not None
        return {
            "episode_id": state.episode_id,
            "tick": state.tick,
            "ticks_advanced": ticks_advanced,
            "state": self._telemetry(),
            "events": events,
            "info": {
                "track_id": state.track_id,
                "off_track": abs(state.lateral_offset_m) > self.track_half_width_m,
            },
        }

    def _telemetry(self) -> dict[str, Any]:
        state = self.state
        assert state is not None
        assert self.lookahead_count is not None

        lookahead: list[list[float]] = []
        mask: list[int] = []
        for index in range(self.lookahead_count):
            distance = (index + 1) * self.lookahead_spacing_m
            future_progress = state.progress_m + distance
            if future_progress <= state.track_length_m:
                curvature = self._curvature(future_progress)
                right_m = (
                    0.5 * curvature * distance**2
                    - state.lateral_offset_m
                    - math.tan(state.heading_error_rad) * distance
                )
                lookahead.append([distance, right_m, 0.0, curvature])
                mask.append(1)
            else:
                lookahead.append([0.0, 0.0, 0.0, 0.0])
                mask.append(0)

        yaw = state.heading_error_rad
        return {
            "position_m": [state.lateral_offset_m, 0.0, state.progress_m],
            "quaternion_xyzw": [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)],
            "local_velocity_mps": [0.0, 0.0, state.speed_mps],
            "angular_velocity_radps": [0.0, state.yaw_rate_radps, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
            "pitch_rad": 0.0,
            "roll_rad": 0.0,
            "wheel_contacts": [1, 1, 1, 1] if not state.crashed else [0, 0, 0, 0],
            "checkpoint_index": state.checkpoint_index,
            "elapsed_s": state.tick * self.fixed_dt_s,
            "previous_action": state.previous_action.to_wire(),
            "track": {
                "progress_m": state.progress_m,
                "length_m": state.track_length_m,
                "half_width_m": self.track_half_width_m,
                "lateral_offset_m": state.lateral_offset_m,
                "heading_error_rad": state.heading_error_rad,
                "lookahead": lookahead,
                "lookahead_mask": mask,
            },
        }

    def close(self) -> None:
        self._closed = True


def make_mock_env(**kwargs: Any) -> Any:
    """Convenience factory kept import-light for vectorized trainers."""

    from polybot.env import PolyTrackEnv

    return PolyTrackEnv(MockSimulatorTransport(), **kwargs)
