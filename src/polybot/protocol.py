"""Versioned, transport-independent simulator protocol.

The game adapter reports physical facts. Reward shaping and Gymnasium episode
limits intentionally live in :mod:`polybot.env`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL_NAME = "polybot.sim"
PROTOCOL_VERSION = 1


class ProtocolViolation(RuntimeError):
    """Raised when a simulator message violates the wire contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolViolation(f"{name} must be an object")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolViolation(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolViolation(f"{name} must be finite")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolViolation(f"{name} must be an integer >= {minimum}")
    return value


def _vector(value: object, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolViolation(f"{name} must be an array")
    if len(value) != size:
        raise ProtocolViolation(f"{name} must contain exactly {size} numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class Action:
    """A digital driving action held for one or more fixed simulation ticks."""

    steer: int = 0
    throttle: bool = False
    brake: bool = False

    def __post_init__(self) -> None:
        if self.steer not in (-1, 0, 1):
            raise ValueError("steer must be -1, 0, or 1")

    @classmethod
    def from_policy(cls, value: Sequence[int] | np.ndarray) -> Action:
        """Decode Gymnasium ``MultiDiscrete([3, 2, 2])`` values."""

        array = np.asarray(value)
        if array.shape != (3,):
            raise ValueError("action must have shape (3,)")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(array, np.floor(array))):
                raise ValueError("action entries must be integers")
        values = tuple(int(item) for item in array)
        if values[0] not in (0, 1, 2) or values[1] not in (0, 1) or values[2] not in (0, 1):
            raise ValueError("action is outside MultiDiscrete([3, 2, 2])")
        return cls(steer=values[0] - 1, throttle=bool(values[1]), brake=bool(values[2]))

    def to_policy(self) -> np.ndarray:
        return np.asarray([self.steer + 1, int(self.throttle), int(self.brake)], dtype=np.int64)

    def to_wire(self) -> dict[str, float]:
        return {
            "steer": float(self.steer),
            "throttle": float(self.throttle),
            "brake": float(self.brake),
        }

    @classmethod
    def from_wire(cls, value: object) -> Action:
        data = _mapping(value, "action")
        steer = _finite(data.get("steer"), "action.steer")
        throttle = _finite(data.get("throttle"), "action.throttle")
        brake = _finite(data.get("brake"), "action.brake")
        if steer not in (-1.0, 0.0, 1.0):
            raise ProtocolViolation("action.steer must be -1, 0, or 1")
        if throttle not in (0.0, 1.0) or brake not in (0.0, 1.0):
            raise ProtocolViolation("action throttle and brake must be 0 or 1")
        return cls(int(steer), bool(throttle), bool(brake))


@dataclass(frozen=True, slots=True)
class Telemetry:
    """One fixed-tick simulator observation.

    Axes use a right-handed car-local frame: +X right, +Y up, +Z forward.
    Rotations are radians and quaternions are ordered XYZW.
    """

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    local_velocity_mps: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]
    up_vector: tuple[float, float, float]
    route_progress_m: float
    track_length_m: float
    track_half_width_m: float
    lateral_offset_m: float
    heading_error_rad: float
    pitch_rad: float
    roll_rad: float
    wheel_contacts: tuple[float, float, float, float]
    checkpoint_index: int
    elapsed_s: float
    previous_action: Action
    lookahead: tuple[tuple[float, float, float, float], ...]
    lookahead_mask: tuple[float, ...]

    @classmethod
    def from_wire(cls, value: object, *, lookahead_count: int) -> Telemetry:
        state = _mapping(value, "state")
        track = _mapping(state.get("track"), "state.track")

        raw_lookahead = track.get("lookahead")
        if not isinstance(raw_lookahead, Sequence) or isinstance(raw_lookahead, (str, bytes)):
            raise ProtocolViolation("state.track.lookahead must be an array")
        if len(raw_lookahead) != lookahead_count:
            raise ProtocolViolation(
                f"state.track.lookahead must contain exactly {lookahead_count} points"
            )
        lookahead = tuple(
            _vector(point, f"state.track.lookahead[{index}]", 4)
            for index, point in enumerate(raw_lookahead)
        )
        mask = _vector(track.get("lookahead_mask"), "state.track.lookahead_mask", lookahead_count)
        if any(item not in (0.0, 1.0) for item in mask):
            raise ProtocolViolation("state.track.lookahead_mask values must be 0 or 1")

        track_length = _finite(track.get("length_m"), "state.track.length_m")
        half_width = _finite(track.get("half_width_m"), "state.track.half_width_m")
        if track_length <= 0 or half_width <= 0:
            raise ProtocolViolation("track length and half-width must be positive")

        contacts = _vector(state.get("wheel_contacts"), "state.wheel_contacts", 4)
        if any(item not in (0.0, 1.0) for item in contacts):
            raise ProtocolViolation("wheel contact values must be 0 or 1")

        return cls(
            position_m=_vector(state.get("position_m"), "state.position_m", 3),
            quaternion_xyzw=_vector(state.get("quaternion_xyzw"), "state.quaternion_xyzw", 4),
            local_velocity_mps=_vector(
                state.get("local_velocity_mps"), "state.local_velocity_mps", 3
            ),
            angular_velocity_radps=_vector(
                state.get("angular_velocity_radps"), "state.angular_velocity_radps", 3
            ),
            up_vector=_vector(state.get("up_vector"), "state.up_vector", 3),
            route_progress_m=_finite(track.get("progress_m"), "state.track.progress_m"),
            track_length_m=track_length,
            track_half_width_m=half_width,
            lateral_offset_m=_finite(track.get("lateral_offset_m"), "state.track.lateral_offset_m"),
            heading_error_rad=_finite(
                track.get("heading_error_rad"), "state.track.heading_error_rad"
            ),
            pitch_rad=_finite(state.get("pitch_rad"), "state.pitch_rad"),
            roll_rad=_finite(state.get("roll_rad"), "state.roll_rad"),
            wheel_contacts=contacts,
            checkpoint_index=_integer(state.get("checkpoint_index"), "state.checkpoint_index"),
            elapsed_s=_finite(state.get("elapsed_s"), "state.elapsed_s"),
            previous_action=Action.from_wire(state.get("previous_action")),
            lookahead=lookahead,
            lookahead_mask=mask,
        )

    @staticmethod
    def vector_size(lookahead_count: int) -> int:
        # Motion/orientation (9), route relationship (5), contacts (4),
        # previous action (3), lookahead points (4N), lookahead mask (N).
        return 21 + 5 * lookahead_count

    def to_vector(self) -> np.ndarray:
        """Return a normalized, track-invariant feature vector for the policy."""

        width = max(self.track_half_width_m, 0.01)
        base = [
            *(np.clip(self.local_velocity_mps, -200.0, 200.0) / 100.0),
            *(np.clip(self.angular_velocity_radps, -50.0, 50.0) / 10.0),
            *np.clip(self.up_vector, -1.0, 1.0),
            float(np.clip(self.route_progress_m / self.track_length_m, -1.0, 2.0)),
            float(np.clip(self.lateral_offset_m / width, -5.0, 5.0)),
            float(np.clip(self.heading_error_rad / math.pi, -1.0, 1.0)),
            float(np.clip(self.pitch_rad / math.pi, -1.0, 1.0)),
            float(np.clip(self.roll_rad / math.pi, -1.0, 1.0)),
            *(np.asarray(self.wheel_contacts, dtype=np.float64)),
            float(self.previous_action.steer),
            float(self.previous_action.throttle),
            float(self.previous_action.brake),
        ]
        future: list[float] = []
        for forward_m, right_m, up_m, curvature_inv_m in self.lookahead:
            future.extend(
                (
                    float(np.clip(forward_m / 100.0, -5.0, 5.0)),
                    float(np.clip(right_m / 50.0, -5.0, 5.0)),
                    float(np.clip(up_m / 50.0, -5.0, 5.0)),
                    float(np.clip(curvature_inv_m / 0.2, -5.0, 5.0)),
                )
            )
        vector = np.asarray([*base, *future, *self.lookahead_mask], dtype=np.float32)
        expected = self.vector_size(len(self.lookahead))
        if vector.shape != (expected,):  # Defensive check for schema edits.
            raise AssertionError(
                f"feature schema produced {vector.size} values, expected {expected}"
            )
        return vector

    def to_info(self) -> dict[str, Any]:
        return {
            "position_m": self.position_m,
            "quaternion_xyzw": self.quaternion_xyzw,
            "route_progress_m": self.route_progress_m,
            "track_length_m": self.track_length_m,
            "checkpoint_index": self.checkpoint_index,
            "elapsed_s": self.elapsed_s,
            "lateral_offset_m": self.lateral_offset_m,
            "heading_error_rad": self.heading_error_rad,
        }


@dataclass(frozen=True, slots=True)
class Transition:
    episode_id: str
    tick: int
    ticks_advanced: int
    telemetry: Telemetry
    events: tuple[str, ...]
    simulator_info: Mapping[str, Any]

    @classmethod
    def from_wire(cls, value: object, *, lookahead_count: int) -> Transition:
        data = _mapping(value, "result")
        episode_id = data.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProtocolViolation("result.episode_id must be a non-empty string")
        raw_events = data.get("events", [])
        if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
            raise ProtocolViolation("result.events must be an array")
        if any(not isinstance(event, str) for event in raw_events):
            raise ProtocolViolation("result.events must contain strings")
        info = data.get("info", {})
        return cls(
            episode_id=episode_id,
            tick=_integer(data.get("tick"), "result.tick"),
            ticks_advanced=_integer(data.get("ticks_advanced"), "result.ticks_advanced"),
            telemetry=Telemetry.from_wire(data.get("state"), lookahead_count=lookahead_count),
            events=tuple(raw_events),
            simulator_info=_mapping(info, "result.info"),
        )


def request_message(request_id: int, op: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if request_id < 0:
        raise ValueError("request_id must be non-negative")
    if not op:
        raise ValueError("op cannot be empty")
    return {
        "protocol": PROTOCOL_NAME,
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "op": op,
        "params": dict(params),
    }


def response_result(message: object, *, expected_id: int) -> Mapping[str, Any]:
    data = _mapping(message, "response")
    if data.get("protocol") != PROTOCOL_NAME:
        raise ProtocolViolation("response protocol name is missing or unsupported")
    if data.get("v") != PROTOCOL_VERSION:
        raise ProtocolViolation(f"unsupported response protocol version: {data.get('v')!r}")
    if data.get("id") != expected_id:
        raise ProtocolViolation(
            f"response id {data.get('id')!r} does not match request id {expected_id}"
        )
    if data.get("ok") is not True:
        error = _mapping(data.get("error", {}), "response.error")
        code = error.get("code", "remote_error")
        message_text = error.get("message", "simulator rejected the request")
        raise ProtocolViolation(f"{code}: {message_text}")
    return _mapping(data.get("result"), "response.result")


def success_response(request: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Create a response; primarily useful to simulator implementations and tests."""

    return {
        "protocol": PROTOCOL_NAME,
        "v": PROTOCOL_VERSION,
        "id": request.get("id"),
        "ok": True,
        "result": dict(result),
    }


def error_response(
    request_id: object, code: str, message: str, *, version: int = PROTOCOL_VERSION
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "v": version,
        "id": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }
