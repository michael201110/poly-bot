"""Shared policy observation schema for every algorithm."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from polybot.protocol import Telemetry

SCHEMA = "polybot.observation.v2"
FEATURE_SCHEMA = "polybot.observation-features.v1"
CONTROLLER_STATE_SCHEMA = "polybot.observation.v2.pwm-state"
TRAINING_STATE_SCHEMA = "polybot.observation.v2.training-state"
TRAINING_STATE_SIZE = 12


def schema_for(config: object) -> str:
    settings = getattr(config, "grtqc", None)
    if getattr(settings, "critic_environment_state", False):
        return TRAINING_STATE_SCHEMA
    return CONTROLLER_STATE_SCHEMA if getattr(settings, "critic_controller_state", False) else SCHEMA


def extra_size(config: object) -> int:
    settings = getattr(config, "grtqc", None)
    return (4 * bool(getattr(settings, "critic_controller_state", False))
            + TRAINING_STATE_SIZE * bool(getattr(settings, "critic_environment_state", False)))


def observe(telemetry: Telemetry) -> np.ndarray:
    """45 normalized state values, then 4 values and 1 mask per lookahead point."""
    return telemetry.to_vector()


def describe_observation(
    telemetry: Telemetry,
    observation: np.ndarray,
    *,
    extra_features: Sequence[tuple[str, str, str, str, float]] = (),
) -> list[dict[str, Any]]:
    """Describe the exact policy vector with source values and display metadata."""
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    definitions: list[tuple[str, str, str, str, float, float, float]] = []

    def add(
        key: str, label: str, group: str, unit: str, raw: float,
        minimum: float, maximum: float,
    ) -> None:
        definitions.append((key, label, group, unit, float(raw), minimum, maximum))

    axes = ("right", "up", "forward")
    for index, axis in enumerate(axes):
        add(f"velocity.{axis}", f"{axis.title()} velocity", "vehicle", "m/s",
            telemetry.local_velocity_mps[index], -2.0, 2.0)
    for index, axis in enumerate(axes):
        add(f"acceleration.{axis}", f"{axis.title()} acceleration", "vehicle", "m/s²",
            telemetry.local_acceleration_mps2[index], -2.0, 2.0)
    for index, axis in enumerate(axes):
        label = "Yaw rate (about up/Y)" if axis == "up" else f"Angular velocity {axis.upper()}"
        add(f"angular_velocity.{axis}", label, "vehicle", "rad/s",
            telemetry.angular_velocity_radps[index], -5.0, 5.0)
    for index, axis in enumerate(axes):
        add(f"up_vector.{axis}", f"Up vector {axis.upper()}", "vehicle", "normalized",
            telemetry.up_vector[index], -1.0, 1.0)
    add("route.progress", "Route progress", "track", "m", telemetry.route_progress_m, -1.0, 2.0)
    add("route.lateral_offset", "Lateral offset", "track", "m",
        telemetry.lateral_offset_m, -5.0, 5.0)
    add("route.heading_error", "Heading error", "track", "rad",
        telemetry.heading_error_rad, -1.0, 1.0)
    add("vehicle.pitch", "Pitch", "vehicle", "rad", telemetry.pitch_rad, -1.0, 1.0)
    add("vehicle.roll", "Roll", "vehicle", "rad", telemetry.roll_rad, -1.0, 1.0)
    wheel_names = ("wheel_1", "wheel_2", "wheel_3", "wheel_4")
    for index, wheel in enumerate(wheel_names):
        add(f"{wheel}.contact", f"Wheel {index + 1} contact", "vehicle", "contact",
            telemetry.wheel_contacts[index], 0.0, 1.0)
    for index, wheel in enumerate(wheel_names):
        add(f"{wheel}.suspension_length", f"Wheel {index + 1} suspension length",
            "vehicle", "m", telemetry.suspension_lengths_m[index], -1.0, 1.0)
    for index, wheel in enumerate(wheel_names):
        add(f"{wheel}.suspension_velocity", f"Wheel {index + 1} suspension velocity",
            "vehicle", "m/s", telemetry.suspension_velocities_mps[index], -2.0, 2.0)
    for index, wheel in enumerate(wheel_names):
        add(f"{wheel}.skid", f"Wheel {index + 1} skid", "vehicle", "m/s",
            telemetry.wheel_skids[index], -2.0, 2.0)
    add("controls.actual_steering", "Actual steering", "controls", "normalized",
        telemetry.actual_steering, -1.0, 1.0)
    for index, axis in enumerate(axes):
        add(f"ghost.relative_position.{axis}", f"Ghost relative position {axis}",
            "ghost", "m", telemetry.ghost_relative_position_m[index], -5.0, 5.0)
    add("ghost.heading_error", "Ghost heading error", "ghost", "rad",
        telemetry.ghost_heading_error_rad, -1.0, 1.0)
    add("ghost.target_speed", "Ghost target speed", "ghost", "m/s",
        telemetry.ghost_target_speed_mps, 0.0, 2.0)
    for label, value in (
        ("Expert steer", telemetry.expert_action.steer),
        ("Expert throttle", telemetry.expert_action.throttle),
        ("Expert brake", telemetry.expert_action.brake),
        ("Previous steer", telemetry.previous_action.steer),
        ("Previous throttle", telemetry.previous_action.throttle),
        ("Previous brake", telemetry.previous_action.brake),
    ):
        key = label.lower().replace(" ", "_")
        add(f"action.{key}", label, "control_history", "normalized", value, -1.0, 1.0)
    for point_index, (forward, right, up, curvature) in enumerate(telemetry.lookahead):
        for key, label, value in (
            ("forward", "forward distance", forward),
            ("right", "right offset", right),
            ("up", "height offset", up),
            ("curvature", "curvature", curvature),
        ):
            unit = "1/m" if key == "curvature" else "m"
            add(
                f"lookahead.{point_index}.{key}",
                f"Lookahead {point_index + 1} {label}",
                "lookahead",
                unit,
                value,
                -5.0,
                5.0,
            )
    for point_index, mask in enumerate(telemetry.lookahead_mask):
        add(
            f"lookahead.{point_index}.mask",
            f"Lookahead {point_index + 1} available",
            "lookahead",
            "mask",
            mask,
            0.0,
            1.0,
        )
    for key, label, group, unit, raw in extra_features:
        add(key, label, group, unit, raw, -5.0, 5.0)

    if len(definitions) != values.size:
        raise ValueError(
            f"observation feature metadata has {len(definitions)} entries for "
            f"a {values.size}-value vector"
        )
    result: list[dict[str, Any]] = []
    for index, (key, label, group, unit, raw, minimum, maximum) in enumerate(definitions):
        normalized = float(values[index])
        if not math.isfinite(normalized) or not math.isfinite(raw):
            raise ValueError(f"observation feature {key} is not finite")
        result.append({
            "index": index,
            "key": key,
            "label": label,
            "group": group,
            "unit": unit,
            "raw_value": raw,
            "value": normalized,
            "minimum": minimum,
            "maximum": maximum,
        })
    return result


def size(lookahead_count: int) -> int:
    return Telemetry.vector_size(lookahead_count)
