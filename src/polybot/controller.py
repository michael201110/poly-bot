"""Hand-written controller used to validate the environment before RL."""

from __future__ import annotations

import numpy as np

from polybot.protocol import Action, Telemetry


class CenterlineController:
    """A deliberately small curvature-aware centreline follower."""

    def __init__(self, *, steering_deadband: float = 0.12) -> None:
        self.steering_deadband = steering_deadband

    def action(self, telemetry: Telemetry) -> Action:
        valid_curvatures = [
            point[3]
            for point, valid in zip(telemetry.lookahead, telemetry.lookahead_mask, strict=True)
            if valid
        ]
        max_curvature = max((abs(value) for value in valid_curvatures[:8]), default=0.0)

        valid_points = [
            point
            for point, valid in zip(
                telemetry.lookahead, telemetry.lookahead_mask, strict=True
            )
            if valid
        ]
        pursuit = valid_points[min(2, len(valid_points) - 1)] if valid_points else (10, 0, 0, 0)
        pursuit_angle = float(np.arctan2(pursuit[1], max(2.0, pursuit[0])))

        steering_signal = (
            1.8 * pursuit_angle
            - 0.08 * telemetry.lateral_offset_m
            - 0.70 * telemetry.heading_error_rad
        )
        if steering_signal > self.steering_deadband:
            steer = 1
        elif steering_signal < -self.steering_deadband:
            steer = -1
        else:
            steer = 0

        speed = telemetry.local_velocity_mps[2]
        turn_demand = max(abs(pursuit_angle), 8.0 * max_curvature)
        target_speed = float(np.clip(18.0 / (1.0 + 3.5 * turn_demand), 6.0, 18.0))
        unstable = (
            abs(telemetry.heading_error_rad) > 0.45
            or abs(telemetry.lateral_offset_m) > telemetry.track_half_width_m * 0.65
        )
        brake = speed > target_speed + 1.0 or (unstable and speed > 7.0)
        throttle = not brake and speed < target_speed
        return Action(steer=steer, throttle=throttle, brake=brake)

    def policy_action(self, telemetry: Telemetry) -> np.ndarray:
        return self.action(telemetry).to_policy()
