"""Hand-written controller used to validate the environment before RL."""

from __future__ import annotations

import numpy as np

from polybot.protocol import Action, Telemetry


class CenterlineController:
    """A deliberately small curvature-aware centreline follower."""

    def __init__(
        self,
        *,
        steering_deadband: float = 0.18,
        steering_sign: int = 1,
        max_speed_mps: float = 18.0,
    ) -> None:
        if steering_sign not in (-1, 1):
            raise ValueError("steering_sign must be -1 or 1")
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        self.steering_deadband = steering_deadband
        self.steering_sign = steering_sign
        self.max_speed_mps = max_speed_mps

    def action(self, telemetry: Telemetry) -> Action:
        valid_curvatures = [
            point[3]
            for point, valid in zip(telemetry.lookahead, telemetry.lookahead_mask, strict=True)
            if valid
        ]
        max_curvature = max((abs(value) for value in valid_curvatures[:8]), default=0.0)

        valid_points = [
            point
            for point, valid in zip(telemetry.lookahead, telemetry.lookahead_mask, strict=True)
            if valid
        ]
        pursuit = valid_points[min(4, len(valid_points) - 1)] if valid_points else (20, 0, 0, 0)
        pursuit_angle = float(np.arctan2(pursuit[1], max(2.0, pursuit[0])))

        steering_signal = (
            1.2 * pursuit_angle
            - 0.05 * telemetry.lateral_offset_m
            - 0.50 * telemetry.heading_error_rad
        ) * self.steering_sign
        if steering_signal > self.steering_deadband:
            steer = 1
        elif steering_signal < -self.steering_deadband:
            steer = -1
        else:
            steer = 0

        speed = telemetry.local_velocity_mps[2]
        turn_demand = max(abs(pursuit_angle), 8.0 * max_curvature)
        target_speed = float(
            np.clip(
                self.max_speed_mps / (1.0 + 3.5 * turn_demand),
                min(6.0, self.max_speed_mps),
                self.max_speed_mps,
            )
        )
        unstable = (
            abs(telemetry.heading_error_rad) > 0.45
            or abs(telemetry.lateral_offset_m) > telemetry.track_half_width_m * 0.65
        )
        brake = speed > target_speed + 0.75 or (unstable and speed > 5.0)
        throttle = not brake and speed < target_speed
        return Action(steer=steer, throttle=throttle, brake=brake)

    def policy_action(self, telemetry: Telemetry) -> np.ndarray:
        return self.action(telemetry).to_policy()
