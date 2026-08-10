"""Hand-written controller used to validate the environment before RL."""

from __future__ import annotations

import numpy as np

from polybot.protocol import Action, Telemetry


class CenterlineController:
    """A deliberately small curvature-aware centreline follower."""

    def __init__(self, *, steering_deadband: float = 0.08) -> None:
        self.steering_deadband = steering_deadband

    def action(self, telemetry: Telemetry) -> Action:
        valid_curvatures = [
            point[3]
            for point, valid in zip(telemetry.lookahead, telemetry.lookahead_mask, strict=True)
            if valid
        ]
        near_curvature = float(np.mean(valid_curvatures[:4])) if valid_curvatures else 0.0
        max_curvature = max((abs(value) for value in valid_curvatures[:8]), default=0.0)

        steering_signal = (
            38.0 * near_curvature
            - 0.22 * telemetry.lateral_offset_m
            - 1.15 * telemetry.heading_error_rad
        )
        if steering_signal > self.steering_deadband:
            steer = 1
        elif steering_signal < -self.steering_deadband:
            steer = -1
        else:
            steer = 0

        speed = telemetry.local_velocity_mps[2]
        target_speed = float(np.clip(34.0 / (1.0 + 70.0 * max_curvature), 12.0, 34.0))
        unstable = (
            abs(telemetry.heading_error_rad) > 0.75
            or abs(telemetry.lateral_offset_m) > telemetry.track_half_width_m
        )
        brake = speed > target_speed + 2.0 or (unstable and speed > 10.0)
        throttle = not brake and speed < target_speed
        return Action(steer=steer, throttle=throttle, brake=brake)

    def policy_action(self, telemetry: Telemetry) -> np.ndarray:
        return self.action(telemetry).to_policy()
