"""Gymnasium environment for PolyTrack-compatible simulators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polybot.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Action,
    ProtocolViolation,
    Telemetry,
    Transition,
    request_message,
    response_result,
)
from polybot.transport import SimulatorTransport


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients kept independent from the game adapter."""

    progress_per_m: float = 0.0
    elapsed_cost_per_s: float = 0.0
    on_track_speed_per_m: float = 0.0
    airborne_speed_per_m: float = 0.0
    airborne_brake_bonus_per_s: float = 0.0
    takeoff_target_speed_mps: float = 45.0
    takeoff_speed_reward_per_mps: float = 0.0
    takeoff_speed_reward_limit: float = 0.0
    imitation_bonus_per_s: float = 18.0
    imitation_position_scale_m: float = 2.0
    imitation_rotation_scale_rad: float = 0.349066  # 20 degrees
    unsafe_speed_penalty_per_m: float = 0.0
    barrier_contact_penalty: float = 0.0
    barrier_collision_impulse_threshold: float = 0.0
    off_track_landing_penalty: float = 0.0
    airborne_spin_penalty_per_rad: float = 0.0
    airborne_spin_deadzone_radps: float = 0.0349066  # 2 degrees per second
    airborne_pitch_deadzone_radps: float = 1.570796  # 90 degrees per second
    airborne_tilt_penalty_per_s: float = 0.0
    airborne_roll_penalty_per_s: float = 0.0
    airborne_pitch_tolerance_rad: float = 1.047198  # 60 degrees
    airborne_roll_limit_rad: float = 1.047198  # 60 degrees
    airborne_roll_timeout_s: float = 0.10
    airborne_roll_failure_penalty: float = 0.0
    ground_slip_tolerance_rad: float = 0.0872665  # 5 degrees
    ground_slip_penalty_per_rad_s: float = 0.0
    checkpoint_bonus: float = 0.0
    checkpoint_fast_bonus: float = 0.0
    checkpoint_target_s: float = 30.0
    finish_bonus: float = 1000.0
    finish_fast_bonus: float = 2000.0
    finish_target_s: float = 22.0
    finish_pace_decay_per_s: float = 1.5
    curriculum_section_bonus: float = 0.0
    crash_penalty: float = 0.0
    stall_penalty: float = 0.0
    off_track_penalty: float = 0.0
    early_off_track_penalty: float = 0.0
    action_change_penalty: float = 0.0
    max_forward_progress_per_step_m: float = 10.0
    max_reverse_progress_per_step_m: float = 3.0
    stall_speed_threshold_mps: float = 5.0
    stall_timeout_s: float = 5.0
    off_track_lateral_ratio: float = 1.05
    off_track_heading_ratio: float = 0.80
    off_track_heading_rad: float = 1.10
    off_track_wall_ride_roll_rad: float = 0.261799  # 15 degrees
    off_track_wall_ride_min_grounded_wheels: int = 3
    off_track_timeout_s: float = 1.25
    off_track_min_grounded_wheels: int = 2
    landing_grace_s: float = 2.0
    early_run_s: float = 20.0


def _has_off_track_evidence(telemetry: Telemetry, config: RewardConfig) -> bool:
    """Reject geometric off-track evidence while the car is airborne."""

    grounded_wheels = sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
    if grounded_wheels < config.off_track_min_grounded_wheels:
        return False
    # Banked wall-ride sections deliberately put the car far from the ghost's
    # centre line. Lateral distance is not valid off-track evidence there.
    if (
        grounded_wheels >= config.off_track_wall_ride_min_grounded_wheels
        and abs(telemetry.roll_rad) >= config.off_track_wall_ride_roll_rad
    ):
        return False
    width = max(0.1, telemetry.track_half_width_m)
    lateral_ratio = abs(telemetry.lateral_offset_m) / width
    return lateral_ratio >= config.off_track_lateral_ratio or (
        lateral_ratio >= config.off_track_heading_ratio
        and abs(telemetry.heading_error_rad) >= config.off_track_heading_rad
    )


def _checkpoint_reward(telemetry: Telemetry, config: RewardConfig) -> float:
    """Reward each checkpoint, with extra credit for a fast average split."""

    checkpoint_number = max(1, telemetry.checkpoint_index)
    target_elapsed_s = config.checkpoint_target_s * checkpoint_number
    pace_factor = float(np.clip(1.0 - telemetry.elapsed_s / target_elapsed_s, 0.0, 1.0))
    return config.checkpoint_bonus + config.checkpoint_fast_bonus * pace_factor


def _finish_reward(telemetry: Telemetry, config: RewardConfig) -> float:
    """Reward a valid finish, with additional credit for completing it quickly."""

    seconds_over_target = max(0.0, telemetry.elapsed_s - config.finish_target_s)
    pace_factor = float(np.exp(-config.finish_pace_decay_per_s * seconds_over_target))
    return config.finish_bonus + config.finish_fast_bonus * pace_factor


def _ghost_pose_reward(
    simulator_info: Mapping[str, Any], config: RewardConfig, dt: float
) -> float:
    """Reward proximity and full 3D orientation agreement with the ghost pose."""

    try:
        position_error = float(simulator_info["ghost_position_error_m"])
        rotation_error = float(simulator_info["ghost_rotation_error_rad"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not np.isfinite(position_error) or not np.isfinite(rotation_error):
        return 0.0
    position_similarity = np.exp(
        -max(0.0, position_error) / config.imitation_position_scale_m
    )
    rotation_similarity = np.exp(
        -max(0.0, rotation_error) / config.imitation_rotation_scale_rad
    )
    return float(config.imitation_bonus_per_s * dt * position_similarity * rotation_similarity)


def _airborne_spin_penalty(telemetry: Telemetry, config: RewardConfig, dt: float) -> float:
    """Penalize strong rotation in the air while tolerating normal jump pitch."""

    if any(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    pitch_rate, yaw_rate, roll_rate = telemetry.angular_velocity_radps
    excess_spin = (
        max(0.0, abs(pitch_rate) - config.airborne_pitch_deadzone_radps)
        + max(0.0, abs(yaw_rate) - config.airborne_spin_deadzone_radps)
        + max(0.0, abs(roll_rate) - config.airborne_spin_deadzone_radps)
    )
    return config.airborne_spin_penalty_per_rad * excess_spin * dt


def _airborne_brake_reward(
    telemetry: Telemetry, action: Action, config: RewardConfig, dt: float
) -> float:
    """Slightly reward braking only while every wheel is off the ground."""

    if not action.brake or any(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    return config.airborne_brake_bonus_per_s * dt


def _airborne_tilt_penalty(telemetry: Telemetry, config: RewardConfig, dt: float) -> float:
    """Penalize tilted and inverted flight even after the car stops rotating."""

    if any(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    roll_error = abs(telemetry.roll_rad) / (np.pi / 2.0)
    pitch_error = max(0.0, abs(telemetry.pitch_rad) - config.airborne_pitch_tolerance_rad) / (
        np.pi / 2.0
    )
    roll_error = min(2.0, roll_error)
    pitch_error = min(2.0, pitch_error)
    return (
        config.airborne_roll_penalty_per_s * roll_error
        + config.airborne_tilt_penalty_per_s * pitch_error
    ) * dt


def _ground_slip_penalty(
    telemetry: Telemetry, config: RewardConfig, dt: float
) -> float:
    """Penalize tyre-scrubbing slip only while all four wheels are grounded."""

    if not all(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    lateral_speed = abs(telemetry.local_velocity_mps[0])
    forward_speed = abs(telemetry.local_velocity_mps[2])
    slip_angle = float(np.arctan2(lateral_speed, max(forward_speed, 1e-6)))
    excess_slip = max(0.0, slip_angle - config.ground_slip_tolerance_rad)
    return config.ground_slip_penalty_per_rad_s * excess_slip * dt


class PolyTrackEnv(gym.Env[np.ndarray, np.ndarray]):
    """Synchronous Gymnasium wrapper around a PolyTrack simulator adapter."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: SimulatorTransport,
        *,
        track_id: str = "mock/gentle-s",
        lookahead_count: int = 12,
        frame_skip: int = 4,
        max_episode_steps: int = 2_000,
        reward_config: RewardConfig | None = None,
        request_timeout_s: float = 10.0,
        curriculum_last_fraction: float = 0.0,
        curriculum_probability: float = 0.0,
        curriculum_start_ratio: float | None = None,
        curriculum_end_ratio: float | None = None,
        curriculum_start_s: float | None = None,
        curriculum_end_s: float | None = None,
    ) -> None:
        super().__init__()
        if lookahead_count < 1:
            raise ValueError("lookahead_count must be positive")
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not track_id:
            raise ValueError("track_id cannot be empty")
        if not 0.0 <= curriculum_last_fraction <= 1.0:
            raise ValueError("curriculum_last_fraction must be in [0, 1]")
        if not 0.0 <= curriculum_probability <= 1.0:
            raise ValueError("curriculum_probability must be in [0, 1]")
        if (curriculum_start_ratio is None) != (curriculum_end_ratio is None):
            raise ValueError("curriculum section start and end must be provided together")
        if curriculum_start_ratio is not None and not (
            0.0 <= curriculum_start_ratio < curriculum_end_ratio <= 1.0
        ):
            raise ValueError("curriculum section must satisfy 0 <= start < end <= 1")
        if (curriculum_start_s is None) != (curriculum_end_s is None):
            raise ValueError("timed curriculum start and end must be provided together")
        if curriculum_start_s is not None and not (
            0.0 <= curriculum_start_s < curriculum_end_s
        ):
            raise ValueError("timed curriculum must satisfy 0 <= start < end")
        if curriculum_start_ratio is not None and curriculum_start_s is not None:
            raise ValueError("progress and timed curriculum cannot be combined")

        self.transport = transport
        self.track_id = track_id
        self.lookahead_count = lookahead_count
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.request_timeout_s = request_timeout_s
        self.curriculum_last_fraction = curriculum_last_fraction
        self.curriculum_probability = curriculum_probability
        self.curriculum_start_ratio = curriculum_start_ratio
        self.curriculum_end_ratio = curriculum_end_ratio
        self.curriculum_start_s = curriculum_start_s
        self.curriculum_end_s = curriculum_end_s

        self.action_space = spaces.MultiDiscrete(np.asarray([3, 2, 2], dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(Telemetry.vector_size(lookahead_count),),
            dtype=np.float32,
        )

        self._next_request_id = 0
        self._handshake_complete = False
        self._closed = False
        self._episode_id: str | None = None
        self._episode_steps = 0
        self._previous_progress_m = 0.0
        self._previous_action = Action()
        self._episode_done = True
        self._stationary_s = 0.0
        self._off_track_s = 0.0
        self._barrier_contact_s = 0.0
        self._airborne_roll_s = 0.0
        self._landing_grace_s = 0.0
        self._was_airborne = False
        self.latest_telemetry: Telemetry | None = None
        self.simulator_capabilities: Mapping[str, Any] = {}

    def _exchange(self, op: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("environment is closed")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = request_message(request_id, op, params)
        response = self.transport.request(request, timeout_s=self.request_timeout_s)
        return response_result(response, expected_id=request_id)

    def _handshake(self) -> None:
        if self._handshake_complete:
            return
        result = self._exchange(
            "hello",
            {
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "lookahead_count": self.lookahead_count,
            },
        )
        if result.get("protocol") != PROTOCOL_NAME:
            raise ProtocolViolation("simulator hello returned an unsupported protocol")
        if result.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolViolation("simulator hello returned an unsupported protocol version")
        if result.get("lookahead_count") != self.lookahead_count:
            raise ProtocolViolation("simulator cannot provide the requested lookahead count")
        fixed_dt_s = result.get("fixed_dt_s")
        if (
            isinstance(fixed_dt_s, bool)
            or not isinstance(fixed_dt_s, (int, float))
            or not 0 < fixed_dt_s <= 1
        ):
            raise ProtocolViolation("simulator fixed_dt_s must be in (0, 1]")
        max_ticks = result.get("max_ticks_per_step")
        if (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or max_ticks < self.frame_skip
        ):
            raise ProtocolViolation("simulator cannot advance the requested frame_skip")
        self.simulator_capabilities = dict(result)
        self._handshake_complete = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._handshake()
        options = options or {}
        track_id = options.get("track_id", self.track_id)
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("options['track_id'] must be a non-empty string")
        simulator_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )
        start_progress_ratio = 0.0
        if self.curriculum_start_ratio is not None:
            start_progress_ratio = self.curriculum_start_ratio
        elif (
            self.curriculum_last_fraction > 0.0
            and self.np_random.random() < self.curriculum_probability
        ):
            start_progress_ratio = float(
                self.np_random.uniform(1.0 - self.curriculum_last_fraction, 0.95)
            )
        result = self._exchange(
            "reset",
            {
                "seed": simulator_seed,
                "track_id": track_id,
                "start_progress_ratio": start_progress_ratio,
                "start_time_s": self.curriculum_start_s,
                "native_restart": self.curriculum_start_s is None,
            },
        )
        transition = Transition.from_wire(result, lookahead_count=self.lookahead_count)
        if transition.ticks_advanced != 0:
            raise ProtocolViolation("reset must not advance simulation ticks")

        self._episode_id = transition.episode_id
        self._episode_steps = 0
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._previous_action = transition.telemetry.previous_action
        self._episode_done = False
        self._stationary_s = 0.0
        self._off_track_s = 0.0
        self._barrier_contact_s = 0.0
        self._airborne_roll_s = 0.0
        self._landing_grace_s = 0.0
        self._was_airborne = False
        self.latest_telemetry = transition.telemetry
        observation = transition.telemetry.to_vector()
        info = self._info(transition, reward_terms=None, simulator_seed=simulator_seed)
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_id is None or self._episode_done:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside {self.action_space}")
        decoded_action = Action.from_policy(action)
        result = self._exchange(
            "step",
            {
                "episode_id": self._episode_id,
                "action": decoded_action.to_wire(),
                "ticks": self.frame_skip,
            },
        )
        transition = Transition.from_wire(result, lookahead_count=self.lookahead_count)
        if transition.episode_id != self._episode_id:
            raise ProtocolViolation("simulator returned a stale or unexpected episode_id")
        if transition.ticks_advanced > self.frame_skip:
            raise ProtocolViolation("simulator advanced more ticks than requested")

        dt = transition.ticks_advanced * float(self.simulator_capabilities["fixed_dt_s"])
        telemetry = transition.telemetry
        width = max(0.1, telemetry.track_half_width_m)
        lateral_ratio = abs(telemetry.lateral_offset_m) / width
        grounded_wheels = sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
        airborne = grounded_wheels < self.reward_config.off_track_min_grounded_wheels
        clean_takeoff = airborne and not self._was_airborne
        off_track_landing = (
            not airborne
            and self._was_airborne
            and lateral_ratio >= self.reward_config.off_track_lateral_ratio
        )
        if airborne != self._was_airborne:
            self._landing_grace_s = self.reward_config.landing_grace_s
        else:
            self._landing_grace_s = max(0.0, self._landing_grace_s - dt)
        self._was_airborne = airborne
        landing_grace = self._landing_grace_s > 0.0

        speed = float(np.linalg.norm(telemetry.local_velocity_mps))
        if landing_grace:
            self._stationary_s = 0.0
        elif speed < self.reward_config.stall_speed_threshold_mps:
            self._stationary_s += dt
        else:
            self._stationary_s = 0.0
        stalled = self._stationary_s >= self.reward_config.stall_timeout_s

        off_track_candidate = _has_off_track_evidence(telemetry, self.reward_config)
        if off_track_candidate and not landing_grace:
            self._off_track_s += dt
        else:
            self._off_track_s = max(0.0, self._off_track_s - 2.0 * dt)
        off_track = self._off_track_s >= self.reward_config.off_track_timeout_s
        early_off_track = off_track and telemetry.elapsed_s <= self.reward_config.early_run_s

        raw_collision_impulses = transition.simulator_info.get("collision_impulses", ())
        try:
            collision_impulse = float(
                np.linalg.norm(np.asarray(raw_collision_impulses, dtype=np.float64))
            )
        except (TypeError, ValueError):
            collision_impulse = 0.0
        if not np.isfinite(collision_impulse):
            collision_impulse = 0.0
        barrier_contact = collision_impulse > self.reward_config.barrier_collision_impulse_threshold
        self._barrier_contact_s = dt if barrier_contact else 0.0

        fully_airborne = grounded_wheels == 0
        if fully_airborne and abs(telemetry.roll_rad) >= self.reward_config.airborne_roll_limit_rad:
            self._airborne_roll_s += dt
        else:
            self._airborne_roll_s = max(0.0, self._airborne_roll_s - 2.0 * dt)
        airborne_roll_failure = self._airborne_roll_s >= self.reward_config.airborne_roll_timeout_s
        curriculum_section_complete = bool(
            (
                self.curriculum_end_ratio is not None
                and telemetry.route_progress_m
                >= telemetry.track_length_m * self.curriculum_end_ratio
            )
            or (
                self.curriculum_end_s is not None
                and telemetry.elapsed_s >= self.curriculum_end_s - self.curriculum_start_s
            )
        )

        reward, reward_terms = self._reward(
            transition,
            decoded_action,
            stalled=stalled,
            off_track=off_track,
            early_off_track=early_off_track,
            barrier_contact=barrier_contact,
            off_track_landing=off_track_landing,
            clean_takeoff=clean_takeoff,
            airborne_roll_failure=airborne_roll_failure,
            curriculum_section_complete=curriculum_section_complete,
        )
        self._episode_steps += 1
        events = set(transition.events)
        crash = "crash" in events and not landing_grace
        terminated = (
            "finish" in events
            or crash
            or stalled
            or off_track
            or barrier_contact
            or airborne_roll_failure
            or curriculum_section_complete
        )
        truncated = "time_limit" in events or self._episode_steps >= self.max_episode_steps
        self._episode_done = terminated or truncated
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._previous_action = decoded_action
        self.latest_telemetry = transition.telemetry

        observation = transition.telemetry.to_vector()
        info = self._info(transition, reward_terms=reward_terms)
        if stalled:
            info["events"] = (*transition.events, "stalled")
            info["stationary_s"] = self._stationary_s
        if off_track:
            info["events"] = tuple(dict.fromkeys((*info["events"], "off_track")))
            info["off_track_s"] = self._off_track_s
            info["off_track_lateral_ratio"] = lateral_ratio
            info["early_off_track"] = early_off_track
        if barrier_contact:
            info["events"] = tuple(dict.fromkeys((*info["events"], "barrier_contact")))
            info["barrier_contact_s"] = self._barrier_contact_s
        if off_track_landing:
            info["events"] = tuple(dict.fromkeys((*info["events"], "off_track_landing")))
        if airborne_roll_failure:
            info["events"] = tuple(dict.fromkeys((*info["events"], "airborne_roll_failure")))
            info["airborne_roll_s"] = self._airborne_roll_s
        if curriculum_section_complete:
            info["events"] = tuple(
                dict.fromkeys((*info["events"], "curriculum_section_complete"))
            )
        if landing_grace:
            info["landing_grace_s"] = self._landing_grace_s
        if truncated and "time_limit" not in events:
            info["wrapper_time_limit"] = True
        return observation, reward, terminated, truncated, info

    def _reward(
        self,
        transition: Transition,
        action: Action,
        *,
        stalled: bool = False,
        off_track: bool = False,
        early_off_track: bool = False,
        barrier_contact: bool = False,
        off_track_landing: bool = False,
        clean_takeoff: bool = False,
        airborne_roll_failure: bool = False,
        curriculum_section_complete: bool = False,
    ) -> tuple[float, dict[str, float]]:
        config = self.reward_config
        raw_delta = transition.telemetry.route_progress_m - self._previous_progress_m
        progress_delta = float(
            np.clip(
                raw_delta,
                -config.max_reverse_progress_per_step_m,
                config.max_forward_progress_per_step_m,
            )
        )
        events = transition.events
        telemetry = transition.telemetry
        dt = transition.ticks_advanced * float(self.simulator_capabilities["fixed_dt_s"])
        forward_speed = max(0.0, telemetry.local_velocity_mps[2])
        width = max(0.1, telemetry.track_half_width_m)
        center_factor = float(np.clip(1.0 - abs(telemetry.lateral_offset_m) / width, 0, 1))
        heading_factor = max(0.0, float(np.cos(telemetry.heading_error_rad)))
        on_track_factor = center_factor * heading_factor
        airborne = (
            sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
            < config.off_track_min_grounded_wheels
        )
        airborne_stability = float(
            np.clip(1.0 - abs(telemetry.roll_rad) / (np.pi / 4.0), 0.0, 1.0)
            * np.clip(
                1.0
                - max(
                    0.0,
                    abs(telemetry.pitch_rad) - config.airborne_pitch_tolerance_rad,
                )
                / (np.pi / 4.0),
                0.0,
                1.0,
            )
        )
        distance_at_speed = forward_speed * dt
        takeoff_speed_reward = 0.0
        if clean_takeoff:
            takeoff_speed_reward = float(
                np.clip(
                    (forward_speed - config.takeoff_target_speed_mps)
                    * config.takeoff_speed_reward_per_mps,
                    -config.takeoff_speed_reward_limit,
                    config.takeoff_speed_reward_limit,
                )
            )
        imitation_reward = _ghost_pose_reward(transition.simulator_info, config, dt)
        terms = {
            "progress": config.progress_per_m * progress_delta,
            "elapsed": config.elapsed_cost_per_s * dt,
            "on_track_speed": config.on_track_speed_per_m * distance_at_speed * on_track_factor,
            "airborne_speed": config.airborne_speed_per_m * distance_at_speed * airborne_stability
            if airborne
            else 0.0,
            "airborne_brake": _airborne_brake_reward(telemetry, action, config, dt),
            "takeoff_speed": takeoff_speed_reward,
            "ghost_imitation": imitation_reward,
            "unsafe_speed": config.unsafe_speed_penalty_per_m
            * distance_at_speed
            * (1.0 - on_track_factor),
            "airborne_spin": _airborne_spin_penalty(telemetry, config, dt),
            "airborne_tilt": _airborne_tilt_penalty(telemetry, config, dt),
            "ground_slip": _ground_slip_penalty(telemetry, config, dt),
            "barrier_contact": (config.barrier_contact_penalty if barrier_contact else 0.0),
            "off_track_landing": (config.off_track_landing_penalty if off_track_landing else 0.0),
            "airborne_roll_failure": (
                config.airborne_roll_failure_penalty if airborne_roll_failure else 0.0
            ),
            "checkpoint": _checkpoint_reward(telemetry, config) * events.count("checkpoint"),
            "finish": _finish_reward(telemetry, config) if "finish" in events else 0.0,
            "curriculum_section": (
                config.curriculum_section_bonus if curriculum_section_complete else 0.0
            ),
            "crash": config.crash_penalty if "crash" in events else 0.0,
            "stall": config.stall_penalty if stalled else 0.0,
            "off_track": (
                config.early_off_track_penalty
                if early_off_track
                else config.off_track_penalty
                if off_track
                else 0.0
            ),
            "action_change": config.action_change_penalty
            * (
                abs(action.steer - self._previous_action.steer)
                + int(action.throttle != self._previous_action.throttle)
                + int(action.brake != self._previous_action.brake)
            ),
        }
        reward = float(sum(terms.values()))
        if not np.isfinite(reward):
            raise ProtocolViolation("reward became non-finite")
        return reward, terms

    def _info(
        self,
        transition: Transition,
        *,
        reward_terms: Mapping[str, float] | None,
        simulator_seed: int | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            **transition.telemetry.to_info(),
            "tick": transition.tick,
            "ticks_advanced": transition.ticks_advanced,
            "events": transition.events,
            "simulator_info": dict(transition.simulator_info),
        }
        if reward_terms is not None:
            info["reward_terms"] = dict(reward_terms)
        if simulator_seed is not None:
            info["simulator_seed"] = simulator_seed
        return info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.transport.close()
        super().close()
