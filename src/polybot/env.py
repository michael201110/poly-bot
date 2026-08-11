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

    progress_per_m: float = 0.10
    elapsed_cost_per_s: float = 0.0
    on_track_speed_per_m: float = 0.06
    unsafe_speed_penalty_per_m: float = -0.08
    barrier_proximity_penalty_per_m: float = -1.00
    barrier_proximity_start_ratio: float = 0.55
    barrier_launch_penalty: float = -15.0
    barrier_contact_penalty: float = -25.0
    barrier_contact_ratio: float = 0.90
    barrier_contact_timeout_s: float = 0.25
    airborne_spin_penalty_per_rad: float = -0.30
    airborne_spin_deadzone_radps: float = 0.0349066  # 2 degrees per second
    airborne_tilt_penalty_per_s: float = -15.0
    airborne_roll_penalty_per_s: float = -40.0
    airborne_pitch_tolerance_rad: float = 0.523599  # 30 degrees
    checkpoint_bonus: float = 10.0
    checkpoint_fast_bonus: float = 10.0
    checkpoint_target_s: float = 30.0
    finish_bonus: float = 100.0
    crash_penalty: float = -10.0
    stall_penalty: float = -5.0
    off_track_penalty: float = -5.0
    early_off_track_penalty: float = -15.0
    action_change_penalty: float = -0.002
    max_forward_progress_per_step_m: float = 10.0
    max_reverse_progress_per_step_m: float = 3.0
    stall_speed_threshold_mps: float = 5.0
    stall_timeout_s: float = 5.0
    off_track_lateral_ratio: float = 1.05
    off_track_heading_ratio: float = 0.80
    off_track_heading_rad: float = 1.10
    off_track_timeout_s: float = 1.25
    off_track_min_grounded_wheels: int = 2
    landing_grace_s: float = 2.0
    early_run_s: float = 20.0


def _has_off_track_evidence(telemetry: Telemetry, config: RewardConfig) -> bool:
    """Reject geometric off-track evidence while the car is airborne."""

    grounded_wheels = sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
    if grounded_wheels < config.off_track_min_grounded_wheels:
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


def _airborne_spin_penalty(
    telemetry: Telemetry, config: RewardConfig, dt: float
) -> float:
    """Penalize strong rotation in the air while tolerating normal jump pitch."""

    if sum(contact >= 0.5 for contact in telemetry.wheel_contacts) >= 2:
        return 0.0
    angular_speed = float(np.linalg.norm(telemetry.angular_velocity_radps))
    excess_spin = max(0.0, angular_speed - config.airborne_spin_deadzone_radps)
    return config.airborne_spin_penalty_per_rad * excess_spin * dt


def _airborne_tilt_penalty(
    telemetry: Telemetry, config: RewardConfig, dt: float
) -> float:
    """Penalize tilted and inverted flight even after the car stops rotating."""

    if sum(contact >= 0.5 for contact in telemetry.wheel_contacts) >= 2:
        return 0.0
    roll_error = abs(telemetry.roll_rad) / (np.pi / 2.0)
    pitch_error = max(
        0.0, abs(telemetry.pitch_rad) - config.airborne_pitch_tolerance_rad
    ) / (np.pi / 2.0)
    roll_error = min(2.0, roll_error)
    pitch_error = min(2.0, pitch_error)
    return (
        config.airborne_roll_penalty_per_s * roll_error
        + config.airborne_tilt_penalty_per_s * pitch_error
    ) * dt


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

        self.transport = transport
        self.track_id = track_id
        self.lookahead_count = lookahead_count
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.request_timeout_s = request_timeout_s

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
        result = self._exchange(
            "reset",
            {
                "seed": simulator_seed,
                "track_id": track_id,
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
        barrier_launch = (
            airborne
            and not self._was_airborne
            and lateral_ratio >= self.reward_config.barrier_proximity_start_ratio
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

        barrier_contact_candidate = (
            grounded_wheels >= self.reward_config.off_track_min_grounded_wheels
            and lateral_ratio >= self.reward_config.barrier_contact_ratio
        )
        if barrier_contact_candidate and not landing_grace:
            self._barrier_contact_s += dt
        else:
            self._barrier_contact_s = max(0.0, self._barrier_contact_s - 2.0 * dt)
        barrier_contact = (
            self._barrier_contact_s >= self.reward_config.barrier_contact_timeout_s
        )

        reward, reward_terms = self._reward(
            transition,
            decoded_action,
            stalled=stalled,
            off_track=off_track,
            early_off_track=early_off_track,
            barrier_launch=barrier_launch,
            barrier_contact=barrier_contact,
        )
        self._episode_steps += 1
        events = set(transition.events)
        crash = "crash" in events and not landing_grace
        terminated = "finish" in events or crash or stalled or off_track or barrier_contact
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
            info["events"] = tuple(
                dict.fromkeys((*info["events"], "barrier_contact"))
            )
            info["barrier_contact_s"] = self._barrier_contact_s
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
        barrier_launch: bool = False,
        barrier_contact: bool = False,
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
        lateral_ratio = abs(telemetry.lateral_offset_m) / width
        barrier_factor = float(
            np.clip(
                (lateral_ratio - config.barrier_proximity_start_ratio)
                / (1.0 - config.barrier_proximity_start_ratio),
                0.0,
                1.0,
            )
        )
        distance_at_speed = forward_speed * dt
        terms = {
            "progress": config.progress_per_m * progress_delta,
            "elapsed": config.elapsed_cost_per_s * dt,
            "on_track_speed": config.on_track_speed_per_m
            * distance_at_speed
            * on_track_factor,
            "unsafe_speed": config.unsafe_speed_penalty_per_m
            * distance_at_speed
            * (1.0 - on_track_factor),
            "barrier_proximity": config.barrier_proximity_penalty_per_m
            * distance_at_speed
            * barrier_factor**2,
            "airborne_spin": _airborne_spin_penalty(telemetry, config, dt),
            "airborne_tilt": _airborne_tilt_penalty(telemetry, config, dt),
            "barrier_launch": config.barrier_launch_penalty if barrier_launch else 0.0,
            "barrier_contact": (
                config.barrier_contact_penalty if barrier_contact else 0.0
            ),
            "checkpoint": _checkpoint_reward(telemetry, config)
            * events.count("checkpoint"),
            "finish": config.finish_bonus if "finish" in events else 0.0,
            "crash": config.crash_penalty if "crash" in events else 0.0,
            "stall": config.stall_penalty if stalled else 0.0,
            "off_track": (
                config.early_off_track_penalty
                if early_off_track
                else config.off_track_penalty if off_track else 0.0
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
