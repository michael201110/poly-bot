"""Gymnasium environment for PolyTrack-compatible simulators."""

from __future__ import annotations

import time
from collections.abc import Mapping
from itertools import groupby
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polybot.control.actions import ActionAdapter, AppliedAction, ControlDemand, DigitalActionAdapter
from polybot.environment.observations import observe, size
from polybot.environment.rewards import (
    RewardConfig,
    RewardContext,
    _has_off_track_evidence,
    _reference_corridor_width,
    calculate_reward,
)
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
        max_episode_s: float | None = None,
        reward_config: RewardConfig | None = None,
        request_timeout_s: float = 10.0,
        curriculum_last_fraction: float = 0.0,
        curriculum_probability: float = 0.0,
        curriculum_lead_in_ratio: float = 0.05,
        curriculum_spawn_ratio: float | None = None,
        curriculum_start_ratio: float | None = None,
        curriculum_end_ratio: float | None = None,
        curriculum_start_s: float | None = None,
        curriculum_end_s: float | None = None,
        curriculum_random_quarters: bool = False,
        action_adapter: ActionAdapter | None = None,
    ) -> None:
        super().__init__()
        if lookahead_count < 1:
            raise ValueError("lookahead_count must be positive")
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if max_episode_s is not None and max_episode_s <= 0:
            raise ValueError("max_episode_s must be positive when provided")
        if not track_id:
            raise ValueError("track_id cannot be empty")
        if not 0.0 <= curriculum_last_fraction <= 1.0:
            raise ValueError("curriculum_last_fraction must be in [0, 1]")
        if not 0.0 <= curriculum_probability <= 1.0:
            raise ValueError("curriculum_probability must be in [0, 1]")
        if not 0.0 <= curriculum_lead_in_ratio < 1.0:
            raise ValueError("curriculum_lead_in_ratio must be in [0, 1)")
        if curriculum_spawn_ratio is None and curriculum_start_ratio is not None:
            curriculum_spawn_ratio = max(0.0, curriculum_start_ratio - curriculum_lead_in_ratio)
        if (curriculum_spawn_ratio is None) != (curriculum_start_ratio is None):
            raise ValueError("curriculum spawn and target start must be provided together")
        if (curriculum_start_ratio is None) != (curriculum_end_ratio is None):
            raise ValueError("curriculum section start and end must be provided together")
        if curriculum_spawn_ratio is not None and not (
            0.0 <= curriculum_spawn_ratio <= curriculum_start_ratio < curriculum_end_ratio <= 1.0
        ):
            raise ValueError("curriculum must satisfy 0 <= spawn <= start < end <= 1")
        if curriculum_start_ratio is not None and not (
            0.0 <= curriculum_start_ratio < curriculum_end_ratio <= 1.0
        ):
            raise ValueError("curriculum section must satisfy 0 <= start < end <= 1")
        if (curriculum_start_s is None) != (curriculum_end_s is None):
            raise ValueError("timed curriculum start and end must be provided together")
        if curriculum_start_s is not None and not (0.0 <= curriculum_start_s < curriculum_end_s):
            raise ValueError("timed curriculum must satisfy 0 <= start < end")
        if curriculum_start_ratio is not None and curriculum_start_s is not None:
            raise ValueError("progress and timed curriculum cannot be combined")
        if curriculum_random_quarters and (
            curriculum_start_ratio is not None or curriculum_start_s is not None
        ):
            raise ValueError("random quarters cannot be combined with another curriculum")

        self.transport = transport
        self.track_id = track_id
        self.lookahead_count = lookahead_count
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.max_episode_s = max_episode_s
        self.reward_config = reward_config or RewardConfig()
        self.request_timeout_s = request_timeout_s
        self.curriculum_last_fraction = curriculum_last_fraction
        self.curriculum_probability = curriculum_probability
        self.curriculum_lead_in_ratio = curriculum_lead_in_ratio
        self.curriculum_spawn_ratio = curriculum_spawn_ratio
        self.curriculum_start_ratio = curriculum_start_ratio
        self.curriculum_end_ratio = curriculum_end_ratio
        self.curriculum_start_s = curriculum_start_s
        self.curriculum_end_s = curriculum_end_s
        self.curriculum_random_quarters = curriculum_random_quarters
        self._episode_curriculum_end_ratio: float | None = None
        self._episode_curriculum_start_ratio: float | None = None
        self._episode_curriculum_spawn_ratio: float | None = None
        self._episode_curriculum_quarter: int | None = None
        self.action_adapter = action_adapter or DigitalActionAdapter()
        self.action_space = self.action_adapter.action_space
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(size(lookahead_count),),
            dtype=np.float32,
        )

        self._next_request_id = 0
        self._handshake_complete = False
        self._closed = False
        self._episode_id: str | None = None
        self._episode_steps = 0
        self._previous_progress_m = 0.0
        self._highest_progress_m = 0.0
        self._episode_start_progress_m = 0.0
        self._previous_action = Action()
        self._previous_control = ControlDemand.from_action(self._previous_action)
        self._episode_done = True
        self._airborne_time_s = 0.0
        self._air_brake_time_s = 0.0
        self._air_brake_active_time_s = 0.0
        self._air_brake_reward = 0.0
        self._air_brake_events = 0
        self._air_braking_previous = False
        self._max_air_brake_duty = 0.0
        self._air_brake_windows: dict[str, float] = {}
        self._stationary_s = 0.0
        self._off_track_s = 0.0
        self._barrier_contact_s = 0.0
        self._airborne_roll_s = 0.0
        self._landing_grace_s = 0.0
        self._was_airborne = False
        self.latest_telemetry: Telemetry | None = None
        self._air_brake_request = False
        self._air_brake_base_action: np.ndarray | None = None
        self.simulator_capabilities: Mapping[str, Any] = {}
        self._native_finish_restart_pending = False
        self._curriculum_reset_diagnostics: dict[str, Any] | None = None

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
            or max_ticks < 1
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
        # PolyTrack replaces its simulation worker shortly after displaying a
        # finish. Avoid sending the next reset to that retiring worker.
        if self._native_finish_restart_pending:
            time.sleep(1.0)
            self._native_finish_restart_pending = False
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
        self._episode_curriculum_spawn_ratio = None
        self._episode_curriculum_start_ratio = None
        self._episode_curriculum_end_ratio = self.curriculum_end_ratio
        self._episode_curriculum_quarter = None
        if self.curriculum_random_quarters:
            quarter = int(self.np_random.integers(0, 4))
            target_start_ratio = quarter / 4.0
            start_progress_ratio = max(0.0, target_start_ratio - self.curriculum_lead_in_ratio)
            self._episode_curriculum_end_ratio = (quarter + 1) / 4.0
            self._episode_curriculum_quarter = quarter + 1
            self._episode_curriculum_spawn_ratio = start_progress_ratio
            self._episode_curriculum_start_ratio = target_start_ratio
        elif self.curriculum_start_ratio is not None:
            start_progress_ratio = self.curriculum_spawn_ratio or 0.0
            self._episode_curriculum_spawn_ratio = start_progress_ratio
            self._episode_curriculum_start_ratio = self.curriculum_start_ratio
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
                "native_restart": False,
            },
        )
        transition = Transition.from_wire(result, lookahead_count=self.lookahead_count)
        if transition.ticks_advanced != 0:
            raise ProtocolViolation("reset must not advance simulation ticks")

        self._episode_id = transition.episode_id
        self._episode_steps = 0
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._highest_progress_m = self._previous_progress_m
        self._episode_start_progress_m = self._previous_progress_m
        self._previous_action = transition.telemetry.previous_action
        self._previous_control = ControlDemand.from_action(self._previous_action)
        self._episode_done = False
        self._airborne_time_s = 0.0
        self._air_brake_time_s = 0.0
        self._air_brake_active_time_s = 0.0
        self._air_brake_reward = 0.0
        self._air_brake_events = 0
        self._air_braking_previous = False
        self._max_air_brake_duty = 0.0
        self._air_brake_windows = {}
        self._stationary_s = 0.0
        self._off_track_s = 0.0
        self._barrier_contact_s = 0.0
        self._airborne_roll_s = 0.0
        self._landing_grace_s = 0.0
        self._was_airborne = False
        self.latest_telemetry = transition.telemetry
        self.action_adapter.reset()
        observation = self._policy_observation(transition.telemetry)
        info = self._info(transition, reward_terms=None, simulator_seed=simulator_seed)
        self._add_curriculum_info(info, transition.telemetry.route_progress_m)
        if self._episode_curriculum_start_ratio is not None:
            telemetry = transition.telemetry
            info["curriculum_reset_diagnostics"] = {
                "initial_speed_mps": float(np.linalg.norm(telemetry.local_velocity_mps)),
                "initial_local_velocity_mps": list(telemetry.local_velocity_mps),
                "initial_acceleration_mps2": list(telemetry.local_acceleration_mps2),
                "initial_angular_velocity_radps": list(telemetry.angular_velocity_radps),
                "initial_previous_action": telemetry.previous_action.to_wire(),
                "initial_actual_steering": telemetry.actual_steering,
                "initial_wheel_contacts": list(telemetry.wheel_contacts),
                "initial_suspension_lengths_m": list(telemetry.suspension_lengths_m),
                "route_progress_m": telemetry.route_progress_m,
                "track_length_m": telemetry.track_length_m,
                "heading_error_rad": telemetry.heading_error_rad,
                "lateral_offset_m": telemetry.lateral_offset_m,
                "initial_lookahead": [list(point) for point in telemetry.lookahead],
                "initial_lookahead_mask": list(telemetry.lookahead_mask),
            }
            self._curriculum_reset_diagnostics = info["curriculum_reset_diagnostics"]
        else:
            self._curriculum_reset_diagnostics = None
        if self._episode_curriculum_quarter is not None:
            info["curriculum_quarter"] = self._episode_curriculum_quarter
            info["curriculum_spawn_ratio"] = start_progress_ratio
            info["curriculum_start_ratio"] = self._episode_curriculum_start_ratio
            info["curriculum_end_ratio"] = self._episode_curriculum_end_ratio
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_id is None or self._episode_done:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside {self.action_space}")
        transitions: list[Transition] = []
        air_brake_step = (
            self._air_brake_request and self.latest_telemetry is not None
            and all(contact < 0.5 for contact in self.latest_telemetry.wheel_contacts)
            and self._air_brake_base_action is not None
        )
        self._air_brake_request = False
        base_action = self._air_brake_base_action
        self._air_brake_base_action = None
        if air_brake_step:
            # Query one physics tick at a time only during the air-brake window.
            # Once any wheel touches down, resume the base policy action for the
            # remainder of this frame-skip block; the search layer never brakes on ground.
            tick_controls = []
            landed = False
            for _ in range(self.frame_skip):
                current_action = base_action if landed else action
                current = self.action_adapter.apply(current_action, 1)
                control = current.ticks[0]
                tick_controls.append(control)
                result = self._exchange(
                    "step", {"episode_id": self._episode_id,
                             "action": control.to_wire(), "ticks": 1}
                )
                item = Transition.from_wire(result, lookahead_count=self.lookahead_count)
                transitions.append(item)
                landed = landed or any(contact >= 0.5 for contact in item.telemetry.wheel_contacts)
                if "finish" in item.events or "crash" in item.events:
                    break
            reward_action = ControlDemand(
                float(np.mean([item.steer for item in tick_controls])),
                float(np.mean([item.throttle for item in tick_controls])),
                float(np.mean([item.brake for item in tick_controls])),
            )
            applied = AppliedAction(reward_action, tick_controls)
        else:
            applied = self.action_adapter.apply(action, self.frame_skip)
            reward_action = applied.demand
            tick_controls = applied.ticks
        if air_brake_step:
            pass
        elif self.action_adapter.sequence:
            features = self.simulator_capabilities.get("features", ())
            if "action_sequence" in features:
                max_ticks = int(self.simulator_capabilities["max_ticks_per_step"])
                for start in range(0, len(tick_controls), max_ticks):
                    controls = tick_controls[start:start + max_ticks]
                    result = self._exchange(
                        "step",
                        {
                            "episode_id": self._episode_id,
                            "actions": [control.to_wire() for control in controls],
                            "ticks": len(controls),
                        },
                    )
                    item = Transition.from_wire(result, lookahead_count=self.lookahead_count)
                    transitions.append(item)
                    if "finish" in item.events or "crash" in item.events:
                        break
            else:
                max_ticks = int(self.simulator_capabilities["max_ticks_per_step"])
                for control, values_in_run in groupby(tick_controls):
                    remaining = sum(1 for _ in values_in_run)
                    while remaining:
                        run_ticks = min(remaining, max_ticks)
                        result = self._exchange(
                            "step",
                            {
                                "episode_id": self._episode_id,
                                "action": control.to_wire(),
                                "ticks": run_ticks,
                            },
                        )
                        item = Transition.from_wire(result, lookahead_count=self.lookahead_count)
                        transitions.append(item)
                        remaining -= run_ticks
                        if "finish" in item.events or "crash" in item.events:
                            break
                    if transitions and (
                        "finish" in transitions[-1].events or "crash" in transitions[-1].events
                    ):
                        break
        else:
            max_ticks = int(self.simulator_capabilities["max_ticks_per_step"])
            digital = tick_controls[0].to_wire()
            for start in range(0, self.frame_skip, max_ticks):
                run_ticks = min(max_ticks, self.frame_skip - start)
                result = self._exchange(
                    "step",
                    {"episode_id": self._episode_id, "action": digital, "ticks": run_ticks},
                )
                item = Transition.from_wire(result, lookahead_count=self.lookahead_count)
                transitions.append(item)
                if "finish" in item.events or "crash" in item.events:
                    break
        last = transitions[-1]
        transition = Transition(
            episode_id=last.episode_id,
            tick=last.tick,
            ticks_advanced=sum(item.ticks_advanced for item in transitions),
            telemetry=last.telemetry,
            events=tuple(event for item in transitions for event in item.events),
            simulator_info=last.simulator_info,
        )
        if transition.episode_id != self._episode_id:
            raise ProtocolViolation("simulator returned a stale or unexpected episode_id")
        if transition.ticks_advanced > self.frame_skip:
            raise ProtocolViolation("simulator advanced more ticks than requested")
        executed = tick_controls[:transition.ticks_advanced]
        if executed:
            reward_action = ControlDemand(
                sum(item.steer for item in executed) / len(executed),
                sum(item.throttle for item in executed) / len(executed),
                sum(item.brake for item in executed) / len(executed),
            )

        dt = transition.ticks_advanced * float(self.simulator_capabilities["fixed_dt_s"])
        telemetry = transition.telemetry
        width = _reference_corridor_width(telemetry, self.reward_config)
        lateral_ratio = abs(telemetry.lateral_offset_m) / width
        grounded_wheels = sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
        airborne = grounded_wheels < self.reward_config.off_track_min_grounded_wheels
        clean_takeoff = airborne and not self._was_airborne
        landed_this_step = not airborne and self._was_airborne
        off_track_landing = (
            landed_this_step
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
        # The simulator reports an untyped collision impulse. A touchdown can
        # produce one even when no barrier was hit, so do not end that step as
        # a barrier contact. Native crash/off-track checks still apply.
        barrier_contact = (
            not landed_this_step
            and collision_impulse > self.reward_config.barrier_collision_impulse_threshold
        )
        self._barrier_contact_s = dt if barrier_contact else 0.0

        fully_airborne = grounded_wheels == 0
        if fully_airborne and abs(telemetry.roll_rad) >= self.reward_config.airborne_roll_limit_rad:
            self._airborne_roll_s += dt
        else:
            self._airborne_roll_s = max(0.0, self._airborne_roll_s - 2.0 * dt)
        airborne_roll_failure = self._airborne_roll_s >= self.reward_config.airborne_roll_timeout_s
        curriculum_section_complete = bool(
            (
                self._episode_curriculum_end_ratio is not None
                and telemetry.route_progress_m
                >= telemetry.track_length_m * self._episode_curriculum_end_ratio
            )
            or (
                self.curriculum_end_s is not None
                and telemetry.elapsed_s >= self.curriculum_end_s - self.curriculum_start_s
            )
        )
        timed_out = (
            "finish" not in transition.events
            and not curriculum_section_complete
            and (
                "time_limit" in transition.events
                or self._episode_steps + 1 >= self.max_episode_steps
                or (self.max_episode_s is not None and telemetry.elapsed_s >= self.max_episode_s)
            )
        )

        reward, reward_terms, reward_groups = self._reward(
            transition,
            reward_action,
            stationary_s=self._stationary_s,
            stalled=stalled,
            off_track=off_track,
            early_off_track=early_off_track,
            barrier_contact=barrier_contact,
            off_track_landing=off_track_landing,
            clean_takeoff=clean_takeoff,
            airborne_roll_failure=airborne_roll_failure,
            curriculum_section_complete=curriculum_section_complete,
            timed_out=timed_out,
        )
        fully_airborne = all(contact < 0.5 for contact in telemetry.wheel_contacts)
        braking_in_air = fully_airborne and reward_action.brake > 0
        if fully_airborne:
            self._airborne_time_s += dt
        if braking_in_air:
            self._air_brake_time_s += dt * reward_action.brake
            self._air_brake_active_time_s += dt
            self._max_air_brake_duty = max(self._max_air_brake_duty, reward_action.brake)
            progress_ratio = telemetry.route_progress_m / max(1.0, telemetry.track_length_m)
            window_start = min(0.95, max(0.0, int(progress_ratio * 20) / 20))
            window = f"{window_start:.2f}-{window_start + 0.05:.2f}"
            self._air_brake_windows[window] = (
                self._air_brake_windows.get(window, 0.0) + dt * reward_action.brake
            )
            if not self._air_braking_previous:
                self._air_brake_events += 1
        self._air_braking_previous = braking_in_air
        self._air_brake_reward += reward_terms.get("airborne_brake", 0.0)
        self._episode_steps += 1
        events = set(transition.events)
        crash = "crash" in events and not landing_grace
        if "finish" in events and self.simulator_capabilities.get("simulator") != "mock-kinematic":
            self._native_finish_restart_pending = True
        terminated = (
            "finish" in events
            or crash
            or stalled
            or off_track
            or barrier_contact
            or airborne_roll_failure
            or curriculum_section_complete
        )
        truncated = (
            "time_limit" in events
            or self._episode_steps >= self.max_episode_steps
            or (self.max_episode_s is not None and telemetry.elapsed_s >= self.max_episode_s)
        )
        self._episode_done = terminated or truncated
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._previous_action = tick_controls[0]
        self._previous_control = reward_action
        self.latest_telemetry = transition.telemetry

        observation = self._policy_observation(transition.telemetry)
        info = self._info(transition, reward_terms=reward_terms)
        self._add_curriculum_info(info, telemetry.route_progress_m)
        if self._episode_steps == 1 and self._curriculum_reset_diagnostics is not None:
            info["curriculum_reset_diagnostics"] = self._curriculum_reset_diagnostics
            self._curriculum_reset_diagnostics = None
        info["reward_groups"] = reward_groups
        if not off_track:
            # The adapter's `off_track` event uses its fixed nominal width,
            # which can disagree with the reward profile's wider corridor.
            info["events"] = tuple(event for event in info["events"] if event != "off_track")
        info["requested_control_duty"] = {
            "steer": applied.demand.steer,
            "throttle": applied.demand.throttle,
            "brake": applied.demand.brake,
        }
        if executed:
            info["applied_control_fraction"] = {
                "steer": sum(item.steer for item in executed) / len(executed),
                "throttle": sum(item.throttle for item in executed) / len(executed),
                "brake": sum(item.brake for item in executed) / len(executed),
            }
        info["air_brake_summary"] = {
            "airborne_time_s": self._airborne_time_s,
            "air_brake_time_s": self._air_brake_time_s,
            "air_brake_fraction": self._air_brake_time_s / self._airborne_time_s
            if self._airborne_time_s else 0.0,
            "air_brake_reward": self._air_brake_reward,
            "air_brake_events": self._air_brake_events,
            "average_air_brake_duty": self._air_brake_time_s / self._air_brake_active_time_s
            if self._air_brake_active_time_s else 0.0,
            "max_air_brake_duty": self._max_air_brake_duty,
            "air_brake_windows": self._air_brake_windows.copy(),
        }
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
            info["events"] = tuple(dict.fromkeys((*info["events"], "curriculum_section_complete")))
        if landing_grace:
            info["landing_grace_s"] = self._landing_grace_s
        if truncated and "time_limit" not in events:
            info["wrapper_time_limit"] = True
            info["events"] = tuple(dict.fromkeys((*info["events"], "time_limit")))
        return observation, reward, terminated, truncated, info

    def _policy_observation(self, telemetry: Telemetry) -> np.ndarray:
        return observe(telemetry)

    def _add_curriculum_info(self, info: dict[str, Any], progress_m: float) -> None:
        if self._episode_curriculum_end_ratio is None:
            info["curriculum_stage"] = "full track"
            info["section_progress"] = None
            info["curriculum_in_lead_in"] = False
            return
        start = self._episode_curriculum_start_ratio or 0.0
        end = self._episode_curriculum_end_ratio
        ratio = progress_m / max(1.0, float(info.get("track_length_m", 1.0)))
        info.update({
            "curriculum_spawn_ratio": self._episode_curriculum_spawn_ratio or 0.0,
            "curriculum_start_ratio": start,
            "curriculum_end_ratio": end,
            "curriculum_in_lead_in": ratio < start,
            "section_progress": min(1.0, max(0.0, (ratio - start) / (end - start))),
            "curriculum_stage": "lead-in" if ratio < start else "target section",
        })

    def _reward(
        self,
        transition: Transition,
        action: ControlDemand,
        *,
        stationary_s: float = 0.0,
        stalled: bool = False,
        off_track: bool = False,
        early_off_track: bool = False,
        barrier_contact: bool = False,
        off_track_landing: bool = False,
        clean_takeoff: bool = False,
        airborne_roll_failure: bool = False,
        curriculum_section_complete: bool = False,
        timed_out: bool = False,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        context = RewardContext(
            transition=transition,
            action=action,
            previous_control=self._previous_control,
            config=self.reward_config,
            fixed_dt_s=float(self.simulator_capabilities["fixed_dt_s"]),
            previous_progress_m=self._previous_progress_m,
            highest_progress_m=self._highest_progress_m,
            episode_start_progress_m=self._episode_start_progress_m,
            stationary_s=stationary_s,
            stalled=stalled,
            off_track=off_track,
            early_off_track=early_off_track,
            barrier_contact=barrier_contact,
            off_track_landing=off_track_landing,
            clean_takeoff=clean_takeoff,
            airborne_roll_failure=airborne_roll_failure,
            curriculum_section_complete=curriculum_section_complete,
            timed_out=timed_out,
        )
        result = calculate_reward(context)
        self._highest_progress_m = max(
            self._highest_progress_m, transition.telemetry.route_progress_m
        )
        return result

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
        if self._episode_curriculum_quarter is not None:
            info["curriculum_quarter"] = self._episode_curriculum_quarter
        return info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.transport.close()
        super().close()
