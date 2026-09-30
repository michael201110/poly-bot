"""Algorithm-independent reward settings and reward terms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

import numpy as np

from polybot.control.actions import ControlDemand
from polybot.protocol import ProtocolViolation, Telemetry, Transition


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients kept independent from the game adapter."""

    progress_per_m: float = 0.0
    elapsed_cost_per_s: float = 0.0
    on_track_speed_per_m: float = 0.0
    speed_pace_reward_per_m_per_mps: float = 0.0
    speed_pace_limit_mps: float = 50.0
    airborne_speed_per_m: float = 0.0
    airborne_brake_bonus_per_s: float = 0.0
    ground_brake_penalty_per_s: float = 0.0
    takeoff_target_speed_mps: float = 45.0
    takeoff_speed_reward_per_mps: float = 0.0
    takeoff_speed_reward_limit: float = 0.0
    imitation_bonus_per_s: float = 18.0
    imitation_position_scale_m: float = 2.0
    imitation_rotation_scale_rad: float = 0.349066  # 20 degrees
    expert_action_bonus_per_s: float = 0.0
    ghost_speed_bonus_per_s: float = 0.0
    ghost_speed_scale_mps: float = 5.0
    guidance_reward_scale: float = 0.05
    guidance_min_forward_speed_mps: float = 5.0
    guidance_min_on_track_factor: float = 0.5
    low_speed_penalty_per_s: float = -5.0
    low_speed_grace_s: float = 1.0
    unsafe_speed_penalty_per_m: float = 0.0
    barrier_contact_penalty: float = -50.0
    barrier_early_penalty: float = 0.0
    barrier_collision_impulse_threshold: float = 0.0
    failure_progress_clawback_per_m: float = 0.0
    failure_early_penalty: float = 0.0
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
    ground_spin_deadzone_radps: float = 5.0
    ground_spin_penalty_per_rad_s: float = 0.0
    ground_spin_min_grounded_wheels: int = 2
    checkpoint_bonus: float = 0.0
    checkpoint_fast_bonus: float = 0.0
    checkpoint_target_s: float = 30.0
    checkpoint_speed_bonus_per_mps: float = 0.0
    checkpoint_speed_bonus_limit_mps: float = 45.0
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
    reference_corridor_scale: float = 1.0
    off_track_lateral_ratio: float = 1.05
    off_track_heading_ratio: float = 0.80
    off_track_heading_rad: float = 1.10
    off_track_wall_ride_roll_rad: float = 0.261799  # 15 degrees
    off_track_wall_ride_min_grounded_wheels: int = 3
    off_track_timeout_s: float = 1.25
    off_track_min_grounded_wheels: int = 2
    landing_grace_s: float = 2.0
    early_run_s: float = 20.0


def summer_1_reward_config() -> RewardConfig:
    """Balanced dense-to-sparse curriculum for a fresh Summer 1 policy."""

    return RewardConfig(
        progress_per_m=2.0,
        elapsed_cost_per_s=-0.20,
        on_track_speed_per_m=0.50,
        airborne_speed_per_m=0.20,
        airborne_brake_bonus_per_s=0.10,
        ground_brake_penalty_per_s=-3.0,
        takeoff_target_speed_mps=35.0,
        takeoff_speed_reward_per_mps=0.25,
        takeoff_speed_reward_limit=6.0,
        imitation_bonus_per_s=15.0,
        unsafe_speed_penalty_per_m=-0.50,
        barrier_contact_penalty=-1000.0,
        barrier_early_penalty=0.0,
        barrier_collision_impulse_threshold=0.0,
        failure_progress_clawback_per_m=0.0,
        failure_early_penalty=-2500.0,
        off_track_landing_penalty=-100.0,
        airborne_spin_penalty_per_rad=-2.0,
        airborne_tilt_penalty_per_s=-2.0,
        airborne_roll_penalty_per_s=-2.0,
        airborne_roll_failure_penalty=-1000.0,
        ground_slip_penalty_per_rad_s=-20.0,
        checkpoint_bonus=100.0,
        checkpoint_fast_bonus=75.0,
        checkpoint_target_s=12.0,
        checkpoint_speed_bonus_per_mps=1.0,
        checkpoint_speed_bonus_limit_mps=45.0,
        finish_bonus=1200.0,
        finish_fast_bonus=1800.0,
        finish_target_s=22.0,
        finish_pace_decay_per_s=0.35,
        curriculum_section_bonus=250.0,
        crash_penalty=-300.0,
        stall_penalty=-200.0,
        off_track_penalty=-250.0,
        early_off_track_penalty=-350.0,
        action_change_penalty=-0.005,
    )


def summer_1_pace_reward_config() -> RewardConfig:
    """Second-stage Summer 1 profile for a safe policy that needs more pace."""

    return dataclass_replace(
        summer_1_reward_config(),
        progress_per_m=0.85,
        elapsed_cost_per_s=-2.5,
        on_track_speed_per_m=0.50,
        takeoff_target_speed_mps=40.0,
        takeoff_speed_reward_per_mps=0.40,
        takeoff_speed_reward_limit=8.0,
        imitation_bonus_per_s=20.0,
        ground_brake_penalty_per_s=-4.0,
        unsafe_speed_penalty_per_m=-0.40,
        checkpoint_bonus=75.0,
        checkpoint_fast_bonus=250.0,
        checkpoint_target_s=8.0,
        checkpoint_speed_bonus_per_mps=10.0,
        checkpoint_speed_bonus_limit_mps=45.0,
        finish_bonus=1000.0,
        finish_fast_bonus=2200.0,
        finish_pace_decay_per_s=0.50,
        action_change_penalty=-0.005,
    )


def summer_1_bootstrap_reward_config() -> RewardConfig:
    """Full-track shaping that lets a fresh policy distinguish early attempts."""

    return dataclass_replace(
        summer_1_reward_config(),
        progress_per_m=4.0,
        elapsed_cost_per_s=-0.05,
        on_track_speed_per_m=0.75,
        airborne_speed_per_m=0.25,
        ground_brake_penalty_per_s=-1.0,
        imitation_bonus_per_s=20.0,
        unsafe_speed_penalty_per_m=-0.10,
        barrier_contact_penalty=-150.0,
        failure_early_penalty=-200.0,
        off_track_landing_penalty=-50.0,
        airborne_spin_penalty_per_rad=-0.5,
        airborne_tilt_penalty_per_s=-0.5,
        airborne_roll_penalty_per_s=-0.5,
        airborne_roll_failure_penalty=-150.0,
        ground_slip_penalty_per_rad_s=-5.0,
        checkpoint_bonus=250.0,
        checkpoint_fast_bonus=100.0,
        finish_bonus=2000.0,
        finish_fast_bonus=2000.0,
        crash_penalty=-100.0,
        stall_penalty=-100.0,
        off_track_penalty=-100.0,
        early_off_track_penalty=-150.0,
        action_change_penalty=-0.001,
    )


def summer_1_bootstrap_pace_reward_config() -> RewardConfig:
    """Move a bootstrapped full-track policy from survival toward forward pace."""

    return dataclass_replace(
        summer_1_bootstrap_reward_config(),
        elapsed_cost_per_s=-0.75,
        speed_pace_reward_per_m_per_mps=0.15,
        barrier_contact_penalty=-500.0,
        ground_brake_penalty_per_s=-3.0,
        stall_penalty=-350.0,
    )


def summer_1_ghost_learning_reward_config() -> RewardConfig:
    """Dense full-track shaping centered on copying the loaded ghost lap."""

    return dataclass_replace(
        summer_1_bootstrap_reward_config(),
        progress_per_m=3.0,
        elapsed_cost_per_s=-0.25,
        on_track_speed_per_m=1.0,
        speed_pace_reward_per_m_per_mps=0.03,
        imitation_bonus_per_s=100.0,
        expert_action_bonus_per_s=60.0,
        ghost_speed_bonus_per_s=30.0,
        ghost_speed_scale_mps=5.0,
        barrier_contact_penalty=-600.0,
        failure_early_penalty=-300.0,
        checkpoint_bonus=500.0,
        checkpoint_fast_bonus=250.0,
        finish_bonus=4000.0,
        finish_fast_bonus=4000.0,
        stall_penalty=-400.0,
    )


def summer_1_recovery_reward_config() -> RewardConfig:
    """Use the loaded ghost to teach safe starts as well as fast forward progress."""

    return dataclass_replace(
        summer_1_ghost_learning_reward_config(),
        guidance_reward_scale=1.0,
        guidance_min_forward_speed_mps=0.0,
        guidance_min_on_track_factor=0.25,
        low_speed_penalty_per_s=-5.0,
        low_speed_grace_s=1.0,
    )


def _reference_corridor_width(telemetry: Telemetry, config: RewardConfig) -> float:
    return max(0.1, telemetry.track_half_width_m * config.reference_corridor_scale)


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
    width = _reference_corridor_width(telemetry, config)
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
    forward_speed = max(0.0, telemetry.local_velocity_mps[2])
    speed_bonus = config.checkpoint_speed_bonus_per_mps * min(
        forward_speed, config.checkpoint_speed_bonus_limit_mps
    )
    return config.checkpoint_bonus + config.checkpoint_fast_bonus * pace_factor + speed_bonus


def _finish_reward(telemetry: Telemetry, config: RewardConfig) -> float:
    """Reward a valid finish, with additional credit for completing it quickly."""

    seconds_over_target = max(0.0, telemetry.elapsed_s - config.finish_target_s)
    pace_factor = float(np.exp(-config.finish_pace_decay_per_s * seconds_over_target))
    return config.finish_bonus + config.finish_fast_bonus * pace_factor


def _barrier_contact_reward(telemetry: Telemetry, config: RewardConfig) -> float:
    """Penalize early termination more heavily than a late-run collision."""

    progress_ratio = float(
        np.clip(telemetry.route_progress_m / telemetry.track_length_m, 0.0, 1.0)
    )
    return config.barrier_contact_penalty + config.barrier_early_penalty * (
        1.0 - progress_ratio
    )


def _failure_progress_clawback(
    telemetry: Telemetry,
    config: RewardConfig,
    *,
    episode_start_progress_m: float = 0.0,
    highest_progress_m: float | None = None,
) -> float:
    """Claw back progress earned in this episode, including a later rollback."""

    peak = telemetry.route_progress_m
    if highest_progress_m is not None:
        peak = max(peak, highest_progress_m)
    return config.failure_progress_clawback_per_m * max(
        0.0, peak - episode_start_progress_m
    )


def _failure_early_reward(telemetry: Telemetry, config: RewardConfig) -> float:
    """Make an incomplete outcome costly while still valuing farther exploration."""

    progress_ratio = float(
        np.clip(telemetry.route_progress_m / telemetry.track_length_m, 0.0, 1.0)
    )
    return config.failure_early_penalty * (1.0 - progress_ratio)


def _credited_progress_delta(
    current_m: float, previous_m: float, highest_m: float, config: RewardConfig
) -> float:
    """Credit new route distance once, even after a backward projection jump."""

    if current_m < previous_m:
        return max(current_m - previous_m, -config.max_reverse_progress_per_step_m)
    return min(max(0.0, current_m - highest_m), config.max_forward_progress_per_step_m)


def _ghost_pose_reward(simulator_info: Mapping[str, Any], config: RewardConfig, dt: float) -> float:
    """Reward proximity and full 3D orientation agreement with the ghost pose."""

    try:
        position_error = float(simulator_info["ghost_position_error_m"])
        rotation_error = float(simulator_info["ghost_rotation_error_rad"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not np.isfinite(position_error) or not np.isfinite(rotation_error):
        return 0.0
    position_similarity = np.exp(-max(0.0, position_error) / config.imitation_position_scale_m)
    rotation_similarity = np.exp(-max(0.0, rotation_error) / config.imitation_rotation_scale_rad)
    return float(config.imitation_bonus_per_s * dt * position_similarity * rotation_similarity)


def _expert_action_reward(
    action: ControlDemand, telemetry: Telemetry, config: RewardConfig, dt: float
) -> float:
    expert = telemetry.expert_action
    matches = (
        max(0.0, 1.0 - abs(action.steer - expert.steer) / 2.0)
        + 1.0 - abs(action.throttle - float(expert.throttle))
        + 1.0 - abs(action.brake - float(expert.brake))
    )
    position_error_sq = float(np.square(telemetry.ghost_relative_position_m).sum())
    speed_error = telemetry.local_velocity_mps[2] - telemetry.ghost_target_speed_mps
    confidence = np.exp(
        -position_error_sq / 8.0
        -telemetry.ghost_heading_error_rad**2 / (2 * 0.35**2)
        -speed_error**2 / (2 * 10.0**2)
    )
    return float(config.expert_action_bonus_per_s * (matches / 3.0) * dt * confidence)


def _ghost_speed_reward(telemetry: Telemetry, config: RewardConfig, dt: float) -> float:
    speed_error = abs(
        max(0.0, telemetry.local_velocity_mps[2]) - telemetry.ghost_target_speed_mps
    )
    scale = max(0.1, config.ghost_speed_scale_mps)
    return config.ghost_speed_bonus_per_s * float(np.exp(-speed_error / scale)) * dt


def _ghost_guidance_weight(
    progress_delta: float,
    forward_speed: float,
    on_track_factor: float,
    config: RewardConfig,
    *,
    incomplete_failure: bool,
) -> float:
    if (
        incomplete_failure
        or forward_speed < config.guidance_min_forward_speed_mps
        or on_track_factor < config.guidance_min_on_track_factor
    ):
        return 0.0
    return config.guidance_reward_scale * on_track_factor


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
    telemetry: Telemetry, action: ControlDemand, config: RewardConfig, dt: float
) -> float:
    """Slightly reward braking only while every wheel is off the ground."""

    if not action.brake or any(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    return config.airborne_brake_bonus_per_s * dt * float(action.brake)


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


def _ground_slip_penalty(telemetry: Telemetry, config: RewardConfig, dt: float) -> float:
    """Penalize tyre-scrubbing slip only while all four wheels are grounded."""

    if not all(contact >= 0.5 for contact in telemetry.wheel_contacts):
        return 0.0
    lateral_speed = abs(telemetry.local_velocity_mps[0])
    forward_speed = abs(telemetry.local_velocity_mps[2])
    slip_angle = float(np.arctan2(lateral_speed, max(forward_speed, 1e-6)))
    excess_slip = max(0.0, slip_angle - config.ground_slip_tolerance_rad)
    return config.ground_slip_penalty_per_rad_s * excess_slip * dt


def _ground_spin_penalty(telemetry: Telemetry, config: RewardConfig, dt: float) -> float:
    """Penalize excessive yaw rotation while the car has firm ground contact."""

    grounded = sum(contact >= 0.5 for contact in telemetry.wheel_contacts)
    if grounded < config.ground_spin_min_grounded_wheels:
        return 0.0
    yaw_rate = abs(telemetry.angular_velocity_radps[1])
    excess_spin = max(0.0, yaw_rate - config.ground_spin_deadzone_radps)
    return config.ground_spin_penalty_per_rad_s * excess_spin * dt


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Physical facts and episode state consumed by reward components."""

    transition: Transition
    action: ControlDemand
    previous_control: ControlDemand
    config: RewardConfig
    fixed_dt_s: float
    previous_progress_m: float
    highest_progress_m: float
    episode_start_progress_m: float
    stationary_s: float = 0.0
    stalled: bool = False
    off_track: bool = False
    early_off_track: bool = False
    barrier_contact: bool = False
    off_track_landing: bool = False
    clean_takeoff: bool = False
    airborne_roll_failure: bool = False
    curriculum_section_complete: bool = False
    timed_out: bool = False

    @property
    def telemetry(self) -> Telemetry:
        return self.transition.telemetry

    @property
    def dt(self) -> float:
        return self.transition.ticks_advanced * self.fixed_dt_s

    @property
    def progress_delta(self) -> float:
        return _credited_progress_delta(
            self.telemetry.route_progress_m, self.previous_progress_m,
            self.highest_progress_m, self.config,
        )

    @property
    def forward_speed(self) -> float:
        return max(0.0, self.telemetry.local_velocity_mps[2])

    @property
    def distance_at_speed(self) -> float:
        return self.forward_speed * self.dt

    @property
    def on_track_factor(self) -> float:
        width = _reference_corridor_width(self.telemetry, self.config)
        center = float(np.clip(1 - abs(self.telemetry.lateral_offset_m) / width, 0, 1))
        return center * max(0.0, float(np.cos(self.telemetry.heading_error_rad)))

    @property
    def airborne(self) -> bool:
        return (
            sum(contact >= 0.5 for contact in self.telemetry.wheel_contacts)
            < self.config.off_track_min_grounded_wheels
        )

    @property
    def incomplete_failure(self) -> bool:
        return any((
            self.barrier_contact, self.airborne_roll_failure, self.stalled,
            self.off_track, self.timed_out, "crash" in self.transition.events,
        ))


@dataclass(frozen=True, slots=True)
class RewardComponent:
    name: str
    category: str
    description: str
    unit: str
    calculate: Callable[[RewardContext], dict[str, float]]


def _progress_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    return {
        "progress": p.progress_per_m * c.progress_delta,
        "elapsed": p.elapsed_cost_per_s * c.dt,
        "on_track_speed": p.on_track_speed_per_m * c.distance_at_speed * c.on_track_factor,
        "speed_pace": p.speed_pace_reward_per_m_per_mps * c.distance_at_speed
        * min(c.forward_speed, p.speed_pace_limit_mps) * c.on_track_factor,
    }


def _guidance_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    weight = _ghost_guidance_weight(
        c.progress_delta, c.forward_speed, c.on_track_factor, p,
        incomplete_failure=c.incomplete_failure,
    )
    return {
        "ghost_imitation": _ghost_pose_reward(c.transition.simulator_info, p, c.dt) * weight,
        "expert_action_imitation": _expert_action_reward(
            c.action, c.telemetry, p, c.dt
        ) * weight,
        "ghost_speed": _ghost_speed_reward(c.telemetry, p, c.dt) * weight,
    }


def _driving_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    low_speed_s = min(c.dt, max(0.0, c.stationary_s - p.low_speed_grace_s))
    return {
        "ground_brake": p.ground_brake_penalty_per_s * c.dt * c.action.brake
        if any(contact >= 0.5 for contact in c.telemetry.wheel_contacts) else 0.0,
        "low_speed": p.low_speed_penalty_per_s * low_speed_s,
        "unsafe_speed": p.unsafe_speed_penalty_per_m * c.distance_at_speed
        * (1.0 - c.on_track_factor),
        "ground_slip": _ground_slip_penalty(c.telemetry, p, c.dt),
        "ground_spin": _ground_spin_penalty(c.telemetry, p, c.dt),
        "action_change": p.action_change_penalty * (
            abs(c.action.steer - c.previous_control.steer)
            + abs(c.action.throttle - c.previous_control.throttle)
            + abs(c.action.brake - c.previous_control.brake)
        ),
    }


def _airborne_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    t = c.telemetry
    stability = float(
        np.clip(1 - abs(t.roll_rad) / (np.pi / 4), 0, 1)
        * np.clip(
            1 - max(0.0, abs(t.pitch_rad) - p.airborne_pitch_tolerance_rad)
            / (np.pi / 4), 0, 1,
        )
    )
    takeoff = float(np.clip(
        (c.forward_speed - p.takeoff_target_speed_mps) * p.takeoff_speed_reward_per_mps,
        -p.takeoff_speed_reward_limit, p.takeoff_speed_reward_limit,
    )) if c.clean_takeoff else 0.0
    return {
        "airborne_speed": p.airborne_speed_per_m * c.distance_at_speed * stability
        if c.airborne else 0.0,
        "airborne_brake": _airborne_brake_reward(t, c.action, p, c.dt),
        "takeoff_speed": takeoff,
        "airborne_spin": _airborne_spin_penalty(t, p, c.dt),
        "airborne_tilt": _airborne_tilt_penalty(t, p, c.dt),
        "off_track_landing": p.off_track_landing_penalty if c.off_track_landing else 0.0,
        "airborne_roll_failure": p.airborne_roll_failure_penalty
        if c.airborne_roll_failure else 0.0,
    }


def _milestone_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    return {
        "checkpoint": _checkpoint_reward(c.telemetry, p)
        * c.transition.events.count("checkpoint"),
        "finish": _finish_reward(c.telemetry, p)
        if "finish" in c.transition.events else 0.0,
        "curriculum_section": p.curriculum_section_bonus
        if c.curriculum_section_complete else 0.0,
    }


def _failure_terms(c: RewardContext) -> dict[str, float]:
    p = c.config
    return {
        "barrier_contact": _barrier_contact_reward(c.telemetry, p)
        if c.barrier_contact else 0.0,
        "failure_progress_clawback": _failure_progress_clawback(
            c.telemetry, p, episode_start_progress_m=c.episode_start_progress_m,
            highest_progress_m=max(c.highest_progress_m, c.telemetry.route_progress_m),
        ) if c.incomplete_failure else 0.0,
        "failure_early": _failure_early_reward(c.telemetry, p)
        if c.incomplete_failure else 0.0,
        "crash": p.crash_penalty if "crash" in c.transition.events else 0.0,
        "stall": p.stall_penalty if c.stalled else 0.0,
        "off_track": p.early_off_track_penalty if c.early_off_track else (
            p.off_track_penalty if c.off_track else 0.0
        ),
    }


COMPONENTS = (
    RewardComponent("progress", "Progress", "Forward distance and pace", "points", _progress_terms),
    RewardComponent("guidance", "Guidance", "Optional ghost guidance", "points", _guidance_terms),
    RewardComponent("driving", "Driving quality", "Control and traction", "points", _driving_terms),
    RewardComponent("airborne", "Airborne behaviour", "Jump and landing quality", "points", _airborne_terms),
    RewardComponent("milestones", "Milestones", "Checkpoints and finishes", "points", _milestone_terms),
    RewardComponent("failure", "Failure", "Crash and termination cost", "points", _failure_terms),
)


def calculate_reward(c: RewardContext) -> tuple[float, dict[str, float], dict[str, float]]:
    terms: dict[str, float] = {}
    groups: dict[str, float] = {}
    for component in COMPONENTS:
        part = component.calculate(c)
        overlap = terms.keys() & part.keys()
        if overlap:
            raise AssertionError(f"duplicate reward terms: {overlap}")
        terms.update(part)
        groups[component.category] = float(sum(part.values()))
    total = float(sum(groups.values()))
    if not np.isfinite(total):
        raise ProtocolViolation("reward became non-finite")
    return total, terms, groups
