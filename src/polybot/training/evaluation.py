"""Deterministic full-track evaluation and champion ranking."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class PrefixObservationReference:
    """Run the immutable actor on its original prefix during paired validation.

    This adapter is only for the known appended controller-state layout. It
    never changes the candidate's inputs or permits old replay to be reused.
    """

    def __init__(self, model: Any, *, extra_features: int) -> None:
        self.model = model
        self.width = int(np.prod(model.observation_space.shape))
        self.extra_features = extra_features

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def predict(self, observation: np.ndarray, **kwargs: Any) -> Any:
        values = np.asarray(observation)
        if values.shape[-1] != self.width + self.extra_features:
            raise ValueError("reference observation does not have the declared controller-state suffix")
        return self.model.predict(values[..., :self.width], **kwargs)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    finish_rate: float
    median_progress: float
    mean_progress: float
    best_lap_s: float | None
    median_lap_s: float | None
    crash_rate: float
    off_track_rate: float
    stall_rate: float
    airborne_time_s: float = 0.0
    air_brake_time_s: float = 0.0
    air_brake_fraction: float = 0.0
    air_brake_reward: float = 0.0
    mean_steering_disagreement: float = 0.0
    max_steering_disagreement: float = 0.0
    p95_steering_disagreement: float = 0.0
    mean_longitudinal_disagreement: float = 0.0
    max_longitudinal_disagreement: float = 0.0
    p95_longitudinal_disagreement: float = 0.0
    max_position_deviation_m: float = 0.0
    median_position_deviation_m: float = 0.0
    max_heading_deviation_rad: float = 0.0
    max_speed_deviation_mps: float = 0.0
    max_progress_deviation_m: float = 0.0
    lap_time_delta_s: float | None = None
    barrier_contact_steps: int = 0
    max_barrier_impulse: float = 0.0
    barrier_contact_progress: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rank(self) -> tuple[float, ...]:
        reliable = self.finish_rate == 1.0 and self.median_progress == 1.0
        return (
            self.finish_rate,
            self.median_progress,
            -(self.median_lap_s if self.median_lap_s is not None else float("inf"))
            if reliable else self.mean_progress,
            self.mean_progress if reliable else
            -(self.median_lap_s if self.median_lap_s is not None else float("inf")),
            -self.crash_rate,
            -self.off_track_rate,
            -self.stall_rate,
        )

    def confirms_target_lap(self, target_lap_s: float) -> bool:
        """Require a reliable full-track evaluation before stopping on pace."""
        return (
            target_lap_s > 0
            and self.episodes >= 5
            and self.finish_rate == 1.0
            and self.median_progress == 1.0
            and self.median_lap_s is not None
            and self.median_lap_s < target_lap_s
        )


def evaluate_model(
    model: Any, env_factory: Callable[[], Any], *, episodes: int, seed: int,
    observation_sink: list[np.ndarray] | None = None,
    reference_model: Any | None = None,
    action_noise_std: tuple[float, float] | None = None,
    action_noise_probability: float = 1.0,
    telemetry_sink: list[list[dict[str, Any]]] | None = None,
    reference_telemetry_sink: list[list[dict[str, Any]]] | None = None,
    transition_sink: list[dict[str, Any]] | None = None,
) -> EvaluationResult:
    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    if not 0.0 <= action_noise_probability <= 1.0:
        raise ValueError("action-noise probability must be in [0, 1]")
    progress: list[float] = []
    laps: list[float] = []
    crashes = off_tracks = stalls = 0
    airborne_time = air_brake_time = air_brake_reward = 0.0
    barrier_contact_steps = 0
    max_barrier_impulse = 0.0
    barrier_contact_progress: list[float] = []
    steer_drift: list[float] = []
    longitudinal_drift: list[float] = []
    position_drift: list[float] = []
    heading_drift: list[float] = []
    speed_drift: list[float] = []
    progress_drift: list[float] = []
    lap_deltas: list[float] = []
    noise_rng = np.random.default_rng(seed)
    env = env_factory()
    model.policy.set_training_mode(False)
    try:
        for index in range(episodes):
            reference_path: list[dict[str, Any]] = []
            reference_samples: list[dict[str, Any]] = []
            reference_lap = None
            if reference_model is not None:
                ref_obs, _ = env.reset(seed=seed + index)
                while True:
                    ref_action, _ = reference_model.predict(ref_obs, deterministic=True)
                    if getattr(reference_model, "_air_brake_active", False):
                        env._air_brake_request = True
                        env._air_brake_base_action = reference_model._air_brake_base_action
                    ref_obs, _, ref_terminated, ref_truncated, ref_info = env.step(ref_action)
                    reference_path.append(ref_info)
                    reference_samples.append(_telemetry_sample(ref_info, ref_action))
                    if ref_terminated or ref_truncated:
                        break
                if "finish" in set(ref_info.get("events", ())):
                    reference_lap = float(ref_info["elapsed_s"])
                if reference_telemetry_sink is not None:
                    reference_telemetry_sink.append(reference_samples)
            observation, _ = env.reset(seed=seed + index)
            candidate_path: list[dict[str, Any]] = []
            candidate_samples: list[dict[str, Any]] = []
            while True:
                if observation_sink is not None:
                    observation_sink.append(np.array(observation, copy=True))
                action, _ = model.predict(observation, deterministic=True)
                if getattr(model, "_air_brake_active", False):
                    env._air_brake_request = True
                    env._air_brake_base_action = model._air_brake_base_action
                if action_noise_std is not None:
                    mask = float(noise_rng.random() < action_noise_probability)
                    noise = mask * noise_rng.normal(0.0, np.asarray(action_noise_std), size=2)
                    action = np.clip(np.asarray(action) + noise, -1.0, 1.0).astype(np.float32)
                if reference_model is not None:
                    reference_action, _ = reference_model.predict(observation, deterministic=True)
                    diff = np.abs(np.asarray(action).reshape(-1) - np.asarray(reference_action).reshape(-1))
                    if diff.size >= 2:
                        steer_drift.append(float(diff[0]))
                        longitudinal_drift.append(float(diff[1]))
                before = np.array(observation, copy=True) if transition_sink is not None else None
                recorded = action
                if transition_sink is not None and getattr(model, "critic_raw_actions", False):
                    if action_noise_std is not None:
                        raise ValueError("raw-action evaluation replay does not support post-transform noise")
                    recorded, _ = model.policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                if transition_sink is not None:
                    transition_sink.append({
                        "observation": before, "next_observation": np.array(observation, copy=True),
                        "action": np.array(recorded, copy=True), "reward": reward,
                        "done": terminated or truncated,
                        "timeout": truncated and not terminated,
                    })
                if reference_model is not None:
                    candidate_path.append(info)
                candidate_samples.append(_telemetry_sample(info, action))
                if terminated or truncated:
                    break
            if telemetry_sink is not None:
                telemetry_sink.append(candidate_samples)
            events = set(info.get("events", ()))
            if reference_model is not None:
                for current in candidate_path:
                    if not reference_path:
                        break
                    progress_m = float(current.get("route_progress_m", 0.0))
                    nearest = min(reference_path, key=lambda sample: abs(
                        float(sample.get("route_progress_m", 0.0)) - progress_m
                    ))
                    a = np.asarray(current.get("position_m", ()), dtype=float)
                    b = np.asarray(nearest.get("position_m", ()), dtype=float)
                    if a.size >= 3 and b.size >= 3:
                        position_drift.append(float(np.linalg.norm(a[:3] - b[:3])))
                    heading_drift.append(abs(float(current.get("heading_error_rad", 0.0)) -
                                             float(nearest.get("heading_error_rad", 0.0))))
                    va = np.asarray(current.get("local_velocity_mps", ()), dtype=float)
                    vb = np.asarray(nearest.get("local_velocity_mps", ()), dtype=float)
                    if va.size >= 3 and vb.size >= 3:
                        speed_drift.append(abs(float(np.linalg.norm(va)) - float(np.linalg.norm(vb))))
                    progress_drift.append(abs(progress_m - float(nearest.get("route_progress_m", 0.0))))
                if reference_lap is not None and "finish" in events:
                    lap_deltas.append(float(info["elapsed_s"]) - reference_lap)
            progress.append(min(1.0, max(0.0, info["route_progress_m"] / info["track_length_m"])))
            if "finish" in events:
                laps.append(float(info["elapsed_s"]))
            crashes += int(
                "crash" in events or "airborne_roll_failure" in events
            )
            off_tracks += int("off_track" in events)
            stalls += int("stalled" in events)
            contacts = info.get("barrier_contact_progress", ())
            barrier_contact_steps += len(contacts)
            barrier_contact_progress.extend(float(value) for value in contacts)
            max_barrier_impulse = max(
                max_barrier_impulse,
                float(info.get("nonlanding_impulse_peak", 0.0)),
            )
            summary = info.get("air_brake_summary", {})
            airborne_time += float(summary.get("airborne_time_s", 0.0))
            air_brake_time += float(summary.get("air_brake_time_s", 0.0))
            air_brake_reward += float(summary.get("air_brake_reward", 0.0))
    finally:
        env.close()
        model.policy.set_training_mode(True)
    return EvaluationResult(
        episodes, len(laps) / episodes, statistics.median(progress),
        statistics.fmean(progress), min(laps) if laps else None,
        statistics.median(laps) if laps else None,
        crashes / episodes, off_tracks / episodes, stalls / episodes,
        airborne_time / episodes, air_brake_time / episodes,
        air_brake_time / airborne_time if airborne_time else 0.0,
        air_brake_reward / episodes,
        float(np.mean(steer_drift)) if steer_drift else 0.0,
        max(steer_drift, default=0.0), float(np.percentile(steer_drift, 95)) if steer_drift else 0.0,
        float(np.mean(longitudinal_drift)) if longitudinal_drift else 0.0,
        max(longitudinal_drift, default=0.0),
        float(np.percentile(longitudinal_drift, 95)) if longitudinal_drift else 0.0,
        max(position_drift, default=0.0), float(np.median(position_drift)) if position_drift else 0.0,
        max(heading_drift, default=0.0), max(speed_drift, default=0.0),
        max(progress_drift, default=0.0), float(np.median(lap_deltas)) if lap_deltas else None,
        barrier_contact_steps, max_barrier_impulse, tuple(barrier_contact_progress),
    )


def _telemetry_sample(info: dict[str, Any], action: Any) -> dict[str, Any]:
    sample = dict(info)
    values = np.asarray(action).reshape(-1)
    if values.size >= 2:
        sample["policy_steering"] = float(values[0])
        sample["policy_longitudinal"] = float(values[1])
        sample["policy_throttle"] = max(0.0, float(values[1]))
        sample["policy_brake"] = max(0.0, -float(values[1]))
    velocity = np.asarray(info.get("local_velocity_mps", ()), dtype=float)
    sample["speed_mps"] = float(np.linalg.norm(velocity)) if velocity.size else 0.0
    contacts = np.asarray(info.get("wheel_contacts", ()), dtype=float)
    sample["grounded_wheels"] = int(np.count_nonzero(contacts >= 0.5))
    sample["airborne"] = bool(contacts.size == 4 and np.all(contacts < 0.5))
    return sample
