"""Transfer a frozen TQC driving policy into continuous PPO, then fine-tune it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch as th

from polybot.algorithms.registry import backend_for
from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA
from polybot.models.registry import PPO_ACTION_SEMANTICS, ModelMetadata, ModelRegistry
from polybot.training.config import EvaluationConfig, PPOConfig, TrainingConfig
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.promotion import promote_directory
from polybot.training.reward_profiles import RewardProfileStore
from polybot.training.runner import TrainingRunner

ACTION_SCHEMA = "continuous-pwm-v2"
ACTION_SEMANTICS = PPO_ACTION_SEMANTICS
DATASET_SCHEMA = "polybot.teacher-dataset.v1"
DAGGER_DATASET_SCHEMA = "polybot.dagger-dataset.v3"


@dataclass(slots=True)
class TeacherDataset:
    observations: np.ndarray
    raw_actions: np.ndarray
    final_actions: np.ndarray
    driving_actions: np.ndarray
    trajectory_ids: np.ndarray
    progress: np.ndarray
    elapsed_s: np.ndarray
    speed_mps: np.ndarray
    position_m: np.ndarray
    heading_error_rad: np.ndarray
    wheel_contacts: np.ndarray
    overlay_active: np.ndarray
    successful_trajectory_ids: list[int]
    teacher_id: str
    sources: np.ndarray | None = None
    dagger_rounds: np.ndarray | None = None
    sample_weights: np.ndarray | None = None
    student_actions: np.ndarray | None = None
    outcomes: np.ndarray | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = DATASET_SCHEMA if self.sources is None else "polybot.teacher-dataset.v2"
        arrays = {
            name: value for name, value in asdict(self).items()
            if name not in {"successful_trajectory_ids", "teacher_id"} and value is not None
        }
        np.savez_compressed(
            path,
            schema=np.asarray(schema),
            teacher_id=np.asarray(self.teacher_id),
            successful_trajectory_ids=np.asarray(self.successful_trajectory_ids, dtype=np.int64),
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> TeacherDataset:
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"].item()) not in {DATASET_SCHEMA, "polybot.teacher-dataset.v2"}:
                raise ValueError("unsupported teacher dataset schema")
            return cls(
                **{name: np.array(data[name]) for name in (
                    "observations", "raw_actions", "final_actions", "trajectory_ids",
                    "driving_actions",
                    "progress", "elapsed_s", "speed_mps", "position_m",
                    "heading_error_rad", "wheel_contacts", "overlay_active",
                )},
                successful_trajectory_ids=data["successful_trajectory_ids"].astype(int).tolist(),
                teacher_id=str(data["teacher_id"].item()),
                **{
                    name: np.array(data[name]) if name in data.files else None
                    for name in (
                        "sources", "dagger_rounds", "sample_weights",
                        "student_actions", "outcomes",
                    )
                },
            )


@dataclass(slots=True)
class DaggerDataset:
    """Student-visited states labeled by the frozen teacher."""

    observations: np.ndarray
    student_actions: np.ndarray
    teacher_raw_actions: np.ndarray
    teacher_actions: np.ndarray
    teacher_driving_actions: np.ndarray
    action_errors: np.ndarray
    progress: np.ndarray
    episode_ids: np.ndarray
    episode_outcomes: np.ndarray
    sources: np.ndarray
    dagger_rounds: np.ndarray
    elapsed_s: np.ndarray
    time_to_failure_s: np.ndarray
    speed_mps: np.ndarray
    position_m: np.ndarray
    heading_error_rad: np.ndarray
    wheel_contacts: np.ndarray
    airborne: np.ndarray
    air_brake_active: np.ndarray
    overlay_active: np.ndarray
    failure_progress: np.ndarray
    teacher_id: str
    collection_deterministic: bool = True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema=np.asarray(DAGGER_DATASET_SCHEMA),
            teacher_id=np.asarray(self.teacher_id),
            **{key: value for key, value in asdict(self).items() if key != "teacher_id"},
        )

    @classmethod
    def load(cls, path: Path) -> DaggerDataset:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"].item())
            if schema not in {
                "polybot.dagger-dataset.v1", "polybot.dagger-dataset.v2",
                DAGGER_DATASET_SCHEMA,
            }:
                raise ValueError("unsupported DAgger dataset schema")
            names = (
                "observations", "student_actions", "teacher_raw_actions",
                "teacher_actions", "teacher_driving_actions", "action_errors",
                "progress", "episode_ids", "episode_outcomes", "dagger_rounds",
                "elapsed_s", "time_to_failure_s", "speed_mps", "position_m",
                "heading_error_rad", "wheel_contacts", "airborne",
                "air_brake_active", "overlay_active", "failure_progress",
            )
            values = {name: np.array(data[name]) for name in names}
            values["sources"] = (
                np.array(data["sources"]).astype(str)
                if "sources" in data.files
                else np.full(len(values["observations"]), "dagger", dtype="U8")
            )
            return cls(
                **values,
                teacher_id=str(data["teacher_id"].item()),
                collection_deterministic=(
                    bool(data["collection_deterministic"].item())
                    if "collection_deterministic" in data.files else False
                ),
            )


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_compatibility(teacher_metadata: ModelMetadata, student_config: TrainingConfig) -> None:
    problems = []
    if teacher_metadata.algorithm != "tqc":
        problems.append("teacher must be TQC")
    if teacher_metadata.observation_schema != OBSERVATION_SCHEMA:
        problems.append("observation schema differs")
    if teacher_metadata.lookahead_count != student_config.lookahead_count:
        problems.append("lookahead count differs")
    if teacher_metadata.action_schema != ACTION_SCHEMA:
        problems.append("teacher action is not continuous Box(2)")
    if teacher_metadata.track_id != student_config.track_id:
        problems.append("track differs")
    if problems:
        raise ValueError("teacher/student incompatibility: " + ", ".join(problems))


def split_trajectories(
    trajectory_ids: np.ndarray, successful_ids: list[int], *, seed: int = 0,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(set(int(value) for value in successful_ids)), dtype=np.int64)
    if ids.size < 2:
        raise ValueError("at least two successful teacher laps are needed for trajectory validation")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    validation_count = min(ids.size - 1, max(1, round(ids.size * validation_fraction)))
    validation_ids = ids[:validation_count]
    train_ids = ids[validation_count:]
    train = np.flatnonzero(np.isin(trajectory_ids, train_ids))
    validation = np.flatnonzero(np.isin(trajectory_ids, validation_ids))
    return train, validation


def split_aggregated_trajectories(
    trajectory_ids: np.ndarray, sources: np.ndarray, *, seed: int = 0,
    validation_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep episodes intact while retaining validation examples from each source."""
    trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
    sources = np.asarray(sources).astype(str)
    if trajectory_ids.shape != sources.shape:
        raise ValueError("trajectory ids and source labels must align")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    validation_ids: set[int] = set()
    for source in np.unique(sources):
        ids = np.unique(trajectory_ids[sources == source])
        if ids.size < 2:
            continue
        rng.shuffle(ids)
        count = min(ids.size - 1, max(1, round(ids.size * validation_fraction)))
        validation_ids.update(int(value) for value in ids[:count])
    if not validation_ids:
        raise ValueError("aggregated validation needs at least two episodes")
    validation_mask = np.isin(trajectory_ids, list(validation_ids))
    if not np.any(~validation_mask):
        raise ValueError("aggregated training split is empty")
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _deterministic_mean(model: Any, observations: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    outputs = []
    model.policy.set_training_mode(False)
    with th.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = th.as_tensor(observations[start:start + batch_size], device=model.device)
            distribution = model.policy.get_distribution(obs).distribution
            outputs.append(distribution.mean.clamp(-1.0, 1.0).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 2), dtype=np.float32)


def _high_level_teacher_action(teacher: Any, observation: np.ndarray, raw_action: np.ndarray) -> np.ndarray:
    """Apply bakeable actor overlays while leaving tick-level air brake below PPO."""
    progress = float(np.asarray(observation)[12])
    action = np.asarray(raw_action, dtype=np.float32).reshape(2).copy()
    for start, end, amount in getattr(teacher, "speed_bias_schedule", ()):
        fade = min((progress - start) / 0.02, (end - progress) / 0.02)
        action[1] += float(amount) * float(np.clip(fade, 0.0, 1.0))
    for layer in getattr(teacher, "policy_overlays", ()):
        if layer.get("kind") == "air_brake":
            continue
        start, end = float(layer["start"]), float(layer["end"])
        taper = min(float(layer.get("taper", 0.01)), (end - start) / 2)
        enter_x = np.clip((progress - start) / max(taper, 1e-9), 0.0, 1.0)
        leave_x = np.clip((end - progress) / max(taper, 1e-9), 0.0, 1.0)
        enter = enter_x * enter_x * (3.0 - 2.0 * enter_x)
        leave = leave_x * leave_x * (3.0 - 2.0 * leave_x)
        fade = min(enter, leave)
        amount = float(layer.get("amount", 0.0))
        if layer["kind"] == "steer_bias":
            action[0] += amount * fade
        elif layer["kind"] == "steer_gain":
            action[0] *= 1.0 + (amount - 1.0) * fade
        elif layer["kind"] == "drive_bias":
            action[1] += amount * fade
        elif layer["kind"] == "drive_gain":
            action[1] *= 1.0 + (amount - 1.0) * fade
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def imitation_metrics(
    model: Any, dataset: TeacherDataset, indices: np.ndarray
) -> dict[str, Any]:
    predicted = _deterministic_mean(model, dataset.observations[indices])
    target = dataset.final_actions[indices]
    error = np.abs(predicted - target)
    sections = {}
    for section in range(10):
        mask = (
            (dataset.progress[indices] >= section / 10)
            & (dataset.progress[indices] < (section + 1) / 10)
        )
        if mask.any():
            sections[f"{section * 10:02d}-{(section + 1) * 10:02d}%"] = {
                "steering_mae": float(error[mask, 0].mean()),
                "longitudinal_mae": float(error[mask, 1].mean()),
                "samples": int(mask.sum()),
            }
    overlay_mask = dataset.overlay_active[indices].astype(bool)
    return {
        "samples": int(len(indices)),
        "steering_mse": float(np.square(error[:, 0]).mean()),
        "longitudinal_mse": float(np.square(error[:, 1]).mean()),
        "steering_mae": float(error[:, 0].mean()),
        "longitudinal_mae": float(error[:, 1].mean()),
        "max_steering_error": float(error[:, 0].max(initial=0.0)),
        "max_longitudinal_error": float(error[:, 1].max(initial=0.0)),
        "max_action_error": float(error.max(initial=0.0)),
        "overlay_steering_mae": float(error[overlay_mask, 0].mean()) if overlay_mask.any() else None,
        "overlay_longitudinal_mae": float(error[overlay_mask, 1].mean()) if overlay_mask.any() else None,
        "sections": sections,
    }


def pretrain_actor(
    model: Any, dataset: TeacherDataset, *, epochs: int = 100, batch_size: int = 1024,
    patience: int = 10, seed: int = 0, learning_rate: float = 3e-4,
    train_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
    initial_action_std: float = 0.15,
) -> dict[str, Any]:
    """Regress the teacher action mean, preserve the critic, and calibrate PPO noise."""
    if train_indices is None or validation_indices is None:
        if dataset.sources is not None:
            train_indices, validation_indices = split_aggregated_trajectories(
                dataset.trajectory_ids, dataset.sources, seed=seed
            )
        else:
            train_indices, validation_indices = split_trajectories(
                dataset.trajectory_ids, dataset.successful_trajectory_ids, seed=seed
            )
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size and patience must be positive")
    if not 0 < initial_action_std <= 1:
        raise ValueError("initial action standard deviation must be in (0, 1]")
    weights = sample_weights if sample_weights is not None else dataset.sample_weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(dataset.observations),) or not np.all(np.isfinite(weights)):
            raise ValueError("sample weights must be finite and match the dataset")
        if np.any(weights < 0) or weights[train_indices].sum() <= 0:
            raise ValueError("training sample weights must be nonnegative with positive mass")
    actor_modules = [model.policy.mlp_extractor.policy_net, model.policy.action_net]
    actor_parameters = [p for module in actor_modules for p in module.parameters()]
    actor_parameters.extend([model.policy.log_std])
    critic_before = {
        name: value.detach().clone() for name, value in model.policy.state_dict().items()
        if "value" in name or "critic" in name
    }
    optimizer = th.optim.Adam(actor_parameters, lr=learning_rate)
    # Offline regression trains the deterministic mean, so it cannot calibrate
    # PPO's Gaussian exploration scale. Keep live PPO rollouts near the teacher
    # instead of inheriting the broad default noise from an unrelated checkpoint.
    with th.no_grad():
        model.policy.log_std.fill_(float(np.log(initial_action_std)))
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state = None
    stale = 0
    history = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        if weights is None:
            rng.shuffle(train_indices)
            batches = (
                train_indices[start:start + batch_size]
                for start in range(0, len(train_indices), batch_size)
            )
        else:
            probabilities = weights[train_indices] / weights[train_indices].sum()
            batch_count = max(1, int(np.ceil(len(train_indices) / batch_size)))
            batches = (
                rng.choice(train_indices, size=batch_size, replace=True, p=probabilities)
                for _ in range(batch_count)
            )
        for rows in batches:
            obs = th.as_tensor(dataset.observations[rows], device=model.device)
            target = th.as_tensor(dataset.final_actions[rows], device=model.device)
            mean = model.policy.get_distribution(obs).distribution.mean
            per_sample = th.nn.functional.mse_loss(mean, target, reduction="none").mean(dim=1)
            # Weighted rows were already sampled with probability proportional
            # to their sample weights. Multiplying the loss by those weights a
            # second time would square the intended source/error weighting.
            loss = per_sample.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
        validation = imitation_metrics(model, dataset, validation_indices)
        val_loss = validation["steering_mse"] + validation["longitudinal_mse"]
        history.append({"epoch": epoch + 1, "validation_loss": val_loss, **validation})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {
                name: value.detach().clone()
                for name, value in model.policy.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("actor pretraining failed to produce a finite validation checkpoint")
    model.policy.load_state_dict(best_state)
    critic_after = model.policy.state_dict()
    if any(not th.equal(value, critic_after[name]) for name, value in critic_before.items()):
        raise RuntimeError("actor-only imitation unexpectedly changed PPO value-function weights")
    final = imitation_metrics(model, dataset, validation_indices)
    by_source = {}
    if dataset.sources is not None:
        validation_sources = np.asarray(dataset.sources).astype(str)[validation_indices]
        for source in np.unique(validation_sources):
            source_indices = validation_indices[validation_sources == source]
            by_source[str(source)] = imitation_metrics(model, dataset, source_indices)
    return {
        "epochs_run": len(history), "best_validation_loss": best_loss,
        "validation": final, "validation_by_source": by_source, "history": history,
        "action_std": np.exp(model.policy.log_std.detach().cpu().numpy()).tolist(),
        "train_trajectories": sorted(set(dataset.trajectory_ids[train_indices].tolist())),
        "validation_trajectories": sorted(set(dataset.trajectory_ids[validation_indices].tolist())),
        "critic_unchanged": True,
    }


def collect_teacher_data(
    teacher: Any, env: Any, *, successful_laps: int = 20, max_attempts: int | None = None,
    seed: int = 0, teacher_id: str = "unknown",
) -> TeacherDataset:
    if successful_laps < 2:
        raise ValueError("collect at least two successful laps for trajectory-level validation")
    max_attempts = max_attempts or successful_laps * 3
    rows: dict[str, list[Any]] = {name: [] for name in (
        "observations", "raw_actions", "final_actions", "trajectory_ids", "progress",
        "driving_actions",
        "elapsed_s", "speed_mps", "position_m", "heading_error_rad", "wheel_contacts",
        "overlay_active",
    )}
    successful = []
    original_overlays = list(getattr(teacher, "policy_overlays", []))
    original_schedule = list(getattr(teacher, "speed_bias_schedule", []))
    teacher.policy.set_training_mode(False)
    try:
        for trajectory in range(max_attempts):
            if len(successful) >= successful_laps:
                break
            obs, _ = env.reset(seed=seed + trajectory)
            episode_rows = {name: [] for name in rows}
            final_info: dict[str, Any] = {}
            while True:
                raw_action, _ = teacher.policy.predict(obs, deterministic=True)
                final_action, _ = teacher.predict(obs, deterministic=True)
                air_brake_active = bool(getattr(teacher, "_air_brake_active", False))
                high_level_action = _high_level_teacher_action(teacher, obs, raw_action)
                if air_brake_active:
                    env._air_brake_request = True
                    env._air_brake_base_action = high_level_action.copy()
                obs_copy = np.array(obs, dtype=np.float32, copy=True)
                obs, _, terminated, truncated, final_info = env.step(final_action)
                velocity = np.asarray(final_info.get("local_velocity_mps", ()), dtype=float)
                contacts = np.asarray(final_info.get("wheel_contacts", (0, 0, 0, 0)), dtype=float)
                progress = float(obs_copy[12])
                active_overlay = any(
                    float(layer["start"]) <= progress <= float(layer["end"])
                    for layer in original_overlays
                )
                values = {
                    "observations": obs_copy,
                    "raw_actions": np.asarray(raw_action, dtype=np.float32).reshape(2),
                    "final_actions": high_level_action,
                    "driving_actions": np.asarray(final_action, dtype=np.float32).reshape(2),
                    "trajectory_ids": trajectory,
                    "progress": progress,
                    "elapsed_s": float(final_info.get("elapsed_s", 0.0)),
                    "speed_mps": float(np.linalg.norm(velocity)) if velocity.size else 0.0,
                    "position_m": np.asarray(final_info.get("position_m", (0, 0, 0)), dtype=np.float32)[:3],
                    "heading_error_rad": float(final_info.get("heading_error_rad", 0.0)),
                    "wheel_contacts": np.pad(contacts[:4], (0, max(0, 4 - contacts.size)))[:4],
                    "overlay_active": active_overlay,
                }
                for key, value in values.items():
                    episode_rows[key].append(value)
                if terminated or truncated:
                    break
            succeeded = "finish" in set(final_info.get("events", ()))
            for name in rows:
                rows[name].extend(episode_rows[name])
            if succeeded:
                successful.append(trajectory)
        if len(successful) < 2:
            raise RuntimeError(
                f"teacher collection produced only {len(successful)} successful laps "
                f"after {max_attempts} attempts"
            )
    finally:
        teacher.policy_overlays = original_overlays
        teacher.speed_bias_schedule = original_schedule
    return TeacherDataset(
        **{key: np.asarray(value) for key, value in rows.items()},
        successful_trajectory_ids=successful,
        teacher_id=teacher_id,
    )


def _episode_outcome(info: dict[str, Any], truncated: bool) -> str:
    events = set(info.get("events", ()))
    for event, outcome in (
        ("finish", "finish"), ("crash", "crash"), ("barrier_contact", "crash"),
        ("airborne_roll_failure", "crash"), ("off_track", "off_track"),
        ("stalled", "stall"), ("stall", "stall"),
    ):
        if event in events:
            return outcome
    return "timeout" if truncated else "terminated"


def collect_dagger_data(
    student: Any, teacher: Any, env: Any, *, episodes: int = 8,
    dagger_round: int = 1, seed: int = 0, teacher_id: str = "unknown",
    failure_window_s: float = 2.0,
    deterministic: bool = True,
) -> DaggerDataset:
    """Let PPO drive while the frozen TQC labels each exact student observation."""
    if episodes < 1 or dagger_round < 1 or failure_window_s <= 0:
        raise ValueError("DAgger episodes and round must be positive")
    names = (
        "observations", "student_actions", "teacher_raw_actions", "teacher_actions",
        "teacher_driving_actions", "action_errors", "progress", "episode_ids",
        "episode_outcomes", "sources", "dagger_rounds", "elapsed_s", "time_to_failure_s",
        "speed_mps", "position_m", "heading_error_rad", "wheel_contacts",
        "airborne", "air_brake_active", "overlay_active", "failure_progress",
    )
    rows: dict[str, list[Any]] = {name: [] for name in names}
    teacher.policy.set_training_mode(False)
    student.policy.set_training_mode(False)
    overlays = list(getattr(teacher, "policy_overlays", ()))
    schedule = list(getattr(teacher, "speed_bias_schedule", ()))
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        trajectory: list[dict[str, Any]] = []
        terminal_info: dict[str, Any] = {}
        truncated = False
        while True:
            # Copy once and pass this same immutable state to both controllers.
            current = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
            student_action, _ = student.predict(current, deterministic=deterministic)
            student_action = np.clip(
                np.asarray(student_action, dtype=np.float32).reshape(2), -1.0, 1.0
            )
            teacher_raw, _ = teacher.policy.predict(current, deterministic=True)
            teacher_drive, _ = teacher.predict(current, deterministic=True)
            teacher_action = _high_level_teacher_action(teacher, current, teacher_raw)
            teacher_brake = bool(getattr(teacher, "_air_brake_active", False))
            next_observation, _, terminated, truncated, info = env.step(student_action)
            contacts = np.asarray(info.get("wheel_contacts", current[17:21]), dtype=np.float32)
            contacts = np.pad(contacts[:4], (0, max(0, 4 - contacts.size)))[:4]
            velocity = np.asarray(info.get("local_velocity_mps", ()), dtype=float)
            progress = float(current[12])
            active_overlay = any(
                float(layer.get("start", 1.0)) <= progress <= float(layer.get("end", 0.0))
                for layer in overlays
            ) or any(start <= progress <= end for start, end, _ in schedule)
            trajectory.append({
                "observations": current,
                "student_actions": student_action.copy(),
                "teacher_raw_actions": np.asarray(teacher_raw, dtype=np.float32).reshape(2),
                "teacher_actions": teacher_action,
                "teacher_driving_actions": np.asarray(teacher_drive, dtype=np.float32).reshape(2),
                "action_errors": np.abs(student_action - teacher_action),
                "progress": progress,
                "elapsed_s": float(info.get("elapsed_s", 0.0)),
                "speed_mps": float(np.linalg.norm(velocity)) if velocity.size else 0.0,
                "position_m": np.asarray(info.get("position_m", (0, 0, 0)), dtype=np.float32)[:3],
                "heading_error_rad": float(info.get("heading_error_rad", 0.0)),
                "wheel_contacts": contacts,
                "airborne": bool(np.all(current[17:21] < 0.5)),
                "air_brake_active": teacher_brake,
                "overlay_active": active_overlay,
                "terminal_failure": bool(terminated or truncated),
            })
            terminal_info = dict(info)
            observation = next_observation
            if terminated or truncated:
                break
        outcome = _episode_outcome(terminal_info, truncated)
        final_elapsed = float(terminal_info.get("elapsed_s", 0.0))
        progress_end = max((row["progress"] for row in trajectory), default=0.0)
        # The action that actually terminates in a crash is generally terminal
        # chaos; keep the preceding student recovery opportunities instead.
        useful = trajectory
        if outcome in {"crash", "off_track", "stall"} and trajectory:
            useful = [
                row for row in trajectory[:-1]
                if final_elapsed - float(row["elapsed_s"]) <= failure_window_s
            ]
        episode_id = (dagger_round - 1) * 1_000_000 + episode
        for row in useful:
            for name in names:
                if name in {
                    "episode_ids", "episode_outcomes", "sources", "dagger_rounds",
                    "time_to_failure_s", "failure_progress",
                }:
                    continue
                rows[name].append(row[name])
            rows["episode_ids"].append(episode_id)
            rows["episode_outcomes"].append(outcome)
            rows["sources"].append("dagger")
            rows["dagger_rounds"].append(dagger_round)
            delta = final_elapsed - float(row["elapsed_s"])
            rows["time_to_failure_s"].append(max(0.0, delta) if outcome != "finish" else np.inf)
            rows["failure_progress"].append(progress_end)
    student.policy.set_training_mode(True)
    return DaggerDataset(
        **{key: np.asarray(value) for key, value in rows.items()},
        teacher_id=teacher_id,
        collection_deterministic=deterministic,
    )


def dagger_action_error_report(dataset: DaggerDataset) -> dict[str, Any]:
    """Summarize student/teacher divergence in measured 5% track windows."""
    bins: dict[str, Any] = {}
    combined = dataset.action_errors.sum(axis=1)
    global_mean = float(combined.mean()) if combined.size else 0.0
    global_std = float(combined.std()) if combined.size else 0.0
    first_divergence = None
    for section in range(20):
        lower, upper = section * 0.05, (section + 1) * 0.05
        mask = (dataset.progress >= lower) & (
            (dataset.progress < upper) if section < 19 else (dataset.progress <= upper)
        )
        if not mask.any():
            continue
        errors = dataset.action_errors[mask]
        mean_total = float(errors.sum(axis=1).mean())
        key = f"{section * 5:02d}-{(section + 1) * 5:02d}%"
        bins[key] = {
            "samples": int(mask.sum()),
            "mean_action_error": mean_total,
            "max_action_error": float(errors.sum(axis=1).max(initial=0.0)),
            "mean_steering_error": float(errors[:, 0].mean()),
            "max_steering_error": float(errors[:, 0].max(initial=0.0)),
            "mean_longitudinal_error": float(errors[:, 1].mean()),
            "max_longitudinal_error": float(errors[:, 1].max(initial=0.0)),
        }
        if first_divergence is None and mean_total > global_mean + global_std:
            first_divergence = [lower, upper]
    failures = []
    for episode_id in np.unique(dataset.episode_ids):
        indices = np.flatnonzero(dataset.episode_ids == episode_id)
        if indices.size:
            outcome = str(dataset.episode_outcomes[indices[0]])
            if outcome != "finish":
                failures.append({
                    "episode_id": int(episode_id), "outcome": outcome,
                    "progress": float(dataset.failure_progress[indices[0]]),
                })
    return {
        "samples": int(len(dataset.observations)),
        "first_major_divergence_progress": first_divergence,
        "failure_progress": failures,
        "mean_steering_error": float(dataset.action_errors[:, 0].mean()) if len(dataset.observations) else 0.0,
        "mean_longitudinal_error": float(dataset.action_errors[:, 1].mean()) if len(dataset.observations) else 0.0,
        "windows_5_percent": bins,
    }


def aggregate_teacher_datasets(
    nominal: TeacherDataset, dagger_rounds: list[DaggerDataset], *,
    nominal_weight: float = 0.6, recovery_weight: float = 0.4,
    error_weight_cap: float = 3.0, failure_window_s: float = 2.0,
) -> tuple[TeacherDataset, dict[str, Any]]:
    """Aggregate immutable nominal data with every accepted recovery round."""
    if nominal_weight < 0 or recovery_weight < 0 or nominal_weight + recovery_weight <= 0:
        raise ValueError("nominal/recovery weights must be nonnegative with positive total")
    if any(item.teacher_id != nominal.teacher_id for item in dagger_rounds):
        raise ValueError("DAgger data was labeled by a different teacher")
    if not dagger_rounds:
        return nominal, {"nominal_samples": len(nominal.observations), "recovery_samples": 0}
    sample_fields = (
        "observations", "raw_actions", "final_actions", "driving_actions", "trajectory_ids",
        "progress", "elapsed_s", "speed_mps", "position_m", "heading_error_rad",
        "wheel_contacts", "overlay_active", "student_actions", "outcomes", "sources",
        "dagger_rounds",
    )
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in sample_fields}
    nominal_count = len(nominal.observations)
    next_id = int(np.max(nominal.trajectory_ids, initial=-1)) + 1
    nominal_sources = np.full(nominal_count, "nominal", dtype="U8")
    nominal_success = set(int(value) for value in nominal.successful_trajectory_ids)
    arrays["observations"].append(nominal.observations)
    arrays["raw_actions"].append(nominal.raw_actions)
    arrays["final_actions"].append(nominal.final_actions)
    arrays["driving_actions"].append(nominal.driving_actions)
    arrays["trajectory_ids"].append(nominal.trajectory_ids.astype(np.int64))
    for name in (
        "progress", "elapsed_s", "speed_mps", "position_m", "heading_error_rad",
        "wheel_contacts", "overlay_active",
    ):
        arrays[name].append(getattr(nominal, name))
    arrays["student_actions"].append(np.full_like(nominal.final_actions, np.nan, dtype=np.float32))
    nominal_outcomes = np.asarray([
        "finish" if int(trajectory) in nominal_success else "nominal"
        for trajectory in nominal.trajectory_ids
    ], dtype="U16")
    arrays["outcomes"].append(nominal_outcomes)
    arrays["sources"].append(nominal_sources)
    arrays["dagger_rounds"].append(np.full(nominal_count, -1, dtype=np.int32))
    all_successful = set(nominal_success)
    recovery_count = 0
    focus_bins: set[str] = set()
    failure_focus = [np.zeros(nominal_count, dtype=bool)]
    measured_failure_progress: list[float] = []
    for dagger in dagger_rounds:
        count = len(dagger.observations)
        round_failure_focus = np.zeros(count, dtype=bool)
        id_map: dict[int, int] = {}
        for episode in np.unique(dagger.episode_ids):
            id_map[int(episode)] = next_id
            next_id += 1
        remapped = np.asarray([id_map[int(value)] for value in dagger.episode_ids], dtype=np.int64)
        arrays["observations"].append(dagger.observations)
        arrays["raw_actions"].append(dagger.teacher_raw_actions)
        arrays["final_actions"].append(dagger.teacher_actions)
        arrays["driving_actions"].append(dagger.teacher_driving_actions)
        arrays["trajectory_ids"].append(remapped)
        arrays["progress"].append(dagger.progress)
        arrays["elapsed_s"].append(dagger.elapsed_s)
        arrays["speed_mps"].append(dagger.speed_mps)
        arrays["position_m"].append(dagger.position_m)
        arrays["heading_error_rad"].append(dagger.heading_error_rad)
        arrays["wheel_contacts"].append(dagger.wheel_contacts)
        arrays["overlay_active"].append(dagger.overlay_active)
        arrays["student_actions"].append(dagger.student_actions)
        arrays["outcomes"].append(dagger.episode_outcomes.astype("U16"))
        arrays["sources"].append(np.full(count, "dagger", dtype="U8"))
        arrays["dagger_rounds"].append(dagger.dagger_rounds.astype(np.int32))
        for old_id, new_id in id_map.items():
            old_mask = dagger.episode_ids == old_id
            if np.any(dagger.episode_outcomes[old_mask] == "finish"):
                all_successful.add(new_id)
            else:
                # Recorded recovery trajectories are valid split units even if
                # PPO did not finish them.
                all_successful.add(new_id)
        recovery_count += count
        for episode_id in np.unique(dagger.episode_ids):
            episode_indices = np.flatnonzero(dagger.episode_ids == episode_id)
            if not episode_indices.size:
                continue
            outcome = str(dagger.episode_outcomes[episode_indices[0]])
            if outcome in {"crash", "off_track", "stall"}:
                failure_progress = float(dagger.failure_progress[episode_indices[0]])
                measured_failure_progress.append(failure_progress)
                round_failure_focus[episode_indices] = (
                    np.abs(dagger.progress[episode_indices] - failure_progress) <= 0.05 + 1e-6
                )
        failure_focus.append(round_failure_focus)
        report = dagger_action_error_report(dagger)
        window = report["first_major_divergence_progress"]
        if window is not None:
            focus_bins.add(f"{int(window[0] * 100):02d}-{int(window[1] * 100):02d}%")
    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    failure_focus_mask = np.concatenate(failure_focus)
    sources = merged["sources"].astype(str)
    errors = np.zeros((len(sources), 2), dtype=np.float32)
    dagger_mask = sources == "dagger"
    if np.any(dagger_mask):
        offset = nominal_count
        errors[offset:, :] = np.concatenate([item.action_errors for item in dagger_rounds], axis=0)
    normalized_nominal = nominal_weight / max(nominal_weight + recovery_weight, 1e-9)
    normalized_recovery = recovery_weight / max(nominal_weight + recovery_weight, 1e-9)
    sample_weights = np.zeros(len(sources), dtype=np.float32)
    for source, mass in (("nominal", normalized_nominal), ("dagger", normalized_recovery)):
        mask = sources == source
        if not np.any(mask):
            continue
        within = np.ones(int(mask.sum()), dtype=np.float64)
        if source == "dagger":
            action_error = errors[mask].sum(axis=1)
            within *= np.minimum(1.0 + action_error / 0.25, error_weight_cap)
            recovery_progress = merged["progress"][mask]
            if focus_bins:
                focus_mask = np.zeros(int(mask.sum()), dtype=bool)
                for value in focus_bins:
                    lo, hi = [int(part) for part in value.removesuffix("%").split("-")]
                    focus_mask |= (recovery_progress >= lo / 100) & (recovery_progress < hi / 100)
                within *= np.where(focus_mask, 2.0, 1.0)
            within *= np.where(failure_focus_mask[mask], 2.0, 1.0)
            # The measured final two seconds before a failure are useful for
            # recovery; later terminal transitions were filtered at collection.
            fail_times = np.concatenate([item.time_to_failure_s for item in dagger_rounds])
            within *= np.where(fail_times <= failure_window_s, 1.5, 1.0)
        sample_weights[mask] = (within / within.sum() * mass).astype(np.float32)
    trajectory_ids = merged["trajectory_ids"]
    successful_ids = sorted(all_successful)
    aggregated = TeacherDataset(
        observations=merged["observations"], raw_actions=merged["raw_actions"],
        final_actions=merged["final_actions"], driving_actions=merged["driving_actions"],
        trajectory_ids=trajectory_ids, progress=merged["progress"], elapsed_s=merged["elapsed_s"],
        speed_mps=merged["speed_mps"], position_m=merged["position_m"],
        heading_error_rad=merged["heading_error_rad"], wheel_contacts=merged["wheel_contacts"],
        overlay_active=merged["overlay_active"], successful_trajectory_ids=successful_ids,
        teacher_id=nominal.teacher_id, sources=sources, dagger_rounds=merged["dagger_rounds"],
        sample_weights=sample_weights, student_actions=merged["student_actions"],
        outcomes=merged["outcomes"],
    )
    return aggregated, {
        "nominal_samples": nominal_count, "recovery_samples": recovery_count,
        "total_samples": len(aggregated.observations), "nominal_weight": normalized_nominal,
        "recovery_weight": normalized_recovery,
        "first_major_divergence_windows": sorted(focus_bins),
        "measured_failure_progress": measured_failure_progress,
        "failure_focus_samples": int(failure_focus_mask.sum()),
        "action_error_report": [dagger_action_error_report(item) for item in dagger_rounds],
    }


def _config_from_teacher(metadata: ModelMetadata, output_root: Path, *, device: str) -> TrainingConfig:
    value = dict(metadata.training_config)
    value.update({
        "algorithm": "ppo", "device": device, "timesteps": 10_000_000,
        "output_root": str(output_root), "ppo": asdict(PPOConfig(
            architecture=metadata.architecture, learning_rate=3e-5, rollout_steps=1024,
            batch_size=128, epochs=3, entropy_coefficient=1e-4,
            target_lap_s=22.0,
        )),
        "dqn": None, "tqc": None,
        "evaluation": asdict(EvaluationConfig(interval_steps=5_000, episodes=5)),
    })
    return TrainingConfig.from_dict(value)


def _apply_reward_profile(config: TrainingConfig, profile: str | None) -> TrainingConfig:
    """Override the teacher's inherited shaping when a run names a profile."""

    if profile is not None:
        config.rewards = RewardProfileStore().load(profile)
        config.reward_profile = profile
    return config


def _gym_env(config: TrainingConfig, observation_size: int) -> gym.Env:
    """Provide shape-correct spaces without opening the live simulator for cloning."""
    env = gym.Env()
    env.observation_space = gym.spaces.Box(
        -5.0, 5.0, shape=(observation_size,), dtype=np.float32
    )
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    return env


def _run_with_stop_file(
    runner: TrainingRunner, stop_file: Path | None, **run_kwargs: Any,
) -> Path:
    """Let a stop-file request reach an active PPO rollout, not just the next run."""
    finished = threading.Event()

    def watch_stop_file() -> None:
        while not finished.wait(0.25):
            if stop_file is not None and stop_file.exists():
                runner.stop()
                return

    watcher = threading.Thread(target=watch_stop_file, name="teacher-student-stop", daemon=True)
    if stop_file is not None and stop_file.exists():
        runner.stop()
    watcher.start()
    try:
        return runner.run(**run_kwargs)
    finally:
        finished.set()
        watcher.join(timeout=1.0)


def _teacher_and_config(
    teacher_path: Path, output_root: Path, device: str
) -> tuple[Any, ModelMetadata, TrainingConfig, TrainingRunner]:
    registry = ModelRegistry(teacher_path.parents[3])
    metadata = registry.read_metadata(teacher_path)
    config = _config_from_teacher(metadata, output_root, device=device)
    verify_compatibility(metadata, config)
    teacher_config = TrainingConfig.from_dict(dict(metadata.training_config))
    runner = TrainingRunner(teacher_config)
    env = runner._environment()
    try:
        teacher = backend_for("tqc").load_model(
            teacher_path / "policy.zip", env, device, resume=False
        )
    except BaseException:
        env.close()
        raise
    teacher.policy_overlays = list(metadata.policy_overlays)
    return teacher, metadata, config, runner


def _new_student(config: TrainingConfig, observation_size: int) -> Any:
    env = _gym_env(config, observation_size)
    return backend_for("ppo").create_model(config, env, config.device)


def _initial_dagger_student(output: Path, registry: ModelRegistry, track_name: str) -> Path:
    """Prefer the PPO actor distilled from this teacher over the global champion."""
    pretrained = output / "pretrained"
    if (pretrained / "policy.zip").is_file():
        return pretrained
    champion = registry.slot(track_name, "ppo", "champion")
    if (champion / "policy.zip").is_file():
        return champion
    return registry.slot(track_name, "ppo", "latest")


def _dagger_seed_student(
    requested: Path | None, output: Path, registry: ModelRegistry, track_name: str,
) -> Path:
    """Use an explicit evaluated PPO seed when supplied, else the teacher-pretrained actor."""
    if requested is None:
        return _initial_dagger_student(output, registry, track_name)
    if not (requested / "policy.zip").is_file() or not (requested / "metadata.json").is_file():
        raise FileNotFoundError(f"DAgger seed must contain policy.zip and metadata.json: {requested}")
    return requested


def _ensure_ppo_teacher_anchor(anchor_dir: Path, source: Path) -> Path:
    """Snapshot the starting PPO actor once for a stable, run-wide KL anchor."""
    anchor_policy = anchor_dir / "policy.zip"
    if anchor_policy.is_file():
        return anchor_policy
    if anchor_dir.exists():
        raise FileNotFoundError(
            f"incomplete PPO teacher anchor; expected {anchor_policy}"
        )
    if not (source / "policy.zip").is_file() or not (source / "metadata.json").is_file():
        raise FileNotFoundError(
            f"PPO teacher anchor source must contain policy.zip and metadata.json: {source}"
        )
    anchor_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, anchor_dir)
    return anchor_policy


def _evaluation_rank(evaluation: dict[str, Any]) -> tuple[float, float, float]:
    """Rank reliable finishes first, then pace; use progress before first finish."""
    finish_rate = float(evaluation.get("finish_rate", 0.0) or 0.0)
    progress = float(evaluation.get("median_progress", 0.0) or 0.0)
    lap = evaluation.get("median_lap_s")
    if lap is None:
        lap = evaluation.get("best_lap_s")
    if finish_rate > 0 and lap is not None:
        return finish_rate, -float(lap), progress
    return finish_rate, progress, 0.0


def _runner_evaluation_rank(evaluation: dict[str, Any]) -> tuple[float, ...]:
    fields = EvaluationResult.__dataclass_fields__
    return EvaluationResult(**{
        name: evaluation[name] for name in fields if name in evaluation
    }).rank()


def _evaluation_confirms_target(
    evaluation: dict[str, Any], target_lap_s: float,
) -> bool:
    fields = EvaluationResult.__dataclass_fields__
    result = EvaluationResult(**{
        name: evaluation[name] for name in fields if name in evaluation
    })
    return result.confirms_target_lap(target_lap_s)


def _partial_progress_gate_passed(
    evaluation: dict[str, Any], minimum_progress: float,
) -> bool:
    return float(evaluation.get("median_progress", 0.0) or 0.0) >= minimum_progress


def _dagger_rounds_remain_after_reliable_gate(
    *, continue_after_reliable: bool, first_round: int,
    next_round: int, rounds_to_run: int | None,
) -> bool:
    """Whether to keep collecting requested recovery data before PPO fine-tuning."""
    return bool(
        continue_after_reliable
        and rounds_to_run is not None
        and next_round < first_round + rounds_to_run
    )


def _should_resume_ppo_champion(
    candidate: dict[str, Any], champion: dict[str, Any], *,
    progress_tolerance: float = 0.05, lap_tolerance_s: float = 0.5,
) -> bool:
    """Roll back only a material loss, so small noisy dips do not erase learning."""
    candidate_finish = float(candidate.get("finish_rate", 0.0) or 0.0)
    champion_finish = float(champion.get("finish_rate", 0.0) or 0.0)
    if candidate_finish < champion_finish:
        return True
    candidate_progress = float(candidate.get("median_progress", 0.0) or 0.0)
    champion_progress = float(champion.get("median_progress", 0.0) or 0.0)
    if candidate_progress < champion_progress - progress_tolerance:
        return True
    if candidate_finish > 0 and candidate_finish >= champion_finish:
        candidate_lap = candidate.get("median_lap_s")
        champion_lap = champion.get("median_lap_s")
        if (
            candidate_lap is not None and champion_lap is not None
            and float(candidate_lap) > float(champion_lap) + lap_tolerance_s
        ):
            return True
    return False


def _promote_ppo_champion_if_better(
    source_registry: ModelRegistry, destination_registry: ModelRegistry, track_name: str,
) -> Path | None:
    """Copy an evaluated isolated PPO champion only when it beats the main registry."""
    source = source_registry.slot(track_name, "ppo", "champion")
    source_metadata_path = source / "metadata.json"
    if not source_metadata_path.is_file():
        return None
    candidate = source_registry.read_metadata(source)
    if candidate.algorithm != "ppo" or not candidate.evaluation:
        return None
    candidate_rank = _runner_evaluation_rank(candidate.evaluation)
    destination = destination_registry.slot(track_name, "ppo", "champion")
    incumbent_metadata_path = destination / "metadata.json"
    if incumbent_metadata_path.is_file():
        incumbent = destination_registry.read_metadata(destination)
        if incumbent.evaluation:
            incumbent_rank = _runner_evaluation_rank(incumbent.evaluation)
            if candidate_rank <= incumbent_rank:
                return None
    if source.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}-staging-{uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        promote_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _save_student(model: Any, config: TrainingConfig, directory: Path, *, report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / "policy.zip"))
    counts = backend_for("ppo").parameter_counts(model)
    registry = ModelRegistry(config.output_root)
    metadata = ModelMetadata(
        algorithm="ppo", architecture=config.ppo.architecture,
        actor_parameters=counts["actor"], critic_parameters=counts["critic"],
        total_trainable_parameters=counts["total"],
        observation_schema=OBSERVATION_SCHEMA, action_schema=ACTION_SCHEMA,
        action_semantics=ACTION_SEMANTICS, track_name=config.track_name,
        track_id=config.track_id, lookahead_count=config.lookahead_count,
        reward_profile=config.reward_profile, curriculum=asdict(config.curriculum),
        training_config=config.to_dict(), training_timesteps=int(model.num_timesteps),
        simulator_ticks=0, wall_seconds=0.0, seed=config.seed, device=config.device,
        finishes=0, crashes=0, policy_overlays=[
            layer for layer in report.get("teacher_overlays", [])
            if layer.get("kind") == "air_brake"
        ],
    )
    registry.write_metadata(directory, metadata)
    (directory / "teacher-student.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def offline_validate(model: Any, dataset: TeacherDataset) -> dict[str, Any]:
    indices = np.flatnonzero(np.isin(dataset.trajectory_ids, dataset.successful_trajectory_ids))
    return imitation_metrics(model, dataset, indices)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("collect", "pretrain", "validate", "value_warmup", "finetune", "dagger", "full"),
        required=True,
    )
    parser.add_argument(
        "--teacher", type=Path,
        default=Path("models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"),
        help="frozen TQC champion directory containing policy.zip and metadata.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--dataset", type=Path, default=Path("runs/teacher-student/summer-1-teacher.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reward-profile", help="named reward profile for DAgger validation and PPO training",
    )
    parser.add_argument("--laps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--supervised-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--student-architecture", choices=("tiny", "compact", "standard"),
        help="override the distilled PPO student's network size",
    )
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--warmup-steps", type=int, default=2_048)
    parser.add_argument("--rounds", type=int, default=3, help="maximum DAgger rounds")
    parser.add_argument("--episodes-per-round", type=int, default=8)
    parser.add_argument(
        "--dagger-run-name", default="dagger",
        help="isolated DAgger series name, for example dagger-mean-policy",
    )
    parser.add_argument(
        "--dagger-initial-student", type=Path,
        help="evaluated PPO checkpoint to use as the first DAgger student",
    )
    parser.add_argument(
        "--stochastic-dagger", action="store_true",
        help="sample PPO actions during DAgger; default follows deterministic evaluation",
    )
    parser.add_argument(
        "--initial-action-std", type=float, default=0.15,
        help="PPO Gaussian action standard deviation after teacher regression",
    )
    parser.add_argument(
        "--failure-window-s", type=float, default=2.0,
        help="seconds of PPO-driven recovery states to retain before failure",
    )
    parser.add_argument("--nominal-weight", type=float, default=0.6)
    parser.add_argument("--recovery-weight", type=float, default=0.4)
    parser.add_argument(
        "--ppo-learning-rate", type=float, default=1e-5,
        help="on-policy fine-tuning rate; the lower default limits destructive actor jumps",
    )
    parser.add_argument(
        "--ppo-target-kl", type=float, default=0.003,
        help="early-stop PPO minibatches at this approximate KL divergence",
    )
    parser.add_argument(
        "--ppo-anchor-kl", type=float, default=0.1,
        help="penalize drift from the run's immutable starting PPO actor",
    )
    parser.add_argument(
        "--ppo-rollback-progress-tolerance", type=float, default=0.02,
        help="allow this much median-progress loss before restoring the champion",
    )
    parser.add_argument("--until-finishing", action="store_true")
    parser.add_argument("--continue-to-rl", action="store_true")
    parser.add_argument(
        "--continue-to-rl-on-partial", action="store_true",
        help="allow continuous PPO fine-tuning once the best student reaches the progress gate",
    )
    parser.add_argument(
        "--continue-dagger-after-reliable", action="store_true",
        help="complete all configured DAgger rounds before PPO fine-tuning, even if the current best already finishes",
    )
    parser.add_argument(
        "--partial-rl-min-progress", type=float, default=0.35,
        help="minimum best median track progress for the partial-progress PPO gate",
    )
    parser.add_argument(
        "--rl-output-root", type=Path,
        help="isolated model registry for PPO fine-tuning; defaults to a run-specific subdirectory",
    )
    parser.add_argument("--reliability-finishes", type=int, default=5)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument(
        "--max-rounds", type=int, default=1,
        help="fine-tuning blocks; set to 0 in full mode to repeat until the 22-second target",
    )
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    if not args.dagger_run_name or any(char in args.dagger_run_name for char in "/\\"):
        parser.error("--dagger-run-name must be a non-empty path-free name")
    if not 0 < args.initial_action_std <= 1:
        parser.error("--initial-action-std must be in (0, 1]")
    if args.failure_window_s <= 0:
        parser.error("--failure-window-s must be positive")
    if not 0.0 <= args.partial_rl_min_progress <= 1.0:
        parser.error("--partial-rl-min-progress must be in [0, 1]")
    if args.ppo_learning_rate <= 0 or args.ppo_target_kl <= 0:
        parser.error("PPO learning rate and target KL must be positive")
    if not np.isfinite(args.ppo_anchor_kl) or args.ppo_anchor_kl < 0:
        parser.error("--ppo-anchor-kl must be finite and nonnegative")
    if not 0 <= args.ppo_rollback_progress_tolerance < 1:
        parser.error("--ppo-rollback-progress-tolerance must be in [0, 1)")
    teacher_path = args.teacher.resolve()
    teacher, teacher_meta, config, teacher_runner = _teacher_and_config(
        teacher_path, args.output_root, args.device
    )
    _apply_reward_profile(config, args.reward_profile)
    if args.student_architecture is not None:
        if config.ppo is None:
            config.ppo = PPOConfig(architecture=args.student_architecture)
        else:
            config.ppo.architecture = args.student_architecture
    teacher_digest = tree_sha256(teacher_path)
    output = args.output_root / "summer-1" / "ppo" / "teacher-student"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(teacher_path / "metadata.json", output / "teacher-metadata.json")
    manifest_path = output / "teacher-manifest.json"
    teacher_manifest: dict[str, Any] = {
        "teacher_path": str(teacher_path), "teacher_sha256": teacher_digest,
        "teacher_git_commit": teacher_meta.git_commit,
        "teacher_metadata": json.loads((teacher_path / "metadata.json").read_text(encoding="utf-8")),
        "teacher_overlay_stack": teacher_meta.policy_overlays,
        "teacher_action_semantics": teacher_meta.action_semantics or ACTION_SEMANTICS,
        "observation_schema": teacher_meta.observation_schema,
        "lookahead_count": teacher_meta.lookahead_count,
        "recorded_best_lap_s": (teacher_meta.evaluation or {}).get("best_lap_s"),
    }
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("teacher_sha256") == teacher_digest:
            teacher_manifest.update(previous_manifest)
    manifest_path.write_text(
        json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.stage == "dagger":
        if args.rounds < 1 or args.episodes_per_round < 2 or not 1 <= args.reliability_finishes <= 5:
            raise ValueError("DAgger needs positive rounds, at least two episodes, and a 1–5 finish gate")
        nominal = TeacherDataset.load(args.dataset)
        if nominal.teacher_id != teacher_digest:
            raise ValueError("nominal teacher dataset does not match the frozen TQC teacher")
        registry = ModelRegistry(args.output_root)
        dagger_root = output / args.dagger_run_name
        dagger_data_root = args.dataset.parent / args.dagger_run_name
        dagger_root.mkdir(parents=True, exist_ok=True)
        dagger_data_root.mkdir(parents=True, exist_ok=True)
        completed = []
        for round_dir in dagger_root.glob("round-[0-9][0-9][0-9]"):
            if (round_dir / "student" / "policy.zip").is_file():
                try:
                    completed.append((int(round_dir.name.rsplit("-", 1)[1]), round_dir))
                except ValueError:
                    continue
        if completed:
            last_round = max(round_index for round_index, _ in completed)
            ranked = []
            for round_index, round_dir in sorted(completed):
                evaluation_path = round_dir / "live-evaluation.json"
                if evaluation_path.is_file():
                    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                    ranked.append((_evaluation_rank(evaluation), round_index, round_dir))
            if ranked:
                _rank, _best_index, best_dir = max(ranked, key=lambda row: row[0])
                student_path = best_dir / "student"
            else:
                student_path = max(completed, key=lambda row: row[0])[1] / "student"
            first_round = last_round + 1
        else:
            student_path = _dagger_seed_student(
                args.dagger_initial_student, output, registry, config.track_name,
            )
            first_round = 1
        if not (student_path / "policy.zip").is_file():
            raise FileNotFoundError("DAgger requires the saved pretrained PPO student")
        student_meta = registry.read_metadata(student_path)
        if student_meta.algorithm != "ppo" or student_meta.action_schema != ACTION_SCHEMA:
            raise ValueError("DAgger student must use the continuous PPO action schema")
        student = backend_for("ppo").load_model(
            student_path / "policy.zip", _gym_env(config, nominal.observations.shape[1]), args.device
        )
        if args.reward_profile is not None:
            # This starting actor comes from the old reward objective; discard its
            # optimizer moments before either DAgger regression or value warmup.
            student.policy.optimizer.state.clear()
        with th.no_grad():
            student.policy.log_std.fill_(float(np.log(args.initial_action_std)))
        baseline_student_path = dagger_root / "baseline-student"
        _save_student(
            student, config, baseline_student_path,
            report={"teacher_overlays": teacher_meta.policy_overlays,
                    "action_std": [args.initial_action_std, args.initial_action_std]},
        )
        air_brake_overlays = [
            layer for layer in teacher_meta.policy_overlays if layer.get("kind") == "air_brake"
        ]
        evaluator = TrainingRunner(config)
        evaluator._ppo_air_brake_overlays = air_brake_overlays
        baseline = evaluate_model(
            student, lambda: evaluator._environment(), episodes=5, seed=args.seed + 20_000,
        )
        baseline_report = baseline.to_dict()
        (dagger_root / "baseline-evaluation.json").write_text(
            json.dumps(baseline_report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"dagger_baseline": baseline_report}, separators=(",", ":")), flush=True)
        best_student_path = student_path
        best_evaluation = baseline_report
        historical_best = []
        for _round_index, round_dir in sorted(completed):
            evaluation_path = round_dir / "live-evaluation.json"
            if evaluation_path.is_file():
                historical_best.append((
                    _evaluation_rank(json.loads(evaluation_path.read_text(encoding="utf-8"))),
                    round_dir / "student",
                ))
        if historical_best:
            prior_rank, prior_path = max(historical_best, key=lambda row: row[0])
            if prior_rank > _evaluation_rank(best_evaluation):
                best_student_path = prior_path
                best_evaluation = json.loads(
                    (prior_path.parent / "live-evaluation.json").read_text(encoding="utf-8")
                )
            else:
                best_student_path = baseline_student_path
        else:
            best_student_path = baseline_student_path
        if best_student_path != baseline_student_path:
            student = backend_for("ppo").load_model(
                best_student_path / "policy.zip",
                _gym_env(config, nominal.observations.shape[1]), args.device,
            )
        student_path = best_student_path
        all_datasets = sorted(dagger_data_root.glob("round-[0-9][0-9][0-9].npz"))
        rounds_to_run = None if args.until_finishing else args.rounds
        round_index = first_round
        last_evaluation = baseline
        partial_rl_gate_passed = (
            args.continue_to_rl and args.continue_to_rl_on_partial
            and _partial_progress_gate_passed(best_evaluation, args.partial_rl_min_progress)
        )
        if partial_rl_gate_passed:
            print(json.dumps({
                "partial_progress_gate_passed": True,
                "best_median_progress": best_evaluation.get("median_progress"),
                "required_median_progress": args.partial_rl_min_progress,
                "student_checkpoint": str(best_student_path),
            }, separators=(",", ":")), flush=True)
        while (
            (rounds_to_run is None or round_index < first_round + rounds_to_run)
            and not partial_rl_gate_passed
        ):
            if args.stop_file is not None and args.stop_file.exists():
                print(json.dumps(
                    {"dagger_stopped": True, "next_round": round_index},
                    separators=(",", ":"),
                ), flush=True)
                break
            round_name = f"round-{round_index:03d}"
            round_dir = dagger_root / round_name
            round_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = dagger_data_root / f"{round_name}.npz"
            if dataset_path.is_file():
                dagger_data = DaggerDataset.load(dataset_path)
            else:
                collector = TrainingRunner(config)
                collector._ppo_air_brake_overlays = air_brake_overlays
                env = collector._environment()
                try:
                    dagger_data = collect_dagger_data(
                        student, teacher, env, episodes=args.episodes_per_round,
                        dagger_round=round_index, seed=args.seed + round_index * 1_000,
                        teacher_id=teacher_digest,
                        deterministic=not args.stochastic_dagger,
                        failure_window_s=args.failure_window_s,
                    )
                    dagger_data.save(dataset_path)
                finally:
                    env.close()
            all_datasets = sorted(dagger_data_root.glob("round-[0-9][0-9][0-9].npz"))
            dagger_collection = [DaggerDataset.load(path) for path in all_datasets]
            aggregated, aggregate_report = aggregate_teacher_datasets(
                nominal, dagger_collection, nominal_weight=args.nominal_weight,
                recovery_weight=args.recovery_weight,
                failure_window_s=args.failure_window_s,
            )
            aggregate_path = round_dir / "aggregated-dataset.npz"
            aggregated.save(aggregate_path)
            report = pretrain_actor(
                student, aggregated, epochs=args.epochs, seed=args.seed + round_index,
                learning_rate=args.supervised_learning_rate,
                sample_weights=aggregated.sample_weights,
                initial_action_std=args.initial_action_std,
            )
            report.update({
                "round": round_index, "teacher_sha256": teacher_digest,
                "teacher_lap_s": (teacher_meta.evaluation or {}).get("best_lap_s"),
                "aggregate": aggregate_report,
                "nominal_sample_count": len(nominal.observations),
                "recovery_sample_count": len(aggregated.observations) - len(nominal.observations),
                "teacher_overlays": list(teacher_meta.policy_overlays),
                "action_schema": ACTION_SCHEMA, "action_semantics": ACTION_SEMANTICS,
            })
            _save_student(student, config, round_dir / "student", report=report)
            (round_dir / "pretraining-report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            evaluator._ppo_air_brake_overlays = air_brake_overlays
            last_evaluation = evaluate_model(
                student, lambda: evaluator._environment(), episodes=5,
                seed=args.seed + 30_000 + round_index * 5,
            )
            evaluation_report = last_evaluation.to_dict()
            (round_dir / "live-evaluation.json").write_text(
                json.dumps(evaluation_report, indent=2) + "\n", encoding="utf-8"
            )
            summary = {
                "dagger_round": round_index,
                "dagger_run_name": args.dagger_run_name,
                "collection_deterministic": dagger_data.collection_deterministic,
                "failure_window_s": args.failure_window_s,
                "dataset": str(dataset_path),
                "samples": len(dagger_data.observations),
                "failure_progress": dagger_action_error_report(dagger_data)["failure_progress"],
                "action_error": dagger_action_error_report(dagger_data),
                "nominal_validation": report.get("validation_by_source", {}).get("nominal"),
                "dagger_validation": report.get("validation_by_source", {}).get("dagger"),
                "finish_rate": last_evaluation.finish_rate,
                "median_progress": last_evaluation.median_progress,
                "median_lap_s": last_evaluation.median_lap_s,
                "crashes": last_evaluation.crash_rate,
                "off_track": last_evaluation.off_track_rate,
                "stalls": last_evaluation.stall_rate,
                "student_checkpoint": str(round_dir / "student"),
            }
            (round_dir / "round-report.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            teacher_manifest.setdefault("dagger_rounds", []).append(summary)
            (output / "teacher-manifest.json").write_text(
                json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, separators=(",", ":")), flush=True)
            candidate_path = round_dir / "student"
            if _evaluation_rank(evaluation_report) > _evaluation_rank(best_evaluation):
                best_evaluation = evaluation_report
                best_student_path = candidate_path
                student_path = candidate_path
            else:
                student = backend_for("ppo").load_model(
                    best_student_path / "policy.zip",
                    _gym_env(config, nominal.observations.shape[1]), args.device,
                )
                student.policy.set_training_mode(False)
                student_path = best_student_path
            round_index += 1
            if _evaluation_rank(best_evaluation)[0] >= args.reliability_finishes / 5:
                if not _dagger_rounds_remain_after_reliable_gate(
                    continue_after_reliable=args.continue_dagger_after_reliable,
                    first_round=first_round, next_round=round_index,
                    rounds_to_run=rounds_to_run,
                ):
                    print(json.dumps({"reliability_gate_passed": True,
                                      "finish_rate": best_evaluation.get("finish_rate"),
                                      "student_checkpoint": str(student_path)}, separators=(",", ":")), flush=True)
                    break
                print(json.dumps({
                    "reliability_gate_passed": True,
                    "continuing_configured_dagger_rounds": True,
                    "finish_rate": best_evaluation.get("finish_rate"),
                    "student_checkpoint": str(student_path),
                    "next_round": round_index,
                }, separators=(",", ":")), flush=True)
            partial_rl_gate_passed = (
                args.continue_to_rl and args.continue_to_rl_on_partial
                and _partial_progress_gate_passed(best_evaluation, args.partial_rl_min_progress)
            )
            if partial_rl_gate_passed:
                print(json.dumps({
                    "partial_progress_gate_passed": True,
                    "best_median_progress": best_evaluation.get("median_progress"),
                    "required_median_progress": args.partial_rl_min_progress,
                    "student_checkpoint": str(best_student_path),
                }, separators=(",", ":")), flush=True)
                break
        reliability_gate_passed = (
            _evaluation_rank(best_evaluation)[0] >= args.reliability_finishes / 5
        )
        if (
            args.continue_to_rl and (reliability_gate_passed or partial_rl_gate_passed)
            and not (args.stop_file is not None and args.stop_file.exists())
        ):
            rl_output_root = args.rl_output_root or (
                output / "ppo-rl" / args.dagger_run_name
            )
            config.output_root = rl_output_root
            rl_registry = ModelRegistry(rl_output_root)
            isolated_latest = rl_registry.slot(config.track_name, "ppo", "latest")
            if (isolated_latest / "metadata.json").is_file():
                isolated_champion = rl_registry.slot(config.track_name, "ppo", "champion")
                latest_evaluation = rl_registry.read_metadata(isolated_latest).evaluation or {}
                champion_evaluation = (
                    rl_registry.read_metadata(isolated_champion).evaluation or {}
                    if (isolated_champion / "metadata.json").is_file() else {}
                )
                resume_champion = bool(champion_evaluation) and _should_resume_ppo_champion(
                    latest_evaluation, champion_evaluation,
                    progress_tolerance=args.ppo_rollback_progress_tolerance,
                    lap_tolerance_s=0.5,
                )
                student_path = isolated_champion if resume_champion else isolated_latest
                print(json.dumps({
                    "ppo_rl_resume": True, "checkpoint": str(student_path),
                    "timesteps": rl_registry.read_metadata(student_path).training_timesteps,
                    "resuming_from_champion": resume_champion,
                }, separators=(",", ":")), flush=True)
            else:
                config.timesteps = args.warmup_steps
                warmup = TrainingRunner(config)
                warmup._ppo_air_brake_overlays = air_brake_overlays
                student_path = _run_with_stop_file(
                    warmup, args.stop_file,
                    resume=student_path, freeze_ppo_actor=True,
                    allow_ppo_reward_change=args.reward_profile is not None,
                )
            if args.ppo_anchor_kl > 0:
                anchor_policy = _ensure_ppo_teacher_anchor(
                    rl_output_root / "ppo-teacher-anchor", student_path,
                )
                config.ppo.teacher_model = str(anchor_policy.resolve())
                config.ppo.teacher_kl_coefficient = args.ppo_anchor_kl
                print(json.dumps({
                    "ppo_teacher_anchor": str(anchor_policy),
                    "teacher_kl_coefficient": args.ppo_anchor_kl,
                }, separators=(",", ":")), flush=True)
            else:
                config.ppo.teacher_model = None
                config.ppo.teacher_kl_coefficient = 0.0
            while True:
                if args.stop_file is not None and args.stop_file.exists():
                    print(json.dumps({
                        "ppo_rl_stopped": True, "checkpoint": str(student_path),
                        "timesteps": rl_registry.read_metadata(student_path).training_timesteps,
                    }, separators=(",", ":")), flush=True)
                    break
                config.timesteps = args.timesteps
                config.ppo.learning_rate = args.ppo_learning_rate
                config.ppo.entropy_coefficient = 1e-4
                config.ppo.target_kl = args.ppo_target_kl
                runner = TrainingRunner(config)
                # Keep updates across mild eval noise while restoring a measured
                # champion immediately after a material regression.
                latest = _run_with_stop_file(
                    runner, args.stop_file,
                    resume=student_path, rollback_to_champion=True,
                    ppo_rollback_progress_tolerance=args.ppo_rollback_progress_tolerance,
                    ppo_rollback_lap_tolerance_s=0.5,
                    allow_ppo_reward_change=args.reward_profile is not None,
                )
                latest_meta = rl_registry.read_metadata(latest)
                evaluation = latest_meta.evaluation or {}
                best_lap = evaluation.get("best_lap_s")
                isolated_champion = rl_registry.slot(config.track_name, "ppo", "champion")
                champion_evaluation = (
                    rl_registry.read_metadata(isolated_champion).evaluation or {}
                )
                resume_champion = _should_resume_ppo_champion(
                    evaluation, champion_evaluation,
                    progress_tolerance=args.ppo_rollback_progress_tolerance,
                    lap_tolerance_s=0.5,
                )
                promoted = _promote_ppo_champion_if_better(
                    rl_registry, registry, config.track_name,
                )
                print(json.dumps({
                    "target_reached": _evaluation_confirms_target(evaluation, 22.0),
                    "best_lap_s": best_lap, "finish_rate": evaluation.get("finish_rate"),
                    "median_progress": evaluation.get("median_progress"),
                    "timesteps": latest_meta.training_timesteps, "checkpoint": str(latest),
                    "isolated_champion": str(isolated_champion),
                    "resuming_from_champion": resume_champion,
                    "next_resume": str(isolated_champion if resume_champion else latest),
                    "promoted_to_main_champion": str(promoted) if promoted else None,
                }, indent=2), flush=True)
                student_path = isolated_champion if resume_champion else latest
                if _evaluation_confirms_target(evaluation, 22.0):
                    break
        elif (
            args.continue_to_rl and (reliability_gate_passed or partial_rl_gate_passed)
            and args.stop_file is not None and args.stop_file.exists()
        ):
            print(json.dumps({
                "ppo_rl_stopped": True, "checkpoint": str(best_student_path),
                "reason": "stop requested after DAgger evaluation",
            }, separators=(",", ":")), flush=True)
        if tree_sha256(teacher_path) != teacher_digest:
            raise RuntimeError("frozen TQC teacher changed during DAgger")
    if args.stage in {"validate", "full"}:
        teacher_eval = evaluate_model(
            teacher, lambda: teacher_runner._environment(), episodes=5,
            seed=args.seed + 5_000,
        )
        teacher_manifest["teacher_live_evaluation"] = teacher_eval.to_dict()
        (output / "teacher-manifest.json").write_text(
            json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"teacher_baseline": teacher_eval.to_dict()}, indent=2))
    dataset = None
    try:
        if args.stage in {"collect", "full"}:
            env = teacher_runner._environment()
            try:
                dataset = collect_teacher_data(
                    teacher, env, successful_laps=args.laps, seed=args.seed,
                    teacher_id=teacher_digest,
                )
                dataset.save(args.dataset)
            finally:
                env.close()
        if args.stage in {"pretrain", "validate", "full"}:
            dataset = dataset or TeacherDataset.load(args.dataset)
            student_path = output / "pretrained" / "policy.zip"
            student = (
                backend_for("ppo").load_model(student_path, None, args.device)
                if args.stage == "validate" and student_path.is_file()
                else _new_student(config, dataset.observations.shape[1])
            )
            if args.stage != "validate":
                report = pretrain_actor(
                    student, dataset, epochs=args.epochs, seed=args.seed,
                    learning_rate=args.supervised_learning_rate,
                    initial_action_std=args.initial_action_std,
                )
                report.update({
                    "teacher_path": str(teacher_path), "teacher_sha256": teacher_digest,
                    "teacher_lap_s": (teacher_meta.evaluation or {}).get("best_lap_s"),
                    "action_schema": ACTION_SCHEMA, "action_semantics": ACTION_SEMANTICS,
                    "dataset_samples": len(dataset.observations),
                    "successful_laps": len(dataset.successful_trajectory_ids),
                    "observation_schema": teacher_meta.observation_schema,
                    "lookahead_count": teacher_meta.lookahead_count,
                    "teacher_overlays": list(teacher_meta.policy_overlays),
                })
                _save_student(student, config, output / "pretrained", report=report)
                (output / "pretraining-report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
            print(json.dumps(offline_validate(student, dataset), indent=2))
            if args.stage in {"validate", "full"}:
                student_runner = TrainingRunner(config)
                student_runner._ppo_air_brake_overlays = [
                    layer for layer in teacher_meta.policy_overlays
                    if layer.get("kind") == "air_brake"
                ]
                live = evaluate_model(
                    student, lambda: student_runner._environment(), episodes=5,
                    seed=args.seed + 10_000,
                )
                report_path = output / "live-validation.json"
                report_path.write_text(json.dumps(live.to_dict(), indent=2) + "\n", encoding="utf-8")
                teacher_manifest["student_live_evaluation"] = live.to_dict()
                (output / "teacher-manifest.json").write_text(
                    json.dumps(teacher_manifest, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps({"live_validation": live.to_dict()}, indent=2))
        if args.stage in {"value_warmup", "finetune", "full"}:
            if args.stage == "full":
                live = json.loads((output / "live-validation.json").read_text(encoding="utf-8"))
                finish_rate = float(live.get("finish_rate", 0.0))
                if finish_rate < args.reliability_finishes / 5:
                    print(json.dumps({
                        "reliability_gate_blocked": True,
                        "finish_rate": finish_rate,
                        "required_finish_rate": args.reliability_finishes / 5,
                        "next_stage": "dagger",
                    }, indent=2), flush=True)
                    raise SystemExit(0)
            else:
                gate_registry = ModelRegistry(args.output_root)
                gate_path = gate_registry.slot(config.track_name, "ppo", "champion")
                if not (gate_path / "metadata.json").is_file():
                    raise ValueError("PPO needs an evaluated reliable champion before RL; run DAgger first")
                gate_evaluation = gate_registry.read_metadata(gate_path).evaluation or {}
                if gate_evaluation.get("finish_rate", 0.0) < args.reliability_finishes / 5:
                    raise ValueError(
                        "PPO has not passed the finish-rate reliability gate; "
                        "run DAgger before value warmup or fine-tuning"
                    )
            student_dir = output / "pretrained"
            if args.stage in {"finetune", "full"}:
                latest = args.output_root / "summer-1" / "ppo" / "latest"
                if (latest / "metadata.json").is_file():
                    student_dir = latest
            student_path = student_dir / "policy.zip"
            if not student_path.is_file():
                raise FileNotFoundError("PPO student checkpoint missing; run collect and pretrain first")
            # Live stages are launched through TrainingRunner to preserve the normal
            # evaluation, checkpoint, rollback, and champion promotion guarantees.
            if args.stage in {"value_warmup", "full"}:
                config.timesteps = args.warmup_steps
                warmup_runner = TrainingRunner(config)
                warmup_latest = warmup_runner.run(
                    resume=output / "pretrained", freeze_ppo_actor=True,
                    allow_ppo_reward_change=args.reward_profile is not None,
                )
                student_dir = warmup_latest
            if args.stage in {"finetune", "full"}:
                rounds = args.max_rounds if args.max_rounds > 0 else None
                round_index = 0
                while rounds is None or round_index < rounds:
                    config.timesteps = args.timesteps
                    config.ppo.learning_rate = 3e-5
                    config.ppo.entropy_coefficient = 1e-4
                    config.ppo.target_kl = 0.01
                    runner = TrainingRunner(config)
                    latest = runner.run(
                        resume=student_dir, rollback_to_champion=True,
                        allow_ppo_reward_change=args.reward_profile is not None,
                    )
                    latest_meta = ModelRegistry(config.output_root).read_metadata(latest)
                    evaluation = latest_meta.evaluation or {}
                    best_lap = evaluation.get("best_lap_s")
                    if _evaluation_confirms_target(evaluation, 22.0):
                        print(json.dumps({
                            "target_reached": True, "algorithm": "ppo",
                            "lap_s": best_lap, "timesteps": latest_meta.training_timesteps,
                            "checkpoint": str(latest),
                        }, indent=2))
                        break
                    student_dir = latest
                    round_index += 1
                    print(json.dumps({
                        "target_reached": False, "round": round_index,
                        "best_lap_s": best_lap, "finish_rate": evaluation.get("finish_rate"),
                        "timesteps": latest_meta.training_timesteps,
                        "checkpoint": str(latest),
                    }, indent=2))
        if tree_sha256(teacher_path) != teacher_digest:
            raise RuntimeError("frozen TQC teacher changed during the student pipeline")
    finally:
        teacher_env = teacher.get_env()
        if teacher_env is not None:
            teacher_env.close()
        teacher_runner.sink.close() if teacher_runner.sink is not None else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
