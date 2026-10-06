"""Snapshot-safe supervised baking of proven TQC policy overlays into its actor.

This is deliberately separate from RL training and the live section optimizer:
the actor alone is distilled, all candidate artifacts remain under a run folder,
and the champion changes only after a separate validation and explicit bake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import statistics
import time
import zipfile
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch as th

from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelMetadata, ModelRegistry, git_commit
from polybot.training.config import TrainingConfig
from polybot.training.evaluation import evaluate_model
from polybot.training.promotion import promote_directory
from polybot.training.runner import TrainingRunner

BAKEABLE_KINDS = frozenset({"steer_bias", "steer_gain", "drive_bias", "drive_gain"})
SCHEDULE_KIND = "speed_bias_schedule"
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_TOLERANCE_S = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(directory: Path) -> tuple[str, str, int, int]:
    metadata = directory / "metadata.json"
    policy = directory / "policy.zip"
    replay = directory / "replay.pkl"
    return (sha256_file(metadata), sha256_file(policy), replay.stat().st_size, replay.stat().st_mtime_ns)


def _read_config(path: Path) -> TrainingConfig:
    return TrainingConfig.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


def _load_run(run_dir: Path) -> tuple[dict[str, Any], TrainingConfig]:
    teacher_info = json.loads((run_dir / "teacher.json").read_text(encoding="utf-8"))
    config = TrainingConfig.from_dict(teacher_info["training_config"])
    return teacher_info, config


def _make_env(config: TrainingConfig, *, mock: bool = False) -> PolyTrackEnv:
    runner = TrainingRunner(config)
    if mock:
        return PolyTrackEnv(
            MockSimulatorTransport(), action_adapter=runner.backend.action_adapter(config),
        )
    return runner._environment()


def _load_model(directory: Path, config: TrainingConfig, *, device: str = "cpu") -> Any:
    if config.algorithm != "tqc":
        raise ValueError("distillation currently supports TQC models only")
    runner = TrainingRunner(config)
    env = _make_env(config, mock=True)
    try:
        model = runner.backend.load_model(directory / "policy.zip", env, device, resume=True)
    finally:
        env.close()
    metadata = ModelRegistry(config.output_root).read_metadata(directory)
    model.policy_overlays = deepcopy(metadata.policy_overlays)
    model.policy.set_training_mode(False)
    return model


def _actor_action(model: Any, observation: np.ndarray) -> np.ndarray:
    tensor = th.as_tensor(observation, device=model.device).reshape(1, -1)
    with th.no_grad():
        action = _actor_forward_actions(model.actor, tensor)
    return np.asarray(action.detach().cpu().numpy()[0], dtype=np.float32)


def _actor_forward_actions(actor: Any, observations: th.Tensor) -> th.Tensor:
    """Return every action in a batch for actors with either SB3 return shape."""
    result = actor(observations, deterministic=True)
    # TQC's actor returns a tensor directly; tolerate actors returning
    # (actions, auxiliary) without accidentally selecting just batch row zero.
    return result[0] if isinstance(result, tuple) else result


def _predict_pair(model: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return raw actor action and actual post-overlay action for one state."""
    raw = _actor_action(model, observation)
    final, _ = model.predict(observation, deterministic=True)
    return raw, np.asarray(final, dtype=np.float32).reshape(-1), bool(
        getattr(model, "_air_brake_active", False)
    )


def _bakeable(metadata: ModelMetadata) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bake, keep = [], []
    for overlay in metadata.policy_overlays:
        (bake if overlay.get("kind") in BAKEABLE_KINDS else keep).append(deepcopy(overlay))
    return bake, keep


def _select_bake(
    metadata: ModelMetadata,
    teacher_info: dict[str, Any],
    bake_kinds: Iterable[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Choose a safe subset of smooth overlays to bake; retain the rest live."""
    bakeable, _ = _bakeable(metadata)
    schedule_present = bool(teacher_info.get("speed_bias_schedule"))
    available = {overlay["kind"] for overlay in bakeable}
    if schedule_present:
        available.add(SCHEDULE_KIND)
    selected = available if bake_kinds is None else set(bake_kinds)
    unknown = selected - BAKEABLE_KINDS - {SCHEDULE_KIND}
    if unknown:
        raise ValueError(f"unsupported bake kinds: {', '.join(sorted(unknown))}")
    unavailable = selected - available
    if unavailable:
        raise ValueError(f"requested bake kinds are not present: {', '.join(sorted(unavailable))}")
    bake = [overlay for overlay in bakeable if overlay["kind"] in selected]
    keep = [
        deepcopy(overlay) for overlay in metadata.policy_overlays
        if overlay.get("kind") not in selected or overlay.get("kind") not in BAKEABLE_KINDS
    ]
    return bake, keep, schedule_present and SCHEDULE_KIND in selected


def _snapshot_payload(directory: Path, metadata: ModelMetadata,
                      speed_bias_schedule: list[list[float]]) -> dict[str, Any]:
    bake, keep = _bakeable(metadata)
    bake.extend({
        "kind": "drive_bias", "start": float(start), "end": float(end),
        "amount": float(amount), "taper": 0.02, "source": "speed_bias_schedule",
    } for start, end, amount in speed_bias_schedule)
    model_config = deepcopy(metadata.training_config)
    # This snapshot is loaded offline for collection/training; the saved TQC
    # hyperparameters remain intact, while environment collection still uses the
    # configured live websocket when explicitly requested.
    return {
        "schema": "polybot.distillation-teacher.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "teacher_directory_hash": sha256_file(directory / "policy.zip"),
        "champion_id": metadata.saved_at,
        "champion_lap_s": (metadata.evaluation or {}).get("median_lap_s"),
        "overlay_stack": deepcopy(metadata.policy_overlays),
        "bakeable_overlays": bake,
        "retained_overlays": keep,
        "speed_bias_schedule": speed_bias_schedule,
        "git_commit": metadata.git_commit,
        "snapshot_git_commit": git_commit(),
        "training_config": model_config,
    }


def create_teacher_snapshot(config_path: Path, run_id: str | None = None) -> Path:
    """Atomically copy one stable champion generation into a private run folder."""
    config = _read_config(config_path)
    if config.algorithm != "tqc":
        raise ValueError("distillation requires a TQC configuration")
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion", track_slug=config.track_slug)
    metadata = registry.read_metadata(champion)
    if not metadata.evaluation or metadata.evaluation.get("median_lap_s") is None:
        raise ValueError("distillation requires a fully evaluated champion")
    stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root = (
        registry.algorithm_dir(config.track_name, "tqc", track_slug=config.track_slug)
        / "distillation" / stamp
    )
    if root.exists():
        raise FileExistsError(f"distillation run already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.with_name(f".{root.name}.snapshot-{uuid4().hex}")
    try:
        for attempt in range(4):
            shutil.rmtree(stage, ignore_errors=True)
            try:
                before = _fingerprint(champion)
                current_metadata = registry.read_metadata(champion)
                shutil.copytree(champion, stage)
                after = _fingerprint(champion)
                copied = _fingerprint(stage)
            except (FileNotFoundError, PermissionError):
                if attempt == 3:
                    raise
                time.sleep(0.1 * (attempt + 1))
                continue
            if before != after or copied != before:
                if attempt == 3:
                    raise RuntimeError("champion changed repeatedly while snapshotting")
                time.sleep(0.1 * (attempt + 1))
                continue
            with zipfile.ZipFile(stage / "policy.zip") as archive:
                archive_data = json.loads(archive.read("data").decode("utf-8"))
            schedule = [list(map(float, item)) for item in archive_data.get("speed_bias_schedule", ())]
            payload = _snapshot_payload(stage, current_metadata, schedule)
            payload.update(run_id=stamp, champion_hash=copied[1])
            (stage / "teacher.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            for rename_attempt in range(8):
                try:
                    if root.exists():
                        raise FileExistsError(f"distillation run already exists: {root}")
                    stage.rename(root)
                    break
                except PermissionError:
                    if rename_attempt == 7:
                        raise
                    time.sleep(0.25 * (rename_attempt + 1))
            return root
        raise RuntimeError("could not obtain a stable champion snapshot")
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _progress(observation: np.ndarray) -> float:
    return float(np.asarray(observation).reshape(-1)[12])


def _overlay_features(progress: float, overlays: Iterable[dict[str, Any]]) -> tuple[bool, bool]:
    bakeable_active = False
    boundary = False
    for overlay in overlays:
        start, end = float(overlay.get("start", 0)), float(overlay.get("end", 0))
        if overlay.get("kind") in BAKEABLE_KINDS and start <= progress <= end:
            bakeable_active = True
            boundary = boundary or min(abs(progress - start), abs(progress - end)) <= 0.01
    return bakeable_active, boundary


def collect_teacher_data(
    run_dir: Path, *, episodes: int = 20, seed: int | None = None,
    perturbation_probability: float = 0.0, steering_perturbation_std: float = 0.005,
    longitudinal_perturbation_std: float = 0.015,
) -> dict[str, Any]:
    if episodes < 2:
        raise ValueError("collect at least two full laps to keep lap-grouped validation possible")
    if not 0.0 <= perturbation_probability <= 1.0:
        raise ValueError("perturbation probability must be between zero and one")
    if min(steering_perturbation_std, longitudinal_perturbation_std) < 0:
        raise ValueError("perturbation standard deviations must be nonnegative")
    clean_episode_count = episodes if perturbation_probability == 0 else episodes // 2
    if clean_episode_count < 2:
        raise ValueError("collect at least two unperturbed full laps")
    if simulator_service_active():
        raise RuntimeError("teacher collection cannot share the simulator with an active training/search process")
    run_dir = run_dir.resolve()
    teacher_info, config = _load_run(run_dir)
    snapshot = run_dir
    model = _load_model(snapshot, config)
    overlays = deepcopy(teacher_info["bakeable_overlays"])
    progress_rows: list[float] = []
    speed_rows: list[float] = []
    contacts_rows: list[np.ndarray] = []
    observation_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    delta_rows: list[float] = []
    overlay_rows: list[bool] = []
    boundary_rows: list[bool] = []
    airbrake_rows: list[bool] = []
    airborne_rows: list[bool] = []
    active_overlay_rows: list[list[bool]] = []
    perturbed_rows: list[bool] = []
    section_rows: list[int] = []
    lap_rows: list[int] = []
    runner = TrainingRunner(config)
    env = runner._environment()
    actual_seed = config.seed + 8_000_000 if seed is None else int(seed)
    finished = 0
    clean_finished = 0
    perturbed_finished = 0
    discarded_perturbed_episodes = 0
    episode_arrays = (
        observation_rows, raw_rows, target_rows, progress_rows, contacts_rows, speed_rows,
        delta_rows, overlay_rows, boundary_rows, airbrake_rows, airborne_rows,
        active_overlay_rows, perturbed_rows, section_rows, lap_rows,
    )
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=actual_seed + episode)
            episode_rng = np.random.default_rng(actual_seed + episode)
            perturb_episode = episode >= episodes // 2 and perturbation_probability > 0
            episode_offsets = [len(rows) for rows in episode_arrays]
            terminated = truncated = False
            while not (terminated or truncated):
                obs = np.asarray(observation, dtype=np.float32).reshape(-1)
                raw, target, airbrake = _predict_pair(model, obs)
                progress = _progress(obs)
                bakeable_active, boundary = _overlay_features(progress, overlays)
                active_overlay_rows.append([
                    overlay.get("kind") in BAKEABLE_KINDS
                    and float(overlay.get("start", 0.0)) <= progress <= float(overlay.get("end", 0.0))
                    for overlay in overlays
                ])
                contacts = obs[17:21].copy()
                velocity = np.asarray(info.get("local_velocity_mps", (0, 0, 0)), dtype=np.float32)
                observation_rows.append(obs.copy())
                raw_rows.append(raw)
                target_rows.append(target)
                progress_rows.append(progress)
                contacts_rows.append(contacts)
                speed_rows.append(float(np.linalg.norm(velocity)))
                delta_rows.append(float(np.max(np.abs(target - raw))))
                overlay_rows.append(bakeable_active)
                boundary_rows.append(boundary)
                airbrake_rows.append(airbrake)
                airborne_rows.append(bool(np.all(contacts < 0.5)))
                perturb_action = perturb_episode and episode_rng.random() < perturbation_probability
                perturbed_rows.append(perturb_action)
                applied_action = target.copy()
                if perturb_action:
                    noise = episode_rng.normal(
                        0.0, (steering_perturbation_std, longitudinal_perturbation_std), size=2,
                    ).astype(np.float32)
                    if airbrake:
                        noise[1] = 0.0
                    applied_action = np.clip(applied_action + noise, -1.0, 1.0)
                section_rows.append(min(9, max(0, int(progress * 10))))
                lap_rows.append(episode)
                if airbrake:
                    env._air_brake_request = True
                    env._air_brake_base_action = model._air_brake_base_action
                observation, _, terminated, truncated, info = env.step(applied_action)
            if "finish" in set(info.get("events", ())):
                finished += 1
                if perturb_episode:
                    perturbed_finished += 1
                else:
                    clean_finished += 1
            elif perturb_episode:
                # Keep only complete perturbed trajectories. Partial states
                # preceding a crash teach an unsafe failure path, not recovery.
                for rows, offset in zip(episode_arrays, episode_offsets, strict=True):
                    del rows[offset:]
                discarded_perturbed_episodes += 1
            print(json.dumps({
                "type": "distillation_progress",
                "message": f"Teacher data collection: {episode + 1}/{episodes} episodes; "
                           f"{finished} finished laps, {len(progress_rows):,} samples.",
            }), flush=True)
        if clean_finished != clean_episode_count:
            raise RuntimeError(
                f"teacher snapshot finished only {clean_finished}/{clean_episode_count} unperturbed laps"
            )
    finally:
        env.close()
    arrays = {
        "observations": np.stack(observation_rows).astype(np.float32),
        "raw_actions": np.stack(raw_rows).astype(np.float32),
        "target_actions": np.stack(target_rows).astype(np.float32),
        "progress": np.asarray(progress_rows, dtype=np.float32),
        "speed_mps": np.asarray(speed_rows, dtype=np.float32),
        "wheel_contacts": np.stack(contacts_rows).astype(np.float32),
        "action_delta": np.asarray(delta_rows, dtype=np.float32),
        "bakeable_overlay_active": np.asarray(overlay_rows, dtype=np.bool_),
        "overlay_boundary": np.asarray(boundary_rows, dtype=np.bool_),
        "air_brake_active": np.asarray(airbrake_rows, dtype=np.bool_),
        "airborne": np.asarray(airborne_rows, dtype=np.bool_),
        "active_overlay_mask": np.asarray(active_overlay_rows, dtype=np.bool_),
        "perturbed": np.asarray(perturbed_rows, dtype=np.bool_),
        "section": np.asarray(section_rows, dtype=np.int8),
        "lap_id": np.asarray(lap_rows, dtype=np.int32),
    }
    dataset_path = run_dir / "dataset.npz"
    tmp = dataset_path.with_name(f".{dataset_path.name}.{uuid4().hex}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(dataset_path)
    weights = sample_weights(arrays)
    dataset_info = {
        "schema": "polybot.distillation-dataset.v1", "created_at": datetime.now(UTC).isoformat(),
        "episodes": episodes, "finished_episodes": finished, "samples": len(progress_rows),
        "finished_clean_episodes": clean_finished,
        "finished_perturbed_episodes": perturbed_finished,
        "discarded_perturbed_episodes": discarded_perturbed_episodes,
        "seed": actual_seed, "dataset_sha256": sha256_file(dataset_path),
        "teacher_hash": teacher_info["champion_hash"], "git_commit": git_commit(),
        "bakeable_overlay_names": [
            f"{index}:{overlay.get('kind')}:{overlay.get('start', 0)}-{overlay.get('end', 1)}"
            for index, overlay in enumerate(overlays)
        ],
        "overlay_modified_samples": int(np.count_nonzero(arrays["action_delta"] > 1e-6)),
        "perturbed_samples": int(np.count_nonzero(arrays["perturbed"])),
        "perturbation": {
            "episodes": list(range(episodes // 2, episodes)) if perturbation_probability > 0 else [],
            "probability": perturbation_probability,
            "steering_std": steering_perturbation_std,
            "longitudinal_std": longitudinal_perturbation_std,
        },
        "weighted_samples_total": float(np.sum(weights)),
    }
    _write_json_atomic(run_dir / "dataset.json", dataset_info)
    return dataset_info


def sample_weights(data: dict[str, np.ndarray], *, overlay_weight: float = 8.0,
                   boundary_weight: float = 2.0, airborne_weight: float = 2.0,
                   delta_scale: float = 0.05) -> np.ndarray:
    """Prioritize changed actions, taper boundaries, and airborne transitions."""
    if min(overlay_weight, boundary_weight, airborne_weight, delta_scale) < 0 or delta_scale == 0:
        raise ValueError("sample weighting parameters must be nonnegative with a positive delta scale")
    changed = np.clip(np.asarray(data["action_delta"], dtype=np.float32) / delta_scale, 0.0, 1.0)
    weights = 1.0 + overlay_weight * changed
    weights += boundary_weight * np.asarray(data["overlay_boundary"], dtype=np.float32)
    weights += airborne_weight * np.asarray(data["air_brake_active"], dtype=np.float32)
    if "perturbed" in data:
        weights += 2.0 * np.asarray(data["perturbed"], dtype=np.float32)
    return weights.astype(np.float64)


def _split_by_lap(lap_ids: np.ndarray, seed: int, validation_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    groups = np.unique(lap_ids)
    if len(groups) < 2:
        raise ValueError("at least two laps are required for a lap-grouped validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * validation_fraction)))
    validation_groups = set(groups[:validation_count].tolist())
    validation = np.flatnonzero(np.isin(lap_ids, list(validation_groups)))
    training = np.flatnonzero(~np.isin(lap_ids, list(validation_groups)))
    return training, validation


def _same_state(left: Any, right: Any) -> bool:
    if isinstance(left, th.Tensor):
        return isinstance(right, th.Tensor) and th.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def _action_metrics(model: Any, observations: np.ndarray, targets: np.ndarray,
                    sections: np.ndarray, overlay_active: np.ndarray) -> dict[str, Any]:
    predictions = []
    model.policy.set_training_mode(False)
    for observation in observations:
        action, _ = model.predict(observation, deterministic=True)
        predictions.append(np.asarray(action, dtype=np.float32).reshape(-1))
    error = np.abs(np.stack(predictions) - targets)
    result: dict[str, Any] = {
        "samples": int(len(error)),
        "mean_steering_error": float(error[:, 0].mean()),
        "max_steering_error": float(error[:, 0].max()),
        "mean_longitudinal_error": float(error[:, 1].mean()),
        "max_longitudinal_error": float(error[:, 1].max()),
        "mean_action_error": float(error.mean()),
        "max_action_error": float(error.max()),
        "overlay_mean_action_error": float(error[overlay_active].mean()) if np.any(overlay_active) else 0.0,
        "overlay_samples": int(np.count_nonzero(overlay_active)),
        "by_section": {},
    }
    for section in np.unique(sections):
        selected = sections == section
        result["by_section"][str(int(section))] = {
            "samples": int(selected.sum()), "mean_action_error": float(error[selected].mean()),
            "max_action_error": float(error[selected].max()),
        }
    return result


def train_student(run_dir: Path, *, epochs: int = DEFAULT_EPOCHS, learning_rate: float = DEFAULT_LR,
                  patience: int = DEFAULT_PATIENCE, batch_size: int = 512, seed: int = 20260929,
                  steering_weight: float = 1.0, longitudinal_weight: float = 1.0,
                  overlay_weight: float = 8.0, boundary_weight: float = 2.0,
                  bake_kinds: Iterable[str] | None = None) -> dict[str, Any]:
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, patience, batch size and learning rate must be positive")
    if min(steering_weight, longitudinal_weight, overlay_weight, boundary_weight) <= 0:
        raise ValueError("distillation loss and overlay sample weights must be positive")
    run_dir = run_dir.resolve()
    bake_kinds = None if bake_kinds is None else tuple(bake_kinds)
    teacher_info, config = _load_run(run_dir)
    dataset_path = run_dir / "dataset.npz"
    manifest = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("distillation dataset hash does not match its manifest")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    observations = data["observations"]
    targets = data["target_actions"]
    if observations.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("distillation expects flat observations and two-dimensional TQC actions")
    train_indices, validation_indices = _split_by_lap(data["lap_id"], seed)
    weights = sample_weights(data, overlay_weight=overlay_weight, boundary_weight=boundary_weight)
    runner = TrainingRunner(config)
    student = _load_model(run_dir, config)
    bakeable, retained, bake_schedule = _select_bake(
        runner.registry.read_metadata(run_dir), teacher_info, bake_kinds,
    )
    if not bakeable and not bake_schedule:
        raise ValueError("teacher has no bakeable policy overlays or speed schedule")
    student.policy_overlays = retained
    if bake_schedule:
        student.speed_bias_schedule = []
    actor = student.actor
    critic_before = [p.detach().cpu().clone() for p in student.critic.parameters()]
    targets_critic_before = [p.detach().cpu().clone() for p in student.critic_target.parameters()]
    entropy_before = {
        "ent_coef": deepcopy(student.ent_coef),
        "log_ent_coef": deepcopy(student.log_ent_coef.detach().cpu())
        if isinstance(getattr(student, "log_ent_coef", None), th.Tensor) else None,
        "optimizer": deepcopy(student.ent_coef_optimizer.state_dict())
        if getattr(student, "ent_coef_optimizer", None) is not None else None,
    }
    actor_optimizer_before = deepcopy(student.actor.optimizer.state_dict())
    optimizer = th.optim.Adam(actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    observations_t = th.as_tensor(observations, device=student.device, dtype=th.float32)
    targets_t = th.as_tensor(targets, device=student.device, dtype=th.float32)
    action_weights = th.as_tensor([steering_weight, longitudinal_weight], device=student.device, dtype=th.float32)
    # Preserve the low-level landing-aware air-brake controller. It owns the
    # longitudinal action on airborne samples and is not an actor target.
    air_mask = th.as_tensor(data["air_brake_active"], device=student.device, dtype=th.bool)
    sample_mask = th.ones_like(targets_t)
    sample_mask[air_mask, 1] = 0.0
    train_weights = weights[train_indices].astype(np.float64)
    train_weights /= train_weights.sum()
    actor.eval()
    with th.no_grad():
        initial_prediction = _actor_forward_actions(actor, observations_t[validation_indices])
        initial_diff = initial_prediction - targets_t[validation_indices]
        initial_mask = sample_mask[validation_indices]
        initial_loss = float(
            ((initial_diff.square() * initial_mask * action_weights).sum(dim=1)
             / initial_mask.sum(dim=1).clamp_min(1.0)).mean().cpu()
        )
    best_loss = initial_loss
    best_epoch = 0
    stale = 0
    best_state: dict[str, th.Tensor] | None = {
        key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
    }
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        actor.train()
        batch_count = max(1, int(np.ceil(len(train_indices) / batch_size)))
        epoch_losses = []
        for _ in range(batch_count):
            indices = rng.choice(train_indices, size=min(batch_size, len(train_indices)),
                                 replace=len(train_indices) < batch_size, p=train_weights)
            predicted = _actor_forward_actions(actor, observations_t[indices])
            diff = predicted - targets_t[indices]
            mask = sample_mask[indices]
            per_action = th.square(diff) * mask * action_weights
            per_sample = per_action.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            loss = per_sample.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        actor.eval()
        with th.no_grad():
            predicted = _actor_forward_actions(actor, observations_t[validation_indices])
            diff = predicted - targets_t[validation_indices]
            mask = sample_mask[validation_indices]
            validation_loss = float(
                ((diff.square() * mask * action_weights).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).mean()
                .detach().cpu()
            )
        record = {"epoch": epoch, "train_loss": statistics.fmean(epoch_losses),
                  "validation_loss": validation_loss}
        history.append(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({
                "type": "distillation_progress",
                "message": f"Actor distillation: epoch {epoch}, validation MSE {validation_loss:.6g}.",
            }), flush=True)
        if validation_loss < best_loss - 1e-9:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("distillation did not produce a finite validation checkpoint")
    actor.load_state_dict(best_state)
    if any(
        not th.equal(before, after.detach().cpu())
        for before, after in zip(critic_before, student.critic.parameters(), strict=True)
    ):
        raise RuntimeError("distillation unexpectedly changed critic weights")
    if any(
        not th.equal(before, after.detach().cpu())
        for before, after in zip(targets_critic_before, student.critic_target.parameters(), strict=True)
    ):
        raise RuntimeError("distillation unexpectedly changed target critics")
    entropy_after = {
        "ent_coef": student.ent_coef,
        "log_ent_coef": student.log_ent_coef.detach().cpu()
        if isinstance(getattr(student, "log_ent_coef", None), th.Tensor) else None,
        "optimizer": student.ent_coef_optimizer.state_dict()
        if getattr(student, "ent_coef_optimizer", None) is not None else None,
    }
    if not _same_state(entropy_before, entropy_after):
        raise RuntimeError("distillation unexpectedly changed entropy state")
    if not _same_state(actor_optimizer_before, student.actor.optimizer.state_dict()):
        raise RuntimeError("distillation unexpectedly changed the RL actor optimizer state")
    action_metrics = _action_metrics(student, observations, targets, data["section"], data["bakeable_overlay_active"])
    student_dir = run_dir / "student"
    staging = run_dir / f".student-staging-{uuid4().hex}"
    runner.backend.save_model(student, staging, resume=True)
    teacher_metadata = runner.registry.read_metadata(run_dir)
    staged_metadata = replace(
        teacher_metadata, evaluation=None, policy_overlays=retained,
        critic_adaptation_required=True, adaptation_stage="distilled_unvalidated",
        saved_at=datetime.now(UTC).isoformat(), git_commit=git_commit(),
    )
    runner.registry.write_metadata(staging, staged_metadata)
    if student_dir.exists():
        shutil.rmtree(student_dir)
    for attempt in range(8):
        try:
            staging.replace(student_dir)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))
    report = {
        "schema": "polybot.distillation-training.v1", "created_at": datetime.now(UTC).isoformat(),
        "epochs_run": len(history), "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "initial_validation_loss": initial_loss,
        "history": history, "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)), "teacher_hash": teacher_info["champion_hash"],
        "bakeable_overlay_count": len(bakeable), "retained_overlay_count": len(retained),
        "speed_bias_schedule_baked": bake_schedule,
        "selected_bake_kinds": sorted(set(bake_kinds) if bake_kinds is not None else
                                       ({row["kind"] for row in bakeable} |
                                        ({SCHEDULE_KIND} if bake_schedule else set()))),
        "learning_rate": learning_rate, "seed": seed, "action_metrics": action_metrics,
        "sample_weights": {"overlay": overlay_weight, "boundary": boundary_weight},
        "git_commit": git_commit(), "actor_only": True,
    }
    _write_json_atomic(run_dir / "training.json", report)
    return report


def validate_student(run_dir: Path, *, episodes: int = 5, tolerance_s: float = DEFAULT_TOLERANCE_S) -> dict[str, Any]:
    if episodes < 5 or tolerance_s < 0:
        raise ValueError("live validation requires at least five episodes and nonnegative tolerance")
    if simulator_service_active():
        raise RuntimeError("live validation cannot share the simulator with an active training/search process")
    run_dir = run_dir.resolve()
    teacher_info, config = _load_run(run_dir)
    teacher = _load_model(run_dir, config)
    student = _load_model(run_dir / "student", config)
    runner = TrainingRunner(config)
    teacher_telemetry: list[list[dict[str, Any]]] = []
    student_telemetry: list[list[dict[str, Any]]] = []
    teacher_result = evaluate_model(teacher, runner._environment, episodes=episodes,
                                    seed=config.seed + 9_000_000, telemetry_sink=teacher_telemetry)
    student_result = evaluate_model(student, runner._environment, episodes=episodes,
                                     seed=config.seed + 9_000_000, reference_model=teacher,
                                     telemetry_sink=student_telemetry)
    teacher_roll_failures = sum(
        "airborne_roll_failure" in sample.get("events", ())
        for episode in teacher_telemetry for sample in episode
    )
    student_roll_failures = sum(
        "airborne_roll_failure" in sample.get("events", ())
        for episode in student_telemetry for sample in episode
    )
    delta = None
    if teacher_result.median_lap_s is not None and student_result.median_lap_s is not None:
        delta = float(student_result.median_lap_s - teacher_result.median_lap_s)
    accepted = (
        teacher_result.finish_rate == 1.0 and student_result.finish_rate == 1.0
        and student_result.median_progress == 1.0 and student_result.crash_rate == 0.0
        and student_result.off_track_rate == 0.0 and student_result.stall_rate == 0.0
        and student_roll_failures == 0
        and delta is not None and delta <= tolerance_s
    )
    report = {
        "schema": "polybot.distillation-validation.v1", "validated_at": datetime.now(UTC).isoformat(),
        "teacher_hash": teacher_info["champion_hash"], "episodes": episodes,
        "tolerance_s": tolerance_s, "teacher": teacher_result.to_dict(),
        "student": student_result.to_dict(), "lap_delta_s": delta,
        "airborne_roll_failure_events": {
            "teacher": teacher_roll_failures, "student": student_roll_failures,
        },
        "action_comparison": {
            "mean_steering_error": student_result.mean_steering_disagreement,
            "max_steering_error": student_result.max_steering_disagreement,
            "mean_longitudinal_error": student_result.mean_longitudinal_disagreement,
            "max_longitudinal_error": student_result.max_longitudinal_disagreement,
            "max_position_deviation_m": student_result.max_position_deviation_m,
            "max_heading_deviation_rad": student_result.max_heading_deviation_rad,
        },
        "accepted": accepted,
    }
    _write_json_atomic(run_dir / "validation.json", report)
    return report


def simulator_service_active(host: str = "127.0.0.1", port: int = 8765) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def bake_student(run_dir: Path, *, tolerance_s: float = DEFAULT_TOLERANCE_S) -> Path:
    """Promote only a passing staged student whose source remains champion."""
    if simulator_service_active():
        raise RuntimeError("cannot bake/promote while the live simulator service is active")
    run_dir = run_dir.resolve()
    teacher_info, config = _load_run(run_dir)
    validation = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("accepted") or float(validation.get("tolerance_s", tolerance_s)) > tolerance_s:
        raise ValueError("distilled student has not passed the requested live validation gate")
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion", track_slug=config.track_slug)
    current_hash = sha256_file(champion / "policy.zip")
    if current_hash != teacher_info["champion_hash"]:
        raise RuntimeError("champion changed after snapshot; refusing to replace a newer champion")
    student_metadata = registry.read_metadata(run_dir / "student")
    student_metadata = replace(
        student_metadata, evaluation=validation["student"], critic_adaptation_required=True,
        adaptation_stage="critic_adaptation_required", saved_at=datetime.now(UTC).isoformat(),
        git_commit=git_commit(),
    )
    staging = champion.parent / f".{champion.name}-distillation-staging-{uuid4().hex}"
    shutil.copytree(run_dir / "student", staging)
    registry.write_metadata(staging, student_metadata)
    promotion_record = {
        "status": "prepared", "prepared_at": datetime.now(UTC).isoformat(),
        "source_teacher_hash": current_hash,
        "promoted_policy_hash": sha256_file(staging / "policy.zip"),
        "student_validation": validation, "retained_overlays": student_metadata.policy_overlays,
        "critic_adaptation_required": True,
    }
    _write_json_atomic(run_dir / "promotion.json", promotion_record)
    try:
        if simulator_service_active() or sha256_file(champion / "policy.zip") != teacher_info["champion_hash"]:
            raise RuntimeError("simulator or champion changed before atomic distillation promotion")
        promote_directory(staging, champion, require_replay=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    promotion_record.update(status="promoted", promoted_at=datetime.now(UTC).isoformat())
    _write_json_atomic(run_dir / "promotion.json", promotion_record)
    return champion


def rollback_student(run_dir: Path) -> Path:
    """Restore the exact snapshotted teacher only if the baked student is still current."""
    if simulator_service_active():
        raise RuntimeError("cannot roll back while the live simulator service is active")
    run_dir = run_dir.resolve()
    teacher_info, config = _load_run(run_dir)
    promotion_path = run_dir / "promotion.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion", track_slug=config.track_slug)
    current_hash = sha256_file(champion / "policy.zip")
    if current_hash != promotion.get("promoted_policy_hash"):
        raise RuntimeError("champion changed after distillation; refusing to overwrite newer work")
    if current_hash == teacher_info["champion_hash"]:
        raise ValueError("the snapshotted teacher is already champion")
    staging = champion.parent / f".{champion.name}-distillation-rollback-{uuid4().hex}"
    staging.mkdir(parents=True)
    control_names = {
        "teacher.json", "dataset.npz", "dataset.json", "training.json", "validation.json",
        "promotion.json", "rollback.json", "student",
    }
    try:
        for item in run_dir.iterdir():
            if item.name in control_names:
                continue
            destination = staging / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            elif item.is_file():
                shutil.copy2(item, destination)
        restored_metadata = registry.read_metadata(staging)
        if sha256_file(staging / "policy.zip") != teacher_info["champion_hash"]:
            raise RuntimeError("snapshotted teacher policy hash does not match the rollback record")
        registry.write_metadata(staging, restored_metadata)
        if simulator_service_active() or sha256_file(champion / "policy.zip") != current_hash:
            raise RuntimeError("simulator or champion changed before rollback promotion")
        promote_directory(staging, champion, require_replay=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _write_json_atomic(run_dir / "rollback.json", {
        "rolled_back_at": datetime.now(UTC).isoformat(), "restored_policy_hash": teacher_info["champion_hash"],
        "replaced_policy_hash": current_hash,
    })
    return champion


def run_full_workflow(
    config_path: Path, *, episodes: int = 20, validation_episodes: int = 5,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
) -> dict[str, Any]:
    """Run all offline/live checks and leave any passing candidate staged.

    Promotion is intentionally a separate CLI/GUI action so the operator can
    inspect the exact validation report before replacing a champion.
    """
    if episodes < 2 or validation_episodes < 5 or tolerance_s < 0:
        raise ValueError(
            "full workflow needs at least two collection laps, five validation laps, "
            "and a nonnegative tolerance"
        )
    directory = create_teacher_snapshot(config_path)
    collected = collect_teacher_data(directory, episodes=episodes)
    trained = train_student(directory)
    validation = validate_student(
        directory, episodes=validation_episodes, tolerance_s=tolerance_s,
    )
    return {
        "run_dir": str(directory), "collection": collected, "training": trained,
        "validation": validation, "ready_to_bake": bool(validation["accepted"]),
        "promotion": None,
    }


def _cmd_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    directory = create_teacher_snapshot(args.config, args.run_id)
    return {"type": "distillation_snapshot_created", "run_dir": str(directory),
            "teacher": json.loads((directory / "teacher.json").read_text(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="snapshot the current evaluated champion")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--run-id")
    collect = sub.add_parser("collect", help="collect final teacher actions on live deterministic laps")
    collect.add_argument("--run-dir", type=Path, required=True)
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--seed", type=int)
    collect.add_argument("--perturbation-probability", type=float, default=0.0)
    collect.add_argument("--steering-perturbation-std", type=float, default=0.005)
    collect.add_argument("--longitudinal-perturbation-std", type=float, default=0.015)
    train = sub.add_parser("train", help="train an actor-only supervised student")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--seed", type=int, default=20260929)
    train.add_argument("--bake-kinds", nargs="+", choices=sorted(BAKEABLE_KINDS | {SCHEDULE_KIND}))
    validate = sub.add_parser("validate", help="compare teacher and student on live laps")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--episodes", type=int, default=5)
    validate.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    bake = sub.add_parser("bake", help="promote a validated staged student")
    bake.add_argument("--run-dir", type=Path, required=True)
    bake.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    rollback = sub.add_parser("rollback", help="restore a baked run's snapshotted teacher")
    rollback.add_argument("--run-dir", type=Path, required=True)
    full = sub.add_parser("full", help="snapshot, collect, train, validate, and promote")
    full.add_argument("--config", type=Path, required=True)
    full.add_argument("--episodes", type=int, default=20)
    full.add_argument("--validation-episodes", type=int, default=5)
    full.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_S)
    args = parser.parse_args()
    if args.command == "snapshot":
        result = _cmd_snapshot(args)
    elif args.command == "collect":
        result = collect_teacher_data(
            args.run_dir, episodes=args.episodes, seed=args.seed,
            perturbation_probability=args.perturbation_probability,
            steering_perturbation_std=args.steering_perturbation_std,
            longitudinal_perturbation_std=args.longitudinal_perturbation_std,
        )
    elif args.command == "train":
        result = train_student(args.run_dir, epochs=args.epochs, learning_rate=args.learning_rate,
                               patience=args.patience, batch_size=args.batch_size, seed=args.seed,
                               bake_kinds=args.bake_kinds)
    elif args.command == "validate":
        result = validate_student(args.run_dir, episodes=args.episodes, tolerance_s=args.tolerance)
    elif args.command == "bake":
        result = {"champion": str(bake_student(args.run_dir, tolerance_s=args.tolerance))}
    elif args.command == "rollback":
        result = {"champion": str(rollback_student(args.run_dir))}
    else:
        result = run_full_workflow(
            args.config, episodes=args.episodes,
            validation_episodes=args.validation_episodes, tolerance_s=args.tolerance,
        )
    print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
