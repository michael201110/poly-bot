"""Read-only diagnosis of the frozen TQC teacher and directly distilled PPO actor.

No optimizer is constructed or run here. Live mode needs the game worker connected
to the websocket listener, exactly like an ordinary evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from polybot.algorithms.registry import backend_for
from polybot.environment.env import AirBrakeActionWrapper
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig
from polybot.training.runner import TrainingRunner
from polybot.training.teacher_student import (
    TeacherDataset,
    _deterministic_mean,
    _high_level_teacher_action,
)

ACTION_THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.1)
POSITION_THRESHOLDS_M = (0.01, 0.05, 0.1, 0.5, 1.0)
HEADING_THRESHOLDS_DEG = (0.1, 0.5, 1.0, 5.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _error_summary(error: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(error, dtype=np.float64).reshape(-1)
    return {
        "samples": int(values.size),
        "mae": float(np.mean(np.abs(values))) if values.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "max": float(np.max(np.abs(values))) if values.size else 0.0,
        **{
            f"p{percentile}": float(np.percentile(np.abs(values), percentile)) if values.size else 0.0
            for percentile in (50, 90, 95, 99)
        },
    }


def compare_same_observations(
    teacher: Any, student: Any, dataset: TeacherDataset,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare both actors on precisely the same successful teacher states."""
    mask = np.isin(dataset.trajectory_ids, dataset.successful_trajectory_ids)
    obs = dataset.observations[mask]
    target = dataset.final_actions[mask]
    student_actions = []
    raw_actions = []
    for start in range(0, len(obs), 512):
        batch = obs[start:start + 512]
        raw, _ = teacher.policy.predict(batch, deterministic=True)
        predicted, _ = student.predict(batch, deterministic=True)
        raw_actions.append(np.asarray(raw, dtype=np.float32))
        student_actions.append(np.asarray(predicted, dtype=np.float32))
    raw = np.concatenate(raw_actions)
    predicted = np.concatenate(student_actions)
    # Air-brake overrides are intentionally below the distilled actor. Compare
    # student against the bakeable teacher target, and audit that target itself.
    calculated = np.stack([
        _high_level_teacher_action(teacher, observation, action)
        for observation, action in zip(obs, raw, strict=True)
    ])
    errors = predicted - target
    progress = dataset.progress[mask]
    sections = {}
    for index in range(20):
        selected = (progress >= index * 0.05) & (progress < (index + 1) * 0.05)
        if np.any(selected):
            sections[f"{index * 5:02d}-{(index + 1) * 5:02d}%"] = {
                "steering": _error_summary(errors[selected, 0]),
                "longitudinal": _error_summary(errors[selected, 1]),
            }
    contacts = dataset.wheel_contacts[mask]
    speed = dataset.speed_mps[mask]
    regimes = {
        "high_speed": speed >= 45.0,
        "airborne": np.all(contacts < 0.5, axis=1),
        "overlay": dataset.overlay_active[mask].astype(bool),
        "jump_approach": (progress >= 0.20) & (progress < 0.35),
        "air_brake_region": (progress >= 0.688575) & (progress <= 0.8125),
    }
    report = {
        "teacher_samples": int(len(obs)),
        "observation_shape": list(obs.shape),
        "observation_dtype": str(obs.dtype),
        "steering": _error_summary(errors[:, 0]),
        "longitudinal": _error_summary(errors[:, 1]),
        "teacher_target_max_difference": float(np.max(np.abs(calculated - target))),
        "student_deterministic_repeat_max_difference": float(
            np.max(np.abs(predicted[:min(512, len(obs))] - np.asarray(
                student.predict(obs[:min(512, len(obs))], deterministic=True)[0]
            )))
        ),
        "student_predict_vs_clipped_mean_max_difference": float(
            np.max(np.abs(predicted - _deterministic_mean(student, obs)))
        ),
        "sections": sections,
        "regimes": {
            name: {
                "steering": _error_summary(errors[selected, 0]),
                "longitudinal": _error_summary(errors[selected, 1]),
            }
            for name, selected in regimes.items()
        },
    }
    return report, {
        "observations": obs, "teacher_raw": raw,
        "teacher_targets": target, "student": predicted, "progress": progress,
    }


def first_divergences(teacher: list[dict[str, Any]], student: list[dict[str, Any]]) -> dict[str, Any]:
    """Find first decision-aligned differences; missing samples remain unknown."""
    pairs = list(zip(teacher, student, strict=False))
    result: dict[str, Any] = {"compared_steps": len(pairs)}
    for field, thresholds in (
        ("action", ACTION_THRESHOLDS),
        ("position_m", POSITION_THRESHOLDS_M),
        ("heading_deg", HEADING_THRESHOLDS_DEG),
    ):
        result[field] = {}
        for threshold in thresholds:
            first = next((
                index for index, (left, right) in enumerate(pairs)
                if field in left and field in right
                and float(
                    np.linalg.norm(np.asarray(left[field]) - np.asarray(right[field]))
                    if field == "position_m"
                    else np.max(np.abs(np.asarray(left[field]) - np.asarray(right[field])))
                ) > threshold
            ), None)
            result[field][str(threshold)] = first
    for name in ("wheel_contacts", "speed_mps", "steering", "longitudinal"):
        tolerance = 0.0 if name == "wheel_contacts" else 0.01
        result[name] = next((
            index for index, (left, right) in enumerate(pairs)
            if name in left and name in right
            and float(np.max(np.abs(np.asarray(left[name]) - np.asarray(right[name])))) > tolerance
        ), None)
    return result


def _record(index: int, observation: np.ndarray, action: np.ndarray, info: dict[str, Any],
            *, raw: np.ndarray | None = None, comparison: np.ndarray | None = None,
            air_brake_base: np.ndarray | None = None) -> dict[str, Any]:
    velocity = np.asarray(info.get("local_velocity_mps", (0, 0, 0)), dtype=float)
    heading = float(info.get("heading_error_rad", 0.0))
    return {
        "step": index, "tick": int(info.get("tick", 0)),
        "ticks_advanced": int(info.get("ticks_advanced", 0)),
        "elapsed_s": float(info.get("elapsed_s", 0.0)),
        "progress": float(observation[12]),
        "position_m": list(info.get("position_m", ())),
        "velocity_mps": velocity.tolist(), "speed_mps": float(np.linalg.norm(velocity)),
        "heading_deg": math.degrees(heading),
        "angular_velocity_radps": list(info.get("angular_velocity_radps", ())),
        "wheel_contacts": list(info.get("wheel_contacts", ())),
        "airborne": all(float(value) < 0.5 for value in info.get("wheel_contacts", ())),
        "observation": observation.tolist(),
        "action": np.asarray(action).tolist(),
        "raw_actor_action": None if raw is None else np.asarray(raw).tolist(),
        "air_brake_base_action": None if air_brake_base is None else np.asarray(air_brake_base).tolist(),
        "other_policy_same_observation": None if comparison is None else np.asarray(comparison).tolist(),
        "steering": float(action[0]), "longitudinal": float(action[1]),
        "requested_control_duty": info.get("requested_control_duty"),
        "applied_control_fraction": info.get("applied_control_fraction"),
        "executed_tick_controls": info.get("executed_tick_controls"),
        "events": list(info.get("events", ())),
    }


def run_episode(env: Any, model: Any, other: Any, *, seed: int, teacher: bool) -> list[dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    records = []
    while True:
        raw = model.policy.predict(observation, deterministic=True)[0] if teacher else None
        action, _ = model.predict(observation, deterministic=True)
        other_action, _ = other.predict(observation, deterministic=True)
        if teacher and getattr(model, "_air_brake_active", False):
            env._air_brake_request = True
            env._air_brake_base_action = model._air_brake_base_action
        air_brake_base = getattr(env.unwrapped, "_air_brake_base_action", None)
        next_observation, _, terminated, truncated, info = env.step(action)
        records.append(_record(len(records), observation, action, info, raw=raw,
                               comparison=other_action, air_brake_base=air_brake_base))
        observation = next_observation
        if terminated or truncated:
            return records


def replay_actions(env: Any, records: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Drive an identical reset with the saved actions and saved air-brake base."""
    observation, _ = env.reset(seed=seed)
    replay = []
    for row in records:
        previous_observation = observation
        action = np.asarray(row["action"], dtype=np.float32)
        base = row.get("air_brake_base_action")
        if base is not None:
            env._air_brake_request = True
            env._air_brake_base_action = np.asarray(base, dtype=np.float32)
        observation, _, terminated, truncated, info = env.step(action)
        replay.append(_record(len(replay), previous_observation, action, info))
        if terminated or truncated:
            break
    return replay


def perturb_teacher(env: Any, teacher: Any, *, seed: int, axis: int,
                    perturbation: float) -> list[dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    records = []
    while True:
        action, _ = teacher.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).copy()
        action[axis] = np.clip(action[axis] + perturbation, -1.0, 1.0)
        if getattr(teacher, "_air_brake_active", False):
            env._air_brake_request = True
            base = teacher._air_brake_base_action.copy()
            base[axis] = np.clip(base[axis] + perturbation, -1.0, 1.0)
            env._air_brake_base_action = base
        next_observation, _, terminated, truncated, info = env.step(action)
        records.append(_record(len(records), observation, action, info))
        observation = next_observation
        if terminated or truncated:
            return records


def _outcome(records: list[dict[str, Any]]) -> dict[str, Any]:
    last = records[-1]
    return {"steps": len(records), "elapsed_s": last["elapsed_s"],
            "progress": last["progress"], "events": last["events"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=Path(
        "models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"))
    parser.add_argument("--student", type=Path, default=Path(
        "models/experiments/ppo-wallspin-standard-20260930/summer-1/ppo/teacher-student/pretrained"))
    parser.add_argument("--dataset", type=Path, default=Path("runs/teacher-student/summer-1-teacher.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/teacher-student/transfer-diagnosis"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--experiments", action="store_true",
                        help="also replay teacher actions and test signed perturbations")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    teacher_metadata = ModelRegistry().read_metadata(args.teacher)
    student_metadata = ModelRegistry().read_metadata(args.student)
    teacher = backend_for("tqc").load_model(args.teacher / "policy.zip", None, "cpu")
    student = backend_for("ppo").load_model(args.student / "policy.zip", None, "cpu")
    teacher.policy_overlays = list(teacher_metadata.policy_overlays)
    dataset = TeacherDataset.load(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    report, arrays = compare_same_observations(teacher, student, dataset)
    report["checkpoints"] = {
        "teacher": str(args.teacher), "teacher_sha256": _sha256(args.teacher / "policy.zip"),
        "student": str(args.student), "student_sha256": _sha256(args.student / "policy.zip"),
        "dataset": str(args.dataset), "dataset_sha256": _sha256(args.dataset),
        "teacher_frame_skip": teacher_metadata.training_config["frame_skip"],
        "student_frame_skip": student_metadata.training_config["frame_skip"],
        "teacher_action_schema": teacher_metadata.action_schema,
        "student_action_schema": student_metadata.action_schema,
        "teacher_observation_schema": teacher_metadata.observation_schema,
        "student_observation_schema": student_metadata.observation_schema,
    }
    np.savez_compressed(args.output / "same-observation.npz", **arrays)
    if args.live:
        config = TrainingConfig.from_dict(dict(teacher_metadata.training_config))
        runner = TrainingRunner(config)
        env = runner._environment()
        env.capture_tick_controls = True
        student_env = AirBrakeActionWrapper(env, list(student_metadata.policy_overlays))
        try:
            reference = run_episode(env, teacher, student, seed=args.seed, teacher=True)
            candidate = run_episode(student_env, student, teacher, seed=args.seed, teacher=False)
            if args.experiments:
                replay = replay_actions(env, reference, seed=args.seed)
                perturbations = {}
                for axis, name in ((0, "steering"), (1, "longitudinal")):
                    for magnitude in (0.0001, 0.0005, 0.001, 0.005, 0.01):
                        for sign in (-1, 1):
                            delta = sign * magnitude
                            rows = perturb_teacher(env, teacher, seed=args.seed,
                                                   axis=axis, perturbation=delta)
                            perturbations[f"{name}:{delta:+g}"] = _outcome(rows)
        finally:
            env.close()
        report["live"] = {
            "seed": args.seed,
            "teacher": _outcome(reference), "student": _outcome(candidate),
            "first_divergences": first_divergences(reference, candidate),
        }
        for name, records in (("teacher", reference), ("student", candidate)):
            with (args.output / f"{name}-trajectory.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
        if args.experiments:
            report["replay"] = {
                "outcome": _outcome(replay),
                "first_divergences": first_divergences(reference, replay),
            }
            report["perturbations"] = perturbations
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"sections", "regimes"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
