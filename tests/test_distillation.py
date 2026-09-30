from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th

from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA
from polybot.models.registry import ModelMetadata, ModelRegistry
from polybot.training.config import EvaluationConfig, TQCConfig, TrainingConfig
from polybot.training.distillation import (
    _actor_forward_actions,
    _bakeable,
    _load_model,
    _predict_pair,
    _select_bake,
    bake_student,
    create_teacher_snapshot,
    rollback_student,
    run_full_workflow,
    sample_weights,
    train_student,
)
from polybot.training.runner import TrainingRunner


def _saved_teacher(tmp_path: Path) -> tuple[TrainingConfig, Path, Path]:
    config = TrainingConfig(
        algorithm="tqc", backend="mock", track_name="Summer 1", track_id="current",
        device="cpu", timesteps=32, evaluation=EvaluationConfig(32, 1),
        output_root=tmp_path / "models", log_root=tmp_path / "logs",
        tqc=TQCConfig(architecture="tiny", learning_starts=8, batch_size=8, replay_capacity=128),
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    runner = TrainingRunner(config)
    env = runner._environment()
    model = runner.backend.create_model(config, env, "cpu")
    model.policy_overlays = [
        {"kind": "steer_bias", "start": 0.2, "end": 0.8, "amount": 0.02, "taper": 0.01},
        {"kind": "air_brake", "start": 0.4, "end": 0.6, "duty": 1.0, "taper": 0.003},
    ]
    model.speed_bias_schedule = [(0.3, 0.7, 0.01)]
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    runner.backend.save_model(model, champion, resume=True)
    runner.registry.write_metadata(champion, ModelMetadata(
        algorithm="tqc", architecture="tiny", actor_parameters=1, critic_parameters=1,
        total_trainable_parameters=2, observation_schema=OBSERVATION_SCHEMA,
        action_schema=runner.backend.action_adapter(config).schema, track_name=config.track_name,
        track_id=config.track_id, lookahead_count=config.lookahead_count, reward_profile=None,
        curriculum={}, training_config=config.to_dict(), training_timesteps=32, simulator_ticks=0,
        wall_seconds=0, seed=config.seed, device="cpu", finishes=1, crashes=0,
        evaluation={"episodes": 5, "finish_rate": 1.0, "median_progress": 1.0,
                    "best_lap_s": 24.0, "median_lap_s": 24.0, "crash_rate": 0.0,
                    "off_track_rate": 0.0, "stall_rate": 0.0},
        policy_overlays=model.policy_overlays,
    ))
    env.close()
    return config, config_path, champion


def test_teacher_snapshot_records_overlays_and_schedule_and_final_action(tmp_path) -> None:
    config, config_path, _ = _saved_teacher(tmp_path)
    run_dir = create_teacher_snapshot(config_path, "test-run")
    teacher_info = json.loads((run_dir / "teacher.json").read_text(encoding="utf-8"))
    assert teacher_info["champion_lap_s"] == 24.0
    assert len(teacher_info["retained_overlays"]) == 1
    assert len(teacher_info["bakeable_overlays"]) == 2
    assert teacher_info["speed_bias_schedule"] == [[0.3, 0.7, 0.01]]
    metadata = ModelRegistry(config.output_root).read_metadata(run_dir)
    bakeable, retained = _bakeable(metadata)
    assert [row["kind"] for row in bakeable] == ["steer_bias"]
    assert [row["kind"] for row in retained] == ["air_brake"]

    teacher = _load_model(run_dir, config)
    observation = np.zeros(105, dtype=np.float32)
    observation[12] = 0.5
    observation[17:21] = 1.0
    raw, final, airbrake = _predict_pair(teacher, observation)
    assert final[0] == pytest.approx(raw[0] + 0.02, abs=2e-5)
    assert final[1] == pytest.approx(raw[1] + 0.01, abs=2e-5)
    assert not airbrake

    observation[12] = 0.5
    observation[17:21] = 0.0
    _raw, _final, airbrake = _predict_pair(teacher, observation)
    assert airbrake


def test_sample_weights_oversample_modified_boundaries_and_flights() -> None:
    data = {
        "action_delta": np.array([0.0, 0.01, 0.05], dtype=np.float32),
        "overlay_boundary": np.array([False, True, False]),
        "air_brake_active": np.array([False, False, True]),
    }
    weights = sample_weights(data)
    assert weights[0] == 1
    assert weights[1] > weights[0]
    assert weights[2] > weights[1]
    perturbed = sample_weights({
        **data,
        "perturbed": np.array([False, False, True]),
    })
    assert perturbed[2] > weights[2]


def test_full_workflow_leaves_passing_student_staged_for_explicit_bake(tmp_path, monkeypatch) -> None:
    import polybot.training.distillation as distillation

    run_dir = tmp_path / "staged-run"
    calls: list[str] = []
    monkeypatch.setattr(distillation, "create_teacher_snapshot", lambda _config: run_dir)
    monkeypatch.setattr(
        distillation, "collect_teacher_data",
        lambda _run, *, episodes: calls.append(f"collect:{episodes}") or {"samples": 10},
    )
    monkeypatch.setattr(distillation, "train_student", lambda _run: {"best_validation_loss": 0.01})
    monkeypatch.setattr(
        distillation, "validate_student",
        lambda _run, *, episodes, tolerance_s: {"accepted": True, "episodes": episodes},
    )
    monkeypatch.setattr(
        distillation, "bake_student",
        lambda *_args, **_kwargs: pytest.fail("full workflow must not implicitly promote"),
    )

    result = run_full_workflow(tmp_path / "config.json", episodes=20, validation_episodes=5)

    assert result["ready_to_bake"] is True
    assert result["promotion"] is None
    assert result["run_dir"] == str(run_dir)
    assert calls == ["collect:20"]


def test_partial_bake_keeps_unselected_smooth_overlays_and_schedule(tmp_path) -> None:
    config, config_path, _ = _saved_teacher(tmp_path)
    run_dir = create_teacher_snapshot(config_path, "partial-bake-run")
    metadata = ModelRegistry(config.output_root).read_metadata(run_dir)
    teacher_info = json.loads((run_dir / "teacher.json").read_text(encoding="utf-8"))
    metadata.policy_overlays.append(
        {"kind": "drive_gain", "start": 0.3, "end": 0.7, "amount": 1.1, "taper": 0.01},
    )
    bake, keep, bake_schedule = _select_bake(metadata, teacher_info, {"steer_bias"})
    assert [overlay["kind"] for overlay in bake] == ["steer_bias"]
    assert {overlay["kind"] for overlay in keep} == {"drive_gain", "air_brake"}
    assert bake_schedule is False


def test_actor_batch_forward_returns_one_action_per_observation(tmp_path) -> None:
    config, config_path, _ = _saved_teacher(tmp_path)
    run_dir = create_teacher_snapshot(config_path, "batch-forward-run")
    model = _load_model(run_dir, config)
    observations = np.random.default_rng(11).normal(0.0, 0.1, size=(8, 105)).astype(np.float32)
    batch = th.as_tensor(observations, device=model.device)
    predicted = _actor_forward_actions(model.actor, batch)
    individual = np.stack([_predict_pair(model, observation)[0] for observation in observations])
    assert predicted.shape == (8, 2)
    np.testing.assert_allclose(predicted.detach().cpu().numpy(), individual, atol=1e-6)


def test_actor_training_bakes_smooth_overlays_without_touching_critics(tmp_path, monkeypatch) -> None:
    config, config_path, champion = _saved_teacher(tmp_path)
    run_dir = create_teacher_snapshot(config_path, "train-run")
    teacher = _load_model(run_dir, config)
    rng = np.random.default_rng(4)
    observations = rng.normal(0.0, 0.1, size=(48, 105)).astype(np.float32)
    observations[:, 12] = np.linspace(0.0, 1.0, len(observations), dtype=np.float32)
    observations[:, 17:21] = 1.0
    targets = np.stack([_predict_pair(teacher, obs)[1] for obs in observations])
    raw = np.stack([_predict_pair(teacher, obs)[0] for obs in observations])
    delta = np.max(np.abs(targets - raw), axis=1)
    active = (observations[:, 12] >= 0.2) & (observations[:, 12] <= 0.8)
    arrays = {
        "observations": observations, "raw_actions": raw, "target_actions": targets,
        "progress": observations[:, 12], "speed_mps": np.zeros(len(observations), dtype=np.float32),
        "wheel_contacts": observations[:, 17:21], "action_delta": delta,
        "bakeable_overlay_active": active, "overlay_boundary": np.zeros(len(observations), dtype=bool),
        "air_brake_active": np.zeros(len(observations), dtype=bool),
        "section": np.minimum(9, (observations[:, 12] * 10).astype(np.int8)),
        "lap_id": np.repeat(np.arange(2), len(observations) // 2).astype(np.int32),
    }
    dataset_path = run_dir / "dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    from polybot.training.distillation import sha256_file

    (run_dir / "dataset.json").write_text(json.dumps({"dataset_sha256": sha256_file(dataset_path)}),
                                           encoding="utf-8")
    student_before = _load_model(run_dir, config)
    critic_before = [p.detach().cpu().clone() for p in student_before.critic.parameters()]
    target_before = [p.detach().cpu().clone() for p in student_before.critic_target.parameters()]
    ent_before = student_before.log_ent_coef.detach().cpu().clone()
    report = train_student(run_dir, epochs=8, learning_rate=0.001, patience=4,
                           batch_size=16, seed=9)
    assert report["actor_only"] is True
    assert report["best_epoch"] >= 1
    student_meta = ModelRegistry(config.output_root).read_metadata(run_dir / "student")
    assert [row["kind"] for row in student_meta.policy_overlays] == ["air_brake"]
    trained = _load_model(run_dir / "student", config)
    assert any(
        not th.equal(before.detach().cpu(), after.detach().cpu())
        for before, after in zip(teacher.actor.parameters(), trained.actor.parameters(), strict=True)
    )
    assert all(
        th.equal(a, b.detach().cpu())
        for a, b in zip(critic_before, trained.critic.parameters(), strict=True)
    )
    assert all(
        th.equal(a, b.detach().cpu())
        for a, b in zip(target_before, trained.critic_target.parameters(), strict=True)
    )
    assert th.equal(ent_before, trained.log_ent_coef.detach().cpu())
    assert (run_dir / "student" / "replay.pkl").is_file()

    before_hash = sha256_file(champion / "policy.zip")
    (run_dir / "validation.json").write_text(json.dumps({"accepted": False, "tolerance_s": 0.02}),
                                              encoding="utf-8")
    monkeypatch.setattr("polybot.training.distillation.simulator_service_active", lambda: False)
    with pytest.raises(ValueError, match="has not passed"):
        bake_student(run_dir)
    assert sha256_file(champion / "policy.zip") == before_hash
    assert not (run_dir / "promotion.json").exists()

    student_evaluation = json.loads((champion / "metadata.json").read_text(encoding="utf-8"))["evaluation"]
    (run_dir / "validation.json").write_text(json.dumps({
        "accepted": True, "tolerance_s": 0.02, "student": student_evaluation,
    }), encoding="utf-8")
    bake_student(run_dir)
    promoted_hash = sha256_file(champion / "policy.zip")
    assert promoted_hash != before_hash
    promoted_metadata = ModelRegistry(config.output_root).read_metadata(champion)
    assert promoted_metadata.critic_adaptation_required
    assert promoted_metadata.adaptation_stage == "critic_adaptation_required"
    rollback_student(run_dir)
    assert sha256_file(champion / "policy.zip") == before_hash
    assert (run_dir / "rollback.json").is_file()
