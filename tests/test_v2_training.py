from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from polybot.algorithms.registry import ALGORITHMS, backend_for
from polybot.environment.curriculum import build_plan
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import IncompatibleModelError, ModelRegistry
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    EvaluationConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.evaluation import EvaluationResult
from polybot.training.runner import TrainingRunner


def configuration(tmp_path, algorithm: str) -> TrainingConfig:
    specific = (
        {"ppo": PPOConfig(architecture="tiny", rollout_steps=32, batch_size=16, epochs=1)}
        if algorithm == "ppo" else
        {"tqc": TQCConfig(architecture="tiny", learning_starts=100,
                          batch_size=16, replay_capacity=1000)}
    )
    return TrainingConfig(
        algorithm=algorithm, device="cpu", timesteps=32,
        evaluation=EvaluationConfig(32, 1), checkpoint_interval=16,
        output_root=tmp_path / "models", log_root=tmp_path / "logs", **specific,
    )


def test_registry_contains_only_two_equal_backends() -> None:
    assert set(ALGORITHMS) == {"ppo", "tqc"}
    with pytest.raises(ValueError, match="unknown algorithm"):
        backend_for("sac")


def test_config_roundtrip_and_algorithm_specific_validation(tmp_path) -> None:
    for name in ALGORITHMS:
        config = configuration(tmp_path, name)
        assert TrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="TQC settings"):
        TrainingConfig(algorithm="ppo", ppo=PPOConfig(), tqc=TQCConfig())
    with pytest.raises(ValueError, match="only v2"):
        TrainingConfig.from_dict({"schema": "polybot.config.v1"})


def test_curriculum_budget_is_global() -> None:
    for mode, phase_count in (
        ("full", 1), ("quarters", 5), ("quarters-randomised", 1),
        ("q4-full", 2),
    ):
        plan = build_plan(CurriculumConfig(mode), 103)
        assert len(plan.phases) == phase_count
        assert plan.total_steps == 103
        assert all(phase.steps > 0 for phase in plan.phases)
    assert build_plan(CurriculumConfig("section", .25, .5), 50).phases[0].end_ratio == .5
    assert build_plan(CurriculumConfig("timed", start_s=3, end_s=6), 50).phases[0].end_s == 6
    custom = CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("section", 30, .75, 1.0),
        CurriculumPhaseConfig("full", 70),
    ))
    assert build_plan(custom, 100).total_steps == 100
    with pytest.raises(ValueError, match="sum"):
        build_plan(custom, 101)
    cfg = TrainingConfig(curriculum=custom, timesteps=100, tqc=TQCConfig())
    assert TrainingConfig.from_dict(cfg.to_dict()) == cfg


def test_champion_rank_uses_deterministic_finish_and_progress() -> None:
    weak = EvaluationResult(3, 0, .7, .75, None, None, 0, 0, 0)
    strong = EvaluationResult(3, 1/3, .5, .6, 25, 25, 1/3, 0, 0)
    assert strong.rank() > weak.rank()
    quicker = replace(strong, best_lap_s=22, median_lap_s=22)
    assert quicker.rank() > strong.rank()


@pytest.mark.parametrize("algorithm", ["ppo", "tqc"])
def test_short_train_save_resume_and_evaluate(tmp_path, algorithm: str) -> None:
    config = configuration(tmp_path, algorithm)
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, algorithm, "champion")
    assert (latest / "policy.zip").is_file()
    assert (champion / "policy.zip").is_file()
    assert (champion / "metadata.json").is_file()
    assert not (champion / "replay.pkl").exists()
    assert (latest / "replay.pkl").exists() == (algorithm == "tqc")
    metadata = registry.read_metadata(latest)
    assert metadata.training_config == config.to_dict()
    assert metadata.observation_schema == "polybot.observation.v2"
    assert metadata.evaluation["episodes"] == 1
    assert metadata.actor_parameters > 0
    assert metadata.total_trainable_parameters >= metadata.actor_parameters
    assert any(event["type"] == "evaluation" for event in events)
    assert any(event["type"] == "champion" for event in events)
    assert list(config.log_root.glob("*.jsonl"))
    before = metadata.training_timesteps
    TrainingRunner(config).run(resume=latest)
    assert registry.read_metadata(latest).training_timesteps > before
    with pytest.raises(IncompatibleModelError, match="track"):
        registry.validate(metadata, replace(config, track_id="mock/gentle-s"),
                          backend_for(algorithm).action_adapter(config).schema)


def test_tqc_warmup_replay_action_matches_executed_action(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                       action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        action, replay = model._sample_action(config.tqc.learning_starts, n_envs=1)
        np.testing.assert_array_equal(action, replay)
        _, _ = env.reset(seed=0)
        _, _, _, _, info = env.step(action[0])
        assert info["requested_control_duty"]["throttle"] == pytest.approx(max(0, action[0, 1]))
        assert info["requested_control_duty"]["brake"] == pytest.approx(max(0, -action[0, 1]))
    finally:
        env.close()


def test_saved_v2_metadata_rejects_v1(tmp_path) -> None:
    cfg = configuration(tmp_path, "ppo")
    path = ModelRegistry(cfg.output_root).slot(cfg.track_name, "ppo", "latest")
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"schema": "polybot.model.v1"}))
    with pytest.raises(IncompatibleModelError, match="only v2"):
        ModelRegistry(cfg.output_root).read_metadata(path)
