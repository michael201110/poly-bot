from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from polybot.algorithms.registry import ALGORITHMS, backend_for
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.environment.curriculum import build_plan
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import IncompatibleModelError, ModelRegistry
from polybot.protocol import Action
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    DQNConfig,
    EvaluationConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.evaluation import EvaluationResult
from polybot.training.runner import TrainingRunner


def configuration(tmp_path, algorithm: str) -> TrainingConfig:
    specific = {
        "ppo": {"ppo": PPOConfig(architecture="tiny", rollout_steps=32, batch_size=16, epochs=1)},
        "dqn": {"dqn": DQNConfig(architecture="tiny", learning_starts=8, batch_size=8,
                                 replay_capacity=1000, train_frequency=1, target_update_interval=16)},
        "tqc": {"tqc": TQCConfig(architecture="tiny", learning_starts=100,
                                 batch_size=16, replay_capacity=1000)},
    }[algorithm]
    return TrainingConfig(
        algorithm=algorithm, device="cpu", timesteps=32,
        evaluation=EvaluationConfig(32, 1), checkpoint_interval=16,
        output_root=tmp_path / "models", log_root=tmp_path / "logs", **specific,
    )


def test_registry_contains_three_equal_backends() -> None:
    assert set(ALGORITHMS) == {"ppo", "dqn", "tqc"}
    with pytest.raises(ValueError, match="unknown algorithm"):
        backend_for("sac")


def test_config_roundtrip_and_algorithm_specific_validation(tmp_path) -> None:
    for name in ALGORITHMS:
        config = configuration(tmp_path, name)
        assert TrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="TQC settings"):
        TrainingConfig(algorithm="ppo", ppo=PPOConfig(), tqc=TQCConfig())
    with pytest.raises(ValueError, match="DQN"):
        TrainingConfig(algorithm="ppo", ppo=PPOConfig(), dqn=DQNConfig())
    old_v2 = configuration(tmp_path, "tqc").to_dict()
    del old_v2["dqn"]
    assert TrainingConfig.from_dict(old_v2).dqn is None
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


def test_curriculum_sections_spawn_with_lead_in_and_keep_global_budget() -> None:
    expected = [(0.0, 0.0, 0.25), (0.20, 0.25, 0.50), (0.45, 0.50, 0.75),
                (0.70, 0.75, 1.0)]
    plan = build_plan(CurriculumConfig("quarters"), 1000)
    assert plan.total_steps == 1000
    for phase, (spawn, start, end) in zip(plan.phases[:4], expected, strict=True):
        assert (phase.spawn_ratio, phase.start_ratio, phase.end_ratio) == (spawn, start, end)
    randomised = build_plan(CurriculumConfig("quarters-randomised"), 100)
    assert randomised.total_steps == 100
    assert randomised.phases[0].env_kwargs()["curriculum_lead_in_ratio"] == 0.05
    q4_full = build_plan(CurriculumConfig("q4-full"), 100)
    assert (q4_full.phases[0].spawn_ratio, q4_full.phases[0].start_ratio,
            q4_full.phases[0].end_ratio) == (0.70, 0.75, 1.0)
    section = build_plan(CurriculumConfig("section", .25, .5), 20)
    assert (section.phases[0].spawn_ratio, section.phases[0].start_ratio) == (0.20, 0.25)
    custom = build_plan(CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("section", 20, .5, .75, lead_in_ratio=.1),
        CurriculumPhaseConfig("full", 30),
    )), 50)
    assert (custom.phases[0].spawn_ratio, custom.phases[0].start_ratio,
            custom.phases[0].end_ratio) == (.4, .5, .75)

    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", curriculum_random_quarters=True,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        _, info = env.reset(seed=17)
        assert info["curriculum_start_ratio"] - info["curriculum_spawn_ratio"] == pytest.approx(.05)
        assert info["curriculum_end_ratio"] - info["curriculum_start_ratio"] == pytest.approx(.25)
    finally:
        env.close()


def test_curriculum_reset_info_is_moving_and_section_relative() -> None:
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=4,
        curriculum_spawn_ratio=.20, curriculum_start_ratio=.25, curriculum_end_ratio=.5,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        _, reset = env.reset(seed=4)
        assert reset["ticks_advanced"] == 0
        assert reset["route_progress_m"] > 0
        assert reset["local_velocity_mps"][2] == pytest.approx(20.0)
        assert reset["previous_action"] == Action().to_wire()
        assert reset["actual_steering"] == 0
        assert reset["section_progress"] == 0
        assert reset["curriculum_in_lead_in"] is True
        _, _, _, _, info = env.step(1)
        assert info["ticks_advanced"] == 4
        assert info["section_progress"] == pytest.approx(0.0)
        assert info["curriculum_stage"] == "lead-in"
    finally:
        env.close()

    q1 = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight", frame_skip=4)
    try:
        _, reset = q1.reset(seed=4)
        assert reset["ticks_advanced"] == 0
        assert reset["route_progress_m"] == 0
        assert reset["local_velocity_mps"][2] == 0
        assert reset["curriculum_stage"] == "full track"
    finally:
        q1.close()


def test_champion_rank_uses_deterministic_finish_and_progress() -> None:
    weak = EvaluationResult(3, 0, .7, .75, None, None, 0, 0, 0)
    strong = EvaluationResult(3, 1/3, .5, .6, 25, 25, 1/3, 0, 0)
    assert strong.rank() > weak.rank()
    quicker = replace(strong, best_lap_s=22, median_lap_s=22)
    assert quicker.rank() > strong.rank()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn", "tqc"])
def test_short_train_save_resume_and_evaluate(tmp_path, algorithm: str) -> None:
    config = configuration(tmp_path, algorithm)
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, algorithm, "champion")
    assert (latest / "policy.zip").is_file()
    assert (champion / "policy.zip").is_file()
    assert (champion / "metadata.json").is_file()
    assert (champion / "replay.pkl").exists() == (algorithm in {"dqn", "tqc"})
    assert (latest / "replay.pkl").exists() == (algorithm in {"dqn", "tqc"})
    metadata = registry.read_metadata(latest)
    assert metadata.training_config == config.to_dict()
    assert metadata.observation_schema == "polybot.observation.v2"
    assert metadata.evaluation["episodes"] == 1
    assert (metadata.actor_parameters == 0) == (algorithm == "dqn")
    assert metadata.critic_parameters > 0
    assert metadata.total_trainable_parameters >= metadata.actor_parameters
    if algorithm == "ppo":
        registry.write_metadata(latest, replace(metadata, polybot_version="2.0.0"))
        assert registry.read_metadata(latest).polybot_version == "2.0.0"
    assert any(event["type"] == "evaluation" for event in events)
    assert any(event["type"] == "champion" for event in events)
    assert list(config.log_root.glob("*.jsonl"))
    before = metadata.training_timesteps
    if algorithm == "dqn":
        backend = backend_for("dqn")
        env = PolyTrackEnv(MockSimulatorTransport(), track_id=config.track_id,
                           action_adapter=backend.action_adapter(config))
        try:
            loaded = backend.load_model(latest / "policy.zip", env, "cpu", resume=True)
            assert loaded.action_space.n == 9
            assert loaded.replay_buffer.size() > 0
            assert backend.metrics(loaded)["updates"] > 0
            deterministic, _ = loaded.predict(env.reset(seed=5)[0], deterministic=True)
            assert 0 <= int(deterministic) < 9
            best = backend.load_model(champion / "policy.zip", env, "cpu", resume=True)
            assert best.replay_buffer.size() > 0
        finally:
            env.close()
    TrainingRunner(config).run(resume=latest)
    assert registry.read_metadata(latest).training_timesteps > before
    with pytest.raises(IncompatibleModelError, match="track"):
        registry.validate(metadata, replace(config, track_id="mock/gentle-s"),
                          backend_for(algorithm).action_adapter(config).schema)


def test_older_tqc_champion_without_replay_can_continue(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    TrainingRunner(config).run()
    champion = ModelRegistry(config.output_root).slot(config.track_name, "tqc", "champion")
    (champion / "replay.pkl").unlink()
    resumed = replace(
        config, timesteps=16, evaluation=EvaluationConfig(16, 1),
        tqc=replace(config.tqc, learning_starts=8),
    )
    runner = TrainingRunner(resumed)
    runner.run(resume=champion, fresh_replay=True)
    assert runner.model.replay_buffer.size() >= 16
    assert runner.model.learning_starts >= runner.model.num_timesteps
    assert runner.model._refill_replay_from_policy
    assert runner.model._n_updates == 0


@pytest.mark.parametrize("timesteps", (24, 32))
def test_continue_best_restores_champion_after_weaker_evaluation(
    tmp_path, monkeypatch, timesteps: int
) -> None:
    import polybot.training.runner as runner_module

    config = replace(
        configuration(tmp_path, "tqc"), timesteps=timesteps,
        evaluation=EvaluationConfig(16, 1),
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    weak = EvaluationResult(1, 0.0, 0.2, 0.2, None, None, 0.0, 1.0, 0.0)
    evaluations = iter((strong, weak))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    assert (champion / "replay.pkl").is_file()
    assert registry.read_metadata(champion).training_timesteps == 16
    assert registry.read_metadata(latest).training_timesteps == timesteps
    assert registry.read_metadata(latest).evaluation == strong.to_dict()
    assert any(event["type"] == "rollback" and event["replay_source"] == "champion"
               for event in events)


def test_dqn_exploration_follows_full_budget_across_evaluation_chunks(tmp_path) -> None:
    config = replace(
        configuration(tmp_path, "dqn"),
        timesteps=1_000,
        evaluation=EvaluationConfig(16, 1),
        checkpoint_interval=0,
        dqn=replace(configuration(tmp_path, "dqn").dqn, exploration_fraction=1.0),
    )
    runner = None

    def on_event(event: dict) -> None:
        if event["type"] == "evaluation":
            runner.stop()

    runner = TrainingRunner(config, on_event)
    latest = runner.run()
    env = PolyTrackEnv(MockSimulatorTransport(), track_id=config.track_id,
                       action_adapter=backend_for("dqn").action_adapter(config))
    try:
        model = backend_for("dqn").load_model(latest / "policy.zip", env, "cpu")
        assert model.num_timesteps == 16
        assert model.exploration_rate > 0.9
    finally:
        env.close()


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
